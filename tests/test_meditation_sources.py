"""Meditation uses verified source data and keeps its independent roles apart."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.chat import ChatService
from lucifer.meditation_sources import (
    _group_rows, commit_source_revision, invalidate_meditation_sources, prepare_meditation_sources,
    revalidate_meditation_sources,
)
from lucifer.memory import invalidate_candidate_dependents
from lucifer.store import Store
from lucifer.tools import ToolContext
from lucifer.tools import ToolRegistry, ToolPolicyError
from lucifer.tools import web


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "runtime" / "state.db")
    chat = ChatService(store, SimpleNamespace())
    yield store, chat, tmp_path
    store.close()


async def source(chat, text, *, cid=None, training=True):
    cid = cid or chat.create(training_enabled=training)["id"]
    doc = await chat.knowledge.attach(cid, filename="source.txt", content_base64=base64.b64encode(text.encode()).decode())
    if chat.store.get("conversations", cid).get("logging_enabled"):
        for chunk in range(0, doc["chunks"], 5):
            chat.knowledge.read(cid, doc["id"], chunk=chunk, max_chunks=5)
    return cid, doc


@pytest.mark.asyncio
async def test_local_read_ingests_pinned_bytes_and_exact_public_excerpt(env):
    store, chat, root = env
    cid = chat.create(training_enabled=True)["id"]
    raw = b"First documented fact.\nSecond documented fact.\n"
    (root / "download.txt").write_bytes(raw)
    registry = ToolRegistry()
    context = ToolContext(root, knowledge=chat.knowledge, conversation_id=cid, operation_id="local_read", run_id="agent")
    result = await registry.execute("read_file", {"path": "download.txt", "max_lines": 1}, context)
    doc = store.list("knowledge_documents")[0]
    assert doc["sha256"] == hashlib.sha256(raw).hexdigest()
    assert doc["metadata"]["origin"] == "successful_local_read"
    assert result.content["knowledge_source"]["document_id"] == doc["id"]
    receipt = store.list("knowledge_interactions")[0]
    assert receipt["kind"] == "local_read" and receipt["operation_id"] == "local_read"
    assert receipt["returned_sha256"] == hashlib.sha256(result.content["content"].encode()).hexdigest()
    assert receipt["run_id"] == "agent"
    version = registry.source_version("import_document", {"path": "download.txt"}, context)
    read_version = registry.source_version("read_file", {"path": "download.txt"}, context)
    chat.knowledge.delete(cid, doc["id"])
    assert registry.source_version("import_document", {"path": "download.txt"}, context) != version
    assert registry.source_version("read_file", {"path": "download.txt"}, context) == read_version


@pytest.mark.asyncio
async def test_local_document_import_is_confined_and_only_read_chunks_enter_meditation(env):
    store, chat, root = env
    cid = chat.create(training_enabled=True)["id"]
    workspace = root / "project"
    workspace.mkdir()
    (workspace / "download.csv").write_text("planet,moons\nEarth,1\nMars,2\n")
    (root / "outside.csv").write_text("private,source")
    registry = ToolRegistry()
    context = ToolContext(workspace, knowledge=chat.knowledge, conversation_id=cid,
                          operation_id="import_download", run_id="agent")
    with pytest.raises(ToolPolicyError):
        await registry.execute("import_document", {"path": "../outside.csv"}, context)
    imported = await registry.execute("import_document", {"path": "download.csv"}, context)
    assert imported.content["chunks"] > 0 and imported.content["instructions_trusted"] is False
    document = store.get("knowledge_documents", imported.content["document_id"])
    assert document["source"] == (workspace / "download.csv").as_uri()
    assert document["metadata"]["import_origins"][0]["operation_id"] == "import_download"
    assert not store.list("knowledge_interactions")
    bundle = await prepare_meditation_sources(store, {"mode": "unsupervised"})
    assert not bundle["examples"]
    read = await registry.execute("read_knowledge", {"document_id": imported.content["document_id"]}, context)
    assert read.content
    bundle = await prepare_meditation_sources(store, {"mode": "unsupervised"})
    assert bundle["examples"] and all(row["source_kind"] == "document" for row in bundle["examples"])
    assert revalidate_meditation_sources(store, bundle)


@pytest.mark.asyncio
async def test_duplicate_local_import_retains_uploaded_identity_and_authorized_path(env):
    store, chat, root = env
    cid = chat.create(training_enabled=True)["id"]
    raw = b"planet,moons\nEarth,1\n"
    existing = await chat.knowledge.attach(cid, filename="uploaded.csv", content_base64=base64.b64encode(raw).decode())
    (root / "downloaded.csv").write_bytes(raw)
    registry = ToolRegistry()
    context = ToolContext(root, knowledge=chat.knowledge, conversation_id=cid,
                          operation_id="import_downloaded", run_id="agent")
    imported = await registry.execute("import_document", {"path": "downloaded.csv"}, context)
    assert imported.content["document_id"] == existing["id"]
    document = store.get("knowledge_documents", existing["id"])
    assert document["source"] == "upload:uploaded.csv"
    assert document["metadata"]["import_origins"][0]["source"] == (root / "downloaded.csv").as_uri()
    assert len(store.list("knowledge_documents")) == 1 and not store.list("knowledge_interactions")
    version = chat.knowledge.revision(cid)
    await registry.execute("import_document", {"path": "downloaded.csv"}, context)
    assert chat.knowledge.revision(cid) == version


@pytest.mark.asyncio
async def test_supervised_followup_keeps_public_task_context_without_reasoning(env):
    from lucifer.meditation_sources import _native_rows
    store, chat, _ = env
    cid = chat.create(training_enabled=True)["id"]
    turns = [("user", "Compare TCP and UDP for this audio application."),
             ("assistant", "TCP is option one. UDP is option two."),
             ("user", "Explain the second option."), ("assistant", "UDP sends datagrams.")]
    for index, (role, content) in enumerate(turns):
        store.put("messages", {"id": f"turn-{index}", "conversation_id": cid, "role": role, "content": content,
                               "reasoning_content": "DO_NOT_TRAIN_REASONING"})
    rows = _native_rows(store, [cid], "supervised")
    assert rows[-1]["messages"] == [{"role": role, "content": text} for role, text in turns]
    assert rows[-1]["source_message_ids"] == [f"turn-{index}" for index in range(4)]
    assert "DO_NOT_TRAIN_REASONING" not in json.dumps(rows)


@pytest.mark.asyncio
async def test_web_ingestion_reuses_download_and_records_exact_operation(env, monkeypatch):
    store, chat, root = env
    cid = chat.create(training_enabled=True)["id"]
    calls = []
    raw = b"<html><title>Verified source</title><script>NEVER_TRAIN_SCRIPT</script><p>" + b"evidence " * 500 + b"</p></html>"
    async def fetch(url, **kwargs):
        calls.append(url)
        return web.FetchResult(raw, "text/html", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    web._CACHE.clear()
    context = ToolContext(root, knowledge=chat.knowledge, conversation_id=cid, operation_id="owned_read", run_id="mission")
    result = await web.execute_web("read_web_page", {"url": "https://example.com/source", "max_chars": 100}, context)
    again = await web.execute_web("read_web_page", {"url": "https://example.com/source", "max_chars": 100}, context)
    assert len(calls) == 1 and again["cached"]
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(store.list("knowledge_documents")) == 1
    receipts = store.list("knowledge_interactions")
    assert len(receipts) == 1
    assert receipts[0]["operation_id"] == "owned_read" and receipts[0]["run_id"] == "mission"
    assert receipts[0]["returned_sha256"] == hashlib.sha256(result["content"].encode()).hexdigest()
    assert "NEVER_TRAIN_SCRIPT" not in json.dumps(store.list("knowledge_chunks"))
    assert "content" not in receipts[0] and "reasoning" not in receipts[0]
    assert chat.knowledge.list(cid)[0]["training_eligible"]


@pytest.mark.asyncio
async def test_web_read_with_training_off_does_not_opt_chat_in(env, monkeypatch):
    store, chat, root = env
    cid = chat.create()["id"]
    async def fetch(url, **kwargs):
        return web.FetchResult(b"Known public passage", "text/plain", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    web._CACHE.clear()
    result = await web.execute_web("read_web_page", {"url": "https://example.com/plain"},
        ToolContext(root, knowledge=chat.knowledge, conversation_id=cid))
    assert result["status"] == "ok" and not store.get("conversations", cid)["training_enabled"]
    assert not chat.knowledge.list(cid)[0]["training_eligible"]
    bundle = await prepare_meditation_sources(store, {})
    assert not bundle["examples"] and not bundle["holdout_examples"]
    with pytest.raises(ValueError, match="consent"):
        await prepare_meditation_sources(store, {"conversation_ids": [cid]})


@pytest.mark.asyncio
async def test_private_interaction_is_transient_and_deleted_with_source(env):
    store, chat, _ = env
    cid = chat.create(logging_enabled=False)["id"]
    _, doc = await source(chat, "private source passage", cid=cid)
    chat.knowledge.read(cid, doc["id"], operation_id="private_op")
    assert store.list("knowledge_interactions") == []
    assert chat.knowledge.interactions(cid)[0]["operation_id"] == "private_op"
    chat.patch(cid, {"logging_enabled": True})
    assert len(store.list("knowledge_interactions")) == 1
    chat.patch(cid, {"logging_enabled": False})
    assert store.list("knowledge_interactions") == []
    chat.knowledge.delete(cid, doc["id"])
    assert chat.knowledge.interactions(cid) == []


@pytest.mark.asyncio
async def test_connected_duplicate_documents_do_not_cross_roles_and_refresh_is_stable(env):
    store, chat, _ = env
    first, _ = await source(chat, "Identical independent document.")
    second, _ = await source(chat, "Identical independent document.")
    for index in range(16):
        await source(chat, f"Independent document number {index}: factual passage {index}.")
    bundle = await prepare_meditation_sources(store, {"max_generated_examples": 0}, run_id="meditation_test")
    assert revalidate_meditation_sources(store, bundle)
    assert all(bundle[key] for key in ("examples", "development_examples", "holdout_examples"))
    groups = bundle["manifest"]["groups"]
    matched = [group for group in groups if "conversation:" + first in group["members"]]
    assert len(matched) == 1 and "conversation:" + second in matched[0]["members"]
    row_groups = {}
    for key in ("examples", "development_examples", "holdout_examples"):
        for row in bundle[key]:
            assert row["meditation_group_id"] not in row_groups or row_groups[row["meditation_group_id"]] == key
            row_groups[row["meditation_group_id"]] = key
    repeat = await prepare_meditation_sources(store, {"max_generated_examples": 0}, run_id="meditation_test")
    assert repeat["manifest"] == bundle["manifest"]
    tampered = copy.deepcopy(bundle)
    tampered["examples"][0]["messages"][1]["content"] += "forged"
    with pytest.raises(ValueError):
        revalidate_meditation_sources(store, tampered)


@pytest.mark.asyncio
async def test_new_document_cannot_merge_reserved_training_and_admission_groups(env):
    store, chat, _ = env
    for index in range(8):
        await source(chat, f"Independent source number {index}.")
    bundle = await prepare_meditation_sources(store, {"max_generated_examples": 0})
    groups = bundle["manifest"]["groups"]
    def conversation(role):
        group = next(group for group in groups if group["partition"] == role)
        return next(member.split(":", 1)[1] for member in group["members"] if member.startswith("conversation:"))
    await source(chat, "New shared document joins two existing roles.", cid=conversation("train"))
    await source(chat, "New shared document joins two existing roles.", cid=conversation("heldout"))
    with pytest.raises(ValueError, match="reserved"):
        await prepare_meditation_sources(store, {"max_generated_examples": 0})


@pytest.mark.asyncio
async def test_source_content_and_interaction_tampering_fail_revalidation(env):
    store, chat, _ = env
    cid, doc = await source(chat, "Reviewed source with distinct words.")
    chat.knowledge.read(cid, doc["id"])
    bundle = await prepare_meditation_sources(store, {"max_generated_examples": 0})
    assert revalidate_meditation_sources(store, bundle)
    receipt = store.list("knowledge_interactions")[0]
    store.put("knowledge_interactions", {**receipt, "kind": "forged"})
    with pytest.raises(ValueError, match="receipt"):
        revalidate_meditation_sources(store, bundle)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["supervised", "unsupervised"])
async def test_public_training_chats_do_not_need_manual_reviews(env, mode):
    store, chat, _ = env
    for index in range(4):
        cid = chat.create(training_enabled=True)["id"]
        store.put("messages", {"id": f"u{index}", "conversation_id": cid, "role": "user",
                              "content": f"Public user text {index}."})
        store.put("messages", {"id": f"a{index}", "conversation_id": cid, "role": "assistant",
                              "content": f"Public final answer {index}.", "reasoning_content": "RAW_PRIVATE_REASONING"})
    bundle = await prepare_meditation_sources(store, {"mode": mode})
    assert revalidate_meditation_sources(store, bundle)
    assert bundle["metrics"]["manual_review_required"] is False
    assert bundle["pending"] == [] and "RAW_PRIVATE_REASONING" not in json.dumps(bundle)
    rows = bundle["examples"] + bundle["development_examples"] + bundle["holdout_examples"]
    assert rows and all(row["training_mode"] == mode for row in rows)
    assert all(bundle[key] for key in ("examples", "development_examples", "holdout_examples"))
    if mode == "unsupervised":
        assert all("text" in row and "messages" not in row for row in rows)
        assert {row["source_authority"] for row in rows} == {"literal_user_text", "public_model_final"}
    else:
        assert all(row["messages"][-1]["role"] == "assistant" for row in rows)


@pytest.mark.asyncio
async def test_supervised_targets_use_corrections_and_exclude_explicit_negative_feedback(env):
    store, chat, _ = env
    cid = chat.create(training_enabled=True)["id"]
    store.put("messages", {"id": "user", "conversation_id": cid, "role": "user", "content": "Which port?"})
    store.put("messages", {"id": "answer", "conversation_id": cid, "role": "assistant", "content": "9999"})
    review = store.put("feedback", {"conversation_id": cid, "message_id": "answer", "label": "bad"})
    assert not (await prepare_meditation_sources(store, {"mode": "supervised"}))["examples"]
    store.update("feedback", review["id"], correction="8080")
    bundle = await prepare_meditation_sources(store, {"mode": "supervised"})
    assert bundle["examples"][0]["messages"][-1]["content"] == "8080"
    assert bundle["examples"][0]["source_authority"] == "user_correction"
    assert revalidate_meditation_sources(store, bundle)


@pytest.mark.asyncio
async def test_empty_consented_chats_are_selected_but_not_consumed_evidence(env):
    store, chat, _ = env
    empty = chat.create(training_enabled=True)["id"]
    for index in range(3):
        await source(chat, f"Independent document {index}.")
    bundle = await prepare_meditation_sources(store, {"mode": "unsupervised"})
    assert empty not in bundle["source_conversation_ids"]
    assert empty in bundle["manifest"]["selected_conversation_ids"]
    assert bundle["metrics"]["empty_selected_chats"] == 1
    assert all(bundle[key] for key in ("examples", "development_examples", "holdout_examples"))
    assert revalidate_meditation_sources(store, bundle)


def test_many_independent_sources_can_supply_more_than_one_admission_group(env):
    store, chat, _ = env
    examples = []
    for index in range(100):
        cid = chat.create(id=f"independent_{index}", training_enabled=True)["id"]
        examples.append({"conversation_id": cid, "fingerprint": f"independent_example_{index}"})
    groups = _group_rows(store, examples)
    assert sum(group["partition"] == "heldout" for group in groups) >= 8
    assert sum(group["partition"] == "development" for group in groups) >= 8
    assert sum(group["partition"] == "train" for group in groups) >= 40


@pytest.mark.asyncio
async def test_immutable_source_refresh_retains_old_lineage_and_reuses_identical_revision(env):
    store, chat, _ = env
    for index in range(4):
        await source(chat, f"Initial independently consented document {index}.")
    ident = "meditation_refresh"
    root = store.path.parent / "meditation" / ident
    old = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    first = commit_source_revision(store, ident, root, old)
    first_path = Path(first["source_cache_path"])
    first_bytes, first_mtime = first_path.read_bytes(), first_path.stat().st_mtime_ns
    same = commit_source_revision(store, ident, root, old, first["source_revision_refs"])
    assert not first["data_refreshed"] and not same["data_refreshed"]
    assert same["source_revision_refs"] == first["source_revision_refs"]
    assert first_path.read_bytes() == first_bytes and first_path.stat().st_mtime_ns == first_mtime
    await source(chat, "Additional independent newly consented source.")
    refreshed = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    second = commit_source_revision(store, ident, root, refreshed, first["source_revision_refs"])
    assert second["data_refreshed"] and len(second["source_revision_refs"]) == 2
    assert second["source_cache_path"] != first["source_cache_path"]
    assert second["dataset_sha256"] != first["dataset_sha256"]
    assert first_path.read_bytes() == first_bytes
    data = json.loads(Path(second["dataset_path"]).read_text())
    assert all(data["datasets"][role] for role in ("train", "development", "holdout"))
    assert all("text" in row for values in data["datasets"].values() for row in values)


@pytest.mark.asyncio
async def test_adding_duplicate_source_preserves_original_example_binding(env):
    store, chat, _ = env
    for index in range(3):
        await source(chat, f"Independent original source {index}.")
    ident = "meditation_duplicate"
    root = store.path.parent / "meditation" / ident
    old = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    first = commit_source_revision(store, ident, root, old)
    await source(chat, "Independent original source 0.")
    new = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    second = commit_source_revision(store, ident, root, new, first["source_revision_refs"])
    assert not second["data_refreshed"]
    original_refs = {row["fingerprint"]: row for row in old["manifest"]["example_refs"]}
    assert all(original_refs[row["fingerprint"]] == row for row in new["manifest"]["example_refs"])


@pytest.mark.asyncio
async def test_source_refresh_rejects_corrupt_or_revoked_consumed_revision(env):
    store, chat, _ = env
    ids = []
    for index in range(3):
        cid, _ = await source(chat, f"Independent source before withdrawal {index}.")
        ids.append(cid)
    ident = "meditation_revocation"
    root = store.path.parent / "meditation" / ident
    old = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    first = commit_source_revision(store, ident, root, old)
    path = Path(first["source_cache_path"])
    raw = path.read_bytes()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="changed"):
        commit_source_revision(store, ident, root, old, first["source_revision_refs"])
    path.write_bytes(raw)
    chat.patch(ids[0], {"training_enabled": False})
    remaining = await prepare_meditation_sources(store, {"mode": "unsupervised"}, run_id=ident)
    with pytest.raises(ValueError):
        commit_source_revision(store, ident, root, remaining, first["source_revision_refs"])


def test_revocation_removes_source_token_diagnostics_and_redacts_metadata_but_keeps_weights(env):
    store, chat, _ = env
    cid = chat.create(training_enabled=True)["id"]
    run_id = "meditation_diagnostics"
    root = store.path.parent / "meditation" / run_id
    checkpoint = root / "segment-1" / "checkpoint"
    checkpoint.mkdir(parents=True)
    head = checkpoint / "head.bf16"
    head.write_bytes(b"learned weights retained for blocked history")
    optimizer = checkpoint / "optimizer.json"
    optimizer.write_text('{"data_revision":"hash-only"}')
    config = checkpoint.parent / "worker_config.json"
    config.write_text('{"prepared":{"diagnostic_ids":[98123,98124]}}')
    diagnostic = root / "native-verify-1"
    diagnostic.mkdir()
    (diagnostic / "source-token-ids.txt").write_text("98123,98124")
    store.put("meditation_runs", {"id": run_id, "source_conversation_ids": [cid], "status": "paused"})
    store.put("meditation_operations", {"id": "owned", "run_id": run_id,
        "binding": {"config": {"prepared": {"diagnostic_ids": [98123, 98124], "messages": [{"content": "sensitive source"}]}}}})
    chat.patch(cid, {"training_enabled": False})
    assert not config.exists() and not diagnostic.exists()
    assert head.is_file() and optimizer.is_file()
    record = store.get("meditation_operations", "owned")
    assert record["binding"]["config"]["prepared"]["diagnostic_ids"]["unavailable"]
    assert "98123" not in json.dumps(record) and "sensitive source" not in json.dumps(record)
    before = store.revision("meditation_operations", "meditation_runs")
    invalidate_meditation_sources(store, conversation_id=cid, reason="source_consent_revoked")
    assert before == store.revision("meditation_operations", "meditation_runs")


@pytest.mark.parametrize("action", ["training_off", "delete", "review_changed"])
def test_revocation_invalidates_native_descendants_and_erases_only_owned_cache(env, action):
    store, chat, root = env
    cid = chat.create(training_enabled=True)["id"]
    features = store.path.parent / "meditation" / "job" / "features.bin"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"source features")
    source_bundle = features.parent / "sources.json"
    token_input = features.parent / "dataset.json"
    source_bundle.write_text('{"source":"private"}')
    token_input.write_text('{"tokens":[1,2,3]}')
    outside = root / "external-source.bin"
    outside.write_bytes(b"unrelated")
    alias = features.parent / "linked.bin"
    alias.symlink_to(outside)
    store.put("meditation_datasets", {"id": "data", "run_id": "run", "source_conversation_ids": [cid],
        "example_refs": [{"candidate_id": "reviewed"}], "source_cache_path": str(source_bundle)})
    store.put("meditation_runs", {"id": "run", "status": "running", "source_conversation_ids": [cid]})
    store.put("meditation_features", {"id": "features", "run_id": "run", "feature_path": str(features)})
    store.put("meditation_features", {"id": "link", "run_id": "run", "feature_path": str(alias)})
    store.put("base_candidates", {"id": "base", "run_id": "run", "status": "admitted"})
    store.put("base_candidates", {"id": "child", "parent_base_id": "base", "status": "admitted"})
    store.put("meditation_admissions", {"id": "receipt", "base_candidate_id": "child", "status": "valid"})
    if action == "training_off":
        chat.patch(cid, {"training_enabled": False})
    elif action == "delete":
        chat.delete(cid)
    else:
        invalidate_candidate_dependents(store, "reviewed")
    assert not features.exists() and not alias.is_symlink() and outside.read_bytes() == b"unrelated"
    assert not source_bundle.exists() and not token_input.exists()
    assert store.get("base_candidates", "child")["source_invalidated"]
    assert store.get("base_candidates", "base")["status"] == "revoked"
    assert store.get("meditation_admissions", "receipt")["status"] == "revoked"
    before = store.revision(*("meditation_runs", "base_candidates", "meditation_admissions"))
    invalidate_meditation_sources(store, **({"candidate_id": "reviewed"} if action == "review_changed" else {"conversation_id": cid}),
        reason="candidate_review_changed" if action == "review_changed" else "source_consent_revoked")
    assert store.revision(*("meditation_runs", "base_candidates", "meditation_admissions")) == before
