import base64
import copy
import json
from types import SimpleNamespace

import pytest

from lucifer.chat import ChatService
from lucifer.document_extract import extract_document, make_chunks
from lucifer.meditation_sources import prepare_meditation_sources, revalidate_meditation_sources
from lucifer.problem_solution import FORMAT, parse_record, source_family
from lucifer.store import Store


def record(ident="example", uri="https://github.com/example/agent/blob/main/guide.md"):
    return {"format": FORMAT, "id": ident,
            "question": "A completed operation receipt exists after a reconnect. What should the agent do?",
            "answer": "Reuse the completed result and continue from the next unfinished step.",
            "sources": [{"url": uri, "sha256": "a" * 64}],
            "verification": {"method": "deterministic_state_fixture", "receipt_sha256": "b" * 64}}


def test_problem_solution_is_literal_bounded_and_never_runs_a_verifier():
    row = record()
    raw = json.dumps(row).encode()
    doc = extract_document("problems.jsonl", raw)
    assert doc.metadata["problem_solution_format"] == FORMAT
    assert parse_record(make_chunks(doc.sections)[0]["text"]) == row
    row["verification"] = {"program": "raise RuntimeError('DO_NOT_EXECUTE')"}
    with pytest.raises(ValueError, match="verification"):
        extract_document("problems.jsonl", json.dumps(row).encode())
    row = record()
    row["reasoning_content"] = "PRIVATE_REASONING"
    with pytest.raises(ValueError, match="public fields"):
        extract_document("problems.jsonl", json.dumps(row).encode())
    row = record()
    row["answer"] = "long" * 1000
    with pytest.raises(ValueError, match="chunk bound"):
        extract_document("problems.jsonl", json.dumps(row).encode())
    with pytest.raises(ValueError, match="Duplicate"):
        extract_document("problems.jsonl", raw + b"\n" + raw)
    plain = b'{"question":"ordinary jsonl data", "answer":"literal"}'
    assert "problem_solution_format" not in extract_document("plain.jsonl", plain).metadata


@pytest.mark.parametrize(("left", "right"), [
    ("https://github.com/Owner/Repo/blob/main/a.md?q=1", "https://raw.githubusercontent.com/owner/repo/REV/b.md"),
    ("https://arxiv.org/pdf/2210.03629v1.pdf", "https://export.arxiv.org/abs/2210.03629v3"),
    ("https://huggingface.co/docs/transformers/v4.51/en/a", "https://huggingface.co/docs/transformers/main/en/b"),
    ("https://example.org/a?utm_source=one", "https://www.example.org/b#heading"),
])
def test_related_source_urls_cannot_create_independence(left, right):
    assert source_family(left) == source_family(right)


def test_independent_repositories_remain_distinct_and_local_paths_do_not_merge_a_whole_folder():
    assert source_family("https://github.com/owner/one") != source_family("https://github.com/owner/two")
    assert source_family("file:///home/Lucifer/knowledge/a.jsonl") is None
    assert source_family("https://huggingface.co/docs/transformers/main/en/cache") == source_family(
        "https://github.com/huggingface/transformers/blob/main/docs/cache.md")


@pytest.mark.asyncio
async def test_provided_solutions_need_consent_and_interaction_and_bind_source_families(tmp_path):
    store = Store(tmp_path / "state.db")
    chat = ChatService(store, SimpleNamespace())
    try:
        ids, docs = [], []
        for index, uri in enumerate(("https://github.com/owner/agent/blob/main/a.md",
                                     "https://raw.githubusercontent.com/owner/agent/rev/b.md",
                                     "https://github.com/other/llm/blob/main/a.md")):
            cid = chat.create(training_enabled=True)["id"]
            row = record(str(index), uri)
            row["question"] += f" Fixture {index}."
            row["sources"][0]["sha256"] = str(index) * 64
            doc = await chat.knowledge.attach(cid, filename="tasks.jsonl",
                content_base64=base64.b64encode(json.dumps(row).encode()).decode())
            ids.append(cid)
            docs.append(doc)
        before = await prepare_meditation_sources(store, {"mode": "supervised"})
        assert not before["examples"]
        for cid, doc in zip(ids, docs):
            chat.knowledge.read(cid, doc["id"])
        bundle = await prepare_meditation_sources(store, {"mode": "supervised"})
        assert len(bundle["manifest"]["groups"]) == 2
        shared = next(g for g in bundle["manifest"]["groups"] if "conversation:" + ids[0] in g["members"])
        assert "conversation:" + ids[1] in shared["members"]
        rows = bundle["examples"] + bundle["development_examples"] + bundle["holdout_examples"]
        assert len(rows) == 3
        assert all(r["messages"][0]["content"].startswith(record()["question"]) for r in rows)
        assert all(r["messages"][-1] == {"role": "assistant", "content": record()["answer"]} for r in rows)
        assert all(r["source_authority"] == "document_provided_solution" for r in rows)
        assert all(r["provenance"]["authority_is_truth_verification"] is False for r in rows)
        assert revalidate_meditation_sources(store, bundle)
        forged = copy.deepcopy(bundle)
        forged["examples"][0]["provenance"]["provided_sources"][0]["url"] += "forged"
        with pytest.raises(ValueError):
            revalidate_meditation_sources(store, forged)
        chunk = store.get("knowledge_chunks", docs[0]["id"] + ":0")
        store.update("knowledge_chunks", chunk["id"], text=chunk["text"].replace("Reuse", "Repeat"))
        with pytest.raises(ValueError):
            revalidate_meditation_sources(store, bundle)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_unread_source_url_change_invalidates_group_contract(tmp_path):
    store = Store(tmp_path / "state.db")
    chat = ChatService(store, SimpleNamespace())
    try:
        cid = chat.create(training_enabled=True)["id"]
        doc = await chat.knowledge.attach(cid, filename="doc.txt", content_base64=base64.b64encode(b"Fact.").decode())
        store.update("knowledge_documents", doc["id"], source="https://example.org/paper-one")
        chat.knowledge.read(cid, doc["id"])
        bundle = await prepare_meditation_sources(store, {"mode": "unsupervised"})
        store.update("knowledge_documents", doc["id"], source="https://other.org/paper-two")
        with pytest.raises(ValueError, match="Source content"):
            revalidate_meditation_sources(store, bundle)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_precommitted_roles_cannot_relabel_existing_or_connected_sources(tmp_path):
    store = Store(tmp_path / "state.db")
    chat = ChatService(store, SimpleNamespace())
    try:
        ids = []
        for index in range(10):
            cid = chat.create(training_enabled=True)["id"]
            row = record(str(index), f"https://github.com/independent/repository-{index}/guide.md")
            row["question"] += f" Environment {index}."
            row["sources"][0]["sha256"] = f"{index:064x}"
            doc = await chat.knowledge.attach(cid, filename="problems.jsonl",
                content_base64=base64.b64encode(json.dumps(row).encode()).decode())
            chat.knowledge.read(cid, doc["id"])
            ids.append(cid)
        roles = {cid: "train" if i == 0 else "development" if i == 1 else "heldout" for i, cid in enumerate(ids)}
        request = {"mode": "supervised", "conversation_ids": ids, "source_roles": roles}
        bundle = await prepare_meditation_sources(store, request)
        assert sum(g["partition"] == "heldout" for g in bundle["manifest"]["groups"]) == 8
        assert revalidate_meditation_sources(store, bundle)
        with pytest.raises(ValueError, match="conflicting"):
            await prepare_meditation_sources(store, {**request, "source_roles": {ids[-1]: "train"}})
        with pytest.raises(ValueError, match="consumed eligible"):
            await prepare_meditation_sources(store, {**request, "source_roles": {"missing": "heldout"}})
        # A second page of the training repository cannot be declared a new
        # heldout group, even with different bytes, task, chat and source URL.
        cid = chat.create(training_enabled=True)["id"]
        row = record("alias", "https://raw.githubusercontent.com/independent/repository-0/rev/other.md")
        row["question"] += " A distinct task."
        doc = await chat.knowledge.attach(cid, filename="alias.jsonl",
            content_base64=base64.b64encode(json.dumps(row).encode()).decode())
        chat.knowledge.read(cid, doc["id"])
        with pytest.raises(ValueError, match="conflicting"):
            await prepare_meditation_sources(store, {"mode": "supervised", "source_roles": {cid: "heldout"}})
    finally:
        store.close()


@pytest.mark.asyncio
async def test_common_generated_dataset_lineage_blocks_citation_based_split(tmp_path):
    store = Store(tmp_path / "state.db")
    chat = ChatService(store, SimpleNamespace())
    try:
        roles = {}
        for i, role in enumerate(("train", "heldout")):
            cid = chat.create(training_enabled=True)["id"]
            row = record(str(i), f"https://github.com/independent/repository-{i}/reference.md")
            row["question"] += f" Environment {i}."
            row["sources"][0]["sha256"] = str(i) * 64
            row["lineage"] = ["c" * 64]
            doc = await chat.knowledge.attach(cid, filename="generated.jsonl",
                content_base64=base64.b64encode(json.dumps(row).encode()).decode())
            chat.knowledge.read(cid, doc["id"])
            roles[cid] = role
        with pytest.raises(ValueError, match="conflicting"):
            await prepare_meditation_sources(store, {"mode": "supervised", "source_roles": roles})
        assert not store.list("meditation_source_partitions")
    finally:
        store.close()
