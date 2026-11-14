import json
from types import SimpleNamespace

import pytest

from lucifer.chat import ChatService
from lucifer.memory import (capture_source_snapshot, conflicted_source_messages, initialize_memory, invalidate_memory_sources,
                            reconcile_after_turn, reconcile_memory, register_approved_memory, validate_source_snapshot)
from lucifer.store import Store
from lucifer.schemas import ChatRequest


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "memory.sqlite3")
    yield result
    result.close()


def source(store, cid="c", suffix="", question="Which deployment port?", answer="8080", training=True):
    try:
        store.get("conversations", cid)
    except KeyError:
        store.put("conversations", {"id": cid, "logging_enabled": True, "training_enabled": training})
    user, assistant = cid + "u" + suffix, cid + "a" + suffix
    store.put("messages", {"id": user, "conversation_id": cid, "role": "user", "content": question})
    store.put("messages", {"id": assistant, "conversation_id": cid, "role": "assistant", "content": answer})
    store.put("feedback", {"conversation_id": cid, "message_id": assistant, "label": "good"})
    return user, assistant


def example(store, cid, user, assistant, *, answer="8080", scope="project:alpha", task=None, **extra):
    return {"messages": [{"role": "user", "content": "Which deployment port?"}, {"role": "assistant", "content": answer}],
        "task_id": task or assistant, "scope_id": scope, "fact_key": "deployment_port", "source_message_ids": [user, assistant],
        "source_bindings": [capture_source_snapshot(store, cid, [user, assistant])],
        "validation": {"status": "verified", "method": "reviewed_exact_copy"}, **extra}


def test_snapshot_binds_actual_evidence_but_not_new_turns_or_duplicate_positive_reviews(store):
    user, answer = source(store)
    bound = capture_source_snapshot(store, "c", [user, answer])
    source(store, suffix="later", question="A separate question", answer="Other answer")
    store.put("feedback", {"conversation_id": "c", "message_id": answer, "label": "user_liked"})
    assert validate_source_snapshot(store, bound)
    store.put("feedback", {"conversation_id": "c", "message_id": answer, "label": "correct", "correction": "9090"})
    with pytest.raises(ValueError, match="review changed"):
        validate_source_snapshot(store, bound)
    tampered = {**bound, "message_ids": []}
    with pytest.raises(ValueError, match="snapshot changed"):
        validate_source_snapshot(store, tampered)


def test_reconciliation_is_idempotent_and_same_task_correction_supersedes_stale_target(store):
    _, answer = source(store)
    chat = ChatService(store, None)
    first = reconcile_memory(store)
    assert first["created"] == first["active"] == 1
    old = chat.memory()[0]
    revision = store.revision("memory_items")
    assert reconcile_memory(store)["unchanged"] == 1
    assert store.revision("memory_items") == revision
    store.put("feedback", {"conversation_id": "c", "message_id": answer, "label": "correct", "correction": "9090"})
    invalidate_memory_sources(store, "c", answer, reason="review_corrected")
    result = reconcile_memory(store, ["c"])
    current = chat.memory()[0]
    assert result["created"] == 1 and current["version"] == 2
    assert current["fact_id"] == old["fact_id"] and current["task_id"] == old["task_id"]
    assert "9090" in current["content"] and "8080" not in current["content"]
    assert store.get("memory_items", old["id"])["status"] == "superseded"


def test_conflicting_same_scope_assertions_are_withheld_not_latest_wins(store):
    source(store)
    source(store, suffix="other", answer="9090")
    chat = ChatService(store, None)
    result = reconcile_memory(store)
    assert result["active"] == 0 and chat.memory() == []
    assert {row["status"] for row in store.list("memory_items")} == {"conflicted"}
    assert conflicted_source_messages(store) == {("c", "ca"), ("c", "caother")}
    store.put("feedback", {"conversation_id": "c", "message_id": "caother", "label": "bad"})
    invalidate_memory_sources(store, "c", "caother", reason="negative_review")
    reconcile_memory(store, ["c"])
    assert len(chat.memory()) == 1 and "8080" in chat.memory()[0]["content"]
    assert conflicted_source_messages(store) == set()


def test_later_unreviewed_correction_withholds_old_task_before_new_answer_review(store):
    source(store)
    chat = ChatService(store, None)
    reconcile_memory(store)
    store.put("messages", {"id": "correction-user", "conversation_id": "c", "role": "user", "content": "Actually, use port 9090 instead."})
    store.put("messages", {"id": "unreviewed-answer", "conversation_id": "c", "role": "assistant", "content": "New unreviewed answer"})
    result = reconcile_after_turn(store, "c")
    assert result["invalidated"] == 1 and chat.memory() == []
    assert reconcile_memory(store)["active"] == 0  # Old approved pair cannot re-enter.


def test_explicit_new_task_protects_unrelated_prior_facts(store):
    source(store)
    chat = ChatService(store, None)
    reconcile_memory(store)
    store.put("messages", {"id": "new-task", "conversation_id": "c", "role": "user", "content": "New topic: choose dinner"})
    store.put("messages", {"id": "new-correction", "conversation_id": "c", "role": "user", "content": "Actually, make it vegetarian"})
    reconcile_after_turn(store, "c")
    assert len(chat.memory()) == 1 and "8080" in chat.memory()[0]["content"]


def test_changed_literal_constraint_is_withheld_without_global_latest_wins(store):
    source(store, question="Budget: 50", answer="Use a 50-unit budget")
    reconcile_memory(store)
    store.put("messages", {"id": "budget-change", "conversation_id": "c", "role": "user", "content": "Budget: 100"})
    reconcile_after_turn(store, "c")
    assert ChatService(store, None).memory() == []


def test_negative_positive_review_cycle_gets_new_evidence_epoch(store):
    user, answer = source(store)
    old = capture_source_snapshot(store, "c", [user, answer])
    store.update("messages", answer, learning_revision=1)
    store.put("feedback", {"conversation_id": "c", "message_id": answer, "label": "bad"})
    invalidate_memory_sources(store, "c", answer, reason="negative_review")
    store.update("messages", answer, learning_revision=2)
    store.put("feedback", {"conversation_id": "c", "message_id": answer, "label": "good"})
    current = capture_source_snapshot(store, "c", [user, answer])
    assert current["snapshot_sha256"] != old["snapshot_sha256"]
    assert current["messages"][-1]["learning_revision"] == 2
    with pytest.raises(ValueError, match="review changed"):
        validate_source_snapshot(store, old)
    assert validate_source_snapshot(store, current)


def test_consent_cycle_invalidates_old_training_evidence_but_preserves_manual_note(store):
    user, answer = source(store)
    chat = ChatService(store, None)
    old = capture_source_snapshot(store, "c", [user, answer])
    manual = register_approved_memory(store, {"id": "manual", "conversation_id": "c", "content": "Approved note"}, source_message_id=answer)
    chat.patch("c", {"training_enabled": False})
    chat.patch("c", {"training_enabled": True})
    assert store.get("conversations", "c")["learning_revision"] == 2
    with pytest.raises(ValueError, match="review changed"):
        validate_source_snapshot(store, old)
    assert validate_source_snapshot(store, manual["source_bindings"][0], require_training=False)
    assert [item["id"] for item in chat.memory()] == ["manual"]


def test_cross_conversation_scope_is_explicit_and_supersession_needs_user_evidence(store):
    one = source(store, "one")
    two = source(store, "two", answer="9090")
    chat = ChatService(store, None)
    reconcile_memory(store)
    assert len(chat.memory()) == 2  # Different conversation scopes do not overwrite.
    shared = [example(store, "one", *one), example(store, "two", *two, answer="9090")]
    reconcile_memory(store, examples=shared)
    shared_rows = [row for row in store.list("memory_items") if row.get("scope_id") == "project:alpha"]
    assert {row["status"] for row in shared_rows} == {"conflicted"}
    user, assistant = source(store, "three", question="Correction: replace the old deployment port with 9090", answer="9090")
    update = example(store, "three", user, assistant, answer="9090", task="correction-task",
        supersedes=[one[1]], validation={"status": "verified", "method": "explicit_user_correction", "explicit_supersession": True})
    reconcile_memory(store, examples=[update])
    shared_rows = [row for row in store.list("memory_items") if row.get("scope_id") == "project:alpha"]
    assert all(row["status"] == "superseded" for row in shared_rows if row["task_id"] == one[1])
    assert {row["value_sha256"] for row in shared_rows if row["status"] == "active"} == {next(row["value_sha256"] for row in shared_rows if row["task_id"] == "correction-task")}


def test_model_claimed_supersession_without_explicit_user_correction_is_withheld(store):
    first = source(store, "one")
    second = source(store, "two", answer="9090")
    reconcile_memory(store, examples=[example(store, "one", *first)])
    claimed = example(store, "two", *second, answer="9090", supersedes=[first[1]],
        validation={"status": "verified", "method": "reviewed_exact_copy", "explicit_supersession": True})
    reconcile_memory(store, examples=[claimed])
    assert not any(row.get("active") for row in store.list("memory_items"))


def test_same_task_key_does_not_allow_a_different_approved_source_to_win_by_recency(store):
    first = source(store, "one", answer="TokenABC")
    second = source(store, "two", answer="tokenabc")
    material = [example(store, "one", *first, answer="TokenABC", task="shared-task"),
                example(store, "two", *second, answer="tokenabc", task="shared-task")]
    reconcile_memory(store, examples=material)
    assert {row["status"] for row in store.list("memory_items")} == {"conflicted"}
    assert conflicted_source_messages(store) == {("one", first[1]), ("two", second[1])}


def test_training_snapshot_cannot_authorize_an_empty_evidence_set(store):
    source(store)
    with pytest.raises(ValueError, match="at least one"):
        capture_source_snapshot(store, "c", [])


def test_feedback_memory_recovers_explicit_scope_and_binds_its_antecedent(store):
    first = source(store, question="Scope: project-alpha", answer="Acknowledged")
    source(store, suffix="followup", question="Which deployment port?", answer="8080")
    reconcile_memory(store)
    rows = store.list("memory_items")
    assert {item["scope_id"] for item in rows} == {"project-alpha"}
    latest = next(item for item in rows if item["source_message_id"] == "cafollowup")
    assert first[0] in latest["source_bindings"][0]["message_ids"]
    assert latest["fact_key"] == "task:" + latest["task_id"]
    store.update("messages", first[0], content="Scope: project-beta", learning_revision=1)
    reconcile_after_turn(store, "c")
    assert ChatService(store, None).memory() == []


def test_later_explicit_scope_withholds_earlier_implicitly_cid_scoped_memory(store):
    source(store)
    reconcile_memory(store)
    store.put("messages", {"id": "scope-later", "conversation_id": "c", "role": "user", "content": "Scope: project-alpha"})
    reconcile_after_turn(store, "c")
    assert ChatService(store, None).memory() == []


def test_rejected_draft_withholds_its_memory_and_consuming_adapter(store):
    user, answer = source(store)
    store.put("curriculum_candidates", {"id": "candidate", "status": "verified", "content_hash": "immutable"})
    item = example(store, "c", user, answer, candidate_id="candidate", content_hash="immutable")
    reconcile_memory(store, examples=[item])
    store.put("adapters", {"id": "trained", "status": "candidate", "curriculum_example_refs": [{"candidate_id": "candidate"}]})
    store.update("curriculum_candidates", "candidate", status="obsolete")
    reconcile_memory(store, examples=[])
    assert ChatService(store, None).memory() == []
    assert store.get("adapters", "trained")["source_invalidated"]


def test_rejecting_one_draft_does_not_resolve_conflicting_source_assertions(store):
    one, two = source(store, "one"), source(store, "two", answer="9090")
    material = []
    for cid, ids, target in [("one", one, "8080"), ("two", two, "9090")]:
        store.put("curriculum_candidates", {"id": cid, "status": "verified", "content_hash": cid})
        material.append(example(store, cid, *ids, answer=target, candidate_id=cid, content_hash=cid))
    reconcile_memory(store, examples=material)
    store.update("curriculum_candidates", "one", status="obsolete")
    reconcile_memory(store, examples=[])
    assert conflicted_source_messages(store) == {("one", one[1]), ("two", two[1])}
    assert ChatService(store, None).memory() == []


def test_conflicting_field_sources_also_withhold_their_canonical_parent_notes(store):
    one, two = source(store, "one"), source(store, "two", answer="9090")
    fields = [example(store, "one", *one), example(store, "two", *two, answer="9090")]
    summaries = [example(store, "one", *one, scope="one", fact_key="task:one", answer="A canonical source summary"),
                 example(store, "two", *two, scope="two", fact_key="task:two", answer="Another canonical source summary")]
    reconcile_memory(store, examples=fields + summaries)
    assert any(item["status"] == "active" for item in store.list("memory_items"))
    assert ChatService(store, None).memory() == []


@pytest.mark.asyncio
async def test_actual_chat_commit_withholds_old_memory_before_new_reply_has_feedback(store):
    source(store)
    class Manager:
        specs = {"model": SimpleNamespace(effective_thinking=lambda value: "off")}
        def current(self):
            return {"id": "model"}
        async def stream_events(self, *args, **kwargs):
            yield "delta", {"content": "Unreviewed reply to the correction"}
            yield "finish", {"finish_reason": "stop"}
    chat = ChatService(store, Manager())
    reconcile_memory(store)
    request = ChatRequest(messages=[{"role": "user", "content": "Actually, use port 9090 instead."}])
    events = [event async for event in chat.turn(request, "c")]
    assert events[-1][0] == "done" and chat.memory() == []


def test_negative_review_invalidates_candidates_adapters_descendants_and_run_sources(store):
    user, assistant = source(store)
    snapshot = capture_source_snapshot(store, "c", [user, assistant])
    reconcile_memory(store)
    store.put("curriculum_candidates", {"id": "draft", "source_bindings": [snapshot], "status": "verified", "messages": [{"content": "immutable"}]})
    store.put("adapters", {"id": "a", "status": "candidate", "source_conversation_ids": ["c"]})
    store.put("adapters", {"id": "child", "status": "active", "source_expert_ids": ["a"]})
    store.put("sleep_runs", {"id": "sleep_x", "candidate_adapter_id": "child"})
    store.put("feedback", {"conversation_id": "c", "message_id": assistant, "label": "bad"})
    counts = invalidate_memory_sources(store, "c", assistant, reason="negative_review")
    assert counts == {"memories_invalidated": 1, "candidates_obsoleted": 1, "adapters_invalidated": 2, "runs_invalidated": 1}
    assert store.get("curriculum_candidates", "draft")["status"] == "obsolete"
    assert store.get("curriculum_candidates", "draft")["messages"] == [{"content": "immutable"}]
    assert store.get("adapters", "a")["status"] == "failed"
    assert store.get("adapters", "child")["status"] == "active"
    assert store.get("adapters", "child")["source_revoked"]
    assert store.get("sleep_runs", "sleep_x")["artifact_revoked"]
    assert ChatService(store, None).memory() == []


def test_manual_approved_notes_use_logging_consent_and_legacy_unbound_notes_are_withheld(store):
    _, assistant = source(store)
    store.put("memory_items", {"id": "old", "conversation_id": "c", "content": "Stale legacy answer"})
    store.put("memory_items", {"id": "approved", "conversation_id": "c", "source": "approved_agent_summary", "content": "Approved durable note"})
    chat = ChatService(store, None)
    assert [row["id"] for row in chat.memory()] == ["approved"]
    registered = register_approved_memory(store, {"id": "bound", "conversation_id": "c", "content": "Bound approved note"}, source_message_id=assistant)
    assert registered["source_bindings"][0]["require_training"] is False
    chat.patch("c", {"training_enabled": False})
    assert {row["id"] for row in chat.memory()} == {"approved", "bound"}
    with pytest.raises(ValueError, match="cannot authorize training"):
        validate_source_snapshot(store, registered["source_bindings"][0])
    chat.patch("c", {"logging_enabled": False})
    assert chat.memory() == [] and store.list("memory_items") == []


def test_unverified_or_stale_examples_never_become_retrievable_memory(store):
    user, assistant = source(store)
    valid = example(store, "c", user, assistant)
    unverified = {**valid, "validation": {"status": "unknown"}}
    assert reconcile_memory(store, examples=[unverified])["rejected_examples"] == 1
    store.update("messages", assistant, content="Edited source answer")
    assert reconcile_memory(store, examples=[valid])["rejected_examples"] == 1
    assert ChatService(store, None).memory() == []


def test_privacy_purge_removes_multisource_generated_text_and_owned_evaluations_not_weights(store, tmp_path):
    user, answer = source(store, "private")
    source(store, "other")
    snapshot = capture_source_snapshot(store, "private", [user, answer])
    private = "PRIVATE_DERIVED_FACT"
    store.put("curriculum_candidates", {"id": "generated", "conversation_id": "other", "source_bindings": [snapshot], "messages": [{"content": private}]})
    store.put("curriculum_drafts", {"id": "draft", "source_bindings": [snapshot], "candidate_ids": ["generated"]})
    folder = tmp_path / "evaluations"
    folder.mkdir()
    suite = folder / "sleep_private.suite.json"
    suite.write_text(private)
    weights = tmp_path / "adapters" / "trained.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"trained checkpoint remains")
    outside = tmp_path.parent / (tmp_path.name + "-outside.suite.json")
    outside.write_text("external file")
    binding = {"path": str(suite), "sha256": "frozen", "coverage": {"unsupported_topics": [private]}}
    store.put("sleep_runs", {"id": "sleep_p", "source_conversation_ids": ["private"], "evaluation_suite": binding,
        "curriculum_manifest": {"source_text": private}, "candidate_adapter_id": "trained"})
    store.put("adapters", {"id": "trained", "path": str(weights), "status": "active", "source_conversation_ids": ["private"], "evaluation_suite": binding})
    store.put("evaluation_results", {"id": "evaluation", "adapter_id": "trained", "cases": [{"baseline_answer": private}]})
    store.put("adapters", {"id": "unrelated", "status": "candidate", "source_conversation_ids": ["other"],
        "evaluation_suite": {"path": str(outside), "coverage": {"unsupported_topics": ["KEEP_OTHER_SCOPE"]}}})
    ChatService(store, None).patch("private", {"training_enabled": False})
    assert store.list("curriculum_candidates") == store.list("curriculum_drafts") == store.list("evaluation_results") == []
    assert not suite.exists() and weights.read_bytes() == b"trained checkpoint remains" and outside.exists()
    assert store.get("adapters", "trained")["evaluation_suite"]["unavailable"]
    assert "KEEP_OTHER_SCOPE" in json.dumps(store.get("adapters", "unrelated"))
    assert private not in json.dumps(store.list("sleep_runs") + store.list("adapters"))
    outside.unlink()


def test_reasoning_and_identifiers_do_not_enter_reconciled_content(store):
    source(store, answer="<think>PRIVATE_REASONING</think>Final contact alice@example.com")
    reconcile_memory(store)
    content = json.dumps(store.list("memory_items"))
    assert "PRIVATE_REASONING" not in content and "alice@example.com" not in content
    assert "[EMAIL]" in content


def test_nested_store_transaction_rolls_back_documents_and_registry_revisions(store):
    original = store.revision("memory_items", "feedback", "settings")
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put("memory_items", {"id": "partial", "content": "Partial"})
            with store.transaction():
                store.put("feedback", {"id": "review", "label": "good"})
                store.set_setting("memory_marker", True)
            raise RuntimeError("Abort reconciliation")
    assert store.list("memory_items") == store.list("feedback") == []
    assert store.setting("memory_marker") is None
    assert store.revision("memory_items", "feedback", "settings") == original


def test_reconciliation_has_no_partial_commit_on_write_failure(store, monkeypatch):
    source(store)
    source(store, suffix="second", question="Which deployment host?", answer="localhost")
    original, calls = store.put, []
    def fail(kind, data):
        result = original(kind, data)
        if kind == "memory_items":
            calls.append(data["id"])
            if len(calls) == 2:
                raise RuntimeError("Injected write failure")
        return result
    monkeypatch.setattr(store, "put", fail)
    before = store.revision("memory_items")
    with pytest.raises(RuntimeError, match="Injected"):
        reconcile_memory(store)
    assert store.list("memory_items") == [] and store.revision("memory_items") == before
    assert initialize_memory(store) == {"manual_preserved": 0, "legacy_invalidated": 0}
