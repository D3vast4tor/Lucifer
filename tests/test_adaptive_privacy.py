"""Revoked sources cannot survive in adaptive datasets or admission evidence."""
import json

import pytest

from lucifer.chat import ChatService
from lucifer.memory import invalidate_candidate_dependents, invalidate_memory_sources
from lucifer.store import Store


def artifact(tmp_path, relative, content):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.mark.parametrize("action", ["delete", "training_revoked", "logging_revoked"])
def test_adaptive_source_purge_removes_owned_files_and_revokes_descendant_admissions(tmp_path, action):
    store = Store(tmp_path / "state.db")
    try:
        chat = ChatService(store, None)
        source = chat.create(logging_enabled=True, training_enabled=True)
        unrelated = chat.create(logging_enabled=True, training_enabled=True)
        dataset = artifact(tmp_path, "datasets/revision.jsonl", json.dumps({"conversation_id": source["id"],
            "messages": [{"role": "assistant", "content": "private training target"}]}))
        suite = artifact(tmp_path, "evaluations/development.suite.json", '{"target":"private heldout target"}')
        kept = artifact(tmp_path, "datasets/kept.jsonl", json.dumps({"conversation_id": unrelated["id"]}))
        outside = artifact(tmp_path.parent, tmp_path.name + "-outside.jsonl", "not a managed artifact")
        outside_link = tmp_path.parent / (tmp_path.name + "-outside-link.jsonl")
        outside_link.symlink_to(outside)
        link = tmp_path / "datasets" / "linked.jsonl"
        link.symlink_to(outside)
        store.put("adaptive_datasets", {"id": "revision", "run_id": "sleep_old", "dataset_path": str(dataset),
            "source_conversation_ids": [source["id"]], "sha256": "metadata"})
        store.put("adaptive_datasets", {"id": "external", "run_id": "sleep_outside", "dataset_path": str(outside),
            "source_conversation_ids": [source["id"]]})
        store.put("adaptive_datasets", {"id": "linked", "run_id": "sleep_link", "dataset_path": str(link),
            "source_conversation_ids": [source["id"]]})
        store.put("adaptive_datasets", {"id": "outside_link", "run_id": "sleep_outside_link",
            "dataset_path": str(tmp_path / "datasets" / ".." / ".." / outside_link.name),
            "source_conversation_ids": [source["id"]]})
        store.put("adaptive_datasets", {"id": "unrelated", "run_id": "sleep_new", "dataset_path": str(kept),
            "source_conversation_ids": [unrelated["id"]]})
        store.put("adaptive_development_suites", {"id": "development", "run_id": "sleep_old",
            "evaluation_suite": {"path": str(suite), "sha256": "metadata", "cases": ["private heldout target"]}})
        store.put("sleep_runs", {"id": "sleep_old", "status": "admitted", "candidate_adapter_id": "checkpoint",
            "adaptive_lineage": {"source_conversation_ids": [source["id"]], "revision_ids": ["revision"]}})
        store.put("adapters", {"id": "checkpoint", "status": "admitted", "training_run_id": "sleep_old",
            "adaptive_lineage": {"revision_ids": ["revision"]}})
        store.put("adapters", {"id": "descendant", "status": "admitted", "source_expert_ids": ["checkpoint"]})
        store.put("admission_receipts", {"id": "receipt", "adapter_id": "descendant", "status": "valid"})
        store.put("admission_receipts", {"id": "kept_receipt", "adapter_id": "other", "status": "valid",
            "source_conversation_ids": [unrelated["id"]]})
        if action == "delete":
            chat.delete(source["id"])
        else:
            chat.patch(source["id"], {"training_enabled" if action == "training_revoked" else "logging_enabled": False})
        assert not dataset.exists() and not suite.exists()
        assert not link.is_symlink() and outside.read_text() == "not a managed artifact"
        assert outside_link.is_symlink()
        assert kept.exists() and not store.get("adaptive_datasets", "unrelated").get("artifact_revoked")
        assert store.get("adaptive_datasets", "revision")["artifact_revoked"]
        assert store.get("adaptive_development_suites", "development")["evaluation_suite"] == {
            "path": str(suite), "sha256": "metadata", "unavailable": True, "revoked_conversation_id": source["id"]}
        assert store.get("sleep_runs", "sleep_old")["artifact_revoked"]
        for ident in ("checkpoint", "descendant"):
            candidate = store.get("adapters", ident)
            assert candidate["source_invalidated"] and candidate["source_revoked"] and candidate["status"] == "failed"
        assert store.get("admission_receipts", "receipt")["status"] == "revoked"
        assert store.get("admission_receipts", "kept_receipt")["status"] == "valid"
    finally:
        store.close()


def test_candidate_review_invalidates_all_checkpoints_that_consumed_the_revision(tmp_path):
    store = Store(tmp_path / "state.db")
    try:
        store.put("adaptive_datasets", {"id": "consumed", "run_id": "sleep_old",
            "example_refs": [{"candidate_id": "reviewed_target"}]})
        store.put("sleep_runs", {"id": "sleep_old", "status": "admitted", "candidate_adapter_id": "checkpoint"})
        store.put("adapters", {"id": "checkpoint", "status": "admitted",
            "adaptive_lineage": {"revision_ids": ["consumed"]}})
        store.put("adapters", {"id": "descendant", "status": "admitted", "source_expert_ids": ["checkpoint"]})
        store.put("adaptive_development_suites", {"id": "development", "run_id": "sleep_old"})
        store.put("admission_receipts", {"id": "receipt", "adapter_id": "descendant", "status": "valid"})
        result = invalidate_candidate_dependents(store, "reviewed_target")
        assert result == {"adapters_invalidated": 2}
        assert store.get("adaptive_datasets", "consumed")["source_invalidated"]
        assert store.get("adaptive_development_suites", "development")["artifact_revoked"]
        assert store.get("sleep_runs", "sleep_old")["artifact_revoked"]
        assert store.get("admission_receipts", "receipt")["status"] == "revoked"
        before = store.revision("admission_receipts", "adaptive_datasets")
        invalidate_candidate_dependents(store, "reviewed_target")
        assert store.revision("admission_receipts", "adaptive_datasets") == before
    finally:
        store.close()


def test_nested_adaptive_lineage_is_invalidated_without_opening_dataset_files(tmp_path):
    store = Store(tmp_path / "state.db")
    try:
        store.put("sleep_runs", {"id": "sleep_old", "adaptive": {"lineage": {"source_conversation_ids": ["source"]}}})
        store.put("adaptive_datasets", {"id": "revision", "run_id": "sleep_old", "source_conversation_ids": ["source"],
            "dataset_path": "/not/a/readable/dataset.jsonl"})
        store.put("admission_receipts", {"id": "receipt", "source_conversation_ids": ["source"], "status": "valid"})
        invalidate_memory_sources(store, "source", reason="source_changed")
        assert store.get("sleep_runs", "sleep_old")["source_invalidated"]
        assert store.get("adaptive_datasets", "revision")["artifact_revoked"]
        assert store.get("admission_receipts", "receipt")["status"] == "revoked"
    finally:
        store.close()
