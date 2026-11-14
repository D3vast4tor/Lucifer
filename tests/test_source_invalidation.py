"""Source revocation affects live routing and previously retrieved memories."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.curriculum import prepare_curriculum, review_candidate
from lucifer.inference import ModelManager
from lucifer.memory import conflicted_source_messages, reconcile_memory
from lucifer.store import Store


def _settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", models_file=tmp_path / "missing-models.yaml",
                    eval_file=tmp_path / "missing-evals.yaml")


def _source(store, cid, value):
    store.put("conversations", {"id": cid, "logging_enabled": True, "training_enabled": True})
    store.put("messages", {"id": cid + "u", "conversation_id": cid, "role": "user",
                           "content": "Plan the work trip.\nScope: atlas\nDestination: " + value})
    store.put("messages", {"id": cid + "a", "conversation_id": cid, "role": "assistant",
                           "content": "Destination: " + value})
    store.put("feedback", {"conversation_id": cid, "message_id": cid + "a", "label": "good"})


@pytest.mark.parametrize("action", ["negative_feedback", "corrected_feedback", "training_revoked", "logging_revoked", "deleted", "candidate_rejected"])
def test_api_source_invalidation_detaches_active_adapter_and_next_chat_uses_base(tmp_path, action):
    app = create_app(_settings(tmp_path), autoload="demo")
    with TestClient(app) as client:
        conv = client.post("/conversations", json={"logging_enabled": True, "training_enabled": True}).json()
        answer = client.post("/chat", json={"conversation_id": conv["id"], "messages": [
            {"role": "user", "content": "What is the capital of France?"}]}).json()
        review = {"conversation_id": conv["id"], "message_id": answer["message_id"], "label": "good"}
        assert client.post("/feedback", json=review).status_code == 201
        prepared = asyncio.run(prepare_curriculum(app.state.store, [conv["id"]], options={"max_generated_examples": 0}))
        candidate = (prepared["examples"] + prepared["holdout_examples"])[0]
        app.state.store.put("adapters", {"id": "active-expert", "base_model_id": "demo", "format": "demo", "status": "active",
            "source_conversation_ids": [conv["id"]], "source_message_ids": [answer["message_id"]],
            "curriculum_example_refs": [{"candidate_id": candidate["id"]}]})
        app.state.store.set_setting("active_adapter", "active-expert")
        assert client.post("/models/load", json={"model_id": "demo"}).json()["adapter_id"] == "active-expert"
        if action == "negative_feedback":
            response = client.post("/feedback", json={**review, "label": "bad"})
        elif action == "corrected_feedback":
            response = client.post("/feedback", json={**review, "label": "correct", "correction": "Paris is France's capital."})
        elif action == "training_revoked":
            response = client.patch(f"/conversations/{conv['id']}", json={"training_enabled": False})
        elif action == "logging_revoked":
            response = client.patch(f"/conversations/{conv['id']}", json={"logging_enabled": False})
        elif action == "deleted":
            response = client.delete(f"/conversations/{conv['id']}")
        else:
            response = client.post(f"/sleep/examples/{candidate['id']}/review", json={"decision": "reject", "content_hash": candidate["content_hash"]})
        assert response.status_code in {200, 201}, response.text
        assert client.get("/system/status").json()["active_adapter"] is None
        assert app.state.manager.current()["adapter_id"] is None
        assert app.state.manager._runtime_adapter is None
        assert app.state.store.get("adapters", "active-expert")["status"] == "failed"
        other = client.post("/conversations", json={"logging_enabled": True}).json()
        result = client.post("/chat", json={"conversation_id": other["id"], "messages": [
            {"role": "user", "content": "What is the capital of France?"}]})
        assert result.status_code == 200, result.text
        assert app.state.store.get("messages", result.json()["message_id"])["adapter_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["source_revoked", "source_invalidated"])
async def test_cached_adapter_copy_cannot_bypass_current_registry_revocation(tmp_path, flag):
    settings = _settings(tmp_path)
    store = Store(settings.data_dir / "state.db")
    manager = ModelManager(settings, store)
    try:
        await manager.load("demo")
        cached = store.put("adapters", {"id": "cached", "base_model_id": "demo", "format": "demo", "status": "active"})
        await manager.activate_adapter(cached)
        store.update("adapters", cached["id"], **{flag: True})
        assert not cached.get(flag)
        with pytest.raises(ValueError, match="revoked or superseded"):
            await manager.complete([{"role": "user", "content": "hello"}], adapter=cached)
        with pytest.raises(ValueError, match="revoked or superseded"):
            await manager.activate_adapter(cached)
        # Explicitly supplied invalidation also wins over a clean registry row.
        store.update("adapters", cached["id"], **{flag: False})
        with pytest.raises(ValueError, match="revoked or superseded"):
            await manager.activate_adapter({**cached, flag: True})
        await manager.activate_adapter(None)
        assert await manager.complete([{"role": "user", "content": "2+2"}]) == "4"
    finally:
        await manager.close()
        store.close()


@pytest.mark.parametrize("flag", ["source_revoked", "source_invalidated"])
def test_backend_restart_does_not_restore_invalidated_active_selection(tmp_path, flag):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, autoload="demo")) as client:
        client.app.state.store.put("adapters", {"id": "old", "base_model_id": "demo", "format": "demo", "status": "active", flag: True})
        client.app.state.store.set_setting("active_adapter", "old")
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        assert client.get("/system/status").json()["active_adapter"] is None
        assert app.state.manager.current().get("adapter_id") is None
        assert app.state.store.get("adapters", "old")["status"] == "failed"


@pytest.mark.asyncio
async def test_conflict_witnesses_withhold_existing_notes_after_rejected_drafts_without_retrieval_probes(tmp_path, monkeypatch):
    store = Store(tmp_path / "state.db")
    try:
        _source(store, "rome", "Rome")
        _source(store, "paris", "Paris")
        chat = ChatService(store, None)
        reconcile_memory(store)
        assert len(chat.memory()) == 2  # Earlier positive feedback made task notes.
        prepared = await prepare_curriculum(store, options={"max_generated_examples": 8})
        for row in prepared["pending"]:
            review_candidate(store, row["id"], decision="reject", content_hash=row["content_hash"])
        assert not prepared["examples"] and not prepared["holdout_examples"]
        outcome = reconcile_memory(store, examples=[])
        assert outcome["conflicted"] == 2 and chat.memory() == []
        assert conflicted_source_messages(store) == {("rome", "romea"), ("paris", "parisa")}
        assert {item["status"] for item in store.list("memory_items")} == {"conflicted"}
        revision = store.revision("memory_items")
        marker_revision = store.revision("memory_source_conflicts")
        reconcile_memory(store, examples=[])
        assert store.revision("memory_items") == revision
        assert store.revision("memory_source_conflicts") == marker_revision
        original_list = store.list
        def cached_only(kind, **kwargs):
            assert kind not in {"messages", "feedback", "curriculum_candidates"}, "retrieval must consume reconciled cached status"
            return original_list(kind, **kwargs)
        monkeypatch.setattr(store, "list", cached_only)
        assert chat.memory() == []
        monkeypatch.setattr(store, "list", original_list)
        # Revoking/deleting the source also removes its derived cached markers.
        chat.delete("rome")
        assert all(row["conversation_id"] != "rome" for row in store.list("memory_source_conflicts"))
    finally:
        store.close()


@pytest.mark.asyncio
async def test_changed_source_binding_retires_conflict_markers_at_reconcile(tmp_path):
    store = Store(tmp_path / "state.db")
    try:
        _source(store, "a", "Rome")
        _source(store, "b", "Paris")
        reconcile_memory(store)
        await prepare_curriculum(store, options={"max_generated_examples": 8})
        reconcile_memory(store, examples=[])
        assert len(conflicted_source_messages(store)) == 2
        for cid in ("a", "b"):
            store.put("messages", {"id": cid + "correction", "conversation_id": cid, "role": "user", "content": "Correction: both old trip destinations are withdrawn."})
        reconcile_memory(store, examples=[])
        assert conflicted_source_messages(store) == set()
        assert all(not item["active"] for item in store.list("memory_source_conflicts"))
        assert ChatService(store, None).memory() == []
    finally:
        store.close()
