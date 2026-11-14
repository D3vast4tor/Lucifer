import json

from lucifer.chat import ChatService
from lucifer.store import Store


def test_consent_revocation_removes_dataset_and_invalidates_candidate(tmp_path):
    store = Store(tmp_path / "lucifer.sqlite3")
    chat = ChatService(store, None)
    conv = chat.create(logging_enabled=True, training_enabled=True)
    folder = tmp_path / "datasets"
    folder.mkdir()
    path = folder / "train.jsonl"
    path.write_text(json.dumps({"conversation_id": conv["id"], "messages": [{"role": "user", "content": "private source"}]}))
    run = store.put("training_runs", {"id": "train_1", "status": "awaiting_promotion", "dataset_path": str(path), "candidate_adapter_id": "candidate-1", "source_conversation_ids": [conv["id"]]})
    store.put("adapters", {"id": "candidate-1", "status": "candidate"})
    store.put("feedback", {"conversation_id": conv["id"], "message_id": "m1", "label": "good"})
    chat.patch(conv["id"], {"training_enabled": False})
    assert not path.exists()
    assert store.get("training_runs", run["id"])["artifact_revoked"]
    assert store.get("adapters", "candidate-1")["source_revoked"]
    assert store.get("adapters", "candidate-1")["status"] == "failed"
    assert len(store.list("feedback")) == 1  # Logging consent remains on.
    store.close()


def test_dataset_deletion_cannot_escape_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    store = Store(data_dir / "lucifer.sqlite3")
    chat = ChatService(store, None)
    conv = chat.create(logging_enabled=True, training_enabled=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("sensitive but not a managed artifact")
    store.put("training_runs", {"id": "train_1", "dataset_path": str(outside), "source_conversation_ids": [conv["id"]]})
    chat.delete(conv["id"])
    assert outside.exists()
    store.close()


def test_model_switch_archives_incompatible_active_adapter(tmp_path):
    from fastapi.testclient import TestClient
    from lucifer.api import create_app
    from lucifer.config import Settings
    config = tmp_path / "models.yaml"
    config.write_text("models:\n  - {id: model-a, backend: demo}\n  - {id: model-b, backend: demo}\n")
    app = create_app(Settings(data_dir=tmp_path / "data", models_file=config), autoload="model-a")
    with TestClient(app) as client:
        app.state.store.put("adapters", {"id": "adapter-a", "base_model_id": "model-a", "format": "demo", "status": "active"})
        app.state.store.set_setting("active_adapter", "adapter-a")
        assert client.post("/models/load", json={"model_id": "model-a"}).json()["adapter_id"] == "adapter-a"
        assert client.post("/models/load", json={"model_id": "model-b"}).status_code == 200
        assert client.get("/system/status").json()["active_adapter"] is None
        assert app.state.store.get("adapters", "adapter-a")["status"] == "archived"


def test_second_backend_does_not_mutate_first_registry(tmp_path):
    import pytest
    from fastapi.testclient import TestClient
    from lucifer.api import create_app
    from lucifer.config import Settings
    settings = Settings(data_dir=tmp_path, models_file=tmp_path / "missing.yaml")
    first = create_app(settings, autoload="demo")
    with TestClient(first):
        second = create_app(settings)
        with pytest.raises(RuntimeError, match="Another Lucifer backend"):
            with TestClient(second):
                pass
        assert first.state.manager.current()["status"] == "loaded"
