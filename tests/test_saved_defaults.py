"""Default saving and explicit save consent do not opt into training or interrupt."""
import json

from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings
from lucifer.schemas import AgentRequest
from test_agent_sessions import SessionManager


def _settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", models_file=tmp_path / "missing.yaml", agent_workspace=tmp_path)


def test_new_conversations_persist_by_default_explicit_private_choice_does_not_leak_to_next(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        defaults = client.get("/system/bootstrap").json()["status"]["privacy_defaults"]
        assert defaults == {"logging_enabled": True, "training_enabled": False}
        saved = client.post("/conversations", json={}).json()
        assert saved["logging_enabled"] is True and saved["training_enabled"] is False
        response = client.post("/chat", json={"conversation_id": saved["id"], "messages": [{"role": "user", "content": "2+2"}]})
        assert response.status_code == 200
        assert len(app.state.store.list("messages", conversation_id=saved["id"])) == 2
        private = client.post("/conversations", json={"logging_enabled": False}).json()
        assert private["logging_enabled"] is False and private["training_enabled"] is False
        assert client.post("/chat", json={"conversation_id": private["id"], "messages": [{"role": "user", "content": "Private 2+2"}]}).status_code == 200
        assert app.state.store.list("messages", conversation_id=private["id"]) == []
        next_chat = client.post("/conversations", json={}).json()
        assert next_chat["logging_enabled"] is True and next_chat["training_enabled"] is False
        assert app.state.chat.create()["logging_enabled"] is True  # Internal agent creation shares the default.
    with TestClient(create_app(settings, autoload="demo")) as restarted:
        assert len(restarted.get(f"/conversations/{saved['id']}").json()["messages"]) == 2
        assert restarted.get(f"/conversations/{private['id']}").json()["messages"] == []


def test_saving_running_private_agent_keeps_provider_alive_and_checkpoints_public_context(tmp_path):
    app = create_app(_settings(tmp_path), autoload="demo")
    with TestClient(app) as client:
        manager = SessionManager([{"reasoning": "PRIVATE_PROVIDER_CHANNEL_NOT_FOR_DISK"}])
        app.state.agent.manager = manager
        app.state.chat.manager = manager
        conversation = client.post("/conversations", json={"logging_enabled": False}).json()
        run = client.portal.call(app.state.agent.start, AgentRequest(goal="Retain the public mission", workspace=str(tmp_path),
            conversation_id=conversation["id"], thinking="high"))
        client.portal.call(manager.entered.wait)
        assert app.state.agent.busy and app.state.store.list("agent_checkpoints") == []
        saved = client.patch(f"/conversations/{conversation['id']}", json={"logging_enabled": True})
        assert saved.status_code == 200, saved.text
        assert saved.json()["logging_enabled"] is True and saved.json()["training_enabled"] is False
        checkpoint = app.state.store.get("agent_checkpoints", run["id"])
        assert "Retain the public mission" in json.dumps(checkpoint)
        assert "PRIVATE_PROVIDER_CHANNEL_NOT_FOR_DISK" not in json.dumps(checkpoint)
        assert app.state.agent.busy and not manager.cancelled
        assert not app.state.agent.states[run["id"]].context.cancel.is_set()
        # Other changes still need an idle run; logging consent cannot smuggle
        # training, policy changes or revocation into the exception.
        for changes in ({"training_enabled": True}, {"logging_enabled": False}, {"title": "Changed"},
                        {"logging_enabled": True, "training_enabled": True}):
            assert client.patch(f"/conversations/{conversation['id']}", json=changes).status_code == 409
        assert client.patch(f"/conversations/{conversation['id']}", json={"logging_enabled": True}).status_code == 200
        assert app.state.agent.busy and not manager.cancelled
        assert client.post(f"/agent/runs/{run['id']}/stop").status_code == 200
