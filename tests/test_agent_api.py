import asyncio
import json

from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings


def wire_provider(app, monkeypatch, *, block=False):
    manager = app.state.manager
    manager.capabilities.invalidate("demo")
    manager.capabilities.record("demo", "api-test", declared={"chat_completions": True, "tools": True})
    app.state.store.update("models", "demo", effective_context_length=65536)
    calls = []
    async def count(*args, **kwargs):
        return {"input_tokens": 200, "method": "conservative_estimate", "estimated": True}
    async def events(messages, tools, **kwargs):
        calls.append(messages)
        if block:
            await asyncio.Event().wait()
        yield "delta", {"content": "Agent final answer"}
        yield "usage", {"total_tokens": 210}
        yield "finish", {"finish_reason": "stop"}
    monkeypatch.setattr(manager, "count_tokens", count)
    monkeypatch.setattr(manager, "agent_events", events)
    return calls


def test_agent_api_stream_replay_idempotency_and_feedback_consent(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml", agent_workspace=tmp_path)
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        calls = wire_provider(app, monkeypatch)
        private = client.post("/conversations", json={"logging_enabled": False}).json()
        initial = client.get("/system/bootstrap").json()
        request = {"goal": "Complete a private task", "idempotency_key": "request-one", "conversation_id": private["id"]}
        response = client.post("/agent/runs", json=request)
        assert response.status_code == 202, response.text
        run = response.json()
        streamed = client.get(f"/agent/runs/{run['id']}/events")
        assert streamed.status_code == 200 and "event: final" in streamed.text and "event: done" in streamed.text
        result = client.get(f"/agent/runs/{run['id']}").json()
        assert result["status"] == "completed" and result["answer"] == "Agent final answer"
        assert client.post("/agent/runs", json=request).json()["id"] == run["id"]
        assert len(calls) == 1
        last = result["last_sequence"]
        assert client.get(f"/agent/runs/{run['id']}/events", headers={"Last-Event-ID": str(last)}).text == ""
        telemetry = client.get("/system/telemetry").json()
        assert telemetry["revision"] == initial["revision"]
        assert telemetry["agent_runs"][-1]["status"] == "completed"
        assert telemetry["status"]["agent_busy"] is False
        feedback = {"conversation_id": run["conversation_id"], "message_id": result["message_id"], "label": "good"}
        assert client.post("/feedback", json=feedback).status_code == 400
        assert client.post(f"/agent/runs/{run['id']}/remember", json={}).status_code == 400
        client.patch(f"/conversations/{run['conversation_id']}", json={"logging_enabled": True})
        assert client.post("/feedback", json=feedback).status_code == 201
        assert client.post(f"/agent/runs/{run['id']}/remember", json={}).status_code == 200
        exported = client.get(f"/agent/runs/{run['id']}/export").json()
        assert "reasoning" not in exported and "tool_results" not in exported
        assert "Complete a private task" not in json.dumps(app.state.store.list("agent_runs"))
        client.delete(f"/conversations/{run['conversation_id']}")
        erased = client.get(f"/agent/runs/{run['id']}/export").json()
        assert erased["content_retained"] is False and "answer" not in erased and "goal" not in erased
        assert run["id"] not in app.state.agent.states


def test_active_agent_blocks_model_learning_and_chat_and_can_cancel(tmp_path, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml", agent_workspace=tmp_path), autoload="demo")
    with TestClient(app) as client:
        wire_provider(app, monkeypatch, block=True)
        run = client.post("/agent/runs", json={"goal": "Wait"}).json()
        assert client.post("/models/unload", json={}).status_code == 409
        assert client.post("/train/start", json={"dry_run": True}).status_code == 409
        assert client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]}).status_code == 409
        assert client.post(f"/agent/runs/{run['id']}/approve", json={"approval_id": "missing", "approved": True}).status_code == 400
        assert client.post(f"/agent/runs/{run['id']}/cancel", json={}).json()["status"] == "cancelled"
        assert client.get("/system/status").json()["agent_busy"] is False
        assert app.state.store.list("messages") == []


def test_agent_routes_require_configured_bearer_token(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml", api_token="test-key"), autoload="demo")
    with TestClient(app) as client:
        assert client.get("/agent/tools").status_code == 401
        assert client.post("/agent/runs", json={"goal": "x"}).status_code == 401
        assert client.get("/agent/tools", headers={"Authorization": "Bearer test-key"}).status_code == 200


def test_input_and_resume_retry_keys_survive_lost_http_responses(tmp_path, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path / 'state', models_file=tmp_path / 'models.yaml',
                              agent_workspace=tmp_path), autoload='demo')
    with TestClient(app) as client:
        calls = wire_provider(app, monkeypatch)
        run = client.post('/agent/runs', json={'goal': 'Initial goal.'}).json()
        client.get(f"/agent/runs/{run['id']}/events")
        for kind, body in (
            ('input', {'content': 'Follow-up goal.', 'idempotency_key': 'input-response-lost'}),
            ('resume', {'instruction': 'Another follow-up.', 'idempotency_key': 'resume-response-lost'}),
        ):
            path = f"/agent/runs/{run['id']}/{kind}"
            assert client.post(path, json=body).is_success  # Response deliberately discarded.
            client.get(f"/agent/runs/{run['id']}/events")
            before = client.get(f"/agent/runs/{run['id']}").json()
            replay = client.post(path, json=body)
            assert replay.is_success, replay.text
            assert replay.json()['idempotent_replay'] and replay.json()['segment'] == before['segment']
            changed = {**body, ('content' if kind == 'input' else 'instruction'): 'Conflicting instruction.'}
            assert client.post(path, json=changed).status_code == 400
        assert len(calls) == 3
        transcript = client.get(f"/conversations/{run['conversation_id']}").json()['messages']
        assert len(transcript) == 6
        assert client.post(f"/agent/runs/{run['id']}/input", json={'content': 'x', 'idempotency_key': ''}).status_code == 422
