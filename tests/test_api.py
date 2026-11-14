"""Public API integration: privacy, persistence, SSE failure, jobs and authentication."""
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.client import APIClient, APIError
from lucifer.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml", eval_file=tmp_path / "evals.yaml")


def chat(client, conversation_id, question="What is the capital of France?"):
    response = client.post("/chat", json={"conversation_id": conversation_id, "messages": [{"role": "user", "content": question}]})
    assert response.status_code == 200, response.text
    return response.json()


def finish_job(client, path):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"dry_run", "failed", "cancelled", "awaiting_promotion", "rejected"}:
            return job
        time.sleep(0.01)
    pytest.fail(f"Job did not finish: {job}")


def test_private_chat_save_feedback_target_and_dry_run(settings):
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        conv = client.post("/conversations", json={"logging_enabled": False}).json()
        assert conv["logging_enabled"] is False
        assert conv["training_enabled"] is False
        answer = chat(client, conv["id"], "Email me the capital of France at alice@example.org")
        assert answer["choices"][0]["message"]["content"] == "Paris."
        assert app.state.store.list("messages") == []
        transient = client.get(f"/conversations/{conv['id']}").json()["messages"]
        assert len(transient) == 2
        assert client.post("/feedback", json={"conversation_id": conv["id"], "message_id": answer["message_id"], "label": "good"}).status_code == 400
        assert client.patch(f"/conversations/{conv['id']}", json={"training_enabled": True}).status_code == 400

        saved = client.patch(f"/conversations/{conv['id']}", json={"logging_enabled": True}).json()
        assert saved["training_enabled"] is False
        assert len(app.state.store.list("messages", conversation_id=conv["id"])) == 2
        user_id = transient[0]["id"]
        assert client.post("/feedback", json={"conversation_id": conv["id"], "message_id": user_id, "label": "good"}).status_code == 400
        other = client.post("/conversations", json={"logging_enabled": True}).json()
        assert client.post("/feedback", json={"conversation_id": other["id"], "message_id": answer["message_id"], "label": "good"}).status_code == 400
        assert client.post("/feedback", json={"conversation_id": conv["id"], "message_id": answer["message_id"], "label": "correct", "correction": "Paris is the capital of France."}).status_code == 201
        assert client.patch(f"/conversations/{conv['id']}", json={"training_enabled": True}).status_code == 200

        # Keep the legacy redaction contract covered independently of the
        # context compiler's stricter source-withholding policy.
        response = client.post("/sleep/start", json={"dry_run": True, "context_examples": False})
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        run = finish_job(client, f"/sleep/runs/{run_id}")
        assert run["status"] == "dry_run", run
        assert run["dataset_metrics"]["corrected"] == 1
        dataset = [json.loads(line) for line in Path(run["dataset_path"]).read_text().splitlines()]
        assert dataset[0]["messages"][1]["content"] == "Paris is the capital of France."
        assert "alice@example.org" not in json.dumps(dataset)
        assert "[EMAIL]" in json.dumps(dataset)
        assert Path(run["dataset_path"]).stat().st_mode & 0o777 == 0o600
        assert client.get("/adapters").json() == []
        assert client.get("/eval/results").json() == []
        assert client.get("/system/status").json()["active_adapter"] is None
        assert client.get("/sleep/queue").json() == []
        assert client.get("/sleep/status").json()["status"] == "dry_run"


def test_saved_history_survives_restart_private_history_does_not(settings):
    with TestClient(create_app(settings, autoload="demo")) as client:
        saved = client.post("/conversations", json={"logging_enabled": True}).json()
        private = client.post("/conversations", json={"logging_enabled": False}).json()
        chat(client, saved["id"])
        chat(client, private["id"])
        assert len(client.get(f"/conversations/{private['id']}").json()["messages"]) == 2
    with TestClient(create_app(settings, autoload="demo")) as client:
        assert len(client.get(f"/conversations/{saved['id']}").json()["messages"]) == 2
        assert client.get(f"/conversations/{private['id']}").json()["messages"] == []
        exported = client.get(f"/conversations/{saved['id']}/export").json()
        assert exported["messages"][1]["content"] == "Paris."
        assert client.delete(f"/conversations/{saved['id']}").status_code == 200
        assert client.get(f"/conversations/{saved['id']}").status_code == 404


def test_turn_stream_error_never_persists_partial_answer(settings):
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        conv = client.post("/conversations", json={"logging_enabled": True}).json()
        original = app.state.manager.stream_events
        async def broken(messages, **kwargs):
            yield "delta", {"content": "partial output"}
            raise RuntimeError("Inference process died")
        app.state.manager.stream_events = broken
        response = client.post("/chat/stream", json={"conversation_id": conv["id"], "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert "event: delta" in response.text
        assert "event: error" in response.text
        assert "Inference process died" in response.text
        assert "event: done" not in response.text
        assert client.get(f"/conversations/{conv['id']}").json()["messages"] == []
        assert app.state.store.list("messages") == []
        app.state.manager.stream_events = original
        chat(client, conv["id"])
        assert len(client.get(f"/conversations/{conv['id']}").json()["messages"]) == 2


def test_logging_off_erases_persisted_sources_and_training_consent(settings):
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        conv = client.post("/conversations", json={"logging_enabled": True, "training_enabled": True}).json()
        answer = chat(client, conv["id"])
        client.post("/feedback", json={"conversation_id": conv["id"], "message_id": answer["message_id"], "label": "good"})
        app.state.store.put("memory_items", {"conversation_id": conv["id"], "content": "Personal note"})
        response = client.patch(f"/conversations/{conv['id']}", json={"logging_enabled": False})
        assert response.status_code == 200
        assert response.json()["training_enabled"] is False
        for kind in ("messages", "feedback", "memory_items"):
            assert app.state.store.list(kind, conversation_id=conv["id"]) == []
        assert len(client.get(f"/conversations/{conv['id']}").json()["messages"]) == 2


@pytest.mark.asyncio
async def test_real_terminal_client_auth_and_stream_protocol(settings):
    settings.api_token = "test-local-secret"
    app = create_app(settings, autoload="demo")
    async with app.router.lifespan_context(app):
        unauthorized = APIClient(transport=httpx.ASGITransport(app=app))
        with pytest.raises(APIError, match="401"):
            await unauthorized.get("/health")
        await unauthorized.close()
        client = APIClient(api_token=settings.api_token, transport=httpx.ASGITransport(app=app))
        assert (await client.get("/health"))["status"] == "ok"
        conv = await client.post("/conversations")
        events = [event async for event in client.stream_chat({"conversation_id": conv["id"], "messages": [{"role": "user", "content": "What is 2 + 2?"}]})]
        assert events[0] == ("meta", {"conversation_id": conv["id"]})
        assert "".join(data["content"] for event, data in events if event == "delta") == "4"
        assert events[-1][0] == "done"
        assert events[-1][1]["message_id"]
        await client.close()


def test_api_rejects_invalid_training_and_incomplete_feedback(settings):
    with TestClient(create_app(settings, autoload="demo")) as client:
        assert client.post("/conversations", json={"logging_enabled": False, "training_enabled": True}).status_code == 422
        assert client.post("/train/start", json={"output_name": "../escape"}).status_code == 422
        assert client.post("/train/start", json={"epochs": 1000}).status_code == 422
        assert client.post("/sleep/start", json={"max_episodes": 0}).status_code == 422
        assert client.post("/feedback", json={"conversation_id": "x", "message_id": "y", "label": "correct"}).status_code == 422
        assert client.post("/chat", json={"messages": [{"role": "assistant", "content": "pretend completed"}]}).status_code == 422
        assert client.get("/train").json() == []
        assert client.get("/sleep/runs").json() == []


@pytest.mark.asyncio
async def test_terminal_app_complete_reviewed_sleep_flow(settings):
    from lucifer.tui import LuciferApp
    backend = create_app(settings, autoload="demo")
    async with backend.router.lifespan_context(backend):
        terminal = LuciferApp(client=APIClient(transport=httpx.ASGITransport(app=backend)))
        async with terminal.run_test(size=(120, 40)) as pilot:
            async def submit(text):
                terminal.submit(text)
                await pilot.pause()
                await terminal.workers.wait_for_complete()
                await pilot.pause()
            await submit("What is the capital of France?")
            assert terminal.last_message_id
            await submit("/chat save")
            await submit("/feedback correct Paris is the capital of France.")
            await submit("/privacy training on")
            await submit("/sleep start --dry-run")
            run = backend.state.store.get("sleep_runs", terminal.latest_sleep)
            assert run["status"] == "dry_run", run
            assert run["dataset_metrics"]["corrected"] == 1
            assert terminal.conversation["training_enabled"] is True
            assert backend.state.store.list("adapters") == []
