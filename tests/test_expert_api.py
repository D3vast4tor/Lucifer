"""Public expert creation keeps reviewed sources, alias fields and dry-run proof honest."""
import asyncio
import time

import httpx
import pytest

from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings


def test_expert_api_proposes_new_reviewed_data_and_maps_build_aliases(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml"), autoload="demo")
    with TestClient(app) as client:
        profile = {"id": "trainable", "backend": "vllm", "path": "org/checkpoint"}
        assert client.post("/models/register", json=profile).status_code == 200
        conversation = client.post("/conversations", json={"logging_enabled": True, "training_enabled": True}).json()
        reply = client.post("/chat", json={"conversation_id": conversation["id"],
            "messages": [{"role": "user", "content": "What is the capital of France?"}]}).json()
        proposal = {"conversation_ids": [conversation["id"]], "base_model": "trainable"}
        assert client.post("/experts/propose", json=proposal).json() == []
        assert client.post("/feedback", json={"conversation_id": conversation["id"],
            "message_id": reply["message_id"], "label": "good"}).status_code == 201
        suggested = client.post("/experts/propose", json=proposal).json()
        assert suggested[0]["example_count"] == 1
        response = client.post("/experts/build", json={"name": "Lucifer geography", "topics": ["geography"],
            "description": "Reviewed geography answers", "conversation_ids": [conversation["id"]],
            "base_model": "trainable", "scope": "all", "max_episodes": 1, "dry_run": True})
        assert response.status_code == 202, response.text
        run = response.json()
        for _ in range(100):
            run = client.get(f"/sleep/runs/{run['id']}").json()
            if run["status"] in {"dry_run", "failed"}:
                break
            time.sleep(.01)
        assert run["status"] == "dry_run", run
        assert run["expert"]["expert_name"] == "Lucifer geography"
        assert run["expert"]["expert_topics"] == ["geography"]
        assert run["dataset_metrics"]["examples"] == 1
        assert run["request"]["scope"] == "all"
        assert "name" not in run["request"] and "conversation_ids" not in run["request"]
        assert client.get("/experts").json() == []
        assert client.get("/adapters").json() == []


@pytest.mark.asyncio
async def test_terminal_learning_job_stays_busy_until_model_restoration_finishes(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml"), autoload="demo")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            app.state.store.put("training_runs", {"id": "train_restore", "status": "rejected"})
            release = asyncio.Event()
            task = asyncio.create_task(release.wait())
            app.state.learning.tasks["train_restore"] = task
            before = (await client.get("/system/telemetry")).json()
            assert before["status"]["jobs"] == [] and before["status"]["learning_busy"] is True
            assert (await client.get("/system/bootstrap")).json()["status"]["learning_busy"] is True
            assert (await client.post("/models/unload", json={})).status_code == 409
            assert (await client.post("/agent/runs", json={"goal": "Wait for restore"})).status_code == 409
            release.set()
            await task
            after = (await client.get("/system/telemetry")).json()
            assert after["revision"] == before["revision"]
            assert after["status"]["learning_busy"] is False
            assert (await client.get("/system/status")).json()["learning_busy"] is False
            assert (await client.post("/models/unload", json={})).status_code == 200
