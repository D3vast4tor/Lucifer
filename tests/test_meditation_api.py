"""Native Meditation API uses real source checks without launching a model job."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings


def application(tmp_path, *, token=None):
    return create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml",
        eval_file=tmp_path / "missing-evals.yaml", api_token=token), autoload="demo")


def register_native(client, tmp_path):
    result = client.post("/models/register", json={"id": "native", "backend": "strata",
        "path": str(tmp_path / "never-loaded-pack"), "context_length": 1024,
        "runtime_command": ["/bin/false"], "enable_lora": False})
    assert result.status_code == 200, result.text


def add_sources(client, store, count=4, *, training=True):
    ids = []
    for index in range(count):
        cid = client.post("/conversations", json={"logging_enabled": True, "training_enabled": training}).json()["id"]
        store.put("messages", {"id": f"question_{index}", "conversation_id": cid, "role": "user",
            "content": f"Arithmetic: compute {index + 30} * 17."})
        store.put("messages", {"id": f"answer_{index}", "conversation_id": cid, "role": "assistant",
            "content": str((index + 30) * 17), "reasoning_content": "NEVER_TRAIN_PRIVATE_PROVIDER_REASONING"})
        assert client.post("/feedback", json={"conversation_id": cid, "message_id": f"answer_{index}",
                                              "label": "good"}).status_code == 201
        ids.append(cid)
    return ids


def finished(client, ident):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get("/meditation/runs/" + ident)
        assert response.status_code == 200, response.text
        run = response.json()
        if run.get("progress", {}).get("terminal"):
            return run
        time.sleep(.005)
    raise AssertionError("Meditation source-only job failed to settle")


def test_dry_run_prepares_sources_idempotently_without_native_worker_or_activation(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        register_native(client, tmp_path)
        ids = add_sources(client, app.state.store)
        async def forbidden(*args, **kwargs):
            raise AssertionError("Dry-run preparation may not execute a native worker")
        monkeypatch.setattr(app.state.meditation, "_worker", forbidden)
        payload = {"base_model": "native", "conversation_ids": ids, "dry_run": True, "idempotency_key": "source-preview"}
        response = client.post("/meditation/start", json=payload)
        assert response.status_code == 202, response.text
        run = finished(client, response.json()["id"])
        assert run["status"] == "dry_run", run
        assert run["progress"]["restored"]
        assert run["dataset_metrics"]["partition_coverage"]
        raw = Path(run["source_cache_path"]).read_text()
        assert "NEVER_TRAIN_PRIVATE_PROVIDER_REASONING" not in raw
        bundle = json.loads(raw)
        assert bundle["manifest"]["raw_thinking_retained"] is False
        assert bundle["manifest"]["raw_tool_traces_retained"] is False
        assert all(bundle[key] for key in ("examples", "development_examples", "holdout_examples"))
        again = client.post("/meditation/start", json=payload)
        assert again.status_code == 202 and again.json()["id"] == run["id"]
        assert len(client.get("/meditation/runs").json()) == 1
        assert client.get("/meditation/bases").json() == []
        assert client.get("/models/current").json()["id"] == "demo"
        conflict = client.post("/meditation/start", json={**payload, "seed": 20})
        assert conflict.status_code == 400 and "Idempotency" in conflict.json()["detail"]


def test_plain_chat_owns_model_until_its_turn_finishes(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        register_native(client, tmp_path)
        cid = client.post('/conversations', json={}).json()['id']
        lock = app.state.chat.locks[cid]
        client.portal.call(lock.acquire)
        try:
            status = client.get('/system/status').json()
            assert status['chat_busy'] is True and status['agent_busy'] is False
            response = client.post('/meditation/start', json={
                'base_model': 'native', 'dry_run': True, 'idempotency_key': 'wait-for-chat'})
            assert response.status_code == 409 and 'chat turn' in response.json()['detail']
            response = client.post('/meditation/runs/unused/resume', json={
                'idempotency_key': 'resume-waits-for-chat'})
            assert response.status_code == 409 and 'chat turn' in response.json()['detail']
            assert not any(operation.get('phase') == 'resume' for operation in app.state.store.list('meditation_operations'))
            response = client.post('/sleep/runs/unused/resume', json={'idempotency_key': 'sleep-waits-for-chat'})
            assert response.status_code == 409 and 'chat turn' in response.json()['detail']
            assert client.post('/models/unload', json={}).status_code == 409
            assert client.get('/models/current').json()['id'] == 'demo'
            assert client.get('/meditation/runs').json() == []
        finally:
            client.portal.call(lock.release)
        assert client.get('/system/status').json()['chat_busy'] is False


def test_unconsented_or_missing_sources_cannot_launch_native_work(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        register_native(client, tmp_path)
        cid = add_sources(client, app.state.store, 1, training=False)[0]
        async def forbidden(*args, **kwargs):
            raise AssertionError("Invalid sources cannot execute native training")
        monkeypatch.setattr(app.state.meditation, "_worker", forbidden)
        for index, selected in enumerate(([cid], ["does-not-exist"])):
            response = client.post("/meditation/start", json={"base_model": "native", "conversation_ids": selected,
                "idempotency_key": f"invalid-source-{index}"})
            assert response.status_code == 202
            run = finished(client, response.json()["id"])
            assert run["status"] == "failed" and run["progress"]["restored"]
        assert client.get("/meditation/bases").json() == []


def test_unsupervised_api_prepares_unreviewed_public_text_without_answer_labels(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        register_native(client, tmp_path)
        for index in range(3):
            cid = client.post("/conversations", json={"training_enabled": True}).json()["id"]
            app.state.store.put("messages", {"id": f"unreviewed_{index}", "conversation_id": cid,
                "role": "user", "content": f"Public text for language modelling, source {index}."})
        result = client.post("/meditation/start", json={"mode": "unsupervised", "base_model": "native",
            "dry_run": True, "idempotency_key": "unreviewed-language-data"})
        assert result.status_code == 202
        run = finished(client, result.json()["id"])
        assert run["status"] == "dry_run" and run["dataset_metrics"]["manual_review_required"] is False
        bundle = json.loads(Path(run["source_cache_path"]).read_text())
        assert bundle["manifest"]["training_mode"] == "unsupervised"
        assert all("text" in row and "messages" not in row for key in (
            "examples", "development_examples", "holdout_examples") for row in bundle[key])


def test_meditation_routes_require_api_token_and_reject_invalid_intents(tmp_path):
    app = application(tmp_path, token="local-test-token")
    with TestClient(app) as client:
        for path in ("/meditation/runs", "/meditation/bases"):
            assert client.get(path).status_code == 401
        assert client.post("/meditation/start", json={"idempotency_key": "key"}).status_code == 401
        client.headers["Authorization"] = "Bearer local-test-token"
        assert client.get("/meditation/runs").status_code == 200
        for payload in ({}, {"idempotency_key": " "}, {"idempotency_key": "key", "seed": True},
                        {"idempotency_key": "key", "conversation_ids": []},
                        {"idempotency_key": "key", "conversation_ids": ["same", "same"]},
                        {"idempotency_key": "key", "mode": "fake_training"},
                        {"idempotency_key": "key", "max_rounds": 3}):
            assert client.post("/meditation/start", json=payload).status_code == 422
        assert client.post("/meditation/runs/absent/resume", json={}).status_code == 422
        assert client.post("/meditation/bases/absent/activate", json={}).status_code == 422
        assert client.post("/meditation/rollback", json={}).status_code == 422


def test_meditation_busy_excludes_chat_model_changes_and_activation(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "meditation", SimpleNamespace(busy=True))
        assert client.get("/system/status").json()["learning_busy"] is True
        assert client.get("/system/status").json()["meditation_busy"] is True
        assert client.post("/chat", json={"conversation_id": "unused", "messages": [{"role": "user", "content": "hello"}]}).status_code == 409
        assert client.post("/models/load", json={"model_id": "demo"}).status_code == 409
        assert client.post("/meditation/bases/unused/activate", json={"idempotency_key": "busy"}).status_code == 409
        assert client.post("/meditation/rollback", json={"idempotency_key": "busy"}).status_code == 409
        assert client.post('/sleep/runs/unused/resume', json={'idempotency_key': 'sleep-waits-for-meditation'}).status_code == 409


def test_active_agent_blocks_meditation_start_and_resume(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "agent", SimpleNamespace(busy=True))
        assert client.post("/meditation/start", json={"idempotency_key": "busy-agent"}).status_code == 409
        assert client.post("/meditation/runs/unused/resume", json={"idempotency_key": "busy-agent"}).status_code == 409


def test_bootstrap_uses_cached_meditation_registry_and_progress_heartbeat(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        original = app.state.store.list
        calls = []
        def counted(kind, **kwargs):
            calls.append(kind)
            return original(kind, **kwargs)
        monkeypatch.setattr(app.state.store, "list", counted)
        first = client.get("/system/bootstrap").json()
        assert first["jobs"]["meditation"] == [] and first["base_candidates"] == []
        calls.clear()
        for _ in range(3):
            assert client.get("/system/telemetry").status_code == 200
        assert not calls
        app.state.store.put("meditation_runs", {"id": "source_preview", "status": "dry_run",
            "progress": {"stage": "dry_run", "terminal": True, "restored": True}})
        job_update = client.get("/system/telemetry").json()
        assert job_update["jobs"]["meditation"][0]["id"] == "source_preview"
        assert calls == ["meditation_runs"]
        assert job_update["revision"] == first["revision"]
        calls.clear()
        app.state.store.put("base_candidates", {"id": "saved_native", "status": "candidate"})
        candidate_update = client.get("/system/bootstrap").json()
        assert candidate_update["base_candidates"][0]["id"] == "saved_native"
        assert calls == ["base_candidates"]
        assert candidate_update["revision"] > job_update["revision"]


def test_base_activation_rejects_unproven_candidate_and_preserves_active_parent(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        register_native(client, tmp_path)
        ids = add_sources(client, app.state.store)
        response = client.post("/meditation/start", json={"base_model": "native", "conversation_ids": ids,
            "dry_run": True, "idempotency_key": "source-preview"})
        run = finished(client, response.json()["id"])
        source_path = Path(run["source_cache_path"])
        app.state.store.put("base_candidates", {"id": "unproven", "status": "candidate",
            "source_cache_path": str(source_path), "source_bundle_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_conversation_ids": ids, "admission_report": {"passed": False}})
        response = client.post("/meditation/bases/unproven/activate", json={"idempotency_key": "attempt-activation"})
        assert response.status_code == 400, response.text
        assert client.get("/models/current").json()["id"] == "demo"
        assert app.state.store.setting("active_base_revision") is None
