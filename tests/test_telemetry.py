import pytest
from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings
from lucifer import telemetry


def inventory():
    return {"available": True, "gpus": [{"id": "GPU-one", "index": "0", "name": "Test GPU",
            "backend": "cuda", "total_mb": 12000, "used_mb": 10, "free_mb": 11990}]}


@pytest.mark.asyncio
async def test_live_counters_change_without_rediscovering_hardware(monkeypatch):
    discoveries, queries = [], []
    def discover():
        discoveries.append(True)
        return inventory()
    def sample(args):
        queries.append(args)
        use = 12 if len(queries) == 1 else 87
        return f"GPU-one, 0, 12000, {use * 100}, {12000 - use * 100}, {use}, 45.2, 51\n"
    monkeypatch.setattr(telemetry, "discover_gpus", discover)
    monkeypatch.setattr(telemetry, "_run", sample)
    service = telemetry.TelemetryService(interval=100)
    await service.start()
    try:
        first = service.snapshot()
        await service.sample()
        second = service.snapshot()
        for _ in range(5):
            service.snapshot()
        assert len(discoveries) == 1 and len(queries) == 2
        assert first["gpus"][0]["utilization_pct"] == 12
        assert second["gpus"][0]["utilization_pct"] == 87
        assert second["gpus"][0]["used_mb"] == 8700
        assert second["gpus"][0]["power_w"] == 45.2
        assert not second["stale"]
        second["gpus"][0]["used_mb"] = 0
        assert service.snapshot()["gpus"][0]["used_mb"] == 8700
    finally:
        await service.close()


def test_failed_and_unsupported_counters_are_honest(monkeypatch):
    monkeypatch.setattr(telemetry, "_run", lambda args: "")
    result = telemetry.sample_gpus(inventory())
    assert result["stale"] and result["errors"]
    assert result["gpus"][0]["utilization_pct"] is None
    other = {"gpus": [{"id": "intel", "backend": "vulkan"}]}
    result = telemetry.sample_gpus(other)
    assert result["gpus"][0]["power_w"] is None
    assert result["gpus"][0]["telemetry_available"] is False


def test_heartbeat_reuses_registry_and_invalidates_on_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "discover_gpus", lambda: {"available": False, "gpus": []})
    app = create_app(Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml"), autoload="demo")
    with TestClient(app) as client:
        bootstrap = client.get("/system/bootstrap").json()
        calls = []
        original = app.state.store.list
        def listed(kind, **filters):
            calls.append(kind)
            return original(kind, **filters)
        monkeypatch.setattr(app.state.store, "list", listed)
        for _ in range(4):
            result = client.get("/system/telemetry").json()
            assert result["revision"] == bootstrap["revision"]
        assert calls == []
        # Chat-only documents don't invalidate model/adapter/job panel state.
        client.post("/conversations", json={})
        assert client.get("/system/telemetry").json()["revision"] == bootstrap["revision"]
        app.state.store.put("training_runs", {"id": "run", "status": "dry_run"})
        refreshed = client.get("/system/telemetry").json()
        assert refreshed["revision"] == bootstrap["revision"]
        assert refreshed["jobs"]["train"][0]["id"] == "run"
        assert calls.count("training_runs") == 1
        assert "models" not in calls and "adapters" not in calls
        app.state.store.set_setting("active_adapter", None)
        assert client.get("/system/telemetry").json()["revision"] > bootstrap["revision"]
        capability = client.get("/models/demo/capabilities")
        assert capability.status_code == 200
        assert capability.json()["model_id"] == "demo"


def test_optimizer_heartbeat_updates_preserve_sealed_state_without_registry_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "discover_gpus", lambda: {"available": False, "gpus": []})
    app = create_app(Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml"), autoload="demo")
    with TestClient(app) as client:
        app.state.store.put("sleep_runs", {"id": "sleep_live", "status": "training",
            "adaptive": {"state": {"global_successful_updates": 36, "round_remaining": 14, "adaptive_epoch": 1}},
            "progress": {"cumulative_optimizer_steps": 36, "round_remaining": 14, "adaptive_epoch": 1}})
        bootstrap = client.get("/system/bootstrap").json()
        calls = []
        original = app.state.store.list
        def listed(kind, **filters):
            calls.append(kind)
            return original(kind, **filters)
        monkeypatch.setattr(app.state.store, "list", listed)
        for step in (37, 38):
            app.state.store.update("sleep_runs", "sleep_live", progress={
                "cumulative_optimizer_steps": step, "round_remaining": 50 - step, "adaptive_epoch": 1})
            heartbeat = client.get("/system/telemetry").json()
            assert heartbeat["revision"] == bootstrap["revision"]
            job = heartbeat["jobs"]["sleep"][0]
            assert job["progress"]["cumulative_optimizer_steps"] == step
            assert job["progress"]["round_remaining"] == 50 - step
            assert job["adaptive"]["state"] == {"global_successful_updates": 36, "round_remaining": 14, "adaptive_epoch": 1}
        assert calls == ["sleep_runs", "sleep_runs"]
