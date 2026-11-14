"""Adaptive Sleep controls, admission visibility and cached progress rendering."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
from pydantic import ValidationError
import pytest
from textual.widgets import Button, Static

from lucifer.adapter_map import admission_gate_lines, build_atlas, node_details
from lucifer.api import create_app
from lucifer.client import APIClient
from lucifer.config import Settings
from lucifer.schemas import AdaptiveSchedule, SleepRequest, SleepResumeRequest
from lucifer.tui import LuciferApp, sleep_start_options


def test_schedule_requires_the_fixed_refresh_contract_and_legacy_is_unchanged():
    one_shot = SleepRequest()
    assert one_shot.schedule is None and one_shot.generation_thinking == "off"
    assert SleepRequest(schedule={}).generation_thinking == "max"
    assert SleepRequest(schedule={}, generation_thinking="off").generation_thinking == "max"
    with pytest.raises(ValidationError, match="maximum teacher thinking"):
        SleepRequest(schedule={}, generation_thinking="high")
    assert AdaptiveSchedule(seed=7).model_dump() == {
        "mode": "until_admitted", "refresh_optimizer_steps": 50, "initial_update_budget": 50, "seed": 7}
    for invalid in ({"refresh_optimizer_steps": 49}, {"initial_update_budget": 100},
                    {"mode": "forever"}, {"seed": True}, {"unexpected": 1}):
        with pytest.raises(ValidationError):
            AdaptiveSchedule(**invalid)
    for key in ("", "  ", "x" * 129):
        with pytest.raises(ValidationError):
            SleepResumeRequest(idempotency_key=key)


def test_terminal_adaptive_options_are_explicit_and_default_teacher_to_max():
    assert sleep_start_options([]) == {"dry_run": False, "context_examples": True}
    parsed = sleep_start_options(["--adaptive", "--seed", "19", "--generation-model", "lucifer"])
    assert parsed["schedule"]["seed"] == 19 and parsed["schedule"]["refresh_optimizer_steps"] == 50
    assert parsed["generation_thinking"] == "max" and parsed["generation_model"] == "lucifer"
    for invalid in (["--seed", "3"], ["--adaptive", "--legacy"], ["--thinking", "guess"],
                    ["--seed", "bad"], ["--adaptive", "--adaptive"]):
        with pytest.raises(ValueError):
            sleep_start_options(invalid)


def interface_app(tmp_path):
    return create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml"), autoload="demo")


def test_api_passes_schedule_and_resume_key_without_rejecting_receipt_replay(tmp_path, monkeypatch):
    app = interface_app(tmp_path)
    with TestClient(app) as client:
        submitted, resumed = [], []

        async def start(payload):
            submitted.append(payload)
            return {"id": "sleep_adaptive", "status": "queued"}

        async def resume(run_id, *, idempotency_key):
            resumed.append((run_id, idempotency_key))
            return {"id": run_id, "status": "training"}

        monkeypatch.setattr(app.state.learning, "start_sleep", start)
        response = client.post("/sleep/start", json={"schedule": {"mode": "until_admitted", "seed": 7},
                                                     "generation_thinking": "max"})
        assert response.status_code == 202
        assert submitted[0]["schedule"]["initial_update_budget"] == 50
        assert submitted[0]["generation_thinking"] == "max"
        app.state.learning = SimpleNamespace(busy=True, resume_sleep=resume)
        for _ in range(2):
            response = client.post("/sleep/runs/sleep_adaptive/resume", json={"idempotency_key": "same-intent"})
            assert response.status_code == 202
        assert resumed == [("sleep_adaptive", "same-intent")] * 2
        assert client.post("/sleep/runs/sleep_adaptive/resume", json={}).status_code == 422


@pytest.mark.parametrize("state", ["admitted", "archived"])
def test_select_uses_admission_service_and_never_switches_models(tmp_path, monkeypatch, state):
    app = interface_app(tmp_path)
    with TestClient(app) as client:
        app.state.store.put("adapters", {"id": "eligible", "base_model_id": "demo", "status": state,
                                         "admitted_at": "2026-11-12T23:00:00Z"})
        selected = []

        async def activate(ident):
            selected.append(ident)
            return {"active_adapter": ident}

        async def forbid(*args, **kwargs):
            raise AssertionError("Selecting an admitted adapter must not load a different model")

        monkeypatch.setattr(app.state.learning, "activate_admitted", activate, raising=False)
        monkeypatch.setattr(app.state.manager, "load", forbid)
        assert client.post("/adapters/select", json={"adapter_id": "eligible"}).json() == {"active_adapter": "eligible"}
        assert selected == ["eligible"]


@pytest.mark.asyncio
async def test_client_retains_explicit_resume_key_and_old_sleep_payload():
    requests = []

    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json={"id": "sleep_test"})

    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        await client.start_sleep(dry_run=True)
        await client.resume_sleep("sleep_test", idempotency_key="repeatable")
        await client.resume_sleep("sleep_test", idempotency_key="repeatable")
    finally:
        await client.close()
    assert requests[0] == ("/sleep/start", {"dynamic": True, "dry_run": True, "context_examples": True})
    assert requests[1:] == [("/sleep/runs/sleep_test/resume", {"idempotency_key": "repeatable"})] * 2


@pytest.mark.asyncio
async def test_admitted_atlas_and_sleep_progress_use_the_heartbeat_snapshot(monkeypatch):
    model = {"id": "lucifer", "status": "loaded", "backend": "external"}
    report = {"gates": {"complete_evidence": True, "domain_improvement": {"passed": False, "reason": "No verified gain"}}}
    adapter = {"id": "eligible", "base_model_id": "lucifer", "status": "admitted", "admission_report": report}
    job = {"id": "sleep_test", "status": "awaiting_evidence", "request": {"schedule": {"mode": "until_admitted"}},
           "admission_report": report, "progress": {"terminal": True, "restored": True, "adaptive_epoch": 3,
               "round_budget": 64, "cumulative_optimizer_steps": 150, "data_passes": 2.5, "next_refresh_optimizer_step": 200,
               "randint_min": 3, "randint_max": 9, "round_index": 5}}
    snapshot = {"models": [model], "adapters": [adapter], "status": {"model": model, "active_adapter": None,
        "learning_busy": False, "agent_busy": False}, "jobs": {"train": [], "sleep": [job]}, "revision": 1,
        "gpu": {"available": False, "gpus": [], "stale": False, "errors": []}, "capabilities": {}}
    requests = []

    def handle(request):
        requests.append(request.url.path)
        assert request.url.path in {"/system/bootstrap", "/system/telemetry"}
        return httpx.Response(200, json=snapshot)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 50)) as pilot:
        app.set_view("sleep")
        await pilot.pause()
        counters = app.query_one("#sleep-counters", Static).render().plain
        assert "ADAPTIVE EPOCH // 3" in counters and "DATA PASSES // 2.5" in counters
        assert "ROUND INDEX // 5" in counters and "3 + RANDINT(3, 9)" in counters
        assert "TOTAL OPTIMIZER UPDATES // 150" in counters and "NEXT DATA REFRESH AT UPDATE // 200" in counters
        assert "domain improvement: FAIL — No verified gain" in app.query_one("#sleep-plan", Static).render().plain
        assert not app.query_one("#sleep-start", Button).disabled
        assert ("eligible", "eligible") in app._adapter_options
        rebuilds = []
        original_registry = app.render_registry

        def render_registry():
            rebuilds.append(True)
            original_registry()

        monkeypatch.setattr(app, "render_registry", render_registry)
        bootstrap_reads = requests.count("/system/bootstrap")
        job["progress"].update(randint_min=4, randint_max=10, data_passes=2.75)
        await app.refresh_panels()
        counters = app.query_one("#sleep-counters", Static).render().plain
        assert "3 + RANDINT(4, 10)" in counters and "DATA PASSES // 2.75" in counters
        assert not rebuilds and requests.count("/system/bootstrap") == bootstrap_reads
        node = app.query_one("#brain-canvas").atlas.node("adapter:eligible")
        assert app.can_activate_adapter(node)
        app._current_model = {"id": "other"}
        assert app.can_activate_adapter(node)  # explicit activation loads the matching registered profile
    atlas = build_atlas([model], [adapter])
    assert "not automatically active" in node_details(atlas.node("adapter:eligible"), atlas)
    assert admission_gate_lines({"gates": {"unknown": None}}) == ["unknown: UNKNOWN"]
