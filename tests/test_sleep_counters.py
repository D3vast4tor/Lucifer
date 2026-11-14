"""Sleep counters follow cached heartbeats without changing sealed training state."""
import copy

import httpx
import pytest
from textual.widgets import Static

from lucifer.client import APIClient
from lucifer.tui import LuciferApp


def adaptive_job():
    return {"id": "sleep_resumed", "status": "training", "started_at": "2026-11-12T12:00:00Z",
        "request": {"schedule": {"mode": "until_admitted"}, "base_model": "student"},
        "candidate_adapter_id": "checkpoint-36", "evaluation_id": "evaluation-30",
        "admission_report": {"gates": {"complete_evidence": True, "domain_improvement": False}},
        "adaptive": {"state": {"adaptive_epoch": 1, "round_index": 1, "round_budget": 50,
            "round_remaining": 14, "global_successful_updates": 36, "dataset_passes_completed": 6,
            "next_refresh_at": 50, "randint_min": 1, "randint_max": 7}},
        "progress": {"stage": "training", "label": "Training current segment", "terminal": False, "restored": False,
            "adaptive_epoch": 1, "round_index": 1, "round_budget": 50, "round_remaining": 14,
            "cumulative_optimizer_steps": 36, "optimizer_step": 36, "data_passes": 6,
            "next_refresh_optimizer_step": 50, "native_epoch": 6}}


class Heartbeat:
    def __init__(self, job, *, busy=True):
        model = {"id": "demo", "backend": "demo", "status": "loaded"}
        self.snapshot = {"models": [model], "adapters": [{"id": "checkpoint-30", "base_model_id": "student",
            "status": "failed", "evaluation_id": "evaluation-30"}],
            "status": {"model": model, "learning_busy": busy, "agent_busy": False},
            "jobs": {"train": [], "sleep": [job]}, "revision": 1, "capabilities": {},
            "gpu": {"available": False, "gpus": [], "stale": False, "errors": []}}
        self.paths = []
        self.disconnected = False

    def handle(self, request):
        self.paths.append(request.url.path)
        assert request.url.path in {"/system/bootstrap", "/system/telemetry"}
        if self.disconnected:
            return httpx.Response(503, json={"detail": "Temporarily disconnected"})
        return httpx.Response(200, json=self.snapshot)


@pytest.mark.asyncio
async def test_successful_update_heartbeat_changes_counters_without_registry_or_checkpoint_changes(monkeypatch):
    job = adaptive_job()
    sealed = copy.deepcopy(job["adaptive"]["state"])
    backend = Heartbeat(job)
    # A newer terminal record with unfinished historical metadata must not
    # hide the genuinely running older job after an explicit resume.
    backend.snapshot["jobs"]["sleep"].append({"id": "newer_stopped", "status": "failed",
        "started_at": "2026-11-12T13:00:00Z", "progress": {"terminal": False, "restored": False}})
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view="sleep")
    async with app.run_test(size=(130, 55)) as pilot:
        await pilot.pause()
        assert app.selected_learning_job()[1]["id"] == "sleep_resumed"
        assert "sleep_resumed" in app.query_one("#sleep-run", Static).render().plain
        counter = app.query_one("#sleep-counters", Static)
        changes, rebuilds = [], []
        original_update, original_registry = counter.update, app.render_registry
        def update(value):
            changes.append(True)
            return original_update(value)
        def registry():
            rebuilds.append(True)
            return original_registry()
        monkeypatch.setattr(counter, "update", update)
        monkeypatch.setattr(app, "render_registry", registry)
        job["progress"].update(cumulative_optimizer_steps=37, optimizer_step=37, round_remaining=13, native_epoch=6.25)
        await app.refresh_panels()
        text = counter.render().plain
        assert "TOTAL OPTIMIZER UPDATES // 37" in text and "ROUND UPDATES LEFT // 13" in text
        assert "CHECKPOINTED UPDATES // 36" in text and "ADAPTIVE EPOCH // 1 // budget reset cycle" in text
        assert "DATA PASSES // 6" in text and "TRAINER DATA EPOCH 6.25" in text
        assert app._jobs["sleep"][0]["adaptive"]["state"] == sealed
        await app.refresh_panels()
        assert len(changes) == 1 and not rebuilds
        assert backend.paths.count("/system/bootstrap") == 1
        job["status"] = "queued"
        job["adaptive"].update(development_assessment={"target_families": ["arithmetic"]},
                              last_refresh={"step": 50, "new_verified_examples": 12})
        job["progress"].update(stage="evaluating", label="Evaluating candidate on frozen admission cases")
        await app.refresh_panels()
        assert "EVALUATING // Evaluating candidate on frozen admission cases" in app.query_one("#sleep-stage", Static).render().plain
        assert app.query_one("#sleep-run", Static).render().plain.endswith("/ EVALUATING")
        assert app._jobs["sleep"][0]["status"] == "queued"
        plan = app.query_one("#sleep-plan", Static).render().plain
        assert "PRACTICE TARGETS // arithmetic // independent development" in plan
        assert "LAST DATA REFRESH AT UPDATE // 50" in plan and "NEW VERIFIED EXAMPLES // 12" in plan
        assert not rebuilds


@pytest.mark.asyncio
async def test_paused_counters_use_sealed_state_and_identify_last_evaluated_checkpoint():
    job = adaptive_job()
    job["status"] = "cancelled"
    job["progress"] = {"stage": "finished", "label": "Sleep paused", "terminal": True,
                       "restored": True, "overall_fraction": 1, "native_epoch": 6}
    backend = Heartbeat(job, busy=False)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view="sleep")
    async with app.run_test(size=(130, 55)) as pilot:
        await pilot.pause()
        counters = app.query_one("#sleep-counters", Static).render().plain
        assert "TOTAL OPTIMIZER UPDATES // 36" in counters and "ROUND UPDATES LEFT // 14" in counters
        assert "ADAPTIVE EPOCH // 1" in counters and "DATA PASSES // 6" in counters
        assert "CHECKPOINTED UPDATES // 36" in counters
        plan = app.query_one("#sleep-plan", Static).render().plain
        assert "EXPERT // checkpoint-36" in plan and "no candidate created" not in plan
        assert "ADMISSION CHECKPOINT // checkpoint-30 // last completed evaluation" in plan
        assert "ADMISSION EVALUATION // evaluation-30" in plan
        overall = app.query_one("#sleep-overall-label", Static).render().plain
        assert "PAUSED // admission not reached" in overall and "100%" not in overall
        app._sleep_pending_job = {**job, "status": "queued", "progress": {"terminal": False, "restored": False}}
        assert app.selected_learning_job()[1]["status"] == "queued"


@pytest.mark.asyncio
async def test_disconnected_dashboard_marks_retained_counters_until_reconnection():
    backend = Heartbeat(adaptive_job())
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_view="sleep")
    async with app.run_test(size=(130, 55)) as pilot:
        await pilot.pause()
        backend.disconnected = True
        await app.refresh_panels()
        assert "DISCONNECTED — LAST RECEIVED COUNTERS" in app.query_one("#sleep-run", Static).render().plain
        assert "TOTAL OPTIMIZER UPDATES // 36" in app.query_one("#sleep-counters", Static).render().plain
        backend.disconnected = False
        await app.refresh_panels()
        assert "DISCONNECTED" not in app.query_one("#sleep-run", Static).render().plain
