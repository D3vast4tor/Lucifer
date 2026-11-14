"""Owned worker progress survives actual TQDM/JSON physical-line merging."""
import asyncio
import json
from pathlib import Path

import pytest

from lucifer.learning import LearningService, worker_progress_events
from lucifer.config import Settings
from lucifer.store import Store
from test_experts import Manager, anchors, checkpoint, reviewed


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml", eval_file=tmp_path / "evals.yaml")
    store = Store(settings.data_dir / "state.db")
    yield settings, store
    store.close()


def event(step, **extra):
    return {"event": "training_progress", "step": step, "total_steps": 18, "epoch": .75, "loss": 1.25, **extra}


def test_progress_extracts_carriage_return_ansi_and_multiple_complete_records():
    physical = "\r 33%|█████| 6/18 [00:58<01:49, 9.12s/it]\x1b[0m" + json.dumps(event(6))
    physical += "\r 39%|██████| 7/18 [01:07<01:39, 9.01s/it]" + json.dumps(event(7)) + "\n"
    assert worker_progress_events(physical) == [event(6), event(7)]


def test_progress_ignores_diagnostic_nesting_and_recovers_after_non_json_trainer_prefix():
    nested = json.dumps({"diagnostic": event(17)})
    physical = nested + "\r {'loss': 1.25, 'epoch': .75} \r" + json.dumps(event(6))
    assert worker_progress_events(physical) == [event(6)]


@pytest.mark.parametrize("bad", [event(True), event(-1), event(19), event(1, total_steps=0),
                               event(1, total_steps=True), event(1, event="unrelated"),
                               {"event": "training_progress", "step": 1}])
def test_progress_rejects_invalid_or_unrelated_counters(bad):
    assert worker_progress_events(json.dumps(bad)) == []


def test_progress_rejects_duplicate_keys_and_nonfinite_records():
    assert worker_progress_events('{"event":"training_progress","step":1,"step":2,"total_steps":18}') == []
    assert worker_progress_events(json.dumps(event(1, loss=float('nan')))) == []


@pytest.mark.asyncio
async def test_merged_owned_worker_line_updates_cached_training_progress(env, monkeypatch):
    settings, store = env
    anchors(settings)
    reviewed(store, "source", "Explain arithmetic addition for integers")
    manager = Manager()
    service = LearningService(settings, store, manager)
    reached, release = asyncio.Event(), asyncio.Event()
    physical = "\r 33%|█████| 6/18 [00:58<01:49, 9.12s/it]" + json.dumps(event(6)) + "\n"
    class Output:
        first = True
        async def readline(self):
            if self.first:
                self.first = False
                return physical.encode()
            reached.set()
            await release.wait()
            return b""
    class Process:
        stdout = Output()
        returncode = 0
        async def wait(self):
            return 0
    async def launch(*args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        checkpoint(Path(config["output_dir"]), config["base_model"])
        return Process()
    async def evaluate(ident):
        return {"id": "unverified", "passed": False, "quality_status": "unknown"}
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(service, "evaluate", evaluate)
    try:
        run = await service.start_sleep({"dynamic": True})
        await asyncio.wait_for(reached.wait(), 1)
        current = store.get("sleep_runs", run["id"])
        assert current["status"] == "training"
        assert current["progress"]["optimizer_step"] == current["progress"]["completed"] == 6
        assert current["progress"]["optimizer_steps_total"] == current["progress"]["total"] == 18
        assert current["progress"]["fraction"] == pytest.approx(1 / 3)
        assert current["progress"]["loss"] == 1.25
        assert not current["progress"]["terminal"]
        release.set()
        await service.tasks[run["id"]]
    finally:
        release.set()
        await service.close()
