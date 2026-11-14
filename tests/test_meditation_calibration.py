"""Disposable development-only rate search and its interruption contract."""
from pathlib import Path

import pytest

from lucifer.head_pack import file_hash
from lucifer.meditation_calibration import calibrate_initial_rate
from lucifer.meditation_policy import MeditationPolicy
from test_meditation_service import harness, seed_run


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    head = tmp_path / "baseline.bf16"
    head.write_bytes(b"original baseline")
    exported = {"head_path": str(head), "bf16_head_sha256": file_hash(head), "shape": [2, 2]}
    calls = []
    async def worker(ident, config):
        calls.append(config)
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if config["mode"] == "train":
            assert config["dataset"] == {"role": "train"}
            assert config["head_path"] == str(head) and config.get("resume") is None
            assert config["steps"] == 5 and config["calibration_trial"] is True
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            probe = checkpoint / "head.bf16"
            probe.write_text(str(config["learning_rate"]))
            return {"head_path": str(probe), "checkpoint_path": str(checkpoint), "segment_updates": 5, "successful_updates": 5,
                    "changed_elements": 4, "after_head_sha256": file_hash(probe)}
        assert config["mode"] == "evaluate" and config["datasets"] == {"development": {"role": "development"}}
        rate = float(Path(config["head_path"]).read_text())
        return {"head_sha256": file_hash(config["head_path"]),
                "datasets": {"development": {"loss": {5e-6: 2.1, 1e-5: 1.8, 2e-5: 1.7}[rate]}}}
    monkeypatch.setattr(service, "_worker", worker)
    options = {"exported": exported, "train": {"role": "train"}, "development": {"role": "development"},
               "baseline_loss": 2.0, "feature_contract": "pinned-features", "run_dir": service._run_dir(run["id"]), "seed": 19}
    yield service, store, run, head, calls, options
    store.close()


@pytest.mark.asyncio
async def test_rate_trials_are_disposable_and_completed_search_is_not_replayed(calibration):
    service, store, run, head, calls, options = calibration
    policy = MeditationPolicy()
    report = await calibrate_initial_rate(service, run["id"], policy, **options)
    assert policy.learning_rate == 2e-5 and policy.successful_updates == 0
    assert report["total_successful_updates"] == 15 and report["heldout_used"] is False
    assert report["optimizer_state_retained"] is False and len(calls) == 6
    assert head.read_bytes() == b"original baseline"
    assert all(not Path(row["checkpoint_path"]).exists() for row in report["trials"])
    # The selected rate differs from the initial rate. Resume must retain the
    # committed search rather than expanding a new search around that rate.
    again = await calibrate_initial_rate(service, run["id"], policy, **options)
    assert again == report and len(calls) == 6
    assert store.get("meditation_runs", run["id"])["policy"]["successful_updates"] == 0


@pytest.mark.asyncio
async def test_probe_result_is_sealed_before_cleanup_and_resumes_without_training_again(calibration, monkeypatch):
    import lucifer.meditation_calibration as module
    service, store, run, _, calls, options = calibration
    remove = module.shutil.rmtree
    once = {"fail": True}
    def interrupted(path):
        if once["fail"]:
            once["fail"] = False
            raise RuntimeError("Interrupted after numerical receipt")
        return remove(path)
    monkeypatch.setattr(module.shutil, "rmtree", interrupted)
    policy = MeditationPolicy()
    with pytest.raises(RuntimeError, match="numerical receipt"):
        await calibrate_initial_rate(service, run["id"], policy, **options)
    pending = store.get("meditation_runs", run["id"])["learning_rate_calibration"]
    assert len(pending["trials"]) == 1 and pending["total_successful_updates"] == 5
    report = await calibrate_initial_rate(service, run["id"], policy, **options)
    assert report["status"] == "completed" and len(calls) == 6
    assert report["total_successful_updates"] == 15


@pytest.mark.asyncio
async def test_existing_optimizer_updates_skip_initial_calibration(calibration):
    service, _, run, _, calls, options = calibration
    assert await calibrate_initial_rate(service, run["id"], MeditationPolicy(successful_updates=25), **options) is None
    assert calls == []


@pytest.mark.asyncio
async def test_partial_probe_continues_its_checkpoint_without_repeating_committed_updates(calibration, monkeypatch):
    service, store, run, _, calls, options = calibration
    original_worker, original_update = service._worker, store.update
    once = {"partial": True, "interrupt": True}
    async def worker(ident, config):
        if config["mode"] == "train" and config.get("resume"):
            calls.append(config)
            assert config["steps"] == 3
            assert Path(config["head_path"]).parent == Path(config["resume"])
            checkpoint = Path(config["output_dir"]) / "checkpoint"
            checkpoint.mkdir(parents=True)
            head = checkpoint / "head.bf16"
            head.write_text(str(config["learning_rate"]))
            return {"head_path": str(head), "checkpoint_path": str(checkpoint), "segment_updates": 3,
                    "successful_updates": 5, "changed_elements": 4, "after_head_sha256": file_hash(head)}
        result = await original_worker(ident, config)
        if config["mode"] == "train" and once["partial"]:
            once["partial"] = False
            result.update(segment_updates=2, successful_updates=2)
        return result
    def update(kind, ident, **changes):
        result = original_update(kind, ident, **changes)
        pending = (changes.get("learning_rate_calibration") or {}).get("pending_trial") or {}
        if pending.get("completed_updates") == 2 and once["interrupt"]:
            once["interrupt"] = False
            raise RuntimeError("Pause after partial checkpoint commit")
        return result
    monkeypatch.setattr(service, "_worker", worker)
    monkeypatch.setattr(store, "update", update)
    policy = MeditationPolicy()
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        await calibrate_initial_rate(service, run["id"], policy, **options)
    assert len(calls) == 1
    report = await calibrate_initial_rate(service, run["id"], policy, **options)
    assert report["status"] == "completed" and report["total_successful_updates"] == 15
    assert sum(c["mode"] == "train" and not c.get("resume") for c in calls) == 3
    assert sum(c["mode"] == "train" and c.get("resume") is not None for c in calls) == 1
