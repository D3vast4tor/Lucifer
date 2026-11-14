"""Only an exited, uncommitted, unchanged-input trial may be discarded."""
from pathlib import Path

import pytest

from lucifer.meditation_recovery import capture_inputs, seal_failure, smaller_batch, verify_failure


@pytest.fixture
def config(tmp_path):
    for name in ("head", "features", "labels", "worker"):
        (tmp_path / name).write_bytes(name.encode())
    return {"mode": "train", "head_path": str(tmp_path / "head"),
            "dataset": {"features_path": str(tmp_path / "features"), "labels_path": str(tmp_path / "labels")},
            "output_dir": str(tmp_path / "trial"), "batch_size": 32, "accumulation": 4}


def test_discard_contract_requires_unchanged_inputs_and_no_committed_result(config, tmp_path):
    config["_memory_retry_inputs"] = capture_inputs(config)
    receipt = seal_failure(config, resource="cuda_oom", worker_path=tmp_path / "worker")
    assert receipt["safe_to_discard"] and verify_failure(config, receipt, worker_path=tmp_path / "worker")
    Path(config["output_dir"]).mkdir()
    (Path(config["output_dir"]) / "result.json").write_text('{}')
    assert not verify_failure(config, receipt, worker_path=tmp_path / "worker")


@pytest.mark.parametrize("changed", ["head", "features", "labels", "worker"])
def test_changed_source_checkpoint_or_worker_blocks_memory_retry(config, tmp_path, changed):
    config["_memory_retry_inputs"] = capture_inputs(config)
    receipt = seal_failure(config, resource="vram", worker_path=tmp_path / "worker")
    (tmp_path / changed).write_bytes(b"changed")
    assert not verify_failure(config, receipt, worker_path=tmp_path / "worker")


def test_shape_or_optimizer_request_change_cannot_reuse_failure_receipt(config, tmp_path):
    config["_memory_retry_inputs"] = capture_inputs(config)
    receipt = seal_failure(config, resource="vram", worker_path=tmp_path / "worker")
    assert not verify_failure({**config, "batch_size": 16}, receipt, worker_path=tmp_path / "worker")
    assert seal_failure(config, resource="ram", worker_path=tmp_path / "worker") is None
    assert seal_failure(config, resource="disk", worker_path=tmp_path / "worker") is None


def test_bounded_memory_policy_preserves_effective_batch():
    assert smaller_batch(32, 4, retries=0) == {
        "batch_size": 16, "accumulation": 8, "effective_batch_size": 128, "retry": 1}
    assert smaller_batch(16, 8, retries=1)["accumulation"] == 16
    assert smaller_batch(8, 16, retries=2)["effective_batch_size"] == 128
    assert smaller_batch(4, 32, retries=3) is None
    assert smaller_batch(1, 128, retries=0) is None
    assert smaller_batch(2, 1024, retries=0) is None


def test_memory_policy_handles_odd_batches_without_changing_effective_count():
    result = smaller_batch(7, 4, retries=0)
    assert result["batch_size"] * result["accumulation"] == 28


@pytest.mark.asyncio
async def test_owned_exited_worker_is_discarded_before_memory_retry(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from test_meditation_service import harness, seed_run
    from lucifer.meditation_recovery import WorkerMemoryFailure
    import lucifer.meditation as controller
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    head, features, labels = [tmp_path / name for name in ("head", "features", "labels")]
    for path in (head, features, labels):
        path.write_bytes(b"immutable input")
    output = service._run_dir(run["id"]) / "segment-1"
    config = {"mode": "train", "head_path": str(head), "shape": [2, 2], "dataset": {
        "features_path": str(features), "labels_path": str(labels)}, "output_dir": str(output)}
    async def empty():
        if False:
            yield b""
    async def spawn(*args, **kwargs):
        actual = json.loads((output / "worker_config.json").read_text())
        actual["_memory_retry_inputs"] = capture_inputs(actual)
        receipt = seal_failure(actual, resource="cuda_oom", worker_path=Path(controller.__file__).with_name("meditation_worker.py"))
        (output / "error.json").write_text(json.dumps({"memory_failure": receipt}))
        (output / "checkpoint").mkdir()
        (output / "checkpoint/partial-head").write_bytes(b"uncommitted")
        async def wait():
            return 1
        return SimpleNamespace(pid=123456789, returncode=1, stdout=empty(), wait=wait)
    monkeypatch.setattr(controller.asyncio, "create_subprocess_exec", spawn)
    try:
        with pytest.raises(WorkerMemoryFailure):
            await service._worker(run["id"], config)
        assert not (output / "checkpoint").exists()
        assert head.read_bytes() == b"immutable input"
        assert store.get("meditation_runs", run["id"])["memory_failure_count"] == 1
        operation = next(row for row in store.list("meditation_operations") if row["phase"] == "worker")
        assert operation["status"] == "discarded"
        assert not service.processes
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_controller_reduces_only_microbatch_and_preserves_resume_and_updates(tmp_path, monkeypatch):
    from test_meditation_service import harness, seed_run
    from lucifer.meditation_policy import MeditationPolicy
    from lucifer.meditation_recovery import WorkerMemoryFailure, train_with_recovery
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    policy = MeditationPolicy(successful_updates=50)
    calls = []
    async def worker(ident, config):
        calls.append(config)
        if len(calls) <= 2:
            store.update("meditation_runs", ident, memory_failure_count=len(calls))
            raise WorkerMemoryFailure({"safe_to_discard": True})
        return {"successful_updates": 75}
    monkeypatch.setattr(service, "_worker", worker)
    config = {"mode": "train", "head_path": "original-head", "resume": "immutable-checkpoint", "seed": 19,
              "steps": 25, "batch_size": 32, "accumulation": 4,
              "output_dir": str(service._run_dir(run["id"]) / "segment-3")}
    try:
        result = await train_with_recovery(service, run["id"], policy, config)
        assert result["successful_updates"] == 75 and policy.successful_updates == 50
        assert [(c["batch_size"], c["accumulation"]) for c in calls] == [(32, 4), (16, 8), (8, 16)]
        assert all(c["batch_size"] * c["accumulation"] == 128 for c in calls)
        assert all(c["head_path"] == "original-head" and c["resume"] == "immutable-checkpoint"
                   and c["seed"] == 19 and c["steps"] == 25 for c in calls)
        assert len({c["output_dir"] for c in calls}) == 3
        assert store.get("meditation_runs", run["id"])["policy"]["batch_size"] == 8
    finally:
        await service.close()
        store.close()
