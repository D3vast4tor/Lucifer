"""Quota admission includes the next checkpoint and excludes parent symlinks."""
import pytest

from lucifer.meditation_storage import checkpoint_reserve, feature_reserve, require_room, StorageLimitError


def test_next_artifact_must_fit_before_creation(tmp_path):
    (tmp_path / "existing").write_bytes(b"x" * 80)
    with pytest.raises(StorageLimitError):
        require_room(tmp_path, reserve_bytes=30, limit_bytes=100, free_reserve_bytes=0)
    receipt = require_room(tmp_path, reserve_bytes=20, limit_bytes=100, free_reserve_bytes=0)
    assert receipt["used_bytes"] == 80 and receipt["reserved_bytes"] == 20


def test_shared_parent_symlinks_do_not_charge_original_model_bytes(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "weights").write_bytes(b"x" * 1000)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parent").symlink_to(parent, target_is_directory=True)
    (run / "weights").symlink_to(parent / "weights")
    assert require_room(run, limit_bytes=1, free_reserve_bytes=0)["used_bytes"] == 0


def test_checkpoint_and_feature_reservations_cover_numeric_arrays():
    assert checkpoint_reserve([129280, 5120]) == 9_333_899_264
    datasets = {"train": [{"labels": [-100, 7, 8]}], "holdout": [{"labels": [-100, -100, 9]}]}
    assert feature_reserve(datasets, 5120, raw_limit_bytes=0) == 3 * (5120 * 4 + 8 + 1024) + (512 << 20)


@pytest.mark.asyncio
async def test_reservation_rejects_before_worker_launch_and_prelaunch_intent_can_resume(tmp_path, monkeypatch):
    import asyncio
    from test_meditation_service import harness, seed_run
    from lucifer.meditation import EvidenceRequired
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    output = service._run_dir(run["id"]) / "segment-1"
    config = {"mode": "train", "shape": [2, 2], "output_dir": str(output)}
    def full(*args, **kwargs):
        assert kwargs["reserve_bytes"] == checkpoint_reserve([2, 2])
        raise EvidenceRequired("Disk reservation rejected")
    async def forbidden(*args, **kwargs):
        pytest.fail("Capacity rejection must precede worker launch")
    resumed = []
    async def resume(ident):
        resumed.append(ident)
    monkeypatch.setattr(service, "_check_disk_budget", full)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(service, "_run", resume)
    try:
        with pytest.raises(EvidenceRequired, match="reservation"):
            await service._worker(run["id"], config)
        operation = next(row for row in store.list("meditation_operations") if row["phase"] == "worker")
        assert not operation.get("started_at") and not output.exists()
        store.update("meditation_runs", run["id"], status="awaiting_evidence")
        await service.resume(run["id"], "resume-preflight")
        await service.tasks[run["id"]]
        assert resumed == [run["id"]]
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_native_capacity_rejection_can_retry_but_committed_result_cannot(tmp_path, monkeypatch):
    import asyncio
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from test_meditation_service import harness, seed_run
    from lucifer.head_pack import atomic_json, file_hash
    from lucifer.meditation import EvidenceRequired
    import lucifer.meditation as controller
    from lucifer.meditation_recovery import config_hash
    service, store, _ = harness(tmp_path)
    run = seed_run(service, store)
    output = service._run_dir(run["id"]) / "features"
    config = {"mode": "prepare", "output_dir": str(output)}
    calls = []
    async def empty():
        if False:
            yield b""
    async def spawn(*args, **kwargs):
        calls.append(args)
        assert not (output / "error.json").exists()
        code = 1 if len(calls) == 1 else 0
        if code:
            payload = {"preflight_rejection": {"contract": "native_feature_storage_preflight_v1",
                "config_sha256": config_hash(config),
                "worker_sha256": file_hash(Path(controller.__file__).with_name("meditation_worker.py"))}}
            (output / "error.json").write_text(json.dumps(payload))
        else:
            atomic_json(output / "result.json", {"mode": "prepare", "capacity": "admitted"})
        async def wait():
            return code
        return SimpleNamespace(pid=123456789, returncode=code, stdout=empty(), wait=wait)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    try:
        with pytest.raises(EvidenceRequired, match="disk/cache budget"):
            await service._worker(run["id"], config)
        operation = next(row for row in store.list("meditation_operations") if row["phase"] == "worker")
        assert operation["status"] == "capacity_rejected" and operation["started_at"] is None
        result = await service._worker(run["id"], config)
        assert result["capacity"] == "admitted"
        assert await service._worker(run["id"], config) == result
        assert len(calls) == 2
    finally:
        await service.close()
        store.close()
