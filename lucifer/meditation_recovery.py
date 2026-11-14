"""Proof required before discarding an uncommitted optimizer segment.

This contract permits another finite trial from unchanged input files after the
owning worker has exited. It never authorizes replay of a committed result or an
uncertain live effect, and does not infer retry safety from exception text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .head_pack import file_hash


CONTRACT = "discard_uncommitted_head_segment_v1"


class WorkerMemoryFailure(RuntimeError):
    def __init__(self, receipt):
        super().__init__("Owned optimizer exited with a verified discardable memory failure")
        self.receipt = receipt


def config_hash(config):
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(json.dumps(public, sort_keys=True, allow_nan=False).encode()).hexdigest()


def input_paths(config):
    paths = {str(Path(config["head_path"]).resolve())}
    data = config.get("dataset", config)
    paths.update(str(Path(data[key]).resolve()) for key in ("features_path", "feature_path", "labels_path") if data.get(key))
    resume = config.get("resume") or config.get("resume_from")
    if resume:
        directory = Path(resume)
        state_path = directory / "optimizer.json"
        state = json.loads(state_path.read_text())
        paths.add(str(state_path.resolve()))
        for name in state["hashes"]:
            if Path(name).name != name:
                raise ValueError("Optimizer input binding must use ordinary checkpoint members")
            paths.add(str((directory / name).resolve()))
    return paths


def capture_inputs(config):
    return {path: file_hash(path) for path in sorted(input_paths(config))}


def seal_failure(config, *, resource, worker_path):
    """Called by the worker only for typed CUDA allocation/VRAM-limit errors."""
    if config.get("mode") != "train" or resource not in {"cuda_oom", "vram"}:
        return None
    before = config.get("_memory_retry_inputs")
    if not before:
        return None
    try:
        unchanged = before == capture_inputs(config)
    except (OSError, ValueError, KeyError):
        unchanged = False
    committed = (Path(config["output_dir"]) / "result.json").exists()
    return {"contract": CONTRACT, "safe_to_discard": unchanged and not committed,
            "config_sha256": config_hash(config), "worker_sha256": file_hash(worker_path),
            "resource": resource, "inputs": before, "result_committed": committed,
            "discarded_updates": int(config.get("_uncommitted_successful_updates", 0))}


def verify_failure(config, receipt, *, worker_path):
    """The controller must additionally establish that its worker has exited."""
    if (not isinstance(receipt, dict) or receipt.get("contract") != CONTRACT
            or receipt.get("safe_to_discard") is not True or receipt.get("result_committed") is not False
            or receipt.get("resource") not in {"cuda_oom", "vram"}
            or receipt.get("config_sha256") != config_hash(config)
            or receipt.get("worker_sha256") != file_hash(worker_path)
            or (Path(config["output_dir"]) / "result.json").exists()):
        return False
    try:
        return receipt.get("inputs") == capture_inputs(config)
    except (OSError, ValueError, KeyError):
        return False


def smaller_batch(batch, accumulation, *, retries, max_retries=3):
    """Propose a bounded retry preserving the exact nominal effective batch."""
    if any(type(value) is not int or value < 1 for value in (batch, accumulation, max_retries)):
        raise ValueError("Invalid memory recovery batch dimensions")
    if type(retries) is not int or retries < 0:
        raise ValueError("Invalid memory recovery counter")
    if retries >= max_retries or batch <= 1:
        return None
    effective = batch * accumulation
    candidate = batch // 2
    while candidate > 0:
        if effective % candidate == 0 and effective // candidate <= 1024:
            return {"batch_size": candidate, "accumulation": effective // candidate,
                    "effective_batch_size": effective, "retry": retries + 1}
        candidate -= 1
    return None


async def train_with_recovery(service, ident, policy, config):
    """Retry only a failure already verified/discarded by the owning controller."""
    base = Path(config["output_dir"])
    failures = service.store.get("meditation_runs", ident).get("memory_failure_count", 0)
    current = {**config, "output_dir": str(base.with_name(base.name + f"-memory-{failures}")) if failures else str(base)}
    retries = 0
    while True:
        try:
            return await service._worker(ident, current)
        except WorkerMemoryFailure:
            proposal = smaller_batch(policy.batch_size, policy.accumulation, retries=retries)
            if proposal is None:
                raise
            policy.batch_size = proposal["batch_size"]
            policy.accumulation = proposal["accumulation"]
            retries = proposal["retry"]
            run = service.store.get("meditation_runs", ident)
            if run.get("source_revoked") or run.get("source_invalidated"):
                raise ValueError("Memory recovery cannot reuse revoked sources")
            service.store.update("meditation_runs", ident, policy=policy.snapshot(), last_memory_recovery=proposal)
            service._progress(ident, "memory_recovery", policy=policy.snapshot(),
                reason="Discarded uncommitted trial; smaller microbatch preserves effective example count",
                successful_updates=policy.successful_updates)
            current = {**config, "batch_size": policy.batch_size, "accumulation": policy.accumulation,
                "output_dir": str(base.with_name(base.name + f"-memory-{run.get('memory_failure_count', 0)}"))}
