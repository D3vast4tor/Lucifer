"""Read-only CPU evaluation fallback; never repeats optimizer effects."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path

from .head_pack import file_hash
from .store import now


CONTRACT = "readonly_head_evaluation_cpu_fallback_v1"
PARITY_TOLERANCE = 1e-5


def capture_identity(config):
    """Optimizer policy source is provenance, not frozen-backbone arithmetic."""
    return {key: value for key, value in config.items()
            if key not in {"optimizer_support_sha256", "output_dir"}}


def completed_preparation(service, ident, config):
    """Reuse only a sealed identical capture, keeping its original contract.

    The numeric worker, capture implementation, tokenizer, dataset, model and
    execution plan remain exact matches. Changing calibration/evaluation policy
    does not alter those features or authorize repeating optimizer effects.
    """
    owned = service._run_dir(ident).resolve()
    for operation in service.store.list("meditation_operations"):
        binding = operation.get("binding", {})
        previous = binding.get("config", {})
        if (operation.get("phase") != "worker" or operation.get("status") != "completed"
                or binding.get("run_id") != ident or previous.get("mode") != "prepare"
                or capture_identity(previous) != capture_identity(config)):
            continue
        output = Path(previous["output_dir"])
        result = output / "result.json"
        if (output.is_symlink() or not output.resolve().is_relative_to(owned)
                or not result.is_file() or file_hash(result) != operation.get("result_sha256")):
            raise ValueError("Committed frozen-feature preparation changed")
        return operation
    return None


def abandon_superseded_preparation(service, ident, operation):
    """Reconcile an exited redundant capture without touching its files."""
    config = operation.get("binding", {}).get("config", {})
    if config.get("mode") != "prepare" or completed_preparation(service, ident, config) is None:
        return False
    output = Path(config["output_dir"])
    owned = service._run_dir(ident).resolve()
    private_config = output / "worker_config.json"
    if (output.is_symlink() or not output.resolve().is_relative_to(owned)
            or not private_config.is_file() or json.loads(private_config.read_text()) != config
            or config_is_live(private_config) or (output / "result.json").exists()):
        return False
    service.store.update("meditation_operations", operation["id"], status="abandoned_read_only",
        recovery_contract="superseded_frozen_capture_v1", reconciled_at=now())
    return True


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EvaluationMemoryFailure(RuntimeError):
    def __init__(self, receipt, operation_id=None):
        super().__init__("Read-only head evaluation exceeded its GPU budget")
        self.receipt = receipt
        self.operation_id = operation_id


def config_is_live(config_path):
    """Inspect actual same-user process handles, not a stale job/lock file."""
    wanted = str(Path(config_path).resolve()).encode()
    proc = Path("/proc")
    if not proc.is_dir():
        return True  # No authoritative process visibility: do not reconcile.
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = (path / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except (PermissionError, OSError):
            return True
        if b"--config" in args:
            index = args.index(b"--config") + 1
            if index < len(args) and args[index] == wanted:
                return True
    return False


def evaluation_failure(config, error, *, run, run_dir, worker_path):
    """Recover only a hash-bound VRAM-limit failure in the known reader.

    Older worker error receipts retain frames rather than exception text. Bind
    the frame to the exact pinned source and its explicit VRAM raise, so a RAM,
    data-integrity or arbitrary worker failure cannot authorize this fallback.
    """
    if (config.get("mode") != "evaluate" or config.get("device", "cuda") != "cuda"
            or error.get("mode") != "evaluate" or error.get("error_type") != "ResourceLimitError"):
        return None
    worker_path = Path(worker_path).resolve()
    pinned = (run.get("native_runtime_binding") or {}).get("config", {}).get("optimizer_worker_sha256")
    if not pinned or file_hash(worker_path) != pinned:
        return None
    tree = ast.parse(worker_path.read_text())
    vram_lines = {node.lineno for function in tree.body
                  if isinstance(function, ast.FunctionDef) and function.name == "_resource_check"
                  for node in ast.walk(function) if isinstance(node, ast.Raise)
                  and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
                  and node.exc.func.id == "ResourceLimitError"
                  and any(keyword.arg == "resource" and isinstance(keyword.value, ast.Constant)
                          and keyword.value.value == "vram" for keyword in node.exc.keywords)}
    matching_frame = any(frame.get("function") == "_resource_check" and frame.get("line") in vram_lines
                         and Path(frame.get("file", "")).resolve() == worker_path for frame in error.get("frames", []))
    if not matching_frame:
        return None
    owned = Path(run_dir).resolve()
    output = Path(config["output_dir"])
    config_path = output / "worker_config.json"
    if (output.is_symlink() or not output.resolve().is_relative_to(owned)
            or (output / "result.json").exists() or not config_path.is_file()
            or json.loads(config_path.read_text()) != config or config_is_live(config_path)):
        return None
    expected_head = config.get("head_sha256")
    if not expected_head:
        pending = (run.get("learning_rate_calibration") or {}).get("pending_trial") or {}
        if pending.get("head_path") != config.get("head_path"):
            return None
        expected_head = pending.get("head_sha256")
    reference_hash = config.get("reference_head_sha256")
    expected = {config.get("head_path"): expected_head, config.get("reference_head_path"): reference_hash}
    datasets = config.get("datasets") or {}
    if set(datasets) != {"development"}:
        return None
    for value in datasets.values():
        expected[value.get("features_path", value.get("feature_path"))] = value.get("features_sha256")
        expected[value.get("labels_path")] = value.get("labels_sha256")
    for name, checksum in expected.items():
        if not isinstance(name, str) or not isinstance(checksum, str):
            return None
        path = Path(name)
        if not path.resolve().is_relative_to(owned) or not path.is_file() or file_hash(path) != checksum:
            return None
    return {"contract": CONTRACT, "config_sha256": digest(config), "worker_sha256": pinned,
            "inputs": expected, "resource": "vram", "mutation_replayed": False,
            "no_matching_live_worker": True, "result_committed": False}


def select_cpu(service, ident, receipt, *, operation_id=None):
    run = service.store.get("meditation_runs", ident)
    if run.get("source_revoked") or run.get("source_invalidated"):
        raise ValueError("Evaluation source consent or evidence was revoked")
    if run.get("evaluation_device") == "cpu":
        raise ValueError("CPU evaluation already selected; no repeated fallback is allowed")
    if receipt.get("contract") != CONTRACT:
        raise ValueError("Unknown evaluation recovery contract")
    with service.store.transaction():
        service.store.update("meditation_runs", ident, evaluation_device="cpu", resumable=True, evaluation_fallback={
            **receipt, "selected_at": now(), "prior_operation_id": operation_id})
        if operation_id:
            service.store.update("meditation_operations", operation_id, status="abandoned_read_only",
                recovery_contract=CONTRACT, reconciled_at=now())


async def ensure_cpu_parity(service, ident):
    run = service.store.get("meditation_runs", ident)
    reference = run.get("evaluation_baseline")
    if not reference:
        raise ValueError("CPU evaluation needs the committed GPU development baseline")
    signature = digest(reference)
    previous = run.get("evaluation_device_parity") or {}
    if previous.get("reference_sha256") == signature and previous.get("passed") is True:
        return previous
    service._progress(ident, "evaluation_cpu_parity", evaluation_device="cpu",
        reason="Verify the fixed development baseline before CPU evaluation")
    measured = await service._worker(ident, {"mode": "evaluate", "device": "cpu",
        "head_path": reference["head_path"], "head_sha256": reference["head_sha256"], "shape": reference["shape"],
        "datasets": {"development": reference["dataset"]},
        "output_dir": str(service._run_dir(ident) / "evaluation-cpu-parity" / signature)})
    gpu, cpu = reference["measurement"], measured["datasets"]["development"]
    complete = (measured.get("head_sha256") == reference["head_sha256"] and cpu.get("tokens") == gpu.get("tokens")
                and type(cpu.get("tokens")) is int and cpu["tokens"] > 0
                and bool(gpu.get("group_losses"))
                and cpu.get("group_losses", {}).keys() == gpu.get("group_losses", {}).keys())
    deltas = [abs(cpu["loss"] - gpu["loss"])]
    if complete:
        deltas.extend(abs(cpu["group_losses"][key] - gpu["group_losses"][key]) for key in gpu["group_losses"])
    passed = complete and all(math.isfinite(value) and value <= PARITY_TOLERANCE for value in deltas)
    receipt = {"reference_sha256": signature, "head_sha256": reference["head_sha256"],
               "maximum_loss_delta": max(deltas), "tolerance": PARITY_TOLERANCE, "passed": passed,
               "device": "cpu", "activation_dtype": "F32", "logit_dtype": "F32",
               "bit_identical_claimed": False, "holdout_used": False}
    service.store.update("meditation_runs", ident, evaluation_device_parity=receipt)
    if not passed:
        raise ValueError("CPU/GPU development baseline arithmetic did not pass pinned parity")
    return receipt


async def evaluate_with_fallback(service, ident, config):
    run = service.store.get("meditation_runs", ident)
    if run.get("evaluation_device") == "cpu":
        await ensure_cpu_parity(service, ident)
        service._progress(ident, "evaluation_cpu", evaluation_device="cpu",
            reason="Read-only development evaluation offloaded within the host RAM budget")
        config = {**config, "device": "cpu", "output_dir": str(config["output_dir"]) + "-cpu"}
    try:
        return await service._worker(ident, config)
    except EvaluationMemoryFailure as error:
        select_cpu(service, ident, error.receipt, operation_id=error.operation_id)
        await ensure_cpu_parity(service, ident)
        service._progress(ident, "evaluation_cpu", evaluation_device="cpu",
            reason="Read-only development evaluation offloaded after verified GPU-budget failure")
        return await service._worker(ident, {**config, "device": "cpu", "output_dir": str(config["output_dir"]) + "-cpu"})
