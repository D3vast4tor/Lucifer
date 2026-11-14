"""Finite disposable initial-rate trials; heldout scores are never an input."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from .head_pack import file_hash
from .meditation_policy import MeditationPolicy
from .meditation_retention import retention_receipt
from .meditation_evaluation import evaluate_with_fallback
from .meditation_recovery import train_with_recovery
from .meditation_sources import digest


def discard_checkpoints(run_dir, probe_dir, paths):
    owned = Path(run_dir).resolve()
    for value in paths:
        checkpoint = Path(value)
        if (checkpoint.is_symlink() or checkpoint.parent.is_symlink() or checkpoint.name != "checkpoint"
                or checkpoint.resolve().parent.parent != owned
                or not checkpoint.parent.name.startswith(probe_dir.name)):
            raise ValueError("Calibration checkpoint is outside its owned run")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)


async def calibrate_initial_rate(service, ident, policy, *, exported, train, development,
                                 baseline_loss, feature_contract, run_dir, seed):
    if policy.successful_updates:
        return None  # Continue the established adaptive optimizer after resume.
    run = service.store.get("meditation_runs", ident)
    precision = run.get("training_precision", "bf16")
    inputs = {"protocol": 1, "feature_contract": feature_contract, "compute_precision": precision,
        "retention_contract": run.get("retention_contract"),
        "initial_head_sha256": file_hash(exported["head_path"]), "shape": exported["shape"],
        "seed": seed, "baseline_development_loss": baseline_loss,
        "minimum_rate": policy.minimum_rate, "maximum_rate": policy.maximum_rate}
    input_signature = digest(inputs)
    saved = run.get("learning_rate_calibration") or {}
    if saved.get("input_sha256") == input_signature and saved.get("status") == "completed":
        policy.learning_rate = saved["selected_learning_rate"]
        policy.batch_size, policy.accumulation = saved["batch_size"], saved["accumulation"]
        return saved
    if saved and saved.get("input_sha256") != input_signature:
        history = (run.get("calibration_history") or []) + [saved]
        service.store.update("meditation_runs", ident, calibration_history=history)
        saved = {}
    rates = sorted({max(policy.minimum_rate, min(policy.maximum_rate, policy.learning_rate * factor))
                    for factor in (0.5, 1.0, 2.0)}) if not saved else saved["binding"]["rates"]
    binding = saved.get("binding") or {**inputs, "rates": rates, "steps_per_probe": 5,
        "batch_size": policy.batch_size, "accumulation": policy.accumulation}
    signature = digest(binding)
    report = {"contract_sha256": signature, "input_sha256": input_signature, "binding": binding, "status": "running",
              "trials": saved.get("trials", []), "pending_trial": saved.get("pending_trial"),
              "total_successful_updates": saved.get("total_successful_updates", 0)}
    service.store.update("meditation_runs", ident, learning_rate_calibration=report)
    effective_batch = binding["batch_size"] * binding["accumulation"]
    minimum_batch = policy.batch_size
    for index, rate in enumerate(rates):
        run = service.store.get("meditation_runs", ident)
        if run.get("source_revoked") or run.get("source_invalidated"):
            raise ValueError("Calibration sources were revoked")
        probe_dir = Path(run_dir) / f"calibration-{signature[:16]}-{index}"
        recorded = next((row for row in report["trials"] if row["index"] == index), None)
        if recorded is None:
            trial_policy = MeditationPolicy.restore(policy.snapshot())
            trial_policy.learning_rate = rate
            trial_policy.batch_size, trial_policy.accumulation = minimum_batch, effective_batch // minimum_batch
            service._progress(ident, "calibrating_rate", calibration_trial=index + 1,
                calibration_trials=len(rates), learning_rate=rate, calibration_current_updates=0,
                successful_updates=policy.successful_updates)
            pending = report.get("pending_trial") or {}
            if pending and pending.get("index") != index:
                raise ValueError("Calibration continuation belongs to another rate")
            completed = pending.get("completed_updates", 0)
            checkpoints = list(pending.get("checkpoints", []))
            trained = pending.get("trained_receipt")
            while completed < 5:
                service._check_disk_budget(Path(run_dir))
                if pending and file_hash(pending["head_path"]) != pending["head_sha256"]:
                    raise ValueError("Calibration continuation head changed")
                trained = await train_with_recovery(service, ident, trial_policy, {
                    "mode": "train", "head_path": pending.get("head_path", exported["head_path"]),
                    "compute_precision": precision,
                    "resume": pending.get("checkpoint_path"), "shape": exported["shape"],
                    "dataset": train, "steps": 5 - completed, "learning_rate": rate, "device": "cuda", "seed": seed,
                    "batch_size": trial_policy.batch_size, "accumulation": trial_policy.accumulation,
                    "parent_head_sha256": exported["bf16_head_sha256"], "calibration_trial": True,
                    "vram_limit_bytes": 6 * 1024**3, "ram_limit_bytes": 24 * 1024**3,
                    "output_dir": str(probe_dir if completed == 0 else probe_dir.with_name(probe_dir.name + f"-continue-{completed}"))})
                updates = trained.get("segment_updates", 0)
                if type(updates) is not int or not 0 < updates <= 5 - completed:
                    raise ValueError("Incomplete finite learning-rate trial")
                completed += updates
                if trained.get("successful_updates") != completed:
                    raise ValueError("Calibration checkpoint update counter changed")
                checkpoints.append(trained["checkpoint_path"])
                pending = {"index": index, "completed_updates": completed, "head_path": trained["head_path"],
                    "head_sha256": trained["after_head_sha256"], "checkpoint_path": trained["checkpoint_path"],
                    "checkpoints": checkpoints, "trained_receipt": trained}
                report["pending_trial"] = pending
                service.store.update("meditation_runs", ident, learning_rate_calibration=report, resumable=True)
                # A newer complete checkpoint replaces earlier partial probes.
                # Keep at most two sets during a continuation's atomic commit.
                discard_checkpoints(run_dir, probe_dir, checkpoints[:-1])
            measured = await evaluate_with_fallback(service, ident, {"mode": "evaluate", "head_path": trained["head_path"],
                "head_sha256": trained["after_head_sha256"],
                "shape": exported["shape"], "datasets": {"development": development},
                **({"reference_head_path": exported["head_path"], "reference_head_sha256": exported["bf16_head_sha256"]}
                   if run.get("retention_contract") else {}),
                "output_dir": str(probe_dir.with_name(probe_dir.name + "-development"))})
            loss = measured["datasets"]["development"]["loss"]
            if measured.get("head_sha256") != trained["after_head_sha256"]:
                raise ValueError("Development calibration scored another head")
            if not math.isfinite(loss):
                raise ValueError("Nonfinite development calibration")
            distribution = (retention_receipt(measured, contract=run["retention_contract"],
                head_sha256=trained["after_head_sha256"], reference_sha256=exported["bf16_head_sha256"])
                if run.get("retention_contract") else None)
            recorded = {"index": index, "learning_rate": rate, "development_loss": loss,
                "successful_updates": completed, "head_sha256": trained["after_head_sha256"],
                "last_segment_changed_elements": trained["changed_elements"],
                "head_changed": trained["after_head_sha256"] != binding["initial_head_sha256"],
                "batch_size": trial_policy.batch_size,
                "checkpoint_path": trained["checkpoint_path"], "disposable_checkpoints": checkpoints,
                "resources": trained.get("resources", {}), "distribution_retention": distribution}
            report["trials"].append(recorded)
            report["total_successful_updates"] += completed
            report["pending_trial"] = None
            # Seal the numerical result before deleting its disposable weights.
            service.store.update("meditation_runs", ident, learning_rate_calibration=report)
        minimum_batch = min(minimum_batch, recorded["batch_size"])
        discard_checkpoints(run_dir, probe_dir, recorded.get("disposable_checkpoints", [recorded["checkpoint_path"]]))
    nonregressing = [row for row in report["trials"] if row["development_loss"] <= baseline_loss
                    and (row.get("distribution_retention") or {"passed": True})["passed"]]
    chosen = min(nonregressing, key=lambda row: (row["development_loss"], row["learning_rate"])) if nonregressing else min(
        report["trials"], key=lambda row: row["learning_rate"])
    policy.learning_rate = chosen["learning_rate"]
    policy.batch_size, policy.accumulation = minimum_batch, effective_batch // minimum_batch
    report.update(status="completed", selected_learning_rate=policy.learning_rate,
        batch_size=policy.batch_size, accumulation=policy.accumulation,
        reason="Best nonregressing disposable development probe" if nonregressing else
               "No probe improved development; start the smallest tested conservative rate",
        heldout_used=False, optimizer_state_retained=False)
    service.store.update("meditation_runs", ident, learning_rate_calibration=report, policy=policy.snapshot())
    service._progress(ident, "calibrated_rate", learning_rate=policy.learning_rate,
        calibration_current_updates=0, successful_updates=policy.successful_updates, reason=report["reason"])
    return report
