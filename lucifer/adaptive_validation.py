"""Journaled adaptive validation with scalar-only, reusable call receipts.

All calls use LearningService._evaluation_answer, whose production path disables
thinking and tools. Reconnects never retry an unfinished call. Only an explicit
terminal-run resume can authorize a new attempt under the tools-disabled,
read-only LLM contract; completed receipts are always reused. No response text
is persisted.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid


class ValidationPending(ValueError):
    """Evidence is unavailable or an earlier request has an uncertain outcome."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()



READ_ONLY_RETRY_CONTRACT = "tools_disabled_read_only_validation_v1"
TERMINAL_RESUME_STATES = {"cancelled", "interrupted", "awaiting_evidence", "failed"}


def _attempt(service, operation, *, number=1, authorization_id=None, previous_attempt_id=None, legacy=False):
    ident = "adaptive_validation_attempt_" + uuid.uuid4().hex
    return service.store.put("adaptive_validation_attempts", {
        "id": ident, "operation_id": operation["id"], "run_id": operation["run_id"],
        "fingerprint": operation["fingerprint"], "attempt_number": number,
        "call_id": None if legacy else uuid.uuid4().hex,
        "authorization_id": authorization_id, "previous_attempt_id": previous_attempt_id,
        "retry_contract": READ_ONLY_RETRY_CONTRACT,
        "legacy_without_call_id": legacy, "status": "uncertain" if legacy else "prepared"})


def authorize_validation_resume(service, run_id, authorization_id):
    """Authorize only uncertain pure-validation calls after explicit resume.

    The caller must hold its learning/control lock and have validated all source
    lineage. Call before changing the terminal job to queued. This authorization
    never applies to teacher generation, training, tools or checkpoint effects.
    """
    if not isinstance(authorization_id, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", authorization_id) is None:
        raise ValueError("An opaque bounded validation-resume authorization ID is required")
    ident = "adaptive_validation_resume_" + _digest([run_id, authorization_id])[:32]
    with service.store.transaction():
        try:
            receipt = service.store.get("adaptive_validation_resumes", ident)
        except KeyError:
            receipt = None
        if receipt is not None:
            result = receipt.get("result")
            if (receipt.get("run_id") != run_id or receipt.get("authorization_id") != authorization_id
                    or not isinstance(result, dict) or receipt.get("result_sha256") != _digest(result)):
                raise ValueError("Validation-resume authorization receipt changed")
            return result
        job = service.store.get("sleep_runs", run_id)
        if job.get("status") not in TERMINAL_RESUME_STATES or getattr(service, "busy", False):
            raise ValidationPending("Validation retry needs an explicit resume of a stopped run")
        grants = []
        operations = service.store.list("adaptive_operations", run_id=run_id)
        for operation in operations:
            if operation.get("phase") != "validation_call" or operation.get("status") not in {"started", "uncertain"}:
                continue
            binding = operation.get("binding")
            if not isinstance(binding, dict):
                raise ValidationPending("An uncertain validation call has no verifiable contract")
            fingerprint = _digest(binding)
            if (operation.get("fingerprint") != fingerprint
                    or operation.get("id") != "adaptive_validation_" + fingerprint[:32]
                    or binding.get("run_id") != run_id or binding.get("schema") != 1
                    or binding.get("tools") != "disabled" or binding.get("thinking") != "off"
                    or type(binding.get("temperature")) not in (int, float) or binding["temperature"] != 0
                    or binding.get("kind") not in {"warmup", "measured"}):
                raise ValidationPending("An uncertain call is outside the read-only validation retry contract")
            prior_id = operation.get("current_attempt_id")
            if prior_id:
                prior = service.store.get("adaptive_validation_attempts", prior_id)
                if (prior.get("operation_id") != operation["id"] or prior.get("fingerprint") != fingerprint
                        or prior.get("status") not in {"started", "uncertain"}):
                    raise ValidationPending("The uncertain validation attempt identity changed")
                prior = service.store.update("adaptive_validation_attempts", prior_id, status="uncertain")
            else:
                # Older calls had no attempt IDs. Preserve that fact rather than
                # inventing a historical call ID recorded before their effect.
                prior = _attempt(service, operation, legacy=True)
                prior_id = prior["id"]
            attempt = _attempt(service, operation, number=prior["attempt_number"] + 1,
                               authorization_id=authorization_id, previous_attempt_id=prior_id)
            service.store.update("adaptive_operations", operation["id"], status="prepared",
                                 current_attempt_id=attempt["id"], call_id=attempt["call_id"],
                                 retry_authorization_id=authorization_id,
                                 read_only_retry_contract=READ_ONLY_RETRY_CONTRACT)
            grants.append({"operation_id": operation["id"], "attempt_id": attempt["id"],
                           "call_id": attempt["call_id"]})
        result = {"authorization_id": authorization_id, "retry_contract": READ_ONLY_RETRY_CONTRACT,
                  "authorized_count": len(grants), "prepared_attempts": grants}
        service.store.put("adaptive_validation_resumes", {"id": ident, "run_id": run_id,
            "authorization_id": authorization_id, "status": "completed", "result": result,
            "result_sha256": _digest(result)})
        return result


def _artifact(adapter):
    if adapter is None:
        return {"id": None, "format": "base"}
    return {"id": adapter.get("id"), "base_model_id": adapter.get("base_model_id"),
            "training_model": adapter.get("training_model"), "format": adapter.get("format"),
            "checkpoint_manifest_sha256": adapter.get("checkpoint_manifest_sha256"),
            "weights_sha256": (adapter.get("artifact_verification") or {}).get("sha256")}


def _profile(service, model_id):
    current = service.manager.current() or {}
    if current.get("id") != model_id:
        raise ValidationPending("Load the validation panel's registered student profile first")
    registry = getattr(service.manager, "capabilities", None)
    snapshot = registry.get(model_id) if registry is not None else None
    revision = getattr(snapshot, "profile_revision", None)
    if revision is None and hasattr(service, "_profile_revision"):
        revision = service._profile_revision(current)
    if revision is None:
        revision = _digest({key: current.get(key) for key in (
            "id", "backend", "path", "training_model", "adapter_backend", "context_length", "max_output_tokens")})
    return revision, current


def _completion(result):
    """Explicit scalar allowlist: arbitrary provider fields can contain text."""
    finish = result.get("finish_reason")
    finishes = {"stop", "end_turn", "length", "max_tokens", "tool_calls", "tool_use", "legacy_text", "cancel", "error"}
    dialect = result.get("dialect")
    dialects = {"chat_completions", "anthropic_messages", "responses", "legacy"}
    value = {"complete": result.get("complete") is True,
             "finish_reason": finish if isinstance(finish, str) and finish in finishes else None,
             "dialect": dialect if isinstance(dialect, str) and dialect in dialects else None}
    count = result.get("input_tokens")
    if type(count) is int and count >= 0:
        value["input_tokens"] = count
    if type(result.get("token_count_estimated")) is bool:
        value["token_count_estimated"] = result["token_count_estimated"]
    reasons = {"provider_error", "startup_transport_unknown", "tool_call_or_incomplete_final", "scoring_error"}
    if isinstance(result.get("reason"), str) and result["reason"] in reasons:
        value["reason"] = result["reason"]
    return value


def _claim(service, binding):
    fingerprint = _digest(binding)
    ident = "adaptive_validation_" + fingerprint[:32]
    with service.store.transaction():
        try:
            operation = service.store.get("adaptive_operations", ident)
        except KeyError:
            operation = service.store.put("adaptive_operations", {"id": ident, "run_id": binding["run_id"],
                "phase": "validation_call", "fingerprint": fingerprint, "binding": binding, "status": "prepared"})
        if operation.get("fingerprint") != fingerprint or operation.get("binding") != binding:
            raise ValueError("Validation call identity changed")
        if operation.get("status") == "completed":
            result = operation.get("result")
            if not isinstance(result, dict) or operation.get("result_sha256") != _digest(result):
                raise ValueError("Completed validation receipt changed")
            allowed = {"complete", "pass", "latency", "completion", "answer_sha256"}
            if set(result) != allowed or type(result.get("complete")) is not bool:
                raise ValueError("Invalid scalar validation receipt")
            if result.get("pass") is not None and type(result["pass"]) is not bool:
                raise ValueError("Invalid validation score")
            if type(result.get("latency")) not in (int, float) or not math.isfinite(result["latency"]) or result["latency"] < 0:
                raise ValueError("Invalid validation latency")
            if _completion(result["completion"]) != result["completion"]:
                raise ValueError("Validation completion receipt contains unexpected fields")
            if not isinstance(result.get("answer_sha256"), str) or len(result["answer_sha256"]) != 64:
                raise ValueError("Invalid validation answer digest")
            return operation, result
        if operation.get("status") != "prepared":
            raise ValidationPending("A validation call has an uncertain outcome; completed calls were not repeated")
        attempt_id = operation.get("current_attempt_id")
        if attempt_id:
            attempt = service.store.get("adaptive_validation_attempts", attempt_id)
            if (attempt.get("operation_id") != ident or attempt.get("fingerprint") != fingerprint
                    or attempt.get("status") != "prepared" or not attempt.get("call_id")
                    or attempt.get("retry_contract") != READ_ONLY_RETRY_CONTRACT):
                raise ValidationPending("The prepared validation attempt is unavailable or changed")
            authorization = attempt.get("authorization_id")
            if authorization is not None:
                receipt_id = "adaptive_validation_resume_" + _digest([binding["run_id"], authorization])[:32]
                receipt = service.store.get("adaptive_validation_resumes", receipt_id)
                grant = {"operation_id": ident, "attempt_id": attempt["id"], "call_id": attempt["call_id"]}
                if (receipt.get("status") != "completed" or receipt.get("result_sha256") != _digest(receipt.get("result"))
                        or grant not in receipt.get("result", {}).get("prepared_attempts", [])):
                    raise ValidationPending("The validation retry has no committed explicit authorization")
            elif attempt.get("attempt_number") != 1:
                raise ValidationPending("The validation retry has no explicit authorization")
        else:
            attempt = _attempt(service, operation)
        # Commit both opaque call identity and immutable input fingerprint before
        # any awaited provider effect. Completed attempts are never reopened.
        service.store.update("adaptive_validation_attempts", attempt["id"], status="started")
        operation = service.store.update("adaptive_operations", ident, status="started",
                                         current_attempt_id=attempt["id"], call_id=attempt["call_id"])
    return operation, None


async def _call(service, binding, messages, adapter, model_id, budget, case=None, token_count=None):
    operation, cached = _claim(service, binding)
    if cached is not None:
        return cached
    started = time.perf_counter()
    try:
        result = await service._evaluation_answer(messages, adapter, model_id, budget, token_count)
    except asyncio.CancelledError:
        # The prepared identity remains started: a reconnect cannot resend it.
        raise
    except Exception:
        raise ValidationPending("A validation call has no completion receipt; inspect before resuming") from None
    elapsed = time.perf_counter() - started
    complete = result.get("complete") is True
    answer = result.get("answer")
    if not isinstance(answer, str):
        answer, complete = "", False
    passed = None
    if complete and case is not None:
        from .learning import check_case
        try:
            passed = bool(check_case(case, answer))
        except Exception:
            complete = False
            result = {"complete": False, "reason": "scoring_error"}
    receipt = {"complete": complete, "pass": passed, "latency": elapsed,
               "completion": _completion({**result, "complete": complete}),
               "answer_sha256": hashlib.sha256(answer.encode()).hexdigest()}
    # Discard the response channels before entering the durable store.
    del result, answer
    with service.store.transaction():
        service.store.update("adaptive_operations", operation["id"], status="completed", result=receipt,
                             result_sha256=_digest(receipt))
        service.store.update("adaptive_validation_attempts", operation["current_attempt_id"],
                             status="completed", result_sha256=_digest(receipt))
    return receipt


async def measure_panel(service, suite, model_id, baseline, candidate=None, *, repetitions=3, run_id=None):
    """Balanced paired validation; completed scalar call receipts survive resume.

    run_id=None is a stable standalone panel scope. To request independent fresh
    measurements, use a new run_id rather than changing or deleting receipts.
    Frozen case budgets are honored; an incompatible runtime pauses evaluation.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Validation repetitions must be a positive integer")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("A validation panel needs unique frozen cases")
    profile_revision, current = _profile(service, model_id)
    suite_hash = _digest(suite)
    common = {"schema": 1, "run_id": run_id or "standalone_" + _digest([suite_hash, model_id]),
              "suite_sha256": suite_hash, "suite_freeze_digest": suite.get("metadata", {}).get("freeze_digest"),
              "model_id": model_id, "profile_revision": profile_revision,
              "tools": "disabled", "thinking": "off", "temperature": 0}
    choices = [("baseline", baseline)] if candidate is None else [("baseline", baseline), ("candidate", candidate)]
    pair = {"baseline": _artifact(baseline), "candidate": _artifact(candidate) if candidate is not None else None}
    maximum = current.get("max_output_tokens")
    budgets = suite.get("metadata", {}).get("case_budgets", {})
    for case in cases:
        budget = budgets.get(case["id"], 2048)
        if type(budget) is not int or budget < 1:
            raise ValueError("Frozen validation case budget is invalid")
        if type(maximum) is int and budget > maximum:
            raise ValidationPending("The loaded student cannot serve the frozen validation budget")
    # The physical route identity shares a warm-up if the same checkpoint is
    # subsequently used as the other member of another paired comparison.
    runtime_identity = {key: current[key] for key in ("pid", "created_at", "loaded_at")
                        if current.get(key) is not None}
    for _, adapter in choices:
        binding = {**common, "kind": "warmup", "adapter": _artifact(adapter), "budget": 32}
        # A route warmed in a prior process does not establish warm latency in
        # this runtime. Measured receipts retain their immutable logical identity.
        if runtime_identity:
            binding["runtime_identity"] = runtime_identity
        warm = await _call(service, binding, [{"role": "user", "content": "Reply with OK only."}],
                           adapter, model_id, 32)
        if not warm["complete"]:
            raise ValidationPending("Validation warm-up did not complete")
    rows, done = [], 0
    total = len(cases) * repetitions * len(choices)
    for index, case in enumerate(cases):
        budget = budgets.get(case["id"], 2048)
        messages = [{"role": "user", "content": case["prompt"]}]
        count = await service._evaluation_token_count(messages, model_id, budget)
        for repetition in range(repetitions):
            ordered = choices if (index + repetition) % 2 == 0 else list(reversed(choices))
            row = {"id": case["id"], "suite": case.get("suite", "core"), "repetition": repetition}
            for label, adapter in ordered:
                binding = {**common, "kind": "measured", "pair": pair, "label": label,
                           "adapter": _artifact(adapter), "case_id": case["id"], "case_sha256": _digest(case),
                           "repetition": repetition, "budget": budget}
                receipt = await _call(service, binding, messages, adapter, model_id, budget, case, count)
                row[label + "_pass"] = receipt["pass"]
                row[label + "_latency"] = receipt["latency"]
                row[label + "_completion"] = receipt["completion"]
                row[label + "_answer_sha256"] = receipt["answer_sha256"]
                done += 1
                if run_id:
                    panel = "development" if suite.get("metadata", {}).get("suite_purpose") == "development" else "frozen admission"
                    service._progress(run_id, stage="evaluating", label=f"Evaluating {label} on {panel} cases", completed=done, total=total,
                                      eval_completed=done, eval_total=total, case_id=case["id"], evaluating=label)
            row["scored"] = all(row[label + "_pass"] is not None for label, _ in choices)
            rows.append(row)
    return rows
