"""Scalar validation receipts prevent duplicate provider effects after resume."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from lucifer.adaptive_validation import ValidationPending, authorize_validation_resume, measure_panel
from lucifer.store import Store


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / "test.db")
    class Service:
        def __init__(self):
            self.store = store
            self.calls = []
            self.progress = []
            self.fail_on = None
            self.incomplete_on = None
            self.revision = "profile-1"
            self.manager = SimpleNamespace(current=lambda: {"id": "student", "max_output_tokens": 4096})
        def _profile_revision(self, spec):
            return self.revision
        async def _evaluation_token_count(self, messages, model_id, budget):
            return {"input_tokens": 5, "estimated": False, "method": "test"}
        async def _evaluation_answer(self, messages, adapter, model_id, budget, token_count=None):
            # A committed call ID/fingerprint must already exist before effects.
            started = [r for r in store.list("adaptive_operations")
                       if r["status"] == "started" and r.get("phase") == "validation_call"]
            assert started and all(r["fingerprint"] for r in started)
            self.calls.append((messages[0]["content"], adapter.get("id") if adapter else None, budget))
            if len(self.calls) == self.fail_on:
                raise asyncio.CancelledError
            complete = len(self.calls) != self.incomplete_on
            text = "no" if "safety" in messages[0]["content"] else "42"
            return {"answer": text, "complete": complete, "finish_reason": "stop" if complete else "length",
                    "dialect": "chat_completions", "input_tokens": 5, "token_count_estimated": False,
                    "reasoning": "PRIVATE_REASONING", "nested": {"private": "PRIVATE_TEXT"}}
        def _progress(self, *args, **kwargs):
            self.progress.append(kwargs)
    result = Service()
    yield result
    store.close()


def suite():
    return {"metadata": {"freeze_digest": "frozen-suite", "case_budgets": {"task": 2048, "safety": 2048}},
            "cases": [{"id": "task", "suite": "core", "prompt": "PRIVATE_PROMPT task", "expected_exact": "42"},
                      {"id": "safety", "suite": "safety", "prompt": "PRIVATE_PROMPT safety", "expected_exact": "no"}]}


def adapter(name="checkpoint-a", digest="a" * 64):
    return {"id": name, "base_model_id": "student", "checkpoint_manifest_sha256": digest,
            "artifact_verification": {"sha256": digest}}


@pytest.mark.asyncio
async def test_completed_panel_reuses_all_scalar_receipts_and_balances_order(service):
    rows = await measure_panel(service, suite(), "student", None, adapter(), repetitions=3, run_id="run")
    assert len(rows) == 6 and all(row["scored"] for row in rows)
    assert len(service.calls) == 14  # two warmups + two cases x3x2 routes
    assert [call[1] for call in service.calls[2:8]] == [None, "checkpoint-a", "checkpoint-a", None, None, "checkpoint-a"]
    saved = json.dumps(service.store.list("adaptive_operations"))
    assert "PRIVATE" not in saved
    assert "'answer':" not in saved and '"reasoning"' not in saved
    assert all(row["status"] == "completed" for row in service.store.list("adaptive_operations"))
    again = await measure_panel(service, suite(), "student", None, adapter(), repetitions=3, run_id="run")
    assert again == rows and len(service.calls) == 14
    assert service.progress[-1]["eval_completed"] == service.progress[-1]["eval_total"] == 12


@pytest.mark.asyncio
async def test_unknown_started_request_pauses_without_repeating_completed_or_unknown_calls(service):
    service.fail_on = 4
    with pytest.raises(asyncio.CancelledError):
        await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert len(service.calls) == 4
    operations = service.store.list("adaptive_operations")
    assert sum(row["status"] == "completed" for row in operations) == 3
    assert sum(row["status"] == "started" for row in operations) == 1
    service.fail_on = None
    with pytest.raises(ValidationPending, match="uncertain"):
        await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert len(service.calls) == 4


@pytest.mark.asyncio
async def test_same_physical_adapter_warmup_is_shared_across_pair_labels(service):
    await measure_panel(service, suite(), "student", None, adapter("a"), repetitions=1, run_id="run")
    assert len(service.calls) == 6
    await measure_panel(service, suite(), "student", adapter("a"), adapter("b", "b" * 64), repetitions=1, run_id="run")
    assert len(service.calls) == 11  # only b needs a new warm-up, plus four paired calls
    warmups = [call for call in service.calls if call[2] == 32]
    assert [call[1] for call in warmups] == [None, "a", "b"]


@pytest.mark.asyncio
async def test_identity_changes_start_distinct_panel_operations(service):
    await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert len(service.calls) == 6
    await measure_panel(service, suite(), "student", None, adapter(digest="b" * 64), repetitions=1, run_id="run")
    assert len(service.calls) == 11  # cached base warmup; new candidate warmup + four measured
    service.revision = "profile-2"
    await measure_panel(service, suite(), "student", None, adapter(digest="b" * 64), repetitions=1, run_id="run")
    assert len(service.calls) == 17


@pytest.mark.asyncio
async def test_incomplete_response_stays_unscored_and_is_not_retried(service):
    service.incomplete_on = 3
    rows = await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert rows[0]["baseline_pass"] is None and not rows[0]["scored"]
    assert len(service.calls) == 6
    assert await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run") == rows
    assert len(service.calls) == 6


@pytest.mark.asyncio
async def test_frozen_budget_does_not_silently_shrink(service):
    service.manager.current = lambda: {"id": "student", "max_output_tokens": 512}
    with pytest.raises(ValidationPending, match="frozen validation budget"):
        await measure_panel(service, suite(), "student", None, adapter(), run_id="run")
    assert not service.calls and not service.store.list("adaptive_operations")


@pytest.mark.asyncio
async def test_changed_receipt_fails_closed_without_new_effect(service):
    await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    first = service.store.list("adaptive_operations")[0]
    altered = copy.deepcopy(first["result"])
    altered["latency"] = 0
    service.store.update("adaptive_operations", first["id"], result=altered)
    with pytest.raises(ValueError, match="receipt changed"):
        await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert len(service.calls) == 6


async def interrupted_panel(service):
    service.fail_on = 4
    with pytest.raises(asyncio.CancelledError):
        await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    service.store.put("sleep_runs", {"id": "run", "status": "cancelled"})
    return next(row for row in service.store.list("adaptive_operations") if row["status"] == "started")


def operation_bytes(service):
    return dict(service.store.conn.execute("SELECT id,data FROM documents WHERE kind='adaptive_operations'"))


@pytest.mark.asyncio
async def test_explicit_terminal_resume_authorizes_only_unknown_call_and_keeps_completed_bytes(service):
    unknown = await interrupted_panel(service)
    before = operation_bytes(service)
    immutable = {row["id"]: before[row["id"]] for row in service.store.list("adaptive_operations") if row["status"] == "completed"}
    service.store.put("adaptive_operations", {"id": "training-effect", "run_id": "run",
                     "phase": "training_segment", "status": "started", "binding": {}})
    grant = authorize_validation_resume(service, "run", "opaque-resume-1")
    assert grant["authorized_count"] == 1
    prepared = service.store.get("adaptive_operations", unknown["id"])
    assert prepared["status"] == "prepared"
    assert prepared["fingerprint"] == unknown["fingerprint"] and prepared["binding"] == unknown["binding"]
    assert prepared["call_id"] != unknown["call_id"]
    previous = service.store.get("adaptive_validation_attempts", unknown["current_attempt_id"])
    assert previous["status"] == "uncertain" and previous["call_id"] == unknown["call_id"]
    new = service.store.get("adaptive_validation_attempts", prepared["current_attempt_id"])
    assert new["status"] == "prepared" and new["authorization_id"] == "opaque-resume-1"
    assert new["previous_attempt_id"] == previous["id"]
    snapshot = json.dumps(service.store.list("adaptive_validation_attempts"), sort_keys=True)
    assert authorize_validation_resume(service, "run", "opaque-resume-1") == grant
    assert json.dumps(service.store.list("adaptive_validation_attempts"), sort_keys=True) == snapshot
    service.fail_on = None
    rows = await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert all(row["scored"] for row in rows)
    assert len(service.calls) == 7  # one authorized retry + two as-yet unissued logical calls
    after = operation_bytes(service)
    assert all(after[ident] == raw for ident, raw in immutable.items())
    assert service.store.get("adaptive_operations", "training-effect")["status"] == "started"
    assert "PRIVATE" not in json.dumps(service.store.list("adaptive_validation_attempts"))
    assert "PRIVATE" not in json.dumps(service.store.list("adaptive_validation_resumes"))


@pytest.mark.asyncio
async def test_reusing_authorization_does_not_reauthorize_a_second_unknown_attempt(service):
    unknown = await interrupted_panel(service)
    authorize_validation_resume(service, "run", "resume-once")
    service.fail_on = 5
    with pytest.raises(asyncio.CancelledError):
        await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    before = operation_bytes(service)
    authorize_validation_resume(service, "run", "resume-once")
    assert operation_bytes(service) == before
    assert service.store.get("adaptive_operations", unknown["id"])["status"] == "started"
    again = authorize_validation_resume(service, "run", "resume-twice")
    assert again["authorized_count"] == 1
    current = service.store.get("adaptive_operations", unknown["id"])
    assert service.store.get("adaptive_validation_attempts", current["current_attempt_id"])["attempt_number"] == 3


@pytest.mark.asyncio
async def test_active_run_cannot_authorize_retry(service):
    await interrupted_panel(service)
    service.store.update("sleep_runs", "run", status="training")
    with pytest.raises(ValidationPending, match="stopped run"):
        authorize_validation_resume(service, "run", "resume")
    assert not service.store.list("adaptive_validation_resumes")


@pytest.mark.asyncio
async def test_unknown_call_outside_tools_disabled_contract_cannot_be_retried(service):
    import lucifer.adaptive_validation as validation
    unknown = await interrupted_panel(service)
    changed = copy.deepcopy(unknown)
    changed["binding"]["tools"] = "enabled"
    changed["fingerprint"] = validation._digest(changed["binding"])
    changed["id"] = "adaptive_validation_" + changed["fingerprint"][:32]
    service.store.delete("adaptive_operations", unknown["id"])
    service.store.put("adaptive_operations", changed)
    before = operation_bytes(service)
    with pytest.raises(ValidationPending, match="outside the read-only"):
        authorize_validation_resume(service, "run", "resume")
    assert operation_bytes(service) == before
    assert not service.store.list("adaptive_validation_resumes")


@pytest.mark.asyncio
async def test_legacy_unknown_call_has_honest_archival_record_and_new_call_identity(service):
    unknown = await interrupted_panel(service)
    service.store.delete("adaptive_validation_attempts", unknown["current_attempt_id"])
    old = {key: value for key, value in unknown.items() if key not in {"current_attempt_id", "call_id"}}
    service.store.put("adaptive_operations", old)
    authorize_validation_resume(service, "run", "resume-legacy")
    attempts = [row for row in service.store.list("adaptive_validation_attempts") if row["operation_id"] == old["id"]]
    assert len(attempts) == 2
    legacy = next(row for row in attempts if row["legacy_without_call_id"])
    assert legacy["status"] == "uncertain" and legacy["call_id"] is None
    new = next(row for row in attempts if not row["legacy_without_call_id"])
    assert new["call_id"] and new["status"] == "prepared"
    assert new["previous_attempt_id"] == legacy["id"]


@pytest.mark.asyncio
async def test_completed_validation_calls_are_not_changed_by_resume_authorization(service):
    await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    service.store.put("sleep_runs", {"id": "run", "status": "cancelled"})
    before = operation_bytes(service)
    grant = authorize_validation_resume(service, "run", "resume-completed")
    assert grant["authorized_count"] == 0
    assert operation_bytes(service) == before
    assert len(service.calls) == 6


@pytest.mark.asyncio
async def test_runtime_restart_rewarms_routes_without_changing_completed_measurements(service):
    runtime = {"id": "student", "max_output_tokens": 4096, "pid": 123,
               "created_at": "2026-11-12T20:00:00Z", "loaded_at": "2026-11-12T21:00:00Z"}
    service.manager.current = lambda: dict(runtime)
    first = await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert len(service.calls) == 6
    before = operation_bytes(service)
    measured_ids = {row["id"] for row in service.store.list("adaptive_operations")
                    if row["binding"]["kind"] == "measured"}
    assert all("runtime_identity" not in row["binding"] for row in service.store.list("adaptive_operations")
               if row["id"] in measured_ids)
    runtime["pid"] = 456
    runtime["loaded_at"] = "2026-11-12T22:00:00Z"
    second = await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run")
    assert second == first
    assert len(service.calls) == 8
    assert all(call[2] == 32 for call in service.calls[-2:])
    after = operation_bytes(service)
    assert all(after[ident] == before[ident] for ident in measured_ids)
    warmups = [row for row in service.store.list("adaptive_operations") if row["binding"]["kind"] == "warmup"]
    assert len(warmups) == 4
    assert {row["binding"]["runtime_identity"]["pid"] for row in warmups} == {123, 456}
    # Another reconnect to this same process reuses both warmed routes again.
    assert await measure_panel(service, suite(), "student", None, adapter(), repetitions=1, run_id="run") == first
    assert len(service.calls) == 8
