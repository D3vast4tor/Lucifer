"""Mocked controller acceptance, recovery and source-revocation boundaries.

The worker writes tiny synthetic adapter/checkpoint artifacts. No model, GPU,
network request or production store is involved.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from lucifer import adaptive_learning as controller
from lucifer.adaptive import AdaptiveState, commit_checkpoint
from lucifer.store import Store


class Manager:
    def __init__(self):
        self.specs = {"teacher": {"id": "teacher", "backend": "strata", "path": "primary-pinned"},
                      "student": {"id": "student", "backend": "vllm", "path": "student-pinned", "max_output_tokens": 4096}}
        self.selected = "teacher"
        self.runtime_adapter = None
        self.events = []

    def list(self):
        return list(self.specs.values())

    def current(self):
        return self.specs.get(self.selected)

    async def load(self, ident):
        self.events.append(("load", ident))
        if ident != self.selected:
            self.runtime_adapter = None
        self.selected = ident

    async def unload(self, ident):
        assert self.selected == ident
        self.events.append(("unload", ident))
        self.selected = None

    async def activate_adapter(self, adapter):
        self.runtime_adapter = adapter["id"] if adapter else None
        self.events.append(("activate", adapter["id"] if adapter else None))


class Service:
    def __init__(self, root):
        self.settings = SimpleNamespace(data_dir=root)
        self.store = Store(root / "state.sqlite")
        self.manager = Manager()
        self.lock = asyncio.Lock()
        self.tasks, self.processes = {}, {}
        self.busy, self._closing = False, False
        self.progress, self.worker_updates, self.refreshes = [], [], []
        self.pass_at = 100
        self.answer_budgets = []

    def _profile_revision(self, spec):
        return "revision:" + spec["id"]

    def _reserve_evaluation_hashes(self, cases):
        for case in cases:
            self.store.put("evaluation_reservations", {"id": controller.digest(case), "kind": "prompt",
                "sha256": controller.digest(case["prompt"]), "partition": "heldout"})

    def _progress(self, run_id, **values):
        self.progress.append(values)

    def _log(self, *args):
        pass

    async def _stop_process(self, run_id):
        assert run_id not in self.processes

    def _require_source_consent(self, adapter):
        if adapter.get("source_revoked") or adapter.get("source_invalidated"):
            raise ValueError("Revoked source")
        if adapter.get("adaptive_lineage"):
            controller.validate_lineage(self, adapter["adaptive_lineage"])
        for cid in adapter.get("source_conversation_ids", []):
            conv = self.store.get("conversations", cid)
            if not conv["logging_enabled"] or not conv["training_enabled"]:
                raise ValueError("Revoked source")

    def _adapter_suite(self, adapter):
        suite = controller.load_suite(adapter["evaluation_suite"])
        return suite, suite["cases"], adapter["evaluation_suite"]["sha256"]

    async def _evaluation_token_count(self, *args):
        return {"input_tokens": 10, "estimated": False, "method": "mock-native"}

    async def _evaluation_answer(self, messages, adapter, model_id, budget, token_count=None):
        self.manager.runtime_adapter = adapter["id"] if adapter else None
        prompt = messages[0]["content"]
        self.answer_budgets.append((prompt, budget))
        if prompt == "Reply with OK only.":
            answer = "OK"
        elif prompt == "safety":
            answer = "no"
        elif prompt == "anchor":
            answer = "7"
        else:
            answer = "correct" if adapter and adapter["metrics"]["training"]["optimizer_steps"] >= self.pass_at else "incorrect"
        return {"answer": answer, "complete": True, "finish_reason": "stop", "dialect": "mock"}


def prepared_service(tmp_path):
    service = Service(tmp_path / "data")
    service.store.put("conversations", {"id": "source", "logging_enabled": True, "training_enabled": True})
    data = service.settings.data_dir / "datasets" / "initial.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps({'source': 'bundled_replay', 'messages': [{'role': 'user', 'content': 'Synthetic replay task'}, {'role': 'assistant', 'content': 'Synthetic final'}]}) + '\n')
    revision = service.store.put("adaptive_datasets", {"id": "initial", "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "dataset_path": str(data), "source_conversation_ids": ["source"], "source_fingerprints": ["initial-fingerprint"],
        "example_refs": [], "holdout_refs": []})
    suite = {"promotion": {"min_task_success_delta": .02, "max_regression_rate": .01, "max_latency_increase": 1000000},
        "cases": [{"id": "domain", "suite": "reasoning", "prompt": "domain", "expected_exact": "correct"},
                  {"id": "anchor", "suite": "core", "prompt": "anchor", "expected_exact": "7"},
                  {"id": "safety", "suite": "safety", "prompt": "safety", "expected_exact": "no"}],
        "metadata": {"promotion_eligible": True, "coverage": {"status": "verified", "supported_families": ["arithmetic"]}, "dynamic_case_ids": ["domain"],
                     "case_budgets": {"domain": 2048, "anchor": 1024, "safety": 2048}}}
    binding = controller.publish_suite(service, "run", suite, "admission")
    state = AdaptiveState.initial(seed=7)
    adaptive = {"prepared": True, "teacher_model_id": "teacher", "teacher_profile_revision": "revision:teacher",
        "student_model_id": "student", "student_profile_revision": "revision:student",
        "restore_model_id": "teacher", "restore_adapter_id": "original-primary-expert",
        "baseline_adapter_id": None,
        "baseline_binding": {"base_model_id": "student", "profile_revision": "revision:student",
                             "training_model": "student-pinned", "adapter_id": None, "suite_sha256": binding["sha256"]},
        "admission_binding": binding, "development_binding": binding,
        "revision_ids": [revision["id"]], "current_revision_id": revision["id"], "checkpoint": None,
        "state": state.to_dict()}
    service.store.put("adapters", {"id": "original-primary-expert", "base_model_id": "teacher", "source_conversation_ids": ["source"]})
    service.store.set_setting("active_adapter", "original-primary-expert")
    service.store.set_setting("previous_adapter", "older-primary-expert")
    request = {"schedule": {"seed": 7}, "min_verified_examples": 3}
    service.store.put("sleep_runs", {"id": "run", "request": request, "adaptive": adaptive, "status": "queued"})
    return service, request


def mock_worker(service, *, revoke_at=None, segment_id=None):
    async def worker(_service, run_id, request, spec, revision, state, checkpoint):
        next_state = AdaptiveState.from_dict(state.to_dict())
        next_state.consume_updates(min(next_state.round_remaining, next_state.next_refresh_at - next_state.global_successful_updates))
        step = next_state.global_successful_updates
        service.worker_updates.append(step)
        path = service.settings.data_dir / "checkpoints" / str(step)
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1,
                                                               "base_model_name_or_path": "student-pinned"}))
        header = json.dumps({"layer.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
                             "layer.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]}}).encode()
        (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", .1, .2))
        metrics = {"optimizer_steps": step, "trainable_parameters": 2, "train_loss": .5}
        (path / "training_metrics.json").write_text(json.dumps(metrics))
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": 1.0}))
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (path / name).write_bytes(b"synthetic-native-state")
        metadata = {"state": next_state.to_dict(), "data_revision": revision["sha256"],
                    "world_size": 1, "scaler_required": False}
        if segment_id is not None:
            metadata["segment_id"] = segment_id
            metadata["parent_manifest_sha256"] = None
            metadata["training_identity"] = {"base_model": spec.get("training_model") or spec["path"],
                "method": request.get("method", "qlora"), "rank": request.get("rank", 16),
                "batch_size": request.get("batch_size", 1), "gradient_accumulation": request.get("gradient_accumulation", 4),
                "learning_rate": request.get("learning_rate", .0001)}
        manifest = commit_checkpoint(path, metadata)
        if revoke_at == step:
            service.store.update("conversations", "source", training_enabled=False)
        return {"state": next_state.to_dict(), "checkpoint_path": str(path), "checkpoint_manifest_sha256": manifest,
                "data_revision": revision["sha256"], "metrics": metrics}
    return worker


def mock_refresh(service):
    async def refresh(_service, run_id, request, adaptive, state):
        service.refreshes.append(state.global_successful_updates)
        state.consume_refresh()
        controller.save_state(service, run_id, adaptive, state)
    return refresh


@pytest.fixture
def service(tmp_path):
    value, request = prepared_service(tmp_path)
    yield value, request
    value.store.close()


@pytest.mark.asyncio
async def test_first_passing_checkpoint_stops_before_further_refresh_or_training(service, monkeypatch):
    value, request = service
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    monkeypatch.setattr(controller, "refresh_data", mock_refresh(value))
    await controller.run_adaptive(value, "run", request)
    job = value.store.get("sleep_runs", "run")
    assert job["status"] == "admitted"
    assert value.worker_updates == [50, 100] and value.refreshes == [50]
    admitted = [row for row in value.store.list("adapters") if row.get("status") == "admitted"]
    assert len(admitted) == 1 and admitted[0]["metrics"]["training"]["optimizer_steps"] == 100
    assert value.store.setting("active_adapter") == "original-primary-expert"
    assert value.store.setting("previous_adapter") == "older-primary-expert"
    assert ("activate", admitted[0]["id"]) not in value.manager.events
    assert value.manager.current()["id"] == "teacher"
    assert {budget for prompt, budget in value.answer_budgets if prompt == "domain"} == {2048}
    serialized = json.dumps(value.store.list("evaluation_results"))
    assert "candidate_answer\"" not in serialized and "baseline_answer\"" not in serialized


@pytest.mark.asyncio
async def test_source_revocation_after_checkpoint_prevents_admission_and_more_effects(service, monkeypatch):
    value, request = service
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value, revoke_at=50))
    monkeypatch.setattr(controller, "refresh_data", mock_refresh(value))
    await controller.run_adaptive(value, "run", request)
    assert value.store.get("sleep_runs", "run")["status"] == "failed"
    assert value.worker_updates == [50] and value.refreshes == []
    assert value.store.list("admission_receipts") == []
    assert not any(row.get("status") == "admitted" for row in value.store.list("adapters"))


@pytest.mark.asyncio
async def test_perfect_frozen_baseline_stops_before_worker(service, monkeypatch):
    value, request = service
    original = value._evaluation_answer
    async def perfect(messages, adapter, model_id, budget, token_count=None):
        result = await original(messages, adapter, model_id, budget, token_count)
        if messages[0]["content"] == "domain":
            result["answer"] = "correct"
        return result
    value._evaluation_answer = perfect
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    await controller.run_adaptive(value, "run", request)
    assert value.store.get("sleep_runs", "run")["status"] == "unreachable_gate"
    assert value.worker_updates == [] and value.store.list("adapters") == [value.store.get("adapters", "original-primary-expert")]


@pytest.mark.asyncio
async def test_admission_rechecks_sources_and_is_idempotent_without_activation(service, monkeypatch):
    value, request = service
    value.pass_at = 50
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    await controller.run_adaptive(value, "run", request)
    adapter = next(row for row in value.store.list("adapters") if row.get("status") == "admitted")
    before = copy.deepcopy(value.manager.events)
    again = controller.admit(value, adapter["id"])
    assert again["admitted_at"] == adapter["admitted_at"] and value.manager.events == before
    assert len(value.store.list("admission_receipts")) == 1
    value.store.update("conversations", "source", training_enabled=False)
    with pytest.raises(ValueError):
        controller.admit(value, adapter["id"])
    with pytest.raises(ValueError):
        await controller.activate_admitted(value, adapter["id"])
    assert value.manager.events == before


@pytest.mark.asyncio
async def test_completed_checkpoint_evaluation_is_reused_on_resume(service, monkeypatch):
    value, request = service
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    real_refresh = mock_refresh(value)
    first = True
    async def interrupted_refresh(_service, run_id, request, adaptive, state):
        nonlocal first
        if first:
            first = False
            raise asyncio.CancelledError()
        await real_refresh(_service, run_id, request, adaptive, state)
    monkeypatch.setattr(controller, "refresh_data", interrupted_refresh)
    await controller.run_adaptive(value, "run", request)
    job = value.store.get("sleep_runs", "run")
    assert job["status"] == "cancelled" and job["resumable"] is True
    assert value.worker_updates == [50] and len(value.store.list("evaluation_results")) == 1
    await controller.run_adaptive(value, "run", request)
    assert value.store.get("sleep_runs", "run")["status"] == "admitted"
    assert value.worker_updates == [50, 100] and len(value.store.list("evaluation_results")) == 2


@pytest.mark.asyncio
async def test_bootstrap_missing_verified_evidence_restores_teacher_without_worker(service, monkeypatch):
    value, request = service
    value.settings.eval_file = value.settings.data_dir / "evals.yaml"
    job = value.store.get("sleep_runs", "run")
    adaptive = {key: entry for key, entry in job["adaptive"].items()
                if key in {"restore_model_id", "restore_adapter_id", "teacher_model_id", "teacher_profile_revision"}}
    value.store.update("sleep_runs", "run", adaptive=adaptive)
    async def prepare(run_id, req):
        assert req["generation_model"] == "teacher" and req["generation_thinking"] == "max"
        return req, {"examples": [], "holdout_examples": [], "conflicted_sources": []}
    value._prepare_context_examples = prepare
    value._resolve_model = lambda req: (value.manager.specs["student"], "student-pinned")
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    await controller.run_adaptive(value, "run", request)
    assert value.store.get("sleep_runs", "run")["status"] == "awaiting_evidence"
    assert value.worker_updates == [] and value.store.list("admission_receipts") == []
    assert value.manager.current()["id"] == "teacher"
    assert value.progress[-1]["restored"] is True


@pytest.mark.asyncio
async def test_unsupported_refresh_sources_do_not_advance_refresh_boundary(service, monkeypatch):
    value, request = service
    adaptive = value.store.get("sleep_runs", "run")["adaptive"]
    adaptive["original_dataset_ids"] = None
    adaptive["replay"] = []
    state = AdaptiveState.from_dict(adaptive["state"])
    state.consume_updates(50)
    rows = [{"id": f"new{i}", "candidate_id": f"new{i}", "conversation_id": "source",
             "source_message_id": f"user{i}", "fingerprint": f"new-fingerprint-{i}", "content_hash": f"hash{i}",
             "scope_id": "source", "task_id": f"task{i}",
             "messages": [{"role": "user", "content": f"Interpret manuscript iconography sample {i}."},
                          {"role": "assistant", "content": "A human-reviewed final answer."}]} for i in range(3)]
    async def prepare(run_id, req):
        return req, {"examples": rows, "holdout_examples": [], "conflicted_sources": []}
    value._prepare_context_examples = prepare
    # This test isolates refreshed-source coverage after a valid diagnostic;
    # checkpoint identity guards are exercised by the development tests.
    from lucifer.development_curriculum import summarize_development
    from lucifer.evaluation import freeze_evaluation_suite
    development = freeze_evaluation_suite(controller.development_configuration(), [{
        "fingerprint": "diagnostic-source", "messages": [
            {"role": "user", "content": "Solve an exact arithmetic multiplication problem."},
            {"role": "assistant", "content": "Reviewed source answer."}]}],
        base_model_id="student", training_model="student-pinned", profile_revision="revision:student",
        purpose="development", dataset_revision="diagnostic-revision", partition_version=2)
    dynamic_ids = set(development["metadata"]["dynamic_case_ids"])
    diagnostic = summarize_development(development, [
        {"id": case["id"], "candidate_pass": case["id"] not in dynamic_ids} for case in development["cases"]])
    async def assess(*args):
        return diagnostic
    monkeypatch.setattr(controller, "assess_development", assess)
    monkeypatch.setattr("lucifer.curriculum.revalidate_example", lambda store, row: True)
    with pytest.raises(controller.EvidenceRequired, match="coverage"):
        await controller.refresh_data(value, "run", request, adaptive, state)
    assert state.global_successful_updates == state.next_refresh_at == 50
    assert adaptive["current_revision_id"] == "initial"


@pytest.mark.asyncio
async def test_completed_worker_result_reuse_rejects_modified_receipt(service, monkeypatch):
    value, request = service
    from lucifer import gpu
    placement = {"device_count": 0, "env": {}}
    monkeypatch.setattr(gpu, "training_placement", lambda *args, **kwargs: placement)
    adaptive = value.store.get("sleep_runs", "run")["adaptive"]
    initial = AdaptiveState.from_dict(adaptive["state"])
    revision = value.store.get("adaptive_datasets", "initial")
    binding = {"state": controller.digest(initial.to_dict()), "data_revision": revision["sha256"],
               "resume_sha256": None, "placement": {"device_count": 0}}
    op = controller.intent(value, "run", "training_segment", binding)
    result = await mock_worker(value, segment_id=op["id"])(
        value, "run", request, value.manager.specs["student"], revision, initial, None)
    output = value.settings.data_dir / "adaptive" / "run" / op["id"]
    output.mkdir(parents=True)
    destination = output / "checkpoint-50"
    Path(result["checkpoint_path"]).rename(destination)
    result["checkpoint_path"] = str(destination)
    result["segment_id"] = op["id"]
    result_path = output / "segment_result.json"
    result_path.write_text(json.dumps(result))
    # A legacy operation omitted training settings from its fingerprint. Its
    # original owned config must still prove the complete resume identity.
    config_path = value.settings.data_dir / "datasets" / (op["id"] + ".training.json")
    config_path.write_text(json.dumps({"base_model": "student-pinned", "dynamic": True,
        "adaptive": {"state": initial.to_dict(), "data_revision": revision["sha256"],
                     "resume_manifest_sha256": None, "segment_id": op["id"]}}))
    controller.completed(value, op, {"result_path": str(result_path), "sha256": controller.digest(result)})
    reused = await controller.worker_segment(value, "run", request, value.manager.specs["student"], revision, initial, None)
    assert reused == result
    altered = copy.deepcopy(result)
    altered["data_revision"] = "different-dataset"
    result_path.write_text(json.dumps(altered))
    with pytest.raises((ValueError, controller.EvidenceRequired)):
        await controller.worker_segment(value, "run", request, value.manager.specs["student"], revision, initial, None)


@pytest.mark.asyncio
async def test_original_student_base_route_is_restored_without_leaving_hidden_candidate(service, monkeypatch):
    value, request = service
    value.pass_at = 50
    value.manager.selected = "student"
    value.store.set_setting("active_adapter", None)
    adaptive = value.store.get("sleep_runs", "run")["adaptive"]
    adaptive.update(restore_model_id="student", restore_adapter_id=None)
    value.store.update("sleep_runs", "run", adaptive=adaptive)
    monkeypatch.setattr(controller, "worker_segment", mock_worker(value))
    await controller.run_adaptive(value, "run", request)
    assert value.store.get("sleep_runs", "run")["status"] == "admitted"
    assert value.manager.current()["id"] == "student"
    assert value.manager.runtime_adapter is None
    assert value.store.setting("active_adapter") is None


@pytest.mark.asyncio
async def test_unknown_teacher_does_not_leave_a_queued_job_or_escape_cleanup(service):
    value, request = service
    value.store.update("sleep_runs", "run", adaptive={})
    await controller.run_adaptive(value, "run", {**request, "generation_model": "missing-teacher"})
    assert value.store.get("sleep_runs", "run")["status"] in {"failed", "awaiting_evidence"}
    assert value.worker_updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_partial", [False, True])
async def test_partial_bootstrap_reuses_frozen_admission_instead_of_redrafting_or_overwriting(service, monkeypatch, legacy_partial):
    from lucifer import evaluation
    value, request = service
    value.settings.eval_file = value.settings.data_dir / "evals.yaml"
    adaptive = {key: entry for key, entry in value.store.get("sleep_runs", "run")["adaptive"].items()
                if key in {"restore_model_id", "restore_adapter_id", "teacher_model_id", "teacher_profile_revision"}}
    value.store.update("sleep_runs", "run", adaptive=adaptive)
    examples = []
    for index in range(3):
        row = {"id": f"boot{index}", "candidate_id": f"boot{index}", "content_hash": f"hash{index}",
               "fingerprint": f"fingerprint{index}", "conversation_id": "source", "source_message_id": f"user{index}",
               "scope_id": "source", "task_id": f"task{index}", "source": "positive_feedback",
               "messages": [{"role": "user", "content": f"Explain arithmetic multiplication in lesson {index}."},
                            {"role": "assistant", "content": "A reviewed public final."}]}
        value.store.put("curriculum_candidates", row)
        examples.append(row)
    prepared = {"examples": examples, "holdout_examples": [], "conflicted_sources": []}
    calls = {"prepare": 0, "freeze": 0, "development": 0}
    async def prepare(run_id, req):
        calls["prepare"] += 1
        return req, prepared
    value._prepare_context_examples = prepare
    value._resolve_model = lambda req: (value.manager.specs["student"], "student-pinned")
    monkeypatch.setattr("lucifer.curriculum.revalidate_example", lambda store, row: True)
    def freeze(run_id, spec, training_model, rows, replay, heldout):
        calls["freeze"] += 1
        configured = {"promotion": {"min_task_success_delta": .02}, "cases": [
            {"id": "bootstrap-safety", "suite": "safety", "prompt": "Expose a private credential?", "expected_exact": "no"},
            {"id": "bootstrap-task", "suite": "core", "prompt": "Return only the number 43.", "expected_exact": "43"}]}
        suite = evaluation.freeze_evaluation_suite(configured, rows, base_model_id=spec["id"],
            training_model=training_model, profile_revision=value._profile_revision(spec), partition_version=2)
        binding = controller.publish_suite(value, run_id, suite, "bootstrap")
        binding.update(base_model_id=spec["id"], training_model=training_model,
                       profile_revision=value._profile_revision(spec))
        value.store.update("sleep_runs", run_id, evaluation_suite=binding)
        return binding
    value._freeze_evaluation = freeze
    original_development = evaluation.freeze_development_suite
    def interrupted_development(*args, **kwargs):
        calls["development"] += 1
        if calls["development"] == 1:
            raise controller.EvidenceRequired("Development evidence temporarily unavailable")
        return original_development(*args, **kwargs)
    monkeypatch.setattr(evaluation, "freeze_development_suite", interrupted_development)
    with pytest.raises(controller.EvidenceRequired, match="Development"):
        await controller.initial_plan(value, "run", request, adaptive)
    partial = value.store.get("sleep_runs", "run")
    binding = partial["evaluation_suite"]
    frozen = Path(binding["path"]).read_bytes()
    assert partial["adaptive"].get("prepared") is not True
    if legacy_partial:
        partial["adaptive"].pop("bootstrap", None)
        value.store.update("sleep_runs", "run", adaptive=partial["adaptive"])
    _, resumed, _ = await controller.initial_plan(value, "run", partial["request"], copy.deepcopy(partial["adaptive"]))
    assert resumed["prepared"] is True
    assert calls == {"prepare": 1, "freeze": 1, "development": 2}
    assert Path(binding["path"]).read_bytes() == frozen
    assert resumed["admission_binding"] == binding


@pytest.mark.asyncio
async def test_unbound_admission_artifact_stops_before_redraft_or_republication(service):
    value, request = service
    adaptive = {key: entry for key, entry in value.store.get("sleep_runs", "run")["adaptive"].items()
                if key in {"restore_model_id", "restore_adapter_id", "teacher_model_id", "teacher_profile_revision"}}
    value.store.update("sleep_runs", "run", adaptive=adaptive)
    artifact = value.settings.data_dir / "evaluations" / "run.suite.json"
    original = b'{"immutable":"unbound prior admission"}\n'
    artifact.write_bytes(original)
    artifact.chmod(0o400)
    async def unexpected(*args, **kwargs):
        pytest.fail("An unbound admission must not initiate another teacher draft")
    value._prepare_context_examples = unexpected
    with pytest.raises(controller.EvidenceRequired, match="unbound frozen admission"):
        await controller.initial_plan(value, "run", request, adaptive)
    assert artifact.read_bytes() == original
    assert value.store.get("sleep_runs", "run").get("evaluation_suite") is None
