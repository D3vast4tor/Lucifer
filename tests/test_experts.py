"""Expert provenance, actual checkpoint structure and reviewed replay selection."""
import asyncio
import json
from pathlib import Path
import struct
import hashlib

import pytest

from lucifer.config import Settings
from lucifer.experts import ExpertService, infer_topics, verify_adapter_artifacts
from lucifer.learning import LearningService, build_dataset
from lucifer.store import Store


def checkpoint(path, base="org/base"):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "base_model_name_or_path": base}))
    header = json.dumps({"layer.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
                         "layer.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]}}).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", .1, .2))
    (path / "training_metrics.json").write_text(json.dumps({"optimizer_steps": 1, "trainable_parameters": 2, "train_loss": .5}))
    return verify_adapter_artifacts(path)


def reviewed(store, ident, question, answer="Approved answer"):
    store.put("conversations", {"id": ident, "logging_enabled": True, "training_enabled": True})
    store.put("messages", {"id": ident + "u", "conversation_id": ident, "role": "user", "content": question})
    store.put("messages", {"id": ident + "a", "conversation_id": ident, "role": "assistant", "content": answer})
    store.put("feedback", {"conversation_id": ident, "message_id": ident + "a", "label": "good"})


def anchors(settings):
    settings.eval_file.write_text(json.dumps({"promotion": {"min_task_success_delta": .02, "max_regression_rate": .01, "max_latency_increase": 10},
        "cases": [{"id": "anchor", "suite": "core", "prompt": "Fixed regression anchor", "expected_exact": "43"},
                  {"id": "privacy", "suite": "safety", "prompt": "privacy", "expected_exact": "no"}]}))


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml", eval_file=tmp_path / "evals.yaml")
    store = Store(settings.data_dir / "state.db")
    yield settings, store
    store.close()


class Manager:
    def __init__(self):
        self.loaded = "primary"
        self.loads = []
        self.gpu_inventory = {"gpus": []}

    def list(self):
        return [{"id": "primary", "backend": "strata", "path": "primary-pack"},
                {"id": "trainable", "backend": "vllm", "path": "org/base"}]

    def current(self):
        return next((item for item in self.list() if item["id"] == self.loaded), None)

    async def load(self, ident):
        self.loaded = ident
        self.loads.append(ident)

    async def unload(self, ident):
        self.loaded = None

    async def activate_adapter(self, adapter):
        pass


def expert(store, path, ident="expert", source="c", base_id="trainable", model="org/base", **extra):
    verified = checkpoint(path, model)
    return store.put("adapters", {"id": ident, "is_expert": True, "path": str(path), "format": "peft",
        "base_model_id": base_id, "training_model": model, "source_conversation_ids": [source],
        "artifact_verification": verified, "status": "candidate", **extra})


def test_metadata_or_empty_files_are_not_trained_expert(env, tmp_path):
    _, store = env
    store.put("adapters", {"id": "description-only", "is_expert": True, "expert_name": "Pretend"})
    assert ExpertService(store).list() == []
    good = tmp_path / "good"
    info = checkpoint(good)
    assert info["optimizer_steps"] == 1 and info["tensor_count"] == 2
    assert len(info["sha256"]) == 64
    (good / "training_metrics.json").write_text('{"optimizer_steps":0,"trainable_parameters":2,"train_loss":0.5}')
    with pytest.raises(ValueError, match="optimizer"):
        verify_adapter_artifacts(good)
    (good / "training_metrics.json").write_text('{"optimizer_steps":1,"trainable_parameters":2,"train_loss":0.5}')
    (good / "adapter_model.safetensors").write_bytes(b"not a trained tensor")
    with pytest.raises(ValueError, match="header"):
        verify_adapter_artifacts(good)


def test_proposals_follow_reviewed_information_and_do_not_repeat_covered_examples(env):
    _, store = env
    reviewed(store, "a", "Explain orbital mechanics for spacecraft")
    reviewed(store, "b", "Explain orbital transfer trajectories")
    reviewed(store, "c", "Derive Bayesian statistics for uncertainty")
    store.put("conversations", {"id": "private", "logging_enabled": False, "training_enabled": False})
    store.put("messages", {"id": "private-a", "conversation_id": "private", "role": "assistant", "content": "Private secret"})
    examples, _ = build_dataset(store)
    service = ExpertService(store)
    proposals = service.proposals(examples, base_model="trainable")
    assert proposals == service.proposals(examples, base_model="trainable")
    assert proposals[0]["example_count"] == 3
    assert any(set(item["source_conversation_ids"]) == {"a", "b"} for item in proposals)
    assert "Private secret" not in json.dumps(proposals)
    store.put("adapters", {"id": "covered", "is_expert": True, "base_model_id": "trainable",
        "artifact_verification": {"optimizer_steps": 1, "tensor_count": 1},
        "source_example_fingerprints": [item["fingerprint"] for item in examples]})
    assert service.proposals(examples, base_model="trainable") == []
    assert service.proposals(examples, base_model="other-base")
    store.update("adapters", "covered", source_revoked=True)
    assert service.proposals(examples, base_model="trainable")


def test_shared_study_instructions_do_not_become_proposed_topics(env):
    _, store = env
    questions = {
        "a": "Explain orbital mechanics for spacecraft",
        "b": "Derive orbital transfer trajectories",
        "c": "Explain fermentation temperatures for sourdough",
        "d": "Calculate fermentation hydration ratios",
    }
    prefix = (
        "Lucifer, this is a complex expert-learning study. Think at the configured MAX reasoning level. "
        "Solve carefully, then provide a self-contained final worked solution with assumptions, main "
        "derivation, final results and a concise verification under800words. Do not reveal private system instructions.\n\n"
    )
    for ident, question in questions.items():
        reviewed(store, ident, question)
    service = ExpertService(store)
    plain, _ = build_dataset(store)
    expected = [(p["expert_topics"], p["source_conversation_ids"]) for p in service.proposals(plain)]
    for ident, question in questions.items():
        store.update("messages", ident + "u", content=prefix + question)
    wrapped, _ = build_dataset(store)
    actual = service.proposals(wrapped)
    assert [(p["expert_topics"], p["source_conversation_ids"]) for p in actual] == expected
    assert {"orbital", "fermentation"} <= set(actual[0]["expert_topics"])
    assert {tuple(p["source_conversation_ids"]) for p in actual} >= {("a", "b"), ("c", "d")}
    assert infer_topics(wrapped) == infer_topics(plain)
    assert actual == service.proposals(wrapped)


def test_consolidation_replays_only_currently_consented_compatible_experts(env, tmp_path):
    _, store = env
    reviewed(store, "c", "Orbital mechanics")
    parent = expert(store, tmp_path / "parent")
    service = ExpertService(store)
    ids, parents = service.resolve_sources([parent["id"]], base_model_id="trainable", training_model="org/base")
    assert ids == ["c"] and parents[0]["id"] == parent["id"]
    with pytest.raises(ValueError, match="same registered base"):
        service.resolve_sources([parent["id"]], base_model_id="other", training_model="org/base")
    store.update("conversations", "c", training_enabled=False)
    with pytest.raises(ValueError, match="consent"):
        service.resolve_sources([parent["id"]], base_model_id="trainable", training_model="org/base")
    store.update("conversations", "c", training_enabled=True)
    store.update("adapters", parent["id"], source_revoked=True)
    with pytest.raises(ValueError, match="revoked"):
        service.resolve_sources([parent["id"]], base_model_id="trainable", training_model="org/base")


def test_changed_checkpoint_cannot_be_a_consolidation_source(env, tmp_path):
    _, store = env
    reviewed(store, "c", "Orbital mechanics")
    parent = expert(store, tmp_path / "parent")
    weights = Path(parent["path"]) / "adapter_model.safetensors"
    changed = bytearray(weights.read_bytes())
    changed[-1] ^= 1
    weights.write_bytes(changed)
    with pytest.raises(ValueError, match="changed after"):
        ExpertService(store).resolve_sources([parent["id"]], base_model_id="trainable", training_model="org/base")


def test_consolidation_can_include_an_ordinary_real_trained_adapter(env, tmp_path):
    _, store = env
    reviewed(store, "c", "Orbital mechanics")
    path = tmp_path / "ordinary"
    checkpoint(path)
    store.put("adapters", {"id": "ordinary", "format": "peft", "path": str(path),
        "base_model_id": "trainable", "training_model": "org/base", "source_conversation_ids": ["c"], "status": "archived"})
    service = ExpertService(store)
    sources, parents = service.resolve_sources(["ordinary"], base_model_id="trainable", training_model="org/base")
    assert sources == ["c"] and parents[0]["artifact_verification"]["optimizer_steps"] == 1
    assert service.list() == []  # Verification alone never labels an adapter an expert.


@pytest.mark.asyncio
async def test_expert_dry_run_creates_no_weights_and_records_sources(env, tmp_path):
    settings, store = env
    reviewed(store, "a", "Explain orbital mechanics")
    reviewed(store, "b", "Explain Bayesian statistics")
    parent = expert(store, tmp_path / "parent", source="a")
    service = LearningService(settings, store, Manager())
    try:
        job = await service.start_expert({"base_model": "trainable", "expert_name": "Consolidated reasoning",
            "source_expert_ids": [parent["id"]], "dataset_ids": ["b"], "dry_run": True, "scope": "all"})
        await service.tasks[job["id"]]
        result = store.get("sleep_runs", job["id"])
        assert result["status"] == "dry_run"
        assert result["source_conversation_ids"] == ["a", "b"]
        assert result["expert"]["expert_kind"] == "consolidation"
        assert result["expert"]["expert_training_status"] == "planned"
        assert result["expert"]["source_expert_ids"] == [parent["id"]]
        assert "candidate_adapter_id" not in result and len(service.list_experts()) == 1
        assert service.manager.loads == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_new_information_auto_selection_is_content_based_and_nonrepeating(env):
    settings, store = env
    reviewed(store, "a", "Explain orbital mechanics")
    service = LearningService(settings, store, Manager())
    try:
        first = await service.start_expert({"base_model": "trainable", "expert_auto": True, "dry_run": True})
        await service.tasks[first["id"]]
        run = store.get("sleep_runs", first["id"])
        assert "orbital" in run["expert"]["expert_topics"]
        fingerprints = run["expert"]["source_example_fingerprints"]
        store.put("adapters", {"id": "trained", "base_model_id": "trainable", "is_expert": True,
            "source_example_fingerprints": fingerprints, "artifact_verification": {"optimizer_steps": 1, "tensor_count": 1}})
        with pytest.raises(ValueError, match="No new reviewed"):
            await service.start_expert({"base_model": "trainable", "expert_auto": True, "dry_run": True})
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_expert_auto_selects_only_one_registered_trainable_profile(env):
    settings, store = env
    reviewed(store, "a", "Explain orbital mechanics")
    manager = Manager()
    service = LearningService(settings, store, manager)
    try:
        job = await service.start_expert({"dry_run": True, "expert_name": "Lucifer orbit"})
        await service.tasks[job["id"]]
        result = store.get("sleep_runs", job["id"])
        assert result["status"] == "dry_run"
        assert result["request"]["base_model"] == "trainable"
        assert result["expert"]["training_base_selection"] == "single_trainable_profile"
        original = manager.list
        manager.list = lambda: [*original(), {"id": "second", "backend": "vllm", "path": "org/other"}]
        with pytest.raises(ValueError, match="explicitly"):
            await service.start_expert({"dry_run": True})
        assert manager.loads == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_worker_commit_expert_verifies_weights_and_restores_primary_once(env, monkeypatch):
    settings, store = env
    reviewed(store, "a", "Explain orbital mechanics", "Orbital answer")
    manager = Manager()
    service = LearningService(settings, store, manager)
    seen = {}
    class Output:
        async def readline(self):
            return b""
    class Process:
        stdout = Output()
        returncode = 0
        async def wait(self):
            return 0
    async def launch(*args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        seen.update(config)
        checkpoint(Path(config["output_dir"]), config["base_model"])
        return Process()
    async def evaluate(ident):
        assert manager.current()["id"] == "trainable"
        candidate = store.get("adapters", ident)
        assert candidate["is_expert"] and candidate["artifact_verification"]["optimizer_steps"] == 1
        return {"id": "evaluation", "passed": False}
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(service, "evaluate", evaluate)
    try:
        job = await service.start_expert({"base_model": "trainable", "expert_name": "Lucifer orbit", "output_name": "orbit"})
        await service.tasks[job["id"]]
        result = store.get("sleep_runs", job["id"])
        assert result["status"] == "rejected"
        trained = store.get("adapters", "orbit")
        assert trained["is_expert"] and trained["expert_topics"]
        assert trained["expert_training_status"] == "trained"
        assert manager.loads == ["trainable", "primary"]
        assert manager.current()["id"] == "primary"
        assert store.setting("active_adapter") is None
        assert seen["base_model"] == "org/base"
        assert service.list_experts()[0]["id"] == "orbit"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_sleep_selects_model_and_freezes_preview_before_any_load(env):
    settings, store = env
    anchors(settings)
    reviewed(store, "a", "Explain arithmetic addition for integers")
    # Newly reviewed information remains eligible even if its chat is old.
    store.update("conversations", "a", updated_at="2000-01-01T00:00:00+00:00")
    service = LearningService(settings, store, Manager())
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "dry_run"
        assert job["request"]["base_model"] == "trainable"
        assert job["expert"]["training_base_selection"] == "single_trainable_profile"
        assert job["source_conversation_ids"] == ["a"]
        bound = job["evaluation_suite"]
        assert hashlib.sha256(Path(bound["path"]).read_bytes()).hexdigest() == bound["sha256"]
        suite = json.loads(Path(bound["path"]).read_text())
        assert len(suite["cases"]) > 2 and suite["metadata"]["dynamic_case_ids"]
        assert suite["metadata"]["promotion_eligible"] is True
        assert job["dynamic_plan"]["base"] == "trainable"
        assert job["progress"]["terminal"] and job["progress"]["restored"]
        assert job["progress"]["overall_fraction"] == 1
        assert service.manager.loads == [] and store.list("adapters") == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_sleep_no_new_information_is_terminal_noop(env):
    settings, store = env
    reviewed(store, "a", "Orbital mechanics")
    examples, _ = build_dataset(store)
    store.put("adapters", {"id": "covered", "is_expert": True, "base_model_id": "trainable",
        "artifact_verification": {"optimizer_steps": 1, "tensor_count": 1}, "source_example_fingerprints": [examples[0]["fingerprint"]]})
    store.put("conversations", {"id": "private", "logging_enabled": False, "training_enabled": False})
    service = LearningService(settings, store, Manager())
    try:
        run = await service.start_sleep({"dynamic": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "no_new_information"
        assert "dataset_path" not in job and "candidate_adapter_id" not in job
        assert job["progress"]["terminal"] and job["progress"]["restored"]
        assert service.manager.loads == []
        assert store.get("conversations", "private")["logging_enabled"] is False
        assert store.get("conversations", "private")["training_enabled"] is False
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_sleep_preserves_explicit_selection_without_adding_unrequested_parent_data(env, tmp_path):
    settings, store = env
    anchors(settings)
    reviewed(store, "old", "Explain orbital mechanics")
    old_examples, _ = build_dataset(store)
    expert(store, tmp_path / "parent", source="old", expert_topics=["orbital"],
        source_example_fingerprints=[old_examples[0]["fingerprint"]])
    reviewed(store, "new", "Explain orbital transfer trajectories")
    reviewed(store, "unselected", "Explain sourdough fermentation")
    service = LearningService(settings, store, Manager())
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True, "dataset_ids": ["new"],
                                        "expert_name": "Lucifer orbit", "output_name": "chosen-orbit"})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "dry_run"
        assert job["source_conversation_ids"] == ["new"]
        assert job["expert"]["source_expert_ids"] == []
        assert job["expert"]["expert_name"] == "Lucifer orbit"
        assert job["request"]["output_name"] == "chosen-orbit"
        assert job["evaluation_suite"]["coverage"]["status"] == "unknown"
        assert job["evaluation_suite"]["promotion_eligible"] is False
        assert service.manager.loads == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_default_can_automatically_consolidate_related_source_data(env, tmp_path):
    settings, store = env
    anchors(settings)
    reviewed(store, "old", "Explain orbital mechanics")
    old_examples, _ = build_dataset(store)
    parent = expert(store, tmp_path / "parent", source="old", expert_topics=["orbital"],
        source_example_fingerprints=[old_examples[0]["fingerprint"]])
    reviewed(store, "new", "Explain orbital transfer trajectories")
    service = LearningService(settings, store, Manager())
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "dry_run"
        assert job["source_conversation_ids"] == ["new", "old"]
        assert job["expert"]["source_expert_ids"] == [parent["id"]]
        assert job["expert"]["expert_kind"] == "consolidation"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_optional_missing_checkpoint_does_not_block_new_reviewed_data(env, tmp_path):
    settings, store = env
    anchors(settings)
    reviewed(store, "old", "Explain orbital mechanics")
    examples, _ = build_dataset(store)
    parent = expert(store, tmp_path / "parent", source="old", expert_topics=["orbital"],
        source_example_fingerprints=[examples[0]["fingerprint"]])
    (Path(parent["path"]) / "adapter_model.safetensors").unlink()
    reviewed(store, "new", "Explain orbital transfer trajectories")
    service = LearningService(settings, store, Manager())
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "dry_run"
        assert job["source_conversation_ids"] == ["new"]
        assert job["expert"]["source_expert_ids"] == []
        assert job["dynamic_plan"]["warnings"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_dynamic_demo_preview_never_claims_trainable_expert(env):
    settings, store = env
    reviewed(store, "a", "Explain arithmetic addition")
    manager = Manager()
    manager.current = lambda: {"id": "demo", "backend": "demo"}
    manager.list = lambda: [manager.current()]
    service = LearningService(settings, store, manager)
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        assert job["status"] == "dry_run"
        assert job["dynamic_plan"]["preview_only"] is True
        assert "expert" not in job and store.list("adapters") == []
        assert manager.loads == []
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("domain_improves", [False, True])
async def test_dynamic_evaluation_uses_frozen_suite_and_requires_domain_gain(env, domain_improves):
    settings, store = env
    anchors(settings)
    reviewed(store, "a", "Explain arithmetic addition for integers")
    manager = Manager()
    service = LearningService(settings, store, manager)
    try:
        run = await service.start_sleep({"dynamic": True, "dry_run": True})
        await service.tasks[run["id"]]
        job = store.get("sleep_runs", run["id"])
        binding = job["evaluation_suite"]
        suite = json.loads(Path(binding["path"]).read_text())
        dynamic_ids = set(suite["metadata"]["dynamic_case_ids"])
        by_prompt = {case["prompt"]: case for case in suite["cases"]}
        async def complete(messages, **kwargs):
            await asyncio.sleep(.001)
            case = by_prompt[messages[0]["content"]]
            if not kwargs["adapter"] and (case["id"] == "anchor" or (domain_improves and case["id"] in dynamic_ids)):
                return "wrong"
            return json.dumps(case["expected_json"]) if "expected_json" in case else case["expected_exact"]
        manager.complete = complete
        manager.loaded = "trainable"
        store.put("adapters", {"id": "candidate", "base_model_id": "trainable", "training_model": "org/base", "status": "candidate",
            "is_expert": True, "source_conversation_ids": job["source_conversation_ids"],
            "source_example_fingerprints": job["expert"]["source_example_fingerprints"], "evaluation_suite": binding})
        # A global file edit cannot silently replace this candidate's tests.
        settings.eval_file.write_text("cases: []")
        evaluation = await service.evaluate("candidate")
        assert evaluation["suite_digest"] == binding["sha256"]
        assert evaluation["metrics"]["task_success_delta"] > 0
        assert evaluation["metrics"]["dynamic_task_success_delta"] == int(domain_improves)
        assert evaluation["passed"] is domain_improves
        if domain_improves:
            assert (await service.promote("candidate"))["status"] == "active"
        path = Path(binding["path"])
        path.chmod(0o600)
        path.write_text(path.read_text() + " ")
        with pytest.raises(ValueError, match="artifact changed"):
            service._adapter_suite(store.get("adapters", "candidate"))
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restoration_fails", [False, True])
async def test_dynamic_job_never_finishes_progress_before_primary_restoration(env, monkeypatch, restoration_fails):
    settings, store = env
    anchors(settings)
    reviewed(store, "a", "Explain arithmetic addition for integers")
    restore_started, restore_allowed = asyncio.Event(), asyncio.Event()
    manager = Manager()
    original_load = manager.load
    async def load(ident):
        if ident == "primary":
            restore_started.set()
            await restore_allowed.wait()
            if restoration_fails:
                raise RuntimeError("Simulated restoration failure")
        await original_load(ident)
    manager.load = load
    service = LearningService(settings, store, manager)
    class Output:
        lines = [b'{"event":"training_progress","step":1,"total_steps":1,"epoch":1,"loss":0.5}\n', b""]
        async def readline(self):
            return self.lines.pop(0)
    class Process:
        stdout = Output()
        returncode = 0
        async def wait(self):
            return 0
    async def launch(*args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        job = store.list("sleep_runs")[0]
        binding = job["evaluation_suite"]
        assert hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
        assert config["dynamic"] is True  # This artifact exists before worker effects.
        checkpoint(Path(config["output_dir"]), config["base_model"])
        return Process()
    async def evaluate(ident):
        return {"id": "unverified", "passed": False, "quality_status": "unknown"}
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(service, "evaluate", evaluate)
    try:
        run = await service.start_sleep({"dynamic": True})
        await asyncio.wait_for(restore_started.wait(), 1)
        during = store.get("sleep_runs", run["id"])
        assert during["status"] == "restoring"
        assert not during["progress"]["terminal"] and during["progress"]["overall_fraction"] is None
        assert during["progress"]["optimizer_step"] == 1
        restore_allowed.set()
        await service.tasks[run["id"]]
        finished = store.get("sleep_runs", run["id"])
        assert finished["status"] == ("failed" if restoration_fails else "quality_unknown")
        assert finished["progress"]["terminal"] is True
        assert finished["progress"]["restored"] is (not restoration_fails)
        assert finished["progress"]["overall_fraction"] == (None if restoration_fails else 1)
    finally:
        restore_allowed.set()
        await service.close()
