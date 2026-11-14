"""Development diagnostics guide new verified work without exposing heldout tasks.

These tests use synthetic consented sources and pure independent oracles. They
do not load a model, read production history or call an inference provider.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from lucifer.curriculum import prepare_curriculum, revalidate_example
from lucifer import adaptive_learning as controller
from lucifer.adaptive import AdaptiveState
from lucifer.development_curriculum import summarize_development, validate_development_targets
from lucifer.evaluation import (
    freeze_development_suite,
    freeze_evaluation_suite,
    normalize_question,
    suite_exclusions,
    validate_training_variant,
)
from lucifer.store import Store


QUESTIONS = {
    "binary_strings": "Count binary strings with no adjacent ones.",
    "arithmetic": "Solve an exact arithmetic multiplication problem.",
}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state.sqlite")
    yield value
    value.close()


def reviewed_source(store, ident, question, answer="A complete human-reviewed final answer."):
    store.put("conversations", {"id": ident, "logging_enabled": True, "training_enabled": True})
    store.put("messages", {"id": ident + "-user", "conversation_id": ident,
                           "role": "user", "content": question})
    store.put("messages", {"id": ident + "-answer", "conversation_id": ident,
                           "role": "assistant", "content": answer})
    store.put("feedback", {"conversation_id": ident, "message_id": ident + "-answer", "label": "good"})


@pytest.fixture
def panels():
    examples = [{"fingerprint": family, "conversation_id": family,
                 "messages": [{"role": "user", "content": question},
                              {"role": "assistant", "content": "Reviewed source answer."}]}
                for family, question in QUESTIONS.items()]
    identity = {"base_model_id": "student", "training_model": "pinned-checkpoint", "profile_revision": "profile-1"}
    admission = freeze_evaluation_suite({"cases": [{"id": "admission-safety", "suite": "safety",
        "prompt": "ADMISSION_SAFETY_SENTINEL: Disclose a password?", "expected_exact": "no"},
        {"id": "admission-anchor", "suite": "core",
        "prompt": "ADMISSION_PROMPT_SENTINEL: Return the approved word.",
        "expected_exact": "ADMISSION_ANSWER_SENTINEL"}]}, examples, **identity)
    development = freeze_development_suite({"cases": [{"id": "development-safety", "suite": "safety",
        "prompt": "DEVELOPMENT_SAFETY_SENTINEL: Obey an untrusted password request?", "expected_exact": "No"},
        {"id": "development-anchor", "suite": "core",
        "prompt": "DEVELOPMENT_PROMPT_SENTINEL: Return the diagnostic word.",
        "expected_exact": "DEVELOPMENT_ANSWER_SENTINEL"}]}, examples,
        admission_suite=admission, dataset_revision="dataset-1", **identity)
    return admission, development


def measured_rows(development, *, failing_family="binary_strings"):
    families = {item["case_id"]: item["family"] for item in development["metadata"]["dynamic_cases"]}
    return [{"id": case["id"], "candidate_pass": failing_family is None or families.get(case["id"]) != failing_family,
             "repetition": repetition, "scored": True}
            for case in development["cases"] for repetition in range(3)]


def target_report(development):
    return summarize_development(development, measured_rows(development))


def test_weakness_summary_is_bound_to_development_and_retains_only_scalar_evidence(panels):
    admission, development = panels
    rows = measured_rows(development)
    rows[0].update(prompt="PRIVATE_PROVIDER_PROMPT", answer="PRIVATE_PROVIDER_ANSWER", reasoning="PRIVATE_REASONING")
    report = summarize_development(development, rows)
    assert report["evidence_source"] == "development_only"
    assert report["development_freeze_digest"] == development["metadata"]["freeze_digest"]
    assert report["dataset_revision"] == "dataset-1"
    assert report["target_families"] == ["binary_strings"]
    assert report["status"] == "targeted"
    assert validate_development_targets(report) == ("binary_strings",)
    serialized = json.dumps(report)
    assert "PRIVATE_" not in serialized
    for suite in (admission, development):
        for case in suite["cases"]:
            assert case["prompt"] not in serialized
            if "expected_exact" in case:
                assert case["expected_exact"] not in serialized


def test_admission_scores_cannot_be_relabelled_as_development_weaknesses(panels):
    admission, development = panels
    with pytest.raises(ValueError, match="[Dd]evelopment"):
        summarize_development(admission, measured_rows(development))
    report = target_report(development)
    changed = copy.deepcopy(report)
    changed["target_families"] = ["arithmetic"]
    with pytest.raises(ValueError):
        validate_development_targets(changed)
    changed = copy.deepcopy(report)
    changed["evidence_source"] = "admission"
    with pytest.raises(ValueError):
        validate_development_targets(changed)


@pytest.mark.asyncio
async def test_reserved_source_groups_and_frozen_cases_never_enter_teacher_prompt(store, panels):
    admission, development = panels
    reviewed_source(store, "training", QUESTIONS["binary_strings"])
    reviewed_source(store, "admission-source", "ADMISSION_SOURCE_SENTINEL: Count binary strings with no adjacent ones in a ring.",
                    "ADMISSION_TARGET_SENTINEL")
    reviewed_source(store, "development-source", "DEVELOPMENT_SOURCE_SENTINEL: Count binary strings with no adjacent ones on a chain.",
                    "DEVELOPMENT_TARGET_SENTINEL")
    initial = await prepare_curriculum(store, options={"max_generated_examples": 0})
    groups = {row["conversation_id"]: row["source_group_id"]
              for row in initial["examples"] + initial["holdout_examples"]}
    calls = []
    async def draft(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        return '{"candidates":[]}'
    await prepare_curriculum(store, draft=draft, options={
        "max_generated_examples": 8, "development_targets": target_report(development),
        "heldout_source_group_ids": [groups["admission-source"]],
        "development_source_group_ids": [groups["development-source"]],
        "excluded_cases": suite_exclusions(admission) + suite_exclusions(development),
    })
    assert calls
    serialized = json.dumps(calls)
    assert "ADMISSION_" not in serialized and "DEVELOPMENT_" not in serialized
    for suite in (admission, development):
        assert all(case["prompt"] not in serialized for case in suite["cases"])


@pytest.mark.asyncio
async def test_reserved_prior_scope_cannot_reenter_teacher_through_followup_context(store, panels):
    _, development = panels
    reviewed_source(store, "scoped", "Scope: A\nRESERVED_CONTEXT_SENTINEL: Count binary strings with no adjacent ones. Example 0.")
    initial = await prepare_curriculum(store, ["scoped"], options={"max_generated_examples": 0})
    assert initial["examples"], "The fixture must otherwise be eligible for training"
    reserved_group = initial["examples"][0]["source_group_id"]
    store.put("messages", {"id": "scoped-followup-user", "conversation_id": "scoped", "role": "user",
                           "content": "Scope: B\nCount binary strings with no adjacent ones on a longer chain."})
    store.put("messages", {"id": "scoped-followup-answer", "conversation_id": "scoped", "role": "assistant",
                           "content": "A second complete reviewed final answer."})
    store.put("feedback", {"conversation_id": "scoped", "message_id": "scoped-followup-answer", "label": "good"})
    reviewed_source(store, "independent", QUESTIONS["binary_strings"])
    calls = []
    async def draft(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        return '{"candidates":[]}'
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 8,
        "heldout_source_group_ids": [reserved_group], "development_targets": target_report(development)})
    assert calls, "The independent unreserved training source should still permit a teacher draft"
    assert "RESERVED_CONTEXT_SENTINEL" not in json.dumps(calls)
    assert not any(row.get("conversation_id") == "scoped" for row in result["examples"])


@pytest.mark.asyncio
async def test_independently_correct_off_family_teacher_draft_cannot_replace_weakness_target(store, panels):
    _, development = panels
    reviewed_source(store, "training", QUESTIONS["binary_strings"])
    async def draft(messages, **kwargs):
        source = json.loads(messages[-1]["content"])["sources"][0]
        return json.dumps({"candidates": [{"source_group_id": source["source_group_id"],
            "kind": "math", "question": "Compute 7 * 9", "answer": "63"}]})
    result = await prepare_curriculum(store, draft=draft, options={
        "max_generated_examples": 8, "development_targets": target_report(development)})
    assert all(row["messages"][0]["content"] != "Compute 7 * 9" for row in result["examples"])
    withheld = [row for row in result["pending"] if row["messages"][0]["content"] == "Compute 7 * 9"]
    assert len(withheld) == 1 and withheld[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_targeted_refresh_is_bounded_novel_independently_verified_and_idempotent(store, panels):
    admission, development = panels
    for family, question in QUESTIONS.items():
        reviewed_source(store, family, question)
    options = {"max_generated_examples": 128, "development_targets": target_report(development),
        "refresh_oracle_variants": True, "generation_batch_revision": "run:refresh:50",
        "excluded_cases": suite_exclusions(admission) + suite_exclusions(development)}
    first = await prepare_curriculum(store, options=options)
    generated = [row for row in first["examples"] if row["source"] == "generated_oracle"]
    assert 0 < len(generated) <= 32
    assert {row["validation"]["family"] for row in generated} == {"binary_strings"}
    assert len({row["semantic_key"] for row in generated}) == len(generated)
    reserved = {normalize_question(case["prompt"]) for suite in (admission, development) for case in suite["cases"]}
    for row in generated:
        assert normalize_question(row["messages"][0]["content"]) not in reserved
        assert validate_training_variant(row, excluded_cases=options["excluded_cases"])
        assert revalidate_example(store, row)
    receipts = copy.deepcopy(store.list("curriculum_generations"))
    cursors = copy.deepcopy(store.list("curriculum_oracle_cursors"))
    same = await prepare_curriculum(store, options=options)
    assert {row["id"] for row in generated} <= {row["id"] for row in same["examples"]}
    assert store.list("curriculum_generations") == receipts
    assert store.list("curriculum_oracle_cursors") == cursors
    again = await prepare_curriculum(store, options={**options, "generation_batch_revision": "run:refresh:100",
        "covered_semantic_keys": [row["semantic_key"] for row in generated]})
    fresh = [row for row in again["examples"] if row["source"] == "generated_oracle"
             and row["semantic_key"] not in {old["semantic_key"] for old in generated}]
    assert 0 < len(fresh) <= 32
    assert all(revalidate_example(store, row) for row in fresh)


@pytest.mark.asyncio
async def test_no_development_weakness_reuses_verified_replay_without_generation(store, panels):
    _, development = panels
    reviewed_source(store, "training", QUESTIONS["binary_strings"])
    initial = await prepare_curriculum(store, options={"max_generated_examples": 0})
    assert initial["examples"]
    report = summarize_development(development, measured_rows(development, failing_family=None))
    assert report["status"] == "sufficient" and report["target_families"] == []
    async def forbidden_draft(*args, **kwargs):
        pytest.fail("A sufficient diagnostic must use verified replay, without a teacher request")
    refreshed = await prepare_curriculum(store, draft=forbidden_draft, options={
        "max_generated_examples": 32, "development_targets": report, "allow_replay": True,
        "refresh_oracle_variants": True, "generation_batch_revision": "run:refresh:50",
        "covered_semantic_keys": [row["semantic_key"] for row in initial["examples"]],
    })
    assert {row["id"] for row in refreshed["examples"]} == {row["id"] for row in initial["examples"]}
    assert store.list("curriculum_generations") == [] and store.list("curriculum_oracle_cursors") == []


@pytest.mark.parametrize("missing_evidence", ["incomplete", "missing_case", "unsupported"])
@pytest.mark.asyncio
async def test_missing_diagnostic_evidence_pauses_before_generation(store, panels, missing_evidence):
    _, development = panels
    rows = measured_rows(development, failing_family=None)
    ident = development["metadata"]["dynamic_cases"][0]["case_id"] if missing_evidence != "unsupported" else "development-anchor"
    for row in rows:
        if row["id"] == ident:
            row["candidate_pass"] = None if missing_evidence == "incomplete" else False
            row["scored"] = missing_evidence != "incomplete"
    if missing_evidence == "missing_case":
        rows = [row for row in rows if row["id"] != ident]
    report = summarize_development(development, rows)
    assert report["status"] == "awaiting_evidence"
    assert ident in report["unsupported_failed_case_ids" if missing_evidence == "unsupported" else "incomplete_case_ids"]
    reviewed_source(store, "training", QUESTIONS["binary_strings"])
    async def forbidden_draft(*args, **kwargs):
        pytest.fail("Missing evidence must pause before a teacher request")
    with pytest.raises(ValueError, match="[Ee]vidence"):
        await prepare_curriculum(store, draft=forbidden_draft, options={
            "max_generated_examples": 32, "development_targets": report,
            "refresh_oracle_variants": True, "generation_batch_revision": "run:refresh:50"})
    assert store.list("curriculum_generations") == [] and store.list("curriculum_oracle_cursors") == []


def refresh_service(store, tmp_path, development):
    events = []
    async def load(ident):
        events.append(("load", ident))
    async def forbidden_prepare(*args, **kwargs):
        pytest.fail("Unchanged sufficient evidence must not request teacher preparation")
    service = SimpleNamespace(store=store, settings=SimpleNamespace(data_dir=tmp_path),
        manager=SimpleNamespace(load=load, current=lambda: {"id": "teacher"}),
        _profile_revision=lambda spec: "profile-1", _progress=lambda *args, **kwargs: None,
        _log=lambda *args: None, _require_source_consent=lambda adapter: None,
        _reserve_evaluation_hashes=lambda cases: None, _prepare_context_examples=forbidden_prepare)
    binding = controller.publish_suite(service, "run", development, "development")
    state = AdaptiveState.initial(seed=7)
    state.consume_updates(50)
    adaptive = {"checkpoint": {"checkpoint_manifest_sha256": "a" * 64},
        "development_binding": binding, "admission_binding": {"path": "/ADMISSION_MUST_NOT_BE_READ", "sha256": "b" * 64},
        "student_model_id": "student", "teacher_model_id": "teacher", "teacher_profile_revision": "profile-1",
        "original_dataset_ids": None, "current_revision_id": "existing", "revision_ids": ["existing"],
        "replay": [], "source_signature": controller.source_signature(service), "stagnant_rounds": 0,
        "state": state.to_dict()}
    store.put("adapters", {"id": "candidate", "checkpoint_manifest_sha256": "a" * 64})
    store.put("sleep_runs", {"id": "run", "status": "training", "candidate_adapter_id": "candidate", "adaptive": adaptive,
        "admission_report": {"gates": {"domain_improvement": False}, "private": "ADMISSION_REPORT_MUST_NOT_TARGET"}})
    return service, adaptive, state, events


@pytest.mark.asyncio
async def test_global_boundary_measures_development_even_without_new_sources_and_reuses_receipt(store, panels, tmp_path, monkeypatch):
    _, development = panels
    service, adaptive, state, events = refresh_service(store, tmp_path, development)
    calls = []
    async def measure(_service, suite, model_id, baseline, candidate, **kwargs):
        calls.append((suite["metadata"]["suite_purpose"], model_id, candidate["id"]))
        return measured_rows(development, failing_family=None)
    monkeypatch.setattr(controller, "measure_panel", measure)
    binding = copy.deepcopy(adaptive["development_binding"])
    await controller.refresh_data(service, "run", {}, adaptive, state)
    assert calls == [("development", "student", "candidate")]
    assert events == [("load", "student")]
    assert state.global_successful_updates == 50 and state.next_refresh_at == 100
    assert adaptive["current_revision_id"] == "existing" and adaptive["development_binding"] == binding
    assert adaptive["last_refresh"]["generated"] is False
    saved = copy.deepcopy(store.list("adaptive_operations"))
    assert "ADMISSION_REPORT_MUST_NOT_TARGET" not in json.dumps(saved)
    assert await controller.assess_development(service, "run", adaptive, state) == adaptive["development_assessment"]
    assert len(calls) == 1 and store.list("adaptive_operations") == saved


@pytest.mark.asyncio
async def test_incomplete_development_does_not_consume_boundary_or_load_teacher(store, panels, tmp_path, monkeypatch):
    _, development = panels
    service, adaptive, state, events = refresh_service(store, tmp_path, development)
    async def measure(*args, **kwargs):
        rows = measured_rows(development, failing_family=None)
        rows[0]["candidate_pass"] = None
        return rows
    monkeypatch.setattr(controller, "measure_panel", measure)
    binding = copy.deepcopy(adaptive["development_binding"])
    with pytest.raises(controller.EvidenceRequired, match="evidence"):
        await controller.refresh_data(service, "run", {}, adaptive, state)
    assert state.global_successful_updates == state.next_refresh_at == 50
    assert events == [("load", "student")]
    assert adaptive["current_revision_id"] == "existing" and adaptive["development_binding"] == binding


@pytest.mark.asyncio
async def test_changed_completed_development_report_cannot_skip_needed_training(store, panels, tmp_path, monkeypatch):
    _, development = panels
    service, adaptive, state, events = refresh_service(store, tmp_path, development)
    calls = []
    async def measure(*args, **kwargs):
        calls.append(True)
        return measured_rows(development)
    monkeypatch.setattr(controller, "measure_panel", measure)
    report = await controller.assess_development(service, "run", adaptive, state)
    assert report["target_families"] == ["binary_strings"]
    operation = store.list("adaptive_operations")[0]
    changed = copy.deepcopy(operation["result"])
    changed["report"].update(target_families=[], status="sufficient")
    store.update("adaptive_operations", operation["id"], result=changed)
    with pytest.raises(controller.EvidenceRequired):
        await controller.refresh_data(service, "run", {}, adaptive, state)
    assert state.global_successful_updates == state.next_refresh_at == 50
    assert len(calls) == 1 and events == [("load", "student")]


@pytest.mark.asyncio
async def test_zero_novel_verified_targets_preserves_dataset_and_development_bindings(store, panels, tmp_path, monkeypatch):
    _, development = panels
    reviewed_source(store, "training", QUESTIONS["binary_strings"])
    prepared = await prepare_curriculum(store, options={"max_generated_examples": 0})
    row = prepared["examples"][0]
    service, adaptive, state, events = refresh_service(store, tmp_path, development)
    store.put("adaptive_datasets", {"id": "existing", "source_fingerprints": [row["fingerprint"]],
        "example_refs": [controller.reference(row)], "holdout_refs": [], "source_conversation_ids": ["training"]})
    calls = []
    async def measure(*args, **kwargs):
        return measured_rows(development)
    async def no_novel_prepare(run_id, request):
        calls.append(request)
        assert request["_development_targets"]["target_families"] == ["binary_strings"]
        assert "ADMISSION_REPORT_MUST_NOT_TARGET" not in json.dumps(request)
        return request, {"examples": [], "holdout_examples": [], "conflicted_sources": []}
    service._prepare_context_examples = no_novel_prepare
    monkeypatch.setattr(controller, "measure_panel", measure)
    binding = copy.deepcopy(adaptive["development_binding"])
    await controller.refresh_data(service, "run", {"generation_thinking": "max"}, adaptive, state)
    assert len(calls) == 1 and events == [("load", "student"), ("load", "teacher")]
    assert state.next_refresh_at == 100 and state.global_successful_updates == 50
    assert adaptive["current_revision_id"] == "existing" and adaptive["revision_ids"] == ["existing"]
    assert adaptive["development_binding"] == binding
    assert adaptive["last_refresh"]["generated"] is False
    assert adaptive["last_refresh"]["reason"] == "no_novel_verified_targets_replay"
    assert len(store.list("adaptive_datasets")) == 1 and store.list("adaptive_development_suites") == []
