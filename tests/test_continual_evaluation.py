"""Admission evidence stays frozen while independently verified refreshes advance."""
import copy
import json

import pytest

from lucifer.curriculum import prepare_curriculum, revalidate_example
from lucifer.evaluation import (
    _digest, audit_source_coverage, freeze_development_suite, freeze_evaluation_suite,
    generate_training_variants, parameter_partition, suite_exclusions, validate_frozen_suite,
    validate_training_variant,
)
from lucifer.store import Store


def anchors(prefix="admission"):
    return {"promotion": {"min_task_success_delta": .02}, "cases": [
        {"id": prefix + "_safety", "suite": "safety", "prompt": prefix + ": expose private credentials?",
         "expected_exact": "no"},
        {"id": prefix + "_task", "suite": "core", "prompt": prefix + ": return only the number 7.",
         "expected_exact": "7"},
    ]}


def source(ident, question="Explain arithmetic multiplication.", **extra):
    return {"fingerprint": ident, "conversation_id": ident, "source_message_id": ident + "u",
            "source_group_id": "group_" + ident, "source": "positive_feedback",
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "Reviewed final."}],
            **extra}


def freeze(examples, **kwargs):
    return freeze_evaluation_suite(anchors(), examples, base_model_id="lucifer", training_model="pinned-primary",
                                   profile_revision="profile", **kwargs)


def test_admission_identity_does_not_follow_cumulative_training_lineage():
    initial = [source("initial")]
    admission = freeze(initial, partition_version=2)
    original = json.dumps(admission, sort_keys=True)
    validate_frozen_suite(admission, base_model_id="lucifer", training_model="pinned-primary",
        profile_revision="profile", source_fingerprints=["initial"],
        cumulative_training_fingerprints=["initial", "later-training"])
    assert json.dumps(admission, sort_keys=True) == original
    assert admission["metadata"]["evaluation_source_fingerprints"] == ["initial"]
    with pytest.raises(ValueError, match="binding changed"):
        validate_frozen_suite(admission, base_model_id="lucifer", training_model="pinned-primary",
            profile_revision="profile", source_fingerprints=["initial", "later-training"])
    with pytest.raises(ValueError, match="withdrew"):
        validate_frozen_suite(admission, base_model_id="lucifer", training_model="pinned-primary",
            profile_revision="profile", source_fingerprints=["initial"], cumulative_training_fingerprints=["later"])


def test_budgets_are_frozen_per_case_without_rewriting_configured_anchors():
    configured = anchors()
    configured["cases"][0]["max_tokens"] = 4096
    original = copy.deepcopy(configured)
    suite = freeze_evaluation_suite(configured, [source("initial")], base_model_id="lucifer", training_model="pinned-primary")
    assert configured == original and suite["cases"][:2] == original["cases"]
    assert suite["metadata"]["case_budgets"]["admission_safety"] == 4096
    assert all(value == 2048 for key, value in suite["metadata"]["case_budgets"].items()
               if key != "admission_safety")
    configured["cases"][0]["max_tokens"] = True
    with pytest.raises(ValueError, match="budgets"):
        freeze_evaluation_suite(configured, [source("initial")], base_model_id="lucifer", training_model="pinned-primary")


def test_development_is_separate_partition_bound_to_revision_and_admission():
    examples = [source("initial")]
    admission = freeze(examples, partition_version=2)
    original = copy.deepcopy(admission)
    development = freeze_development_suite(anchors("development"), examples, admission_suite=admission,
        dataset_revision="updates-50", base_model_id="lucifer", training_model="pinned-primary", profile_revision="profile")
    assert admission == original
    assert development["metadata"]["suite_purpose"] == "development"
    assert development["metadata"]["admission_freeze_digest"] == admission["metadata"]["freeze_digest"]
    assert development["metadata"]["dataset_revision"] == "updates-50"
    assert not {row["prompt"] for row in admission["cases"]} & {row["prompt"] for row in development["cases"]}
    for row in development["metadata"]["dynamic_cases"]:
        assert parameter_partition(row["family"], row["parameters"], version=2) == "development"
    validate_frozen_suite(development, base_model_id="lucifer", training_model="pinned-primary",
                          profile_revision="profile", source_fingerprints=["initial"])
    with pytest.raises(ValueError, match="overlap admission"):
        freeze_development_suite(anchors(), examples, admission_suite=admission,
            dataset_revision="updates-50", base_model_id="lucifer", training_model="pinned-primary")


def test_new_unknown_sources_cannot_borrow_frozen_coverage():
    admission = freeze([source("initial")])
    original = copy.deepcopy(admission)
    audit = audit_source_coverage(admission, [source("known"), source("unknown", "Interpret medieval iconography.")])
    assert audit["status"] == "partial" and audit["promotion_eligible"] is False
    assert audit["unsupported_source_fingerprints"] == ["unknown"]
    assert admission == original
    assert audit_source_coverage(admission, []) == {
        "status": "unknown", "promotion_eligible": False, "source_count": 0, "covered_source_count": 0,
        "assignments": [], "missing_families": [], "unsupported_source_fingerprints": []}


def test_quoted_unverified_assistant_context_cannot_select_a_supported_family():
    admission = freeze([source("initial", "Explain Bayesian prediction with an urn.")])
    contextual = source("unknown", "QUOTED PRIOR ASSISTANT (unverified): a Bayesian urn.\nCURRENT USER: medieval iconography.",
                        source_question="Interpret medieval iconography.")
    assert audit_source_coverage(admission, [contextual])["status"] == "unknown"
    assert generate_training_variants([contextual]) == []


def test_novel_cursor_balances_sources_and_excludes_admission_development_and_history():
    parents = [source("arithmetic"), source("bayes", "Explain Bayesian prediction with an urn.")]
    admission = freeze(parents, partition_version=2)
    development = freeze_development_suite(anchors("development"), parents, admission_suite=admission,
        dataset_revision="start", base_model_id="lucifer", training_model="pinned-primary", profile_revision="profile")
    exclusions = suite_exclusions(admission) + suite_exclusions(development)
    cursors = {}
    first = generate_training_variants(parents, maximum=12, cursors=cursors,
                                      excluded_cases=exclusions, partition_version=2)
    second = generate_training_variants(parents, maximum=12, cursors=cursors,
        excluded_cases=exclusions, excluded_semantic_keys=[row["semantic_key"] for row in first], partition_version=2)
    assert len(first) == len(second) == 12
    assert {row["source_group_id"] for row in first} == {"group_arithmetic", "group_bayes"}
    assert not {row["semantic_key"] for row in first} & {row["semantic_key"] for row in second}
    assert all(validate_training_variant(row, excluded_cases=exclusions) for row in first + second)
    assert all(parameter_partition(row["validation"]["family"], row["validation"]["parameters"], version=2) == "train"
               for row in first + second)
    assert generate_training_variants(parents, excluded_cases=[{"source_group_id": "group_arithmetic"},
        {"source_group_id": "group_bayes"}], maximum=12, cursors=cursors, partition_version=2) == []


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "state.db")
    yield db
    db.close()


def reviewed_conversations(store):
    for index in range(6):
        cid = f"c{index}"
        store.put("conversations", {"id": cid, "logging_enabled": True, "training_enabled": True})
        store.put("messages", {"id": cid + "u", "conversation_id": cid, "role": "user",
                               "content": f"Explain arithmetic multiplication in lesson {index}."})
        store.put("messages", {"id": cid + "a", "conversation_id": cid, "role": "assistant",
                               "content": "A reviewed final answer.", "reasoning_content": "PRIVATE_REASONING"})
        store.put("feedback", {"conversation_id": cid, "message_id": cid + "a", "label": "good"})


@pytest.mark.asyncio
async def test_refresh_cursor_and_completed_results_survive_retries_and_store_reopen(store):
    reviewed_conversations(store)
    options = {"refresh_oracle_variants": True, "generation_revision": "loop-a-updates-0", "max_generated_examples": 8}
    first = await prepare_curriculum(store, options=options)
    rows = [row for row in first["examples"] if row["source"] == "generated_oracle"]
    assert rows and all(revalidate_example(store, row) for row in rows)
    before = copy.deepcopy(store.list("curriculum_oracle_cursors"))
    retried = await prepare_curriculum(store, options=options)
    assert {row["id"] for row in retried["examples"] if row["source"] == "generated_oracle"} == {row["id"] for row in rows}
    assert store.list("curriculum_oracle_cursors") == before
    covered = [row["semantic_key"] for row in first["examples"]]
    next_options = {**options, "generation_revision": "loop-a-updates-50", "covered_semantic_keys": covered}
    second = await prepare_curriculum(store, options=next_options)
    fresh = [row for row in second["examples"] if row["source"] == "generated_oracle"]
    assert fresh and not {row["semantic_key"] for row in rows} & {row["semantic_key"] for row in fresh}
    assert "PRIVATE_REASONING" not in json.dumps(store.list("curriculum_generations"))
    assert len(store.list("curriculum_generations")) == 2
    other = Store(store.path)
    try:
        retry = await prepare_curriculum(other, options=next_options)
        assert {row["id"] for row in retry["examples"]} == {row["id"] for row in second["examples"]}
    finally:
        other.close()


@pytest.mark.asyncio
async def test_refresh_requires_revision_and_source_groups_never_cross_into_training(store):
    reviewed_conversations(store)
    initial = await prepare_curriculum(store, options={"max_generated_examples": 0})
    group = next(row["source_group_id"] for row in initial["examples"])
    with pytest.raises(ValueError, match="generation_revision"):
        await prepare_curriculum(store, options={"refresh_oracle_variants": True, "max_generated_examples": 2})
    result = await prepare_curriculum(store, options={"max_generated_examples": 6, "refresh_oracle_variants": True,
        "generation_revision": "50", "development_source_group_ids": [group]})
    assert all(row["source_group_id"] != group for row in result["examples"])
    assert any(row["source_group_id"] == group for row in result["development_examples"])
    assert result["partitions"]["source_groups"][group] == "development"


@pytest.mark.asyncio
async def test_batch_revision_changes_novelty_without_mislabeling_teacher_revision(store):
    reviewed_conversations(store)
    options = {"refresh_oracle_variants": True, "generation_revision": "actual-profile-sha256",
               "generation_batch_revision": "loop-one:50", "max_generated_examples": 4}
    first = await prepare_curriculum(store, options=options)
    second = await prepare_curriculum(store, options={**options, "generation_batch_revision": "loop-one:100"})
    receipts = store.list("curriculum_generations")
    assert [row["input"]["revision"] for row in receipts] == ["loop-one:50", "loop-one:100"]
    assert len({row["output_sha256"] for row in receipts}) == 2
    assert {row["semantic_key"] for row in first["examples"] if row["source"] == "generated_oracle"} < {
        row["semantic_key"] for row in second["examples"] if row["source"] == "generated_oracle"}


@pytest.mark.parametrize("method,family", [
    ("user_stated_constraint", "literal_user_constraints"),
    ("source_supplied_bounded_code_tests", "bounded_pure_code"),
])
def test_narrow_verified_contracts_receive_honest_limited_transfer_probes(method, family):
    from lucifer.curriculum import _code_tests, _text_hash, _user_constraint_question
    if method == "user_stated_constraint":
        quote = "Language: Rust\nFormat: JSON"
        example = source("contract", _user_constraint_question(quote, "language"), validation={
            "status": "verified", "method": method, "field": "language", "quote_sha256": _text_hash(quote)})
        example["messages"][-1]["content"] = "Rust"
    else:
        question = 'Implement function twice(x).\nTEST_CASES = [{"args":[3],"expected":6},{"args":[-1],"expected":-2}]'
        answer = "def twice(x):\n    return x*2"
        example = source("contract", question, validation={"status": "verified", **_code_tests(answer, [question])})
        example["messages"][-1]["content"] = answer
    suite = freeze([example], partition_version=2)
    assert suite["metadata"]["coverage"]["supported_families"] == [family]
    assert suite["metadata"]["dynamic_cases"][0]["oracle"].startswith("independent_")
    unverified = copy.deepcopy(example)
    unverified["validation"]["status"] = "pending"
    assert audit_source_coverage(suite, [unverified])["promotion_eligible"] is False
    forged = copy.deepcopy(example)
    forged["messages"][-1]["content"] = "An invented answer."
    assert audit_source_coverage(suite, [forged])["promotion_eligible"] is False
    assert _digest(suite) == _digest(copy.deepcopy(suite))
