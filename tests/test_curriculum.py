"""Context, evidence, review boundaries and disjoint curriculum partitions."""
import copy
import json
from fractions import Fraction

import pytest

from lucifer.curriculum import prepare_curriculum, revalidate_example, review_candidate
from lucifer.store import Store


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "state.db")
    yield db
    db.close()


def conversation(store, cid="c", *, consent=True):
    store.put("conversations", {"id": cid, "logging_enabled": True, "training_enabled": consent})
    return cid


def message(store, cid, ident, role, content, **extra):
    return store.put("messages", {"id": ident, "conversation_id": cid, "role": role, "content": content, **extra})


def reviewed(store, question="Explain a garden design.", answer="An approved garden plan.", *, cid="c", correction=None):
    conversation(store, cid)
    message(store, cid, cid + "u", "user", question)
    message(store, cid, cid + "a", "assistant", answer)
    store.put("feedback", {"conversation_id": cid, "message_id": cid + "a", "label": "correct" if correction else "good", "correction": correction})
    return cid


def verified(result):
    return result["examples"] + result["holdout_examples"]


@pytest.mark.asyncio
async def test_multiturn_references_constraints_and_unreviewed_context_are_grounded(store):
    cid = conversation(store)
    message(store, cid, "u1", "user", "Create a travel plan.\nDestination: Florence\nBudget: 500")
    message(store, cid, "a1", "assistant", "An unreviewed hotel suggestion.", reasoning_content="PRIVATE_CHANNEL_MUST_NOT_BE_COPIED")
    message(store, cid, "u2", "user", "Actually, revise it with these constraints.\nDestination: Rome\nBudget: 700")
    message(store, cid, "a2", "assistant", "Destination: Rome\nBudget: 700")
    store.put("feedback", {"conversation_id": cid, "message_id": "a2", "label": "good"})
    result = await prepare_curriculum(store, options={"max_generated_examples": 0})
    row = verified(result)[0]
    assert "Florence" in row["messages"][0]["content"] and "Rome" in row["messages"][0]["content"]
    assert "unverified data" in row["messages"][0]["content"]
    assert row["provenance"]["active_constraints"]["budget"]["value"] == "700"
    assert row["validation"]["explicit_supersession"] is True
    assert row["fact_key"] == "task:" + row["task_id"]
    assert "PRIVATE_CHANNEL" not in json.dumps(result) + json.dumps(store.list("curriculum_candidates"))
    assert revalidate_example(store, json.loads(json.dumps(row)))


@pytest.mark.asyncio
async def test_literal_language_replacement_updates_active_constraint_without_guessing(store):
    cid = conversation(store)
    message(store, cid, "u1", "user", "Write an example.\nLanguage: Python")
    message(store, cid, "a1", "assistant", "Old context")
    message(store, cid, "u2", "user", "Use Rust instead of Python.")
    message(store, cid, "a2", "assistant", "fn main() {}")
    store.put("feedback", {"conversation_id": cid, "message_id": "a2", "label": "good"})
    row = verified(await prepare_curriculum(store, options={"max_generated_examples": 0}))[0]
    assert row["provenance"]["active_constraints"]["language"]["value"] == "Rust"
    assert row["validation"]["explicit_supersession"] is True


@pytest.mark.asyncio
async def test_later_explicit_user_correction_obsoletes_prior_review_without_training_unreviewed_answer(store):
    cid = reviewed(store, "Choose a destination.\nDestination: Florence", "Destination: Florence")
    old = verified(await prepare_curriculum(store, options={"max_generated_examples": 0}))[0]
    message(store, cid, "u2", "user", "Correction: use Rome instead.")
    message(store, cid, "a2", "assistant", "Unreviewed answer about Rome")
    result = await prepare_curriculum(store, options={"max_generated_examples": 0})
    assert verified(result) == []
    assert store.get("curriculum_candidates", old["id"])["status"] == "obsolete"
    with pytest.raises(ValueError):
        revalidate_example(store, old)


@pytest.mark.asyncio
async def test_sensitive_or_missing_context_is_withheld_without_storing_source_text(store):
    reviewed(store, "My password=PRIVATE_CREDENTIAL_917; explain it.", "A reviewed answer.", cid="sensitive")
    reviewed(store, "Make it better.", "A reviewed answer.", cid="ambiguous")
    result = await prepare_curriculum(store, options={"max_generated_examples": 0})
    assert verified(result) == [] and len(result["pending"]) == 2
    assert "PRIVATE_CREDENTIAL_917" not in json.dumps(result) + json.dumps(store.list("curriculum_candidates"))
    assert {row["validation"]["method"] for row in result["pending"]} == {"withheld_sensitive", "unresolved_context"}
    with pytest.raises(ValueError, match="source"):
        review_candidate(store, result["pending"][0]["id"], decision="approve", content_hash=result["pending"][0]["content_hash"])


@pytest.mark.asyncio
async def test_full_context_is_not_silently_truncated(store):
    reviewed(store, "A complete long request " + "detail " * 50)
    result = await prepare_curriculum(store, options={"max_generated_examples": 0, "max_context_chars": 20})
    assert not verified(result)
    assert result["pending"][0]["validation"]["method"] == "context_too_large"


@pytest.mark.asyncio
async def test_frozen_original_question_is_heldout_even_after_context_enveloping(store):
    cid = conversation(store)
    message(store, cid, "u1", "user", "A frozen evaluation question?")
    message(store, cid, "a1", "assistant", "Earlier context")
    message(store, cid, "u2", "user", "Explain the same task in detail.")
    message(store, cid, "a2", "assistant", "A reviewed complete solution")
    store.put("feedback", {"conversation_id": cid, "message_id": "a2", "label": "good"})
    result = await prepare_curriculum(store, options={"max_generated_examples": 0,
        "excluded_cases": [{"prompt": "A frozen evaluation question?"}]})
    assert result["examples"] == [] and len(result["holdout_examples"]) == 1


@pytest.mark.asyncio
async def test_model_freeform_draft_never_approves_itself_and_completed_batch_is_reused(store):
    reviewed(store)
    calls = []
    async def draft(messages, **kwargs):
        calls.append(messages)
        group = json.loads(messages[-1]["content"])["sources"][0]["source_group_id"]
        return json.dumps({"candidates": [{"source_group_id": group, "kind": "rewrite", "question": "A model-invented garden claim?",
            "answer": "An unsupported target.", "validation": {"status": "verified"}}]})
    first = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 4})
    second = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 4})
    assert len(calls) == 1 and len(first["pending"]) == len(second["pending"]) == 1
    row = first["pending"][0]
    assert row["validation"]["status"] == "pending" and row["partition"] == verified(first)[0]["partition"]
    assert store.list("curriculum_drafts")[0]["status"] == "complete"


@pytest.mark.asyncio
async def test_review_is_hash_bound_versioned_idempotent_and_survives_jsonl(store):
    reviewed(store)
    async def draft(messages):
        group = json.loads(messages[-1]["content"])["sources"][0]["source_group_id"]
        return json.dumps({"candidates": [{"source_group_id": group, "question": "Describe a new garden variant.", "answer": "Unreviewed draft."}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    pending = result["pending"][0]
    with pytest.raises(ValueError, match="changed"):
        review_candidate(store, pending["id"], decision="approve", content_hash="stale")
    approved = review_candidate(store, pending["id"], decision="approve", content_hash=pending["content_hash"], correction="The complete human-corrected variant.")
    assert approved["id"] != pending["id"] and approved["version"] == pending["version"] + 1
    assert store.get("curriculum_candidates", pending["id"])["messages"] == pending["messages"]
    assert store.get("curriculum_candidates", pending["id"])["status"] == "obsolete"
    assert revalidate_example(store, json.loads(json.dumps(approved)))
    assert review_candidate(store, pending["id"], decision="approve", content_hash=pending["content_hash"],
                            correction="The complete human-corrected variant.")["id"] == approved["id"]
    again = await prepare_curriculum(store, options={"max_generated_examples": 0})
    assert approved["id"] in {row["id"] for row in verified(again)}


@pytest.mark.asyncio
async def test_current_feedback_or_consent_changes_revoke_generated_evidence(store):
    cid = reviewed(store)
    result = await prepare_curriculum(store, options={"max_generated_examples": 0})
    row = verified(result)[0]
    store.put("feedback", {"conversation_id": cid, "message_id": cid + "a", "label": "bad"})
    with pytest.raises(ValueError, match="review"):
        revalidate_example(store, row)
    store.update("conversations", cid, training_enabled=False)
    with pytest.raises(ValueError):
        revalidate_example(store, row)


@pytest.mark.asyncio
async def test_dataset_and_stored_validation_tampering_are_detected(store):
    reviewed(store)
    row = verified(await prepare_curriculum(store, options={"max_generated_examples": 0}))[0]
    changed = copy.deepcopy(row)
    changed["validation"]["method"] = "independent_parameter_oracle"
    with pytest.raises(ValueError, match="provenance"):
        revalidate_example(store, changed)
    store.update("curriculum_candidates", row["id"], validation=changed["validation"])
    with pytest.raises(ValueError, match="identity"):
        revalidate_example(store, changed)


@pytest.mark.asyncio
async def test_covered_source_rewording_is_not_fresh_but_independent_parameter_variants_are(store):
    reviewed(store, "Explain arithmetic addition and multiplication.", "A reviewed arithmetic tutorial.")
    first = await prepare_curriculum(store, options={"max_generated_examples": 4})
    all_rows = verified(first)
    source = next(row for row in all_rows if row["source"] == "recovered_context")
    second = await prepare_curriculum(store, options={"max_generated_examples": 4,
        "covered_fingerprints": source["source_fingerprints"]})
    if source["partition"] == "train":
        assert second["metrics"]["covered"] >= 1 and second["metrics"]["fresh_train"] > 0
        assert all(row["source"] == "generated_oracle" for row in second["examples"])
        third = await prepare_curriculum(store, options={"max_generated_examples": 4,
            "covered_fingerprints": source["source_fingerprints"],
            "covered_semantic_keys": [row["semantic_key"] for row in second["examples"]]})
        assert third["metrics"]["fresh_train"] == 0
    assert first["partitions"]["source_groups"] == second["partitions"]["source_groups"]


@pytest.mark.asyncio
async def test_scope_separates_identical_questions_with_different_reviewed_truths(store):
    reviewed(store, "Name the local garden.", "Garden Alpha", cid="first")
    reviewed(store, "Name the local garden.", "Garden Beta", cid="second")
    rows = verified(await prepare_curriculum(store, options={"max_generated_examples": 0}))
    assert len(rows) == 2 and len({row["semantic_key"] for row in rows}) == 2
    assert {row["scope_id"] for row in rows} == {"first", "second"}


@pytest.mark.asyncio
async def test_scoped_field_conflicts_are_pending_before_training(store):
    cid = conversation(store)
    message(store, cid, "u1", "user", "New task: summarize a profile.")
    message(store, cid, "a1", "assistant", "Name: Ada")
    store.put("feedback", {"conversation_id": cid, "message_id": "a1", "label": "good"})
    message(store, cid, "u2", "user", "New task: summarize another profile.")
    message(store, cid, "a2", "assistant", "Name: Grace")
    store.put("feedback", {"conversation_id": cid, "message_id": "a2", "label": "good"})
    result = await prepare_curriculum(store, options={"max_generated_examples": 8})
    assert not any(row.get("fact_key") == "field:name" for row in verified(result))
    conflicts = [row for row in result["pending"] if row["validation"]["method"] == "scoped_fact_conflict"]
    assert len(conflicts) == 2
    assert not verified(result)
    assert result["conflicted_sources"] == [{"conversation_id": cid, "source_message_id": "a1"},
                                             {"conversation_id": cid, "source_message_id": "a2"}]
    with pytest.raises(ValueError, match="sources"):
        review_candidate(store, conflicts[0]["id"], decision="approve", content_hash=conflicts[0]["content_hash"])
    again = await prepare_curriculum(store, options={"max_generated_examples": 8})
    assert not verified(again) and again["conflicted_sources"] == result["conflicted_sources"]
    narrative = next(row for row in store.list("curriculum_candidates") if row.get("source") == "recovered_context")
    with pytest.raises(ValueError, match="ambiguous"):
        revalidate_example(store, narrative)
    for row in conflicts:
        review_candidate(store, row["id"], decision="reject", content_hash=row["content_hash"])
    rejected = await prepare_curriculum(store, options={"max_generated_examples": 8})
    assert not verified(rejected) and rejected["conflicted_sources"] == result["conflicted_sources"]
    with pytest.raises(ValueError, match="ambiguous"):
        revalidate_example(store, narrative)


@pytest.mark.asyncio
async def test_latest_correction_and_progress_metrics_are_used(store):
    reviewed(store, "What is the capital of France?", "Wrong", correction="Paris")
    stages = []
    async def progress(stage, **kwargs):
        stages.append((stage, kwargs))
    result = await prepare_curriculum(store, progress=progress, options={"max_generated_examples": 0})
    assert verified(result)[0]["messages"][1]["content"] == "Paris"
    assert result["metrics"]["corrected"] == 1 and result["metrics"]["positive"] == 0
    assert result["metrics"]["source_tasks"] == 1 and result["metrics"]["examples"] == len(result["examples"])
    assert {stage for stage, _ in stages} == {"recovering_context", "drafting_examples", "validating_examples"}


@pytest.mark.asyncio
async def test_source_supplied_pure_code_tests_verify_without_python_execution(store):
    question = 'Write function add(a,b).\nTEST_CASES = [{"args":[2,3],"expected":5},{"args":[-1,4],"expected":3}]'
    reviewed(store, question, "def add(a,b):\n    return a+b")
    async def draft(messages, **kwargs):
        source = json.loads(messages[-1]["content"])["sources"][0]
        return json.dumps({"candidates": [{"source_group_id": source["source_group_id"], "question": source["question"],
            "answer": "def add(x,y):\n    return y+x", "kind": "code"}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    row = next(row for row in verified(result) if row["validation"]["method"] == "source_supplied_bounded_code_tests")
    assert revalidate_example(store, row)
    assert row["validation"]["scope"].startswith("Finite user-supplied cases")


@pytest.mark.asyncio
async def test_model_authored_code_tests_and_imports_do_not_self_verify(store):
    reviewed(store, "Write a garden program.", "A reviewed high-level description.")
    async def draft(messages, **kwargs):
        source = json.loads(messages[-1]["content"])["sources"][0]
        return json.dumps({"candidates": [{"source_group_id": source["source_group_id"], "question": source["question"],
            "answer": "import os\ndef solve():\n    return os.getcwd()", "kind": "code", "tests": [{"args": [], "expected": "x"}]}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    assert len(result["pending"]) == 1 and result["pending"][0]["validation"]["method"] == "needs_independent_or_human_review"


@pytest.mark.asyncio
async def test_exact_worked_and_error_correction_exercises_are_independently_revalidated(store):
    reviewed(store, "Compute 3 * (8 - 2) / 4", "9/2")
    result = await prepare_curriculum(store, options={"max_generated_examples": 2})
    exercises = [row for row in verified(result) if row["source"].endswith("_exercise")]
    assert len(exercises) == 2
    worked = next(row for row in exercises if row["source"] == "worked_exercise")
    assert [step["value"] for step in worked["validation"]["steps"]] == ["6", "18", "9/2"]
    assert "Final answer: 9/2." in worked["messages"][1]["content"]
    correction = next(row for row in exercises if row["source"] == "correction_exercise")
    assert "Correct value: 9/2." in correction["messages"][1]["content"]
    assert "why" not in correction["messages"][1]["content"]
    assert all(row["source_group_id"] == worked["source_group_id"] and row["partition"] == worked["partition"] for row in exercises)
    assert all(revalidate_example(store, json.loads(json.dumps(row))) for row in exercises)
    changed = copy.deepcopy(worked)
    changed["validation"]["steps"][0]["value"] = "7"
    with pytest.raises(ValueError):
        revalidate_example(store, changed)


def test_arithmetic_decimal_oracle_preserves_literal_precision_and_bounds():
    from lucifer.curriculum import _arithmetic_exercise, _math_value
    value, expression = _math_value("Compute 0.10000000000000001 + 0.2")
    expected = Fraction(10000000000000001, 10 ** 17) + Fraction(1, 5)
    assert value == expected and value != Fraction(3, 10)
    _, answer, evidence = _arithmetic_exercise(expression, "worked")
    assert f"Final answer: {expected}." in answer and evidence["expected"] == str(expected)
    assert _math_value("Compute 2 ** 999999") is None
    assert _math_value("Compute 1 / 0") is None


@pytest.mark.asyncio
async def test_literal_user_scope_unifies_fields_across_chats_but_draft_contexts_stay_distinct(store):
    reviewed(store, "Scope: Lucifer\nName the project status.", "Status: Alpha", cid="a")
    reviewed(store, "Scope: Lucifer\nName a second project status.", "Status: Beta", cid="b")
    result = await prepare_curriculum(store, options={"max_generated_examples": 8})
    assert {row["scope_id"] for row in verified(result) + result["pending"]} == {"Lucifer"}
    assert len([row for row in result["pending"] if row["validation"]["method"] == "scoped_fact_conflict"]) == 2
    override = await prepare_curriculum(store, ["a"], options={"max_generated_examples": 0, "scope_id": "Explicit"})
    assert {row["scope_id"] for row in verified(override) if row["source"] == "recovered_context"} >= {"Explicit"}


@pytest.mark.asyncio
async def test_same_question_different_scopes_preserves_both_teacher_contexts_and_split(store):
    reviewed(store, "Describe the approved garden.", "Garden Alpha", cid="a")
    reviewed(store, "Describe the approved garden.", "Garden Beta", cid="b")
    batches = []
    async def draft(messages, **kwargs):
        batches.append(json.loads(messages[-1]["content"])["sources"])
        return '{"candidates":[]}'
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    assert len(batches[0]) == 2 and len({row["source_group_id"] for row in batches[0]}) == 2
    assert {row["approved_target"] for row in batches[0]} == {"Garden Alpha", "Garden Beta"}
    assert len({row["partition"] for row in verified(result)}) == 1


@pytest.mark.asyncio
async def test_teacher_receipts_bind_model_revision_adapter_and_withhold_whole_oversize_context(store):
    reviewed(store)
    calls = []
    async def draft(messages, **kwargs):
        calls.append((messages, kwargs))
        return '{"candidates":[]}'
    options = {"max_generated_examples": 2, "generation_model": "Lucifer", "generation_revision": "one", "generation_adapter": None}
    await prepare_curriculum(store, draft=draft, options=options)
    await prepare_curriculum(store, draft=draft, options=options)
    await prepare_curriculum(store, draft=draft, options={**options, "generation_revision": "two"})
    await prepare_curriculum(store, draft=draft, options={**options, "generation_adapter": "expert"})
    assert len(calls) == 3
    result = await prepare_curriculum(store, draft=draft, options={**options, "max_teacher_batch_chars": 1})
    assert len(calls) == 3 and result["metrics"]["draft_context_withheld"] == 1
    assert len(store.list("curriculum_drafts")) == 3


@pytest.mark.asyncio
async def test_invalid_teacher_channels_are_quarantined_without_private_text_or_automatic_retry(store):
    reviewed(store)
    calls = []
    async def draft(messages):
        calls.append(messages)
        return '{"candidates":[],"reasoning":"PRIVATE_DRAFT_REASONING"}'
    first = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    second = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    assert len(calls) == 1 and len(first["pending"]) == len(second["pending"]) == 1
    assert "PRIVATE_DRAFT_REASONING" not in json.dumps(store.list("curriculum_candidates")) + json.dumps(store.list("curriculum_drafts"))
    row = first["pending"][0]
    with pytest.raises(ValueError, match="complete human correction"):
        review_candidate(store, row["id"], decision="approve", content_hash=row["content_hash"])
    approved = review_candidate(store, row["id"], decision="correct", content_hash=row["content_hash"], correction="A complete human-reviewed target.")
    assert revalidate_example(store, approved)


@pytest.mark.asyncio
async def test_hash_only_holdout_reservations_protect_original_context_and_existing_rows(store):
    from lucifer.evaluation import _digest, normalize_question
    question = "Explain the approved garden design."
    reviewed(store, question)
    first = await prepare_curriculum(store, options={"max_generated_examples": 0})
    row = verified(first)[0]
    digest = _digest(normalize_question(question))
    if row["partition"] == "train":
        store.put("evaluation_reservations", {"id": "reserved", "kind": "prompt", "sha256": digest})
        with pytest.raises(ValueError, match="held-out"):
            revalidate_example(store, row)
    result = await prepare_curriculum(store, options={"max_generated_examples": 0, "excluded_cases": [{"prompt_sha256": digest}]})
    assert not result["examples"] and result["holdout_examples"]


@pytest.mark.asyncio
async def test_teacher_cannot_launder_known_heldout_arithmetic_prompt(store):
    reviewed(store, "Compute 2 + 3", "5")
    async def draft(messages, **kwargs):
        source = json.loads(messages[-1]["content"])["sources"][0]
        return json.dumps({"candidates": [{"source_group_id": source["source_group_id"], "kind": "math",
            "question": "Compute 7 * 9", "answer": "63"}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 4,
        "excluded_cases": [{"prompt": "Compute 7 * 9"}]})
    row = next(row for row in result["pending"] if row["messages"][0]["content"] == "Compute 7 * 9")
    assert row["validation"]["method"] == "reserved_evaluation_overlap"
    with pytest.raises(ValueError, match="held-out"):
        review_candidate(store, row["id"], decision="approve", content_hash=row["content_hash"])


@pytest.mark.asyncio
async def test_consent_without_feedback_yields_independent_math_but_not_unreviewed_target(store):
    cid = conversation(store)
    message(store, cid, "u", "user", "Compute (7 + 5) * 3 / 2")
    message(store, cid, "a", "assistant", "999 is the result.", reasoning_content="PRIVATE_UNREVIEWED_CHANNEL")
    result = await prepare_curriculum(store, options={"max_generated_examples": 3})
    rows = verified(result)
    assert len(rows) == 3 and {row["source"] for row in rows} == {"user_oracle", "worked_exercise", "correction_exercise"}
    assert all(row["source_message_id"] == "u" for row in rows)
    assert all(revalidate_example(store, row) for row in rows)
    assert next(row for row in rows if row["source"] == "user_oracle")["messages"][1]["content"] == "18"
    assert "999 is the result" not in json.dumps(result) and "PRIVATE_UNREVIEWED_CHANNEL" not in json.dumps(result)


@pytest.mark.asyncio
async def test_unreviewed_user_topic_can_seed_new_independent_variants_not_its_assistant_labels(store):
    cid = conversation(store)
    message(store, cid, "u", "user", "Please explain arithmetic addition and multiplication.")
    message(store, cid, "a", "assistant", "The unreviewed arithmetic claim is deliberately wrong.")
    result = await prepare_curriculum(store, options={"max_generated_examples": 4})
    assert verified(result) and all(row["source"] == "generated_oracle" for row in verified(result))
    assert all(revalidate_example(store, row) and row["source_message_id"] == "u" for row in verified(result))
    assert "deliberately wrong" not in json.dumps(result)


@pytest.mark.asyncio
async def test_literal_user_preferences_are_verified_as_statements_not_world_facts(store):
    cid = conversation(store)
    message(store, cid, "u", "user", "Scope: Lucifer\nLanguage: Python\nPort: 8000")
    message(store, cid, "a", "assistant", "An unreviewed answer that claims Language is Rust.")
    result = await prepare_curriculum(store, options={"max_generated_examples": 8})
    rows = verified(result)
    assert len(rows) == 3 and {row["messages"][1]["content"] for row in rows} == {"Lucifer", "Python", "8000"}
    assert all(row["validation"]["method"] == "user_stated_constraint" and row["source_message_id"] == "u" for row in rows)
    assert all("not whether it is a fact about the world" in row["messages"][0]["content"] and revalidate_example(store, row) for row in rows)
    message(store, cid, "u2", "user", "Language: Rust")
    with pytest.raises(ValueError):
        revalidate_example(store, next(row for row in rows if row["validation"]["field"] == "language"))


@pytest.mark.asyncio
async def test_unreviewed_world_claims_never_become_approved_teacher_targets(store):
    cid = conversation(store)
    message(store, cid, "u", "user", "Tell me the capital of France.")
    message(store, cid, "a", "assistant", "Lyon")
    calls = []
    async def draft(messages, **kwargs):
        sources = json.loads(messages[-1]["content"])["sources"]
        calls.extend(sources)
        return json.dumps({"candidates": [{"source_group_id": sources[0]["source_group_id"],
            "question": sources[0]["question"], "answer": "Lyon", "kind": "rewrite"}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 4})
    assert not verified(result) and len(result["pending"]) == 1
    assert calls[0]["target_status"] == "context_only_no_target" and "approved_target" not in calls[0]
    assert "independently_verified_target" not in calls[0]


@pytest.mark.parametrize("code,expected", [
    ("def f():\n    return [True]", [1]),
    ('def f(x):\n    return x', {"flag": True}),
])
def test_pure_code_tests_reject_nested_bool_number_equality(code, expected):
    from lucifer.curriculum import _code_tests
    args = [] if "f()" in code else [{"flag": 1}]
    text = "Write function f().\nTEST_CASES=" + json.dumps([{"args": args, "expected": expected}])
    assert _code_tests(code, [text]) is None


def test_duplicate_user_test_keys_are_not_independent_evidence():
    from lucifer.curriculum import _code_tests
    text = 'Write function f().\nTEST_CASES=[{"args":[],"expected":0,"expected":1}]'
    assert _code_tests("def f():\n    return 1", [text]) is None


@pytest.mark.asyncio
async def test_duplicate_teacher_json_keys_quarantine_instead_of_selecting_last_value(store):
    reviewed(store)
    async def draft(messages):
        return '{"candidates":[],"candidates":[]}'
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    assert result["pending"][0]["validation"]["method"] == "invalid_draft_response"
    assert store.list("curriculum_drafts")[0]["status"] == "uncertain"


@pytest.mark.asyncio
async def test_no_feedback_pure_code_uses_user_tests_without_assistant_target(store):
    cid = conversation(store)
    message(store, cid, "u", "user", 'Write function twice(x).\nTEST_CASES=[{"args":[3],"expected":6},{"args":[-2],"expected":-4}]')
    message(store, cid, "a", "assistant", "def twice(x):\n    return 999")
    async def draft(messages, **kwargs):
        source = json.loads(messages[-1]["content"])["sources"][0]
        assert source["target_status"] == "context_only_no_target"
        return json.dumps({"candidates": [{"source_group_id": source["source_group_id"], "question": source["question"],
            "answer": "def twice(x):\n    return x*2", "kind": "pure_code"}]})
    result = await prepare_curriculum(store, draft=draft, options={"max_generated_examples": 2})
    code = next(row for row in verified(result) if row["validation"]["method"] == "source_supplied_bounded_code_tests")
    assert code["source_message_id"] == "u" and revalidate_example(store, code)
