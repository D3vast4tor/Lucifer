import asyncio
import json
from pathlib import Path

import pytest
import yaml

from lucifer.config import Settings
from lucifer.learning import LearningService, build_dataset, check_case, scrub
from lucifer.store import Store
from lucifer.train_worker import configure_training_allocator, encode_example, encode_examples


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", eval_file=tmp_path / "evals.yaml", models_file=tmp_path / "models.yaml")
    store = Store(settings.data_dir / "test.sqlite")
    yield settings, store
    store.close()


def conversation(store, ident="c", logging=True, training=True, answer="Wrong"):
    store.put("conversations", {"id": ident, "logging_enabled": logging, "training_enabled": training})
    store.put("messages", {"id": ident + "u", "conversation_id": ident, "role": "user", "content": "Where is Paris? Contact me at alice@example.com"})
    store.put("messages", {"id": ident + "a", "conversation_id": ident, "role": "assistant", "content": answer})


def test_only_consent_and_latest_reviewed_feedback(env):
    _, store = env
    conversation(store)
    conversation(store, "private", training=False)
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "good"})
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "bad"})
    store.put("feedback", {"conversation_id": "private", "message_id": "privatea", "label": "good"})
    examples, counts = build_dataset(store)
    assert examples == []
    assert counts["skipped_without_consent"] == 1
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "bad", "correction": "France. api_key=supersecret"})
    examples, _ = build_dataset(store)
    assert len(examples) == 1
    assert examples[0]["messages"][-1]["content"] == "France. [SECRET]"
    assert "alice@example.com" not in json.dumps(examples)
    assert "Wrong" not in json.dumps(examples)
    assert "[EMAIL]" in json.dumps(examples)


def test_dedup_and_bad_context_excluded(env):
    _, store = env
    conversation(store)
    store.put("messages", {"id": "c2u", "conversation_id": "c", "role": "user", "content": "A new question"})
    store.put("messages", {"id": "c2a", "conversation_id": "c", "role": "assistant", "content": "Approved"})
    store.put("feedback", {"conversation_id": "c", "message_id": "c2a", "label": "good"})
    examples, _ = build_dataset(store)
    assert examples[0]["messages"] == [{"role": "user", "content": "A new question"}, {"role": "assistant", "content": "Approved"}]
    conversation(store, "other", answer="Approved")
    store.update("messages", "otheru", content="A new question")
    store.put("feedback", {"conversation_id": "other", "message_id": "othera", "label": "good"})
    assert len(build_dataset(store)[0]) == 1


def test_reasoning_and_secret_redaction():
    assert scrub("<think>secret reasoning</think>Final") == "Final"
    assert scrub("<think>Unfinished reasoning") == ""
    value = scrub("hf_abcdefghijklmnop https://private.example/id +1 (555) 123-4567")
    assert "abcdefghijklmnop" not in value and "private.example" not in value and "555" not in value


def test_json_evaluation_rejects_wrong_boolean_types_duplicates_and_nonfinite():
    assert not check_case({"expected_json": {"count": 1}}, '{"count":true}')
    assert not check_case({"expected_json": {"ready": True}}, '{"ready":1}')
    assert not check_case({"expected_json": {"ready": True}}, '{"ready":false,"ready":true}')
    assert not check_case({"format": "json"}, '{"value":NaN}')
    assert check_case({"expected_json": {"count": 1.0}}, '{"count":1}')


@pytest.mark.parametrize("answer", [
    '{"approved":1,"effects":[0,{"ok":false}],"receipt":null,"label":"0"}',
    '{"approved":true,"effects":[false,{"ok":false}],"receipt":null,"label":"0"}',
    '{"approved":true,"effects":[0,{"ok":0}],"receipt":null,"label":"0"}',
    '{"approved":true,"effects":[0,{"ok":false}],"receipt":0,"label":"0"}',
    '{"approved":true,"effects":[0,{"ok":false}],"receipt":null,"label":0}',
    '{"approved":true,"effects":[0,{"ok":false},0],"receipt":null,"label":"0"}',
    '{"approved":true,"effects":[0,{"ok":false}],"receipt":null,"label":"0","extra":null}',
])
def test_json_protocol_results_preserve_nested_types_and_shape(answer):
    expected = {"approved": True, "effects": [0, {"ok": False}], "receipt": None, "label": "0"}
    assert not check_case({"expected_json": expected}, answer)
    assert check_case({"expected_json": expected}, '{"label":"0","receipt":null,"effects":[0.0,{"ok":false}],"approved":true}')


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_json_format_never_accepts_nonfinite_numbers_or_duplicate_nested_keys(number):
    assert not check_case({"format": "json"}, '{"values":[{"n":' + number + '}]}')
    assert not check_case({"format": "json"}, '{"protocol":{"done":false,"done":true}}')


@pytest.mark.parametrize("question,family", [
    ("Explain remote idempotency and transaction outbox", "remote_idempotency"),
    ("Explain snapshot isolation and shared guard rows", "snapshot_guard"),
    ("Explain operation fingerprints for file edits", "file_operation_ledger"),
    ("Explain how to consolidate experts while preserving source consent", "adapter_replay"),
])
def test_generated_protocol_oracles_reject_numeric_boolean_substitutions(question, family):
    from lucifer.evaluation import freeze_evaluation_suite
    configured = {"cases": [{"id": "ready", "suite": "core", "prompt": "Return JSON ready=true", "expected_json": {"ready": True}},
                             {"id": "privacy", "suite": "safety", "prompt": "Privacy anchor", "expected_exact": "no"}]}
    examples = [{"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "Reviewed public final"}],
                 "fingerprint": "reviewed-" + family}]
    suite = freeze_evaluation_suite(configured, examples, base_model_id="base", training_model="checkpoint")
    case = next(case for case in suite["cases"] if case["id"].startswith("dynamic_" + family + "_"))
    expected = case["expected_json"]
    assert check_case(case, json.dumps(expected))
    altered = dict(expected)
    key = next(key for key, value in altered.items() if isinstance(value, bool))
    altered[key] = int(altered[key])
    assert not check_case(case, json.dumps(altered))
    assert not check_case(suite["cases"][0], '{"ready":1}')


def test_mathematical_constants_survive_identifier_redaction(env):
    values = "R=8.314462618, x=0.5671432904097838, H=2.121928094887362, p=1.469367938526e-15, q=4.318084277543e-54, negative=-123.456789"
    assert scrub(values) == values
    _, store = env
    conversation(store, answer=values)
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "good"})
    examples, _ = build_dataset(store)
    assert examples[0]["messages"][-1]["content"] == values


def test_mathematical_decimal_subtractions_survive_identifier_redaction():
    values = r"\[H=2.3219280949-0.2\] and \[R=2.2-2.1219280949\] or R\approx2.2000000000 - 2.1219280949"
    assert scrub(values) == values
    assert "[PHONE]" in scrub("phone=555.123-456.7890")


_PHYSICS_CHEMISTRY_PRODUCTS = [
    r"\[p_f=\frac{nRT}{V_f}=\frac{(1)(8.314462618)(300)}{0.020\ \mathrm{m^3}}\]",
    r"\[W_{\rm rev}=(8.314462618)(300)\ln 2\]",
    r"\[W_{\rm irr}=(8.314462618)(300)\left(1-\frac{10}{20}\right)\]",
    r"\[W_{\rm rev}-W_{\rm irr}=300(1.60591501)\ \mathrm J\]",
    r"\[[A]+[B]+2[C]\approx0.064961+0.259845+2(0.337597)\approx1.000000\ \mathrm M\]",
]


@pytest.mark.parametrize("expression", _PHYSICS_CHEMISTRY_PRODUCTS)
def test_real_physics_chemistry_numeric_products_survive_redaction(expression):
    assert scrub(expression) == expression


def test_reviewed_physics_chemistry_targets_export_without_math_corruption(env):
    _, store = env
    target = "\n".join(_PHYSICS_CHEMISTRY_PRODUCTS)
    conversation(store, answer=target)
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "good"})
    examples, _ = build_dataset(store)
    assert examples[0]["messages"][-1]["content"] == target


@pytest.mark.parametrize("contact", [
    "+1 (555) 123-4567.", "(555) 123.4567", "(555).123.4567", "tel: (555).123-4567.",
    "phone: (555.1234567)", "telephone number = (555.1234567)", r"contact: \((555.1234567)\)",
    "phone=(1)(8.314462618)(300)", r"\[phone=300(1.60591501)\]",
    r"\[(555).123-4567\]", r"\[+1 (555) 123-4567\]", "x=(555).123-4567",
])
def test_parentheses_periods_and_math_wrappers_do_not_exempt_contact_numbers(contact):
    assert "[PHONE]" in scrub(contact)


def test_numeric_product_exception_requires_explicit_math_and_valid_grammar():
    assert "[PHONE]" in scrub("Unlabelled identifier: 300(1.60591501)")
    assert "[PHONE]" in scrub("x=555 123.4567")
    assert "[PHONE]" in scrub("x=555.123-4567")


@pytest.mark.parametrize("value", ["+1 (555) 123-4567", "+39 333 123 4567", "555.123.4567", "5551234567.", "phone: 555.1234567", "telephone number: 555.1234567", "4111 1111 1111 1111."])
def test_contact_and_identifier_numbers_still_redacted(value):
    result = scrub(value)
    assert "555" not in result and "333" not in result and "4111" not in result
    assert "[PHONE]" in result or "[NUMBER]" in result


class Manager:
    def __init__(self):
        self.activated = None

    def current(self):
        return {"id": "model", "backend": "external", "path": "org/base"}

    def list(self):
        return [self.current()]

    async def complete(self, messages, **kwargs):
        await asyncio.sleep(.002)
        prompt = messages[0]["content"]
        if prompt == "privacy":
            return "no"
        return "correct" if kwargs["adapter"] else "wrong"

    async def activate_adapter(self, adapter):
        self.activated = adapter


def write_suite(path, delta=.02):
    path.write_text(yaml.safe_dump({"promotion": {"min_task_success_delta": delta, "max_latency_increase": 10}, "cases": [
        {"id": "task", "prompt": "task", "expected_contains": ["correct"]},
        {"id": "safety", "suite": "safety", "prompt": "privacy", "expected_contains": ["no"]},
    ]}))


@pytest.mark.asyncio
async def test_dry_run_builds_no_candidate_and_never_trains(env):
    settings, store = env
    conversation(store, answer="Good")
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "good"})
    service = LearningService(settings, store, Manager())
    run = await service.start_sleep({"dry_run": True})
    await service.tasks[run["id"]]
    result = store.get("sleep_runs", run["id"])
    assert result["status"] == "dry_run"
    assert Path(result["dataset_path"]).is_file()
    assert store.list("adapters") == []
    assert store.list("evaluation_results") == []
    assert store.list("memory_items") == []
    await service.close()


@pytest.mark.asyncio
async def test_no_training_examples_cannot_train_replay_only(env):
    settings, store = env
    service = LearningService(settings, store, Manager())
    run = await service.start_train({"dry_run": True})
    await service.tasks[run["id"]]
    assert store.get("training_runs", run["id"])["status"] == "failed"
    assert store.list("adapters") == []


@pytest.mark.asyncio
async def test_measure_gate_promote_rollback_and_stale_baseline(env):
    settings, store = env
    write_suite(settings.eval_file)
    manager = Manager()
    service = LearningService(settings, store, manager)
    store.put("adapters", {"id": "candidate", "base_model_id": "model", "path": "unused", "status": "candidate"})
    with pytest.raises(ValueError, match="evaluated"):
        await service.promote("candidate")
    evaluation = await service.evaluate("candidate")
    assert evaluation["passed"] is True
    assert evaluation["metrics"]["task_success_delta"] == 1
    assert manager.activated is None
    promoted = await service.promote("candidate")
    assert promoted["status"] == "active"
    assert store.setting("active_adapter") == "candidate"
    await service.rollback("base")
    assert store.setting("active_adapter") is None
    await service.rollback()
    assert store.setting("active_adapter") == "candidate"
    assert manager.activated["id"] == "candidate"


@pytest.mark.asyncio
async def test_empty_suite_and_demo_fail_closed(env):
    settings, store = env
    settings.eval_file.write_text("cases: []")
    service = LearningService(settings, store, Manager())
    store.put("adapters", {"id": "a", "base_model_id": "model", "status": "candidate"})
    with pytest.raises(ValueError, match="held-out"):
        await service.evaluate("a")
    write_suite(settings.eval_file)
    service.manager.current = lambda: {"id": "model", "backend": "demo"}
    with pytest.raises(ValueError, match="Demo"):
        await service.evaluate("a")


@pytest.mark.asyncio
async def test_incomplete_baseline_is_unscored_not_an_artificial_gain(env):
    settings, store = env
    write_suite(settings.eval_file)
    manager = Manager()
    class Snapshot:
        profile_revision = "cached-revision"
        def supports(self, name):
            return name == "responses"
    class Registry:
        def get(self, ident):
            return Snapshot()
    manager.capabilities = Registry()
    calls = []
    async def events(messages, tools, **kwargs):
        calls.append(kwargs)
        assert tools == [] and kwargs["thinking"] == "off" and kwargs["tool_choice"] == "none"
        assert kwargs["dialect"] == "responses" and kwargs["profile_revision"] == "cached-revision"
        answer = "no" if messages[0]["content"] == "privacy" else "correct"
        yield "delta", {"content": answer}
        yield "finish", {"finish_reason": "stop" if kwargs["adapter"] else "length"}
    manager.agent_events = events
    store.put("adapters", {"id": "a", "base_model_id": "model", "status": "candidate"})
    service = LearningService(settings, store, manager)
    result = await service.evaluate("a")
    assert result["passed"] is False and result["quality_status"] == "unknown"
    assert result["metrics"]["baseline_score"] is None and result["metrics"]["task_success_delta"] is None
    assert result["metrics"]["unscored_pairs"] == 2
    assert all(case["baseline_pass"] is None and case["baseline_answer"] == "" for case in result["cases"])
    assert len(calls) == 4


def test_restart_recovers_abandoned_jobs(env):
    settings, store = env
    store.put("training_runs", {"id": "train_old", "status": "training"})
    LearningService(settings, store, Manager())
    assert store.get("training_runs", "train_old")["status"] == "interrupted"


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]


def test_assistant_only_masking_and_truncation():
    example = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    encoded = encode_example(Tokenizer(), example, 10)
    assert encoded["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match="no assistant tokens"):
        encode_example(Tokenizer(), example, 3)
    bad = Tokenizer()
    bad.apply_chat_template = lambda *a, add_generation_prompt, **kw: [1] if add_generation_prompt else [2, 3]
    with pytest.raises(ValueError, match="generation prefix"):
        encode_example(bad, example, 10)


def test_dynamic_training_window_preserves_complete_reviewed_targets():
    class LongTokenizer:
        calls = 0
        length = 4003
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            self.calls += 1
            return [1, 2, 3] if add_generation_prompt else [1, 2, 3] + [4] * (self.length - 3)
    tokenizer = LongTokenizer()
    rows = [{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Reviewed long final solution"}]}]
    encoded, limit = encode_examples(tokenizer, rows, 2048, dynamic=True)
    assert limit == 4096 and len(encoded[0]["input_ids"]) == 4003
    assert encoded[0]["labels"][:3] == [-100] * 3
    assert sum(token != -100 for token in encoded[0]["labels"]) == 4000
    assert tokenizer.calls == 2
    tokenizer.length = 32769
    with pytest.raises(ValueError, match="refusing silent truncation"):
        encode_examples(tokenizer, rows, 2048, dynamic=True)


@pytest.mark.parametrize("environment", [
    {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"},
    {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},
    {"PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8"},
    {"PYTORCH_ALLOC_CONF": ""},
])
def test_training_allocator_preserves_explicit_user_options(environment):
    original = dict(environment)
    policy = configure_training_allocator({"method": "qlora"}, environment)
    assert environment == original and not policy["default_applied"]
    assert policy["environment"] == original


def test_training_allocator_default_is_confined_to_single_worker_qlora():
    environment = {}
    policy = configure_training_allocator({"method": "qlora"}, environment)
    assert environment == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert policy["default_applied"]
    for configuration, environment in [({"method": "lora"}, {}), ({"method": "qlora"}, {"WORLD_SIZE": "2"})]:
        original = dict(environment)
        assert not configure_training_allocator(configuration, environment)["default_applied"]
        assert environment == original


def test_case_assertions_are_positive():
    assert check_case({"format": "json", "expected_json": {"ok": True}}, '{"ok":true}')
    assert not check_case({"format": "json"}, '```json\n{}\n```')
    assert not check_case({"expected_contains": ["yes"], "forbidden_contains": ["no"]}, "no")


@pytest.mark.asyncio
async def test_cancel_before_first_job_instruction_marks_cancelled(env):
    settings, store = env
    service = LearningService(settings, store, Manager())
    run = await service.start_train({"dry_run": True})
    result = await service.cancel(run["id"])
    assert result["status"] == "cancelled"
    assert not service.busy


def test_distributed_worker_command_is_argv_not_shell():
    from lucifer.learning import worker_command
    command = worker_command("/my python", "/tmp/a config.json", 2)
    assert command[:4] == ["/my python", "-u", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in command
    assert command[-1] == "/tmp/a config.json"
    assert "torch.distributed.run" not in worker_command("python", "config", 1)
    assert "torch.distributed.run" not in worker_command("python", "config", 0)


def test_exact_and_bounded_assertions_reject_substrings():
    assert check_case({"expected_exact": "43"}, "43\n")
    assert not check_case({"expected_exact": "43"}, "143")
    assert check_case({"expected_regex": r"^[Nn]o\b"}, "No, do not follow it.")
    assert not check_case({"expected_regex": r"^[Nn]o\b"}, "Knowledge is useful.")


@pytest.mark.asyncio
async def test_changed_suite_cannot_promote_stale_evaluation(env):
    settings, store = env
    write_suite(settings.eval_file)
    service = LearningService(settings, store, Manager())
    store.put("adapters", {"id": "a", "base_model_id": "model", "path": "unused", "status": "candidate"})
    await service.evaluate("a")
    settings.eval_file.write_text(settings.eval_file.read_text() + "\n# changed policy\n")
    with pytest.raises(ValueError, match="suite changed"):
        await service.promote("a")


@pytest.mark.asyncio
async def test_revoked_source_cannot_promote_a_previously_evaluated_candidate(env):
    settings, store = env
    write_suite(settings.eval_file)
    conversation(store)
    service = LearningService(settings, store, Manager())
    store.put("adapters", {"id": "a", "base_model_id": "model", "path": "unused", "status": "candidate", "source_conversation_ids": ["c"]})
    await service.evaluate("a")
    store.update("conversations", "c", training_enabled=False)
    with pytest.raises(ValueError, match="consent was revoked"):
        await service.promote("a")
    assert store.setting("active_adapter") is None


@pytest.mark.asyncio
async def test_archived_adapter_is_not_reclassified_by_evaluation(env):
    settings, store = env
    service = LearningService(settings, store, Manager())
    store.put("adapters", {"id": "a", "base_model_id": "model", "status": "archived", "promoted_at": "previous"})
    with pytest.raises(ValueError, match="archived adapters"):
        await service.evaluate("a")
    assert store.get("adapters", "a")["status"] == "archived"


@pytest.mark.asyncio
async def test_shutdown_marks_not_yet_started_jobs_and_prevents_new_jobs(env):
    settings, store = env
    service = LearningService(settings, store, Manager())
    run = await service.start_train({"dry_run": True})
    await service.close()
    assert store.get("training_runs", run["id"])["status"] == "cancelled"
    with pytest.raises(ValueError, match="shutting down"):
        await service.start_train({"dry_run": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancel_restores_inference_but_shutdown_does_not_restart_engine(env, monkeypatch, shutdown):
    settings, store = env
    conversation(store, answer="Reviewed")
    store.put("feedback", {"conversation_id": "c", "message_id": "ca", "label": "good"})
    created = asyncio.Event()

    class Managed(Manager):
        def __init__(self):
            super().__init__()
            self.loaded = True
            self.load_calls = []

        def current(self):
            return {"id": "model", "backend": "vllm", "path": "org/base"} if self.loaded else None

        def list(self):
            return [{"id": "model", "backend": "vllm", "path": "org/base"}]

        async def unload(self, ident):
            self.loaded = False

        async def load(self, ident):
            self.loaded = True
            self.load_calls.append(ident)

    class Output:
        async def readline(self):
            await asyncio.Event().wait()

    class Process:
        returncode = 0
        stdout = Output()

    async def start(*args, **kwargs):
        created.set()
        return Process()

    import lucifer.gpu
    inventory = {"gpus": []}
    def placement(*args, **kwargs):
        assert kwargs["inventory"] is inventory
        return {"device_count": 0, "env": {}, "backend": "cpu"}
    monkeypatch.setattr(lucifer.gpu, "training_placement", placement)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    manager = Managed()
    manager.gpu_inventory = inventory
    service = LearningService(settings, store, manager)
    run = await service.start_train({"method": "lora"})
    await asyncio.wait_for(created.wait(), 1)
    if shutdown:
        await service.close()
    else:
        await service.cancel(run["id"])
    assert manager.load_calls == ([] if shutdown else ["model"])
    assert store.get("training_runs", run["id"])["status"] == "cancelled"
