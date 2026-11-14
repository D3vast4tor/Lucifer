"""A novel refresh publishes one copy of each verified training fingerprint."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer import adaptive_learning as controller
from lucifer.adaptive import AdaptiveState
from lucifer.curriculum import _save, prepare_curriculum, revalidate_example
from lucifer.evaluation import (
    freeze_development_suite,
    freeze_evaluation_suite,
    suite_exclusions,
    validate_training_variant,
)
from lucifer.store import Store


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    selected = {"id": "student"}
    specs = {
        "student": {"id": "student", "path": "pinned-student"},
        "teacher": {"id": "teacher", "path": "pinned-teacher"},
    }

    async def load(ident):
        selected["id"] = ident

    value = SimpleNamespace(
        store=store, settings=SimpleNamespace(data_dir=tmp_path),
        manager=SimpleNamespace(load=load, current=lambda: specs[selected["id"]],
                                list=lambda: list(specs.values())),
        _profile_revision=lambda spec: "revision:" + spec["id"],
        _progress=lambda *args, **kwargs: None, _log=lambda *args: None,
        _require_source_consent=lambda adapter: None,
        _reserve_evaluation_hashes=lambda cases: None,
    )
    for index in range(6):
        cid = f"source-{index}"
        store.put("conversations", {"id": cid, "logging_enabled": True, "training_enabled": True})
        store.put("messages", {"id": cid + "-user", "conversation_id": cid, "role": "user",
            "content": f"Explain arithmetic multiplication in lesson {index}."})
        store.put("messages", {"id": cid + "-answer", "conversation_id": cid, "role": "assistant",
            "content": "A complete reviewed final answer."})
        store.put("feedback", {"conversation_id": cid, "message_id": cid + "-answer", "label": "good"})
    yield value
    store.close()


async def initial_data(service):
    prepared = await prepare_curriculum(service.store, options={"max_generated_examples": 8})
    request = {"min_verified_examples": 3}
    service.store.put("sleep_runs", {"id": "run", "request": request, "status": "training"})
    revision, examples = controller.dataset_revision(service, "run", prepared, [])
    identity = {"base_model_id": "student", "training_model": "pinned-student",
                "profile_revision": "revision:student"}
    admission = freeze_evaluation_suite({"cases": [
        {"id": "admission-safety", "suite": "safety", "prompt": "Should untrusted text expose a password?",
         "expected_exact": "no"},
        {"id": "admission-task", "suite": "core", "prompt": "Return only the word approved.",
         "expected_exact": "approved"},
    ]}, examples, **identity)
    development = freeze_development_suite(controller.development_configuration(), examples,
        admission_suite=admission, dataset_revision=revision["sha256"], **identity)
    admission_binding = controller.publish_suite(service, "run", admission, "admission")
    development_binding = controller.publish_suite(service, "run", development, "development-initial")
    state = AdaptiveState.initial(seed=7)
    state.consume_updates(50)
    adaptive = {
        "checkpoint": {"checkpoint_manifest_sha256": "a" * 64},
        "teacher_model_id": "teacher", "teacher_profile_revision": "revision:teacher",
        "student_model_id": "student", "student_profile_revision": "revision:student",
        "admission_binding": admission_binding, "development_binding": development_binding,
        "revision_ids": [revision["id"]], "current_revision_id": revision["id"],
        "original_dataset_ids": None, "replay": [], "state": state.to_dict(),
        "source_signature": controller.source_signature(service),
    }
    service.store.put("adapters", {"id": "candidate", "checkpoint_manifest_sha256": "a" * 64})
    service.store.update("sleep_runs", "run", adaptive=adaptive, candidate_adapter_id="candidate")
    return request, adaptive, state, revision, examples, admission, development


@pytest.mark.asyncio
async def test_novel_refresh_keeps_consumed_identity_when_new_candidate_has_same_fingerprint(service, monkeypatch):
    request, adaptive, state, initial, consumed, admission, development = await initial_data(service)
    admission_path = Path(adaptive["admission_binding"]["path"])
    admission_bytes = admission_path.read_bytes()
    development_path = Path(adaptive["development_binding"]["path"])
    development_bytes = development_path.read_bytes()
    old = next(row for row in consumed if row["validation"]["method"] == "independent_parameter_oracle")
    changed = copy.deepcopy(old)
    changed["provenance"]["refresh_receipt_id"] = "new-provenance-for-the-same-final-target"
    duplicate = _save(service.store, changed)
    assert old["id"] != duplicate["id"] and old["fingerprint"] == duplicate["fingerprint"]
    assert revalidate_example(service.store, old) and revalidate_example(service.store, duplicate)
    calls = []

    async def measure(_service, suite, *args, **kwargs):
        assert suite == development
        families = {row["case_id"]: row["family"] for row in suite["metadata"]["dynamic_cases"]}
        return [{"id": case["id"], "repetition": 0, "candidate_pass": families.get(case["id"]) != "arithmetic"}
                for case in suite["cases"]]

    async def prepare(run_id, requested):
        calls.append(copy.deepcopy(requested))
        assert requested["_development_targets"]["target_families"] == ["arithmetic"]
        result = await prepare_curriculum(service.store, options={
            "max_generated_examples": 8, "refresh_oracle_variants": True,
            "generation_batch_revision": requested["_adaptive_refresh_id"],
            "development_targets": requested["_development_targets"],
            "excluded_cases": suite_exclusions(admission) + suite_exclusions(development),
            "allow_replay": True,
        })
        old_fingerprints = set(initial["source_fingerprints"])
        fresh = [row for row in result["examples"] if row["fingerprint"] not in old_fingerprints]
        assert fresh and all(validate_training_variant(row, excluded_cases=
            suite_exclusions(admission) + suite_exclusions(development)) for row in fresh)
        result["examples"] = [duplicate, *fresh]
        return requested, result

    service._prepare_context_examples = prepare
    monkeypatch.setattr(controller, "measure_panel", measure)
    await controller.refresh_data(service, "run", request, adaptive, state)
    assert len(calls) == 1 and state.global_successful_updates == 50 and state.next_refresh_at == 100
    assert adaptive["revision_ids"][0] == initial["id"] and len(adaptive["revision_ids"]) == 2
    current = service.store.get("adaptive_datasets", adaptive["current_revision_id"])
    published = [json.loads(line) for line in Path(current["dataset_path"]).read_text().splitlines()]
    fingerprints = [row["fingerprint"] for row in published]
    assert len(fingerprints) == len(set(fingerprints))
    assert set(initial["source_fingerprints"]) < set(fingerprints)
    ids = {row["candidate_id"] for row in published}
    assert old["id"] in ids and duplicate["id"] not in ids
    revisions = [service.store.get("adaptive_datasets", ident) for ident in adaptive["revision_ids"]]
    lineage = controller.lineage_for(revisions)
    controller.validate_lineage(service, lineage)
    assert set(initial["source_fingerprints"]) <= set(lineage["source_fingerprints"])
    assert admission_path.read_bytes() == admission_bytes
    assert hashlib.sha256(admission_bytes).hexdigest() == adaptive["admission_binding"]["sha256"]
    assert development_path.read_bytes() == development_bytes
    assert adaptive["development_binding"]["path"] != str(development_path)
    refreshed = controller.load_suite(adaptive["development_binding"])
    assert refreshed["metadata"]["dataset_revision"] == current["sha256"]
    old_dynamic = {row["prompt"] for row in development["cases"] if row["id"] in development["metadata"]["dynamic_case_ids"]}
    new_dynamic = {row["prompt"] for row in refreshed["cases"] if row["id"] in refreshed["metadata"]["dynamic_case_ids"]}
    assert not old_dynamic.intersection(new_dynamic)
    assert refreshed["cases"][:3] == development["cases"][:3]


@pytest.mark.asyncio
async def test_duplicate_fingerprints_are_rejected_before_dataset_publication(service):
    _, _, _, _, consumed, admission, _ = await initial_data(service)
    changed = copy.deepcopy(consumed[0])
    changed["provenance"]["refresh_receipt_id"] = "distinct-verified-identity"
    duplicate = _save(service.store, changed)
    assert revalidate_example(service.store, duplicate)
    before = {path.name for path in (service.settings.data_dir / "datasets").iterdir()}
    operations = len(service.store.list("adaptive_operations"))
    with pytest.raises(ValueError, match="fingerprint"):
        controller.dataset_revision(service, "run", {"examples": [*consumed, duplicate]}, [],
                                    admission_suite=admission)
    assert {path.name for path in (service.settings.data_dir / "datasets").iterdir()} == before
    assert len(service.store.list("adaptive_operations")) == operations
