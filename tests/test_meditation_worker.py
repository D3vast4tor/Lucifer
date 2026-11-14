"""Real CPU optimizer checks on tiny fixtures; native acceptance is separate."""
import json
from pathlib import Path

import pytest

from lucifer.head_pack import file_hash, floats_to_bf16
from lucifer.meditation_worker import ResourceLimitError, evaluate_head, run, train_segment

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_cuda_oom_receipt_proves_discardable_inputs_without_saving_exception_text(tmp_path, head_data, monkeypatch):
    import sys
    import lucifer.meditation_worker as worker
    from lucifer.meditation_recovery import verify_failure
    output = tmp_path / "oom"
    config = {**head_data, "mode": "train", "device": "cuda", "output_dir": str(output)}
    request = tmp_path / "config.json"
    request.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["worker", "--config", str(request)])
    monkeypatch.setattr(worker, "_device", lambda config, torch: "cpu")
    def failed(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("PRIVATE_ALLOCATION_MESSAGE")
    monkeypatch.setattr(torch.nn.functional, "linear", failed)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        worker.main()
    error = json.loads((output / "error.json").read_text())
    assert error["memory_failure"]["safe_to_discard"] is True
    assert verify_failure(config, error["memory_failure"], worker_path=worker.__file__)
    assert "PRIVATE_ALLOCATION_MESSAGE" not in json.dumps(error)
    assert not (output / "result.json").exists()


def test_native_admission_scores_complete_reserved_windows_and_native_references(tmp_path, monkeypatch):
    import hashlib
    import lucifer.meditation_worker as worker
    import lucifer.head_capture as capture
    datasets = {"train": [{"group": "training", "ids": [1, 2], "labels": [-100, 2]}],
                "development": [{"group": "development", "ids": [2, 3], "labels": [-100, 3]}],
                "holdout": [{"group": "selected", "ids": [4, 5, 6], "labels": [-100, -100, 6]},
                            {"group": "unselected", "ids": [7, 8], "labels": [-100, 8]},
                            {"group": "selected", "ids": [9, 10], "labels": [-100, 10]}]}
    provenance = {"dataset_sha256": "fixture-source"}
    paths = {}
    for name in ("engine", "runtime_profile", "head_path"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = str(path)
    expected = {"engine_sha256": file_hash(paths["engine"]), "profile_sha256": file_hash(paths["runtime_profile"]),
                "head_sha256": file_hash(paths["head_path"])}
    prepared = {"provenance": provenance, "execution_plan_sha256": "fixed-plan",
        "tokenized_datasets_sha256": hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest(),
        "datasets": {"holdout": {"native_group_clipped_losses": {"selected": 2.0},
                                   "original_native_group_clipped_losses": {"selected": 1.98}}}}
    monkeypatch.setattr(worker, "tokenize_datasets", lambda config: (datasets, provenance))
    monkeypatch.setattr(worker, "load_pack", lambda path: "parent-pack")
    monkeypatch.setattr(worker, "create_head_pack", lambda *args: {"path": "candidate-pack", "head_sha256": expected["head_sha256"]})
    calls = []
    def score(config):
        calls.append(config)
        assert config["segments"] == [datasets["holdout"][0], datasets["holdout"][2]]
        assert config["pack"] == "candidate-pack" and config["score_only"] is True
        assert config["dump_logits"] is False
        return {"native_group_clipped_losses": {"selected": 1.97}, "execution_plan_sha256": "fixed-plan"}
    monkeypatch.setattr(capture, "capture_native", score)
    config = {**paths, **expected, "pack": "parent-pack", "selected_groups": ["selected"],
              "prepared": prepared, "execution_plan_sha256": "fixed-plan", "output_dir": str(tmp_path / "admission")}
    measured = worker.native_admission(config)
    assert measured["backend"] == "native_engine_teacher_forced" and len(calls) == 1
    assert measured["group_clipped_losses"] == {"selected": 1.97}
    assert measured["converted_group_clipped_losses"] == {"selected": 2.0}
    assert measured["original_group_clipped_losses"] == {"selected": 1.98}
    datasets["holdout"][0]["ids"][0] = 99
    with pytest.raises(ValueError, match="source windows"):
        worker.native_admission({**config, "output_dir": str(tmp_path / "tampered")})
    assert len(calls) == 1  # Refuse changed context before any native effect.


def test_tokenize_rejects_changed_dataset_before_loading_tokenizer(tmp_path, monkeypatch):
    import lucifer.meditation_worker as worker
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{}')
    def forbidden(config):
        pytest.fail("Changed source reached tokenizer/model initialization")
    monkeypatch.setattr(worker, "load_primary_tokenizer", forbidden)
    with pytest.raises(ValueError, match="dataset changed"):
        worker.tokenize_datasets({"dataset_path": str(dataset), "dataset_sha256": "expected"})


def test_oversized_feature_cache_is_rejected_before_export_or_native_engine(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lucifer.meditation_worker as worker
    from lucifer.meditation_storage import StorageLimitError
    from lucifer.meditation_recovery import config_hash
    datasets = {"train": [{"ids": [0, 1], "labels": [-100, 1]}]}
    monkeypatch.setattr(worker, "tokenize_datasets", lambda config: (datasets, {}))
    monkeypatch.setattr(worker, "load_pack", lambda path: SimpleNamespace(head=SimpleNamespace(shape=[2, 2])))
    def forbidden(*args, **kwargs):
        pytest.fail("Storage preflight must precede native/export effects")
    monkeypatch.setattr(worker, "export_artifacts", forbidden)
    monkeypatch.setattr(worker, "_native_gate", forbidden)
    config = {"mode": "prepare", "run_root": str(tmp_path), "output_dir": str(tmp_path / "features"),
              "pack": "fixture-pack", "disk_limit_bytes": 1}
    with pytest.raises(StorageLimitError) as caught:
        worker.prepare(config)
    assert caught.value.preflight_rejection["config_sha256"] == config_hash(config)


@pytest.fixture
def head_data(tmp_path):
    torch.set_num_threads(2)
    shape = [8, 16]
    head = tmp_path / "baseline.bf16"
    head.write_bytes(floats_to_bf16(np.zeros(shape, dtype=np.float32)).tobytes())
    features = np.tile(np.eye(16, dtype=np.float32)[:8], (4, 1))
    labels = np.tile(np.arange(8, dtype=np.int64), 4)
    feature_path, label_path = tmp_path / "features.f32", tmp_path / "labels.i64"
    features.astype("<f4").tofile(feature_path)
    labels.astype("<i8").tofile(label_path)
    dataset = {"features_path": str(feature_path), "labels_path": str(label_path), "rows": len(labels),
               "groups": ["source-a"] * 16 + ["source-b"] * 16, "feature_dtype": "F32"}
    return {"shape": shape, "head_path": str(head), "dataset": dataset, "device": "cpu", "seed": 7,
            "learning_rate": 0.03, "batch_size": 8, "steps": 4,
            "parent_head_sha256": file_hash(head)}


def test_real_gradient_updates_exported_head_and_improves_loss_without_touching_parent(tmp_path, head_data):
    before_hash = file_hash(head_data["head_path"])
    baseline = evaluate_head(head_data)["datasets"]["evaluation"]
    result = run({**head_data, "mode": "train", "output_dir": str(tmp_path / "trained")})
    candidate = evaluate_head({**head_data, "head_path": result["head_path"]})["datasets"]["evaluation"]
    assert result["segment_updates"] == result["successful_updates"] == 4
    assert result["changed_elements"] > 0 and result["delta_l2"] > 0 and result["delta_max_abs"] > 0
    assert result["after_head_sha256"] != result["before_head_sha256"] == before_hash
    assert file_hash(head_data["head_path"]) == before_hash
    assert candidate["loss"] < baseline["loss"]
    assert set(candidate["group_losses"]) == {"source-a", "source-b"}
    committed = json.loads((tmp_path / "trained/result.json").read_text())
    assert committed["after_head_sha256"] == result["after_head_sha256"]
    assert result["training_backend"] == "native_frozen_features_existing_head_cpu_adam"
    checkpoint = Path(result["checkpoint_path"])
    state = json.loads((checkpoint / "optimizer.json").read_text())
    assert state["successful_updates"] == 4
    assert all(file_hash(checkpoint / name) == digest for name, digest in state["hashes"].items())


def test_native_precision_trains_with_full_activations_at_the_exported_bf16_point(tmp_path, head_data, monkeypatch):
    from lucifer.head_pack import bf16_to_floats
    shape = head_data["shape"]
    weights = np.linspace(-0.4, 0.4, np.prod(shape), dtype=np.float32).reshape(shape)
    Path(head_data["head_path"]).write_bytes(floats_to_bf16(weights).tobytes())
    features = np.tile(np.linspace(-0.703, 1.001, shape[1], dtype=np.float32), (32, 1))
    features.tofile(head_data["dataset"]["features_path"])
    np.zeros(32, dtype=np.int64).tofile(head_data["dataset"]["labels_path"])
    actual_head = torch.from_numpy(bf16_to_floats(np.fromfile(head_data["head_path"], dtype="<u2")).reshape(shape))
    x = torch.from_numpy(features[:8])
    y = torch.zeros(8, dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(torch.nn.functional.linear(x, actual_head), y).item()
    legacy = torch.nn.functional.cross_entropy(torch.nn.functional.linear(x.bfloat16(), actual_head.bfloat16()).float(), y).item()
    assert abs(expected - legacy) > 1e-5  # The fixture detects the old objective.
    observations = []
    original_linear = torch.nn.functional.linear
    def observe(inputs, weight, *args, **kwargs):
        observations.append((inputs.dtype, weight.dtype, torch.equal(weight, weight.bfloat16().float()), weight.grad is None))
        return original_linear(inputs, weight, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "linear", observe)
    config = {**head_data, "steps": 1, "accumulation": 2, "compute_precision": "native_f32",
              "parent_head_sha256": file_hash(head_data["head_path"]), "output_dir": str(tmp_path / "native-f32")}
    result = train_segment(config)
    assert len(observations) == 2 and all(row == (torch.float32, torch.float32, True, True) for row in observations)
    assert result["losses"][0] == pytest.approx(expected, abs=1e-7)
    assert result["activation_dtype"] == result["logit_dtype"] == "F32"
    assert result["export_dtype"] == "BF16" and result["changed_elements"] > 0
    assert result["gradient_accumulation"] == "cpu_f32"
    state = json.loads((Path(result["checkpoint_path"]) / "optimizer.json").read_text())
    assert state["compute_precision"] == "native_f32"
    with pytest.raises(ValueError, match="precision changed"):
        train_segment({**config, "head_path": result["head_path"], "resume": result["checkpoint_path"],
                       "compute_precision": "bf16", "output_dir": str(tmp_path / "wrong-precision")})


def test_distribution_retention_matches_independent_uniform_reference_kl(tmp_path, head_data):
    from lucifer.head_pack import bf16_to_floats
    trained = train_segment({**head_data, "output_dir": str(tmp_path / "train-kl")})
    config = {**head_data, "head_path": trained["head_path"],
              "reference_head_path": head_data["head_path"], "reference_head_sha256": file_hash(head_data["head_path"]),
              "datasets": {"development": head_data["dataset"]}}
    measured = evaluate_head(config)
    kl = measured["datasets"]["development"]["forward_kl"]
    weight = bf16_to_floats(np.fromfile(trained["head_path"], dtype="<u2")).reshape(head_data["shape"])
    features = np.fromfile(head_data["dataset"]["features_path"], dtype="<f4").reshape(32, 16)
    logits = (features @ weight.T).astype(np.float64)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    expected = (-np.log(8) - logp).mean(axis=1)
    assert kl["mean"] == pytest.approx(expected.mean(), abs=1e-8)
    assert kl["maximum"] == pytest.approx(expected.max(), abs=1e-8)
    assert kl["tokens"] == 32 and kl["mean"] > 0
    unchanged = evaluate_head({**config, "head_path": head_data["head_path"]})
    assert unchanged["datasets"]["development"]["forward_kl"]["maximum"] < 1e-12
    with pytest.raises(ValueError, match="hash changed"):
        evaluate_head({**config, "reference_head_sha256": "0" * 64})


def test_native_precision_cpu_accumulation_resumes_without_changing_the_update(tmp_path, head_data):
    config = {**head_data, "compute_precision": "native_f32", "accumulation": 3, "batch_size": 7}
    continuous = train_segment({**config, "steps": 2, "output_dir": str(tmp_path / "continuous-f32")})
    first = train_segment({**config, "steps": 1, "output_dir": str(tmp_path / "first-f32")})
    resumed = train_segment({**config, "steps": 1, "resume": first["checkpoint_path"],
        "head_path": first["head_path"], "output_dir": str(tmp_path / "resumed-f32")})
    assert resumed["successful_updates"] == continuous["successful_updates"] == 2
    assert resumed["segment_updates"] == 1
    assert resumed["after_head_sha256"] == continuous["after_head_sha256"]
    for name in ("master.f32", "moment1.f32", "moment2.f32"):
        assert file_hash(Path(resumed["checkpoint_path"]) / name) == file_hash(Path(continuous["checkpoint_path"]) / name)


def test_unchanged_bf16_probe_is_measured_but_cannot_pass_as_retained_training(tmp_path, head_data):
    config = {**head_data, "steps": 1, "learning_rate": 1e-45}
    with pytest.raises(ValueError, match="did not change"):
        train_segment({**config, "output_dir": str(tmp_path / "retained")})
    result = train_segment({**config, "calibration_trial": True, "output_dir": str(tmp_path / "probe")})
    assert result["segment_updates"] == 1 and result["changed_elements"] == 0
    assert result["before_head_sha256"] == result["after_head_sha256"]


@pytest.mark.asyncio
async def test_disposable_rate_search_uses_real_cpu_updates_and_development_losses(tmp_path, head_data, monkeypatch):
    from test_meditation_service import harness, seed_run
    from lucifer.meditation_calibration import calibrate_initial_rate
    from lucifer.meditation_policy import MeditationPolicy
    service, store, _ = harness(tmp_path)
    run_record = seed_run(service, store)
    initial = file_hash(head_data["head_path"])
    baseline = evaluate_head(head_data)["datasets"]["evaluation"]["loss"]
    async def worker(ident, config):
        return run({**config, "device": "cpu"})
    monkeypatch.setattr(service, "_worker", worker)
    try:
        policy = MeditationPolicy()
        report = await calibrate_initial_rate(service, run_record["id"], policy,
            exported={"head_path": head_data["head_path"], "bf16_head_sha256": initial, "shape": head_data["shape"]},
            train=head_data["dataset"], development=head_data["dataset"], baseline_loss=baseline,
            feature_contract="tiny-numerical-fixture", run_dir=service._run_dir(run_record["id"]), seed=7)
        assert report["selected_learning_rate"] == 2e-5
        assert all(row["development_loss"] < baseline for row in report["trials"])
        assert all(row["head_changed"] for row in report["trials"])
        assert report["total_successful_updates"] == 15 and policy.successful_updates == 0
        assert file_hash(head_data["head_path"]) == initial
        assert all(not Path(row["checkpoint_path"]).exists() for row in report["trials"])
    finally:
        await service.close()
        store.close()


def test_resume_matches_uninterrupted_optimizer_and_preserves_prior_checkpoint(tmp_path, head_data):
    first = train_segment({**head_data, "steps": 2, "output_dir": str(tmp_path / "first")})
    checkpoint_hash = file_hash(Path(first["checkpoint_path"]) / "master.f32")
    resumed = train_segment({**head_data, "head_path": first["head_path"], "steps": 2,
                             "resume": first["checkpoint_path"], "output_dir": str(tmp_path / "resumed")})
    uninterrupted = train_segment({**head_data, "output_dir": str(tmp_path / "uninterrupted")})
    assert resumed["successful_updates"] == uninterrupted["successful_updates"] == 4
    assert resumed["segment_updates"] == 2
    assert resumed["after_head_sha256"] == uninterrupted["after_head_sha256"]
    assert file_hash(Path(first["checkpoint_path"]) / "master.f32") == checkpoint_hash


def test_corrupt_resume_is_rejected_before_new_candidate(tmp_path, head_data):
    first = train_segment({**head_data, "steps": 1, "output_dir": str(tmp_path / "first")})
    path = Path(first["checkpoint_path"]) / "moment1.f32"
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="Corrupt"):
        train_segment({**head_data, "resume": first["checkpoint_path"], "output_dir": str(tmp_path / "resumed")})
    assert not (tmp_path / "resumed/checkpoint").exists()


def test_nonfinite_features_cannot_produce_a_candidate(tmp_path, head_data):
    path = Path(head_data["dataset"]["features_path"])
    values = np.fromfile(path, dtype="<f4")
    values[:] = float("nan")
    values.tofile(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        train_segment({**head_data, "output_dir": str(tmp_path / "failed")})
    assert not (tmp_path / "failed/checkpoint").exists()


def test_admission_limits_fail_before_optimizer_allocation(tmp_path, head_data):
    with pytest.raises(ResourceLimitError, match="RAM"):
        train_segment({**head_data, "ram_limit_bytes": 1, "output_dir": str(tmp_path / "failed")})


def test_changed_dataset_requires_explicit_resume_refresh(tmp_path, head_data):
    first = train_segment({**head_data, "steps": 1, "output_dir": str(tmp_path / "first")})
    path = Path(head_data["dataset"]["features_path"])
    values = np.fromfile(path, dtype="<f4")
    values[0] += 0.1
    values.tofile(path)
    with pytest.raises(ValueError, match="dataset changed"):
        train_segment({**head_data, "resume": first["checkpoint_path"], "output_dir": str(tmp_path / "failed")})
    refreshed = train_segment({**head_data, "steps": 1, "resume": first["checkpoint_path"], "allow_data_refresh": True,
                               "output_dir": str(tmp_path / "refreshed")})
    assert refreshed["successful_updates"] == 2


def test_group_loss_clipping_and_invalid_labels(tmp_path, head_data):
    baseline = evaluate_head(head_data)["datasets"]["evaluation"]
    assert baseline["loss"] == pytest.approx(np.log(8), rel=1e-6)
    assert baseline["clipped_loss"] == baseline["loss"]
    np.array([-1] * 32, dtype="<i8").tofile(head_data["dataset"]["labels_path"])
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate_head(head_data)


def test_prepare_windows_supervised_targets_and_rejects_cross_partition_groups(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from lucifer.meditation_worker import tokenize_datasets
    class Tokenizer:
        chat_template = "fixture exact assistant-prefix template"
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return [0, 1] if add_generation_prompt else [0, 1, 2, 3]
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    data = tmp_path / "dataset.json"
    datasets = {part: [{"messages": [{"role": "user", "content": "reviewed question"},
                                    {"role": "assistant", "content": "reviewed answer"}], "group": part}]
                for part in ("train", "development", "holdout")}
    data.write_text(json.dumps({"datasets": datasets}))
    config = {"dataset_path": str(data), "tokenizer_path": str(tokenizer), "max_seq_length": 4}
    rows, receipt = tokenize_datasets(config)
    assert rows["train"][0]["labels"] == [-100, -100, 2, 3]
    assert receipt["dataset_sha256"] == file_hash(data)
    windowed, receipt = tokenize_datasets({**config, "max_seq_length": 3})
    assert [label for segment in windowed["train"] for label in segment["labels"] if label != -100] == [2, 3]
    assert all(len(segment["ids"]) <= 3 for segment in windowed["train"])
    assert receipt["supervised_long_context_policy"] == "assistant_windows_with_recent_context_and_unique_targets"
    datasets["development"][0]["group"] = "train"
    data.write_text(json.dumps({"datasets": datasets}))
    with pytest.raises(ValueError, match="cross"):
        tokenize_datasets(config)


def test_unsupervised_text_windows_cover_each_target_once_without_fake_answers(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from lucifer.meditation_worker import tokenize_datasets
    class Tokenizer:
        chat_template = "unused for unsupervised source text"
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens
            return list(range(20))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    data = tmp_path / "dataset.json"
    datasets = {part: [{"text": "consented public source text", "group": part, "fingerprint": part}]
                for part in ("train", "development", "holdout")}
    data.write_text(json.dumps({"datasets": datasets}))
    rows, receipt = tokenize_datasets({"dataset_path": str(data), "tokenizer_path": str(tokenizer),
                                      "training_mode": "unsupervised", "max_seq_length": 8})
    targets = [target for segment in rows["train"] for target in segment["labels"] if target != -100]
    assert targets == list(range(1, 20))
    assert all(len(segment["ids"]) <= 8 and segment["group"] == "train" for segment in rows["train"])
    assert receipt["training_mode"] == "unsupervised" and receipt["window_context_overlap"] == 7
    assert all("messages" not in segment for segment in rows["train"])


def test_native_verification_rejects_loader_that_ignores_candidate_pointer(tmp_path, monkeypatch):
    """A fixture loader proves the verification contract, not live GPU support."""
    from lucifer import head_capture
    from lucifer.head_pack import (load_pack, read_tensor, dequantize_q5_k,
                                   bf16_to_floats, tensor_hash)
    from lucifer.meditation_worker import verify_native_candidate
    from test_head_pack import fixture_pack
    parent = load_pack(fixture_pack(tmp_path))
    old_hash = tensor_hash(parent)
    original = dequantize_q5_k(read_tensor(parent, "head.weight"), parent.head.shape)
    changed = original.copy()
    changed[0] += 0.125
    head = tmp_path / "updated.bf16"
    head.write_bytes(floats_to_bf16(changed).tobytes())
    ignores_pointer = {"value": False}
    def fixture_capture(config):
        pack = load_pack(config["pack"])
        output = Path(config["output_dir"])
        output.mkdir(parents=True)
        if pack.head.dtype == "Q5_K" or ignores_pointer["value"]:
            weight = original
        else:
            weight = bf16_to_floats(np.fromfile(head, dtype="<u2")).reshape(parent.head.shape)
        features = np.ones((1, 256), dtype="<f4")
        feature_path = output / "features.f32"
        features.tofile(feature_path)
        (features @ weight.T).astype("<f4").tofile(output / "native-logits.f32")
        return {"output_dir": str(output), "features_path": str(feature_path), "rows": 1,
                "native_rows": 1, "width": 256, "row_map": [{"native_row": 0}], "engine_sha256": "fixture", "observed_engine_feed_unit": 16,
                "execution_plan": head_capture.controlled_execution_plan(config),
                "execution_plan_sha256": head_capture.execution_plan_hash(head_capture.controlled_execution_plan(config))}
    monkeypatch.setattr(head_capture, "capture_native", fixture_capture)
    config = {"pack": str(parent.path), "head_path": str(head), "engine": "fixture", "diagnostic_ids": [0, 1]}
    result = verify_native_candidate({**config, "output_dir": str(tmp_path / "valid")})
    assert result["passed"] and result["rollback_probe_passed"] and result["original_head_unchanged"]
    assert result["logits_changed_elements"] and result["projection_parity"]["passed"]
    assert not result["activation_performed"] and not result["quality_admission_implied"]
    assert tensor_hash(parent) == old_hash
    ignores_pointer["value"] = True
    with pytest.raises(ValueError, match="does not match"):
        verify_native_candidate({**config, "output_dir": str(tmp_path / "ignored-pointer")})


def test_evaluation_preserves_native_f32_activations_and_logits(tmp_path, head_data):
    """BF16 optimizer rounding must not become an admission improvement."""
    weights = np.zeros(head_data["shape"], dtype=np.float32)
    weights[0, 0] = 64.0
    Path(head_data["head_path"]).write_bytes(floats_to_bf16(weights).tobytes())
    features = np.zeros((32, 16), dtype="<f4")
    features[:, 0] = 1.00390625  # exact F32 halfway between adjacent BF16 values
    features.tofile(head_data["dataset"]["features_path"])
    np.ones(32, dtype="<i8").tofile(head_data["dataset"]["labels_path"])
    measured = evaluate_head(head_data)
    assert measured["datasets"]["evaluation"]["loss"] == pytest.approx(64.25, abs=1e-5)
    assert measured["datasets"]["evaluation"]["clipped_loss"] == 20.0
    assert measured["activation_dtype"] == measured["logit_dtype"] == "F32" and measured["tf32"] is False


def test_installed_primary_tokenizer_matches_runtime_without_hf_metadata(tmp_path, monkeypatch):
    """Optional local integration: real tokenizer assets, no model/GPU calls."""
    import importlib
    from types import SimpleNamespace, MethodType
    from lucifer.meditation_worker import load_primary_tokenizer, tokenize_datasets
    project = Path(__file__).resolve().parents[1]
    profile_path = project / "var/strata/lucifer-profile.json"
    if not profile_path.is_file():
        pytest.skip("Installed native primary profile is unavailable on this machine")
    profile = json.loads(profile_path.read_text())
    directory = Path(profile["tokenizer"])
    runtime = Path(profile["cwd"])
    if not (directory / "vocab.json").is_file() or not (runtime / "serve/server.py").is_file():
        pytest.skip("Installed native tokenizer references are unavailable")
    config = {"tokenizer_path": str(directory), "runtime_profile": str(profile_path)}
    tokenizer = load_primary_tokenizer(config)
    assert tokenizer.provenance["backend"] == "installed_strata_reference"
    for text in ("Hello Lucifer.", "Italiano: è già pronto. 中文1234567 😀", "Quoted <think> and <｜User｜>."):
        assert tokenizer.encode(text, add_special_tokens=True) == [0] + tokenizer.native.encode(text, parse_special=False)
    monkeypatch.syspath_prepend(str(runtime))
    monkeypatch.syspath_prepend(str(runtime / "tools"))
    server = importlib.import_module("serve.server")
    svc = SimpleNamespace(tok=tokenizer.native, template=tokenizer.template, prompts=None,
                          literals=tokenizer.literal_tags, effort_end=False)
    svc.render_prompt = MethodType(server.Service.render_prompt, svc)
    messages = [{"role": "user", "content": "Quote <think> and <｜User｜>, then answer 2 + 3."},
                {"role": "assistant", "content": "5"}]
    for subset, generation in ((messages[:-1], True), (messages, False)):
        actual = tokenizer.apply_chat_template(subset, add_generation_prompt=generation)
        runtime_ids = server.Service.encode_prompt(svc, subset, None,
            {"add_generation_prompt": generation, "enable_thinking": False}, store=False)
        assert actual == runtime_ids
    for mode in ("supervised", "unsupervised"):
        data = tmp_path / f"{mode}.json"
        row = {"text": "Public tokenizer smoke test. 中文 😀"} if mode == "unsupervised" else {"messages": messages}
        data.write_text(json.dumps({"datasets": {part: [{**row, "group": part}]
                                                 for part in ("train", "development", "holdout")}}))
        datasets, receipt = tokenize_datasets({**config, "dataset_path": str(data), "training_mode": mode})
        assert all(datasets.values()) and receipt["tokenizer_backend"]["reference_sha256"]
        assert any(x != -100 for x in datasets["train"][0]["labels"])
