"""Isolated native-head export, capture, evaluation and optimizer worker.

Run with the dedicated training Python. Imports never allocate GPU memory.
The primary architecture stays inside its native inference process; this worker
optimizes only the existing independent output matrix on frozen native features.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys

from .head_pack import (atomic_json, create_head_pack, export_head_bf16,
                        extract_head, file_hash, load_pack)


class ResourceLimitError(RuntimeError):
    def __init__(self, message, *, resource=None):
        super().__init__(message)
        self.resource = resource


def _event(**value):
    print(json.dumps({"meditation": True, **value}, allow_nan=False), flush=True)


def _output(config):
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    if (output / "result.json").exists():
        raise FileExistsError("Worker output already committed")
    return output


def _private_bytes(path, chunks):
    path = Path(path)
    with path.open("xb") as handle:
        path.chmod(0o600)
        for chunk in chunks:
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def export_artifacts(config):
    output = _output(config)
    pack = load_pack(config["pack"])
    original = extract_head(pack, output / "parent-head.raw")
    converted = export_head_bf16(pack, output / "baseline-head.bf16")
    if converted["source_head_sha256"] != original["sha256"]:
        raise ValueError("Parent head changed during export")
    original_pack = create_head_pack(pack, output / "no-op-pack", original["path"], dtype=original["dtype"])
    bf16_pack = create_head_pack(pack, output / "bf16-pack", converted["path"])
    return {"schema_version": 1, "mode": "export", "head_path": converted["path"],
            "shape": list(pack.head.shape), "parent_head_sha256": original["sha256"],
            "bf16_head_sha256": converted["sha256"], "parent_manifest_sha256": pack.manifest_sha256,
            "no_op_pack_path": original_pack["path"], "bf16_pack_path": bf16_pack["path"],
            "parent_head_path": original["path"], "pack_path": str(pack.path),
            "native_loader_validated": False, "training_started": False}


def _reference_module(path, label):
    """Load the locally installed, hash-bound runtime reference without a server."""
    path = Path(path).resolve()
    name = "_lucifer_meditation_" + label + "_" + file_hash(path)[:16]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Installed native tokenizer reference is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class NativePackTokenizer:
    """The deployed Strata tokenizer and template, not a guessed HF variant.

    Strata's tokenizer.json is a tiny pre-tokenizer/special-id config, while its
    vocab/merges/types live separately. It is not a tokenizers library JSON.
    """

    def __init__(self, directory, runtime_root):
        directory, runtime_root = Path(directory), Path(runtime_root).resolve()
        tokenizer_source = runtime_root / "tools/strata_tokenizer.py"
        frontend_source = runtime_root / "serve/frontend.py"
        reference = _reference_module(tokenizer_source, "tokenizer")
        self.frontend = _reference_module(frontend_source, "frontend")
        vocab = json.loads((directory / "vocab.json").read_text())
        if (not isinstance(vocab, dict) or not vocab or any(type(i) is not int for i in vocab.values()) or
                sorted(vocab.values()) != list(range(len(vocab)))):
            raise ValueError("Native tokenizer ids must form a complete unique vocabulary")
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        types = json.loads((directory / "token_type.json").read_text())
        if not isinstance(types, list) or len(types) != len(tokens):
            raise ValueError("Native tokenizer token types disagree with vocabulary")
        settings = json.loads((directory / "tokenizer.json").read_text())
        self.native = reference.Tokenizer(tokens, (directory / "merges.txt").read_text().split("\n"), types,
                                          settings["pre"], settings.get("special_ids"))
        self.template = self.frontend.ChatTemplate(directory / "chat_template.jinja")
        self.chat_template = self.template.source
        self.literal_tags = self.frontend.literal_tags(self.native.control_tokens)
        self.provenance = {"backend": "installed_strata_reference", "pre": settings["pre"],
                           "special_ids": self.native.special_ids,
                           "reference_source": str(tokenizer_source), "reference_sha256": file_hash(tokenizer_source),
                           "frontend_source": str(frontend_source), "frontend_sha256": file_hash(frontend_source),
                           "unsupervised_control_literals": "parse_special_false",
                           "unsupervised_add_bos": True, "unsupervised_add_eos": False,
                           "chat_template_kwargs": {"enable_thinking": False}}

    def encode(self, text, *, add_special_tokens=True):
        ids = self.native.encode(text, parse_special=False)
        if add_special_tokens:
            bos = self.native.special_ids.get("bos")
            if type(bos) is not int or not 0 <= bos < len(self.native.tokens):
                raise ValueError("Native tokenizer lacks a valid BOS policy")
            ids = [bos] + ids
        return ids

    def apply_chat_template(self, messages, *, tokenize=True, return_dict=False,
                            add_generation_prompt=False, enable_thinking=False):
        if not tokenize or return_dict or enable_thinking:
            raise ValueError("Native training requires token ids and thinking disabled")
        # Same escaping/template/tokenization as Service.encode_prompt. Only
        # template-authored controls are controls; document/message literals
        # remain ordinary text, including quoted thinking delimiters.
        marked, tools, changed = self.frontend.mark_think_literals(messages, None, self.literal_tags)
        rendered = self.template.render(marked, tools=tools, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False)
        if changed:
            rendered, plain = self.frontend.unmark_think_literals(rendered, self.literal_tags)
            return self.native.encode(rendered, parse_special=True, plain=plain)
        return self.native.encode(rendered, parse_special=True)


def load_primary_tokenizer(config):
    directory = Path(config["tokenizer_path"]).resolve()
    settings_path = directory / "tokenizer.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    if isinstance(settings, dict) and "pre" in settings and "model" not in settings:
        runtime_root = config.get("runtime_root")
        if not runtime_root and config.get("runtime_profile"):
            runtime_root = json.loads(Path(config["runtime_profile"]).read_text()).get("cwd")
        runtime_root = runtime_root or Path(__file__).resolve().parents[1] / "var/strata/data/terminal/current"
        return NativePackTokenizer(directory, runtime_root)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def tokenize_datasets(config):
    path = Path(config["dataset_path"])
    if config.get("dataset_sha256") and file_hash(path) != config["dataset_sha256"]:
        raise ValueError("Immutable dataset changed before tokenization")
    if path.stat().st_size > 128 << 20:
        raise ValueError("Dataset exceeds bounded preparation input")
    raw = json.loads(path.read_text())
    datasets = raw.get("datasets", raw)
    tokenizer_path = Path(config["tokenizer_path"]).resolve()
    if not tokenizer_path.is_dir():
        raise ValueError("A pinned local tokenizer directory is required")
    tokenizer_hashes = {p.name: file_hash(p) for p in tokenizer_path.iterdir() if p.is_file()}
    if config.get("tokenizer_files") is not None and tokenizer_hashes != config["tokenizer_files"]:
        raise ValueError("Pinned tokenizer files changed before tokenization")
    for filename, expected in config.get("tokenizer_modules", {}).items():
        if file_hash(filename) != expected:
            raise ValueError("Pinned native tokenizer implementation changed")
    tokenizer = load_primary_tokenizer(config)
    limit = int(config.get("max_seq_length", 512))
    if limit < 2 or limit > 32768:
        raise ValueError("Invalid teacher-forced sequence limit")
    result = {}
    training_mode = config.get("training_mode", "supervised")
    if training_mode not in {"supervised", "unsupervised"}:
        raise ValueError("Unknown Meditation training mode")
    for partition in ("train", "development", "holdout"):
        rows = datasets.get(partition)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Missing independent {partition} examples")
        segments = []
        for index, row in enumerate(rows):
            group = row.get("group") or row.get("meditation_group_id")
            if not group:
                raise ValueError("Every example needs an independent source group")
            fingerprint = str(row.get("fingerprint", f"{partition}-{index}"))
            if training_mode == "unsupervised":
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Unsupervised Meditation requires consented source text")
                ids = tokenizer.encode(text, add_special_tokens=True)
                if len(ids) < 2:
                    raise ValueError("Unsupervised source has no next-token targets")
                # Context overlap is explicit; every source next-token target
                # occurs exactly once despite bounded native capture windows.
                overlap = min(32, limit - 1)
                start, window_index = 0, 0
                while start < len(ids) - 1:
                    window = ids[start:start + limit]
                    prefix = 1 if start == 0 else overlap
                    if len(window) <= prefix:
                        break
                    labels = [-100] * prefix + window[prefix:]
                    segments.append({"name": f"{fingerprint}:{window_index}", "group": str(group),
                                     "ids": window, "labels": labels, "source_token_start": start,
                                     "source_target_start": start + prefix})
                    if start + len(window) >= len(ids):
                        break
                    start += limit - overlap
                    window_index += 1
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                raise ValueError("Supervised examples must end with a final assistant target")
            if any(set(m) - {"role", "content"} for m in messages):
                raise ValueError("Training messages contain unsupported channels/metadata")
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                 add_generation_prompt=False, enable_thinking=False)
            if full[:len(prompt)] != prompt:
                raise ValueError("Primary chat template has no exact assistant prefix")
            if len(full) <= len(prompt):
                raise ValueError("Reviewed example has no final assistant tokens")
            labels = [-100] * len(prompt) + full[len(prompt):]
            if len(full) <= limit:
                segments.append({"name": fingerprint, "group": str(group), "ids": full, "labels": labels})
            else:
                overlap = min(32, limit - 1)
                start, window_index = max(0, len(prompt) - overlap), 0
                while start < len(full) - 1:
                    window = full[start:start + limit]
                    window_labels = labels[start:start + limit]
                    prefix = 1 if window_index == 0 else overlap
                    window_labels[:prefix] = [-100] * prefix
                    if any(label != -100 for label in window_labels):
                        segments.append({"name": f"{fingerprint}:{window_index}", "group": str(group),
                            "ids": window, "labels": window_labels, "source_token_start": start,
                            "source_target_start": start + prefix})
                    if start + len(window) >= len(full):
                        break
                    start += limit - overlap
                    window_index += 1
        result[partition] = segments
    groups = {name: {x["group"] for x in segments} for name, segments in result.items()}
    for index, name in enumerate(groups):
        for other in list(groups)[index + 1:]:
            if groups[name] & groups[other]:
                raise ValueError("Source groups cross training/evaluation partitions")
    if {p.name: file_hash(p) for p in tokenizer_path.iterdir() if p.is_file()} != tokenizer_hashes:
        raise ValueError("Tokenizer files changed during tokenization")
    return result, {"tokenizer_path": str(tokenizer_path), "files": tokenizer_hashes,
                    "dataset_sha256": file_hash(path), "template_hash": hashlib.sha256(
                        str(tokenizer.chat_template).encode()).hexdigest(), "training_mode": training_mode,
                    "text_special_tokens": True if training_mode == "unsupervised" else None,
                    "window_context_overlap": min(32, limit - 1),
                    "supervised_long_context_policy": "assistant_windows_with_recent_context_and_unique_targets" if training_mode == "supervised" else None,
                    "tokenizer_backend": getattr(tokenizer, "provenance", {"backend": "local_transformers"})}


def _native_gate(config, exports, segment, output):
    import numpy as np
    from .head_capture import capture_native, compare_native_logits
    # Diagnostic labels deliberately score every position of a short real-token
    # prefix; never substitute these diagnostic rows for reviewed training data.
    ids = segment["ids"][:int(config.get("parity_tokens", 24))]
    probe = {"ids": ids, "diagnostic": True, "engine": config["engine"], "dump_logits": True,
             "reserve_mib": config.get("reserve_mib", 2048), "env": config.get("env", {}),
             "cpu_threads": config.get("cpu_threads", 12), "experts": config.get("experts", 6),
             "disk_limit_bytes": config.get("disk_limit_bytes", 50_000_000_000)}
    captures = {}
    for name, pack in (("parent", config["pack"]), ("no_op", exports["no_op_pack_path"]),
                       ("bf16", exports["bf16_pack_path"])):
        _event(stage="native_parity", variant=name)
        captures[name] = capture_native({**probe, "pack": pack, "output_dir": str(output / name)})
    shape = (len(ids) - 1, exports["shape"][0])
    parent = np.memmap(output / "parent/native-logits.f32", dtype="<f4", mode="r", shape=shape)
    no_op = np.memmap(output / "no_op/native-logits.f32", dtype="<f4", mode="r", shape=shape)
    bf16 = np.memmap(output / "bf16/native-logits.f32", dtype="<f4", mode="r", shape=shape)
    identical = np.array_equal(parent, no_op)
    max_no_op = float(np.abs(parent - no_op).max())
    if not identical and not np.allclose(parent, no_op, atol=1e-5, rtol=1e-6):
        raise ValueError("Byte-identical native sidecar failed logits parity")
    features = {name: np.memmap(value["features_path"], dtype="<f4", mode="r",
                                shape=(value["rows"], value["width"])) for name, value in captures.items()}
    feature_identity = bool(np.array_equal(features["parent"], features["bf16"]))
    plans = {value["execution_plan_sha256"] for value in captures.values()}
    execution_parity = {"passed": feature_identity and len(plans) == 1,
                        "parent_bf16_features_identical": feature_identity,
                        "parent_bf16_feature_max_abs_delta": float(np.abs(features["parent"] - features["bf16"]).max()),
                        "execution_plan_sha256": captures["parent"]["execution_plan_sha256"],
                        "execution_plan": captures["parent"]["execution_plan"]}
    atomic_json(output / "execution-parity.json", execution_parity)
    if not execution_parity["passed"]:
        raise ValueError("Controlled parent/BF16 backbone features are not identical")
    parity = compare_native_logits(load_pack(exports["bf16_pack_path"]), captures["bf16"])
    if not parity["passed"]:
        raise ValueError("Reconstructed final native features failed logits parity")
    # Conversion drift is measured separately; it is not a learned improvement.
    def distribution(values):
        values = np.asarray(values, dtype=np.float64)
        shifted = values - values.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    p, q = distribution(parent), distribution(bf16)
    kl = np.sum(np.exp(p) * (p - q), axis=1)
    targets = np.asarray(ids[1:], dtype=np.int64)
    loss_drift = float((-q[np.arange(len(targets)), targets]).mean() -
                       (-p[np.arange(len(targets)), targets]).mean())
    if not np.isfinite(kl).all() or float(kl.max()) > 0.02 or abs(loss_drift) > 0.05:
        raise ValueError("Untrained BF16 conversion exceeds the fixed numerical drift envelope")
    receipt = {"passed": True, "native_loader_validated": True, "no_op_bit_identical": identical,
               "no_op_max_abs_error": max_no_op, "feature_parity": parity,
               "conversion_kl_mean": float(kl.mean()), "conversion_kl_max": float(kl.max()),
               "conversion_nll_drift": loss_drift, "conversion_kl_max_limit": 0.02,
               "conversion_nll_drift_limit": 0.05, "diagnostic_tokens": len(ids),
               "diagnostic_ids": ids,
               "execution_parity": execution_parity, "execution_plan": captures["parent"]["execution_plan"],
               "execution_plan_sha256": captures["parent"]["execution_plan_sha256"],
               "observed_engine_feed_unit": captures["parent"]["observed_engine_feed_unit"],
               "engine_sha256": captures["parent"]["engine_sha256"]}
    atomic_json(output / "native-gates.json", receipt)
    # Full logits are transient diagnostics. Keep hashes and numerical receipts,
    # not an unbounded vocabulary-sized cache.
    for name in captures:
        for filename in ("native-logits.f32", "native-features.bin"):
            (output / name / filename).unlink(missing_ok=True)
    return receipt


def prepare(config):
    from .head_capture import capture_batched, capture_native
    output = _output(config)
    datasets, provenance = tokenize_datasets(config)
    if config.get("run_root"):
        from .meditation_storage import checkpoint_reserve, feature_reserve, require_room, StorageLimitError
        from .meditation_recovery import config_hash
        root = Path(config["run_root"]).resolve()
        if not output.is_relative_to(root):
            raise ValueError("Feature preparation is outside its owned run")
        shape = load_pack(config["pack"]).head.shape
        limit = int(config.get("disk_limit_bytes", 50_000_000_000))
        raw_limit = min(int(config.get("raw_batch_limit_bytes", 4 << 30)), limit // 4)
        reserve = feature_reserve(datasets, shape[1], raw_limit_bytes=raw_limit) + max(0, checkpoint_reserve(shape) - raw_limit)
        try:
            require_room(root, reserve_bytes=reserve, limit_bytes=limit)
        except StorageLimitError as error:
            error.preflight_rejection = {"contract": "native_feature_storage_preflight_v1",
                "config_sha256": config_hash(config), "worker_sha256": file_hash(__file__)}
            raise
    supplied = config.get("export") or config.get("export_receipt")
    if isinstance(supplied, str):
        supplied = json.loads(Path(supplied).read_text())
    exports = supplied or export_artifacts({**config, "output_dir": str(output / "export")})
    if exports["parent_manifest_sha256"] != load_pack(config["pack"]).manifest_sha256:
        raise ValueError("Prepared export belongs to another primary pack")
    if file_hash(exports["head_path"]) != exports["bf16_head_sha256"]:
        raise ValueError("Untrained BF16 head changed before native gates")
    gates = _native_gate(config, exports, datasets["train"][0], output / "gates")
    receipts, original_scores = {}, {}
    for name, segments in datasets.items():
        _event(stage="capturing", partition=name, examples=len(segments))
        original = capture_native({**config, "segments": segments, "tokens_path": None, "score_only": True,
                                   "output_dir": str(output / (name + "-original-scores")), "dump_logits": False,
                                   "provenance": provenance})
        receipt = capture_batched({**config, "pack": exports["bf16_pack_path"], "segments": segments,
                                   "tokens_path": None, "output_dir": str(output / name), "dump_logits": False,
                                   "score_only": False, "provenance": provenance})
        if receipt["execution_plan_sha256"] != gates["execution_plan_sha256"]:
            raise ValueError("BF16 feature capture changed its proven execution plan")
        receipt["count"] = receipt["rows"]
        receipt["groups"] = [row["group"] for row in receipt["row_map"]]
        receipt.update(original_native_nll=original["native_nll"],
                       original_native_group_losses=original["native_group_losses"],
                       original_native_group_clipped_losses=original["native_group_clipped_losses"],
                       original_score_reference="controlled_parent_not_deployed_default")
        receipts[name] = receipt
        original_scores[name] = original
        # Keep compact supervised features, hashes and scored receipts only.
        (output / name / "native-features.bin").unlink(missing_ok=True)
    bf16_evaluation = evaluate_head({**config, "head_path": exports["head_path"], "shape": exports["shape"],
                                    "datasets": receipts})
    return {"schema_version": 1, "mode": "prepare", "datasets": receipts, "export": exports,
            "tokenized_datasets_sha256": hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest(),
            "head_path": exports["head_path"], "shape": exports["shape"], "native_gates": gates,
            "provenance": provenance, "baseline_bf16": bf16_evaluation,
            "execution_plan": gates["execution_plan"], "execution_plan_sha256": gates["execution_plan_sha256"],
            "observed_engine_feed_unit": gates["observed_engine_feed_unit"],
            "baseline_original": {name: {"loss": value["native_nll"], "group_losses": value["native_group_losses"],
                                         "group_clipped_losses": value["native_group_clipped_losses"],
                                         "reference": "controlled_parent_not_deployed_default"}
                                  for name, value in original_scores.items()}}


def native_admission(config):
    """Score the reserved whole source windows with the actual candidate head."""
    from .head_capture import capture_native
    output = _output(config)
    prepared = config["prepared"]
    for path, expected in ((config["engine"], config["engine_sha256"]),
                           (config["runtime_profile"], config["profile_sha256"]),
                           (config["head_path"], config["head_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("Native admission input changed")
    datasets, provenance = tokenize_datasets(config)
    if provenance != prepared["provenance"] or hashlib.sha256(
            json.dumps(datasets, sort_keys=True).encode()).hexdigest() != prepared["tokenized_datasets_sha256"]:
        raise ValueError("Native admission source windows differ from preparation")
    groups = set(config["selected_groups"])
    segments = [segment for segment in datasets["holdout"] if segment["group"] in groups]
    if not groups or {segment["group"] for segment in segments} != groups:
        raise ValueError("Reserved admission groups are absent from heldout sources")
    candidate = create_head_pack(load_pack(config["pack"]), output / "candidate-pack", config["head_path"])
    if candidate["head_sha256"] != config["head_sha256"]:
        raise ValueError("Native admission pack references another head")
    contract = {"segments": segments, "groups": sorted(groups), "provenance": provenance,
                "head_sha256": config["head_sha256"], "execution_plan_sha256": config["execution_plan_sha256"]}
    atomic_json(output / "selection.json", contract)
    measured = capture_native({**config, "pack": candidate["path"], "segments": segments,
        "tokens_path": None, "score_only": True, "dump_logits": False,
        "output_dir": str(output / "scores"), "provenance": provenance})
    if measured["execution_plan_sha256"] != prepared["execution_plan_sha256"]:
        raise ValueError("Native admission execution differs from preparation")
    heldout = prepared["datasets"]["holdout"]
    candidate_losses = measured["native_group_clipped_losses"]
    if set(candidate_losses) != groups:
        raise ValueError("Native admission did not score every reserved group")
    return {"mode": "native_admission", "backend": "native_engine_teacher_forced",
            "head_sha256": config["head_sha256"], "selection_sha256": file_hash(output / "selection.json"),
            "native_receipt": measured, "group_clipped_losses": candidate_losses,
            "converted_group_clipped_losses": {g: heldout["native_group_clipped_losses"][g] for g in groups},
            "original_group_clipped_losses": {g: heldout["original_native_group_clipped_losses"][g] for g in groups
                if g in heldout.get("original_native_group_clipped_losses", {})}}


def _torch():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    return torch


def verify_native_candidate(config):
    """Load updated weights in the real engine and rehearse parent restoration."""
    import numpy as np
    from .head_capture import capture_native, compare_native_logits
    from .head_pack import tensor_hash
    output = _output(config)
    parent = load_pack(config["pack"])
    updated_hash = file_hash(config["head_path"])
    if config.get("head_sha256") and config["head_sha256"] != updated_hash:
        raise ValueError("Candidate head hash disagrees with its training receipt")
    before_hash = tensor_hash(parent)
    prepared = config.get("prepared") or config.get("prepare_receipt") or {}
    if isinstance(prepared, str):
        prepared = json.loads(Path(prepared).read_text())
    ids = config.get("diagnostic_ids") or prepared.get("native_gates", {}).get("diagnostic_ids")
    if not isinstance(ids, list) or len(ids) < 2:
        raise ValueError("Native verification requires source-bound diagnostic token ids")
    candidate = create_head_pack(parent, output / "candidate-pack", config["head_path"])
    if candidate["head_sha256"] != updated_hash:
        raise ValueError("Candidate manifest does not reference the trained tensor")
    captures = {}
    for name, pack in (("parent_before", str(parent.path)), ("candidate", candidate["path"]),
                       ("parent_restored", str(parent.path))):
        _event(stage="native_candidate_verification", variant=name)
        captures[name] = capture_native({**config, "pack": pack, "ids": ids,
                                         "segments": None, "tokens_path": None,
                                         "diagnostic": True, "dump_logits": True,
                                         "output_dir": str(output / name)})
    plans = {value["execution_plan_sha256"] for value in captures.values()}
    expected_plan = prepared.get("execution_plan_sha256")
    if len(plans) != 1 or expected_plan and expected_plan not in plans:
        raise ValueError("Native verification changed the frozen execution plan")
    shape = (len(ids) - 1, parent.head.shape[0])
    logits = {name: np.memmap(output / name / "native-logits.f32", dtype="<f4", mode="r", shape=shape)
              for name in captures}
    original, restored, updated = logits["parent_before"], logits["parent_restored"], logits["candidate"]
    if not all(np.isfinite(value).all() for value in logits.values()):
        raise ValueError("Native candidate/rollback returned nonfinite logits")
    rollback_identical = bool(np.array_equal(original, restored))
    rollback_error = float(np.abs(original - restored).max())
    if not rollback_identical and not np.allclose(original, restored, atol=1e-5, rtol=1e-6):
        raise ValueError("Native parent rollback probe failed")
    projection = compare_native_logits(load_pack(candidate["path"]), captures["candidate"])
    if not projection["passed"]:
        raise ValueError("Loaded updated head does not match frozen-feature projection")
    changed_logits = int(np.count_nonzero(original != updated))
    if not changed_logits:
        raise ValueError("Updated head produced no observable native logit change")
    after_hash = tensor_hash(parent)
    if before_hash != after_hash:
        raise ValueError("Original head changed during native candidate verification")
    receipt = {"schema_version": 1, "mode": "verify", "passed": True,
               "candidate_pack_path": candidate["path"], "candidate_pack": candidate,
               "updated_head_sha256": updated_hash, "parent_head_sha256_before": before_hash,
               "parent_head_sha256_after": after_hash, "original_head_unchanged": True,
               "projection_parity": projection, "rollback_probe_passed": True,
               "rollback_bit_identical": rollback_identical, "rollback_max_abs_error": rollback_error,
               "logits_changed_elements": changed_logits,
               "max_abs_logit_delta": float(np.abs(updated - original).max()),
               "mean_abs_logit_delta": float(np.abs(updated - original).mean()),
               "diagnostic_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               "native_logit_hashes": {name: file_hash(output / name / "native-logits.f32") for name in captures},
               "engine_sha256": captures["candidate"]["engine_sha256"],
               "execution_plan": captures["candidate"]["execution_plan"],
               "execution_plan_sha256": captures["candidate"]["execution_plan_sha256"],
               "observed_engine_feed_unit": captures["candidate"]["observed_engine_feed_unit"],
               "activation_performed": False, "quality_admission_implied": False}
    for name in captures:
        for filename in ("native-logits.f32", "native-features.bin"):
            (output / name / filename).unlink(missing_ok=True)
    return receipt


def _device(config, torch):
    requested = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("The native-head worker supports one CPU or CUDA device")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested native training CUDA device is unavailable")
    if requested == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return requested


def _resource_check(config, torch, device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    limit = int(config.get("ram_limit_bytes", 24 << 30))
    if rss > limit:
        raise ResourceLimitError("Native head worker exceeded its RAM admission limit")
    gpu = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    if gpu > int(config.get("vram_limit_bytes", 6 << 30)):
        raise ResourceLimitError("Native head worker exceeded its VRAM admission limit", resource="vram")
    return {"peak_rss_bytes": rss, "peak_cuda_allocated_bytes": gpu,
            "rss_bytes": rss, "peak_vram_bytes": gpu}


def _dataset(value, shape):
    import numpy as np
    width = shape[1]
    count = int(value.get("rows", value.get("count", 0)))
    feature_path = Path(value.get("features_path", value.get("feature_path", "")))
    labels_path = Path(value["labels_path"])
    dtype = value.get("feature_dtype", "F32")
    if count < 1 or dtype not in {"F32", "BF16"}:
        raise ValueError("Invalid frozen-feature dataset shape/dtype")
    if feature_path.stat().st_size != count * width * (4 if dtype == "F32" else 2) or labels_path.stat().st_size != count * 8:
        raise ValueError("Frozen feature/label file size disagrees with metadata")
    for path, key in ((feature_path, "features_sha256"), (labels_path, "labels_sha256")):
        if value.get(key) and file_hash(path) != value[key]:
            raise ValueError("Frozen dataset hash changed")
    features = np.memmap(feature_path, dtype="<f4" if dtype == "F32" else "<u2", mode="r", shape=(count, width))
    labels = np.memmap(labels_path, dtype="<i8", mode="r", shape=(count,))
    if int(labels.min()) < 0 or int(labels.max()) >= shape[0]:
        raise ValueError("Head target labels exceed the actual vocabulary")
    groups = value.get("groups")
    if groups is None and value.get("row_map"):
        groups = [row["group"] for row in value["row_map"]]
    groups = groups or ["unpartitioned"] * count
    if len(groups) != count:
        raise ValueError("Group vector does not match feature rows")
    return features, labels, groups, dtype


def _head(config, torch, device):
    import numpy as np
    shape = tuple(config["shape"])
    if len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("Invalid native head shape")
    path = Path(config["head_path"])
    if path.stat().st_size != math.prod(shape) * 2:
        raise ValueError("Native BF16 head size mismatch")
    if config.get("head_sha256") and file_hash(path) != config["head_sha256"]:
        raise ValueError("Native head hash changed")
    array = np.memmap(path, dtype="<u2", mode="c", shape=shape)
    value = torch.from_numpy(array).view(torch.bfloat16).to(device)
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite native head")
    return value, shape


def _batch(features, labels, indices, dtype, torch, device, *, compute_dtype=None):
    import numpy as np
    from .head_pack import bf16_to_floats
    data = np.asarray(features[indices])
    if dtype == "BF16":
        data = bf16_to_floats(data)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite frozen features")
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(
        device=device, dtype=compute_dtype or torch.bfloat16)
    y = torch.from_numpy(np.array(labels[indices], dtype=np.int64, copy=True)).to(device)
    return x, y


def evaluate_head(config):
    torch = _torch()
    device = _device(config, torch)
    # The native BF16 matrix takes F32 activations and returns F32 logits.
    # Evaluation must not inherit the optimizer's BF16 activation/output
    # rounding or TF32 approximations and turn those into apparent learning.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    batch = int(config.get("batch_size", 32))
    if batch < 1 or batch > 4096:
        raise ValueError("Invalid evaluation batch size")
    head, shape = _head(config, torch, device)
    head = head.float()
    reference = None
    reference_hash = None
    if config.get("reference_head_path"):
        reference_hash = config.get("reference_head_sha256")
        if not isinstance(reference_hash, str) or len(reference_hash) != 64:
            raise ValueError("Distribution retention needs its pinned reference head hash")
        reference, reference_shape = _head({"head_path": config["reference_head_path"], "shape": list(shape),
            "head_sha256": reference_hash}, torch, device)
        if reference_shape != shape:
            raise ValueError("Reference head shape changed")
    datasets = config.get("datasets")
    if datasets is None:
        datasets = {"evaluation": config.get("dataset", config)}
    results = {}
    with torch.no_grad():
        for name, value in datasets.items():
            features, labels, groups, dtype = _dataset(value, shape)
            losses, clipped, counts = {}, {}, {}
            kl_total, kl_maximum, kl_count = 0.0, 0.0, 0
            for start in range(0, len(labels), batch):
                stop = min(start + batch, len(labels))
                x, y = _batch(features, labels, slice(start, stop), dtype, torch, device,
                              compute_dtype=torch.float32)
                logits = torch.nn.functional.linear(x, head)
                values = torch.nn.functional.cross_entropy(logits, y, reduction="none").cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite evaluation loss")
                for offset, loss in enumerate(values):
                    group = str(groups[start + offset])
                    losses[group] = losses.get(group, 0.0) + loss
                    clipped[group] = clipped.get(group, 0.0) + min(20.0, loss)
                    counts[group] = counts.get(group, 0) + 1
                if reference is not None:
                    reference_logits = torch.empty_like(logits)
                    # Keep the reference in BF16 storage and convert bounded
                    # row chunks, avoiding a second full F32 GPU matrix.
                    for row in range(0, shape[0], 4096):
                        span = slice(row, min(row + 4096, shape[0]))
                        reference_logits[:, span] = torch.nn.functional.linear(x, reference[span].float())
                    reference_log = reference_logits.double().log_softmax(-1)
                    divergences = (reference_log.exp() * (reference_log - logits.double().log_softmax(-1))).sum(-1)
                    if not torch.isfinite(divergences).all() or float(divergences.min()) < -1e-7:
                        raise ValueError("Invalid development distribution divergence")
                    divergences = divergences.clamp_min(0)
                    kl_total += float(divergences.sum().cpu())
                    kl_maximum = max(kl_maximum, float(divergences.max().cpu()))
                    kl_count += len(values)
                    del reference_logits, reference_log, divergences
                del logits, x, y
                _resource_check(config, torch, device)
            results[name] = {"loss": sum(losses.values()) / len(labels),
                             "clipped_loss": sum(clipped.values()) / len(labels), "tokens": len(labels),
                             "group_losses": {g: losses[g] / counts[g] for g in counts},
                             "group_clipped_losses": {g: clipped[g] / counts[g] for g in counts},
                             "group_tokens": counts}
            if reference is not None:
                results[name]["forward_kl"] = {"mean": kl_total / kl_count, "maximum": kl_maximum,
                    "tokens": kl_count, "direction": "untrained_bf16_reference_to_candidate"}
    resources = _resource_check(config, torch, device)
    del head, reference
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"schema_version": 1, "mode": "evaluate", "datasets": results,
            "head_sha256": file_hash(config["head_path"]), "reference_head_sha256": reference_hash,
            "shape": list(shape), "resources": resources,
            "evaluation_backend": "frozen_native_features_fp32_projection_bf16_weights",
            "activation_dtype": "F32", "logit_dtype": "F32", "tf32": False, "device": device}


def _save_tensor(path, tensor, *, rows=128):
    return _private_bytes(path, (tensor[start:start + rows].detach().contiguous().view(-1).numpy().tobytes()
                                 for start in range(0, tensor.shape[0], rows)))


def train_segment(config):
    import numpy as np
    torch = _torch()
    device = _device(config, torch)
    precision = config.get("compute_precision", "bf16")
    if precision not in {"bf16", "native_f32"}:
        raise ValueError("Unknown native-head training precision")
    compute_dtype = torch.float32 if precision == "native_f32" else torch.bfloat16
    if precision == "native_f32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    output = _output(config)
    shape = tuple(config["shape"])
    if len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("Invalid training head shape")
    params = math.prod(shape)
    if params * 16 + (256 << 20) > int(config.get("ram_limit_bytes", 24 << 30)):
        raise ResourceLimitError("Head state exceeds RAM admission capacity")
    checkpoint_size = params * 14 + (64 << 20)
    if checkpoint_size > int(config.get("disk_limit_bytes", 50_000_000_000)):
        raise ResourceLimitError("Head checkpoint exceeds disk admission capacity")
    if shutil.disk_usage(output).free < checkpoint_size + int(config.get("disk_reserve_bytes", 1 << 30)):
        raise ResourceLimitError("Insufficient free disk for an atomic head checkpoint")
    data = config.get("dataset", config)
    features, labels, groups, feature_dtype = _dataset(data, shape)
    data_revision = hashlib.sha256((file_hash(data.get("features_path", data.get("feature_path"))) +
                                   file_hash(data["labels_path"])).encode()).hexdigest()
    steps = int(config.get("steps", 25))
    batch = int(config.get("batch_size", 32))
    accumulation = int(config.get("accumulation", 1))
    lr = float(config.get("learning_rate", 1e-4))
    max_norm = float(config.get("max_grad_norm", 1.0))
    if not 1 <= steps <= 1000 or not 1 <= batch <= 4096 or not 1 <= accumulation <= 1024:
        raise ValueError("Invalid finite segment dimensions")
    if not math.isfinite(lr) or not 0 < lr <= 0.1 or not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Invalid finite optimizer parameters")
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    initial_hash = file_hash(config["head_path"])
    expected_parent = config.get("parent_head_sha256", initial_hash)
    resume = config.get("resume") or config.get("resume_from")
    successful = 0
    cursor, order = 0, rng.permutation(len(labels))
    if resume:
        resume = Path(resume)
        state = json.loads((resume / "optimizer.json").read_text())
        if state["shape"] != list(shape) or state["parent_head_sha256"] != expected_parent:
            raise ValueError("Head checkpoint belongs to another base")
        if state.get("compute_precision", "bf16") != precision:
            raise ValueError("Checkpoint training precision changed; preserve it and start a new run")
        for name, digest in state["hashes"].items():
            if file_hash(resume / name) != digest:
                raise ValueError("Corrupt native-head optimizer checkpoint")
        if state["data_revision"] != data_revision and not config.get("allow_data_refresh", False):
            raise ValueError("Resume dataset changed without an explicit refresh")
        arrays = [np.memmap(resume / name, dtype="<f4", mode="c", shape=shape)
                  for name in ("master.f32", "moment1.f32", "moment2.f32")]
        master, moment1, moment2 = (torch.from_numpy(a) for a in arrays)
        successful = int(state["successful_updates"])
        if state["data_revision"] == data_revision:
            rng.bit_generator.state = state["rng"]
            order = np.asarray(state["order"], dtype=np.int64)
            cursor = int(state["cursor"])
            if sorted(order.tolist()) != list(range(len(labels))) or not 0 <= cursor <= len(labels):
                raise ValueError("Invalid resumed sampler state")
        weight = torch.nn.Parameter(master.to(device=device, dtype=torch.bfloat16).to(dtype=compute_dtype))
    else:
        loaded, _ = _head(config, torch, "cpu")
        master = loaded.float()
        del loaded
        moment1, moment2 = torch.zeros_like(master), torch.zeros_like(master)
        weight = torch.nn.Parameter(master.to(device=device, dtype=torch.bfloat16).to(dtype=compute_dtype))
    # A second full F32 GPU gradient during ordinary autograd accumulation would
    # exceed the pilot VRAM cap. Accumulate native-precision microgradients in
    # host RAM and release each GPU gradient before the next backward pass.
    cpu_gradient = torch.zeros_like(master) if precision == "native_f32" else None
    stop = {"requested": False}
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.update(requested=True))
    losses = []
    starting_updates = successful
    try:
        for step in range(steps):
            if stop["requested"]:
                break
            weight.grad = None
            if cpu_gradient is not None:
                cpu_gradient.zero_()
            loss_value = 0.0
            for micro in range(accumulation):
                # Complete every microbatch across permutation boundaries, so
                # halving batch/doubling accumulation preserves the actual
                # example count as well as its nominal configuration.
                portions, remaining = [], batch
                while remaining:
                    if cursor >= len(order):
                        order, cursor = rng.permutation(len(labels)), 0
                    selected = order[cursor:min(cursor + remaining, len(order))]
                    portions.append(selected)
                    cursor += len(selected)
                    remaining -= len(selected)
                indices = np.concatenate(portions)
                x, y = _batch(features, labels, indices, feature_dtype, torch, device, compute_dtype=compute_dtype)
                logits = torch.nn.functional.linear(x, weight).float()
                loss = torch.nn.functional.cross_entropy(logits, y) / accumulation
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite native-head training loss")
                loss.backward()
                loss_value += float(loss.detach().cpu())
                del logits, x, y, loss
                if cpu_gradient is not None:
                    current = weight.grad
                    if current is None:
                        raise ValueError("Native head produced no microbatch gradient")
                    for row in range(0, shape[0], 128):
                        span = slice(row, min(row + 128, shape[0]))
                        cpu_gradient[span].add_(current[span].to(device="cpu", dtype=torch.float32))
                    weight.grad = None
                    del current
            gradient = cpu_gradient if cpu_gradient is not None else weight.grad
            if gradient is None:
                raise ValueError("Native head produced no gradient")
            squared = 0.0
            for row in range(0, shape[0], 128):
                part = gradient[row:row + 128].float()
                if not torch.isfinite(part).all():
                    raise ValueError("Nonfinite native-head gradient")
                squared += float(part.square().sum().cpu())
                del part
            norm = math.sqrt(squared)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Native head gradient is zero or invalid")
            scale = min(1.0, max_norm / norm)
            successful += 1
            correction1, correction2 = 1 - beta1 ** successful, math.sqrt(1 - beta2 ** successful)
            # CPU optimizer touches bounded row chunks, avoiding full-size FP32
            # gradient and denominator copies and foreach optimizer temporaries.
            with torch.no_grad():
                for row in range(0, shape[0], 128):
                    span = slice(row, min(row + 128, shape[0]))
                    grad = gradient[span].to(device="cpu", dtype=torch.float32).mul_(scale)
                    moment1[span].mul_(beta1).add_(grad, alpha=1 - beta1)
                    moment2[span].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denominator = moment2[span].sqrt().div_(correction2).add_(epsilon)
                    master[span].addcdiv_(moment1[span], denominator, value=-lr / correction1)
                    if not torch.isfinite(master[span]).all():
                        raise ValueError("Nonfinite native head after optimizer update")
                    # The F32 shadow holds exactly the exported BF16 values,
                    # while retaining native F32 activations/logits/gradients.
                    # Optimize at the deployed quantized point, not an
                    # unrounded F32 master that serving would never read.
                    weight[span].copy_(master[span].to(device=device, dtype=torch.bfloat16).to(dtype=compute_dtype))
                    del grad, denominator
            weight.grad = None
            config["_uncommitted_successful_updates"] = successful - starting_updates
            losses.append(loss_value)
            resources = _resource_check(config, torch, device)
            _event(stage="training", successful_updates=successful, segment_updates=successful - starting_updates,
                   loss=loss_value, learning_rate=lr, gradient_norm=norm, **resources)
        if successful == starting_updates:
            raise ValueError("No successful optimizer updates were completed")
        checkpoint = output / "checkpoint"
        checkpoint.mkdir(mode=0o700)
        _save_tensor(checkpoint / "master.f32", master)
        _save_tensor(checkpoint / "moment1.f32", moment1)
        _save_tensor(checkpoint / "moment2.f32", moment2)
        # Export the actual BF16 values that serving will read; an FP32 master
        # update that rounds back to identical BF16 is not weight-update proof.
        candidate_path = checkpoint / "head.bf16"
        _private_bytes(candidate_path, (weight[row:row + 128].detach().to(device="cpu", dtype=torch.bfloat16).view(torch.uint16).numpy().tobytes()
                                        for row in range(0, shape[0], 128)))
        import numpy as np
        before = np.memmap(config["head_path"], dtype="<u2", mode="r", shape=shape)
        after = np.memmap(candidate_path, dtype="<u2", mode="r", shape=shape)
        from .head_pack import bf16_to_floats
        changed, delta_square, max_delta = 0, 0.0, 0.0
        for row in range(0, shape[0], 128):
            a, b = before[row:row + 128], after[row:row + 128]
            changed += int(np.count_nonzero(a != b))
            delta = bf16_to_floats(b) - bf16_to_floats(a)
            delta_square += float(np.square(delta.astype(np.float64)).sum())
            max_delta = max(max_delta, float(np.abs(delta).max()))
        if not changed and not config.get("calibration_trial", False):
            raise ValueError("Optimizer ran but exported BF16 head did not change")
        names = ("master.f32", "moment1.f32", "moment2.f32", "head.bf16")
        hashes = {name: file_hash(checkpoint / name) for name in names}
        state = {"schema_version": 1, "shape": list(shape), "successful_updates": successful,
                 "parent_head_sha256": expected_parent, "initial_segment_head_sha256": initial_hash,
                 "data_revision": data_revision, "rng": rng.bit_generator.state, "order": order.tolist(),
                 "cursor": cursor, "hashes": hashes, "compute_precision": precision, "learning_rate": lr, "beta1": beta1,
                 "beta2": beta2, "epsilon": epsilon, "batch_size": batch, "accumulation": accumulation}
        atomic_json(checkpoint / "optimizer.json", state)
        return {"schema_version": 1, "mode": "train", "status": "paused" if stop["requested"] else "completed",
                "head_path": str(candidate_path), "checkpoint_path": str(checkpoint), "shape": list(shape),
                "before_head_sha256": initial_hash, "after_head_sha256": hashes["head.bf16"],
                "parent_head_sha256": expected_parent, "successful_updates": successful,
                "segment_updates": successful - starting_updates, "changed_elements": changed,
                "delta_l2": math.sqrt(delta_square), "delta_max_abs": max_delta,
                "losses": losses, "data_revision": data_revision, "resources": _resource_check(config, torch, device),
                "learning_rate": lr, "batch_size": batch, "accumulation": accumulation, "device": device,
                "compute_precision": precision, "activation_dtype": "F32" if precision == "native_f32" else "BF16",
                "logit_dtype": "F32" if precision == "native_f32" else "BF16", "export_dtype": "BF16",
                "gradient_accumulation": "cpu_f32" if cpu_gradient is not None else "gpu_bf16",
                "training_backend": "native_frozen_features_existing_head_cpu_adam"}
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if device == "cuda":
            torch.cuda.empty_cache()


def run(config):
    mode = config.get("mode")
    if mode == "export":
        result = export_artifacts(config)
    elif mode == "prepare":
        result = prepare(config)
    elif mode == "capture":
        from .head_capture import capture_native
        output = _output(config)
        result = capture_native({**config, "output_dir": str(output / "native")})
    elif mode == "train":
        from .meditation_recovery import capture_inputs
        config["_memory_retry_inputs"] = capture_inputs(config)
        result = train_segment(config)
    elif mode == "evaluate":
        _output(config)
        result = evaluate_head(config)
    elif mode == "verify":
        result = verify_native_candidate(config)
    elif mode == "native_admission":
        result = native_admission(config)
    else:
        raise ValueError("Unknown native-head worker mode")
    atomic_json(Path(config["output_dir"]) / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        result = run(config)
    except Exception as error:
        # Class/frame locations are sufficient for diagnosis without persisting
        # message text, model content or arbitrary third-party exception values.
        import traceback
        frames = [{"file": frame.filename, "function": frame.name, "line": frame.lineno}
                  for frame in traceback.extract_tb(error.__traceback__)]
        from .meditation_recovery import seal_failure
        resource_kind = getattr(error, "resource", None)
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError) and config.get("device") == "cuda":
            resource_kind = "cuda_oom"
        failure = seal_failure(config, resource=resource_kind, worker_path=__file__)
        atomic_json(Path(config["output_dir"]) / "error.json",
                    {"mode": config.get("mode"), "error_type": type(error).__name__, "frames": frames,
                     "memory_failure": failure, "preflight_rejection": getattr(error, "preflight_rejection", None)})
        raise
    _event(stage="finished", mode=config["mode"], result_path=str(Path(config["output_dir"]) / "result.json"),
           successful_updates=result.get("successful_updates"), head_sha256=result.get("after_head_sha256"))


if __name__ == "__main__":
    main()
