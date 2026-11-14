"""Owned native-engine feature capture; no model or GPU is touched on import.

Uses the deployed engine's last-layer debug contract. Outputs are private
numerical training artifacts, never provider reasoning or chat logs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from .head_pack import (atomic_json, bf16_to_floats, dequantize_q5_k, file_hash,
                        load_pack, read_tensor)


_RECORD = struct.Struct("<16sqqqq")


def controlled_execution_plan(config):
    """One frozen-backbone numerical contract shared by capture and serving."""
    threads, experts = int(config.get("cpu_threads", 12)), int(config.get("experts", 6))
    if not 1 <= threads <= 256 or not 1 <= experts <= 384:
        raise ValueError("Invalid controlled native execution dimensions")
    plan = {"schema_version": 1, "name": "cpu_routed_experts_window_path",
            "argv": ["--vram-experts", "0", "--no-expert-cache", "--cpu-threads", str(threads),
                     "--prefill-chunk", "0", "--threads", "off", "--tp", "1", "--dspark", "none",
                     "--experts", str(experts)],
            "env": {"DS4_CHUNK": "0", "DS4_STREAM": "0", "DS4_STREAM_WIN": "0", "DS4_LAND": "0",
                    "DS4_STORE_ZC": "0", "DS4_RECENT": "0"},
            "window": 16, "compiled_feed_unit": 16, "cpu_threads": threads, "routed_experts": experts,
            "vram_experts": 0, "expert_cache": False, "prefill_chunk": 0, "threads": False,
            "tensor_parallel": 1, "dspark": "none", "controlled_baseline_not_deployed_default": True}
    if config.get("execution_plan") is not None and config["execution_plan"] != plan:
        raise ValueError("Configured native execution plan disagrees with the canonical worker contract")
    if config.get("execution_plan_sha256") and config["execution_plan_sha256"] != execution_plan_hash(plan):
        raise ValueError("Configured native execution plan hash disagrees with the worker contract")
    return plan


def execution_plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observed_feed_unit(log_path, expected):
    with Path(log_path).open("rb") as handle:
        startup = handle.read(4 << 20).decode("utf-8", errors="replace")
    match = re.search(r"prompts run in windows of ([0-9]+) tokens", startup)
    if match is None or int(match.group(1)) != expected:
        raise ValueError("Native engine did not confirm the pinned compiled feed unit")
    return int(match.group(1))


def iter_native_records(path, *, max_record_bytes=8 << 20):
    with Path(path).open("rb") as handle:
        while True:
            raw = handle.read(_RECORD.size)
            if not raw:
                return
            if len(raw) != _RECORD.size:
                raise ValueError("Truncated native feature record header")
            name, layer, position, rows, count = _RECORD.unpack(raw)
            if layer < 0 or position < 0 or rows < 1 or count < 1 or count > max_record_bytes:
                raise ValueError("Invalid native feature record dimensions")
            payload = handle.read(count)
            if len(payload) != count:
                raise ValueError("Truncated native feature record payload")
            yield name.split(b"\0", 1)[0].decode("ascii"), layer, position, rows, payload


def reconstruct_features(hidden, pre, norm, epsilon):
    import numpy as np
    hidden, pre, norm = (np.asarray(value, dtype=np.float32) for value in (hidden, pre, norm))
    if hidden.ndim != 3 or pre.ndim != 2 or hidden.shape[0] != pre.shape[0] or pre.shape[1] < hidden.shape[1]:
        raise ValueError("Invalid final residual/mix shape")
    if norm.shape != (hidden.shape[2],) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid final norm")
    if not all(np.isfinite(x).all() for x in (hidden, pre[:, :hidden.shape[1]], norm)):
        raise ValueError("Nonfinite native feature inputs")
    # Preserve the native stream accumulation order; reduction rounding remains
    # an explicitly measured parity condition against native scores/logits.
    combined = np.zeros((hidden.shape[0], hidden.shape[2]), dtype=np.float32)
    for stream in range(hidden.shape[1]):
        combined += pre[:, stream, None] * hidden[:, stream, :]
    inv = np.reciprocal(np.sqrt(np.mean(combined * combined, axis=1, dtype=np.float32) + epsilon))
    features = combined * inv[:, None] * norm
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite reconstructed head features")
    return features.astype("<f4", copy=False)


def reconstruct_post_moe(hidden, moe, post, combination):
    """Reconstruct native hc_post when a build does not export final h_out.

    Combination rows are stored with a 64-float row stride, but only hc*hc
    entries are meaningful. Post/mix rows have stride eight. Padding must not
    be interpreted as model data or required to contain initialized values.
    """
    import numpy as np
    hidden, moe, post, combination = (np.asarray(value, dtype=np.float32)
                                      for value in (hidden, moe, post, combination))
    if hidden.ndim != 3:
        raise ValueError("Invalid native post-MoE residual shape")
    rows, hc, width = hidden.shape
    if moe.shape != (rows, width) or post.shape != (rows, 8) or combination.shape != (rows, 64):
        raise ValueError("Invalid native post-MoE mix shape")
    post, combination = post[:, :hc], combination[:, :hc * hc].reshape(rows, hc, hc)
    if not all(np.isfinite(value).all() for value in (hidden, moe, post, combination)):
        raise ValueError("Nonfinite native post-MoE inputs")
    result = post[:, :, None] * moe[:, None, :]
    # CUDA evaluates these updates with fused multiply-add. F64 products of F32
    # operands and one F32 rounding retain that arithmetic much more closely
    # than separate F32 multiply/add; native logit parity remains mandatory.
    for source in range(hc):
        result = (combination[:, source, :, None].astype(np.float64) *
                  hidden[:, source, None, :].astype(np.float64) + result.astype(np.float64)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed final residual")
    return result


def _segments(config):
    if config.get("tokens_path"):
        source = Path(config["tokens_path"])
        if source.stat().st_size > 128 << 20:
            raise ValueError("Token input exceeds bounded capture input")
        raw = json.loads(source.read_text())
    elif config.get("segments") is not None:
        raw = config["segments"]
    elif "ids" in config:
        raw = [{"name": "probe", "ids": config["ids"], "labels": config.get("labels")}]
    else:
        raise ValueError("Native capture requires explicit teacher-forced tokens")
    raw = raw.get("segments", []) if isinstance(raw, dict) else raw
    if not isinstance(raw, list) or not raw:
        raise ValueError("Native capture has no token segments")
    result = []
    for index, value in enumerate(raw):
        ids = value.get("ids")
        if not isinstance(ids, list) or len(ids) < 2 or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("Invalid capture token ids")
        labels = value.get("labels")
        if labels is None:
            # Explicit diagnostic probes may score every token; production
            # dataset callers must provide their assistant-only labels.
            if not config.get("diagnostic", False):
                raise ValueError("Assistant-only labels are required")
            labels = ids[:]
        if len(labels) != len(ids) or any(type(x) is not int or x != -100 and x != ids[i]
                                          for i, x in enumerate(labels)):
            raise ValueError("Labels must mask or equal their original token")
        name = str(value.get("name", index))
        result.append({"name": hashlib.sha256(name.encode()).hexdigest()[:24], "source_name": name,
                       "group": str(value.get("group", name)), "ids": ids, "labels": labels})
    return result


def native_command(config, segments, output):
    pack = load_pack(config["pack"])
    engine = Path(config["engine"]).resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise ValueError("Native capture engine is unavailable")
    width, vocab = pack.head.shape[1], pack.head.shape[0]
    if any(t >= vocab for segment in segments for t in segment["ids"]):
        raise ValueError("Token ids exceed the actual primary vocabulary")
    capture_input = output / "native-ids.json"
    atomic_json(capture_input, {"segments": [{"name": x["name"], "ids": x["ids"]} for x in segments]})
    plan = controlled_execution_plan(config)
    args = [str(engine), "--pack", str(pack.path), "--window", str(plan["window"]), *plan["argv"],
            "--vision", "off", "--reserve-mib", str(config.get("reserve_mib", 2048)),
            "--profile-readonly", "--max-context",
            str(max(len(x["ids"]) for x in segments) + 8)]
    if config.get("dump_logits"):
        if len(segments) != 1:
            raise ValueError("Full-logit diagnostic capture supports one segment")
        ids_file = output / "native-ids.txt"
        ids_file.write_text(",".join(map(str, segments[0]["ids"][:-1])))
        ids_file.chmod(0o600)
        args += ["--ids", "@" + str(ids_file), "--n", "0", "--dump-logits", str(output / "native-logits.f32")]
    else:
        args += ["--score", str(capture_input), "--score-out", str(output / "native-scores.tsv")]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DS4_DUMP") and key not in {"DS4_SCORE_PREFIX", "DS4_FEEDLOG"}}
    env.update(config.get("env", {}))
    env.update(plan["env"])
    # Overrides are forced after caller-provided env, not merely CLI defaults.
    env.update(DS4_CHUNK="0", DS4_DUMP_S=str(output / "native-features.bin"),
               DS4_DUMP_H=str(output / "native-h.bin"),
               DS4_DUMP_S_LAYERS=str(int(pack.metadata["num_hidden_layers"]) - 1),
               DS4_DUMP_S_NAMES="f_pre", STRATA_DS4_SLOT_KEEP="0")
    for name in ("DS4_DUMP_L", "DS4_DSPARK_DUMP", "DS4_SCORE_PREFIX"):
        env.pop(name, None)
    if config.get("score_only"):
        for name in ("DS4_DUMP_S", "DS4_DUMP_H", "DS4_DUMP_S_NAMES", "DS4_DUMP_S_LAYERS"):
            env.pop(name, None)
    libs = [str(engine.parent)]
    if env.get("LD_LIBRARY_PATH"):
        libs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return args, env, pack, width


def score_metrics(path, segments, row_map):
    """Bind every native score row to its exact teacher-forced source token."""
    selected = {row["native_row"]: str(row["group"]) for row in row_map}
    losses, clipped, counts, total = {}, {}, {}, 0
    with Path(path).open() as handle:
        for segment in segments:
            for position, target in enumerate(segment["ids"][1:]):
                line = handle.readline()
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8 or int(fields[0]) != position or int(fields[1]) != target:
                    raise ValueError("Native scoring rows do not match source positions/targets")
                lp = float(fields[2])
                if not math.isfinite(lp) or lp > 1e-5:
                    raise ValueError("Invalid native next-token log probability")
                if total in selected:
                    group = selected[total]
                    losses[group] = losses.get(group, 0.0) - lp
                    clipped[group] = clipped.get(group, 0.0) + min(20.0, -lp)
                    counts[group] = counts.get(group, 0) + 1
                total += 1
        if handle.readline():
            raise ValueError("Native scoring output has unexpected trailing rows")
    if sum(counts.values()) != len(row_map) or not counts:
        raise ValueError("Native scores do not cover selected targets")
    return {"native_nll": sum(losses.values()) / len(row_map),
            "native_group_losses": {g: losses[g] / counts[g] for g in counts},
            "native_group_clipped_losses": {g: clipped[g] / counts[g] for g in counts}, "native_group_tokens": counts}


def score_only_receipt(segments, output):
    row_map, source_counts, total = [], {}, 0
    for index, segment in enumerate(segments):
        for position, label in enumerate(segment["labels"][1:]):
            if label != -100:
                row_map.append({"native_row": total + position, "segment": index, "position": position,
                                "group": segment["group"]})
                name = segment["source_name"]
                source_counts[name] = source_counts.get(name, 0) + 1
        total += len(segment["ids"]) - 1
    return {"score_only": True, "rows": len(row_map), "native_rows": total, "row_map": row_map,
            "source_counts": source_counts, **score_metrics(output / "native-scores.tsv", segments, row_map)}


def parse_capture(pack, segments, output, *, dump_logits=False):
    import numpy as np
    width, hc = pack.head.shape[1], int(pack.metadata["hc_mult"])
    last_layer = int(pack.metadata["num_hidden_layers"]) - 1
    norm = np.frombuffer(read_tensor(pack, "norm.weight"), dtype="<f4")
    epsilon = float(pack.metadata["rms_norm_eps"])
    seg_index, position, selected, total = 0, 0, 0, 0
    pending, contract = {}, None
    trajectory_path = output / "native-h.bin"
    layers = int(pack.metadata["num_hidden_layers"])
    trajectory = trajectory_path.open("rb") if trajectory_path.exists() else None
    row_map, source_counts = [], {}
    feature_path, label_path = output / "features.f32", output / "labels.i64"
    from contextlib import ExitStack
    with ExitStack() as stack:
        feature_file = stack.enter_context(feature_path.open("xb"))
        label_file = stack.enter_context(label_path.open("xb"))
        if trajectory is not None:
            stack.enter_context(trajectory)
        feature_path.chmod(0o600)
        label_path.chmod(0o600)
        for name, layer, p0, rows, payload in iter_native_records(output / "native-features.bin"):
            if seg_index >= len(segments) or layer != last_layer or p0 != position:
                raise ValueError("Native feature stream has unexpected layer/position")
            segment = segments[seg_index]
            remaining = len(segment["ids"]) - 1 - position
            if rows > remaining:
                raise ValueError("Native feature rows exceed the source tokens")
            if contract is None:
                if trajectory is not None and name == "f_pre":
                    contract = ("f_pre",)
                elif name == "h_attn":
                    contract = ("h_attn", "f_pre", "f_post", "f_comb", "moe_out")
                elif name == "f_pre":
                    contract = ("f_pre", "h_out")
                else:
                    raise ValueError("Native capture does not begin with a supported complete contract")
            if name != contract[len(pending)]:
                raise ValueError("Native feature window has missing, duplicate or reordered records")
            expected_shape = {"h_attn": (rows, hc, width), "h_out": (rows, hc, width),
                              "f_pre": (rows, 8), "f_post": (rows, 8), "f_comb": (rows, 64),
                              "moe_out": (rows, width)}[name]
            expected_bytes = math.prod(expected_shape) * 4
            if len(payload) != expected_bytes or pending and next(iter(pending.values())).shape[0] != rows:
                raise ValueError("Native feature record size disagrees with its window")
            pending[name] = np.frombuffer(payload, dtype="<f4").reshape(expected_shape)
            if len(pending) == len(contract):
                hidden = pending.get("h_out")
                if trajectory is not None:
                    expected_h_bytes = rows * layers * hc * width * 4
                    raw_h = trajectory.read(expected_h_bytes)
                    if len(raw_h) != expected_h_bytes:
                        raise ValueError("Native residual trajectory lacks complete window/layer coverage")
                    # DS4_DUMP_H's outer serial loop emits [token,layer,hc,dim]
                    # after each layer returns, including GPU-resident paths.
                    hidden = np.frombuffer(raw_h, dtype="<f4").reshape(rows, layers, hc, width)[:, -1]
                elif hidden is None:
                    hidden = reconstruct_post_moe(pending["h_attn"], pending["moe_out"], pending["f_post"],
                                                   pending["f_comb"])
                features = reconstruct_features(hidden, pending["f_pre"], norm, epsilon)
                labels = np.asarray(segment["labels"][p0 + 1:p0 + rows + 1], dtype="<i8")
                mask = labels != -100
                feature_file.write(features[mask].tobytes())
                label_file.write(labels[mask].tobytes())
                for local in np.nonzero(mask)[0]:
                    row_map.append({"native_row": total + int(local), "segment": seg_index,
                                    "position": p0 + int(local), "group": segment["group"]})
                count = int(mask.sum())
                selected += count
                source_counts[segment["source_name"]] = source_counts.get(segment["source_name"], 0) + count
                total += rows
                position += rows
                pending = {}
                if position == len(segment["ids"]) - 1:
                    seg_index += 1
                    position = 0
        if pending or seg_index != len(segments) or position or not selected:
            raise ValueError("Native capture is incomplete or has no assistant targets")
        if trajectory is not None and trajectory.read(1):
            raise ValueError("Native residual trajectory has unexpected trailing rows/layers")
        for handle in (feature_file, label_file):
            handle.flush()
            os.fsync(handle.fileno())
    scoring = {"native_nll": None, "native_group_losses": {}, "native_group_clipped_losses": {},
               "native_group_tokens": {}}
    if not dump_logits:
        scoring = score_metrics(output / "native-scores.tsv", segments, row_map)
    return {"feature_path": str(feature_path), "features_path": str(feature_path), "labels_path": str(label_path),
            "feature_dtype": "F32", "rows": selected, "width": width, "shape": list(pack.head.shape),
            "native_rows": total, **scoring, "row_map": row_map, "source_counts": source_counts,
            "native_record_contract": list(contract or ()) + (["residual_trajectory"] if trajectory is not None else []),
            "features_sha256": file_hash(feature_path), "labels_sha256": file_hash(label_path)}


def capture_native(config):
    """Run finite native teacher-forced capture; caller owns inference exclusion."""
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    segments = _segments(config)
    args, env, pack, width = native_command(config, segments, output)
    limit = int(config.get("disk_limit_bytes", 50_000_000_000))
    expected = sum(len(s["ids"]) - 1 for s in segments)
    hc = int(pack.metadata["hc_mult"])
    layers = int(pack.metadata["num_hidden_layers"])
    estimated = expected * ((0 if config.get("score_only") else (width * hc * layers + 8) * 4 + width * 4) +
                            (pack.head.shape[0] * 4 if config.get("dump_logits") else 1500))
    if estimated > limit:
        raise ValueError("Capture exceeds its disk admission budget")
    log_path = output / "native-engine.log"
    with log_path.open("xb") as log:
        log_path.chmod(0o600)
        # The native child stays in the isolated worker's owned process group.
        # A service cancellation of that group must also release this GPU child.
        process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, umask=0o077)
        try:
            code = process.wait(timeout=config.get("timeout_seconds"))
            if code:
                raise RuntimeError(f"Native capture exited with code {code}; private diagnostic retained")
        except BaseException:
            try:
                process.terminate()
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    plan = controlled_execution_plan(config)
    feed_unit = observed_feed_unit(log_path, plan["compiled_feed_unit"])
    receipt = (score_only_receipt(segments, output) if config.get("score_only") else
               parse_capture(pack, segments, output, dump_logits=bool(config.get("dump_logits"))))
    receipt.update(schema_version=3, native_capture=True,
                   reconstruction=None if config.get("score_only") else "final_layer_residual_trajectory_f_pre_rmsnorm",
                   pack_path=str(pack.path), pack_manifest_sha256=pack.manifest_sha256,
                   engine_sha256=file_hash(config["engine"]),
                   norm_sha256=hashlib.sha256(read_tensor(pack, "norm.weight")).hexdigest(),
                   provenance=config.get("provenance", {}),
                   token_segments_sha256=hashlib.sha256(json.dumps(segments, sort_keys=True).encode()).hexdigest(),
                   output_dir=str(output), effective_chunk_size=0, speculation=False,
                   observed_engine_feed_unit=feed_unit, capture_window=plan["window"],
                   execution_plan=plan, execution_plan_sha256=execution_plan_hash(plan))
    atomic_json(output / "capture.json", receipt)
    # The all-layer trajectory is only a bridge to the exact final-layer
    # features. Keep neither unrelated layers nor an unbounded raw cache.
    (output / "native-h.bin").unlink(missing_ok=True)
    return receipt


def capture_batched(config):
    """Bound temporary all-layer dumps while preserving complete source groups.

    Native teacher-forced segments already have an explicit context policy.
    Batching whole segments preserves that policy and never truncates labels.
    """
    segments = _segments(config)
    if config.get("score_only") or config.get("dump_logits"):
        return capture_native(config)
    pack = load_pack(config["pack"])
    per_token = pack.head.shape[1] * int(pack.metadata["hc_mult"]) * int(pack.metadata["num_hidden_layers"]) * 4
    limit = min(int(config.get("raw_batch_limit_bytes", 4 << 30)),
                int(config.get("disk_limit_bytes", 50_000_000_000)) // 4)
    if limit < per_token:
        raise ValueError("Native trajectory batch budget is too small")
    batches, batch, used = [], [], 0
    for segment in segments:
        size = (len(segment["ids"]) - 1) * per_token
        if size > limit:
            raise ValueError("A complete source window exceeds the native trajectory budget; reduce max_seq_length")
        if batch and used + size > limit:
            batches.append(batch)
            batch, used = [], 0
        batch.append(segment)
        used += size
    batches.append(batch)
    if len(batches) == 1:
        return capture_native(config)
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    feature_path, labels_path = output / "features.f32", output / "labels.i64"
    row_map, source_counts, shards, counts, loss_sums, clipped_sums = [], {}, [], {}, {}, {}
    rows, native_rows, segment_offset, reference = 0, 0, 0, None
    with feature_path.open("xb") as features, labels_path.open("xb") as labels:
        feature_path.chmod(0o600)
        labels_path.chmod(0o600)
        for index, batch in enumerate(batches):
            print(json.dumps({"meditation": True, "stage": "native_capture", "shard": index + 1,
                              "shards": len(batches)}), flush=True)
            raw_segments = [{"name": segment["source_name"], "ids": segment["ids"],
                             "labels": segment["labels"], "group": segment["group"]} for segment in batch]
            shard_path = output / f"shard-{index:05d}"
            receipt = capture_native({**config, "segments": raw_segments, "tokens_path": None,
                                      "output_dir": str(shard_path)})
            if reference is None:
                reference = receipt
            for key in ("pack_manifest_sha256", "engine_sha256", "norm_sha256", "execution_plan_sha256"):
                if reference[key] != receipt[key]:
                    raise ValueError("Native feature dependency changed between bounded shards")
            with Path(receipt["features_path"]).open("rb") as handle:
                shutil.copyfileobj(handle, features, 4 << 20)
            with Path(receipt["labels_path"]).open("rb") as handle:
                shutil.copyfileobj(handle, labels, 4 << 20)
            row_map.extend({**row, "native_row": row["native_row"] + native_rows,
                            "segment": row["segment"] + segment_offset} for row in receipt["row_map"])
            for name, count in receipt["source_counts"].items():
                source_counts[name] = source_counts.get(name, 0) + count
            for group, count in receipt["native_group_tokens"].items():
                counts[group] = counts.get(group, 0) + count
                loss_sums[group] = loss_sums.get(group, 0.0) + receipt["native_group_losses"][group] * count
                clipped_sums[group] = clipped_sums.get(group, 0.0) + receipt["native_group_clipped_losses"][group] * count
            rows += receipt["rows"]
            native_rows += receipt["native_rows"]
            segment_offset += len(batch)
            shards.append({"path": str(shard_path), "receipt_sha256": file_hash(shard_path / "capture.json"),
                           "rows": receipt["rows"], "native_rows": receipt["native_rows"]})
            for path in {receipt["features_path"], receipt["labels_path"]}:
                Path(path).unlink()
        for handle in (features, labels):
            handle.flush()
            os.fsync(handle.fileno())
    result = {**reference, "output_dir": str(output), "features_path": str(feature_path),
              "feature_path": str(feature_path), "labels_path": str(labels_path), "rows": rows,
              "native_rows": native_rows, "row_map": row_map, "source_counts": source_counts,
              "native_nll": sum(loss_sums.values()) / rows,
              "native_group_losses": {g: loss_sums[g] / counts[g] for g in counts},
              "native_group_clipped_losses": {g: clipped_sums[g] / counts[g] for g in counts},
              "native_group_tokens": counts, "features_sha256": file_hash(feature_path),
              "labels_sha256": file_hash(labels_path), "shards": shards, "raw_batch_limit_bytes": limit,
              "token_segments_sha256": hashlib.sha256(json.dumps(segments, sort_keys=True).encode()).hexdigest()}
    atomic_json(output / "capture.json", result)
    return result


def compare_native_logits(pack, capture, *, rows_per_chunk=128, atol=0.02, rtol=0.005):
    """Independently compare reconstructed-feature projections to native logits.

    F32 CPU projection and native quantized GEMV need not be bit-identical.
    Explicit measured tolerances are returned, never converted to quality proof.
    """
    import numpy as np
    output = Path(capture["output_dir"])
    count, width = capture["rows"], capture["width"]
    vocab = pack.head.shape[0]
    features = np.memmap(capture["features_path"], dtype="<f4", mode="r", shape=(count, width))
    native = np.memmap(output / "native-logits.f32", dtype="<f4", mode="r",
                       shape=(capture["native_rows"], vocab))
    positions = [row["native_row"] for row in capture["row_map"]]
    max_abs, sum_square, elements, violations = 0.0, 0.0, 0, 0
    head = pack.head
    row_bytes = head.size // vocab
    with pack.files[head.file_index].open("rb") as handle:
        handle.seek(head.offset)
        for start in range(0, vocab, rows_per_chunk):
            n = min(rows_per_chunk, vocab - start)
            raw = handle.read(n * row_bytes)
            if len(raw) != n * row_bytes:
                raise ValueError("Truncated head during parity projection")
            if head.dtype == "Q5_K":
                weight = dequantize_q5_k(raw, (n, width))
            elif head.dtype == "BF16":
                weight = bf16_to_floats(np.frombuffer(raw, dtype="<u2")).reshape(n, width)
            else:
                raise ValueError("Unsupported native parity head")
            actual = np.asarray(features) @ weight.T
            expected = np.asarray(native[positions, start:start + n])
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError("Nonfinite parity logits")
            difference = np.abs(actual - expected)
            max_abs = max(max_abs, float(difference.max()))
            sum_square += float(np.square(difference.astype(np.float64)).sum())
            elements += difference.size
            violations += int((difference > atol + rtol * np.abs(expected)).sum())
    return {"passed": violations == 0, "max_abs_error": max_abs, "rmse": math.sqrt(sum_square / elements),
            "elements": elements, "violations": violations, "atol": atol, "rtol": rtol,
            "native_engine_logits_compared": True}
