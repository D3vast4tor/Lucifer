"""Record/mask correctness; no native model or GPU is loaded by these tests."""
import struct
from pathlib import Path

import pytest

from lucifer.head_capture import (iter_native_records, native_command, parse_capture,
                                  reconstruct_features, _segments)
from lucifer.head_pack import load_pack
from test_head_pack import fixture_pack

np = pytest.importorskip("numpy")


def write_record(handle, name, position, value):
    rows = value.shape[0]
    payload = value.astype("<f4").tobytes()
    handle.write(struct.pack("<16sqqqq", name.encode(), 0, position, rows, len(payload)))
    handle.write(payload)


def test_capture_preserves_next_token_masks_and_independent_group_losses(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    out = tmp_path / "capture"
    out.mkdir()
    segments = [{"name": "a", "source_name": "document-a", "group": "group-a", "ids": [0, 1, 0, 1],
                 "labels": [-100, -100, 0, 1]},
                {"name": "b", "source_name": "document-b", "group": "group-b", "ids": [0, 2],
                 "labels": [-100, 2]}]
    with (out / "native-features.bin").open("wb") as handle:
        for rows in (3, 1):
            pre = np.zeros((rows, 8), dtype=np.float32)
            pre[:, :2] = [0.25, 0.75]
            hidden = np.ones((rows, 2, 256), dtype=np.float32)
            write_record(handle, "f_pre", 0, pre)
            write_record(handle, "h_out", 0, hidden)
    (out / "native-scores.tsv").write_text("".join(f"{pos}\t{target}\t{-loss}\t0\t0\t0\t0\t0\n"
        for pos, target, loss in ((0, 1, 0.1), (1, 0, 0.2), (2, 1, 0.4), (0, 2, 0.8))))
    result = parse_capture(pack, segments, out)
    assert result["rows"] == 3 and result["native_rows"] == 4
    np.testing.assert_array_equal(np.fromfile(result["labels_path"], dtype="<i8"), [0, 1, 2])
    assert [row["native_row"] for row in result["row_map"]] == [1, 2, 3]
    assert result["native_nll"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    assert result["native_group_losses"] == pytest.approx({"group-a": 0.3, "group-b": 0.8})
    assert result["source_counts"] == {"document-a": 2, "document-b": 1}
    assert not Path(result["features_path"]).stat().st_mode & 0o077


@pytest.mark.parametrize("data", [b"x", struct.pack("<16sqqqq", b"h_out", 0, 0, 1, 400) + b"x",
                                   struct.pack("<16sqqqq", b"h_out", 0, 0, 1, 9 << 20)])
def test_truncated_or_unbounded_native_records_are_rejected(tmp_path, data):
    path = tmp_path / "bad"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(iter_native_records(path))


def test_mismatched_record_position_never_produces_valid_capture(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    out = tmp_path / "capture"
    out.mkdir()
    with (out / "native-features.bin").open("wb") as handle:
        write_record(handle, "f_pre", 1, np.ones((1, 8)))
    with pytest.raises(ValueError, match="position"):
        parse_capture(pack, [{"ids": [0, 1], "labels": [-100, 1]}], out)


def test_final_norm_reconstruction_uses_mix_and_norm_not_residual_mean():
    hidden = np.array([[[1, 3], [9, -5]]], dtype=np.float32)
    pre = np.array([[0.1, 0.9]], dtype=np.float32)
    norm = np.array([2., 0.5], dtype=np.float32)
    expected = np.array([[8.2, -4.2]], dtype=np.float32)
    expected = expected / np.sqrt(np.mean(expected * expected) + 1e-5) * norm
    np.testing.assert_allclose(reconstruct_features(hidden, pre, norm, 1e-5), expected, rtol=1e-6)
    with pytest.raises(ValueError, match="Nonfinite"):
        reconstruct_features(hidden * float("nan"), pre, norm, 1e-5)


def test_capture_requires_assistant_masks_outside_diagnostic_mode():
    with pytest.raises(ValueError, match="labels"):
        _segments({"ids": [0, 1]})
    with pytest.raises(ValueError, match="original token"):
        _segments({"ids": [0, 1], "labels": [-100, 2]})
    assert _segments({"ids": [0, 1], "diagnostic": True})[0]["labels"] == [0, 1]


def test_native_command_forces_window_path_despite_ambient_overrides(tmp_path, monkeypatch):
    pack = fixture_pack(tmp_path)
    engine = tmp_path / "engine"
    engine.write_text("#!/bin/sh\nexit 0\n")
    engine.chmod(0o700)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("DS4_CHUNK", "4096")
    monkeypatch.setenv("DS4_DUMP_H", "/unsafe/other")
    config = {"pack": str(pack), "engine": str(engine), "env": {"DS4_CHUNK": "2048", "DS4_SCORE_PREFIX": "50"}}
    args, env, _, _ = native_command(config, [{"name": "a", "ids": [0, 1]}], out)
    assert env["DS4_CHUNK"] == "0" and "DS4_SCORE_PREFIX" not in env
    assert env["DS4_DUMP_H"] == str(out / "native-h.bin")
    assert env["DS4_DUMP_S_NAMES"] == "f_pre"
    assert args[args.index("--prefill-chunk") + 1] == "0"
    assert args[args.index("--dspark") + 1] == "none" and "--profile-readonly" in args


def test_cancelling_owned_worker_group_also_stops_native_child(tmp_path):
    """A sleeping fixture exercises real Linux process-group cancellation."""
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    pack = fixture_pack(tmp_path)
    pid_file = tmp_path / "native.pid"
    engine = tmp_path / "native-fixture"
    engine.write_text(f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
                      "Path(os.environ['CAPTURE_PIDFILE']).write_text(str(os.getpid()))\ntime.sleep(60)\n")
    engine.chmod(0o700)
    config = {"pack": str(pack), "engine": str(engine), "ids": [0, 1], "diagnostic": True,
              "output_dir": str(tmp_path / "capture"), "env": {"CAPTURE_PIDFILE": str(pid_file)}}
    code = "import json,sys; from lucifer.head_capture import capture_native; capture_native(json.loads(sys.argv[1]))"
    worker = subprocess.Popen([sys.executable, "-c", code, json.dumps(config)], start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists(), "fixture native child did not start"
        child = int(pid_file.read_text())
        assert os.getpgid(child) == worker.pid
        os.killpg(worker.pid, signal.SIGTERM)
        worker.wait(timeout=5)
        deadline = time.monotonic() + 5
        while Path(f"/proc/{child}/stat").exists() and time.monotonic() < deadline:
            if Path(f"/proc/{child}/stat").read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        assert (not Path(f"/proc/{child}/stat").exists() or
                Path(f"/proc/{child}/stat").read_text().split()[2] == "Z")
    finally:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait(timeout=5)


def test_supported_post_moe_records_reconstruct_final_head_input_with_strict_coverage(tmp_path):
    """The deployed build exports these five points but has no h_out hook."""
    from lucifer.head_capture import reconstruct_post_moe
    pack = load_pack(fixture_pack(tmp_path))
    output = tmp_path / "post-capture"
    output.mkdir()
    hidden = np.ones((2, 2, 256), dtype=np.float32)
    hidden[:, 0] = np.arange(256, dtype=np.float32)
    hidden[:, 1] = 2
    pre, post, combination = (np.full((2, columns), float("nan"), dtype=np.float32) for columns in (8, 8, 64))
    pre[:, :2], post[:, :2] = [0.25, 0.75], [2, 3]
    combination[:, :4] = [1, 2, 3, 4]
    moe = np.full((2, 256), 7, dtype=np.float32)
    expected_final = np.stack([np.arange(256) + 20, 2 * np.arange(256) + 29], axis=0).astype(np.float32)
    np.testing.assert_array_equal(reconstruct_post_moe(hidden, moe, post, combination),
                                  np.stack([expected_final, expected_final]))
    with (output / "native-features.bin").open("wb") as handle:
        for name, value in (("h_attn", hidden), ("f_pre", pre), ("f_post", post),
                            ("f_comb", combination), ("moe_out", moe)):
            write_record(handle, name, 0, value)
    (output / "native-scores.tsv").write_text("0\t1\t-0.2\t0\t0\t0\t0\t0\n1\t2\t-0.4\t0\t0\t0\t0\t0\n")
    segments = [{"ids": [0, 1, 2], "labels": [-100, 1, 2], "group": "source", "source_name": "source"}]
    result = parse_capture(pack, segments, output)
    assert result["rows"] == result["native_rows"] == 2
    assert result["native_record_contract"] == ["h_attn", "f_pre", "f_post", "f_comb", "moe_out"]
    combined = np.arange(256, dtype=np.float32) * 1.75 + 26.75
    expected = combined / np.sqrt(np.mean(combined * combined) + 1e-5)
    actual = np.fromfile(result["features_path"], dtype="<f4").reshape(2, 256)
    np.testing.assert_allclose(actual, np.stack([expected, expected]), rtol=1e-6)


def test_partial_post_moe_contract_cannot_be_mistaken_for_final_features(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    output = tmp_path / "incomplete"
    output.mkdir()
    with (output / "native-features.bin").open("wb") as handle:
        write_record(handle, "h_attn", 0, np.ones((1, 2, 256)))
        write_record(handle, "f_pre", 0, np.ones((1, 8)))
        write_record(handle, "moe_out", 0, np.ones((1, 256)))
    with pytest.raises(ValueError, match="missing, duplicate or reordered"):
        parse_capture(pack, [{"ids": [0, 1], "labels": [-100, 1]}], output)


@pytest.mark.parametrize("bad_bytes,match", [(b"truncated", "complete window/layer"), (b"", "trailing")])
def test_residual_trajectory_requires_exact_layer_token_byte_coverage(tmp_path, bad_bytes, match):
    pack_path = fixture_pack(tmp_path)
    metadata = pack_path / "ds4.txt"
    metadata.write_text(metadata.read_text().replace("num_hidden_layers 1", "num_hidden_layers 3"))
    pack = load_pack(pack_path)
    output = tmp_path / "trajectory"
    output.mkdir()
    pre = np.zeros((1, 8), dtype=np.float32)
    pre[:, :2] = [0.25, 0.75]
    with (output / "native-features.bin").open("wb") as handle:
        payload = pre.astype("<f4").tobytes()
        handle.write(struct.pack("<16sqqqq", b"f_pre", 2, 0, 1, len(payload)))
        handle.write(payload)
    full = np.ones((1, 3, 2, 256), dtype="<f4").tobytes()
    (output / "native-h.bin").write_bytes(bad_bytes if bad_bytes else full + b"extra")
    with pytest.raises(ValueError, match=match):
        parse_capture(pack, [{"ids": [0, 1], "labels": [-100, 1], "group": "g", "source_name": "g"}], output)


def test_outer_residual_trajectory_selects_final_layer_after_each_gpu_layer_returns(tmp_path):
    pack_path = fixture_pack(tmp_path)
    metadata = pack_path / "ds4.txt"
    metadata.write_text(metadata.read_text().replace("num_hidden_layers 1", "num_hidden_layers 3"))
    pack = load_pack(pack_path)
    output = tmp_path / "trajectory"
    output.mkdir()
    rng = np.random.default_rng(65)
    trajectory = rng.normal(size=(3, 3, 2, 256)).astype("<f4")
    trajectory.tofile(output / "native-h.bin")
    pre = np.full((3, 8), float("nan"), dtype="<f4")
    pre[:, :2] = [0.2, 0.8]
    with (output / "native-features.bin").open("wb") as handle:
        for start, rows in ((0, 2), (2, 1)):
            payload = pre[start:start + rows].tobytes()
            handle.write(struct.pack("<16sqqqq", b"f_pre", 2, start, rows, len(payload)))
            handle.write(payload)
    (output / "native-scores.tsv").write_text("".join(f"{pos}\t{target}\t-0.2\t0\t0\t0\t0\t0\n"
        for pos, target in enumerate([1, 2, 1])))
    segments = [{"ids": [0, 1, 2, 1], "labels": [-100, 1, 2, 1], "group": "g", "source_name": "g"}]
    result = parse_capture(pack, segments, output)
    actual = np.fromfile(result["features_path"], dtype="<f4").reshape(3, 256)
    expected = reconstruct_features(trajectory[:, -1], pre, np.ones(256), 1e-5)
    wrong_layer = reconstruct_features(trajectory[:, 0], pre, np.ones(256), 1e-5)
    np.testing.assert_array_equal(actual, expected)
    assert not np.allclose(actual, wrong_layer)
    assert result["native_record_contract"] == ["f_pre", "residual_trajectory"]
    assert result["rows"] == 3


def test_controlled_execution_plan_is_stable_and_overrides_gpu_promotion(tmp_path):
    from lucifer.head_capture import controlled_execution_plan, execution_plan_hash
    pack = fixture_pack(tmp_path)
    engine = tmp_path / "engine"
    engine.write_text("#!/bin/sh\nexit 0\n")
    engine.chmod(0o700)
    output = tmp_path / "out"
    output.mkdir()
    config = {"pack": str(pack), "engine": str(engine), "score_only": True,
              "env": {"DS4_STREAM": "1", "DS4_STREAM_WIN": "1", "DS4_LAND": "1", "DS4_STORE_ZC": "1"}}
    args, env, _, _ = native_command(config, [{"name": "a", "ids": [0, 1]}], output)
    assert args[args.index("--vram-experts") + 1] == "0" and "--no-expert-cache" in args
    assert args[args.index("--cpu-threads") + 1] == "12"
    assert all(env[name] == "0" for name in ("DS4_STREAM", "DS4_STREAM_WIN", "DS4_LAND", "DS4_STORE_ZC", "DS4_RECENT"))
    assert "DS4_DUMP_H" not in env and "DS4_DUMP_S" not in env
    plan = controlled_execution_plan(config)
    assert execution_plan_hash(plan) == execution_plan_hash(controlled_execution_plan({}))
    assert execution_plan_hash(plan) != execution_plan_hash(controlled_execution_plan({"cpu_threads": 24}))


def test_native_scores_reject_shifted_labels_even_when_counts_match(tmp_path):
    from lucifer.head_capture import score_only_receipt
    (tmp_path / "native-scores.tsv").write_text("0\t2\t-0.2\t0\t0\t0\t0\t0\n")
    segments = [{"ids": [0, 1], "labels": [-100, 1], "group": "g", "source_name": "g"}]
    with pytest.raises(ValueError, match="positions/targets"):
        score_only_receipt(segments, tmp_path)


def test_bounded_native_shards_preserve_feature_order_and_group_metrics(tmp_path, monkeypatch):
    import json
    from lucifer import head_capture
    from lucifer.head_pack import atomic_json, file_hash
    pack = load_pack(fixture_pack(tmp_path))
    plan = head_capture.controlled_execution_plan({})
    def fixture_capture(config):
        output = Path(config["output_dir"])
        output.mkdir()
        segment = config["segments"][0]
        number = int(segment["name"])
        value = [0.2, 0.4, 0.8][number]
        features = output / "features.f32"
        labels = output / "labels.i64"
        np.full((2, 256), number, dtype="<f4").tofile(features)
        np.array([1, 2], dtype="<i8").tofile(labels)
        group = segment["group"]
        result = {"features_path": str(features), "feature_path": str(features), "labels_path": str(labels),
                  "rows": 2, "native_rows": 2, "shape": [3, 256], "width": 256, "feature_dtype": "F32",
                  "pack_manifest_sha256": pack.manifest_sha256, "engine_sha256": "fixture", "norm_sha256": "fixture",
                  "execution_plan": plan, "execution_plan_sha256": head_capture.execution_plan_hash(plan),
                  "source_counts": {segment["name"]: 2}, "native_nll": value,
                  "native_group_losses": {group: value}, "native_group_clipped_losses": {group: value},
                  "native_group_tokens": {group: 2},
                  "row_map": [{"native_row": pos, "position": pos, "segment": 0, "group": group} for pos in range(2)]}
        atomic_json(output / "capture.json", result)
        return result
    monkeypatch.setattr(head_capture, "capture_native", fixture_capture)
    segments = [{"name": str(i), "ids": [0, 1, 2], "labels": [-100, 1, 2], "group": "a" if i < 2 else "b"}
                for i in range(3)]
    output = tmp_path / "batched"
    result = head_capture.capture_batched({"pack": str(pack.path), "segments": segments,
                                          "output_dir": str(output), "raw_batch_limit_bytes": 4096})
    assert len(result["shards"]) == 3 and result["rows"] == 6
    assert result["native_group_losses"] == pytest.approx({"a": 0.3, "b": 0.8})
    assert result["native_nll"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    actual = np.fromfile(result["features_path"], dtype="<f4").reshape(6, 256)
    np.testing.assert_array_equal(actual[:, 0], [0, 0, 1, 1, 2, 2])
    assert [row["native_row"] for row in result["row_map"]] == list(range(6))
    assert [row["segment"] for row in result["row_map"]] == [0, 0, 1, 1, 2, 2]
    assert result["features_sha256"] == file_hash(result["features_path"])
    assert all(not (Path(shard["path"]) / "features.f32").exists() for shard in result["shards"])
    assert json.loads((output / "capture.json").read_text())["rows"] == 6


def test_native_compiled_feed_unit_requires_observed_log_evidence(tmp_path):
    from lucifer.head_capture import controlled_execution_plan, execution_plan_hash, observed_feed_unit
    log = tmp_path / "native.log"
    log.write_text("ds4: prompts run in windows of 16 tokens (--prefill-chunk 0)\n")
    plan = controlled_execution_plan({})
    assert plan["window"] == plan["compiled_feed_unit"] == observed_feed_unit(log, 16) == 16
    with pytest.raises(ValueError, match="feed unit"):
        observed_feed_unit(log, 8)
    log.write_text("no interface evidence\n")
    with pytest.raises(ValueError, match="feed unit"):
        observed_feed_unit(log, 16)
    with pytest.raises(ValueError, match="execution plan"):
        controlled_execution_plan({"execution_plan": {**plan, "window": 8}})
    assert controlled_execution_plan({"execution_plan": plan, "execution_plan_sha256": execution_plan_hash(plan)}) == plan
