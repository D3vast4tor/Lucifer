"""Native head I/O checks; fixtures never stand in for live model acceptance."""
import hashlib
import struct

import pytest

from lucifer.head_pack import (bf16_to_floats, create_head_pack, dequantize_q5_k,
                               export_head_bf16, extract_head, floats_to_bf16,
                               load_pack, read_tensor, tensor_hash)

np = pytest.importorskip("numpy")


def fixture_pack(tmp_path, *, rows=3, width=256):
    model, pack = tmp_path / "model", tmp_path / "pack"
    model.mkdir()
    pack.mkdir()
    block = bytearray(176)
    struct.pack_into("<ee", block, 0, 0.25, 0.125)
    block[4:16] = bytes([1] * 12)
    block[48:176] = bytes([0x31] * 128)
    head = bytes(block) * (rows * width // 256)
    norm = np.ones(width, dtype="<f4").tobytes()
    (model / "weights.bin").write_bytes(head + norm)
    (pack / "ds4.txt").write_text(f"model_dir {model}\nnum_hidden_layers 1\nhc_mult 2\nrms_norm_eps 1e-5\n")
    (pack / "tensors.txt").write_text(
        f"file 0 weights.bin\nt head.weight Q5_K 0 0 {len(head)} {rows} {width}\n"
        f"t norm.weight F32 0 {len(head)} {len(norm)} {width}\n")
    (pack / "engram.bin").write_bytes(b"fixture-only")
    (pack / "expert-profile.bin").write_bytes(b"fixture-profile")
    (pack / "tokenizer").mkdir()
    return pack


def test_q5k_decoder_matches_independent_scalar_bit_layout():
    rng = np.random.default_rng(91)
    block = bytearray(rng.integers(0, 256, size=176, dtype=np.uint8).tobytes())
    struct.pack_into("<ee", block, 0, 0.25, 0.125)
    scale, minimum = struct.unpack_from("<ee", block)
    scales, expected = block[4:16], []
    for group in range(8):
        s = scales[group] & 63 if group < 4 else (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)
        m = scales[group + 4] & 63 if group < 4 else (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
        for index in range(32):
            nibble = (block[48 + group // 2 * 32 + index] >> (4 * (group % 2))) & 15
            q = nibble | (((block[16 + index] >> group) & 1) << 4)
            expected.append(scale * s * q - minimum * m)
    actual = dequantize_q5_k(bytes(block), (1, 256))
    np.testing.assert_array_equal(actual[0], np.array(expected, dtype=np.float32))
    with pytest.raises(ValueError, match="complete"):
        dequantize_q5_k(bytes(block[:-1]))


def test_stream_export_and_candidate_keep_parent_bytes_and_auxiliary_assets(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    before = tensor_hash(pack)
    extracted = extract_head(pack, tmp_path / "head.q5")
    assert extracted["sha256"] == before
    converted = export_head_bf16(pack, tmp_path / "head.bf16", rows_per_chunk=1)
    decoded = dequantize_q5_k((tmp_path / "head.q5").read_bytes(), pack.head.shape)
    restored = bf16_to_floats(np.fromfile(converted["path"], dtype="<u2")).reshape(pack.head.shape)
    np.testing.assert_array_equal(restored, bf16_to_floats(floats_to_bf16(decoded)))
    receipt = create_head_pack(pack, tmp_path / "candidate", converted["path"])
    candidate = load_pack(receipt["path"])
    assert candidate.head.offset == 0 and candidate.head.dtype == "BF16"
    assert candidate.head.shape == pack.head.shape
    assert candidate.files[0] == pack.files[0] and receipt["frozen_shards_reused"]
    assert (candidate.path / "engram.bin").is_symlink()
    assert (candidate.path / "tokenizer").is_symlink()
    assert (candidate.path / "expert-profile.bin").read_bytes() == b"fixture-profile"
    assert tensor_hash(pack) == before
    assert tensor_hash(candidate) == converted["sha256"]
    assert read_tensor(pack, "norm.weight", max_bytes=4096)
    with pytest.raises(ValueError, match="bounded"):
        read_tensor(pack, "head.weight", max_bytes=1)
    assert not (candidate.path / "tensors.txt").stat().st_mode & 0o077


def test_noop_q5_sidecar_preserves_exact_payload(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    head = extract_head(pack, tmp_path / "copy.q5")
    receipt = create_head_pack(pack, tmp_path / "no-op", head["path"], dtype="Q5_K")
    assert tensor_hash(load_pack(receipt["path"])) == tensor_hash(pack)
    assert receipt["head_dtype"] == "Q5_K"


@pytest.mark.parametrize("mutation,match", [
    (lambda text: text + text.splitlines()[1] + "\n", "Duplicate"),
    (lambda text: text.replace("Q5_K 0 0", "Q5_K 0 9999999"), "range"),
    (lambda text: text.replace("file 0 weights.bin", "file 1 weights.bin"), "contiguous"),
    (lambda text: text.replace("weights.bin", "/absolute/weights.bin"), "relative"),
])
def test_ambiguous_or_out_of_range_manifests_are_rejected(tmp_path, mutation, match):
    path = fixture_pack(tmp_path)
    manifest = path / "tensors.txt"
    manifest.write_text(mutation(manifest.read_text()))
    with pytest.raises(ValueError, match=match):
        load_pack(path)


def test_candidate_rejects_wrong_size_and_never_overwrites(tmp_path):
    pack = load_pack(fixture_pack(tmp_path))
    bad = tmp_path / "bad.bf16"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="size"):
        create_head_pack(pack, tmp_path / "invalid", bad)
    assert not (tmp_path / "invalid").exists()
    destination = tmp_path / "head.raw"
    destination.write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        extract_head(pack, destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == hashlib.sha256(b"retained").hexdigest()


def test_bf16_rounding_is_nearest_even_and_refuses_nonfinite():
    values = np.array([1.0, -2.5, 1.00390625, 1.01171875], dtype=np.float32)
    np.testing.assert_array_equal(bf16_to_floats(floats_to_bf16(values)), [1., -2.5, 1., 1.015625])
    with pytest.raises(ValueError, match="Nonfinite"):
        floats_to_bf16([float("nan")])


def test_q5_decoder_matches_installed_hf_reference():
    torch = pytest.importorskip("torch")
    from transformers.integrations.gguf.dequant import GGML_Q5_K, dequantize
    block = bytearray(np.random.default_rng(3).integers(0, 256, size=176 * 3, dtype=np.uint8).tobytes())
    for offset in range(0, len(block), 176):
        struct.pack_into("<ee", block, offset, 0.01, 0.02)
    actual = dequantize_q5_k(bytes(block))
    expected = dequantize(torch.tensor(list(block), dtype=torch.uint8), GGML_Q5_K).numpy()
    np.testing.assert_array_equal(actual, expected)
