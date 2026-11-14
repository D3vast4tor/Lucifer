"""Strict, bounded I/O for native Lucifer output-head candidate packs.

This module never writes parent weights and imports numerical packages only in
conversion functions. GGML Q5_K layout: 256 values in 176 bytes (two FP16
scales, twelve packed scale/min bytes, 32 high-bit bytes, 128 nibble bytes).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    file_index: int
    offset: int
    size: int
    shape: tuple[int, ...]

    @property
    def parameters(self):
        return math.prod(self.shape)


@dataclass(frozen=True)
class PackManifest:
    path: Path
    model_dir: Path
    files: dict[int, Path]
    tensors: dict[str, TensorSpec]
    metadata: dict[str, str]
    manifest_sha256: str

    @property
    def head(self):
        return self.tensors["head.weight"]


def file_hash(path, *, offset=0, size=None, chunk_size=4 << 20):
    path = Path(path)
    if offset < 0 or size is not None and size < 0:
        raise ValueError("Invalid hash range")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = path.stat().st_size - offset if size is None else size
        if remaining < 0:
            raise ValueError("Hash range exceeds file")
        while remaining:
            data = handle.read(min(chunk_size, remaining))
            if not data:
                raise ValueError("Tensor/file ended before declared range")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def load_pack(path) -> PackManifest:
    path = Path(path).resolve()
    raw = (path / "tensors.txt").read_bytes()
    if len(raw) > 32 << 20:
        raise ValueError("Manifest exceeds supported size")
    metadata = {}
    for line in (path / "ds4.txt").read_text().splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(" ")
        if not sep or key in metadata:
            raise ValueError("Invalid/duplicate pack metadata")
        metadata[key] = value
    model_dir = Path(metadata.get("model_dir", ""))
    if not model_dir.is_absolute() or not model_dir.is_dir():
        raise ValueError("Pack model_dir must be an existing absolute directory")
    files, tensors = {}, {}
    for line in raw.decode().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "file":
            if len(fields) != 3:
                raise ValueError("Invalid file manifest row")
            index = int(fields[1])
            if index < 0 or index in files or "\0" in fields[2]:
                raise ValueError("Invalid/duplicate manifest file id")
            if Path(fields[2]).is_absolute():
                raise ValueError("Native loader requires relative file entries")
            files[index] = (model_dir / fields[2]).resolve()
        elif fields[0] == "t":
            if len(fields) < 7:
                raise ValueError("Invalid tensor manifest row")
            name, dtype = fields[1:3]
            fi, offset, size = map(int, fields[3:6])
            shape = tuple(map(int, fields[6:]))
            if name in tensors or fi < 0 or offset < 0 or size <= 0 or any(d <= 0 for d in shape):
                raise ValueError("Duplicate tensor or invalid range/shape")
            tensors[name] = TensorSpec(name, dtype, fi, offset, size, shape)
        else:
            raise ValueError("Unknown manifest record")
    if not files or set(files) != set(range(max(files) + 1)):
        raise ValueError("Manifest file ids must be contiguous")
    ranges = {}
    for tensor in tensors.values():
        source = files.get(tensor.file_index)
        if source is None or not source.is_file() or tensor.offset + tensor.size > source.stat().st_size:
            raise ValueError("Tensor range exceeds its declared file")
        ranges.setdefault(tensor.file_index, []).append((tensor.offset, tensor.offset + tensor.size))
    for spans in ranges.values():
        spans.sort()
        if any(current[0] < previous[1] for previous, current in zip(spans, spans[1:])):
            raise ValueError("Overlapping tensor storage is ambiguous")
    if "head.weight" not in tensors or len(tensors["head.weight"].shape) != 2:
        raise ValueError("Pack lacks a matrix head.weight")
    return PackManifest(path, model_dir, files, tensors, metadata, hashlib.sha256(raw).hexdigest())


def read_tensor(pack: PackManifest, name: str, *, max_bytes: int = 1 << 20):
    tensor = pack.tensors[name]
    if tensor.size > max_bytes:
        raise ValueError("Tensor exceeds bounded read limit; use streaming")
    with pack.files[tensor.file_index].open("rb") as handle:
        handle.seek(tensor.offset)
        data = handle.read(tensor.size)
    if len(data) != tensor.size:
        raise ValueError("Truncated tensor")
    return data


def tensor_hash(pack: PackManifest, name="head.weight"):
    tensor = pack.tensors[name]
    return file_hash(pack.files[tensor.file_index], offset=tensor.offset, size=tensor.size)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract_head(pack: PackManifest, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise FileExistsError(destination)
    head = pack.head
    digest = hashlib.sha256()
    try:
        with pack.files[head.file_index].open("rb") as source, destination.open("xb") as target:
            destination.chmod(0o600)
            source.seek(head.offset)
            remaining = head.size
            while remaining:
                data = source.read(min(4 << 20, remaining))
                if not data:
                    raise ValueError("Truncated source head")
                target.write(data)
                digest.update(data)
                remaining -= len(data)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"path": str(destination.resolve()), "sha256": digest.hexdigest(), "dtype": head.dtype,
            "shape": list(head.shape), "bytes": head.size, "parent_manifest_sha256": pack.manifest_sha256}


def dequantize_q5_k(data, shape=None):
    """Decode standard GGML Q5_K blocks to F32, independently of model loaders."""
    import numpy as np
    blocks = np.frombuffer(data, dtype=np.uint8)
    if not len(blocks) or len(blocks) % 176:
        raise ValueError("Q5_K requires complete 176-byte blocks")
    blocks = blocks.reshape(-1, 176)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    scales = blocks[:, 4:16]
    scale = np.concatenate((scales[:, :4] & 63,
                            (scales[:, 8:12] & 15) | ((scales[:, :4] >> 6) << 4)), axis=1).astype(np.float32)
    minimum = np.concatenate((scales[:, 4:8] & 63,
                              (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)), axis=1).astype(np.float32)
    lo = blocks[:, 48:176].reshape(-1, 4, 1, 32)
    lo = ((lo >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(-1, 8, 32)
    hi = ((blocks[:, 16:48, None].transpose(0, 2, 1) >>
           np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1)
    q = lo | (hi << 4)
    result = ((d * scale)[:, :, None] * q - (dmin * minimum)[:, :, None]).reshape(-1)
    if shape is not None:
        if math.prod(shape) != result.size:
            raise ValueError("Q5_K bytes do not match requested shape")
        result = result.reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite dequantized head")
    return result


def floats_to_bf16(values):
    import numpy as np
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite head/features")
    bits = values.view("<u4")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")


def bf16_to_floats(values):
    import numpy as np
    return (np.asarray(values, dtype="<u2").astype("<u4") << 16).view("<f4")


def export_head_bf16(pack: PackManifest, destination, *, rows_per_chunk=64):
    import numpy as np
    head = pack.head
    rows, width = head.shape
    if rows_per_chunk < 1 or rows_per_chunk > 1024:
        raise ValueError("Invalid conversion chunk")
    if head.dtype == "Q5_K" and (width % 256 or head.size != rows * width // 256 * 176):
        raise ValueError("Unexpected Q5_K head layout")
    if head.dtype not in {"Q5_K", "BF16", "F32"}:
        raise ValueError("Unsupported training head dtype")
    bytes_per_row = width // 256 * 176 if head.dtype == "Q5_K" else width * (2 if head.dtype == "BF16" else 4)
    if head.size != rows * bytes_per_row:
        raise ValueError("Head size does not match dtype/shape")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise FileExistsError(destination)
    original = hashlib.sha256()
    candidate = hashlib.sha256()
    try:
        with pack.files[head.file_index].open("rb") as source, destination.open("xb") as target:
            destination.chmod(0o600)
            source.seek(head.offset)
            for row in range(0, rows, rows_per_chunk):
                count = min(rows_per_chunk, rows - row)
                data = source.read(bytes_per_row * count)
                if len(data) != bytes_per_row * count:
                    raise ValueError("Truncated head during conversion")
                original.update(data)
                if head.dtype == "BF16":
                    output = data
                else:
                    values = (dequantize_q5_k(data) if head.dtype == "Q5_K" else
                              np.frombuffer(data, dtype="<f4"))
                    output = floats_to_bf16(values).tobytes()
                candidate.update(output)
                target.write(output)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"path": str(destination.resolve()), "sha256": candidate.hexdigest(), "dtype": "BF16",
            "shape": list(head.shape), "bytes": rows * width * 2, "source_head_sha256": original.hexdigest(),
            "parent_manifest_sha256": pack.manifest_sha256, "conversion_only": True}


def create_head_pack(parent: PackManifest, destination, head_path, *, dtype="BF16"):
    """Create a complete candidate pack referencing immutable original shards."""
    if not isinstance(parent, PackManifest):
        parent = load_pack(parent)
    head_path = Path(head_path).resolve()
    destination = Path(destination).resolve()
    if dtype not in {"BF16", "Q5_K"}:
        raise ValueError("Unsupported native candidate dtype")
    rows, width = parent.head.shape
    expected = rows * width * 2 if dtype == "BF16" else rows * width // 256 * 176
    if dtype == "Q5_K" and width % 256 or head_path.stat().st_size != expected:
        raise ValueError("Candidate head size/dtype mismatch")
    if head_path in parent.files.values():
        raise ValueError("A candidate head must be separate from original shards")
    if destination.exists():
        raise FileExistsError(destination)
    if not (parent.path / "engram.bin").is_file():
        raise ValueError("Parent pack lacks mandatory engram.bin")
    destination.mkdir(parents=True, mode=0o700)
    try:
        new_index = max(parent.files) + 1
        relative = os.path.relpath(head_path, parent.model_dir)
        if any(c.isspace() for c in relative):
            raise ValueError("Native manifest paths cannot contain whitespace")
        lines = (parent.path / "tensors.txt").read_text().splitlines()
        replacement = f"t head.weight {dtype} {new_index} 0 {expected} {rows} {width}"
        found = 0
        for index, line in enumerate(lines):
            fields = line.split()
            if len(fields) > 1 and fields[:2] == ["t", "head.weight"]:
                lines[index] = replacement
                found += 1
        if found != 1:
            raise ValueError("Expected exactly one original head row")
        lines.append(f"file {new_index} {relative}")
        (destination / "tensors.txt").write_text("\n".join(lines) + "\n")
        (destination / "ds4.txt").write_text((parent.path / "ds4.txt").read_text())
        for name in ("engram.bin", "tokenizer"):
            if (parent.path / name).exists():
                (destination / name).symlink_to(parent.path / name, target_is_directory=(parent.path / name).is_dir())
        if (parent.path / "expert-profile.bin").is_file():
            shutil.copyfile(parent.path / "expert-profile.bin", destination / "expert-profile.bin")
        for name in ("tensors.txt", "ds4.txt", "expert-profile.bin"):
            if (destination / name).exists():
                (destination / name).chmod(0o600)
        candidate = load_pack(destination)
        receipt = {"path": str(destination), "head_path": str(head_path), "head_sha256": file_hash(head_path),
                   "head_dtype": dtype, "shape": [rows, width], "manifest_sha256": candidate.manifest_sha256,
                   "parent_pack": str(parent.path), "parent_manifest_sha256": parent.manifest_sha256,
                   "frozen_shards_reused": all(candidate.files[i] == p for i, p in parent.files.items())}
        atomic_json(destination / "head-receipt.json", receipt)
        return receipt
    except BaseException:
        shutil.rmtree(destination)
        raise
