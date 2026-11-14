"""Conservative storage admission before creating an owned training artifact."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat


class StorageLimitError(RuntimeError):
    pass


def checkpoint_reserve(shape):
    if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
        raise ValueError("Invalid checkpoint shape")
    # Three FP32 master/moment arrays, exported BF16 head, metadata margin.
    return shape[0] * shape[1] * 14 + (64 << 20)


def feature_reserve(datasets, dimension, *, raw_limit_bytes=4 << 30):
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Invalid feature dimension")
    if type(raw_limit_bytes) is not int or raw_limit_bytes < 0:
        raise ValueError("Invalid native trajectory reservation")
    targets = sum(label != -100 for segments in datasets.values() for segment in segments
                  for label in segment["labels"][1:])
    # Compact F32 features, I64 labels, conservative row/provenance metadata,
    # plus one bounded all-layer trajectory and fixed diagnostic/config margin.
    return targets * (dimension * 4 + 8 + 1024) + raw_limit_bytes + (512 << 20)


def require_room(run_dir, *, reserve_bytes=0, limit_bytes=50_000_000_000, free_reserve_bytes=1 << 30):
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Invalid storage reservation")
    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Storage admission requires an existing owned directory")
    used = 0
    def failed(error):
        raise error
    for base, directories, files in os.walk(root, followlinks=False, onerror=failed):
        directories[:] = [name for name in directories if not (Path(base) / name).is_symlink()]
        for name in files:
            try:
                info = (Path(base) / name).lstat()
            except FileNotFoundError:
                continue  # Concurrent source revocation may remove files.
            if stat.S_ISREG(info.st_mode):
                used += info.st_size
    free = shutil.disk_usage(root).free
    if used + reserve_bytes > limit_bytes or free < reserve_bytes + free_reserve_bytes:
        raise StorageLimitError("Meditation cannot reserve its next artifact within the disk/cache budget")
    return {"used_bytes": used, "reserved_bytes": reserve_bytes, "limit_bytes": limit_bytes, "free_bytes": free}
