#!/usr/bin/env python3
"""Read-only CPU comparison of native-style and training-style head arithmetic.

No model loading/control, GPU allocation, optimizer, source mutation or holdout
selection occurs. CPU BF16 kernels are diagnostic, not a bitwise GPU oracle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lucifer.meditation_worker import _dataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.limit <= 128:
        raise SystemExit("Use 1..128 bounded diagnostic rows per non-holdout split")
    if args.output.exists():
        raise SystemExit("Existing measurement retained; do not overwrite it")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    prepared = json.loads(args.prepared.read_text())
    shape = tuple(prepared["shape"])
    assert args.head.stat().st_size == np.prod(shape) * 2
    raw = np.memmap(args.head, dtype="<u2", mode="c", shape=shape)
    bf16 = torch.from_numpy(raw).view(torch.bfloat16)
    full = bf16.float()
    result = {"measured_at": datetime.now(timezone.utc).isoformat(), "device": "cpu", "threads": 4,
              "head_sha256": sha(args.head), "prepared_sha256": sha(args.prepared),
              "script_sha256": sha(Path(__file__)), "shape": list(shape),
              "rows_per_split_at_most": args.limit, "holdout_examined": False,
              "optimizer_or_weight_update": False, "cpu_bf16_is_bitwise_gpu_oracle": False, "datasets": {}}
    with torch.no_grad():
        for name in ("train", "development"):
            spec = prepared["datasets"][name]
            features, labels, _, dtype = _dataset(spec, shape)
            indices = np.linspace(0, len(labels) - 1, min(args.limit, len(labels)), dtype=np.int64)
            x = np.array(features[indices], copy=True)
            if dtype == "BF16":
                from lucifer.head_pack import bf16_to_floats
                x = bf16_to_floats(x)
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(np.array(labels[indices], copy=True)).long()
            native = torch.nn.functional.linear(x, full)
            rounded_inputs = torch.nn.functional.linear(x.bfloat16().float(), full)
            training = torch.nn.functional.linear(x.bfloat16(), bf16).float()
            variants = {"native_style_f32": native, "bf16_inputs_f32_logits": rounded_inputs,
                        "bf16_inputs_and_output": training, "f32_inputs_bf16_output": native.bfloat16().float()}
            values = {}
            reference_log = native.double().log_softmax(-1)
            reference_prob = reference_log.exp()
            for label, logits in variants.items():
                losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
                kl = (reference_prob * (reference_log - logits.double().log_softmax(-1))).sum(-1)
                values[label] = {"mean_loss": losses.mean().item(),
                                 "loss_delta_from_native": (losses - torch.nn.functional.cross_entropy(native, y, reduction="none")).mean().item(),
                                 "maximum_absolute_logit_delta": (logits - native).abs().max().item(),
                                 "mean_forward_kl_from_native": kl.mean().item(),
                                 "maximum_forward_kl_from_native": kl.max().item()}
            result["datasets"][name] = {"rows": len(indices), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                                        "variants": values}
            print(name, json.dumps(values), flush=True)
            del native, rounded_inputs, training, variants, reference_log, reference_prob
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
