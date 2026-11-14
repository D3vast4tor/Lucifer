#!/usr/bin/env python3
"""Read-only capacity assessment of the installed Lucifer pack; never trains."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = PROJECT / "var/strata/models/packs/deepseek-v4.1-flash-ist-das-2.00bpw"


def assess(pack, api_url):
    manifest = pack / "tensors.txt"
    raw = manifest.read_bytes()
    groups, payload, tensors = defaultdict(int), defaultdict(int), Counter()
    selected, aliases, seen = {}, [], set()
    for line in raw.decode().splitlines():
        fields = line.split()
        if not fields or fields[0] != "t":
            continue
        name, dtype = fields[1:3]
        index, offset, size = map(int, fields[3:6])
        shape = list(map(int, fields[6:]))
        key = index, offset, size
        if key in seen:
            aliases.append(name)
        seen.add(key)
        groups[index] += math.prod(shape)
        payload[index] += size
        tensors[index] += 1
        if name in {"head.weight", "embed.weight", "norm.weight"}:
            selected[name] = {"dtype": dtype, "file_index": index, "offset": offset,
                              "stored_bytes": size, "shape": shape, "parameters": math.prod(shape)}
    primary = sum(groups[index] for index in range(5))
    head = selected["head.weight"]["parameters"]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key] = int(value.split()[0]) * 1024
    status, gpu, errors = {}, {}, []
    for route in ("/system/status", "/system/gpu"):
        try:
            request = urllib.request.Request(api_url.rstrip("/") + route)
            with urllib.request.urlopen(request, timeout=5) as response:
                value = json.load(response)
            if route.endswith("status"):
                model = value.get("model") or {}
                status = {"agent_busy": value.get("agent_busy"), "learning_busy": value.get("learning_busy"),
                          "model_id": model.get("id"), "backend": model.get("backend"), "path": model.get("path")}
            else:
                gpu = value
        except (OSError, ValueError) as error:
            errors.append({"route": route, "error_type": type(error).__name__})
    free = shutil.disk_usage(PROJECT)
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), None)
    full = {"bf16_weights": primary * 2, "bf16_gradients": primary * 2,
            "two_fp32_adam_moments": primary * 8, "optional_fp32_master": primary * 4,
            "bf16_weights_gradients_fp32_moments_total": primary * 12,
            "same_with_fp32_master_total": primary * 16}
    partial = {"bf16_head_weights": head * 2, "bf16_head_gradients": head * 2,
               "cpu_fp32_master": head * 4, "cpu_two_fp32_moments": head * 8,
               "mixed_weights_gradients_moments_master_total": head * 16,
               "extra_cpu_fp32_gradient_copy": head * 4,
               "serving_bf16_increase_over_current_head": head * 2 - selected["head.weight"]["stored_bytes"],
               "bf16_features_per_token": selected["head.weight"]["shape"][1] * 2,
               "f32_features_per_token": selected["head.weight"]["shape"][1] * 4,
               "bf16_features_100000_tokens": selected["head.weight"]["shape"][1] * 2 * 100000}
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "read_only": True,
            "training_or_gpu_workload_started": False, "storage_throughput_measured": False,
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "parameters_by_file": dict(groups), "quantized_payload_bytes_by_file": dict(payload),
            "tensor_count_by_file": dict(tensors), "duplicate_storage_aliases": aliases,
            "primary_parameters": primary, "all_pack_parameters": sum(groups.values()),
            "selected_tensors": selected, "cpu": cpu, "memory_bytes": memory,
            "filesystem_bytes": {"total": free.total, "used": free.used, "available": free.free},
            "gpu_cached_snapshot": gpu, "runtime": status, "api_errors": errors,
            "full_training_state_bytes": full, "partial_head_training_bytes": partial,
            "decisions": {
                "ordinary_full_weight_adam_training": "not_feasible_on_observed_machine",
                "head_only_training": "capacity_feasible_engineering_pilot_not_end_to_end_validated",
                "other_custom_low_state_full_training": "unvalidated_research_not_proven_impossible"},
            "caveats": [
                "12/16-byte totals are explicit Adam scenarios, not universal lower bounds.",
                "Activations, framework workspace, extra copies and checkpoints are additional.",
                "Inference memory use does not establish backward or optimizer compatibility.",
                "The head-only route modifies a subset of actual Lucifer weights, not the entire model.",
                "Training would require an idle model slot and separate capture/train/reload phases."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = assess(args.pack.resolve(), args.api_url)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
