"""GPU inventory without importing a GPU runtime into the control process.

Backend engines decide architecture support. A connected display adapter is not
necessarily usable by CUDA/ROCm. llama.cpp can target devices exposed by its
compiled Vulkan/Metal/CUDA backend, including heterogeneous Vulkan devices.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import subprocess


def _run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=4)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def discover_gpus() -> dict:
    gpus = []
    text = _run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
    for row in csv.reader(text.splitlines()):
        if len(row) != 7:
            continue
        index, ident, pci, name, total, used, free = [v.strip() for v in row]
        def memory(value):
            try:
                return int(float(value))
            except ValueError:
                return None
        gpus.append({"index": index, "id": ident, "pci_bus_id": pci.lower(), "name": name,
                     "vendor": "NVIDIA", "backend": "cuda", "compute_available": True,
                     "total_mb": memory(total), "used_mb": memory(used), "free_mb": memory(free)})
    raw = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    try:
        cards = json.loads(raw) if raw else {}
    except ValueError:
        cards = {}
    for card, values in cards.items():
        if not card.startswith("card") or not isinstance(values, dict):
            continue
        def megabytes(key):
            try:
                return int(values[key]) // (1024 * 1024)
            except (KeyError, TypeError, ValueError):
                return None
        total, used = megabytes("VRAM Total Memory (B)"), megabytes("VRAM Total Used Memory (B)")
        gpus.append({"index": card[4:], "id": card, "name": values.get("Card Series", values.get("Card Model", "AMD GPU")),
                     "vendor": "AMD", "backend": "rocm", "compute_available": True,
                     "total_mb": total, "used_mb": used, "free_mb": total - used if total is not None and used is not None else None})
    # Enumerate other connected adapters; avoid claiming a compute runtime exists.
    pci_text = _run(["lspci", "-Dmm"])
    for line in pci_text.splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) < 4 or not any(s in parts[1].lower() for s in ("vga", "3d controller", "display controller")):
            continue
        pci, _, vendor, name = parts[:4]
        if any(g.get("pci_bus_id", "").endswith(pci.lower()[4:]) for g in gpus):
            continue
        if "amd" in vendor.lower() and any(g["backend"] == "rocm" for g in gpus):
            continue
        gpus.append({"index": pci, "id": pci, "pci_bus_id": pci, "name": name, "vendor": vendor,
                     "backend": "vulkan", "compute_available": False,
                     "total_mb": None, "used_mb": None, "free_mb": None})
    if platform.system() == "Darwin":
        raw = _run(["system_profiler", "SPDisplaysDataType", "-json"])
        try:
            displays = json.loads(raw).get("SPDisplaysDataType", [])
        except ValueError:
            displays = []
        for index, display in enumerate(displays):
            gpus.append({"index": str(index), "id": f"metal:{index}", "name": display.get("sppci_model", "Apple GPU"),
                         "vendor": display.get("spdisplays_vendor", "Apple"), "backend": "metal", "compute_available": True,
                         "total_mb": None, "used_mb": None, "free_mb": None})
    return {"available": bool(gpus), "gpus": gpus,
            "cuda_device_count": sum(g["backend"] == "cuda" for g in gpus),
            "rocm_device_count": sum(g["backend"] == "rocm" for g in gpus)}


def training_placement(gpu_devices: list[str] | str = "all", multi_gpu: bool = True,
                       *, inventory: dict | None = None, environ: dict | None = None) -> dict:
    """Select physical CUDA/ROCm IDs, preserving inherited masks for 'all'.

    Explicit IDs can be physical indices or UUIDs. CPU is an empty list.
    Cross-vendor process groups are rejected; each engine needs one runtime.
    """
    inventory = discover_gpus() if inventory is None else inventory
    environ = os.environ if environ is None else environ
    available = [g for g in inventory.get("gpus", []) if g["backend"] in {"cuda", "rocm"}]
    if gpu_devices == "all":
        selected = []
        for backend, mask_name in (("cuda", "CUDA_VISIBLE_DEVICES"), ("rocm", "HIP_VISIBLE_DEVICES")):
            cards = [g for g in available if g["backend"] == backend]
            # ROCm also honors ROCR_VISIBLE_DEVICES; never widen an inherited restriction.
            mask = environ.get(mask_name)
            if backend == "rocm" and mask is None:
                mask = environ.get("ROCR_VISIBLE_DEVICES")
            if mask is None:
                selected.extend(cards)
            else:
                ids = [v.strip() for v in mask.split(",") if v.strip() and v.strip() != "-1"]
                selected.extend(g for g in cards if str(g["index"]) in ids or g["id"] in ids)
    else:
        if not isinstance(gpu_devices, list) or not all(isinstance(v, str) for v in gpu_devices):
            raise ValueError("gpu_devices must be 'all' or a list of GPU indices/UUIDs")
        if len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError("gpu_devices must not contain duplicates")
        selected = []
        for ident in gpu_devices:
            matches = [g for g in available if str(g["index"]) == ident or g["id"] == ident]
            if len(matches) != 1:
                raise ValueError(f"GPU {ident!r} is unavailable or ambiguous; use a UUID from /gpu list")
            if any(g["backend"] == matches[0]["backend"] and g["id"] == matches[0]["id"] for g in selected):
                raise ValueError("gpu_devices contains duplicate aliases for the same physical GPU")
            selected.append(matches[0])
    backends = {g["backend"] for g in selected}
    if len(backends) > 1:
        raise ValueError("A vLLM/training job cannot combine CUDA and ROCm GPUs. Select one vendor's devices; use a compatible llama.cpp Vulkan build for heterogeneous inference.")
    if not multi_gpu:
        selected = selected[:1]
    backend = next(iter(backends), "cpu")
    env = {}
    if backend != "cpu":
        mask = ",".join(str(g["index"]) for g in selected)
        env["CUDA_VISIBLE_DEVICES" if backend == "cuda" else "HIP_VISIBLE_DEVICES"] = mask
        if backend == "rocm":
            env["ROCR_VISIBLE_DEVICES"] = mask
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    return {"env": env, "device_count": len(selected), "backend": backend, "devices": selected}
