"""Shared GPU sampling: inventory once, changing counters in the background."""
from __future__ import annotations

import asyncio
import copy
import csv
import json
import time
from datetime import datetime, timezone

from .gpu import _run, discover_gpus


def number(value):
    try:
        return float(str(value).replace("%", "").replace("W", "").replace("c", "").strip())
    except (TypeError, ValueError):
        return None


def sample_gpus(inventory: dict) -> dict:
    """Read counters only for discovered vendors; missing metrics remain unknown."""
    result = copy.deepcopy(inventory)
    errors = []
    for card in result.get("gpus", []):
        card.update(utilization_pct=None, power_w=None, temperature_c=None, telemetry_available=False)
    if any(g.get("backend") == "cuda" for g in result.get("gpus", [])):
        raw = _run(["nvidia-smi", "--query-gpu=uuid,index,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"])
        matched = set()
        for row in csv.reader(raw.splitlines()):
            if len(row) != 8:
                continue
            ident, index, *values = [v.strip() for v in row]
            for card in result["gpus"]:
                if card.get("backend") == "cuda" and (card.get("id") == ident or str(card.get("index")) == index):
                    for key, value in zip(("total_mb", "used_mb", "free_mb", "utilization_pct", "power_w", "temperature_c"), values):
                        card[key] = number(value)
                    card["telemetry_available"] = True
                    matched.add(str(card.get("id")))
        if any(g.get("backend") == "cuda" and str(g.get("id")) not in matched for g in result["gpus"]):
            errors.append("Live NVIDIA counters unavailable for one or more GPUs")
    if any(g.get("backend") == "rocm" for g in result.get("gpus", [])):
        raw = _run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showpower", "--showtemp", "--json"])
        try:
            cards = json.loads(raw) if raw else {}
        except ValueError:
            cards = {}
        for card in result["gpus"]:
            if card.get("backend") != "rocm":
                continue
            values = cards.get("card" + str(card["index"]), {})
            if not values:
                errors.append(f"Live AMD counters unavailable for GPU {card['index']}")
                continue
            total, used = number(values.get("VRAM Total Memory (B)")), number(values.get("VRAM Total Used Memory (B)"))
            card.update(total_mb=total / 1048576 if total is not None else None,
                        used_mb=used / 1048576 if used is not None else None,
                        free_mb=(total - used) / 1048576 if total is not None and used is not None else None,
                        utilization_pct=number(values.get("GPU use (%)")), telemetry_available=True)
            for key, value in values.items():
                if "power" in key.lower() and "(w)" in key.lower():
                    card["power_w"] = number(value)
                if "temperature" in key.lower() and "edge" in key.lower():
                    card["temperature_c"] = number(value)
    result.update(sampled_at=datetime.now(timezone.utc).isoformat(), errors=errors,
                  stale=bool(errors))
    return result


class TelemetryService:
    def __init__(self, inventory: dict | None = None, *, interval: float = 1.0):
        self.inventory = inventory
        self.interval = interval
        self._snapshot: dict = {"gpus": [], "sampled_at": None, "stale": True, "errors": []}
        self._sample_time = 0.0
        self._task: asyncio.Task | None = None

    async def start(self):
        if self.inventory is None:
            self.inventory = await asyncio.to_thread(discover_gpus)
        await self.sample()
        self._task = asyncio.create_task(self._run(), name="lucifer-gpu-telemetry")

    async def sample(self):
        try:
            self._snapshot = await asyncio.to_thread(sample_gpus, self.inventory)
            self._sample_time = time.monotonic()
        except Exception as exc:
            self._snapshot = {**self._snapshot, "stale": True, "errors": [str(exc)]}

    async def _run(self):
        while True:
            await asyncio.sleep(self.interval)
            await self.sample()

    def snapshot(self) -> dict:
        result = copy.deepcopy(self._snapshot)
        result["age_seconds"] = round(time.monotonic() - self._sample_time, 2) if self._sample_time else None
        result["stale"] = result["stale"] or not self._sample_time or result["age_seconds"] > max(5, self.interval * 3)
        return result

    async def close(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
