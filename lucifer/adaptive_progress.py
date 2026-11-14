"""Typed heartbeats from an owned adaptive worker, without raw diagnostics.

Native Trainer global_step counts optimizer attempts, including AMP-overflow
skips, and can reset with a dataset revision. Only the worker's explicit
successful-update channel may advance Lucifer's cumulative progress.
"""
from __future__ import annotations

import json
import math


_INTEGER_FIELDS = ("global_successful_updates", "segment_successful_updates", "round_remaining",
                   "randint_min", "randint_max", "next_refresh_at")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate adaptive progress field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite adaptive progress constant")


def progress_events(line, *, segment_id, state, last_successful):
    """Return clean heartbeats, advancing no counters from malformed records.

    `state` is the immutable segment-start AdaptiveState (or its dictionary).
    `last_successful` is the last accepted worker count, initially the segment's
    committed global count. The caller carries the final accepted count into the
    next call; multiple records on one physical line are checked sequentially.
    """
    if not isinstance(line, str) or not isinstance(segment_id, str) or not segment_id:
        return []
    def field(name):
        return state.get(name) if isinstance(state, dict) else getattr(state, name, None)
    start = field("global_successful_updates")
    remaining = field("round_remaining")
    refresh = field("next_refresh_at")
    low, high = field("randint_min"), field("randint_max")
    if (any(type(value) is not int for value in (start, remaining, refresh, low, high, last_successful))
            or start < 0 or remaining < 1 or refresh <= start or low < 1 or high - low != 6):
        return []
    maximum = min(start + remaining, refresh)
    if not start <= last_successful <= maximum:
        return []
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    events, cursor, previous = [], 0, last_successful
    while (begin := line.find("{", cursor)) != -1:
        try:
            event, cursor = decoder.raw_decode(line, begin)
        except (ValueError, RecursionError):
            cursor = begin + 1
            continue
        if (not isinstance(event, dict) or event.get("event") != "adaptive_progress"
                or event.get("segment_id") != segment_id):
            continue
        if any(type(event.get(key)) is not int for key in _INTEGER_FIELDS):
            continue
        skipped = event.get("optimizer_skipped")
        epoch = event.get("native_epoch")
        if type(skipped) is not bool or type(epoch) not in (int, float) or not math.isfinite(epoch) or epoch < 0:
            continue
        current = event["global_successful_updates"]
        delta = current - start
        if (current != previous + (0 if skipped else 1) or not start <= current <= maximum
                or event["segment_successful_updates"] != delta
                or event["round_remaining"] != remaining - delta
                or event["randint_min"] != low or event["randint_max"] != high
                or event["next_refresh_at"] != refresh):
            continue
        events.append({"optimizer_step": current, "cumulative_optimizer_steps": current,
                       "committed_optimizer_steps": start, "round_remaining": event["round_remaining"],
                       "randint_min": low, "randint_max": high, "next_refresh_optimizer_step": refresh,
                       "native_epoch": epoch, "optimizer_skipped": skipped})
        previous = current
    return events
