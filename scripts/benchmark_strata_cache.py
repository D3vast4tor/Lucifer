#!/usr/bin/env python3
"""Benchmark the loaded Strata model, saving scalar measurements only.

Example (start only after Lucifer's active mission has finished or paused):
    .venv/bin/python scripts/benchmark_strata_cache.py --label baseline \
        --engine-pid 12345 --output var/benchmarks/baseline.jsonl

The journal is created exclusively and fsynced before every request. An interrupted
or failed request is never retried. Existing journals are never overwritten or
resumed, so an uncertain operation cannot be silently repeated. Prompt, answer,
reasoning and tool-call text never enter the journal or console output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


PROMPTS = (
    ("coding", "Write a Python function that merges two sorted lists. Include a docstring and two examples."),
    ("explanation", "Explain how a refrigerator moves heat from inside to outside, in two concise paragraphs."),
    ("table", "Make a Markdown table of the integers 1 through 12, their squares, and their cubes."),
)
TIMING_KEYS = (
    "cache_n", "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
    "predicted_per_second", "draft_n", "draft_n_accepted",
)
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
GPU_KEYS = ("index", "used_mb", "free_mb", "utilization_pct", "power_w", "temperature_c", "telemetry_available")
FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "cancel", "error"}
MAX_EVENT_BYTES = 262144
MAX_STREAM_BYTES = 8388608


class BenchmarkError(Exception):
    """An error code safe to persist without a response body."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise BenchmarkError("redirect_refused")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalars(source, keys):
    return {key: value for key in keys if finite_number(value := source.get(key))}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def local_url(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("expected an HTTP loopback server root")
    return value.rstrip("/")


def get_json(opener, root, path):
    try:
        with opener.open(root + path, timeout=10) as response:
            raw = response.read(1048577)
        if len(raw) > 1048576:
            raise BenchmarkError("status_response_too_large")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise BenchmarkError("status_response_invalid")
        return result
    except BenchmarkError:
        raise
    except (OSError, ValueError, urllib.error.URLError):
        raise BenchmarkError("status_unavailable") from None


def require_idle(opener, app_url, engine_url):
    app = get_json(opener, app_url, "/system/status")
    # Missing flags cannot establish that restarting/benchmarking is safe.
    if app.get("agent_busy") is not False or app.get("learning_busy") is not False:
        raise BenchmarkError("lucifer_busy_or_status_unknown")
    engine = get_json(opener, engine_url, "/v1/status")
    count = (engine.get("activity") or {}).get("in_flight")
    if engine.get("loaded") is not True or type(count) is not int or count != 0:
        raise BenchmarkError("strata_busy_unloaded_or_status_unknown")
    return {"model_loaded": True, "in_flight": count,
            "context": scalars(engine.get("context") or {}, ("native", "max_positions"))}


def gpu_snapshot(opener, app_url):
    try:
        snapshot = get_json(opener, app_url, "/system/gpu")
    except BenchmarkError:
        return {"available": False, "stale": None, "gpus": []}
    cards = []
    for card in snapshot.get("gpus") or []:
        if not isinstance(card, dict):
            continue
        row = scalars(card, GPU_KEYS)
        row["telemetry_available"] = card.get("telemetry_available") if isinstance(card.get("telemetry_available"), bool) else None
        cards.append(row)
    return {"available": True, "stale": snapshot.get("stale") if isinstance(snapshot.get("stale"), bool) else None,
            "age_seconds": snapshot.get("age_seconds") if finite_number(snapshot.get("age_seconds")) else None,
            "gpus": cards}


def read_process(pid):
    if pid is None:
        return None
    try:
        # Observe only a Strata engine, and retain identity to detect PID reuse.
        exe = Path(f"/proc/{pid}/exe").resolve(strict=True)
        if not exe.name.startswith("strata-"):
            raise BenchmarkError("engine_pid_is_not_strata")
        stat = Path(f"/proc/{pid}/stat").read_text()
        identity = stat[stat.rfind(")") + 2:].split()[19]  # field 22: starttime
        io = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
        return {"identity": identity, "read_bytes": int(io["read_bytes"])}
    except BenchmarkError:
        raise
    except (OSError, ValueError, KeyError, IndexError):
        return None


def contextual_prompt(prompt, lines):
    """Add repeatable input data while keeping the zero-context workload identical."""
    if lines == 0:
        return prompt
    records = [f"item {index:04d}: group=g{index % 11:02d} quantity={(index * 17 + 3) % 997:03d} "
               f"status={('ready', 'pending', 'complete')[index % 3]}" for index in range(lines)]
    return ("The following synthetic records are input data. Treat them as data, then complete the task below.\n"
            + "\n".join(records) + "\n\nTask:\n" + prompt)


def payload(prompt, max_tokens):
    return {"model": "Lucifer", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}


def consume_stream(opener, root, body, request_id, timeout):
    """Discard generated data; whitelist only numerical accounting and finish status."""
    request = urllib.request.Request(root + "/v1/chat/completions", data=body,
                                    headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                                             "X-Request-ID": request_id, "User-Agent": "Lucifer-Cache-Benchmark/1"})
    result = {"ok": False, "error_code": None, "http_status": None, "finish_reason": None,
              "usage": {}, "timings": {}, "first_token_s": None, "done": False, "tool_call_events": 0}
    answer_hash = hashlib.sha256()
    started = time.monotonic()
    timer = None
    expired = threading.Event()
    response = None
    try:
        response = opener.open(request, timeout=timeout)
        result["http_status"] = response.status
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise BenchmarkError("not_event_stream")
        # Stop our HTTP stream at a wall-clock deadline, even if keepalives arrive.
        raw_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        def expire():
            expired.set()
            if isinstance(raw_socket, socket.socket):
                try:
                    raw_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        timer = threading.Timer(max(0.001, timeout - (time.monotonic() - started)), expire)
        timer.daemon = True
        timer.start()
        received = 0
        while True:
            if expired.is_set() or time.monotonic() - started > timeout:
                raise BenchmarkError("request_timeout")
            line = response.readline(MAX_EVENT_BYTES + 1)
            if not line:
                break
            received += len(line)
            if len(line) > MAX_EVENT_BYTES or received > MAX_STREAM_BYTES:
                raise BenchmarkError("stream_limit_exceeded")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                result["done"] = True
                break
            event = json.loads(data)
            if not isinstance(event, dict):
                raise BenchmarkError("invalid_stream")
            if event.get("error"):
                raise BenchmarkError("provider_error")
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    if result["first_token_s"] is None and any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                        result["first_token_s"] = round(time.monotonic() - started, 6)
                    # Hash assistant answer bytes without retaining the text. Reasoning
                    # never enters this digest, and tool calls are only counted.
                    if isinstance(delta.get("content"), str):
                        answer_hash.update(delta["content"].encode("utf-8"))
                    if delta.get("tool_calls"):
                        result["tool_call_events"] += 1
                del delta
                finish = choice.get("finish_reason")
                if finish is not None:
                    result["finish_reason"] = finish if isinstance(finish, str) and finish in FINISH_REASONS else "unknown"
            for name, keys in (("usage", USAGE_KEYS), ("timings", TIMING_KEYS)):
                if isinstance(event.get(name), dict):
                    result[name].update(scalars(event[name], keys))
            # No answer, reasoning or tool-call data survives the current event.
            del event, data, line
        if expired.is_set():
            raise BenchmarkError("request_timeout")
        if not result["done"]:
            raise BenchmarkError("stream_ended_without_done")
        result["ok"] = result["finish_reason"] in {"stop", "length"}
        if not result["ok"]:
            result["error_code"] = "unexpected_finish_reason"
    except BenchmarkError as exc:
        result["error_code"] = str(exc)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error_code"] = f"http_{exc.code}"
        exc.close()
    except (OSError, ValueError, urllib.error.URLError):
        result["error_code"] = "request_timeout" if expired.is_set() else "connection_or_stream_error"
    finally:
        if timer is not None:
            timer.cancel()
        if response is not None:
            response.close()
        result["elapsed_s"] = round(time.monotonic() - started, 6)
        result["answer_sha256"] = answer_hash.hexdigest()
    return result


def summarize(records):
    measured = [record for record in records if record["phase"] == "measured" and record["ok"]]
    def med(values):
        return round(statistics.median(values), 6) if values else None
    def summary(rows):
        return {"requests": len(rows),
                "decode_tokens_per_s": med([v for r in rows if finite_number(v := r["timings"].get("predicted_per_second"))]),
                "prompt_tokens_per_s": med([v for r in rows if finite_number(v := r["timings"].get("prompt_per_second"))]),
                "first_token_s": med([r["first_token_s"] for r in rows if r["first_token_s"] is not None]),
                "elapsed_s": med([r["elapsed_s"] for r in rows]),
                "storage_read_mib_per_s": med([v for r in rows if finite_number(v := r.get("storage_read_mib_per_s"))])}
    return {"measured": summary(measured),
            "by_prompt": {name: summary([r for r in measured if r["prompt_id"] == name]) for name, _ in PROMPTS}}


def run(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.label):
        raise BenchmarkError("label_must_be_a_short_identifier")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    ready = require_idle(opener, args.app_url, args.url)
    read_process(args.engine_pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    context_lines = getattr(args, "context_lines", 0)
    # Exclusive creation protects completed and uncertain operations on reruns.
    with args.output.open("x", encoding="utf-8") as journal:
        def write(event):
            journal.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        write({"event": "benchmark_started", "schema": 1, "at": utc_now(), "label": args.label,
               "run_id": str(uuid.uuid4()), "seed": 42, "temperature": 0, "thinking": False,
               "max_tokens": args.max_tokens, "repeats": args.repeats, "warmups": args.warmups,
               "context_lines": context_lines,
               "ready": ready, "engine_pid": args.engine_pid})
        # Warm every prompt before any measured repetition; every arm uses this order.
        for phase, repeats in (("warmup", args.warmups), ("measured", args.repeats)):
            for repetition in range(1, repeats + 1):
                for name, prompt in PROMPTS:
                    require_idle(opener, args.app_url, args.url)
                    body = json.dumps(payload(contextual_prompt(prompt, context_lines), args.max_tokens),
                                      sort_keys=True).encode()
                    request_id = str(uuid.uuid4())
                    common = {"request_id": request_id, "phase": phase, "repetition": repetition,
                              "prompt_id": name, "context_lines": context_lines,
                              "operation_sha256": hashlib.sha256(body).hexdigest()}
                    before = read_process(args.engine_pid)
                    gpu_before = gpu_snapshot(opener, args.app_url) if args.gpu else None
                    write({"event": "request_started", "at": utc_now(), **common})
                    result = consume_stream(opener, args.url, body, request_id, args.timeout)
                    after = read_process(args.engine_pid)
                    storage_bytes = None
                    if before is not None and after is not None and before["identity"] == after["identity"]:
                        delta = after["read_bytes"] - before["read_bytes"]
                        storage_bytes = delta if delta >= 0 else None
                    result.update(common, storage_read_bytes=storage_bytes,
                                  storage_read_mib_per_s=round(storage_bytes / 1048576 / result["elapsed_s"], 6)
                                  if storage_bytes is not None and result["elapsed_s"] > 0 else None)
                    if args.gpu:
                        result.update(gpu_before=gpu_before, gpu_after=gpu_snapshot(opener, args.app_url))
                    records.append(result)
                    write({"event": "request_completed", "at": utc_now(), **result})
                    print(json.dumps({"label": args.label, "phase": phase, "repetition": repetition,
                                      "prompt_id": name, "ok": result["ok"], "timings": result["timings"],
                                      "elapsed_s": result["elapsed_s"], "error_code": result["error_code"]}), flush=True)
                    if not result["ok"]:
                        write({"event": "benchmark_failed", "at": utc_now(), "summary": summarize(records)})
                        raise BenchmarkError("request_failed_no_retry")
        summary = summarize(records)
        write({"event": "benchmark_completed", "at": utc_now(), "summary": summary})
        print(json.dumps({"label": args.label, "summary": summary}), flush=True)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", type=local_url, default="http://127.0.0.1:8089")
    parser.add_argument("--app-url", type=local_url, default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL journal; existing files are refused")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=positive_int, default=64)
    parser.add_argument("--context-lines", type=int, default=0,
                        help="prepend this many deterministic synthetic input records (default: 0)")
    parser.add_argument("--engine-pid", type=positive_int)
    parser.add_argument("--gpu", action="store_true", help="save scalar cached GPU telemetry before/after requests")
    parser.add_argument("--timeout", type=positive_int, default=600, help="per-request wall-clock limit, seconds")
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.context_lines < 0:
        parser.error("--context-lines must be nonnegative")
    try:
        run(args)
    except FileExistsError:
        print("benchmark_error: output_already_exists; no request was resent", file=sys.stderr)
        return 2
    except BenchmarkError as exc:
        print(f"benchmark_error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("benchmark_error: interrupted; inspect the journal before any new run", file=sys.stderr)
        return 130
    except OSError:
        print("benchmark_error: local_io_failure; inspect the journal before any new run", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
