"""Privacy and no-resend checks for the numeric-only cache benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "benchmark_strata_cache", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_strata_cache.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}


def stream(*events):
    return Response(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
                    + b"data: [DONE]\n\n")


class Opener:
    def __init__(self, *, busy=False, provider_error=False):
        self.busy = busy
        self.provider_error = provider_error
        self.requests = 0

    def open(self, request, timeout):
        if isinstance(request, str):
            if request.endswith("/system/status"):
                data = {"agent_busy": self.busy, "learning_busy": False}
            else:
                data = {"loaded": True, "activity": {"in_flight": 0}, "context": {"native": 1048576}}
            return Response(json.dumps(data).encode())
        self.requests += 1
        if self.provider_error:
            return stream({"error": {"message": "PRIVATE_REASONING_AND_ERROR"}})
        return stream(
            {"choices": [{"delta": {"reasoning_content": "PRIVATE_REASONING"}}]},
            {"choices": [{"delta": {"content": "PRIVATE_ANSWER"}}]},
            {"choices": [{"delta": {"content": "_TAIL"}, "finish_reason": "length"}]},
            {"usage": {"prompt_tokens": 19, "completion_tokens": 64, "private": "PRIVATE_USAGE"},
             "timings": {"predicted_per_second": 2.0, "predicted_ms": 32000, "private": "PRIVATE_TIMING"}})


def args(tmp_path):
    return argparse.Namespace(url="http://127.0.0.1:8089", app_url="http://127.0.0.1:8000",
                              label="baseline", output=tmp_path / "evidence.jsonl", repeats=1,
                              warmups=0, max_tokens=64, engine_pid=None, gpu=False, timeout=10)


def test_journal_excludes_generated_text_and_has_intent_before_effect(tmp_path, monkeypatch, capsys):
    options = args(tmp_path)
    opener = Opener()
    monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *_: opener)
    bench.run(options)
    saved = options.output.read_text()
    assert "PRIVATE" not in saved
    assert "PRIVATE" not in capsys.readouterr().out
    records = [json.loads(line) for line in saved.splitlines()]
    completed = [row for row in records if row["event"] == "request_completed"]
    assert len(completed) == len(bench.PROMPTS)
    assert all(row["answer_sha256"] == hashlib.sha256(b"PRIVATE_ANSWER_TAIL").hexdigest() for row in completed)
    for index, row in enumerate(records):
        if row["event"] == "request_completed":
            assert records[index - 1]["event"] == "request_started"
            assert records[index - 1]["request_id"] == row["request_id"]
            assert len(row["operation_sha256"]) == 64
    assert records[-1]["summary"]["measured"]["decode_tokens_per_s"] == 2.0
    with pytest.raises(FileExistsError):
        bench.run(options)
    assert opener.requests == len(bench.PROMPTS)


def test_busy_guard_makes_no_inference_or_journal(tmp_path, monkeypatch):
    options = args(tmp_path)
    opener = Opener(busy=True)
    monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *_: opener)
    with pytest.raises(bench.BenchmarkError, match="lucifer_busy"):
        bench.run(options)
    assert opener.requests == 0
    assert not options.output.exists()


def test_provider_failure_is_sanitized_and_not_retried(tmp_path, monkeypatch, capsys):
    options = args(tmp_path)
    opener = Opener(provider_error=True)
    monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *_: opener)
    with pytest.raises(bench.BenchmarkError, match="request_failed_no_retry"):
        bench.run(options)
    assert opener.requests == 1
    assert "PRIVATE" not in options.output.read_text()
    assert "PRIVATE" not in capsys.readouterr().out
    records = [json.loads(line) for line in options.output.read_text().splitlines()]
    assert records[-2]["error_code"] == "provider_error"
    assert records[-1]["event"] == "benchmark_failed"


def test_tool_calls_are_counted_without_retaining_or_executing_them():
    class ToolOpener:
        def open(self, request, timeout):
            return stream({"choices": [{"delta": {"tool_calls": [{"function": {
                "name": "PRIVATE_TOOL", "arguments": "PRIVATE_ARGUMENTS"}}]}, "finish_reason": "tool_calls"}]})
    result = bench.consume_stream(ToolOpener(), "http://127.0.0.1:8089", b"{}", "request", 10)
    assert result["tool_call_events"] == 1
    assert not result["ok"]
    assert result["finish_reason"] == "tool_calls"
    assert "PRIVATE" not in json.dumps(result)


def test_context_records_are_deterministic_and_zero_context_is_identical():
    prompt = bench.PROMPTS[0][1]
    assert bench.contextual_prompt(prompt, 0) == prompt
    expanded = bench.contextual_prompt(prompt, 100)
    assert expanded == bench.contextual_prompt(prompt, 100)
    assert expanded.count("\nitem ") == 100
    assert expanded.endswith("\n\nTask:\n" + prompt)
    assert "input data" in expanded
    original = json.dumps(bench.payload(prompt, 64), sort_keys=True).encode()
    default = json.dumps(bench.payload(bench.contextual_prompt(prompt, 0), 64), sort_keys=True).encode()
    assert original == default
    assert hashlib.sha256(original).hexdigest() != hashlib.sha256(
        json.dumps(bench.payload(expanded, 64), sort_keys=True).encode()).hexdigest()


def test_context_size_is_journaled_without_input_text(tmp_path, monkeypatch):
    options = args(tmp_path)
    options.context_lines = 100
    opener = Opener()
    monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *_: opener)
    bench.run(options)
    saved = options.output.read_text()
    records = [json.loads(line) for line in saved.splitlines()]
    assert records[0]["context_lines"] == 100
    assert all(row["context_lines"] == 100 for row in records if row["event"].startswith("request_"))
    assert "item 0000" not in saved
