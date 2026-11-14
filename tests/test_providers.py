"""Agent protocols: complete calls, cross-dialect history, cancellation and budgets."""
import asyncio
from contextlib import aclosing
import json

import httpx
import pytest

from lucifer.capabilities import CapabilityRegistry
from lucifer.config import Settings
from lucifer.inference import ModelManager, ModelSpec
from lucifer.providers import (ProviderAdapter, ProviderError, canonical_messages,
    conservative_count, validate_structured)
from lucifer.store import Store

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read bounded workspace text",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"], "additionalProperties": False}}}]
MESSAGES = [{"role": "system", "content": "You are Lucifer."}, {"role": "user", "content": "Read README.md"}]


def snapshot(**extra):
    declarations = {name: True for name in ("chat_completions", "anthropic_messages", "responses", "streaming", "tools", "reasoning", "vision", "structured_output", "token_counting")}
    declarations.update(extra)
    return CapabilityRegistry().record("test", "revision", declared=declarations, effective_context_length=8192)


def frame(value, event=None):
    return ("event: " + event + "\n" if event else "") + "data: " + json.dumps(value) + "\n\n"


def chat_stream(*, arguments='{"path":"README.md"}', finish="tool_calls", second_id=None):
    out = frame({"choices": [{"index": 0, "delta": {"reasoning_content": "Actual provider reasoning."}}]})
    out += frame({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_one", "type": "function", "function": {"name": "read_file", "arguments": arguments[:8]}}]}}]})
    out += frame({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments[8:]}}]}}]})
    if second_id:
        out += frame({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": second_id, "function": {"name": "read_file", "arguments": '{}'}}]}}]})
    out += frame({"choices": [{"delta": {}, "finish_reason": finish}]})
    out += frame({"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51}})
    return out + 'data: [DONE]\n\n'


def anthropic_stream(*, stop="tool_use"):
    out = frame({"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 0}}})
    out += frame({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})
    out += frame({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Actual provider reasoning."}})
    out += frame({"type": "content_block_stop", "index": 0})
    out += frame({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call_one", "name": "read_file", "input": {}}})
    out += frame({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path":'}})
    out += frame({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"README.md"}'}})
    out += frame({"type": "content_block_stop", "index": 1})
    out += frame({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 9}})
    return out + frame({"type": "message_stop"})


def responses_stream(*, completed=True):
    out = frame({"type": "response.created", "response": {"status": "in_progress"}})
    out += frame({"type": "response.reasoning_text.delta", "delta": "Actual provider reasoning."})
    item = {"id": "fc_one", "type": "function_call", "call_id": "call_one", "name": "read_file", "arguments": "", "status": "in_progress"}
    out += frame({"type": "response.output_item.added", "output_index": 0, "item": item})
    out += frame({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"path":"README.md"}'})
    item = {**item, "arguments": '{"path":"README.md"}', "status": "completed" if completed else "incomplete"}
    out += frame({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": item["arguments"]})
    out += frame({"type": "response.output_item.done", "output_index": 0, "item": item})
    return out + frame({"type": "response.completed", "response": {"status": "completed", "output": [item], "usage": {"input_tokens": 42, "output_tokens": 9, "total_tokens": 51}}})


async def events(dialect, body, *, payload=None, snap=None):
    adapter = ProviderAdapter(dialect, snap or snapshot())
    payload = payload or adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=1024,
                temperature=0, thinking="off", compatibility=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
        return [item async for item in adapter.stream(client, "http://provider/v1", {}, payload)]


@pytest.mark.parametrize("dialect,body", [("chat_completions", chat_stream()), ("anthropic_messages", anthropic_stream()), ("responses", responses_stream())])
@pytest.mark.asyncio
async def test_all_dialects_produce_one_validated_complete_call(dialect, body):
    result = await events(dialect, body)
    assert [data for event, data in result if event == "tool_call"] == [{"id": "call_one", "name": "read_file", "arguments": {"path": "README.md"}}]
    assert result[-1] == ("finish", {"finish_reason": "tool_calls"})
    assert ("reasoning", {"content": "Actual provider reasoning."}) in result
    assert any(event == "usage" for event, _ in result)


@pytest.mark.parametrize("dialect", ["chat_completions", "anthropic_messages", "responses"])
def test_canonical_tool_history_and_images_are_converted_for_each_dialect(dialect):
    messages = MESSAGES + [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_one", "name": "read_file", "arguments": {"path": "README.md"}}]},
        {"role": "tool", "tool_call_id": "call_one", "content": "README content"},
        {"role": "user", "content": [{"type": "text", "text": "What color?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}]}]
    payload = ProviderAdapter(dialect, snapshot()).payload(model="Lucifer", messages=messages, tools=TOOLS,
        max_tokens=2048, temperature=0, thinking="high", compatibility=True)
    assert payload["model"] == "Lucifer"
    text = json.dumps(payload)
    assert "call_one" in text and "README content" in text
    assert "reasoning_content" not in text and "Actual provider reasoning" not in text
    if dialect == "chat_completions":
        assert payload["messages"][2]["content"] is None
        assert payload["messages"][3]["tool_call_id"] == "call_one"
    elif dialect == "anthropic_messages":
        assert payload["system"] == "You are Lucifer."
        assert payload["messages"][1]["content"][0]["type"] == "tool_use"
        assert payload["messages"][2]["content"][0]["type"] == "tool_result"
        assert payload["messages"][2]["content"][-1]["type"] == "image"
        assert payload["output_config"]["effort"] == "high"
    else:
        assert payload["store"] is False and "previous_response_id" not in payload
        assert any(m["type"] == "function_call_output" for m in payload["input"])
        assert payload["reasoning"]["effort"] == "high"


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "tool_call_id": "missing", "content": "wrong"}],
    MESSAGES + [{"role": "assistant", "content": None, "tool_calls": [{"id": "one", "name": "read_file", "arguments": {}}]}],
    MESSAGES + [{"role": "assistant", "content": None, "tool_calls": [{"id": "one", "name": "read_file", "arguments": {}}]}, {"role": "user", "content": "skip result"}],
])
def test_invalid_tool_history_is_not_sent(messages):
    with pytest.raises(ProviderError):
        canonical_messages(messages)


@pytest.mark.parametrize("body", [chat_stream(arguments='{"path":'), chat_stream(arguments='[]'),
    chat_stream(arguments='{"path":"one","path":"two"}'), chat_stream(arguments='{"path":NaN}'),
    chat_stream(finish="length"), chat_stream(finish="stop"), chat_stream(second_id="call_one"),
    chat_stream().replace('"read_file"', '"withdrawn_tool"'), chat_stream().replace('"call_one"', '""'),
    chat_stream().replace('"finish_reason": "tool_calls"', '"finish_reason": null').replace('data: [DONE]\n\n', ''),
])
@pytest.mark.asyncio
async def test_bad_or_truncated_calls_never_reach_the_executor(body):
    emitted = []
    adapter = ProviderAdapter("chat_completions", snapshot())
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
        with pytest.raises(ProviderError):
            async for item in adapter.stream(client, "http://provider/v1", {}, payload):
                emitted.append(item)
    assert not any(event == "tool_call" for event, _ in emitted)


@pytest.mark.parametrize("dialect,body", [("anthropic_messages", anthropic_stream(stop="max_tokens")), ("responses", responses_stream(completed=False))])
@pytest.mark.asyncio
async def test_non_chat_dialects_reject_truncated_function_blocks(dialect, body):
    with pytest.raises(ProviderError):
        await events(dialect, body)


class Paused(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
    async def __aiter__(self):
        yield frame({"choices": [{"delta": {"reasoning_content": "live"}}]}).encode()
        await asyncio.Future()
    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_typed_stream_closes_the_owned_http_request():
    stream = Paused()
    provider = ProviderAdapter("chat_completions", snapshot())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))) as client:
        payload = provider.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
        async with aclosing(provider.stream(client, "http://provider/v1", {}, payload)) as output:
            assert await anext(output) == ("reasoning", {"content": "live"})
        assert stream.closed


@pytest.mark.asyncio
async def test_only_explicit_pre_stream_429_retries_with_retry_after(monkeypatch):
    calls, delays = [], []
    async def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "0.02"}) if len(calls) < 3 else httpx.Response(200, text=chat_stream())
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr("lucifer.providers.asyncio.sleep", sleep)
    adapter = ProviderAdapter("chat_completions", snapshot())
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = [item async for item in adapter.stream(client, "http://provider/v1", {}, payload)]
    assert len(calls) == 3 and delays == [.02, .02]
    assert len([event for event, _ in result if event == "tool_call"]) == 1


@pytest.mark.asyncio
async def test_midstream_error_is_not_retried_even_after_a_complete_call_fragment():
    calls = []
    body = chat_stream().replace('data: [DONE]\n\n', frame({"error": {"message": "engine died"}}))
    def handle(request):
        calls.append(request)
        return httpx.Response(200, text=body)
    adapter = ProviderAdapter("chat_completions", snapshot())
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ProviderError, match="Provider stream failed"):
            _ = [item async for item in adapter.stream(client, "http://provider/v1", {}, payload)]
    assert len(calls) == 1


def test_structured_validation_uses_full_local_schema_without_remote_references():
    fmt = {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["Rome"]}}, "required": ["answer"], "additionalProperties": False}}}
    assert validate_structured('{"answer":"Rome"}', fmt) == {"answer": "Rome"}
    for raw in ('{"answer":3}', '{"answer":"Paris"}', '{"answer":"Rome","extra":1}', '{}'):
        with pytest.raises(ProviderError):
            validate_structured(raw, fmt)
    remote = {"type": "json_schema", "json_schema": {"schema": {"$ref": "https://example.test/schema"}}}
    with pytest.raises(ProviderError, match="local JSON schema"):
        validate_structured('{}', remote)


@pytest.mark.asyncio
async def test_invalid_structured_result_is_buffered_and_never_emitted_as_answer():
    adapter = ProviderAdapter("chat_completions", snapshot())
    fmt = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=[], max_tokens=512, temperature=0, thinking="off", response_format=fmt)
    body = frame({"choices": [{"delta": {"content": '{"answer":"wrong"}'}, "finish_reason": "stop"}]})
    emitted = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
        with pytest.raises(ProviderError):
            async for item in adapter.stream(client, "http://provider/v1", {}, payload, response_format=fmt):
                emitted.append(item)
    assert not any(event == "delta" for event, _ in emitted)


def test_unsupported_dialects_features_and_combinations_fail_clearly():
    with pytest.raises(ValueError, match="Unknown provider dialect"):
        ProviderAdapter("made_up_alias", snapshot())
    denied = ProviderAdapter("chat_completions", snapshot(tools=False, vision=False))
    with pytest.raises(ProviderError, match="function tools"):
        denied.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
    fmt = {"type": "json_object"}
    limited = snapshot(structured_output={"constraints": {"chat_completions_response_format_with_tools": False}})
    with pytest.raises(ProviderError, match="combined"):
        ProviderAdapter("chat_completions", limited).payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off", response_format=fmt)
    with pytest.raises(ProviderError, match="Anthropic"):
        ProviderAdapter("anthropic_messages", snapshot()).payload(model="Lucifer", messages=MESSAGES, tools=[], max_tokens=512, temperature=0, thinking="off", response_format=fmt)


def test_conservative_budget_counts_schemas_unicode_and_image_expansion():
    plain = conservative_count(MESSAGES, [])
    with_tools = conservative_count(MESSAGES, TOOLS)
    image = conservative_count(MESSAGES + [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}]}], TOOLS)
    assert plain["estimated"] and plain["method"] == "conservative_estimate"
    assert with_tools["input_tokens"] > plain["input_tokens"]
    assert image["input_tokens"] > with_tools["input_tokens"] + 65536


def test_image_transport_bytes_are_not_counted_as_text_or_mutated():
    small_url = "data:image/png;base64,aGVsbG8="
    large_url = "data:image/png;base64," + "QUJD" * 262144
    small = MESSAGES + [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": small_url}},
    ]}]
    large = MESSAGES + [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": large_url}},
    ]}]
    small_count = conservative_count(small, TOOLS)
    large_count = conservative_count(large, TOOLS)
    assert large_count == small_count
    assert isinstance(large_count["input_tokens"], int) and large_count["estimated"]
    assert large[-1]["content"][1]["image_url"]["url"] == large_url
    assert small[-1]["content"][1]["image_url"]["url"] == small_url
    two_images = large[:-1] + [{"role": "user", "content": [
        *large[-1]["content"], {"type": "image_url", "image_url": {"url": small_url}},
    ]}]
    extra_image_cost = conservative_count(two_images, TOOLS)["input_tokens"] - large_count["input_tokens"]
    assert 65536 < extra_image_cost < 65536 + 256
    assert two_images[-1]["content"][1]["image_url"]["url"] == large_url


async def manager(tmp_path, handler):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    result = ModelManager(settings, Store(settings.data_dir / "test.db"))
    result.register(ModelSpec(id="model", backend="external", endpoint="http://provider:9000/v1", context_length=8192,
        capabilities={name: True for name in ("chat_completions", "anthropic_messages", "responses", "tools", "reasoning", "token_counting")}))
    async def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model", "max_model_len": 8192}]})
        if request.url.path in {"/v1/status", "/openapi.json"}:
            return httpx.Response(404)
        return handler(request)
    await result.client.aclose()
    result.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    await result.load("model")
    return result


@pytest.mark.asyncio
async def test_manager_pins_profile_and_reuses_prepared_adapters_without_discovery(tmp_path):
    requests = []
    def handle(request):
        requests.append(request.url.path)
        return httpx.Response(200, text=chat_stream())
    engine = await manager(tmp_path, handle)
    try:
        startup = engine.capabilities.get("model")
        prepared = engine._provider_adapters["model"]["chat_completions"]
        for _ in range(2):
            result = [item async for item in engine.agent_events(MESSAGES, TOOLS, model_id="model", profile_revision=startup.profile_revision, thinking="off", max_tokens=512)]
            assert not any(event == "reasoning" for event, _ in result)
        assert requests == ["/v1/chat/completions"] * 2
        assert engine._provider_adapters["model"]["chat_completions"] is prepared
        assert engine.capabilities.get("model") is startup
        with pytest.raises(ProviderError, match="revision changed"):
            _ = [event async for event in engine.agent_events(MESSAGES, TOOLS, profile_revision="old")]
        with pytest.raises(ProviderError, match="exceeds active context"):
            _ = [event async for event in engine.agent_events(MESSAGES, TOOLS, max_tokens=8192)]
        assert not engine._lock.locked()
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_manager_counts_only_advertised_native_route_and_keeps_estimates_honest(tmp_path):
    requests = []
    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"input_tokens": 123})
    engine = await manager(tmp_path, handle)
    try:
        exact = await engine.count_tokens(MESSAGES, TOOLS, dialect="anthropic_messages")
        assert exact == {"input_tokens": 123, "method": "provider", "estimated": False}
        estimate = await engine.count_tokens(MESSAGES, TOOLS, dialect="chat_completions")
        assert estimate["estimated"] and estimate["input_tokens"] > 123
        assert len(requests) == 1 and requests[0][0] == "/v1/messages/count_tokens"
        assert requests[0][1]["system"] == "You are Lucifer."
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_responses_structured_format_can_defer_validation_during_a_tool_round():
    adapter = ProviderAdapter("responses", snapshot())
    fmt = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off", response_format=fmt)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=responses_stream()))) as client:
        result = [item async for item in adapter.stream(client, "http://provider/v1", {}, payload, response_format=fmt)]
    assert result[-2:] == [("tool_call", {"id": "call_one", "name": "read_file", "arguments": {"path": "README.md"}}), ("finish", {"finish_reason": "tool_calls"})]
    assert payload["text"]["format"]["type"] == "json_schema"


def test_bad_structured_schema_fails_before_the_provider_request_and_counts_its_size():
    adapter = ProviderAdapter("chat_completions", snapshot())
    fmt = {"type": "json_schema", "json_schema": {"schema": {"$ref": "https://example.test/schema"}}}
    with pytest.raises(ProviderError, match="local JSON schema"):
        adapter.payload(model="Lucifer", messages=MESSAGES, tools=[], max_tokens=512, temperature=0, thinking="off", response_format=fmt)
    fmt = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "description": "x" * 500}}}
    assert conservative_count(MESSAGES, [], fmt)["input_tokens"] > conservative_count(MESSAGES, [])["input_tokens"] + 500


@pytest.mark.asyncio
async def test_provider_error_bodies_cannot_smuggle_reasoning_into_persisted_failure_text():
    adapter = ProviderAdapter("chat_completions", snapshot())
    payload = adapter.payload(model="Lucifer", messages=MESSAGES, tools=TOOLS, max_tokens=512, temperature=0, thinking="off")
    body = frame({"error": {"message": "raw provider reasoning or an echoed private prompt", "type": "server_error", "code": "engine_died"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
        with pytest.raises(ProviderError) as failure:
            _ = [item async for item in adapter.stream(client, "http://provider/v1", {}, payload)]
    assert "engine_died" in str(failure.value)
    assert "private prompt" not in str(failure.value) and "raw provider reasoning" not in str(failure.value)


@pytest.mark.asyncio
async def test_manager_agent_cancellation_releases_engine_lock_and_does_not_change_snapshot(tmp_path):
    paused = Paused()
    engine = await manager(tmp_path, lambda request: httpx.Response(200, stream=paused))
    startup = engine.capabilities.get("model")
    turn = engine.agent_events(MESSAGES, TOOLS, thinking="high", max_tokens=512,
                               model_id="model", profile_revision=startup.profile_revision)
    try:
        assert await anext(turn) == ("reasoning", {"content": "live"})
        unload = asyncio.create_task(engine.unload())
        await asyncio.sleep(.005)
        assert not unload.done()
        await turn.aclose()
        await asyncio.wait_for(unload, timeout=1)
        assert paused.closed and not engine._lock.locked()
        assert engine.capabilities.get("model") is startup
    finally:
        await turn.aclose()
        await engine.close()
        engine.store.close()


def test_structured_schema_honors_declared_dialect_and_checks_formats_offline():
    fmt = {"type": "json_schema", "json_schema": {"schema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "dependencies": {"x": ["y"]}}}}
    with pytest.raises(ProviderError, match="failed local"):
        validate_structured('{"x":1}', fmt)
    assert validate_structured('{"x":1,"y":2}', fmt)["y"] == 2
    fmt = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}}}}
    with pytest.raises(ProviderError, match="failed local"):
        validate_structured('{"email":"not-an-email"}', fmt)
    fmt["json_schema"]["schema"]["$schema"] = "https://unknown.test/schema"
    with pytest.raises(ProviderError, match="Unsupported JSON Schema dialect"):
        validate_structured('{}', fmt)
