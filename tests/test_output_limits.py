"""Large output budgets use startup metadata, without losing reasoning channels."""
import json

import httpx
import pytest
import yaml

from lucifer.config import Settings
from lucifer.capabilities import CapabilityRegistry
from lucifer.inference import ModelManager
from lucifer.providers import ProviderAdapter, ProviderError
from lucifer.schemas import AgentRequest, ChatRequest
from lucifer.store import Store


async def loaded(tmp_path, *, advertised=None, configured=262144, context=1048576):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [{"id": "large", "backend": "external",
        "endpoint": "http://provider/v1", "max_output_tokens": configured,
        "capabilities": {"chat_completions": True, "reasoning": True}}]}))
    settings = Settings(data_dir=tmp_path / "state", models_file=path)
    manager = ModelManager(settings, Store(settings.data_dir / "test.db"))
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path == "/v1/models":
            card = {"id": "large", "max_model_len": context}
            if advertised is not None:
                card["max_output_tokens"] = advertised
            return httpx.Response(200, json={"data": [card]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(404)
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    await manager.load("large")
    return manager, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised,configured,context,expected,source", [
    (None, 262144, 1048576, 262144, "profile output budget"),
    (65536, 262144, 1048576, 65536, "runtime output metadata"),
    (262144, 131072, 1048576, 131072, "profile output budget"),
    (None, 262144, 8192, 7936, "active context allowance"),
])
async def test_output_allowance_is_computed_once_from_loaded_profile(tmp_path, advertised, configured, context, expected, source):
    manager, requests = await loaded(tmp_path, advertised=advertised, configured=configured, context=context)
    try:
        current = manager.current()
        assert current["max_output_tokens"] == expected
        assert current["output_limit_source"] == source
        assert current["native_max_output_tokens"] == advertised
        startup_requests = len(requests)
        for _ in range(2):
            assert manager.current()["max_output_tokens"] == expected
        assert len(requests) == startup_requests
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_large_agent_output_reaches_provider_and_smaller_profile_rejects_it(tmp_path):
    manager, requests = await loaded(tmp_path)
    try:
        messages = [{"role": "user", "content": "Lucifer, reply briefly."}]
        result = [event async for event in manager.agent_events(messages, max_tokens=262144, input_tokens=100)]
        assert result[-1] == ("finish", {"finish_reason": "stop"})
        request = next(r for r in requests if r.url.path == "/v1/chat/completions")
        assert json.loads(request.content)["max_tokens"] == 262144
        assert request.extensions["timeout"]["read"] is None
        with pytest.raises(ProviderError, match="configured allowance"):
            _ = [event async for event in manager.agent_events(messages, max_tokens=262145, input_tokens=100)]
    finally:
        await manager.close()
        manager.store.close()


def test_request_schemas_allow_model_sized_output_and_unbounded_agent():
    assert ChatRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=262144).max_tokens == 262144
    request = AgentRequest(goal="Inspect the project", workspace="/tmp", max_tokens=262144)
    assert request.timeout_seconds is None
    assert request.max_rounds is None and request.max_tool_calls is None and request.max_total_tokens is None


def test_actual_reasoning_history_uses_native_separate_field_only():
    messages = [{"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Actual provider thought"},
        {"role": "user", "content": "Correction"}]
    for dialect in ("chat_completions", "anthropic_messages"):
        contract = CapabilityRegistry().record("Lucifer", "native-revision", declared={
            "reasoning": {"constraints": {"history_field": "reasoning_content"}}})
        provider = ProviderAdapter(dialect, contract)
        payload = provider.payload(model="Lucifer", messages=messages, tools=[], max_tokens=2048,
            temperature=0, thinking="max", compatibility=True)
        if dialect == "chat_completions":
            assert payload["messages"][1]["reasoning_content"] == "Actual provider thought"
            assert payload["messages"][1]["content"] == "Answer"
        else:
            assert payload["messages"][1]["content"][0] == {"type": "thinking", "thinking": "Actual provider thought"}
        generic = ProviderAdapter(dialect, None).payload(model="Lucifer", messages=messages, tools=[], max_tokens=2048,
            temperature=0, thinking="max", compatibility=False)
        assert "Actual provider thought" not in json.dumps(generic)
    # Payload preparation never destroys the original RAM history.
    assert messages[1]["reasoning_content"] == "Actual provider thought"
