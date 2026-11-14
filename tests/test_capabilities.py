from dataclasses import FrozenInstanceError
import asyncio

import httpx
import pytest

from lucifer.capabilities import CAPABILITY_NAMES, CapabilityRegistry


def test_runtime_advertisement_keeps_unreported_features_unknown():
    registry = CapabilityRegistry()
    snapshot = registry.record("lucifer", "revision-a", models={"data": [{"id": "Lucifer", "meta": {"n_ctx": 8192}}]},
        status={"service": "strata", "model": "Lucifer", "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
                "cache_max_tokens": 8192, "context": {"native": 8192, "max_positions": 8192},
                "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                      "constrained_decoding": False, "stream_buffered": True},
                "vision": {"enabled": True, "available": True}})
    assert set(snapshot.capabilities) == set(CAPABILITY_NAMES)
    for name in ("chat_completions", "anthropic_messages", "responses", "structured_output", "vision"):
        assert snapshot.capabilities[name]["status"] == "advertised"
        assert snapshot.supports(name) and not snapshot.supports(name, require_verified=True)
    for name in ("tools", "reasoning", "streaming", "usage", "cancellation", "token_counting", "tokenization", "mcp"):
        assert snapshot.capabilities[name]["status"] == "unknown"
        assert not snapshot.supports(name)
    assert snapshot.context["effective_context_length"] == 8192
    assert snapshot.context["native_context_length"] is None
    structured = snapshot.capabilities["structured_output"]["constraints"]
    assert structured["method"] == "prompt_and_validate"
    assert structured["constrained_decoding"] is False and structured["stream_buffered"] is True
    assert structured["local_result_validation_required"] is True


def test_explicit_native_metadata_stays_separate_from_allocated_context():
    snapshot = CapabilityRegistry().record("a", "revision", status={"cache_max_tokens": 131072},
        native_context_length=1048576, effective_context_length=65536)
    assert snapshot.context == {"effective_context_length": 65536, "native_context_length": 1048576}


def test_declarations_and_verification_have_distinct_evidence():
    registry = CapabilityRegistry()
    snapshot = registry.record("a", "revision", status={"capabilities": {"tools": {"status": "verified"}}},
        declared={"tools": {"status": "verified", "evidence": ["pinned parser source"],
                             "constraints": {"strict_schema_enforced": False}},
                  "mcp": False, "reasoning": {"constraints": {"modes": ["off", "low", "medium", "high", "max"]}}},
        verified={"chat_completions": {"evidence": ["local acceptance response"]}})
    assert snapshot.capabilities["tools"]["status"] == "advertised"
    assert snapshot.capabilities["mcp"]["status"] == "unavailable"
    assert snapshot.capabilities["chat_completions"]["status"] == "verified"
    assert snapshot.supports("chat_completions", require_verified=True)
    assert "local acceptance response" in snapshot.capabilities["chat_completions"]["evidence"]
    assert snapshot.capabilities["tools"]["constraints"]["local_argument_validation_required"] is True


def test_snapshots_are_immutable_and_api_copies_cannot_change_the_cache():
    declaration = {"tools": {"constraints": {"formats": ["json"]}}}
    snapshot = CapabilityRegistry().record("a", "revision", declared=declaration)
    declaration["tools"]["constraints"]["formats"].append("unsafe")
    assert snapshot.capabilities["tools"]["constraints"]["formats"] == ("json",)
    with pytest.raises(TypeError):
        snapshot.capabilities["tools"]["status"] = "verified"
    with pytest.raises(FrozenInstanceError):
        snapshot.model_id = "other"
    public = snapshot.as_dict()
    public["capabilities"]["tools"]["constraints"]["formats"].append("changed")
    assert snapshot.as_dict()["capabilities"]["tools"]["constraints"]["formats"] == ["json"]


@pytest.mark.asyncio
async def test_startup_queries_are_cached_and_never_generate_or_probe_per_turn():
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a"}]})
        if request.url.path == "/v1/status":
            return httpx.Response(200, json={"dialects": ["/v1/chat/completions"]})
        raise AssertionError("Capability discovery must not run generation or tools")

    registry = CapabilityRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        first = await registry.discover("a", "revision", client=client, base_url="http://localhost:8089/v1")
        for _ in range(5):
            assert registry.get("a") is first
            assert await registry.discover("a", "revision", client=client, base_url="http://localhost:8089") is first
    assert sorted(requests) == [("GET", "/v1/models"), ("GET", "/v1/status")]


@pytest.mark.asyncio
async def test_injected_readiness_payloads_avoid_duplicate_http_calls():
    def handle(_request):
        raise AssertionError("Already fetched startup metadata must be reused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        snapshot = await CapabilityRegistry().discover("a", "revision", client=client, base_url="http://localhost:8089",
            models={"data": [{"id": "a"}]}, status={"dialects": ["/v1/chat/completions"]})
    assert snapshot.discovery["models"]["status"] == "injected"
    assert snapshot.discovery["status"]["status"] == "injected"


@pytest.mark.asyncio
async def test_concurrent_startup_callers_share_one_discovery():
    requests = []

    async def handle(request):
        requests.append(request.url.path)
        await asyncio.sleep(0)
        return httpx.Response(200, json={"data": [{"id": "a"}]} if request.url.path.endswith("models") else {})

    registry = CapabilityRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        first, second = await asyncio.gather(*[
            registry.discover("a", "revision", client=client, base_url="http://localhost:8089") for _ in range(2)])
    assert first is second
    assert sorted(requests) == ["/v1/models", "/v1/status"]


@pytest.mark.asyncio
async def test_failed_or_unsupported_metadata_does_not_invent_unsupported_features():
    def handle(request):
        return httpx.Response(404 if request.url.path.endswith("status") else 200,
                              json={"detail": "not implemented"} if request.url.path.endswith("status") else {"data": [{"id": "a"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        snapshot = await CapabilityRegistry().discover("a", "revision", client=client, base_url="http://localhost:8089")
    assert snapshot.discovery["status"]["http_status"] == 404
    assert all(entry["status"] == "unknown" for entry in snapshot.capabilities.values())


def test_invalidation_only_changes_the_requested_model_and_revision():
    registry = CapabilityRegistry()
    first = registry.record("a", "old", declared={"tools": False})
    other = registry.record("b", "stable", declared={"tools": True})
    assert registry.record("a", "old", declared={"tools": True}) is first
    registry.invalidate("a")
    assert registry.get("a") is None and registry.get("a", "old") is None
    assert registry.get("b") is other
    changed = registry.record("a", "new", declared={"tools": True})
    assert changed.profile_revision == "new" and changed.capabilities["tools"]["status"] == "advertised"


@pytest.mark.asyncio
async def test_optional_openapi_is_fetched_once_and_local_refs_do_not_fetch_schemas():
    requests = []
    document = {"paths": {"/v1/chat/completions": {"post": {"requestBody": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Request"}}}}}},
        "/v1/messages/count_tokens": {"post": {}}, "/tokenize": {"post": {}}},
        "components": {"schemas": {"Request": {"type": "object", "properties": {
            "tools": {"type": "array"}, "stream": {"type": "boolean"}, "reasoning_effort": {"type": "string"}}}}}}

    def handle(request):
        requests.append(request.url.path)
        assert request.method == "GET" and request.url.path == "/openapi.json"
        return httpx.Response(200, json=document)

    registry = CapabilityRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        snapshot = await registry.discover("a", "revision", client=client, base_url="http://localhost:8089/v1",
            models={}, status={}, include_openapi=True)
        await registry.discover("a", "revision", client=client, base_url="http://localhost:8089/v1", include_openapi=True)
    assert requests == ["/openapi.json"]
    for name in ("chat_completions", "tools", "streaming", "reasoning", "token_counting", "tokenization"):
        assert snapshot.capabilities[name]["status"] == "advertised"
    assert snapshot.capabilities["usage"]["status"] == "unknown"


def test_selected_model_metadata_does_not_borrow_other_models_vision_support():
    snapshot = CapabilityRegistry().record("a", "revision", served_model_name="selected",
        models={"data": [{"id": "other", "architecture": {"input_modalities": ["text", "image"]}},
                         {"id": "selected", "architecture": {"input_modalities": ["text"]}}]})
    assert snapshot.capabilities["vision"]["status"] == "unknown"


def test_current_unavailable_metadata_overrides_profile_and_openapi_advertisements():
    snapshot = CapabilityRegistry().record("a", "revision", declared={"vision": True, "tools": True},
        status={"vision": {"available": True, "enabled": False}, "capabilities": {"tools": False}},
        openapi={"paths": {"/v1/chat/completions": {"post": {"requestBody": {"content": {
            "application/json": {"schema": {"properties": {"tools": {"type": "array"}}}}}}}}}})
    assert snapshot.capabilities["vision"]["status"] == "unavailable"
    assert snapshot.capabilities["tools"]["status"] == "unavailable"
    assert snapshot.capabilities["vision"]["constraints"]["enabled"] is False


def test_malformed_optional_metadata_is_not_treated_as_capability_evidence():
    snapshot = CapabilityRegistry().record("a", "revision", status={"dialects": [{"unexpected": True}]},
        declared={"tools": {"evidence": "a pinned source", "endpoints": None}},
        openapi={"paths": {"/v1/chat/completions": {"post": {"requestBody": {"content": {"application/json": []}}}}}})
    assert snapshot.capabilities["tools"]["evidence"][-1] == "a pinned source"
    assert snapshot.capabilities["tools"]["endpoints"] == ()
    assert snapshot.capabilities["vision"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_cancelled_model_discovery_is_reaped_without_caching_stale_results():
    started = asyncio.Event()
    pending = asyncio.Event()

    async def handle(_request):
        started.set()
        await pending.wait()
        return httpx.Response(200, json={})

    registry = CapabilityRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        discovery = asyncio.create_task(registry.discover("a", "old", client=client, base_url="http://localhost:8089"))
        await started.wait()
        await registry.cancel_model("a")
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert registry.get("a") is None
        fresh = await registry.discover("a", "new", client=client, base_url="http://localhost:8089", models={}, status={})
        assert fresh.profile_revision == "new"
        await registry.close()


@pytest.mark.asyncio
async def test_registry_close_ends_owned_inflight_metadata_before_client_shutdown():
    started = asyncio.Event()

    async def handle(_request):
        started.set()
        await asyncio.Event().wait()

    registry = CapabilityRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        discovery = asyncio.create_task(registry.discover("a", "revision", client=client, base_url="http://localhost:8089"))
        await started.wait()
        await registry.close()
        with pytest.raises(asyncio.CancelledError):
            await discovery
    assert not registry._inflight and not registry._tasks
