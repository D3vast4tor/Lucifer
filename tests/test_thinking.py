"""Thinking selection is per request; only answer content enters chat history."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
import yaml

from lucifer.api import create_app
from lucifer.config import Settings
from lucifer.inference import ModelManager, ModelSpec
from lucifer.schemas import ChatRequest
from lucifer.store import Store


def test_thinking_options_preserve_legacy_defaults_and_allow_template_labels():
    spec = ModelSpec(id="custom", backend="external", endpoint="http://localhost:8990",
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh", "other": "kept"},
        reasoning_effort_map={"max": "xhigh"})
    assert spec.effective_thinking() == "max"
    assert ModelManager._thinking_options(spec, None) == {"chat_template_kwargs": spec.chat_template_kwargs}
    enabled = ModelManager._thinking_options(spec, "max")
    assert enabled["reasoning_effort"] == "xhigh"
    assert enabled["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    disabled = ModelManager._thinking_options(spec, "off")
    assert disabled == {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False, "other": "kept"}}
    assert spec.chat_template_kwargs["enable_thinking"] is True
    assert spec.chat_template_kwargs["reasoning_effort"] == "xhigh"
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "Hi"}], thinking="unknown")


@pytest.mark.parametrize("thinking", ["off", "low", "medium", "high", "max"])
@pytest.mark.asyncio
async def test_all_thinking_modes_reach_backend_without_changing_profile(tmp_path, thinking):
    spec = {"id": "custom", "backend": "external", "endpoint": "http://localhost:8990",
        "chat_template_kwargs": {"enable_thinking": False, "other": "kept"}}
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [spec]}))
    settings = Settings(data_dir=tmp_path / "state", models_file=config)
    manager = ModelManager(settings, Store(settings.data_dir / "test.db"))
    requests = []
    async def upstream(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "custom"}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}\n\ndata: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await manager.load("custom")
        assert await manager.complete([{"role": "user", "content": "Hi"}], thinking=thinking) == "answer"
        payload = json.loads(requests[-1].content)
        assert payload["reasoning_effort"] == ("none" if thinking == "off" else thinking)
        assert payload["chat_template_kwargs"]["enable_thinking"] is (thinking != "off")
        assert payload["chat_template_kwargs"]["other"] == "kept"
        assert manager.specs["custom"].chat_template_kwargs == spec["chat_template_kwargs"]
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_profile_mode_and_answer_only_persistence_through_public_api(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [{"id": "custom", "backend": "external",
        "endpoint": "http://localhost:8990", "thinking_mode": "high"}]}))
    settings = Settings(data_dir=tmp_path / "state", models_file=config)
    app = create_app(settings, autoload="")
    payloads = []
    async def upstream(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "custom"}]})
        payloads.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"PRIVATE_CHAIN"}}]}\n\ndata: {"choices":[{"delta":{"content":"public answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    async with app.router.lifespan_context(app):
        await app.state.manager.client.aclose()
        app.state.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/models/load", json={"model_id": "custom"})
            conversation = (await client.post("/conversations", json={"logging_enabled": True})).json()
            request = {"conversation_id": conversation["id"], "messages": [{"role": "user", "content": "Hi"}]}
            reply = await client.post("/chat", json=request)
            assert reply.status_code == 200, reply.text
            assert reply.json()["thinking"] == "high"
            assert payloads[-1]["reasoning_effort"] == "high"
            stream = await client.post("/chat/stream", json={**request, "thinking": "off"})
            assert stream.status_code == 200
            assert '"thinking": "off"' in stream.text
            assert "PRIVATE_CHAIN" not in stream.text
            assert payloads[-1]["reasoning_effort"] == "none"
            messages = (await client.get(f"/conversations/{conversation['id']}")).json()["messages"]
            assert [m["content"] for m in messages] == ["Hi", "public answer", "Hi", "public answer"]
            assert "PRIVATE_CHAIN" not in json.dumps(app.state.store.list("messages"))


def test_demo_thinking_choice_is_reported_by_chat(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    with TestClient(create_app(settings, autoload="demo")) as client:
        response = client.post("/chat", json={"thinking": "max", "messages": [{"role": "user", "content": "What is 2+2?"}]})
        assert response.status_code == 200
        assert response.json()["thinking"] == "max"
