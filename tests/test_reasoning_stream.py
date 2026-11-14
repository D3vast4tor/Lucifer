"""Provider reasoning is live-only; text consumers and saved history stay clean."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.inference import ModelManager, ModelSpec
from lucifer.schemas import ChatRequest
from lucifer.store import Store


def frame(event):
    return "data: " + json.dumps(event) + "\n\n"


async def provider(tmp_path, response):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    manager = ModelManager(settings, Store(settings.data_dir / "test.db"))
    manager.register(ModelSpec(id="provider", backend="external", endpoint="http://provider.test:8990"))
    async def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "provider"}]})
        if request.url.path.endswith("/status") or request.url.path == "/openapi.json":
            return httpx.Response(404)
        return response() if callable(response) else httpx.Response(200, text=response)
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    await manager.load("provider")
    return manager


def successful_stream():
    return (": provider heartbeat\n\n"
        + frame({"choices": [{"delta": {"reasoning_content": "First provider thought. ", "reasoning": "duplicate alias"}}]})
        + frame({"choices": [{"delta": {"reasoning": "Second provider thought."}}]})
        + frame({"choices": [{"delta": {"content": "# Answer\n\n**Markdown** stays intact."}}]})
        + frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + frame({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}})
        + "data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_typed_stream_preserves_provider_channels_and_text_wrappers(tmp_path):
    manager = await provider(tmp_path, successful_stream())
    try:
        events = [event async for event in manager.stream_events([{"role": "user", "content": "Hi"}], thinking="high")]
        assert events == [
            ("reasoning", {"content": "First provider thought. "}),
            ("reasoning", {"content": "Second provider thought."}),
            ("delta", {"content": "# Answer\n\n**Markdown** stays intact."}),
            ("finish", {"finish_reason": "stop"}),
            ("usage", {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}),
        ]
        text = await manager.complete([{"role": "user", "content": "Hi"}], thinking="high")
        assert text == "# Answer\n\n**Markdown** stays intact."
        assert "provider thought" not in text
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.parametrize("thinking", ["off", "high"])
@pytest.mark.parametrize("logging", [False, True])
@pytest.mark.asyncio
async def test_chat_reasoning_visibility_never_changes_saved_or_transient_history(tmp_path, thinking, logging):
    manager = await provider(tmp_path, successful_stream())
    service = ChatService(manager.store, manager)
    conversation = service.create(logging_enabled=logging)
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}], thinking=thinking)
    try:
        events = [event async for event in service.turn(request, conversation["id"])]
        reasoning = [data["content"] for event, data in events if event == "reasoning"]
        assert reasoning == ([] if thinking == "off" else ["First provider thought. ", "Second provider thought."])
        done = next(data for event, data in events if event == "done")
        assert done["reasoning_events"] == 2
        assert done["reasoning_status"] == ("hidden" if thinking == "off" else "streamed")
        assert done["finish_reason"] == "stop"
        history = service.history(conversation["id"])
        assert [m["content"] for m in history] == ["Hi", "# Answer\n\n**Markdown** stays intact."]
        assert "provider thought" not in json.dumps(history)
        stored = manager.store.conn.execute("SELECT data FROM documents").fetchall()
        assert "provider thought" not in json.dumps(stored)
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.parametrize("broken", [
    "data: {invalid\n\n",
    "data: []\n\n",
    frame({"choices": "invalid"}),
    frame({"choices": [{"delta": "invalid"}]}),
    frame({"choices": [{"delta": {"reasoning": ["invalid"]}}]}),
    frame({"choices": [{"delta": {"content": "partial"}}]}),
])
@pytest.mark.asyncio
async def test_malformed_or_abrupt_stream_never_commits_a_partial_turn(tmp_path, broken):
    manager = await provider(tmp_path, frame({"choices": [{"delta": {"reasoning_content": "live-only reasoning"}}]}) + broken)
    service = ChatService(manager.store, manager)
    conversation = service.create(logging_enabled=True)
    try:
        with pytest.raises(RuntimeError):
            _ = [event async for event in service.turn(ChatRequest(messages=[{"role": "user", "content": "Hi"}], thinking="high"), conversation["id"])]
        assert service.history(conversation["id"]) == []
        assert manager.store.list("messages") == []
        assert not manager._lock.locked()
    finally:
        await manager.close()
        manager.store.close()


class PausedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.release = asyncio.Event()

    async def __aiter__(self):
        yield frame({"choices": [{"delta": {"reasoning": "provider work in progress"}}]}).encode()
        await self.release.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_closing_a_reasoning_turn_closes_upstream_and_releases_engine_lock(tmp_path):
    body = PausedStream()
    manager = await provider(tmp_path, lambda: httpx.Response(200, stream=body))
    service = ChatService(manager.store, manager)
    conversation = service.create(logging_enabled=True)
    turn = service.turn(ChatRequest(messages=[{"role": "user", "content": "Hi"}], thinking="high"), conversation["id"])
    try:
        assert (await anext(turn))[0] == "reasoning"
        unload = asyncio.create_task(manager.unload())
        await asyncio.sleep(.005)
        assert not unload.done()
        await turn.aclose()
        await asyncio.wait_for(unload, timeout=1)
        assert body.closed
        assert not manager._lock.locked()
        assert manager.store.list("messages") == []
    finally:
        await turn.aclose()
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_canceling_a_blocked_provider_turn_closes_stream_without_committing(tmp_path):
    body = PausedStream()
    manager = await provider(tmp_path, lambda: httpx.Response(200, stream=body))
    service = ChatService(manager.store, manager)
    conversation = service.create(logging_enabled=True)
    turn = service.turn(ChatRequest(messages=[{"role": "user", "content": "Hi"}], thinking="high"), conversation["id"])
    try:
        assert (await anext(turn))[0] == "reasoning"
        pending = asyncio.create_task(anext(turn))
        await asyncio.sleep(.005)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert body.closed
        assert not manager._lock.locked()
        assert manager.store.list("messages") == []
    finally:
        await turn.aclose()
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_multiline_sse_and_final_finish_frame_without_done_are_supported(tmp_path):
    manager = await provider(tmp_path, 'data: {"choices":[\ndata: {"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
    try:
        assert await manager.complete([{"role": "user", "content": "Hi"}]) == "ok"
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_native_metadata_is_read_only_at_registry_and_load_boundaries(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = checkpoint / "config.json"
    metadata.write_text(json.dumps({"max_position_embeddings": 131072}))
    reads = []
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == metadata:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    manager = ModelManager(settings, Store(settings.data_dir / "test.db"))
    spec = ModelSpec(id="local", backend="demo", path=str(checkpoint))
    try:
        manager.register(spec)
        assert len(reads) == 1
        await manager.load("local")
        assert len(reads) == 2
        for _ in range(3):
            manager._save(spec, "loaded")
            manager.current()
            manager.list()
        assert len(reads) == 2
        manager.register(spec)
        assert len(reads) == 3  # Explicit re-registration refreshes model metadata.
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_unchanged_adapter_does_not_rewrite_model_state_each_turn(tmp_path, monkeypatch):
    manager = await provider(tmp_path, successful_stream())
    updates = []
    original = manager.store.update
    def update(kind, ident, **changes):
        if kind == "models" and "adapter_id" in changes:
            updates.append(changes)
        return original(kind, ident, **changes)
    monkeypatch.setattr(manager.store, "update", update)
    try:
        await manager.complete([{"role": "user", "content": "One"}])
        await manager.complete([{"role": "user", "content": "Two"}])
        assert updates == []
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_canceling_load_reaps_optional_capability_discovery_before_client_close(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    manager = ModelManager(settings, Store(settings.data_dir / "test.db"))
    manager.register(ModelSpec(id="provider", backend="external", endpoint="http://provider.test:8990"))
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "provider"}]})
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        load = asyncio.create_task(manager.load("provider"))
        await asyncio.wait_for(started.wait(), timeout=1)
        load.cancel()
        with pytest.raises(asyncio.CancelledError):
            await load
        assert cancelled.is_set()
        assert manager.current() is None
        assert manager.capabilities.get("provider") is None
    finally:
        await manager.close()
        manager.store.close()
