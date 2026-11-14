"""Terminal integration checks against a real HTTP client with a local transport."""
import asyncio
import hashlib
import json
from uuid import UUID

import httpx
import pytest
from textual.widgets import Button, Collapsible, Input, Markdown, ProgressBar, Select, Static, TextArea

from lucifer.client import APIClient, APIError
from lucifer.tui import AgentHistoryScreen, ExampleReviewScreen, LuciferApp, SleepProgressBar


class Backend:
    def __init__(self):
        self.requests = []
        self.conv = {"id": "conv_test", "logging_enabled": True, "training_enabled": False}
        self.models = [{"id": "demo", "backend": "demo", "status": "loaded"}]
        self.adapters = []
        self.trains = []
        self.sleeps = []
        self.status = {"active_adapter": None}
        self.gpu = {"available": False, "gpus": [], "sampled_at": "2026-11-12T12:00:00Z", "stale": False, "errors": []}
        self.revision = 1

    def snapshot(self):
        return {"models": self.models, "adapters": self.adapters, "status": self.status,
                "jobs": {"train": self.trains, "sleep": self.sleeps}, "gpu": self.gpu,
                "capabilities": {}, "revision": self.revision}

    def handle(self, request):
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, payload))
        path = request.url.path
        if path == "/system/bootstrap":
            return httpx.Response(200, json=self.snapshot())
        if path == "/system/telemetry":
            return httpx.Response(200, json={key: value for key, value in self.snapshot().items()
                                             if key in {"status", "jobs", "gpu", "revision"}})
        if path == "/models":
            return httpx.Response(200, json=self.models)
        if path in ("/adapters", "/train", "/sleep/runs"):
            return httpx.Response(200, json=[])
        if path == "/system/gpu":
            return httpx.Response(200, json=self.gpu)
        if path == "/system/status":
            return httpx.Response(200, json={"active_adapter": None})
        if path == "/conversations":
            if request.method == "POST":
                self.conv.update(payload)
            return httpx.Response(200, json=self.conv)
        if path == "/conversations/conv_test":
            self.conv.update(payload)
            return httpx.Response(200, json=self.conv)
        if path == "/conversations/conv_test/export":
            return httpx.Response(200, json={**self.conv, "messages": [{"role": "assistant", "content": "reply"}]})
        if path == "/chat/stream":
            return httpx.Response(200, text='event: meta\ndata: {"conversation_id":"conv_test"}\n\nevent: delta\ndata: {"content":"Hello [bold]"}\n\nevent: delta\ndata: {"content":" friend"}\n\nevent: done\ndata: {"message_id":"msg_test","conversation_id":"conv_test"}\n\n', headers={"content-type": "text/event-stream"})
        if path == "/feedback":
            return httpx.Response(200, json={"id": "fb_test", **payload})
        if path == "/models/load":
            return httpx.Response(404, json={"detail": "Unknown model"})
        if path in ("/sleep/start", "/train/start"):
            return httpx.Response(200, json={"id": "run_test", "status": "queued", **payload})
        return httpx.Response(404, json={"detail": path})


async def submit(app, pilot, text):
    app.submit(text)
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_streamed_chat_feedback_and_privacy_controls():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, "Hi [red]literal[/red]")
        assert app.conversation["id"] == "conv_test"
        assert app.last_message_id == "msg_test"
        assert app.query_one(".assistant .answer", Markdown).source == "Hello [bold] friend"
        assert "LUCIFER" in app.query_one(".assistant").border_title
        assert app.query_one(".user").render().plain == "Hi [red]literal[/red]"
        await submit(app, pilot, "/feedback correct A corrected answer")
        feedback = next(p for method, path, p in backend.requests if path == "/feedback")
        assert feedback == {"conversation_id": "conv_test", "message_id": "msg_test", "label": "correct", "correction": "A corrected answer"}
        await submit(app, pilot, "/privacy logging on")
        assert app.conversation["logging_enabled"] is True
        await submit(app, pilot, "/privacy training on")
        assert app.conversation["training_enabled"] is True
        await submit(app, pilot, "/chat new")
        assert app.conversation is None
        assert app.last_message_id is None


@pytest.mark.asyncio
async def test_new_conversations_save_by_default_while_explicit_private_choice_is_honored():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        assert 'LOG: ON' in app.query_one('#session').render().plain
        assert 'TRAIN: OFF' in app.query_one('#session').render().plain
        await submit(app, pilot, 'Save this new conversation automatically.')
        starts = [payload for method, path, payload in backend.requests if method == 'POST' and path == '/conversations']
        assert starts == [{'logging_enabled': True, 'training_enabled': False}]
        assert app.conversation['logging_enabled'] is True and app.conversation['training_enabled'] is False
        await submit(app, pilot, '/privacy logging off')
        await submit(app, pilot, 'Keep this turn private by my explicit choice.')
        assert app.conversation['logging_enabled'] is False and app.conversation['training_enabled'] is False
        assert 'LOG: OFF' in app.query_one('#session').render().plain
        assert len([1 for method, path, _ in backend.requests if method == 'POST' and path == '/conversations']) == 1
        await submit(app, pilot, '/session new')
        assert 'LOG: ON' in app.query_one('#session').render().plain
        await submit(app, pilot, 'This separate conversation saves automatically again.')
        starts = [payload for method, path, payload in backend.requests if method == 'POST' and path == '/conversations']
        assert len(starts) == 2 and all(payload == {'logging_enabled': True, 'training_enabled': False} for payload in starts)


def test_conversation_schema_defaults_to_saved_without_training_and_allows_explicit_privacy():
    from lucifer.schemas import ConversationCreate
    assert ConversationCreate().model_dump() == {'title': 'New conversation', 'logging_enabled': True, 'training_enabled': False}
    assert ConversationCreate(logging_enabled=False).logging_enabled is False
    with pytest.raises(ValueError, match='Enable logging'):
        ConversationCreate(logging_enabled=False, training_enabled=True)


@pytest.mark.asyncio
async def test_command_errors_recover_and_dry_run_remains_explicit(tmp_path):
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(100, 35)) as pilot:
        await submit(app, pilot, "/model load nonexistent")
        assert "Unknown model" in list(app.query(".system"))[-1].render().plain
        assert app.query_one("#send", Button).disabled is False
        assert app.busy is False
        await submit(app, pilot, "/sleep start --dry-run")
        assert ("POST", "/sleep/start", {"dynamic": True, "dry_run": True, "context_examples": True}) in backend.requests
        await submit(app, pilot, "hello")
        target = tmp_path / "chat export.json"
        await submit(app, pilot, f'/chat export "{target}"')
        assert json.loads(target.read_text())["messages"][0]["content"] == "reply"
        assert target.stat().st_mode & 0o777 == 0o600
        await submit(app, pilot, f'/chat export "{target}"')
        assert "Error:" in list(app.query(".system"))[-1].render().plain


@pytest.mark.asyncio
async def test_sse_multiline_final_frame_and_http_error():
    def handle(request):
        if request.url.path == "/chat/stream":
            return httpx.Response(200, text=': keepalive\nevent: delta\ndata: {\ndata: "content":"hello"}\n\nevent: done\ndata: {}')
        return httpx.Response(409, json={"detail": "Model busy"})
    client = APIClient(transport=httpx.MockTransport(handle))
    events = [event async for event in client.stream_chat({})]
    assert events == [("delta", {"content": "hello"}), ("done", {})]
    with pytest.raises(APIError, match="Model busy"):
        await client.get("/models")
    await client.close()


@pytest.mark.asyncio
async def test_invalid_stream_payload_reports_error():
    client = APIClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text='event: delta\ndata: []\n\n')))
    with pytest.raises(APIError, match="invalid stream payload"):
        async for _ in client.stream_chat({}):
            pass
    await client.close()


@pytest.mark.asyncio
async def test_model_load_uses_backend_startup_deadline_instead_of_short_request_timeout():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "lucifer", "status": "loaded"})

    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        assert (await client.load_model("lucifer"))["status"] == "loaded"
        assert requests[0].url.path == "/models/load"
        assert json.loads(requests[0].content) == {"model_id": "lucifer"}
        assert requests[0].extensions["timeout"]["read"] is None
        assert requests[0].extensions["timeout"]["connect"] == 5
    finally:
        await client.close()


@pytest.mark.parametrize("decision", ["approve", "deny"])
@pytest.mark.asyncio
async def test_malformed_approval_command_keeps_busy_frontend_and_draft_available(decision):
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(100, 35)) as pilot:
        app.busy = True
        draft = f"/agent {decision} 'unfinished"
        app.query_one("#prompt", Input).value = draft
        app.submit(draft)
        await pilot.pause()
        assert app.busy is True
        assert app.query_one("#prompt", Input).value == draft
        assert app.crash_report is None
        assert not any(path.endswith("/approve") for _, path, _ in backend.requests)
        app.busy = False


@pytest.mark.asyncio
async def test_keyboard_submission_and_small_terminal_layout():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(72, 24)) as pilot:
        assert app.has_class("compact")
        assert app.query_one("#thinking-controls").display is True
        assert app.query_one("#thinking-picker", Select).display is True
        app.query_one("#prompt", Input).value = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.last_message_id == "msg_test"
        app.query_one("#prompt", Input).value = "/feedback correct It's a literal correction."
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        feedback = next(p for method, path, p in backend.requests if path == "/feedback")
        assert feedback["correction"] == "It's a literal correction."


@pytest.mark.asyncio
async def test_thinking_modes_are_sent_per_request_with_a_complex_answer_budget():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        for mode in ("off", "low", "medium", "high", "max"):
            await submit(app, pilot, f"/thinking {mode}")
            assert app.query_one("#thinking-picker", Select).value == mode
            await submit(app, pilot, "Solve a problem.")
            payload = next(payload for _, path, payload in reversed(backend.requests) if path == "/chat/stream")
            assert payload["thinking"] == mode
            assert payload["max_tokens"] == (512 if mode == "off" else 4096)
            assert app.query_one(".assistant .answer", Markdown).source == "Hello [bold] friend"


@pytest.mark.asyncio
async def test_thinking_picker_and_output_override_survive_new_chats():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(72, 24)) as pilot:
        app.query_one("#thinking-picker", Select).value = "high"
        await pilot.pause()
        assert app.thinking_mode == "high"
        assert "4,096" in app.query_one("#generation-settings").render().plain
        await submit(app, pilot, "/tokens 6000")
        await submit(app, pilot, "First question.")
        await submit(app, pilot, "/chat new")
        assert app.thinking_mode == "high" and app.output_tokens_override == 6000
        assert app.query_one("#thinking-picker", Select).value == "high"
        await submit(app, pilot, "Second question.")
        payload = next(payload for _, path, payload in reversed(backend.requests) if path == "/chat/stream")
        assert payload["thinking"] == "high" and payload["max_tokens"] == 6000
        await submit(app, pilot, "/thinking off")
        assert app.output_budget() == 6000
        await submit(app, pilot, "/tokens auto")
        assert app.output_tokens_override is None and app.output_budget() == 512
        await submit(app, pilot, "/thinking impossible")
        assert "Usage: /thinking" in list(app.query(".system"))[-1].render().plain
        assert app.thinking_mode == "off"
        await submit(app, pilot, "/tokens 1000000")
        assert app.output_tokens_override == 1000000  # Unknown metadata must not impose the former 32K cap.
        await submit(app, pilot, "/tokens -1")
        assert "Usage: /tokens" in list(app.query(".system"))[-1].render().plain
        assert app.output_tokens_override == 1000000


@pytest.mark.asyncio
async def test_effective_context_caps_budget_and_native_metadata_stays_distinct():
    backend = Backend()
    current = {"id": "lucifer", "status": "loaded", "context_length": "auto",
               "effective_context_length": 3072, "native_context_length": 1048576}
    backend.status = {"model": current}

    def handle(request):
        if request.url.path == "/system/status":
            return httpx.Response(200, json={"model": current})
        if request.url.path == "/models/current":
            return httpx.Response(200, json=current)
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, "/thinking max")
        await submit(app, pilot, "Use the available context.")
        payload = next(payload for _, path, payload in reversed(backend.requests) if path == "/chat/stream")
        assert payload["max_tokens"] == 2816
        settings = app.query_one("#generation-settings").render().plain
        assert "3,072 active" in settings and "1,048,576 native" in settings
        await submit(app, pilot, "/model current")
        displayed = list(app.query(".system"))[-1].render().plain
        assert "Active context: 3,072" in displayed and "Native context: 1,048,576" in displayed
        current["effective_context_length"] = 65536
        await app.refresh_panels()
        await submit(app, pilot, "/tokens 32768")
        await submit(app, pilot, "Use a larger loaded profile.")
        payload = next(payload for _, path, payload in reversed(backend.requests) if path == "/chat/stream")
        assert payload["max_tokens"] == 32768
        assert "65,536 active" in app.query_one("#generation-settings").render().plain


@pytest.mark.asyncio
async def test_reasoning_has_a_separate_collapsed_panel_and_never_becomes_the_answer():
    backend = Backend()

    def handle(request):
        if request.url.path == "/chat/stream":
            return httpx.Response(200, text='event: reasoning\ndata: {"content":"private internal text"}\n\nevent: delta\ndata: {"content":"Visible answer"}\n\nevent: done\ndata: {"message_id":"msg_test","thinking":"high"}\n\n')
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, "/thinking high")
        await submit(app, pilot, "Reply.")
        assert app.query_one(".assistant .answer", Markdown).source == "Visible answer"
        panel = app.query_one(".thinking", Collapsible)
        assert panel.collapsed is True
        assert panel.query_one(Markdown).source == "private internal text"
        assert "private internal text" not in app.query_one(".answer", Markdown).source
        exported = await app.client.get("/conversations/conv_test/export")
        assert "private internal text" not in json.dumps(exported)


@pytest.mark.asyncio
async def test_gpu_inventory_lists_every_attached_device_and_unknown_memory():
    backend = Backend()
    cards = [
        {"index": 0, "id": "gpu-a", "vendor": "nvidia", "name": "First card", "total_mb": 12000, "free_mb": 3000},
        {"index": 1, "id": "gpu-b", "vendor": "amd", "name": "Second card", "total_mb": 16000, "free_mb": 9000},
        {"index": 2, "id": "gpu-c", "vendor": "intel", "name": "Third card", "total_mb": None, "free_mb": None},
    ]
    backend.gpu["gpus"] = cards
    def handle(request):
        if request.url.path == "/system/gpu":
            return httpx.Response(200, json={"available": True, "gpus": cards})
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(140, 45)) as pilot:
        inventory = app.query_one("#gpus").render().plain
        assert all(card["name"] in inventory for card in cards)
        assert "VRAM --% ··········" in inventory
        await submit(app, pilot, "/gpu list")
        displayed = list(app.query(".system"))[-1].render().plain
        assert all(card["id"] in displayed for card in cards)
        assert '"vendor": "amd"' in displayed
        assert app.query_one("#sidebar").allow_vertical_scroll is True


@pytest.mark.asyncio
async def test_model_profiles_keep_lucifer_branding_and_loader_ids_private():
    backend = Backend()
    source = "vendor/future-checkpoint-763B"
    models = [
        {"id": "lucifer", "display_name": source, "backend": "strata", "status": "loaded",
         "spec": {"path": f"/models/{source}", "hf_repo": source}},
        {"id": "legacy-checkpoint", "display_name": "Another future checkpoint", "status": "unloaded"},
    ]
    backend.models = models
    backend.status = {"model": models[0]}

    def handle(request):
        if request.url.path == "/models":
            return httpx.Response(200, json=models)
        if request.url.path == "/models/current":
            return httpx.Response(200, json=models[0])
        if request.url.path == "/models/load":
            payload = json.loads(request.content)
            backend.requests.append((request.method, request.url.path, payload))
            return httpx.Response(200, json=models[1])
        if request.url.path == "/models/unload":
            return httpx.Response(200, json={"status": "unloaded"})
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        sidebar = app.query_one("#models").render().plain
        assert "Lucifer · profile 1" in sidebar
        assert "Lucifer · profile 2" in sidebar
        assert source not in sidebar and "legacy-checkpoint" not in sidebar
        assert app._model_options == [("Lucifer · profile 1", "lucifer"), ("Lucifer · profile 2", "legacy-checkpoint")]
        assert app.query_one("#model-picker", Select).prompt == "Load Lucifer…"
        for command in ("/model list", "/model status", "/model current", "/model load 2", "/model unload"):
            await submit(app, pilot, command)
            displayed = list(app.query(".system"))[-1].render().plain
            assert "Lucifer" in displayed
            assert source not in displayed and "/models/" not in displayed and "legacy-checkpoint" not in displayed
        assert ("POST", "/models/load", {"model_id": "legacy-checkpoint"}) in backend.requests
        assert models[0]["spec"]["hf_repo"] == source


@pytest.mark.asyncio
async def test_job_summaries_hide_checkpoint_metadata_and_keep_useful_state():
    backend = Backend()
    source = "vendor/private-checkpoint"
    job = {"id": "train_test", "status": "dry_run", "dry_run": True, "total_examples": 12,
           "request": {"base_model": source}, "dataset_path": f"/models/{source}/dataset.jsonl",
           "logs": [{"message": f"Preparing {source}"}]}
    backend.trains = [job]
    backend.adapters = [{"id": "adapter_test", "base_model_id": source,
                         "training_model": source, "path": f"/models/{source}"}]

    def handle(request):
        if request.url.path == "/train/start":
            return httpx.Response(200, json=job)
        if request.url.path == "/train":
            return httpx.Response(200, json=[job])
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"id": "adapter_test", "base_model_id": source,
                                             "training_model": source, "path": f"/models/{source}"}])
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        for command in ("/train start --dry-run", "/train status"):
            await submit(app, pilot, command)
            displayed = list(app.query(".system"))[-1].render().plain
            assert "Lucifer // TRAIN // dry_run" in displayed
            assert "train_test" in displayed and "Examples: 12" in displayed
            assert "dataset preparation only" in displayed
            assert source not in displayed and "/models/" not in displayed
        await submit(app, pilot, "/adapters")
        displayed = list(app.query(".system"))[-1].render().plain
        assert "Lucifer" in displayed and source not in displayed and "/models/" not in displayed
        assert job["request"]["base_model"] == source


@pytest.mark.asyncio
async def test_model_descriptor_registration_and_registry_reload(tmp_path):
    backend = Backend()
    original = backend.handle
    supplied = {'id': 'custom-profile', 'backend': 'external', 'endpoint': 'http://127.0.0.1:9999/v1', 'served_model_name': 'Lucifer'}
    def handle(request):
        if request.url.path == '/models/register':
            payload = json.loads(request.content)
            backend.requests.append((request.method, request.url.path, payload))
            return httpx.Response(201, json={**payload, 'status': 'ready'})
        if request.url.path == '/models/reload':
            backend.requests.append((request.method, request.url.path, {}))
            return httpx.Response(200, json={'status': 'reloaded'})
        return original(request)
    descriptor = tmp_path / 'profile with spaces.yaml'
    import yaml
    descriptor.write_text(yaml.safe_dump(supplied))
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(100, 35)) as pilot:
        await submit(app, pilot, f'/model register "{descriptor}"')
        assert ('POST', '/models/register', supplied) in backend.requests
        assert not any(path == '/models/load' for _, path, _ in backend.requests)
        await submit(app, pilot, '/model reload')
        assert ('POST', '/models/reload', {}) in backend.requests
        invalid = tmp_path / 'invalid.json'
        invalid.write_text('[]')
        await submit(app, pilot, f'/model register "{invalid}"')
        assert 'one profile object' in list(app.query('.system'))[-1].render().plain
        invalid.write_text('id: [unterminated')
        await submit(app, pilot, f'/model register "{invalid}"')
        assert 'Invalid model descriptor' in list(app.query('.system'))[-1].render().plain
        assert app.busy is False


@pytest.mark.asyncio
async def test_answer_markdown_renders_code_tables_links_and_batches_tiny_fragments(monkeypatch):
    backend = Backend()
    answer = ('## Result\n\nUse **bold** and [the docs](https://example.org).\n\n'
              '```python\nprint("ok")\n```\n\n| Item | Value |\n| --- | --- |\n| A | 3 |\n')
    calls = []
    original = Markdown.append

    def counted(widget, fragment):
        if widget.has_class("answer"):
            calls.append(fragment)
        return original(widget, fragment)

    monkeypatch.setattr(Markdown, "append", counted)

    def handle(request):
        if request.url.path == "/chat/stream":
            frames = [f'event: delta\ndata: {json.dumps({"content": character})}\n\n' for character in answer]
            frames.append('event: done\ndata: {"message_id":"msg_markdown"}\n\n')
            return httpx.Response(200, text="".join(frames))
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(140, 45)) as pilot:
        await submit(app, pilot, "[red]literal user text[/red]")
        markdown = app.query_one(".answer", Markdown)
        assert markdown.source == answer
        assert len(markdown.query("MarkdownH2")) == 1
        assert len(markdown.query("MarkdownTable")) == 1
        assert markdown.query_one("MarkdownFence").code.strip() == 'print("ok")'
        assert app.query_one(".user").render().plain == "[red]literal user text[/red]"
        assert 1 <= len(calls) < 10
        assert "".join(calls) == answer
        assert app.last_message_id == "msg_markdown"


class HeldStream(httpx.AsyncByteStream):
    def __init__(self, first: str, rest: str):
        self.first, self.rest = first, rest
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    async def __aiter__(self):
        yield self.first.encode()
        self.started.set()
        await self.resume.wait()
        yield self.rest.encode()


@pytest.mark.asyncio
async def test_thinking_panel_toggles_while_provider_tokens_continue_and_stays_ephemeral():
    backend = Backend()
    stream = HeldStream(
        'event: reasoning\ndata: {"content":"Provider trace A. "}\n\n',
        'event: reasoning\ndata: {"content":"Provider trace B."}\n\n'
        'event: delta\ndata: {"content":"**Final answer.**"}\n\n'
        'event: done\ndata: {"message_id":"msg_thinking"}\n\n',
    )

    def handle(request):
        if request.url.path == "/chat/stream":
            return httpx.Response(200, stream=stream)
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(140, 45)) as pilot:
        app.thinking_mode = "high"
        app.submit("Consider the problem.")
        await asyncio.wait_for(stream.started.wait(), 2)
        await pilot.pause(0.1)
        panel = app.query_one(".thinking", Collapsible)
        assert panel.collapsed and app.busy
        assert panel.query_one(Markdown).source == "Provider trace A. "
        await pilot.click(".thinking > CollapsibleTitle")
        assert not panel.collapsed
        stream.resume.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not panel.collapsed
        assert panel.query_one(Markdown).source == "Provider trace A. Provider trace B."
        assert app.query_one(".answer", Markdown).source == "**Final answer.**"
        await submit(app, pilot, "/feedback good")
        feedback = next(payload for _, path, payload in backend.requests if path == "/feedback")
        assert feedback == {"conversation_id": "conv_test", "message_id": "msg_thinking", "label": "good"}
        exported = await app.client.get("/conversations/conv_test/export")
        assert "Provider trace" not in json.dumps(exported)
        assert all("Provider trace" not in json.dumps(payload) for _, _, payload in backend.requests)


@pytest.mark.asyncio
async def test_escape_flushes_partial_markdown_and_never_enables_feedback():
    backend = Backend()
    stream = HeldStream(
        'event: reasoning\ndata: {"content":"Transient provider trace"}\n\n'
        'event: delta\ndata: {"content":"**Partial answer"}\n\n',
        'event: done\ndata: {"message_id":"must_not_arrive"}\n\n',
    )

    def handle(request):
        if request.url.path == "/chat/stream":
            return httpx.Response(200, stream=stream)
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(100, 35)) as pilot:
        app.submit("Start a reply.")
        await asyncio.wait_for(stream.started.wait(), 2)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.query_one(".answer", Markdown).source == "**Partial answer"
        assert app.query_one(".assistant").has_class("partial")
        assert app.last_message_id is None and not app.busy
        assert app.query_one("#send", Button).disabled is False
        await submit(app, pilot, "/feedback good")
        assert not any(path == "/feedback" for _, path, _ in backend.requests)


@pytest.mark.asyncio
async def test_bootstrap_once_telemetry_only_then_revision_invalidates_without_rebuilding_unchanged_panels(monkeypatch):
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 45)):
        updates = []
        original = app.update_panel

        def counted(selector, value):
            if value != app._panel_values.get(selector):
                updates.append(selector)
            original(selector, value)

        monkeypatch.setattr(app, "update_panel", counted)
        await app.refresh_panels()
        await app.refresh_panels()
        assert not updates
        assert sum(path == "/system/bootstrap" for _, path, _ in backend.requests) == 1
        assert sum(path == "/system/telemetry" for _, path, _ in backend.requests) >= 2
        assert not any(path in {"/models", "/adapters", "/system/status", "/train", "/sleep/runs", "/system/gpu"}
                       for _, path, _ in backend.requests)
        backend.trains = [{"id": "train_live", "status": "running"}]
        await app.refresh_panels()
        assert "train_live" in app.query_one("#jobs").render().plain
        assert sum(path == "/system/bootstrap" for _, path, _ in backend.requests) == 1
        backend.models.append({"id": "second", "status": "ready"})
        backend.revision += 1
        await app.refresh_panels()
        assert sum(path == "/system/bootstrap" for _, path, _ in backend.requests) == 2
        assert len(app._model_options) == 2
        assert not any(path == "/models/load" for _, path, _ in backend.requests)


@pytest.mark.asyncio
async def test_heartbeat_response_after_terminal_shutdown_does_not_touch_unmounted_widgets():
    backend = Backend()
    started, release = asyncio.Event(), asyncio.Event()
    async def handle(request):
        if request.url.path == '/system/telemetry':
            started.set()
            await release.wait()
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    pending = None
    try:
        async with app.run_test(size=(88, 30)):
            pending = asyncio.create_task(app.refresh_panels())
            await asyncio.wait_for(started.wait(), 2)
    finally:
        release.set()
        if pending:
            await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('job_status', [None, 'rejected', 'training'])
async def test_learning_heartbeat_blocks_generation_until_inference_restoration_finishes(job_status):
    from lucifer.adapter_map import AdapterBrain
    backend = Backend()
    backend.status.update(model=backend.models[0], learning_busy=False, jobs=[])
    backend.adapters = [{'id': 'a1', 'base_model_id': 'demo', 'status': 'archived', 'metrics': {'passed': True}}]
    def handle(request):
        if request.url.path == '/sleep/status':
            backend.requests.append((request.method, request.url.path, {}))
            return httpx.Response(200, json=[])
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(140, 45)) as pilot:
        app.agent_run_id = 'retained_mission'
        brain = app.query_one(AdapterBrain)
        brain.selected_id = 'adapter:a1'
        app.update_readiness()
        assert not app.query_one('#brain-select', Button).disabled
        backend.status['learning_busy'] = True
        if job_status:
            backend.sleeps = [{'id': 'sleep_latest', 'status': job_status}]
        backend.status['jobs'] = backend.sleeps if job_status == 'training' else []
        await app.refresh_panels()
        label = 'OFFLINE LEARNING IN PROGRESS' if job_status == 'training' else 'RESTORING INFERENCE'
        assert label in app.query_one('#status').render().plain
        assert label in app.query_one('#jobs').render().plain
        assert label in app.query_one('#busy').render().plain
        assert label in app.query_one('#mission-status').render().plain
        assert 'IDLE' not in app.query_one('#jobs').render().plain
        assert 'commands remain available' in app.query_one('#prompt', Input).placeholder
        for selector in ('#send', '#model-picker', '#adapter-picker', '#resume-mission', '#brain-select'):
            assert app.query_one(selector).disabled
        await submit(app, pilot, '/sleep status')
        assert any(path == '/sleep/status' for _, path, _ in backend.requests)
        assert app.query_one('#send', Button).disabled
        draft = 'Continue after restoration'
        app.query_one('#prompt', Input).value = draft
        await pilot.press('enter')
        assert app.query_one('#prompt', Input).value == draft
        for command in ('/model load demo', '/adapter select a1', '/resume', '/sleep start', '/experts build Test'):
            app.submit(command)
        blocked_paths = {'/models/load', '/adapters/select', '/chat/stream', '/sleep/start', '/experts/build'}
        assert not any(path in blocked_paths or path.endswith('/resume') for _, path, _ in backend.requests)
        backend.status['learning_busy'] = False
        backend.status['jobs'] = []
        backend.sleeps = []
        await app.refresh_panels()
        assert app.query_one('#prompt', Input).value == draft
        assert app.query_one('#busy').render().plain == ''
        assert 'ONLINE' in app.query_one('#status').render().plain
        for selector in ('#send', '#model-picker', '#adapter-picker', '#resume-mission', '#brain-select'):
            assert not app.query_one(selector).disabled
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1


class SleepBackend(Backend):
    def __init__(self):
        super().__init__()
        self.status.update(model=self.models[0], learning_busy=False, jobs=[])
        self.job = {'id': 'sleep_dynamic', 'status': 'queued',
                    'progress': {'stage': 'queued', 'label': 'Selecting reviewed information',
                                 'fraction': None, 'overall_fraction': None, 'completed': None, 'total': None,
                                 'optimizer_step': None, 'optimizer_steps_total': None,
                                 'eval_completed': None, 'eval_total': None, 'terminal': False, 'restored': False}}

    def handle(self, request):
        if request.url.path == '/sleep/start':
            self.requests.append((request.method, request.url.path, json.loads(request.content)))
            self.sleeps = [self.job]
            self.status.update(learning_busy=True, jobs=[self.job])
            return httpx.Response(202, json=self.job)
        if request.url.path == '/sleep/cancel':
            self.requests.append((request.method, request.url.path, json.loads(request.content)))
            self.job.update(status='cancelled')
            self.job['progress'].update(stage='finished', label='Cancelled; inference restored',
                                        fraction=None, terminal=True, restored=True, overall_fraction=1)
            self.status.update(learning_busy=False, jobs=[])
            return httpx.Response(200, json=self.job)
        return super().handle(request)


class ExampleBackend(SleepBackend):
    def __init__(self):
        super().__init__()
        self.examples = {ident: {'id': ident, 'status': 'pending', 'source_group_id': 'source_'+ident,
            'partition': 'unassigned', 'validation': {'method': 'needs_human_review'},
            'provenance': {'source_conversation_id': 'conv_test'},
            'reasoning_content': '__PROVIDER_PRIVATE_TRACE__',
            'messages': [{'role': 'user', 'content': f'Question [red]literal[/red] {ident}'},
                         {'role': 'assistant', 'content': 'A proposed **literal** answer.'}]}
            for ident in ('example_a', 'example_b')}
        for record in self.examples.values():
            self.hash_record(record)

    @staticmethod
    def hash_record(record):
        record['content_hash'] = hashlib.sha256(json.dumps(record['messages'], sort_keys=True).encode()).hexdigest()

    def handle(self, request):
        if request.url.path == '/sleep/examples':
            self.requests.append((request.method, request.url.path, dict(request.url.params)))
            status = request.url.params.get('status')
            return httpx.Response(200, json=[record for record in self.examples.values() if not status or record['status'] == status])
        if request.url.path.startswith('/sleep/examples/') and request.url.path.endswith('/review'):
            payload = json.loads(request.content)
            self.requests.append((request.method, request.url.path, payload))
            record = self.examples[request.url.path.split('/')[-2]]
            if payload['content_hash'] != record['content_hash']:
                return httpx.Response(409, json={'detail': 'Example content changed; reload it before reviewing'})
            record['status'] = 'verified' if payload['decision'] == 'approve' else 'rejected'
            if payload.get('correction'):
                record['messages'][-1]['content'] = payload['correction']
            record['validation'] = {'method': 'human_reviewed', 'human_reviewed': True}
            self.hash_record(record)
            return httpx.Response(200, json=record)
        return super().handle(request)


@pytest.mark.asyncio
async def test_example_review_is_explicit_literal_hash_bound_and_heartbeat_does_not_poll_queue():
    backend = ExampleBackend()
    original_hash = backend.examples['example_a']['content_hash']
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.press('f4')
        await pilot.click('#sleep-review-examples')
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ExampleReviewScreen)
        assert '[red]literal[/red]' in screen.query_one('#example-question').render().plain
        assert '**literal**' in screen.query_one('#example-answer').render().plain
        assert original_hash in screen.query_one('#example-hash').render().plain
        assert '__PROVIDER_PRIVATE_TRACE__' not in ''.join(screen.query_one(selector, Static).render().plain
            for selector in ('#example-hash', '#example-metadata', '#example-question', '#example-answer', '#example-status'))
        assert 'example_a' in app._example_displayed_hashes and 'example_b' not in app._example_displayed_hashes
        for _ in range(3):
            await app.refresh_panels()
        assert len([1 for method, path, _ in backend.requests if method == 'GET' and path == '/sleep/examples']) == 1
        correction = 'A complete answer.\n```python\n    print("literal")\n```\n'
        screen.query_one('#example-correction', TextArea).load_text(correction)
        screen.query_one('#example-reason', Input).value = 'Checked against the source'
        await pilot.click('#example-approve')
        await app.workers.wait_for_complete()
        payload = next(data for method, path, data in backend.requests if method == 'POST' and path.endswith('/example_a/review'))
        assert payload == {'decision': 'approve', 'content_hash': original_hash, 'correction': correction, 'reason': 'Checked against the source'}
        assert 'human_reviewed' in screen.query_one('#example-metadata').render().plain
        assert 'verified' in screen.query_one('#example-status').render().plain
        screen.query_one('#example-picker', Select).value = 'example_b'
        await pilot.pause()
        b_hash = backend.examples['example_b']['content_hash']
        await pilot.click('#example-reject')
        await app.workers.wait_for_complete()
        payload = next(data for method, path, data in backend.requests if method == 'POST' and path.endswith('/example_b/review'))
        assert payload == {'decision': 'reject', 'content_hash': b_hash}
        assert len([1 for method, path, _ in backend.requests if method == 'GET' and path == '/sleep/examples']) == 1
        assert not any(path in {'/privacy', '/feedback', '/sleep/start'} for _, path, _ in backend.requests)
        await pilot.press('escape')
        assert not isinstance(app.screen, ExampleReviewScreen)


@pytest.mark.asyncio
async def test_stale_example_review_preserves_correction_and_requires_explicit_reload_for_new_hash():
    backend = ExampleBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await submit(app, pilot, '/sleep examples pending')
        screen = app.screen
        old_hash = screen.displayed['content_hash']
        correction = 'The complete corrected answer\nwith another line.'
        screen.query_one('#example-correction', TextArea).load_text(correction)
        backend.examples['example_a']['messages'][-1]['content'] = 'Changed by another review'
        backend.hash_record(backend.examples['example_a'])
        new_hash = backend.examples['example_a']['content_hash']
        await pilot.click('#example-approve')
        await app.workers.wait_for_complete()
        assert 'content changed' in screen.query_one('#example-status').render().plain
        assert screen.query_one('#example-correction', TextArea).text == correction
        posts = [data for method, path, data in backend.requests if method == 'POST' and path.endswith('/review')]
        assert len(posts) == 1 and posts[0]['content_hash'] == old_hash
        assert sum(path == '/sleep/examples' for _, path, _ in backend.requests) == 1
        await pilot.click('#example-reload')
        await app.workers.wait_for_complete()
        assert new_hash in screen.query_one('#example-hash').render().plain
        assert 'Changed by another review' in screen.query_one('#example-answer').render().plain
        assert screen.query_one('#example-correction', TextArea).text == correction
        await pilot.click('#example-approve')
        await app.workers.wait_for_complete()
        posts = [data for method, path, data in backend.requests if method == 'POST' and path.endswith('/review')]
        assert len(posts) == 2 and posts[1]['content_hash'] == new_hash
        assert backend.examples['example_a']['messages'][-1]['content'] == correction
        for selector in ('#example-approve', '#example-reject', '#example-close'):
            assert screen.query_one(selector).region.bottom <= app.size.height


@pytest.mark.asyncio
async def test_example_commands_refuse_unseen_content_and_preserve_complete_correction_literals():
    backend = ExampleBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, '/sleep approve example_b')
        assert not any(method == 'POST' and path.endswith('/review') for method, path, _ in backend.requests)
        await submit(app, pilot, '/sleep examples pending')
        screen = app.screen
        screen.query_one('#example-picker', Select).value = 'example_b'
        await pilot.pause()
        content_hash = screen.displayed['content_hash']
        await pilot.press('escape')
        correction = 'A complete "quoted" answer.\n```python\n    value = 1\n```'
        await submit(app, pilot, '/sleep approve example_b ' + correction)
        posts = [data for method, path, data in backend.requests if method == 'POST' and path.endswith('/review')]
        assert posts == [{'decision': 'approve', 'content_hash': content_hash, 'correction': correction}]
        await submit(app, pilot, '/sleep reject example_b')
        assert len([1 for method, path, _ in backend.requests if method == 'POST' and path.endswith('/review')]) == 1


@pytest.mark.asyncio
async def test_example_review_does_not_cancel_or_repeat_an_inflight_decision():
    backend = ExampleBackend()
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []
    async def handle(request):
        if request.url.path.endswith('/review'):
            attempts.append(json.loads(request.content))
            entered.set()
            await release.wait()
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, '/sleep examples')
        screen = app.screen
        await pilot.click('#example-approve')
        await asyncio.wait_for(entered.wait(), 2)
        screen.approve_clicked()
        screen.reject_clicked()
        screen.reload_clicked()
        await pilot.press('escape')
        assert app.screen is screen and len(attempts) == 1
        release.set()
        await app.workers.wait_for_complete()
        assert len(attempts) == 1 and attempts[0]['decision'] == 'approve'
        assert backend.examples['example_a']['status'] == 'verified'
        assert 'Review saved' in screen.query_one('#example-status').render().plain


@pytest.mark.asyncio
async def test_versioned_review_selects_new_record_and_cannot_reapprove_obsolete_parent():
    import copy
    backend = ExampleBackend()
    def handle(request):
        if request.url.path.endswith('/review'):
            previous = copy.deepcopy(backend.examples['example_a'])
            response = backend.handle(request)
            updated = response.json()
            updated['id'] = 'example_a_v2'
            updated['validation'] = {'method': 'explicit_human_attestation', 'status': 'verified'}
            backend.examples['example_a'] = {**previous, 'status': 'obsolete', 'superseded_by': updated['id']}
            backend.examples[updated['id']] = updated
            return httpx.Response(200, json=updated)
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await submit(app, pilot, '/sleep examples')
        screen = app.screen
        await pilot.click('#example-approve')
        await app.workers.wait_for_complete()
        assert screen.query_one('#example-picker', Select).value == 'example_a_v2'
        assert 'human_reviewed' in screen.query_one('#example-provenance', Collapsible).title
        assert app._example_records['example_a']['status'] == 'obsolete'
        assert 'example_a' not in app._example_displayed_hashes
        screen.query_one('#example-picker', Select).value = 'example_a'
        await pilot.pause()
        assert screen.query_one('#example-approve', Button).disabled
        assert 'example_a' not in app._example_displayed_hashes
        await pilot.press('escape')
        await submit(app, pilot, '/sleep approve example_a')
        assert sum(method == 'POST' and path.endswith('/review') for method, path, _ in backend.requests) == 1


@pytest.mark.asyncio
async def test_review_reload_keeps_orphaned_draft_visible_without_approving_a_different_version():
    backend = ExampleBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await submit(app, pilot, '/sleep examples')
        screen = app.screen
        draft = 'Keep this complete correction until I inspect the new source.'
        screen.query_one('#example-correction', TextArea).load_text(draft)
        backend.examples['example_a']['status'] = 'obsolete'
        backend.examples['example_a_v2'] = {**backend.examples['example_a'], 'id': 'example_a_v2', 'status': 'pending'}
        await pilot.click('#example-reload')
        await app.workers.wait_for_complete()
        assert screen.query_one('#example-correction', TextArea).text == draft
        assert 'left this filter' in screen.query_one('#example-status').render().plain
        assert screen.query_one('#example-approve', Button).disabled
        assert 'example_a' not in app._example_displayed_hashes
        screen.approve_clicked()
        await app.workers.wait_for_complete()
        assert not any(method == 'POST' and path.endswith('/review') for method, path, _ in backend.requests)


@pytest.mark.asyncio
async def test_missing_optional_example_metadata_is_safe_and_missing_canonical_content_cannot_be_approved():
    backend = ExampleBackend()
    backend.examples = {'old': {'id': 'old', 'status': 'pending', 'content_hash': 'b'*64, 'validation': 'legacy', 'messages': None}}
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await submit(app, pilot, '/sleep examples')
        assert app.screen.query_one('#example-approve', Button).disabled
        assert 'not recorded' not in app.screen.query_one('#example-metadata').render().plain
        assert 'legacy' in app.screen.query_one('#example-metadata').render().plain
        assert 'old' not in app._example_displayed_hashes


@pytest.mark.asyncio
async def test_review_dialog_suspends_background_sleep_animation_and_restores_cached_progress():
    backend = ExampleBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.press('f4')
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        bars = [app.main_query(selector, SleepProgressBar) for selector in ('#sleep-overall', '#sleep-stage-bar')]
        assert all(bar.query_one('#bar').auto_refresh == 1/15 for bar in bars)
        await pilot.click('#sleep-review-examples')
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, ExampleReviewScreen)
        assert all(bar.query_one('#bar').auto_refresh is None for bar in bars)
        backend.job['progress'].update(fraction=.5, completed=1, total=2)
        await app.refresh_panels()
        await pilot.press('escape')
        await pilot.pause()
        assert bars[1].percentage == .5
        assert bars[0].query_one('#bar').auto_refresh == 1/15
        assert bars[1].query_one('#bar').auto_refresh is None


@pytest.mark.asyncio
async def test_curriculum_stages_counts_and_review_terminal_use_only_shared_heartbeat():
    backend = SleepBackend()
    backend.job['curriculum'] = {'source_conversations': 2, 'source_tasks': 3, 'source_groups': 2, 'drafted': 8,
        'verified': 3, 'train_examples': 2, 'heldout_examples': 1, 'pending': 2, 'rejected': 1, 'duplicates': 4,
        'obsolete': 0, 'min_verified_examples': 3, 'verified_tasks': 2, 'min_verified_tasks': 2}
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.press('f4')
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        for stage in ('recovering_context', 'drafting_examples', 'validating_examples', 'reconciling_memory', 'preparing_evaluation'):
            backend.job['status'] = stage
            backend.job['progress'].update(stage=stage, label=stage.replace('_', ' '), completed=None, total=None, fraction=None)
            await app.refresh_panels()
            assert stage.upper() in app.query_one('#sleep-stage').render().plain
        counters = app.query_one('#sleep-counters').render().plain
        for text in ('VERIFIED // 3 / MIN 3', 'TRAIN // 2 // HELD OUT 1 // PENDING 2',
                     'REJECTED // 1 // DUPLICATES 4', 'SOURCE GROUPS // 2', 'VERIFIED TASKS // 2 / MIN 2'):
            assert text in counters
        backend.job['status'] = 'needs_review'
        backend.job['progress'].update(stage='finished', terminal=True, restored=False)
        await app.refresh_panels()
        assert app.query_one('#sleep-overall', ProgressBar).percentage is None
        backend.job['progress']['restored'] = True
        await app.refresh_panels()
        assert app.query_one('#sleep-overall', ProgressBar).percentage is None
        backend.status.update(learning_busy=False, jobs=[])
        await app.refresh_panels()
        assert 'REVIEW REQUIRED' in app.query_one('#sleep-overall-label').render().plain
        assert 'NEEDS REVIEW' in app.query_one('#sleep-result').render().plain
        assert app.query_one('#sleep-review', Button).disabled
        assert not app.query_one('#sleep-review-examples', Button).disabled
        assert not any(path == '/sleep/examples' for _, path, _ in backend.requests)
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1
        await pilot.resize_terminal(88, 30)
        await pilot.pause()
        assert str(app.query_one('#sleep-review-examples', Button).label) == 'EXAMPLES'


def test_context_sleep_and_example_review_schema_bounds_are_explicit():
    from pydantic import ValidationError
    from lucifer.schemas import ExampleReview, ExpertBuild, SleepRequest
    defaults = SleepRequest()
    assert defaults.context_examples is True and defaults.min_verified_examples == 3 and defaults.max_generated_examples == 32
    assert SleepRequest(max_generated_examples=0, context_examples=False).max_generated_examples == 0
    assert ExpertBuild().context_examples is True
    for payload in ({'min_verified_examples': 0}, {'min_verified_examples': 5001}, {'max_generated_examples': 257},
                    {'max_generated_examples': -1}, {'generation_model': ''}):
        with pytest.raises(ValidationError):
            SleepRequest.model_validate(payload)
    ExampleReview(decision='approve', content_hash='a'*64, correction='A complete answer')
    for changes in ({'decision': 'accept'}, {'content_hash': 'bad'}, {'correction': 'x'*100001}, {'reason': 'x'*2001}, {'extra': True}):
        with pytest.raises(ValidationError):
            ExampleReview.model_validate({'decision': 'approve', 'content_hash': 'a'*64, **changes})


@pytest.mark.asyncio
async def test_sleep_progress_has_no_idle_or_hidden_timers_and_resumes_visible_unknown_work(monkeypatch):
    backend = SleepBackend()
    backend.job['dynamic_plan'] = {'evaluation_cases': 8}
    timers = []
    original_interval = SleepProgressBar.set_interval
    def interval(widget, *args, **kwargs):
        timers.append(args)
        return original_interval(widget, *args, **kwargs)
    monkeypatch.setattr(SleepProgressBar, 'set_interval', interval)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        bars = [app.query_one(selector, SleepProgressBar) for selector in ('#sleep-overall', '#sleep-stage-bar')]
        bodies = [bar.query_one('#bar') for bar in bars]
        assert all(body.auto_refresh is None for body in bodies)
        await pilot.press('f4')
        await pilot.pause(.1)
        assert all(body.auto_refresh is None for body in bodies)
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        await pilot.pause(.1)
        assert all(body.auto_refresh == 1 / 15 for body in bodies)
        assert 'EVALUATION RESPONSES // unknown / 16 planned' in app.query_one('#sleep-counters').render().plain
        renders = [0, 0]
        for index, body in enumerate(bodies):
            original_render = body.render
            def render(index=index, original=original_render):
                renders[index] += 1
                return original()
            monkeypatch.setattr(body, 'render', render)
        await pilot.press('f3')
        await pilot.pause(.1)
        assert all(body.auto_refresh is None for body in bodies)
        renders[:] = [0, 0]
        await pilot.pause(.2)
        assert renders == [0, 0]
        backend.job['progress'].update(fraction=.5, completed=2, total=4, optimizer_step=2, optimizer_steps_total=4)
        await app.refresh_panels()
        await pilot.pause(.1)
        assert bars[1].percentage is None
        assert app._jobs['sleep'][0]['progress']['fraction'] == .5
        assert all(body.auto_refresh is None for body in bodies)
        await pilot.press('f4')
        await pilot.pause(.1)
        assert bars[1].percentage == .5
        assert bodies[0].auto_refresh == 1 / 15 and bodies[1].auto_refresh is None
        assert 'OPTIMIZER STEPS // 2 / 4' in app.query_one('#sleep-counters').render().plain
        await pilot.press('f3')
        await pilot.pause(.1)
        renders[:] = [0, 0]
        backend.job['progress'].update(fraction=None, completed=None, total=None)
        await app.refresh_panels()
        await pilot.pause(.2)
        assert all(body.auto_refresh is None for body in bodies)
        assert renders == [0, 0]
        await pilot.press('f4')
        await pilot.pause(.2)
        assert all(body.auto_refresh == 1 / 15 for body in bodies)
        assert all(count > 0 for count in renders)
        backend.job['progress'].update(eval_total=12)
        await app.refresh_panels()
        assert 'EVALUATION RESPONSES // unknown / 12' in app.query_one('#sleep-counters').render().plain
        assert '16 planned' not in app.query_one('#sleep-counters').render().plain
        backend.job['status'] = 'no_new_information'
        backend.job['progress'].update(stage='finished', terminal=True, restored=True, overall_fraction=1)
        backend.status.update(learning_busy=False, jobs=[])
        await app.refresh_panels()
        await pilot.pause(.1)
        assert all(body.auto_refresh is None for body in bodies)
        renders[:] = [0, 0]
        await pilot.pause(.2)
        assert renders == [0, 0]
        assert timers == []
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1


@pytest.mark.asyncio
async def test_one_sleep_start_tracks_whole_pipeline_and_reviews_registered_expert_from_heartbeat(monkeypatch):
    from lucifer.adapter_map import AdapterBrain
    backend = SleepBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.press('f4')
        assert app.active_view == 'sleep'
        assert not any(path == '/sleep/start' for _, path, _ in backend.requests)
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        assert [data for _, path, data in backend.requests if path == '/sleep/start'] == [{'dynamic': True, 'dry_run': False, 'context_examples': True}]
        assert app.query_one('#sleep-start', Button).disabled
        assert not app.query_one('#sleep-cancel', Button).disabled
        assert app.query_one('#sleep-overall', ProgressBar).total is None
        assert app.query_one('#sleep-stage-bar', ProgressBar).total is None
        backend.job.update(status='training', dataset_metrics={'examples': 20}, replay_examples=3,
                           source_conversation_ids=['conv_1', 'conv_2'],
                           expert={'expert_name': 'Lucifer procedures', 'expert_topics': ['configuration'],
                                   'training_base_model_id': 'demo', 'source_expert_ids': []})
        backend.job['progress'].update(stage='training', label='Training the selected expert', fraction=.5,
                                        completed=2, total=4, optimizer_step=2, optimizer_steps_total=4)
        await app.refresh_panels()
        assert app.query_one('#sleep-stage-bar', ProgressBar).percentage == .5
        assert 'STAGE PROGRESS // 50%' in app.query_one('#sleep-stage-count').render().plain
        assert 'total work unknown' in app.query_one('#sleep-overall-label').render().plain
        assert 'OPTIMIZER STEPS // 2 / 4' in app.query_one('#sleep-counters').render().plain
        assert 'Lucifer procedures' in app.query_one('#sleep-plan').render().plain
        brain = app.query_one(AdapterBrain)
        graph_updates = []
        original = brain.set_data
        def counted(*args, **kwargs):
            graph_updates.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(brain, 'set_data', counted)
        backend.job['progress'].update(fraction=.75, completed=3, optimizer_step=3)
        backend.job['logs'] = [{'message': 'Another worker progress line'}]
        await app.refresh_panels()
        assert not graph_updates
        assert 'OPTIMIZER STEPS // 3 / 4' in app.query_one('#sleep-counters').render().plain
        candidate = {'id': 'expert_new', 'base_model_id': 'demo', 'status': 'candidate', 'is_expert': True,
                     'expert_name': 'Lucifer procedures', 'expert_training_status': 'trained', 'expert_quality_status': 'passed',
                     'metrics': {'passed': True}, 'artifact_verification': {'optimizer_steps': 4, 'tensor_count': 12}}
        backend.adapters.append(candidate)
        backend.revision += 1
        backend.job.update(status='evaluating', candidate_adapter_id='expert_new')
        backend.job['progress'].update(stage='evaluating', label='Comparing baseline and candidate', fraction=.25,
                                        completed=1, total=4, eval_completed=1, eval_total=4)
        await app.refresh_panels()
        assert brain.atlas.node('adapter:expert_new').label.startswith('E')
        assert 'EVALUATION RESPONSES // 1 / 4' in app.query_one('#sleep-counters').render().plain
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 2
        assert len(graph_updates) == 1
        backend.job['status'] = 'awaiting_promotion'
        backend.status['jobs'] = []
        backend.job['progress'].update(stage='restoring', label='Restoring inference', fraction=None,
                                        completed=None, total=None, overall_fraction=1)
        await app.refresh_panels()
        assert app.query_one('#sleep-overall', ProgressBar).total is None
        assert '100%' not in app.query_one('#sleep-overall-label').render().plain
        assert app.query_one('#sleep-review', Button).disabled
        backend.job['progress'].update(stage='finished', label='Candidate ready for review', terminal=True, restored=True)
        await app.refresh_panels()
        assert app.query_one('#sleep-overall', ProgressBar).total is None
        backend.status['learning_busy'] = False
        await app.refresh_panels()
        assert app.query_one('#sleep-overall', ProgressBar).percentage == 1
        assert app.query_one('#sleep-overall-label').render().plain == 'OVERALL // 100%'
        assert not app.query_one('#sleep-review', Button).disabled
        assert not app.query_one('#sleep-start', Button).disabled
        assert app.query_one('#sleep-cancel', Button).disabled
        assert 'promotion remains manual' in app.query_one('#sleep-result').render().plain
        await pilot.click('#sleep-review')
        await pilot.pause()
        assert app.active_view == 'brain' and brain.selected_id == 'adapter:expert_new'
        assert app.query_one('#brain-select', Button).disabled
        assert 'Lucifer procedures' in app.query_one('#brain-details', Markdown).source
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 2
        assert not any(path in {'/models/load', '/adapters/promote', '/adapters/select', '/sleep/status'} for _, path, _ in backend.requests)
        await pilot.resize_terminal(88, 30)
        await pilot.press('f4')
        for selector in ('#sleep-start', '#sleep-cancel', '#sleep-review'):
            assert app.query_one(selector).region.bottom <= app.size.height - 4


@pytest.mark.asyncio
async def test_sleep_cancel_button_targets_the_selected_owned_job():
    backend = SleepBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await pilot.press('f4')
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        await pilot.click('#sleep-cancel')
        await app.workers.wait_for_complete()
        assert [data for _, path, data in backend.requests if path == '/sleep/cancel'] == [{'run_id': 'sleep_dynamic'}]
        assert 'CANCELLED' in app.query_one('#sleep-run').render().plain
        assert app.query_one('#sleep-review', Button).disabled
        assert not app.query_one('#sleep-start', Button).disabled
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1


@pytest.mark.asyncio
async def test_sleep_start_error_is_visible_in_pipeline_without_losing_manual_review_gates():
    backend = Backend()
    def handle(request):
        if request.url.path == '/sleep/start':
            return httpx.Response(409, json={'detail': 'Enable logging, training consent and reviewed feedback first'})
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await pilot.press('f4')
        await pilot.click('#sleep-start')
        await app.workers.wait_for_complete()
        assert 'training consent' in app.query_one('#sleep-result').render().plain
        assert app.query_one('#sleep-review', Button).disabled
        assert not app.query_one('#sleep-start', Button).disabled
        assert app.query_one('#sleep-overall', ProgressBar).percentage is None


@pytest.mark.asyncio
@pytest.mark.parametrize(('status', 'restored'), [('no_new_information', True), ('dry_run', True),
                                                ('quality_unknown', True), ('failed', False)])
async def test_sleep_terminal_outcomes_distinguish_quality_preview_and_restore_failure(status, restored):
    backend = SleepBackend()
    backend.job.update(status=status, note='No new reviewed information' if status == 'no_new_information' else 'Dataset preview only')
    backend.job['progress'].update(stage='finished', terminal=True, restored=restored,
                                   overall_fraction=1 if restored else None, label='Pipeline finished')
    if status == 'quality_unknown':
        backend.job.update(candidate_adapter_id='expert_unknown', dynamic_plan={
            'base': 'demo', 'evaluation_cases': 4, 'coverage': {'status': 'partial',
            'supported_question_count': 2, 'source_question_count': 3}})
    if status == 'failed':
        backend.job['restoration_error'] = 'Owned inference process failed to start'
    backend.sleeps = [backend.job]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await pilot.press('f4')
        assert status.upper() in app.query_one('#sleep-run').render().plain
        assert not app.query_one('#sleep-start', Button).disabled
        assert app.query_one('#sleep-overall', ProgressBar).percentage == (1 if restored else None)
        assert app.query_one('#sleep-review', Button).disabled is (status != 'quality_unknown')
        if status == 'quality_unknown':
            assert 'coverage is incomplete' in app.query_one('#sleep-result').render().plain
            assert '4 cases' in app.query_one('#sleep-plan').render().plain
            assert 'SUPPORTED SOURCE QUESTIONS // 2 / 3' in app.query_one('#sleep-plan').render().plain
        elif not restored:
            assert '100%' not in app.query_one('#sleep-overall-label').render().plain
            assert 'restoration needs attention' in app.query_one('#sleep-overall-label').render().plain
            assert not app.query_one('#sleep-overall', ProgressBar).display
            assert 'Could not restore inference' in app.query_one('#sleep-result').render().plain


@pytest.mark.asyncio
async def test_live_gpu_current_bars_and_compact_summary_change_without_registry_fetches():
    backend = Backend()
    card = {"index": 0, "backend": "nvidia", "name": "RTX 5070", "utilization_pct": 3,
            "memory_used_mb": 3000, "free_mb": 9000, "total_mb": 12000, "power_w": 21, "temp_c": 42}
    backend.gpu["gpus"] = [card]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 45)) as pilot:
        first = app.query_one("#gpus").render().plain
        assert "3%" in first and "21 W" in first and "42°C" in first
        gpu_region = app.query_one("#gpus").region
        assert gpu_region.y >= 0 and gpu_region.bottom <= app.size.height - 5
        card.update(utilization_pct=87, memory_used_mb=10000, free_mb=2000, power_w=181, temp_c=66)
        backend.gpu["sampled_at"] = "2026-11-12T12:00:01Z"
        await app.refresh_panels()
        live = app.query_one("#gpus").render().plain
        assert "87%" in live and "10,000 / 12,000 MiB" in live and "181 W" in live and "66°C" in live
        assert "░" in live and "█" in live and "87%" in live
        assert "87%" in app.query_one("#session").render().plain
        backend.gpu["stale"] = True
        backend.gpu["errors"] = ["Sampler timed out"]
        await app.refresh_panels()
        assert "STALE" in app.query_one("#gpus").render().plain
        assert "Sampler timed out" in app.query_one("#gpus").render().plain
        assert sum(path == "/system/bootstrap" for _, path, _ in backend.requests) == 1
        await pilot.resize_terminal(88, 30)
        assert app.has_class("compact")
        assert app.query_one("#thinking-picker", Select).display is True
        assert "87%" in app.query_one("#session").render().plain
        assert app.query_one("#session").content_region.height >= 2


@pytest.mark.asyncio
async def test_telemetry_disconnect_marks_stale_then_reconnect_bootstraps_once():
    backend = Backend()
    broken = False

    def handle(request):
        if broken and request.url.path == "/system/telemetry":
            backend.requests.append((request.method, request.url.path, {}))
            return httpx.Response(503, json={"detail": "Sampler offline"})
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)):
        broken = True
        await app.refresh_panels()
        assert "Backend unavailable" in app.query_one("#status").render().plain
        assert "STALE" in app.query_one("#gpus").render().plain
        broken = False
        await app.refresh_panels()
        assert "ONLINE" in app.query_one("#status").render().plain
        assert sum(path == "/system/bootstrap" for _, path, _ in backend.requests) == 2


def test_original_portrait_has_halo_wings_and_no_skull_features():
    from lucifer.tui import angel_portrait
    portrait = angel_portrait()
    assert len(portrait.plain.splitlines()) == 5
    assert "◇" in portrait.plain and "L U C I F E R" in portrait.plain
    assert "-   -" not in portrait.plain
    assert "X  X" not in portrait.plain and "[__]" not in portrait.plain


class AgentBackend(Backend):
    def __init__(self):
        super().__init__()
        self.models = [{"id": "lucifer", "backend": "external", "status": "loaded"}]
        self.status = {"model": self.models[0], "active_adapter": None}
        self.run = {"id": "agent_test", "status": "queued"}
        self.event_requests = []

    def snapshot(self):
        return {**super().snapshot(), "capabilities": {"capabilities": {"tools": {"status": "advertised"}}},
                "agent_runs": [self.run]}

    def handle(self, request):
        path = request.url.path
        if path.startswith('/agent/'):
            payload = json.loads(request.content) if request.content else {}
            self.requests.append((request.method, path, payload))
            if path == '/agent/runs' and request.method == 'POST':
                self.run.update(payload)
                return httpx.Response(202, json=self.run)
            if path == '/agent/runs/agent_test/events':
                self.event_requests.append(dict(request.url.params))
                events = [
                    ('phase', {'sequence': 1, 'phase': 'planning', 'round': 1}),
                    ('tool_requested', {'sequence': 2, 'call_id': 'call1', 'name': 'read_file', 'arguments': {'path': 'README.md'}}),
                    ('tool_started', {'sequence': 3, 'name': 'read_file'}),
                    ('tool_result', {'sequence': 4, 'name': 'read_file', 'summary': 'Read 6 lines'}),
                    ('reasoning', {'sequence': 5, 'content': 'Actual provider reasoning.'}),
                    ('delta', {'sequence': 6, 'content': '## Found\n\nThe definition is in `app.py:12`.'}),
                    ('final', {'sequence': 7, 'content': '## Found\n\nThe definition is in `app.py:12`.', 'message_id': 'agent_answer', 'conversation_id': 'conv_test'}),
                    ('done', {'sequence': 8, 'status': 'completed', 'run_id': 'agent_test'}),
                ]
                return httpx.Response(200, text=''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n' for event, data in events))
            if path == '/agent/runs/agent_test':
                return httpx.Response(200, json=self.run)
            if path == '/agent/runs/agent_test/session':
                return httpx.Response(200, json={'run': self.run, 'conversation': self.conv, 'messages': []})
            if path == '/agent/history':
                return httpx.Response(200, json=[{**self.run, 'run_id': self.run['id'], 'session_available': True}])
            if path.endswith(('/cancel', '/approve')):
                return httpx.Response(200, json={'status': 'cancelled' if path.endswith('/cancel') else 'approved'})
            if path == '/agent/runs':
                return httpx.Response(200, json=[self.run])
        return super().handle(request)


@pytest.mark.asyncio
async def test_single_agent_view_streams_tools_reasoning_and_does_not_toggle_to_chat():
    backend = AgentBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        assert app.input_mode == 'agent' and app.has_class('agent-mode')
        assert app.output_budget() == 2048
        assert app.query_one('#agent-controls').display
        app.thinking_mode = 'high'
        await submit(app, pilot, "Find the definition of the main function")
        starts = [payload for method, path, payload in backend.requests if path == '/agent/runs' and method == 'POST']
        assert len(starts) == 1
        assert starts[0]['access'] == 'read' and starts[0]['auto_approve_commands'] is False
        assert starts[0]['conversation_id'] == 'conv_test'
        assert app.last_message_id == 'agent_answer'
        assert app.query_one('.answer', Markdown).source.count('## Found') == 1
        assert app.query_one('.thinking-content', Markdown).source == 'Actual provider reasoning.'
        assert app.query_one('.tool-timeline').display is True
        assert backend.event_requests == [{'after': '0'}]
        assert all('Actual provider reasoning' not in json.dumps(data) for _, _, data in backend.requests)
        assert not list(app.query('#view-chat'))
        await pilot.press('f2')
        backend.revision += 1
        await app.refresh_panels()
        assert app.input_mode == 'agent' and app.active_view == 'agent'


@pytest.mark.asyncio
async def test_agent_reconnect_reuses_run_and_deduplicates_replayed_events():
    backend = AgentBackend()
    attachments = 0

    def handle(request):
        nonlocal attachments
        if request.url.path.endswith('/events'):
            attachments += 1
            backend.event_requests.append(dict(request.url.params))
            if attachments == 1:
                return httpx.Response(200, text='event: delta\ndata: {"sequence":1,"content":"Hello "}\n\n')
            return httpx.Response(200, text='event: delta\ndata: {"sequence":1,"content":"Hello "}\n\nevent: delta\ndata: {"sequence":2,"content":"world."}\n\nevent: final\ndata: {"sequence":3,"message_id":"final"}\n\nevent: done\ndata: {"sequence":4,"status":"completed"}\n\n')
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, 'Complete the task')
        assert app.query_one('.answer', Markdown).source == 'Hello world.'
        assert len([1 for method, path, _ in backend.requests if method == 'POST' and path == '/agent/runs']) == 1
        assert backend.event_requests == [{'after': '0'}, {'after': '1'}]


@pytest.mark.asyncio
async def test_agent_approval_shows_concrete_arguments_and_requires_user_decision():
    from lucifer.tui import ApprovalScreen
    backend = AgentBackend()
    stream = HeldStream(
        'event: approval_required\ndata: {"sequence":1,"approval_id":"approval_one","name":"run_command","arguments":{"argv":["python3","report.py"],"cwd":"/workspace"},"summary":"Run report.py"}\n\n',
        'event: delta\ndata: {"sequence":2,"content":"Finished."}\n\nevent: done\ndata: {"sequence":3,"status":"completed"}\n\n',
    )

    def handle(request):
        if request.url.path.endswith('/events'):
            return httpx.Response(200, stream=stream)
        if request.url.path.endswith('/approve'):
            result = backend.handle(request)
            stream.resume.set()
            return result
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        app.submit('Run the report')
        await asyncio.wait_for(stream.started.wait(), 2)
        await pilot.pause(.1)
        assert isinstance(app.screen, ApprovalScreen)
        review = app.screen.query_one('#approval-review').query('Static')
        assert any('report.py' in item.render().plain for item in review)
        assert not any(path.endswith('/approve') for _, path, _ in backend.requests)
        await app.refresh_panels()  # Main telemetry remains usable under a modal.
        await pilot.click('#approve-operation')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any(data == {'approval_id': 'approval_one', 'approved': True} for _, path, data in backend.requests if path.endswith('/approve'))
        assert app.query_one('.answer', Markdown).source == 'Finished.'


@pytest.mark.asyncio
async def test_direct_computer_access_and_automatic_commands_are_explicit_ui_choices():
    from textual.widgets import Checkbox
    backend = AgentBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        assert app.agent_access == 'read' and not app.agent_auto_approve
        app.query_one('#agent-access', Select).value = 'computer'
        await pilot.pause()
        assert app.query_one('#agent-auto', Checkbox).display
        assert 'HOST ACCESS' in app.query_one('#mission-status').render().plain
        app.query_one('#agent-auto', Checkbox).value = True
        await pilot.pause()
        await submit(app, pilot, 'Inspect my host environment')
        start = next(data for method, path, data in backend.requests if path == '/agent/runs' and method == 'POST')
        assert start['access'] == 'computer' and start['auto_approve_commands'] is True


@pytest.mark.asyncio
async def test_cortex_navigation_is_interactive_without_extra_registry_polling():
    from lucifer.adapter_map import AdapterBrain
    backend = Backend()
    backend.adapters = [{'id': 'a1', 'base_model_id': 'demo', 'status': 'archived', 'metrics': {'passed': True}},
                        {'id': 'a2', 'base_model_id': 'demo', 'parent_adapter_id': 'a1', 'status': 'candidate'}]
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.press('f3')
        assert app.active_view == 'brain'
        brain = app.query_one(AdapterBrain)
        assert brain.has_focus and len(brain.atlas.nodes) == 3
        roll = brain.roll_angle
        await pilot.press('q')
        assert brain.roll_angle != roll and app.active_view == 'brain'
        await pilot.press('e')
        assert brain.roll_angle == pytest.approx(roll)
        signature, revision = brain.snapshot_signature(), brain.data_revision
        await app.refresh_panels()
        assert brain.snapshot_signature() == signature and brain.data_revision == revision
        await pilot.press('right', 'n', 'enter')
        assert brain.yaw > .1
        assert app.query_one('#brain-details', Markdown).source
        await pilot.resize_terminal(88, 30)
        await pilot.pause()
        assert brain.region.height >= 8
        await pilot.press('f3')
        assert app.active_view == 'agent'
        assert sum(path == '/system/bootstrap' for _, path, _ in backend.requests) == 1


def test_agent_goal_parser_preserves_natural_text_and_accepts_leading_scoped_options():
    options, goal = LuciferApp.parse_agent_goal('--workspace "/project with spaces" --access computer --auto-approve-commands Don\'t repeat completed work.')
    assert options == {'workspace': '/project with spaces', 'access': 'computer', 'auto_approve_commands': True}
    assert goal == "Don't repeat completed work."


@pytest.mark.asyncio
async def test_agent_image_vision_mcp_choices_and_approved_memory_are_explicit():
    backend = AgentBackend()

    def handle(request):
        if request.url.path.endswith('/remember'):
            backend.requests.append((request.method, request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={'status': 'remembered'})
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        assert not app.agent_allow_images and app.agent_images == [] and app.agent_mcp_servers == []
        await submit(app, pilot, '/agent image "fixtures/red square.png"')
        await submit(app, pilot, '/agent mcp notes,docs,notes')
        await submit(app, pilot, 'Describe the attached picture')
        payload = next(data for method, path, data in backend.requests if path == '/agent/runs' and method == 'POST')
        assert payload['images'] == ['fixtures/red square.png'] and payload['allow_images'] is True
        from lucifer.schemas import AgentRequest
        total = payload.get('max_total_tokens', AgentRequest.model_fields['max_total_tokens'].default)
        assert total is None or total > 65536
        assert payload['mcp_servers'] == ['notes', 'docs'] and payload['allow_mcp'] is True
        # An off request ignores even a provider that incorrectly emits reasoning.
        assert app.query_one('.thinking-content', Markdown).source == ''
        await submit(app, pilot, "/agent remember The user's approved finding.")
        assert any(data == {'summary': "The user's approved finding."} for _, path, data in backend.requests if path.endswith('/remember'))
        await submit(app, pilot, '/agent vision off')
        await submit(app, pilot, '/agent mcp none')
        assert not app.agent_allow_images and app.agent_images == [] and app.agent_mcp_servers == []


@pytest.mark.asyncio
@pytest.mark.parametrize('interrupt_key', ['escape', 'ctrl+c'])
async def test_agent_interrupt_pauses_only_owned_run_and_keeps_partial_output(interrupt_key):
    backend = AgentBackend()
    stream = HeldStream(
        'event: phase\ndata: {"sequence":1,"phase":"generating"}\n\nevent: delta\ndata: {"sequence":2,"content":"Partial result"}\n\n',
        'event: done\ndata: {"sequence":3,"status":"paused"}\n\n',
    )

    def handle(request):
        if request.url.path.endswith('/events'):
            return httpx.Response(200, stream=stream)
        if request.url.path.endswith('/stop'):
            backend.requests.append((request.method, request.url.path, {}))
            stream.resume.set()
            return httpx.Response(200, json={'status': 'paused', 'resumable': True})
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        app.submit('Investigate the workspace')
        await asyncio.wait_for(stream.started.wait(), 2)
        await pilot.press(interrupt_key)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any(path == '/agent/runs/agent_test/stop' for _, path, _ in backend.requests)
        assert app.query_one('.answer', Markdown).source == 'Partial result'
        assert app.query_one('.assistant').has_class('partial')
        assert not app.busy and app.last_message_id is None
        assert app.is_running
        stops = [(method, path) for method, path, _ in backend.requests if path.endswith('/stop')]
        assert stops == [('POST', '/agent/runs/agent_test/stop')]
        await pilot.press('ctrl+c')
        assert app.is_running
        assert stops == [(method, path) for method, path, _ in backend.requests if path.endswith('/stop')]


@pytest.mark.asyncio
async def test_idle_ctrl_c_stays_open_and_ctrl_q_is_explicit_quit():
    backend = Backend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.press('ctrl+c')
        assert app.is_running
        assert not any(method != 'GET' for method, _, _ in backend.requests)
        await pilot.press('ctrl+q')
        assert not app.is_running


@pytest.mark.asyncio
async def test_agent_final_event_replaces_intermediate_tool_round_prose():
    backend = AgentBackend()

    def handle(request):
        if request.url.path.endswith('/events'):
            events = [('delta', {'sequence': 1, 'content': 'Interim prose before a tool.'}),
                      ('tool_started', {'sequence': 2, 'name': 'read_file'}),
                      ('delta', {'sequence': 3, 'content': 'Draft answer.'}),
                      ('final', {'sequence': 4, 'content': '## Final\n\nVerified report.', 'message_id': 'report'}),
                      ('done', {'sequence': 5, 'status': 'completed'})]
            return httpx.Response(200, text=''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n' for event, data in events))
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, 'Investigate and report')
        assert app.query_one('.answer', Markdown).source == '## Final\n\nVerified report.'
        assert app.last_message_id == 'report'


class SessionBackend(AgentBackend):
    def __init__(self):
        super().__init__()
        self.run.update(segment=0, session_turn=0, resumable=True)
        self.models[0].update(effective_context_length=1048576, max_output_tokens=262144)
        self.auto_status = 'completed'

    def events(self, segment):
        offset = (segment - 1) * 8
        status = 'completed' if self.run.get('stop_when_done', True) else 'awaiting_input'
        self.auto_status = status
        return [
            ('phase', {'sequence': offset + 1, 'segment': segment, 'session_turn': segment, 'phase': 'generating'}),
            ('reasoning', {'sequence': offset + 2, 'segment': segment, 'content': f'Actual retained thinking {segment}.'}),
            ('delta', {'sequence': offset + 3, 'segment': segment, 'content': f'Turn {segment} answer.'}),
            ('final', {'sequence': offset + 4, 'segment': segment, 'content': f'Turn {segment} answer.',
                       'message_id': f'answer_{segment}', 'conversation_id': 'conv_test'}),
            ('done', {'sequence': offset + 5, 'segment': segment, 'session_turn': segment, 'status': status}),
        ]

    @staticmethod
    def frames(events):
        return ''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n' for event, data in events)

    def handle(self, request):
        path = request.url.path
        if path == '/agent/runs' and request.method == 'POST':
            payload = json.loads(request.content)
            self.requests.append((request.method, path, payload))
            self.run.update(payload, segment=1, session_turn=1, status='running')
            return httpx.Response(202, json=self.run)
        if path.endswith(('/input', '/resume')):
            payload = json.loads(request.content)
            self.requests.append((request.method, path, payload))
            self.run.update(payload.get('options', payload.get('limits', {})))
            self.run['segment'] += 1
            self.run['session_turn'] += 1
            return httpx.Response(200, json=self.run)
        if path.endswith('/events'):
            self.event_requests.append(dict(request.url.params))
            after = int(request.url.params.get('after', 0))
            events = [item for segment in range(1, self.run['segment'] + 1)
                      for item in self.events(segment) if item[1]['sequence'] > after]
            return httpx.Response(200, text=self.frames(events))
        if path.endswith('/stop'):
            payload = json.loads(request.content)
            self.requests.append((request.method, path, payload))
            return httpx.Response(200, json={**self.run, 'status': 'paused'})
        return super().handle(request)


@pytest.mark.asyncio
async def test_agent_followup_and_resume_keep_one_session_and_each_thinking_panel():
    from textual.widgets import Checkbox
    backend = SessionBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        app.query_one('#thinking-picker', Select).value = 'high'
        app.query_one('#agent-stop', Checkbox).value = False
        await pilot.pause()
        await submit(app, pilot, 'Investigate the original goal')
        assert app._agent_status == 'awaiting_input' and not app.busy
        assert not app.query_one('#resume-mission', Button).disabled
        first = list(app.query('.thinking-content'))[0]
        assert first.source == 'Actual retained thinking 1.'
        await submit(app, pilot, 'Correction: focus on the revised requirement')
        starts = [data for method, path, data in backend.requests if method == 'POST' and path == '/agent/runs']
        inputs = [data for method, path, data in backend.requests if path.endswith('/input')]
        assert len(starts) == 1 and len(inputs) == 1
        assert inputs[0]['content'] == 'Correction: focus on the revised requirement'
        assert starts[0]['timeout_seconds'] is None and starts[0]['max_rounds'] is None
        assert starts[0]['stop_when_done'] is False and starts[0]['continue_output'] is True
        panels = list(app.query('.thinking-content'))
        assert [item.source for item in panels] == ['Actual retained thinking 1.', 'Actual retained thinking 2.']
        await submit(app, pilot, '/agent resume')
        assert len([1 for _, path, _ in backend.requests if path.endswith('/resume')]) == 1
        assert app.agent_run_id == 'agent_test' and len(app.query('.thinking-content')) == 3
        assert app.last_message_id == 'answer_3'
        await submit(app, pilot, '/chat new')
        assert app.agent_run_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('queued_old_selection', [False, True])
async def test_correction_while_thinking_interrupts_same_run_and_retains_old_thought(queued_old_selection):
    backend = SessionBackend()
    first = HeldStream(
        SessionBackend.frames([('phase', {'sequence': 1, 'segment': 1, 'phase': 'generating'}),
                               ('reasoning', {'sequence': 2, 'segment': 1, 'content': 'Original provider thought retained.'})]),
        SessionBackend.frames([('done', {'sequence': 3, 'segment': 1, 'status': 'paused'})]),
    )
    attachments = 0

    def handle(request):
        nonlocal attachments
        if request.url.path.endswith('/input'):
            response = backend.handle(request)
            first.resume.set()
            return response
        if request.url.path.endswith('/events'):
            attachments += 1
            if attachments == 1:
                return httpx.Response(200, stream=first)
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    corrected_finished = asyncio.Event()
    original_finish = app.finish_agent_segment

    async def finished(segment, **kwargs):
        await original_finish(segment, **kwargs)
        if segment == 2:
            corrected_finished.set()

    app.finish_agent_segment = finished
    async with app.run_test(size=(140, 48)) as pilot:
        app.query_one('#thinking-picker', Select).value = 'max'
        await pilot.pause()
        app.submit('Start the mission')
        await asyncio.wait_for(first.started.wait(), 2)
        await pilot.pause(.1)
        assert app.busy and app.query_one('#correction-row').display
        assert not app.query_one('#correction', Input).disabled
        old = list(app.query('.assistant'))[0]
        old.thinking_panel.collapsed = False
        if queued_old_selection:
            picker = app.query_one('#thinking-picker', Select)
            app.thinking_selected(Select.Changed(picker, 'off'))
            assert picker.value == 'max' and app.thinking_mode == 'max'
        app.query_one('#correction', Input).value = 'Use the correction instead of the original assumption'
        await pilot.click('#correct-mission')
        # WorkerManager waits a snapshot; wait for this segment's actual final
        # Markdown flush before taking that snapshot or inspecting its widgets.
        await asyncio.wait_for(corrected_finished.wait(), 3)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len([1 for _, path, _ in backend.requests if path == '/agent/runs']) == 1
        assert len([1 for _, path, _ in backend.requests if path.endswith('/input')]) == 1
        panels = list(app.query('.thinking-content'))
        assert panels[0].source == 'Original provider thought retained.'
        assert panels[1].source == 'Actual retained thinking 2.', {
            'panels': len(panels), 'selected_thinking': app.thinking_mode,
            'reply_modes': [reply.thinking_mode for reply in app._agent_replies.values()],
            'reasoning_lengths': [len(reply.thinking_buffer.source) for reply in app._agent_replies.values()],
            'segment': app._agent_target_segment, 'cursor': app._agent_sequence, 'busy': app.busy,
        }
        assert old.thinking_panel.collapsed is False and old.has_class('partial')
        assert app.last_message_id == 'answer_2' and not app.busy


@pytest.mark.asyncio
async def test_dynamic_output_limit_and_continuation_follow_loaded_metadata():
    from textual.widgets import Checkbox
    backend = SessionBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, '/tokens 65536')
        assert app.output_budget() == 65536
        await submit(app, pilot, '/tokens max')
        assert app.output_budget() == 262144
        app.query_one('#agent-continue', Checkbox).value = False
        await pilot.pause()
        await submit(app, pilot, 'Produce a long answer')
        payload = next(data for _, path, data in backend.requests if path == '/agent/runs')
        assert payload['max_tokens'] == 262144 and payload['continue_output'] is False
        await submit(app, pilot, '/tokens 262145')
        assert 'Usage: /tokens' in list(app.query('.system'))[-1].render().plain


@pytest.mark.asyncio
async def test_agent_network_stream_and_controls_have_no_read_deadline():
    captured = []
    def handle(request):
        captured.append((request.url.path, request.extensions.get('timeout', {})))
        if request.url.path.endswith('/events'):
            return httpx.Response(200, text='event: done\ndata: {"status":"paused"}\n\n')
        return httpx.Response(200, json={'id': 'agent_one'})
    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        await client.start_agent({'goal': 'Long goal'})
        await client.agent_input('agent_one', 'Correction')
        await client.resume_agent('agent_one')
        await client.stop_agent('agent_one')
        assert [event async for event in client.stream_agent('agent_one')]
        assert all(timeout['read'] is None and timeout['connect'] == 5 for _, timeout in captured)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_each_agent_control_intent_gets_a_distinct_uuid_key():
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'agent_one'})
    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        for _ in range(2):
            await client.agent_input('agent_one', 'Same wording, a new user turn')
            await client.resume_agent('agent_one')
        keys = [request['idempotency_key'] for request in requests]
        assert len(set(keys)) == 4
        assert all(UUID(key).version == 4 for key in keys)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['input', 'resume'])
async def test_control_transport_failure_requires_explicit_retry_with_original_intent_key(operation):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ReadError('Response lost after acceptance', request=request)
        return httpx.Response(200, json={'id': 'agent_one', 'idempotent_replay': True})
    client = APIClient(transport=httpx.MockTransport(handle))
    try:
        async def send():
            if operation == 'input':
                return await client.agent_input('agent_one', 'Use the correction', interrupt=True,
                                                options={'thinking': 'max'}, idempotency_key='one-user-intent')
            return await client.resume_agent('agent_one', instruction='Keep the previous findings',
                                             limits={'timeout_seconds': None}, idempotency_key='one-user-intent')
        with pytest.raises(APIError, match='Response lost'):
            await send()
        assert len(requests) == 1
        assert (await send())['idempotent_replay'] is True
        assert len(requests) == 2 and requests[0] == requests[1]
        assert requests[0]['idempotency_key'] == 'one-user-intent'
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expert_build_and_consolidation_send_scoped_sources_without_reasoning():
    backend = Backend()
    def handle(request):
        if request.url.path.startswith('/experts'):
            payload = json.loads(request.content) if request.content else {}
            backend.requests.append((request.method, request.url.path, payload))
            return httpx.Response(200, json={'id': 'sleep_expert', 'status': 'queued'} if request.url.path.endswith('/build') else [])
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        app.conversation = backend.conv
        await submit(app, pilot, '/experts propose')
        await submit(app, pilot, '/experts build quantum --base train --topics quantum,algebra')
        await submit(app, pilot, '/experts consolidate merged expert1,expert2 --base train')
        values = [data for _, path, data in backend.requests if path == '/experts/build']
        assert values[0] == {'name': 'quantum', 'base_model': 'train', 'topics': ['quantum', 'algebra'], 'conversation_ids': ['conv_test']}
        assert values[1] == {'name': 'merged', 'source_expert_ids': ['expert1', 'expert2'], 'base_model': 'train'}
        assert app.latest_sleep == 'sleep_expert'


@pytest.mark.asyncio
async def test_reopening_live_backend_session_replays_all_thinking_without_reposting_goal():
    backend = SessionBackend()
    backend.run.update(segment=2, session_turn=2, resumable=True, workspace='/workspace', access='computer',
                       dialect='chat_completions', thinking='max', auto_approve_commands=True,
                       stop_when_done=False, continue_output=True, conversation_id='conv_test', goal='Original mission')
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await submit(app, pilot, '/agent open agent_test')
        assert app.agent_run_id == 'agent_test'
        assert [item.source for item in app.query('.thinking-content')] == [
            'Actual retained thinking 1.', 'Actual retained thinking 2.']
        assert app.agent_access == 'computer' and app.agent_auto_approve
        assert app.query_one('#agent-access', Select).value == 'computer'
        assert not any(method == 'POST' and path == '/agent/runs' for method, path, _ in backend.requests)
        assert not any(path.endswith(('/resume', '/input')) for _, path, _ in backend.requests)


class HistoryBackend(SessionBackend):
    def __init__(self):
        super().__init__()
        self.conv.update(logging_enabled=True, training_enabled=False)
        self.run.update(status='paused', segment=2, session_turn=2, last_sequence=13,
                        workspace='/saved/workspace', access='read', conversation_id='conv_test',
                        session_available=False, content_retained=False, resumable=False,
                        checkpoint_available=True, transcript_available=True, restore_available=True,
                        logging_enabled=True, training_enabled=False, goal='Original mission',
                        control_notice='An accepted follow-up was interrupted before its payload was retained. Resubmit explicitly if still needed.')
        self.history = [{**self.run, 'run_id': 'agent_test', 'title': 'Saved [literal] mission'},
                        {'id': 'private_old', 'run_id': 'private_old', 'title': 'Private old mission',
                         'status': 'paused', 'logging_enabled': False, 'training_enabled': False,
                         'session_available': False, 'checkpoint_available': False,
                         'transcript_available': False, 'restore_available': False,
                         'transcript_unavailable_reason': 'logging_disabled'}]
        self.messages = [{'id': 'user_old', 'role': 'user', 'content': 'A [literal] saved question'},
                         {'id': 'answer_old', 'role': 'assistant', 'content': '## Saved final\n\nA verified result.'}]

    def handle(self, request):
        path = request.url.path
        if path == '/agent/history':
            self.requests.append((request.method, path, {}))
            return httpx.Response(200, json=self.history)
        if path.endswith('/session'):
            self.requests.append((request.method, path, {}))
            if 'private_old' in path:
                return httpx.Response(200, json={'run': self.history[1], 'conversation': None,
                    'messages': [], 'transcript_unavailable_reason': 'logging_disabled',
                    'operations': [{'status': 'completed', 'operation_fingerprint': 'hash_only'}]})
            return httpx.Response(200, json={'run': self.run, 'conversation': self.conv,
                'messages': self.messages, 'reasoning_content': '__NEVER_SAVED_OR_RENDERED__'})
        return super().handle(request)


@pytest.mark.asyncio
async def test_history_is_on_demand_and_loads_checkpoint_transcript_without_execution():
    backend = HistoryBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        assert app.active_view == 'agent' and not list(app.query('#view-chat'))
        assert not any(path == '/agent/history' for _, path, _ in backend.requests)
        await pilot.press('f5')
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, AgentHistoryScreen)
        metadata = screen.query_one('#history-metadata', Static).render().plain
        assert 'LOGGING // ON // TRAINING // OFF' in metadata
        assert 'saved checkpoint' in metadata and 'provider reasoning was never saved' in metadata
        assert 'CONTROL NOTICE // An accepted follow-up was interrupted' in metadata
        assert '[literal]' in screen.query_one('#history-preview', Static).render().plain
        assert '__NEVER_SAVED_OR_RENDERED__' not in screen.query_one('#history-preview', Static).render().plain
        before = [(method, path) for method, path, _ in backend.requests if path.startswith('/agent/')]
        for _ in range(3):
            await app.refresh_panels()
        assert before == [(method, path) for method, path, _ in backend.requests if path.startswith('/agent/')]
        await pilot.click('#history-open')
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.agent_run_id == 'agent_test' and app._agent_sequence == 13
        assert app.agent_workspace == '/saved/workspace' and app.agent_access == 'read'
        assert app.query_one('.user').render().plain == 'A [literal] saved question'
        assert app.query_one('.answer', Markdown).source == '## Saved final\n\nA verified result.'
        assert all(not item.source for item in app.query('.thinking-content'))
        assert app.query_one('.thinking', Collapsible).title == 'Model thinking · not retained'
        assert app.last_message_id == 'answer_old'
        assert 'Resubmit explicitly' in ''.join(item.render().plain for item in app.query('.system'))
        assert not any(method == 'POST' for method, _, _ in backend.requests)
        await pilot.click('#resume-mission')
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert len([1 for method, path, _ in backend.requests if method == 'POST' and path.endswith('/resume')]) == 1
        assert backend.event_requests == [{'after': '13'}]
        assert [item.source for item in app.query('.answer')] == ['## Saved final\n\nA verified result.', 'Turn 3 answer.']


@pytest.mark.asyncio
async def test_history_keeps_private_restart_unavailability_visible_and_never_fabricates_text():
    backend = HistoryBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(88, 30)) as pilot:
        await pilot.click('#view-history')
        await app.workers.wait_for_complete()
        screen = app.screen
        assert isinstance(screen, AgentHistoryScreen)
        assert len(screen.records) == 2
        screen.query_one('#history-picker', Select).value = 'private_old'
        await pilot.pause()
        await app.workers.wait_for_complete()
        metadata = screen.query_one('#history-metadata', Static).render().plain
        assert 'LOGGING // OFF' in metadata and 'logging disabled' in metadata
        assert 'cannot be reconstructed' in metadata
        assert screen.query_one('#history-open', Button).region.right <= 88
        await pilot.click('#history-open')
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.agent_run_id is None and app.last_message_id is None
        assert app.query_one('#resume-mission', Button).disabled
        assert 'message text is unavailable' in list(app.query('.system'))[-1].render().plain
        assert not list(app.query('.user')) and not list(app.query('.assistant'))
        assert not any(method == 'POST' for method, _, _ in backend.requests)


@pytest.mark.asyncio
async def test_history_can_be_inspected_while_busy_but_cannot_switch_the_running_mission():
    backend = HistoryBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        app.busy = True
        app.agent_run_id = 'currently_running'
        await pilot.press('f5')
        await app.workers.wait_for_complete()
        screen = app.screen
        assert isinstance(screen, AgentHistoryScreen)
        assert screen.query_one('#history-open', Button).disabled
        assert screen.query_one('#history-new', Button).disabled
        assert 'Pause the current mission' in screen.query_one('#history-status', Static).render().plain
        await pilot.press('escape')
        assert app.agent_run_id == 'currently_running'
        assert not any(method == 'POST' for method, _, _ in backend.requests)
        app.busy = False


@pytest.mark.asyncio
async def test_history_reopens_ram_session_with_actual_reasoning_and_replaces_old_view():
    backend = HistoryBackend()
    backend.run.update(session_available=True, resumable=True, content_retained=True, thinking='max')
    backend.history[0].update(session_available=True, resumable=True)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await app.message('user', 'Old view must be replaced')
        await pilot.press('f5')
        await app.workers.wait_for_complete()
        await pilot.click('#history-open')
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert backend.event_requests == [{'after': '0'}]
        assert [item.source for item in app.query('.thinking-content')] == [
            'Actual retained thinking 1.', 'Actual retained thinking 2.']
        assert 'Old view must be replaced' not in ''.join(item.render().plain for item in app.query('.user'))
        assert not any(method == 'POST' for method, _, _ in backend.requests)


@pytest.mark.asyncio
async def test_history_replay_does_not_reask_completed_operation_approvals():
    backend = HistoryBackend()
    backend.run.update(session_available=True, resumable=True, content_retained=True, segment=1, status='completed')

    def handle(request):
        if request.url.path.endswith('/events'):
            return httpx.Response(200, text=SessionBackend.frames([
                ('approval_required', {'sequence': 1, 'segment': 1, 'approval_id': 'already_decided', 'name': 'write_file'}),
                ('tool_result', {'sequence': 2, 'segment': 1, 'call_id': 'prior_effect', 'name': 'write_file', 'status': 'completed'}),
                ('final', {'sequence': 3, 'segment': 1, 'content': 'The approved change finished.', 'message_id': 'saved_final'}),
                ('done', {'sequence': 4, 'segment': 1, 'status': 'completed'}),
            ]))
        return backend.handle(request)

    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, '/agent open agent_test')
        assert isinstance(app.screen, type(app._main_screen))
        assert app._pending_approval is None
        assert app.query_one('.answer', Markdown).source == 'The approved change finished.'
        assert not any(method == 'POST' for method, _, _ in backend.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_view", [None, "sleep"])
async def test_startup_session_reads_existing_run_without_resuming_or_saving_reasoning(initial_view):
    backend = SessionBackend()
    backend.run.update(status='paused', segment=2, session_turn=2, resumable=True, workspace='/workspace',
                       access='read', thinking='max', conversation_id='conv_test', goal='Original mission')
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), initial_agent_run_id='agent_test',
                     initial_view=initial_view)
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.agent_run_id == 'agent_test'
        assert [item.source for item in app.query('.thinking-content')] == [
            'Actual retained thinking 1.', 'Actual retained thinking 2.']
        assert all(method == 'GET' for method, _, _ in backend.requests)
        assert app.active_view == (initial_view or 'agent')
        assert app.has_class('sleep-view') is (initial_view == 'sleep')


@pytest.mark.asyncio
async def test_frontend_fatal_diagnostic_excludes_messages_locals_and_source_lines(tmp_path, capsys):
    from textual.worker import WorkerFailed
    backend = Backend()
    path = tmp_path / 'logs' / 'tui-crash.json'
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)), crash_report_path=path)

    async def fail_with_private_content():
        private_response_buffer = '__MUST_NOT_ENTER_CRASH_DIAGNOSTIC__'
        raise RuntimeError(private_response_buffer)

    with pytest.raises(WorkerFailed):
        async with app.run_test(size=(100, 35)) as pilot:
            app.run_worker(fail_with_private_content(), name='safe-diagnostic-test')
            await pilot.pause()
    encoded = path.read_text()
    report = json.loads(encoded)
    assert '__MUST_NOT_ENTER_CRASH_DIAGNOSTIC__' not in encoded
    assert 'private_response_buffer' not in encoded and 'RuntimeError(' not in encoded
    assert any(item['type'] == 'builtins.RuntimeError' and any(frame['function'] == 'fail_with_private_content'
               for frame in item['frames']) for item in report['failures'])
    assert path.stat().st_mode & 0o777 == 0o600
    assert app.return_code == 1
    output = capsys.readouterr()
    assert '__MUST_NOT_ENTER_CRASH_DIAGNOSTIC__' not in output.out + output.err


@pytest.mark.asyncio
async def test_scope_change_cannot_resume_hidden_host_grant_and_new_goal_keeps_conversation():
    backend = SessionBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(140, 48)) as pilot:
        await submit(app, pilot, '/agent access computer')
        await submit(app, pilot, 'First host task')
        await submit(app, pilot, '/agent access read')
        await submit(app, pilot, '/agent resume')
        assert not any(path.endswith('/resume') for _, path, _ in backend.requests)
        assert 'visible scope differs' in list(app.query('.system'))[-1].render().plain
        await submit(app, pilot, 'Next read-only task')
        starts = [data for method, path, data in backend.requests if method == 'POST' and path == '/agent/runs']
        assert len(starts) == 2 and starts[-1]['access'] == 'read'
        assert starts[0]['conversation_id'] == starts[1]['conversation_id'] == 'conv_test'


@pytest.mark.asyncio
async def test_failed_agent_start_does_not_leave_a_waiting_or_feedback_eligible_reply():
    backend = AgentBackend()
    def handle(request):
        if request.url.path == '/agent/runs' and request.method == 'POST':
            return httpx.Response(409, json={'detail': 'Load a compatible model'})
        return backend.handle(request)
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        app.last_message_id = 'previous_answer'
        await submit(app, pilot, 'New mission')
        assert not app.busy and app.last_message_id is None
        assert app.query_one('.assistant').has_class('partial')
        assert 'Load a compatible model' in list(app.query('.system'))[-1].render().plain
        assert not app.query_one('#send', Button).disabled
