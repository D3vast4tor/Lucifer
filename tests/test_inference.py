import asyncio
import json
from pathlib import Path
import socket
import sys

import httpx
import pytest
from pydantic import ValidationError
import yaml

from lucifer.config import Settings
from lucifer.inference import ModelManager, ModelSpec
from lucifer.store import Store


def manager(tmp_path, models):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": models}))
    settings = Settings(data_dir=tmp_path / "data", models_file=config, startup_timeout=.1)
    return ModelManager(settings, Store(settings.data_dir / "test.db"))


@pytest.mark.asyncio
async def test_vllm_openapi_startup_enables_native_turns_without_per_turn_discovery(tmp_path):
    engine = manager(tmp_path, [{"id": "trainable", "backend": "vllm", "path": "/models/local-checkpoint",
                               "endpoint": "http://localhost:8091/v1",
                               "served_model_name": "Lucifer", "context_length": 2048}])
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Lucifer", "max_model_len": 2048}]})
        if request.url.path == "/v1/status":
            return httpx.Response(404, json={"detail": "Not Found"})
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"paths": {"/v1/chat/completions": {"post": {}}}})
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        event = {"choices": [{"index": 0, "delta": {"content": "4"}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

    await engine.client.aclose()
    engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await engine.load("trainable")
        assert all(method == "GET" for method, _ in requests)
        snapshot = engine.capabilities.get("trainable")
        assert snapshot.supports("chat_completions") and not snapshot.supports("chat_completions", require_verified=True)
        assert not snapshot.supports("tools") and not snapshot.supports("vision")
        for _ in range(3):
            events = [event async for event in engine.agent_events([{"role": "user", "content": "2+2"}],
                      tools=[], model_id="trainable", profile_revision=snapshot.profile_revision,
                      max_tokens=32, thinking="off", tool_choice="none")]
            assert ("delta", {"content": "4"}) in events
            assert any(event == "finish" and data["finish_reason"] == "stop" for event, data in events)
            assert engine.capabilities.get("trainable") is snapshot
        assert requests.count(("GET", "/openapi.json")) == 1
        assert requests.count(("GET", "/v1/models")) == 1
        assert requests.count(("GET", "/v1/status")) == 1
        assert requests.count(("POST", "/v1/chat/completions")) == 3
    finally:
        await engine.close()
        engine.store.close()


def fake_strata(tmp_path, *, delay=0, loaded=True):
    """A foreground supervisor with a real HTTP API and a stubborn child."""
    script = tmp_path / "fake-strata.py"
    script.write_text('''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import time

port, model, child_file, delay, loaded = sys.argv[1:]
time.sleep(float(delay))
child_code = "import os,signal,sys,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)"
subprocess.Popen([sys.executable, "-c", child_code, child_file])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        payload = {"model": model, "loaded": loaded == "True"} if self.path == "/v1/status" else {"data": [{"id": model}]}
        self.wfile.write(json.dumps(payload).encode())
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert request["model"] == model
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for content, finish in [("hello", None), (" world", "stop")]:
            event = {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
            self.wfile.write(("data: " + json.dumps(event) + "\\n\\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\\n\\n")
HTTPServer(("127.0.0.1", int(port)), Handler).serve_forever()
''')
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    child_file = tmp_path / "runtime-child.pid"
    return [sys.executable, str(script), str(port), "deepseek-runtime", str(child_file), str(delay), str(loaded)], port, child_file


def process_running(pid):
    # A terminated orphan may briefly remain a zombie until its reaper runs.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.asyncio
async def test_managed_strata_serves_stream_and_cleans_supervisor_children(tmp_path):
    command, port, child_file = fake_strata(tmp_path, delay=.15)
    engine = manager(tmp_path, [{"id": "deepseek", "backend": "strata", "runtime_command": command,
        "port": port, "endpoint": f"http://127.0.0.1:{port}/v1", "served_model_name": "deepseek-runtime", "startup_timeout": 3}])
    try:
        spec = engine.specs["deepseek"]
        assert spec.managed
        assert engine.command(spec) == command
        loaded = await engine.load("deepseek")  # Exceeds the manager's .1s default timeout.
        assert loaded["pid"] is not None
        process = engine._process
        assert await engine.complete([{"role": "user", "content": "Hi"}]) == "hello world"
        with pytest.raises(ValueError, match="inference-only"):
            await engine.activate_adapter({"id": "adapter", "path": "/adapter"})
        for _ in range(100):
            if child_file.exists():
                break
            await asyncio.sleep(.01)
        child_pid = int(child_file.read_text())
        assert process_running(child_pid)
        await engine.unload()
        for _ in range(100):
            if not process_running(child_pid):
                break
            await asyncio.sleep(.01)
        assert not process_running(child_pid)
        assert process.returncode is not None
        assert engine._process is None
        assert engine._log is None
        assert engine.current() is None
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_strata_wrong_model_never_becomes_ready_and_is_cleaned(tmp_path):
    command, port, _ = fake_strata(tmp_path)
    engine = manager(tmp_path, [{"id": "wrong-model", "backend": "strata", "runtime_command": command,
        "port": port, "startup_timeout": .75}])
    try:
        with pytest.raises(RuntimeError, match="does not advertise model 'wrong-model'"):
            await engine.load("wrong-model")
        assert engine.current() is None
        assert engine._process is None
        assert engine._log is None
        assert engine.list()[0]["status"] == "error"
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_strata_advertised_but_unloaded_model_is_not_ready(tmp_path):
    command, port, _ = fake_strata(tmp_path, loaded=False)
    engine = manager(tmp_path, [{"id": "lucifer", "backend": "strata", "runtime_command": command,
        "port": port, "served_model_name": "deepseek-runtime", "startup_timeout": .75}])
    try:
        with pytest.raises(RuntimeError, match="is not loaded"):
            await engine.load("lucifer")
        assert engine.current() is None
        assert engine._process is None
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_borrowed_server_is_not_stopped_on_unload(tmp_path):
    command, port, _ = fake_strata(tmp_path)
    process = await asyncio.create_subprocess_exec(*command, start_new_session=True)
    engine = manager(tmp_path, [{"id": "borrowed", "backend": "external", "endpoint": f"http://127.0.0.1:{port}",
        "served_model_name": "deepseek-runtime", "startup_timeout": 3}])
    try:
        await engine.load("borrowed")
        assert await engine.complete([{"role": "user", "content": "Hi"}]) == "hello world"
        await engine.close()
        assert process.returncode is None
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"http://127.0.0.1:{port}/v1/models")
            assert response.status_code == 200
    finally:
        import os
        import signal
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        await engine.close()
        engine.store.close()


def test_strata_requires_explicit_argv_and_local_owned_endpoint():
    for values in ({}, {"runtime_command": []}, {"runtime_command": [""]}, {"runtime_command": ["strata", "\0"]},
        {"runtime_command": ["strata"], "endpoint": "http://example.com:8081"},
        {"runtime_command": ["strata"], "endpoint": "http://127.0.0.1:9000"},
        {"runtime_command": ["strata"], "extra_args": ["--other"]},
        {"runtime_command": ["strata"], "adapter_backend": "vllm"},
        {"runtime_command": ["strata"], "startup_timeout": 0}):
        with pytest.raises(ValidationError):
            ModelSpec(id="strata", backend="strata", **values)
    with pytest.raises(ValidationError, match="only supported for managed strata"):
        ModelSpec(id="borrowed", backend="external", endpoint="http://localhost:8081", runtime_command=["strata"])


@pytest.mark.asyncio
async def test_managed_port_probe_allows_time_wait_but_rejects_live_listener(tmp_path):
    command, _, _ = fake_strata(tmp_path)
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    command[2] = str(port)
    engine = manager(tmp_path, [{"id": "lucifer", "backend": "strata", "runtime_command": command,
        "port": port, "served_model_name": "deepseek-runtime", "startup_timeout": 3}])
    try:
        with pytest.raises(RuntimeError, match=f"port {port} is unavailable"):
            await engine.load("lucifer")
        assert engine._process is None
        client = socket.create_connection(("127.0.0.1", port))
        accepted, _ = listener.accept()
        accepted.shutdown(socket.SHUT_WR)
        accepted.close()  # The old server is the active TCP closer, entering TIME_WAIT.
        assert client.recv(1) == b""
        client.close()
        listener.close()
        with socket.socket() as probe:
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))  # Reproduces the old probe's failure.
        assert (await engine.load("lucifer"))["status"] == "loaded"
    finally:
        listener.close()
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_external_stream_and_adapter_routing(tmp_path):
    engine = manager(tmp_path, [{"id": "base", "backend": "external", "endpoint": "http://localhost:8900/v1", "adapter_backend": "vllm", "chat_template_kwargs": {"enable_thinking": False}}])
    requests = []
    async def handle(request):
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        if request.url.path == "/v1/load_lora_adapter":
            return httpx.Response(200, text="loaded")
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await engine.load("base")
        candidate = {"id": "candidate", "base_model_id": "base", "path": "/tmp/adapter"}
        assert await engine.complete([{"role": "user", "content": "Hi"}], adapter=candidate) == "hello world"
        assert json.loads(requests[-1].content)["model"] == "candidate"
        assert json.loads(requests[-1].content)["chat_template_kwargs"] == {"enable_thinking": False}
        assert engine.store.setting("active_adapter") is None
        assert await engine.complete([{"role": "user", "content": "Hi"}]) == "hello world"
        assert json.loads(requests[-1].content)["model"] == "base"
        assert sum(r.url.path.endswith("load_lora_adapter") for r in requests) == 1
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_stream_failure_is_not_silent(tmp_path):
    engine = manager(tmp_path, [{"id": "base", "backend": "external", "endpoint": "http://localhost:8900"}])
    async def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await engine.load("base")
        with pytest.raises(RuntimeError, match="completion marker"):
            await engine.complete([{"role": "user", "content": "Hi"}])
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_failed_model_load_cleans_process(tmp_path):
    engine = manager(tmp_path, [{"id": "bad", "backend": "vllm", "path": "unused", "executable": "/bin/false"}])
    try:
        with pytest.raises(RuntimeError):
            await engine.load("bad")
        assert engine.current() is None
        assert engine._process is None
        assert engine._log is None
        assert engine.list()[0]["status"] == "error"
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_process_heartbeat_avoids_sqlite_reads_and_marks_an_exited_engine(tmp_path, monkeypatch):
    engine = manager(tmp_path, [{"id": "demo", "backend": "demo"}])
    try:
        await engine.load("demo")
        reads = []
        original = engine.store.get
        def get(*args, **kwargs):
            reads.append(args)
            return original(*args, **kwargs)
        monkeypatch.setattr(engine.store, "get", get)
        engine.check_process()
        assert reads == []
        engine._process = await asyncio.create_subprocess_exec(sys.executable, "-c", "raise SystemExit(7)", start_new_session=True)
        await engine._process.wait()
        engine.check_process()
        assert reads == []
        assert engine._current is None
        assert original("models", "demo")["status"] == "error"
        assert "code 7" in original("models", "demo")["error"]
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_demo_is_explicit_and_rejects_real_adapters(tmp_path):
    engine = manager(tmp_path, [{"id": "demo", "backend": "demo"}])
    try:
        await engine.load("demo")
        assert "demonstration backend" in await engine.complete([{"role": "user", "content": "Hello"}])
        with pytest.raises(ValueError, match="Real adapters"):
            await engine.activate_adapter({"id": "real", "path": "real"})
        await engine.unload()
        with pytest.raises(ValueError, match="No model loaded"):
            await engine.complete([])
    finally:
        await engine.close()
        engine.store.close()


def test_gguf_requires_export_and_correct_base(tmp_path):
    engine = manager(tmp_path, [{"id": "gguf", "backend": "llama.cpp", "path": "/tmp/model.gguf"}])
    with pytest.raises(ValueError, match="GGUF LoRA"):
        engine._validate_adapter(engine.specs["gguf"], {"id": "peft", "path": "/tmp/adapter", "base_model_id": "gguf"})
    with pytest.raises(ValueError, match="different base"):
        engine._validate_adapter(engine.specs["gguf"], {"id": "peft", "path": "/tmp/adapter", "base_model_id": "different"})
    engine.store.close()


def test_endpoint_and_managed_binding_validation():
    for endpoint in ("file:///tmp", "http://user:password@localhost:8000", "http://localhost/?token=secret"):
        with pytest.raises(ValidationError):
            ModelSpec(id="x", backend="external", endpoint=endpoint)
    with pytest.raises(ValidationError):
        ModelSpec(id="x", backend="vllm", path="model", extra_args=["--host=0.0.0.0"])
    spec = ModelSpec(id="base", backend="vllm", path="Qwen/Qwen3-4B")
    assert spec.base_url == "http://127.0.0.1:8081/v1"


@pytest.mark.asyncio
async def test_engine_switch_waits_for_stream_cancellation(tmp_path):
    engine = manager(tmp_path, [{"id": "demo", "backend": "demo"}])
    try:
        await engine.load("demo")
        stream = engine.stream([{"role": "user", "content": "Hi"}])
        await anext(stream)
        unload = asyncio.create_task(engine.unload())
        await asyncio.sleep(.005)
        assert not unload.done()
        await stream.aclose()
        await unload
        assert engine.current() is None
    finally:
        await engine.close()
        engine.store.close()


def inventory_for(backend="cuda", count=4):
    return {"gpus": [{"index": str(i), "id": f"{backend}-{i}", "backend": backend, "name": f"GPU {i}"} for i in range(count)]}


@pytest.mark.asyncio
async def test_vllm_auto_splits_all_gpus_and_respects_masks(tmp_path, monkeypatch):
    monkeypatch.setattr("lucifer.inference.discover_gpus", lambda: inventory_for())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    engine = manager(tmp_path, [{"id": "multi", "backend": "vllm", "path": "base"}])
    try:
        spec = engine.specs["multi"]
        plan = engine._vllm_placement(spec)
        assert plan["device_count"] == 4
        assert plan["tensor_parallel_size"] == 4
        assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        cmd = engine.command(spec, placement=plan)
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
        pipeline = spec.model_copy(update={"pipeline_parallel_size": 2})
        assert engine._vllm_placement(pipeline)["tensor_parallel_size"] == 2
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
        masked = engine._vllm_placement(spec)
        assert masked["device_count"] == 2
        assert masked["env"]["CUDA_VISIBLE_DEVICES"] == "1,3"
        with pytest.raises(ValueError, match="must equal"):
            engine._vllm_placement(spec.model_copy(update={"tensor_parallel_size": 3}))
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_vllm_vendor_selection_and_cpu_commands(tmp_path, monkeypatch):
    mixed = {"gpus": inventory_for("cuda", 2)["gpus"] + inventory_for("rocm", 2)["gpus"]}
    monkeypatch.setattr("lucifer.inference.discover_gpus", lambda: mixed)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    engine = manager(tmp_path, [{"id": "multi", "backend": "vllm", "path": "base"}])
    try:
        spec = engine.specs["multi"]
        with pytest.raises(ValueError, match="cannot combine CUDA and ROCm"):
            engine._vllm_placement(spec)
        rocm = engine._vllm_placement(spec.model_copy(update={"gpu_backend": "rocm"}))
        assert rocm["tensor_parallel_size"] == 2
        assert rocm["env"]["HIP_VISIBLE_DEVICES"] == "0,1"
        cpu = engine._vllm_placement(spec.model_copy(update={"gpu_devices": []}))
        assert cpu["device_count"] == 0
        assert cpu["tensor_parallel_size"] == 1
        assert cpu["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "--device" not in engine.command(spec.model_copy(update={"gpu_devices": []}), placement=cpu)
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_llama_uses_backend_device_names_and_weight_proportions(tmp_path):
    engine = manager(tmp_path, [{"id": "gguf", "backend": "llama.cpp", "path": "/tmp/model.gguf"}])
    try:
        spec = engine.specs["gguf"]
        automatic = engine.command(spec)
        assert "--device" not in automatic
        assert automatic[automatic.index("--n-gpu-layers") + 1] == "-1"
        split = ModelSpec(id="split", backend="llama.cpp", path="model.gguf", gpu_devices=["CUDA0", "Vulkan0"], tensor_split=[1, 2], main_gpu=1)
        cmd = engine.command(split)
        assert cmd[cmd.index("--device") + 1] == "CUDA0,Vulkan0"
        assert cmd[cmd.index("--tensor-split") + 1] == "1.0,2.0"
        assert "--tensor-parallel-size" not in cmd
        cpu = engine.command(spec.model_copy(update={"gpu_devices": []}))
        assert cpu[cpu.index("--device") + 1] == "none"
        assert cpu[cpu.index("--n-gpu-layers") + 1] == "0"
    finally:
        await engine.close()
        engine.store.close()


def test_device_partition_validation():
    for settings in ({"gpu_devices": ["CUDA0", "CUDA0"]}, {"tensor_split": [1, 0]}, {"gpu_devices": ["CUDA0"], "tensor_split": [1, 1]}, {"gpu_devices": ["0"]}, {"gpu_devices": ["CUDA0"], "main_gpu": 1}):
        with pytest.raises(ValidationError):
            ModelSpec(id="split", backend="llama.cpp", path="model.gguf", **settings)


@pytest.mark.asyncio
async def test_adapter_rejection_exposes_backend_reason_without_activating(tmp_path):
    engine = manager(tmp_path, [{"id": "base", "backend": "external", "endpoint": "http://localhost:8900", "adapter_backend": "vllm"}])
    async def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        return httpx.Response(400, json={"error": "Adapter rank exceeds max_lora_rank"})
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await engine.load("base")
        with pytest.raises(RuntimeError, match="max_lora_rank"):
            await engine.activate_adapter({"id": "too-large", "base_model_id": "base", "path": "/adapter"})
        assert engine._runtime_adapter is None
        assert not engine._loaded_loras
    finally:
        await engine.close()
        engine.store.close()


@pytest.mark.asyncio
async def test_hot_lora_request_routing_keeps_registry_capabilities_and_bootstrap_stable(tmp_path, monkeypatch):
    from lucifer.api import create_app
    from lucifer import telemetry

    monkeypatch.setattr(telemetry, "discover_gpus", lambda: {"available": False, "gpus": []})
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml"))
    load_requests, native_routes = [], []
    async def upstream(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base", "max_model_len": 8192}]})
        if request.url.path == "/v1/status":
            return httpx.Response(200, json={"dialects": ["/v1/chat/completions"]})
        if request.url.path == "/v1/load_lora_adapter":
            load_requests.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/v1/chat/completions":
            native_routes.append(json.loads(request.content)["model"])
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
        return httpx.Response(404)

    async with app.router.lifespan_context(app):
        engine = app.state.manager
        await engine.client.aclose()
        engine.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        engine.register(ModelSpec(id="base", backend="external", endpoint="http://provider.test/v1", adapter_backend="vllm"))
        await engine.load("base")
        first = {"id": "expert-one", "base_model_id": "base", "path": "/trained/one"}
        second = {"id": "expert-two", "base_model_id": "base", "path": "/trained/two"}
        snapshot = engine.capabilities.get("base")
        provider = engine._provider_adapters["base"]["chat_completions"]
        model_revision = engine.store.revision("models")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://app.test") as client:
            bootstrap = (await client.get("/system/bootstrap")).json()
            listings = []
            original = engine.store.list
            def listed(kind, **filters):
                listings.append(kind)
                return original(kind, **filters)
            monkeypatch.setattr(engine.store, "list", listed)
            # This matches evaluation's repeated baseline/candidate alternation.
            for adapter in [None, first, None, second] * 8:
                events = [event async for event in engine.agent_events([{"role": "user", "content": "Check"}],
                    model_id="base", profile_revision=snapshot.profile_revision, adapter=adapter, thinking="off", max_tokens=16)]
                assert events[-1] == ("finish", {"finish_reason": "stop"})
                assert engine.store.revision("models") == model_revision
                assert engine.current()["adapter_id"] is None
                assert engine.capabilities.get("base") is snapshot
                assert engine._provider_adapters["base"]["chat_completions"] is provider
                assert (await client.get("/system/telemetry")).json()["revision"] == bootstrap["revision"]
            assert "models" not in listings and "adapters" not in listings
            assert native_routes == ["base", "expert-one", "base", "expert-two"] * 8
            assert [item["lora_name"] for item in load_requests] == ["expert-one", "expert-two"]
            assert engine._loaded_loras == {"expert-one", "expert-two"}

            # The last request already used this adapter; explicit selection
            # must still publish it exactly once, comparing persisted state.
            assert (await engine.activate_adapter(second))["adapter_id"] == "expert-two"
            assert engine.store.revision("models") == (model_revision[0] + 1,)
            assert (await client.get("/system/telemetry")).json()["revision"] == bootstrap["revision"] + 1
            assert (await engine.activate_adapter(second))["adapter_id"] == "expert-two"
            assert engine.store.revision("models") == (model_revision[0] + 1,)
            assert (await engine.activate_adapter(None))["adapter_id"] is None
            assert engine.store.revision("models") == (model_revision[0] + 2,)
            assert (await engine.activate_adapter(None))["adapter_id"] is None
            assert engine.store.revision("models") == (model_revision[0] + 2,)
            assert engine.capabilities.get("base") is snapshot
