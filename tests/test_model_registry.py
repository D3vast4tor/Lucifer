"""Runtime model descriptors, file reloads and persistent source selection."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
import yaml

from lucifer.api import create_app
from lucifer.config import Settings


def registry_settings(tmp_path, models=(), *, default_model=None):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": list(models), "default_model": default_model}))
    return Settings(data_dir=tmp_path / "state", models_file=config)


def test_arbitrary_backend_descriptors_persist_and_override_file_until_reload(tmp_path):
    settings = registry_settings(tmp_path, [{"id": "custom-dense", "backend": "vllm", "path": "Example/Original"}])
    dense = {"id": "custom-dense", "backend": "vllm", "path": "Example/Replacement", "gpu_devices": []}
    quantized = {"id": "another-quant", "backend": "llama.cpp", "path": str(tmp_path / "other.gguf"), "gpu_devices": []}
    for attempt in range(2):
        app = create_app(settings, autoload="")
        with TestClient(app) as client:
            if not attempt:
                for descriptor in (dense, quantized):
                    response = client.post("/models/register", json=descriptor)
                    assert response.status_code == 200, response.text
                    assert response.json()["registry_source"] == "runtime"
            manager = app.state.manager
            assert manager.specs["custom-dense"].path == "Example/Replacement"
            vllm = manager.command(manager.specs["custom-dense"], placement={"tensor_parallel_size": 1, "pipeline_parallel_size": 1})
            gguf = manager.command(manager.specs["another-quant"])
            assert vllm[:3] == ["vllm", "serve", "Example/Replacement"]
            assert gguf[gguf.index("--model") + 1] == str(tmp_path / "other.gguf")
            assert gguf[gguf.index("--device") + 1] == "none"
            if attempt:
                response = client.post("/models/reload")
                assert response.status_code == 200, response.text
                assert manager.specs["custom-dense"].path == "Example/Original"
                # File reload is an upsert; unrelated runtime registrations remain.
                assert manager.specs["another-quant"].backend == "llama.cpp"


def test_load_accepts_explicit_profile_without_hardcoded_model_names(tmp_path):
    settings = registry_settings(tmp_path)
    with TestClient(create_app(settings, autoload="")) as client:
        response = client.post("/models/load", json={"spec": {"id": "provided-model", "backend": "demo"}})
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "provided-model"
        assert response.json()["display_name"] == "Lucifer"
        chat = client.post("/chat", json={"model": "provided-model", "messages": [{"role": "user", "content": "What is 2+2?"}]})
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["message"]["content"] == "4"
        assert client.post("/models/load", json={"model_id": "missing"}).status_code == 404
        assert client.post("/models/load", json={}).status_code == 422
        assert client.post("/models/load", json={"model_id": "different", "spec": {"id": "x", "backend": "demo"}}).status_code == 422
        invalid = client.post("/models/register", json={"id": "invalid", "backend": "guessed-backend", "path": "whatever.gguf"})
        assert invalid.status_code == 422
        assert client.get("/models/current").json()["id"] == "provided-model"


@pytest.mark.asyncio
async def test_reload_uses_new_external_source_before_loading(tmp_path):
    descriptor = {"id": "provided-endpoint", "backend": "external", "endpoint": "http://old.example:8910"}
    settings = registry_settings(tmp_path, [descriptor])
    app = create_app(settings, autoload="")
    requests = []
    async def upstream(request):
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "provided-endpoint"}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"new source"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    async with app.router.lifespan_context(app):
        await app.state.manager.client.aclose()
        app.state.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            descriptor["endpoint"] = "http://new.example:8920"
            settings.models_file.write_text(yaml.safe_dump({"models": [descriptor]}))
            assert (await client.post("/models/reload")).status_code == 200
            loaded = await client.post("/models/load", json={"model_id": "provided-endpoint"})
            assert loaded.status_code == 200, loaded.text
            reply = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
            assert reply.status_code == 200, reply.text
            assert reply.json()["choices"][0]["message"]["content"] == "new source"
            assert {str(request.url) for request in requests} == {"http://new.example:8920/v1/models", "http://new.example:8920/v1/status", "http://new.example:8920/openapi.json", "http://new.example:8920/v1/chat/completions"}
            assert json.loads(requests[-1].content)["model"] == "provided-endpoint"


def test_reload_is_validated_before_mutation_and_never_reconfigures_loaded_model(tmp_path):
    original = {"id": "running", "backend": "demo", "description": "original"}
    settings = registry_settings(tmp_path, [original])
    app = create_app(settings, autoload="running")
    with TestClient(app) as client:
        before = client.get("/models/current").json()
        unchanged = client.post("/models/reload")
        assert unchanged.status_code == 200
        assert client.get("/models/current").json()["status"] == "loaded"
        changed = {**original, "description": "changed"}
        settings.models_file.write_text(yaml.safe_dump({"models": [{"id": "other", "backend": "demo"}, changed]}))
        rejected = client.post("/models/reload")
        assert rejected.status_code == 409
        assert "Unload model 'running'" in rejected.json()["detail"]
        assert app.state.manager.specs["running"].description == "original"
        assert "other" not in app.state.manager.specs
        assert client.post("/models/register", json=changed).status_code == 409
        assert client.get("/models/current").json()["pid"] == before["pid"]
        client.post("/models/unload", json={})
        assert client.post("/models/reload").status_code == 200
        assert app.state.manager.specs["running"].description == "changed"
        settings.models_file.write_text(yaml.safe_dump({"models": [original, original]}))
        duplicate = client.post("/models/reload")
        assert duplicate.status_code == 400
        assert "duplicate model ids" in duplicate.json()["detail"]
        assert app.state.manager.specs["running"].description == "changed"
        settings.models_file.write_text("models: [broken")
        invalid = client.post("/models/reload")
        assert invalid.status_code == 400
        assert "Invalid model configuration YAML" in invalid.json()["detail"]


def test_reload_updates_next_start_default_without_hot_switch_and_respects_environment(tmp_path, monkeypatch):
    models = [{"id": "first", "backend": "demo"}, {"id": "second", "backend": "demo"}]
    monkeypatch.delenv("LUCIFER_DEFAULT_MODEL", raising=False)
    settings = registry_settings(tmp_path, models, default_model="first")
    with TestClient(create_app(settings)) as client:
        settings.models_file.write_text(yaml.safe_dump({"models": models, "default_model": "second"}))
        response = client.post("/models/reload")
        assert response.json()["default_model"] == "second"
        assert client.get("/models/current").json()["id"] == "first"
        assert client.get("/system/status").json()["default_model"] == "second"
        monkeypatch.setenv("LUCIFER_DEFAULT_MODEL", "first")
        assert client.post("/models/reload").json()["default_model"] == "first"


def test_model_mutations_keep_api_authentication(tmp_path):
    settings = registry_settings(tmp_path)
    settings.api_token = "local-secret"
    with TestClient(create_app(settings, autoload="")) as client:
        assert client.post("/models/register", json={"id": "x", "backend": "demo"}).status_code == 401
        assert client.post("/models/reload").status_code == 401
        assert client.post("/models/load", json={"spec": {"id": "x", "backend": "demo"}}).status_code == 401


def test_launcher_can_refresh_file_snapshot_without_creating_runtime_override(tmp_path):
    profile = {"id": "file-model", "backend": "demo", "description": "before"}
    settings = registry_settings(tmp_path, [profile])
    with TestClient(create_app(settings, autoload="")) as client:
        updated = {**profile, "description": "after"}
        settings.models_file.write_text(yaml.safe_dump({"models": [updated]}))
        response = client.post("/models/load", json={"spec": updated},
                               headers={"X-Lucifer-Profile-Source": "file"})
        assert response.status_code == 200, response.text
        assert response.json()["registry_source"] == "file"
        assert response.json()["description"] == "after"
        assert client.post("/models/unload", json={}).status_code == 200
        # A profile edited after the launcher read it must fail before effects.
        stale = client.post("/models/load", json={"spec": profile},
                            headers={"X-Lucifer-Profile-Source": "file"})
        assert stale.status_code == 400
        assert "changed on disk" in stale.json()["detail"]
        assert client.get("/models/current").json() is None
        assert client.get("/models").json()[0]["description"] == "after"


def test_explicit_inline_model_still_uses_runtime_source(tmp_path):
    profile = {"id": "file-model", "backend": "demo", "description": "file"}
    settings = registry_settings(tmp_path, [profile])
    with TestClient(create_app(settings, autoload="")) as client:
        response = client.post("/models/load", json={"spec": {**profile, "description": "custom"}})
        assert response.status_code == 200, response.text
        assert response.json()["registry_source"] == "runtime"
        assert response.json()["description"] == "custom"


def test_registry_local_paths_follow_configuration_directory_after_move(tmp_path, monkeypatch):
    from lucifer.inference import ModelSpec

    project = tmp_path / "moved-project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    metadata = project / "weights/config.json"
    metadata.parent.mkdir()
    metadata.write_text('{"max_position_embeddings": 8192}')
    config = config_dir / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [
        {"id": "managed", "backend": "strata", "path": "../pack",
         "runtime_command": ["../bin/server", "--profile", "./profile.json"]},
        {"id": "training", "backend": "vllm", "path": "../weights",
         "training_model": "../weights", "executable": "../bin/vllm",
         "context_metadata_file": "../weights/config.json"},
        {"id": "hub", "backend": "vllm", "path": "Example/Checkpoint"},
    ]}))
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=tmp_path / "state", models_file=config)
    with TestClient(create_app(settings, autoload="")) as client:
        records = {row["id"]: row for row in client.get("/models").json()}
        assert records["managed"]["runtime_command"] == [
            str(project / "bin/server"), "--profile", "./profile.json"]
        assert records["managed"]["path"] == str(project / "pack")
        assert records["training"]["training_model"] == str(metadata.parent)
        assert records["training"]["executable"] == str(project / "bin/vllm")
        assert records["training"]["native_context_length"] == 8192
        assert records["hub"]["path"] == "Example/Checkpoint"
        assert records["hub"]["executable"] is None
        before = records["managed"]["spec"]
        assert client.post("/models/reload").json()["updated"] == []
        assert client.get("/models").json()[0]["spec"] == before

    spec = ModelSpec(id="local", backend="vllm", path="~/model", executable="vllm")
    resolved = spec.resolve_local_paths(config_dir)
    assert resolved.path == str(Path("~/model").expanduser().resolve())
    assert resolved.executable == "vllm"
    assert spec.path == "~/model"
