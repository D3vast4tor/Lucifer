"""Requested, native model and effective server context limits stay distinct."""
import json

import httpx
import pytest
import yaml

from lucifer.config import Settings
from lucifer.inference import ModelManager, ModelSpec
from lucifer.store import Store


def engine(tmp_path, spec):
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [spec]}))
    settings = Settings(data_dir=tmp_path / "state", models_file=config)
    return ModelManager(settings, Store(settings.data_dir / "test.db"))


@pytest.mark.asyncio
async def test_auto_context_commands_use_backend_native_defaults(tmp_path):
    manager = engine(tmp_path, {"id": "custom", "backend": "vllm", "path": "Org/Model"})
    try:
        spec = manager.specs["custom"]
        assert spec.context_length == "auto"
        placement = {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}
        assert "--max-model-len" not in manager.command(spec, placement=placement)
        explicit = manager.command(spec.model_copy(update={"context_length": 100000}), placement=placement)
        assert explicit[explicit.index("--max-model-len") + 1] == "100000"
        gguf = ModelSpec(id="gguf", backend="llama.cpp", path="/tmp/custom.gguf")
        command = manager.command(gguf)
        assert command[command.index("--ctx-size") + 1] == "0"
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_local_metadata_native_limit_is_not_inferred_from_requested_limit(tmp_path):
    pack = tmp_path / "custom-pack"
    pack.mkdir()
    (pack / "ds4.txt").write_text("max_position_embeddings 1048576\nmodel_dir /unrelated\n")
    manager = engine(tmp_path, {"id": "custom", "backend": "strata", "path": str(pack),
        "runtime_command": ["unused"], "context_length": "auto"})
    try:
        record = manager.list()[0]
        assert record["native_context_length"] == 1048576
        assert record["native_context_source"] == str(pack / "ds4.txt")
        assert record["effective_context_length"] is None
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}))
        hf = ModelSpec(id="hf", backend="vllm", path=str(checkpoint), context_length=4096)
        assert manager._native_context(hf)["native_context_length"] == 262144
        assert manager._native_context(hf)["native_context_source"] == str(checkpoint / "config.json")
        unknown = ModelSpec(id="unknown", backend="external", endpoint="http://localhost:8990", context_length=4096)
        assert manager._native_context(unknown)["native_context_length"] is None
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_readiness_records_effective_and_independent_native_metadata(tmp_path):
    manager = engine(tmp_path, {"id": "custom", "backend": "external", "endpoint": "http://localhost:8990"})
    async def upstream(request):
        return httpx.Response(200, json={"data": [{"id": "custom", "meta": {"n_ctx": 8192, "n_ctx_train": 131072}}]})
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        loaded = await manager.load("custom")
        assert loaded["effective_context_length"] == 8192
        assert loaded["native_context_length"] == 131072
        assert loaded["native_context_source"] == "http://localhost:8990/v1/models"
    finally:
        await manager.close()
        manager.store.close()


@pytest.mark.asyncio
async def test_strata_status_native_label_is_only_its_effective_configured_context(tmp_path):
    manager = engine(tmp_path, {"id": "custom", "backend": "strata", "runtime_command": ["unused"]})
    async def upstream(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "custom", "meta": {"n_ctx": 8192}}]})
        return httpx.Response(200, json={"model": "custom", "loaded": True, "cache_max_tokens": 100000,
            "context": {"native": 100000, "max_positions": 100000}})
    await manager.client.aclose()
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await manager._wait_ready(manager.specs["custom"])
        assert manager._context_limits["custom"]["effective_context_length"] == 100000
        assert manager._native_context(manager.specs["custom"])["native_context_length"] is None
    finally:
        await manager.close()
        manager.store.close()
