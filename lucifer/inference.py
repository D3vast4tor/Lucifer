"""Owned inference processes and OpenAI-compatible streaming clients.

No shell commands are evaluated. Managed engines bind only to loopback; external
engines are never terminated by this manager. A request holds the engine lock so
switching models or restarting a GGUF adapter cannot interrupt another request.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
from typing import Annotated, AsyncIterator, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from .config import Settings
from .capabilities import CapabilityRegistry
from .store import Store
from .gpu import discover_gpus, training_placement
from .providers import ProviderAdapter, ProviderError, conservative_count, prepare_adapters


ThinkingMode = Literal["off", "low", "medium", "high", "max"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    display_name: str = Field(default="Lucifer", min_length=1, max_length=120)
    backend: Literal["demo", "vllm", "llama.cpp", "strata", "external"]
    path: str = ""
    served_model_name: str | None = None
    endpoint: str | None = None
    port: int = Field(default=8081, ge=1024, le=65535)
    context_length: Annotated[int, Field(ge=128, le=1048576)] | Literal["auto"] = "auto"
    native_context_length: int | None = Field(default=None, ge=128)
    native_context_source: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1,
        description="Optional profile output-budget ceiling; native limits and active context still apply")
    context_metadata_file: str | None = None
    n_gpu_layers: int = Field(default=-1, ge=-1)
    gpu_devices: list[str] | Literal["all"] = "all"
    gpu_backend: Literal["auto", "cuda", "rocm"] = "auto"
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    split_mode: Literal["none", "layer", "row", "tensor"] = "layer"
    tensor_split: list[float] | None = None
    main_gpu: int = Field(default=0, ge=0)
    gpu_memory_utilization: float = Field(default=0.8, gt=0, le=1)
    enable_lora: bool = True
    max_loras: int = Field(default=4, ge=1)
    max_lora_rank: int = Field(default=64, ge=1)
    quantization: str | None = None
    executable: str | None = None
    runtime_command: list[str] | None = Field(default=None, description="Foreground Strata server argv; executed directly without a shell")
    startup_timeout: float | None = Field(default=None, gt=0, le=86400)
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict, description="Nonsecret process environment overrides; inherit secret variables instead")
    api_key_env: str | None = None
    adapter_backend: Literal["vllm"] | None = None
    training_model: str | None = None
    base_revision: str | None = Field(default=None, min_length=1, max_length=128,
        description="Immutable native base weight revision; adapters require this exact revision")
    parent_model_id: str | None = None
    chat_template_kwargs: dict[str, object] = Field(default_factory=dict)
    thinking_mode: ThinkingMode | None = None
    reasoning_effort_map: dict[Literal["low", "medium", "high", "max"], str] = Field(default_factory=dict,
        description="Optional mapping from thinking modes to labels accepted by this model's template")
    capabilities: dict[str, object] = Field(default_factory=dict,
        description="Evidence-backed profile declarations; advertised support is distinct from verified behavior")
    description: str = ""

    @model_validator(mode="after")
    def validate_backend(self):
        if any(not value.strip() for value in self.reasoning_effort_map.values()):
            raise ValueError("reasoning_effort_map labels must be nonempty")
        if self.backend == "external" and not self.endpoint:
            raise ValueError("external models require an endpoint")
        if self.backend in ("vllm", "llama.cpp") and not self.path:
            raise ValueError(f"{self.backend} models require a model path")
        if self.backend == "strata":
            if not self.runtime_command or not self.runtime_command[0].strip() or any("\0" in arg for arg in self.runtime_command):
                raise ValueError("strata models require a nonempty runtime_command argv without NUL characters")
            if self.extra_args or self.executable:
                raise ValueError("strata uses runtime_command instead of executable or extra_args")
            if self.adapter_backend:
                raise ValueError("strata is inference-only and does not support adapter backends")
        elif self.runtime_command is not None:
            raise ValueError("runtime_command is only supported for managed strata models; use external for an existing server")
        if self.endpoint:
            parsed = urlsplit(self.endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("endpoint must be an HTTP(S) base URL without credentials, query or fragment")
            self.endpoint = self.endpoint.rstrip("/")
            if self.backend == "strata" and (parsed.hostname not in ("127.0.0.1", "localhost") or (parsed.port or (443 if parsed.scheme == "https" else 80)) != self.port):
                raise ValueError("managed strata endpoint must use loopback and match its configured port")
        if isinstance(self.gpu_devices, list):
            if any(not d.strip() for d in self.gpu_devices) or len(set(self.gpu_devices)) != len(self.gpu_devices):
                raise ValueError("gpu_devices must contain unique nonempty device identifiers")
            if self.backend == "llama.cpp":
                if any(d.isdigit() for d in self.gpu_devices):
                    raise ValueError("llama.cpp gpu_devices uses backend names such as CUDA0 or Vulkan0; inspect llama-server --list-devices")
                if self.gpu_devices and self.main_gpu >= len(self.gpu_devices):
                    raise ValueError("main_gpu must index one of the selected llama.cpp devices")
        if self.tensor_split is not None:
            if not self.tensor_split or any(not math.isfinite(v) or v <= 0 for v in self.tensor_split):
                raise ValueError("tensor_split requires positive finite proportions")
            if self.backend != "llama.cpp":
                raise ValueError("tensor_split is a llama.cpp setting; vLLM uses tensor_parallel_size")
            if isinstance(self.gpu_devices, list) and len(self.tensor_split) != len(self.gpu_devices):
                raise ValueError("tensor_split must have one proportion per selected llama.cpp device")
            if self.split_mode == "none":
                raise ValueError("tensor_split requires a split mode that uses multiple devices")
        if self.backend == "llama.cpp" and (self.tensor_parallel_size is not None or self.pipeline_parallel_size != 1):
            raise ValueError("llama.cpp uses split_mode/tensor_split; tensor and pipeline parallel sizes are vLLM settings")
        # Host/port flags must not override the loopback binding of managed engines.
        reserved = ("--host", "--port", "--api-key", "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp", "--device", "--tensor-split", "--split-mode", "--main-gpu")
        if any(a in reserved or any(a.startswith(r + "=") for r in reserved) for a in self.extra_args):
            raise ValueError("extra_args cannot override host, port or API key")
        return self

    @property
    def managed(self) -> bool:
        return self.backend == "strata" or (self.backend in ("vllm", "llama.cpp") and not self.endpoint)

    @property
    def base_url(self) -> str:
        endpoint = self.endpoint or f"http://127.0.0.1:{self.port}"
        return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"

    @property
    def model_name(self) -> str:
        return self.served_model_name or self.id

    def resolve_local_paths(self, directory: Path) -> ModelSpec:
        """Resolve explicit local paths against their registry, preserving Hub ids."""
        def local_path(value):
            if value and value.startswith(("./", "../", "~")):
                candidate = Path(value).expanduser()
                return str((directory / candidate).resolve())
            return value
        updates = {name: local_path(getattr(self, name))
                   for name in ("path", "training_model", "executable", "context_metadata_file")}
        if self.runtime_command:
            # Remaining argv entries can be literal application data, including
            # prompts beginning with ./ or ~. Only argv[0] is always a path.
            updates["runtime_command"] = [local_path(self.runtime_command[0]), *self.runtime_command[1:]]
        return self.model_copy(update=updates)

    def effective_thinking(self, requested: ThinkingMode | None = None) -> ThinkingMode | None:
        if requested is not None:
            if requested not in ("off", "low", "medium", "high", "max"):
                raise ValueError(f"Unsupported thinking mode: {requested}")
            return requested
        if self.thinking_mode is not None:
            return self.thinking_mode
        if self.chat_template_kwargs.get("enable_thinking") is True:
            label = self.chat_template_kwargs.get("reasoning_effort")
            for mode, mapped in self.reasoning_effort_map.items():
                if label == mapped:
                    return mode
            return label if label in ("low", "medium", "high", "max") else None
        return "off"


class ModelManager:
    def __init__(self, settings: Settings, store: Store, *, gpu_inventory: dict | None = None):
        self.settings, self.store = settings, store
        self.gpu_inventory = gpu_inventory
        self.capabilities = CapabilityRegistry()
        self._adapter_source_cache = {}
        self._adapter_source_revision = None
        self._provider_adapters: dict[str, dict[str, ProviderAdapter]] = {}
        self.specs: dict[str, ModelSpec] = {}
        self._registry_sources: dict[str, str] = {}
        self._context_limits: dict[str, dict] = {}
        self._native_context_cache: dict[str, tuple[ModelSpec, dict]] = {}
        self._current: str | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._process_stop_timeout = 5.0
        self._log = None
        self._lock = asyncio.Lock()
        self._runtime_adapter: str | None = None
        self.native_base_validator = None
        self._loaded_loras: set[str] = set()
        self._gpu_assignments: dict[str, dict] = {}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout, connect=10.0), trust_env=False)
        # Persistent model records are the registry; transient serving states never survive restart.
        for record in store.list("models"):
            payload = record.get("spec", {k: v for k, v in record.items() if k in ModelSpec.model_fields})
            spec = ModelSpec.model_validate(payload)
            self.specs[spec.id] = spec
            self._registry_sources[spec.id] = record.get("registry_source", "persisted")
        for spec in self._read_models_file()[0]:
            # Runtime registrations are explicit persistent overrides. Reloading
            # the file below is the explicit way to apply its values again.
            if self._registry_sources.get(spec.id) != "runtime":
                self.specs[spec.id] = spec
                self._registry_sources[spec.id] = "file"
        if not self.specs:
            self.specs["demo"] = ModelSpec(id="demo", backend="demo", description="Deterministic demonstration; no LLM weights")
            self._registry_sources["demo"] = "builtin"
        for spec in self.specs.values():
            self._save(spec, "ready")

    def register(self, spec: ModelSpec, *, source: str = "runtime") -> dict:
        if source == "file":
            configured = next((model for model in self._read_models_file(required=True)[0]
                               if model.id == spec.id), None)
            if configured != spec:
                raise ValueError("The model profile changed on disk; launch again to read its current configuration")
        elif source != "runtime":
            raise ValueError("Unsupported model registry source")
        record = self._register(spec, source)
        self.store.audit("model.register", model_id=spec.id, backend=spec.backend)
        return record

    def _register(self, spec: ModelSpec, source: str) -> dict:
        self.current()
        if spec.id == self._current and self.specs[spec.id] != spec:
            raise RuntimeError(f"Unload model '{spec.id}' before changing its configuration")
        unchanged = self.specs.get(spec.id) == spec
        self.specs[spec.id] = spec
        self._registry_sources[spec.id] = source
        if unchanged:
            return self.store.update("models", spec.id, registry_source=source,
                                     **self._native_context(spec, refresh=True))
        self._gpu_assignments.pop(spec.id, None)
        self._context_limits.pop(spec.id, None)
        self._native_context_cache.pop(spec.id, None)
        self.capabilities.invalidate(spec.id)
        self._provider_adapters.pop(spec.id, None)
        return self._save(spec, "ready")

    def _read_models_file(self, *, required: bool = False) -> tuple[list[ModelSpec], str | None]:
        path = self.settings.models_file
        if not path.exists():
            if required:
                raise ValueError(f"Model configuration file does not exist: {path}")
            return [], None
        try:
            payload = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid model configuration YAML: {path}: {exc}") from exc
        if payload is None:
            payload = {}
        if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
            raise ValueError("Model configuration must be a mapping with a models list")
        specs = [ModelSpec.model_validate(record).resolve_local_paths(path.parent)
                 for record in payload.get("models", [])]
        if len({spec.id for spec in specs}) != len(specs):
            raise ValueError("Model configuration contains duplicate model ids")
        default_model = payload.get("default_model")
        if default_model is not None and (not isinstance(default_model, str) or not default_model.strip()):
            raise ValueError("default_model must be a nonempty model id")
        return specs, default_model

    async def reload(self) -> dict:
        """Upsert file profiles without silently replacing a running engine."""
        async with self._lock:
            specs, file_default = self._read_models_file(required=True)
            self.current()
            for spec in specs:
                if spec.id == self._current and self.specs[spec.id] != spec:
                    raise RuntimeError(f"Unload model '{spec.id}' before reloading its changed configuration")
            updated = [spec.id for spec in specs if self.specs.get(spec.id) != spec]
            for spec in specs:
                self._register(spec, "file")
            # Refresh the next-start preference without switching a live model.
            self.settings.default_model = os.environ.get("LUCIFER_DEFAULT_MODEL", file_default)
            self.store.audit("models.reload", models_file=str(self.settings.models_file), updated=updated)
            return {"status": "reloaded", "models_file": str(self.settings.models_file),
                    "updated": updated, "model_ids": [spec.id for spec in specs],
                    "default_model": self.settings.default_model}

    def _save(self, spec: ModelSpec, status: str, **details) -> dict:
        capabilities = self.capabilities.get(spec.id)
        limits = self._context_limits.get(spec.id, {})
        capacity = self._positive_context(limits.get("effective_context_length")) or self._positive_context(spec.context_length)
        choices = [(value, source) for value, source in (
            (self._positive_context(limits.get("native_max_output_tokens")), "runtime output metadata"),
            (spec.max_output_tokens, "profile output budget"),
            (max(1, capacity - 256) if capacity else None, "active context allowance")) if value is not None]
        output_limit, output_source = min(choices, default=(None, None), key=lambda item: item[0])
        return self.store.put("models", {**spec.model_dump(), "spec": spec.model_dump(), "status": status,
            "registry_source": self._registry_sources.get(spec.id, "persisted"),
            **self._native_context(spec), "effective_context_length": None,
            **limits, "max_output_tokens": output_limit, "output_limit_source": output_source,
            "default_thinking_mode": spec.effective_thinking(),
            "capabilities": capabilities.as_dict() if capabilities is not None else None,
            "log_path": str(self.settings.data_dir / "logs" / f"model-{spec.id}.log") if spec.managed else None,
            "gpu_assignment": self._gpu_assignments.get(spec.id), **details})

    @staticmethod
    def _positive_context(value) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    def _native_context(self, spec: ModelSpec, *, refresh: bool = False) -> dict:
        cached = self._native_context_cache.get(spec.id)
        if not refresh and cached is not None and cached[0] is spec:
            return dict(cached[1])
        native, source = spec.native_context_length, spec.native_context_source
        if native is None and (spec.path or spec.context_metadata_file):
            path = Path(spec.context_metadata_file or spec.path).expanduser()
            if not spec.context_metadata_file:
                path = path / ("ds4.txt" if spec.backend == "strata" else "config.json")
            try:
                if path.name == "ds4.txt":
                    values = dict(line.split(None, 1) for line in path.read_text().splitlines() if len(line.split(None, 1)) == 2)
                    native = self._positive_context(int(values.get("max_position_embeddings", "0")))
                else:
                    config = json.loads(path.read_text())
                    text_config = config.get("text_config") or config
                    native = self._positive_context(text_config.get("max_position_embeddings"))
                if native:
                    source = str(path)
            except (OSError, ValueError, TypeError, AttributeError):
                native = None
        result = {"native_context_length": native, "native_context_source": source if native else None}
        self._native_context_cache[spec.id] = (spec, result)
        return dict(result)

    def list(self) -> list[dict]:
        self.current()  # Detect an exited managed process.
        return [self.store.get("models", ident) for ident in self.specs]

    def check_process(self) -> None:
        """Observe owned process death without reading SQLite on a heartbeat."""
        if self._current and self._process and self._process.returncode is not None:
            self._save(self.specs[self._current], "error", error=f"Model process exited with code {self._process.returncode}")
            self._current = None

    def current(self) -> dict | None:
        self.check_process()
        return self.store.get("models", self._current) if self._current else None

    def _headers(self, spec: ModelSpec) -> dict:
        key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _vllm_placement(self, spec: ModelSpec) -> dict:
        if self.gpu_inventory is None:
            self.gpu_inventory = discover_gpus()
        inventory = self.gpu_inventory
        if spec.gpu_backend != "auto":
            inventory = {**inventory, "gpus": [g for g in inventory.get("gpus", []) if g["backend"] == spec.gpu_backend]}
            if not inventory["gpus"] and spec.gpu_devices != []:
                raise ValueError(f"No {spec.gpu_backend} GPU is available; install its runtime or select another backend")
        plan = training_placement(spec.gpu_devices, inventory=inventory, environ={**os.environ, **spec.env})
        count = plan["device_count"]
        workers = max(1, count)
        pp = spec.pipeline_parallel_size
        tp = spec.tensor_parallel_size or max(1, workers // pp)
        if tp * pp != workers:
            raise ValueError(f"tensor_parallel_size ({tp}) × pipeline_parallel_size ({pp}) must equal the number of selected GPUs ({count}), or 1 for CPU. Adjust gpu_devices or parallel sizes.")
        return {**plan, "tensor_parallel_size": tp, "pipeline_parallel_size": pp}

    def command(self, spec: ModelSpec, adapter: dict | None = None, *, placement: dict | None = None) -> list[str]:
        if spec.backend == "vllm":
            placement = self._vllm_placement(spec) if placement is None else placement
            args = [spec.executable or "vllm", "serve", spec.path, "--host", "127.0.0.1", "--port", str(spec.port),
                    "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
                    "--served-model-name", spec.model_name, "--tensor-parallel-size", str(placement["tensor_parallel_size"]),
                    "--pipeline-parallel-size", str(placement["pipeline_parallel_size"])]
            if spec.context_length != "auto":
                args += ["--max-model-len", str(spec.context_length)]
            if spec.enable_lora:
                args += ["--enable-lora", "--max-loras", str(spec.max_loras), "--max-lora-rank", str(spec.max_lora_rank)]
            if spec.quantization:
                args += ["--quantization", spec.quantization]
        elif spec.backend == "llama.cpp":
            args = [spec.executable or "llama-server", "--model", str(Path(spec.path).expanduser()), "--host", "127.0.0.1",
                    "--port", str(spec.port), "--ctx-size", str(0 if spec.context_length == "auto" else spec.context_length), "--n-gpu-layers", str(0 if spec.gpu_devices == [] else spec.n_gpu_layers),
                    "--alias", spec.model_name, "--split-mode", spec.split_mode, "--main-gpu", str(spec.main_gpu)]
            if isinstance(spec.gpu_devices, list):
                args += ["--device", ",".join(spec.gpu_devices) if spec.gpu_devices else "none"]
            if spec.tensor_split:
                args += ["--tensor-split", ",".join(str(v) for v in spec.tensor_split)]
            if adapter:
                args += ["--lora", str(Path(adapter["path"]).expanduser())]
        elif spec.backend == "strata":
            if adapter:
                raise ValueError("Strata models are inference-only and do not support LoRA adapters")
            return list(spec.runtime_command or [])
        else:
            raise ValueError("Only managed vLLM, llama.cpp and Strata models have commands")
        return args + spec.extra_args

    async def load(self, model_id: str) -> dict:
        async with self._lock:
            if model_id not in self.specs:
                raise KeyError(f"Unknown model: {model_id}")
            if self.native_base_validator is not None:
                self.native_base_validator(self.specs[model_id])
            if self.current() and self._current == model_id:
                return self.current()
            spec = self.specs[model_id]
            if spec.api_key_env and not os.environ.get(spec.api_key_env):
                raise ValueError(f"Required API key environment variable is unset: {spec.api_key_env}")
            await self._unload()
            return await self._load(self.specs[model_id])

    async def _load(self, spec: ModelSpec, adapter: dict | None = None) -> dict:
        self._native_context(spec, refresh=True)
        self.capabilities.invalidate(spec.id)
        self._provider_adapters.pop(spec.id, None)
        self._save(spec, "loading")
        try:
            if spec.managed:
                placement = await asyncio.to_thread(self._vllm_placement, spec) if spec.backend == "vllm" else None
                if placement:
                    self._gpu_assignments[spec.id] = {k: v for k, v in placement.items() if k != "env"}
                if spec.backend == "llama.cpp" and not Path(spec.path).expanduser().is_file():
                    raise ValueError(f"GGUF model does not exist: {spec.path}")
                # Refuse a conflicting port rather than attaching to a different server by accident.
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(("127.0.0.1", spec.port))
                    except OSError as exc:
                        raise RuntimeError(f"Cannot start model '{spec.id}': loopback port {spec.port} is unavailable ({exc.strerror})") from exc
                log_dir = self.settings.data_dir / "logs"
                log_dir.mkdir(exist_ok=True)
                self._log = (log_dir / f"model-{spec.id}.log").open("ab")
                env = {**os.environ, **spec.env, **(placement["env"] if placement else {})}
                if spec.backend == "vllm" and spec.enable_lora:
                    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
                # Strata drains requests and tears down engine workers in their
                # own groups. Give its foreground supervisor time to do so.
                self._process_stop_timeout = 75.0 if spec.backend == "strata" else 5.0
                self._process = await asyncio.create_subprocess_exec(*self.command(spec, adapter, placement=placement), stdout=self._log, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True)
            if spec.backend != "demo":
                await self._wait_ready(spec)
            else:
                revision = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()
                self.capabilities.record(spec.id, revision, declared={"chat_completions": True,
                    "streaming": True, "tools": False, "reasoning": False},
                    native_context_length=self._native_context(spec)["native_context_length"],
                    effective_context_length=spec.context_length if isinstance(spec.context_length, int) else 8192)
            self._provider_adapters[spec.id] = prepare_adapters(self.capabilities.get(spec.id))
            self._current = spec.id
            self._runtime_adapter = adapter["id"] if adapter else None
            self._loaded_loras.clear()
            self.store.audit("model.load", model_id=spec.id, managed=spec.managed)
            return self._save(spec, "loaded", pid=self._process.pid if self._process else None, adapter_id=self._runtime_adapter)
        except BaseException as exc:
            await self.capabilities.cancel_model(spec.id)
            self._provider_adapters.pop(spec.id, None)
            await self._stop_process()
            self._current = None
            self._save(spec, "error", error=str(exc) or type(exc).__name__)
            raise

    async def _wait_ready(self, spec: ModelSpec):
        last_error = "not ready"
        startup_timeout = spec.startup_timeout if spec.startup_timeout is not None else self.settings.startup_timeout
        async def wait():
            nonlocal last_error
            while True:
                if self._process and self._process.returncode is not None:
                    log_path = self.settings.data_dir / "logs" / f"model-{spec.id}.log"
                    excerpt = log_path.read_text(errors="replace")[-8192:] if log_path.exists() else ""
                    raise RuntimeError(f"Inference server exited ({self._process.returncode}). Check GPU partition compatibility and engine support. Full log: {log_path}\n{excerpt}")
                try:
                    response = await self.client.get(spec.base_url + "/models", headers=self._headers(spec), timeout=5)
                    response.raise_for_status()
                    available = response.json().get("data", [])
                    if not isinstance(available, list) or any(not isinstance(item, dict) for item in available):
                        raise ValueError("Server returned an invalid model list")
                    if not any(item.get("id") == spec.model_name for item in available):
                        raise ValueError(f"Server does not advertise model '{spec.model_name}'")
                    card = next(item for item in available if item.get("id") == spec.model_name)
                    meta = card.get("meta") if isinstance(card.get("meta"), dict) else {}
                    effective = next((n for value in (card.get("max_model_len"), meta.get("n_ctx"), meta.get("context_length"))
                        if (n := self._positive_context(value)) is not None), None)
                    native = next((n for value in (card.get("native_context_length"), meta.get("n_ctx_train"), meta.get("native_context_length"))
                        if (n := self._positive_context(value)) is not None), None)
                    if spec.backend == "strata":
                        # Strata can advertise an unloaded model for auto-load.
                        # Its status route distinguishes registry visibility
                        # from weights that are actually ready for inference.
                        status_response = await self.client.get(spec.base_url + "/status", headers=self._headers(spec), timeout=5)
                        status_response.raise_for_status()
                        status = status_response.json()
                        if not isinstance(status, dict) or status.get("model") != spec.model_name or status.get("loaded") is not True:
                            raise ValueError(f"Strata model '{spec.model_name}' is not loaded")
                        context = status.get("context") if isinstance(status.get("context"), dict) else {}
                        effective = self._positive_context(status.get("cache_max_tokens")) or self._positive_context(context.get("native")) or effective
                    output = next((n for value in (card.get("max_output_tokens"), card.get("max_completion_tokens"),
                        meta.get("max_output_tokens"), status.get("max_output_tokens") if spec.backend == "strata" else None)
                        if (n := self._positive_context(value)) is not None), None)
                    limits = {"effective_context_length": effective, "native_max_output_tokens": output}
                    if native and self._native_context(spec)["native_context_length"] is None:
                        limits.update(native_context_length=native, native_context_source=spec.base_url + "/models")
                    self._context_limits[spec.id] = limits
                    revision = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()
                    await self.capabilities.discover(spec.id, revision, client=self.client,
                        base_url=spec.base_url, headers=self._headers(spec), models={"data": available},
                        status=status if spec.backend == "strata" else None,
                        declared=spec.capabilities,
                        # OpenAI-compatible servers such as vLLM advertise
                        # routes in origin OpenAPI rather than /v1/status.
                        # Read it once during this initialization, never turns.
                        include_openapi=spec.backend != "strata",
                        served_model_name=spec.model_name,
                        native_context_length=self._native_context(spec)["native_context_length"] or native,
                        effective_context_length=effective)
                    return
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    last_error = str(exc)
                await asyncio.sleep(0.25)
        try:
            await asyncio.wait_for(wait(), timeout=startup_timeout)
        except TimeoutError as exc:
            raise RuntimeError(f"Model startup timed out after {startup_timeout:g}s: {spec.id}; {last_error}") from exc

    async def _stop_process(self):
        process, self._process = self._process, None
        if process:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=self._process_stop_timeout)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            # A runtime supervisor may exit before its children. Reap the leader
            # above, then terminate any remaining members of our owned session.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        if self._log:
            self._log.close()
            self._log = None

    async def _unload(self):
        previous = self._current
        self._current = None
        await self._stop_process()
        self._runtime_adapter = None
        self._loaded_loras.clear()
        if previous:
            self._save(self.specs[previous], "unloaded")
            self.store.audit("model.unload", model_id=previous)

    async def unload(self, model_id: str | None = None):
        async with self._lock:
            if model_id and model_id not in self.specs:
                raise KeyError(f"Unknown model: {model_id}")
            if model_id is None or model_id == self._current:
                await self._unload()
            return {"status": "unloaded", "model_id": model_id}

    def _validate_adapter(self, spec: ModelSpec, adapter: dict | None):
        if not adapter:
            return
        registered = None
        if adapter.get("id"):
            revision = self.store.revision("adapters")
            if revision != self._adapter_source_revision:
                self._adapter_source_cache.clear()
                self._adapter_source_revision = revision
            ident = adapter["id"]
            if ident not in self._adapter_source_cache:
                try:
                    self._adapter_source_cache[ident] = self.store.get("adapters", ident)
                except KeyError:
                    self._adapter_source_cache[ident] = None
            registered = self._adapter_source_cache[ident]
        if any(item.get("source_revoked") or item.get("source_invalidated") for item in (adapter, registered or {})):
            raise ValueError("Adapter source evidence was revoked or superseded; select a current verified candidate")
        if spec.backend == "strata":
            raise ValueError("Strata models are inference-only and do not support LoRA adapters")
        from .model_compatibility import adapter_compatibility
        compatibility = adapter_compatibility(spec, adapter)
        if not compatibility["compatible"]:
            raise ValueError(compatibility["reason"])
        base = adapter.get("base_model_id") or adapter.get("base_model")
        if base and base not in (spec.id, spec.path, spec.training_model):
            raise ValueError("Adapter belongs to a different base model")
        if spec.backend == "demo":
            if adapter.get("format") != "demo" and adapter.get("backend") != "demo":
                raise ValueError("Real adapters cannot be served by the demonstration backend")
        elif spec.backend == "llama.cpp":
            if not str(adapter.get("path", "")).lower().endswith(".gguf"):
                raise ValueError("llama.cpp requires a GGUF LoRA adapter; export PEFT adapters before serving")
            if not spec.managed:
                raise ValueError("Adapter switching for an external llama.cpp server must be managed by that server")
            if not Path(adapter["path"]).expanduser().is_file():
                raise ValueError("GGUF adapter file does not exist")
        elif spec.backend != "vllm" and spec.adapter_backend != "vllm":
            raise ValueError("External endpoint does not declare a supported adapter backend")
        elif not spec.enable_lora:
            raise ValueError("LoRA is disabled for this model")

    async def _select_adapter(self, spec: ModelSpec, adapter: dict | None) -> str:
        """Route one request without changing the public adapter selection.

        vLLM chooses a loaded LoRA by the request's native model name. Only an
        explicit activation persists that selection; llama.cpp process changes
        already publish their real initialization state through _load.
        """
        self._validate_adapter(spec, adapter)
        ident = adapter["id"] if adapter else None
        if spec.backend == "llama.cpp" and self._runtime_adapter != ident:
            await self._unload()
            await self._load(spec, adapter)
        elif adapter and (spec.backend == "vllm" or spec.adapter_backend == "vllm") and ident not in self._loaded_loras:
            response = await self.client.post(spec.base_url + "/load_lora_adapter", headers=self._headers(spec), json={"lora_name": ident, "lora_path": adapter["path"]})
            if response.is_error:
                raise RuntimeError(f"Inference backend rejected adapter {ident} (HTTP {response.status_code}): {response.text[:4096]}")
            self._loaded_loras.add(ident)
        self._runtime_adapter = ident
        return ident if adapter and (spec.backend == "vllm" or spec.adapter_backend == "vllm") else spec.model_name

    async def activate_adapter(self, adapter: dict | None):
        async with self._lock:
            if not self.current():
                raise ValueError("Load a model before selecting an adapter")
            await self._select_adapter(self.specs[self._current], adapter)
            current = self.current()
            ident = adapter["id"] if adapter else None
            if current and current.get("adapter_id") != ident:
                current = self.store.update("models", current["id"], adapter_id=ident)
            return current

    @staticmethod
    def _thinking_options(spec: ModelSpec, requested: ThinkingMode | None) -> dict:
        kwargs = dict(spec.chat_template_kwargs)
        # Older profiles already specify their template's switch and effort.
        # Leave those unchanged unless a request or new profile mode overrides it.
        legacy = requested is None and spec.thinking_mode is None and "enable_thinking" in kwargs
        if legacy:
            return {"chat_template_kwargs": kwargs}
        mode = spec.effective_thinking(requested)
        effort = "none" if mode == "off" else spec.reasoning_effort_map.get(mode, mode)
        kwargs["enable_thinking"] = mode != "off"
        if mode == "off":
            kwargs.pop("reasoning_effort", None)
        else:
            kwargs["reasoning_effort"] = effort
        return {"reasoning_effort": effort, "chat_template_kwargs": kwargs}

    @staticmethod
    async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
        """Read SSE data fields, including multiline frames and a final EOF frame."""
        lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if lines:
                    yield "\n".join(lines)
                    lines.clear()
            elif line.startswith("data:"):
                value = line[5:]
                lines.append(value[1:] if value.startswith(" ") else value)
        if lines:
            yield "\n".join(lines)

    async def stream_events(self, messages: list[dict], model_id: str | None = None, adapter: dict | None = None, temperature: float = 0.7, max_tokens: int = 512, thinking: ThinkingMode | None = None) -> AsyncIterator[tuple[str, dict]]:
        """Provider channels stay separate; this generator owns the lock once."""
        async with self._lock:
            current = self.current()
            if not current:
                raise ValueError("No model loaded. Use /model load <id> first")
            if model_id and self._current != model_id:
                raise ValueError("Requested model is not loaded")
            spec = self.specs[self._current]
            model = await self._select_adapter(spec, adapter)
            if spec.backend == "demo":
                text = self._demo_reply(messages, adapter)
                for offset in range(0, len(text), 12):
                    await asyncio.sleep(0.002)
                    yield "delta", {"content": text[offset:offset + 12]}
                yield "finish", {"finish_reason": "stop"}
                return
            payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
            payload.update(self._thinking_options(spec, thinking))
            async with self.client.stream("POST", spec.base_url + "/chat/completions", headers=self._headers(spec), json=payload) as response:
                if response.is_error:
                    await response.aread()
                    raise RuntimeError(f"Inference backend rejected chat (HTTP {response.status_code}): {response.text[:4096]}")
                finished = False
                async for raw in self._sse_data(response):
                    raw = raw.strip()
                    if raw == "[DONE]":
                        if not finished:
                            yield "finish", {"finish_reason": None}
                        finished = True
                        break
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except ValueError as exc:
                        raise RuntimeError("Inference stream returned malformed JSON") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError("Inference stream returned a non-object event")
                    if "error" in event:
                        raise RuntimeError(f"Inference stream error: {event['error']}")
                    choices = event.get("choices", [])
                    if not isinstance(choices, list):
                        raise RuntimeError("Inference stream returned invalid choices")
                    for choice in choices:
                        if not isinstance(choice, dict):
                            raise RuntimeError("Inference stream returned an invalid choice")
                        delta = choice.get("delta")
                        if delta is None:
                            delta = {}
                        if not isinstance(delta, dict):
                            raise RuntimeError("Inference stream returned an invalid delta")
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content")
                        if reasoning is None or reasoning == "":
                            reasoning = delta.get("reasoning")
                        if any(value is not None and not isinstance(value, str) for value in (content, reasoning)):
                            raise RuntimeError("Inference stream returned a non-text content channel")
                        if reasoning:
                            yield "reasoning", {"content": reasoning}
                        if content:
                            yield "delta", {"content": content}
                        if choice.get("finish_reason") is not None:
                            if not isinstance(choice["finish_reason"], str):
                                raise RuntimeError("Inference stream returned an invalid finish reason")
                            finished = True
                            yield "finish", {"finish_reason": choice["finish_reason"]}
                    if event.get("usage") is not None:
                        if not isinstance(event["usage"], dict):
                            raise RuntimeError("Inference stream returned invalid usage")
                        yield "usage", event["usage"]
                if not finished:
                    raise RuntimeError("Inference stream ended before a completion marker")

    async def stream(self, messages: list[dict], model_id: str | None = None, adapter: dict | None = None, temperature: float = 0.7, max_tokens: int = 512, thinking: ThinkingMode | None = None) -> AsyncIterator[str]:
        """Text-only compatibility path for callers such as evaluation/training."""
        async with aclosing(self.stream_events(messages, model_id, adapter, temperature, max_tokens, thinking)) as events:
            async for event, data in events:
                if event == "delta":
                    yield data["content"]

    def _agent_provider(self, dialect: str, model_id: str | None, profile_revision: str | None):
        current = self.current()
        if not current or (model_id is not None and current["id"] != model_id):
            raise ProviderError("The agent's selected model is no longer loaded")
        spec = self.specs[current["id"]]
        snapshot = self.capabilities.get(spec.id)
        if snapshot is None or (profile_revision is not None and snapshot.profile_revision != profile_revision):
            raise ProviderError("The agent's startup capability revision changed; start a new run")
        adapter = self._provider_adapters.get(spec.id, {}).get(dialect)
        if adapter is None:
            raise ProviderError(f"The loaded model does not advertise provider dialect: {dialect}")
        return current, spec, snapshot, adapter

    async def count_tokens(self, messages: list[dict], tools=(), *, dialect: str = "chat_completions",
                           model_id: str | None = None, profile_revision: str | None = None,
                           thinking: ThinkingMode = "off", max_tokens: int = 2048,
                           response_format: dict | None = None) -> dict:
        """Count a known native dialect, otherwise return a labeled safe estimate.

        A count route is a render/tokenize request, never generation or capability
        discovery. Its availability comes only from the startup snapshot.
        """
        async with self._lock:
            _, spec, snapshot, provider = self._agent_provider(dialect, model_id, profile_revision)
            endpoints = snapshot.capabilities.get("tokenization", {}).get("endpoints", ())
            native_chat = (spec.backend == "vllm" and dialect == "chat_completions" and thinking == "off"
                           and not tools and response_format is None and snapshot.supports("tokenization")
                           and "/tokenize" in endpoints)
            if native_chat:
                # vLLM TokenizeChatRequest uses the same renderer/template as
                # ChatCompletionRequest. Keep this narrow: tool-choice and JSON
                # response-format rendering need contracts not exposed by that
                # endpoint. No route is discovered or probed during a turn.
                payload = provider.payload(model=spec.model_name, messages=messages, tools=[], max_tokens=max_tokens,
                    temperature=.2, thinking=thinking, thinking_options=self._thinking_options(spec, thinking), compatibility=True)
                text_only = all(isinstance(message.get("content"), (str, type(None))) for message in payload["messages"])
                if text_only:
                    template = dict(payload.get("chat_template_kwargs") or {})
                    # ChatCompletionRequest.build_chat_params supplies this
                    # template variable; TokenizeChatRequest has no top-level
                    # reasoning_effort field, so mirror it explicitly.
                    template["reasoning_effort"] = payload.get("reasoning_effort", "none")
                    token_payload = {"model": payload["model"], "messages": payload["messages"],
                        "chat_template_kwargs": template, "add_generation_prompt": True,
                        "continue_final_message": False, "add_special_tokens": False}
                    api_root = spec.base_url[:-3] if spec.base_url.endswith("/v1") else spec.base_url
                    endpoint = api_root.rstrip("/") + "/tokenize"
                    try:
                        response = await self.client.post(endpoint, headers=self._headers(spec), json=token_payload,
                                                          timeout=httpx.Timeout(None, connect=10))
                        response.raise_for_status()
                        result = response.json()
                        value, tokens = result.get("count"), result.get("tokens")
                        if (type(value) is not int or value < 1 or not isinstance(tokens, list) or len(tokens) != value
                                or any(type(token) is not int or token < 0 for token in tokens)):
                            raise ValueError("Invalid native token count")
                    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                        # An unavailable/invalid advertised counter never means
                        # zero input. Preserve the conservative safety bound.
                        return {**conservative_count(messages, [], response_format), "count_error": "native_tokenizer_unavailable"}
                    return {"input_tokens": value, "method": "vllm_tokenize_chat", "estimated": False}
            if dialect != "anthropic_messages" or not snapshot.supports("token_counting"):
                return conservative_count(messages, list(tools), response_format)
            if response_format:
                raise ProviderError("Structured response_format is not supported by Anthropic token counting")
            payload = provider.payload(model=spec.model_name, messages=messages, tools=list(tools),
                max_tokens=max_tokens, temperature=.2, thinking=thinking,
                thinking_options=self._thinking_options(spec, thinking), compatibility=spec.backend == "strata")
            payload.pop("stream", None)
            response = await self.client.post(spec.base_url + "/messages/count_tokens", headers=self._headers(spec),
                                              json=payload, timeout=httpx.Timeout(None, connect=10))
            if response.is_error:
                raise ProviderError(f"Provider rejected token counting (HTTP {response.status_code})")
            try:
                value = response.json()["input_tokens"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError("Provider token count response is invalid") from exc
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProviderError("Provider token count must be a nonnegative integer")
            return {"input_tokens": value, "method": "provider", "estimated": False}

    async def agent_events(self, messages: list[dict], tools=(), *, dialect: str = "chat_completions",
                           model_id: str | None = None, profile_revision: str | None = None,
                           adapter: dict | None = None, temperature: float = .2, max_tokens: int = 2048,
                           thinking: ThinkingMode = "off", tool_choice="auto", response_format: dict | None = None,
                           input_tokens: int | None = None) -> AsyncIterator[tuple[str, dict]]:
        """One pinned provider turn, with complete calls emitted after completion.

        The engine lock covers the entire owned HTTP stream. Model selection,
        profile changes and adapter restarts cannot race this provider request.
        The caller still owns executor policy, full argument validation and the
        idempotency ledger; a capability snapshot is never execution permission.
        """
        async with self._lock:
            current, spec, snapshot, provider = self._agent_provider(dialect, model_id, profile_revision)
            if thinking not in ("off", "low", "medium", "high", "max"):
                raise ValueError("Unsupported agent thinking mode")
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
                raise ValueError("Agent output budget must be a positive integer")
            capacity = (current.get("effective_context_length") or snapshot.context.get("effective_context_length")
                        or (spec.context_length if isinstance(spec.context_length, int) else None))
            if capacity is None:
                raise ProviderError("Agent context capacity is unknown; configure or advertise the active context limit")
            output_limit = current.get("max_output_tokens") or capacity
            if input_tokens is None:
                input_tokens = conservative_count(messages, list(tools), response_format)["input_tokens"]
            if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
                raise ValueError("Agent input token count is invalid")
            if input_tokens + max_tokens > capacity:
                raise ProviderError(f"Agent request exceeds active context: {input_tokens} input + {max_tokens} output > {capacity}")
            if max_tokens > output_limit:
                raise ProviderError(f"Requested output budget exceeds the loaded model's configured allowance ({output_limit})")
            model = await self._select_adapter(spec, adapter)
            # llama.cpp adapter activation can initialize a new owned server.
            _, spec, snapshot, provider = self._agent_provider(dialect, model_id, profile_revision)
            if spec.backend == "demo":
                if tools:
                    raise ProviderError("The demonstration backend does not execute model tool calls")
                yield "delta", {"content": self._demo_reply(messages, adapter)}
                yield "finish", {"finish_reason": "stop"}
                return
            payload = provider.payload(model=model, messages=messages, tools=list(tools),
                max_tokens=max_tokens, temperature=temperature, thinking=thinking,
                thinking_options=self._thinking_options(spec, thinking), tool_choice=tool_choice,
                response_format=response_format, compatibility=spec.backend in {"strata", "vllm", "llama.cpp"})
            async with aclosing(provider.stream(self.client, spec.base_url, self._headers(spec), payload,
                                               response_format=response_format)) as events:
                async for event, data in events:
                    if event != "reasoning" or thinking != "off":
                        yield event, data

    async def complete(self, messages: list[dict], model_id: str | None = None, adapter: dict | None = None, temperature: float = 0.7, max_tokens: int = 512, thinking: ThinkingMode | None = None) -> str:
        return "".join([part async for part in self.stream(messages, model_id, adapter, temperature, max_tokens, thinking)])

    @staticmethod
    def _demo_reply(messages: list[dict], adapter: dict | None) -> str:
        prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        lower = prompt.lower()
        if "2 + 2" in lower or "2+2" in lower:
            answer = "4"
        elif "your name" in lower or "who are you" in lower:
            answer = "I'm Lucifer, running in demonstration mode."
        elif "json" in lower:
            answer = '{"ok": true}'
        elif "capital" in lower and "france" in lower:
            answer = "Paris."
        elif "secret" in lower or "system prompt" in lower:
            answer = "I cannot reveal secrets or private system instructions."
        else:
            answer = "This is the demonstration backend. Configure a llama.cpp, vLLM, or external model to receive real model answers."
        return answer

    async def close(self):
        await self.unload()
        await self.capabilities.close()
        await self.client.aclose()
