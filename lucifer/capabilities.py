"""Startup-only provider capability discovery; no generation or tool execution.

Availability is evidence, not permission. A route/schema advertised by a provider
is distinct from a behavior verified by an explicit acceptance check. Unknown
features stay unknown, including when another feature or provider brand suggests
they might exist.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

import httpx


CAPABILITY_NAMES = (
    "chat_completions", "anthropic_messages", "responses", "streaming", "tools",
    "reasoning", "vision", "structured_output", "usage", "cancellation",
    "token_counting", "tokenization", "mcp",
)
_STATES = {"unknown", "unavailable", "advertised", "verified"}
_DIALECTS = {
    "/v1/chat/completions": "chat_completions",
    "/v1/messages": "anthropic_messages",
    "/v1/responses": "responses",
}
_ROUTES = {
    **{("post", route): name for route, name in _DIALECTS.items()},
    ("post", "/v1/messages/count_tokens"): "token_counting",
    ("post", "/tokenize"): "tokenization",
    ("post", "/detokenize"): "tokenization",
    ("get", "/mcp"): "mcp",
}
_MAX_METADATA_BYTES = 2 * 1024 * 1024


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _positive(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _strings(value):
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple)) else []


@dataclass(frozen=True)
class CapabilitySnapshot:
    model_id: str
    profile_revision: str
    discovered_at: str
    capabilities: Mapping[str, Any]
    context: Mapping[str, Any]
    discovery: Mapping[str, Any]

    def __post_init__(self):
        for key in ("capabilities", "context", "discovery"):
            object.__setattr__(self, key, _freeze(getattr(self, key)))

    def as_dict(self) -> dict:
        return {"schema_version": 1, "model_id": self.model_id, "profile_revision": self.profile_revision,
                "discovered_at": self.discovered_at, "capabilities": _thaw(self.capabilities),
                "context": _thaw(self.context), "discovery": _thaw(self.discovery)}

    def supports(self, name: str, *, require_verified: bool = False) -> bool:
        state = self.capabilities.get(name, {}).get("status", "unknown")
        return state == "verified" if require_verified else state in {"advertised", "verified"}


def _mark(caps, name, *, state="advertised", evidence=None, endpoint=None, constraints=None):
    if name not in caps:
        return
    entry = caps[name]
    entry["status"] = state
    if isinstance(evidence, str):
        evidence = [evidence]
    for item in evidence or []:
        if isinstance(item, str) and item not in entry["evidence"]:
            entry["evidence"].append(item)
    if endpoint and endpoint not in entry["endpoints"]:
        entry["endpoints"].append(endpoint)
    if isinstance(constraints, Mapping):
        entry["constraints"].update(constraints)


def _declarations(caps, declarations, source, *, verified=False):
    if not isinstance(declarations, Mapping):
        return
    for name, value in (declarations or {}).items():
        if name not in caps:
            continue
        if isinstance(value, bool):
            state = ("verified" if verified else "advertised") if value else "unavailable"
            details = {}
        elif isinstance(value, Mapping):
            details = value
            state = value.get("status", "advertised")
            if state not in _STATES:
                continue
            # A provider/profile saying "verified" is still a declaration. Only
            # the separately supplied acceptance ledger can establish verification.
            if state in {"advertised", "verified"}:
                state = "verified" if verified else "advertised"
        else:
            continue
        _mark(caps, name, state=state, evidence=[source, *_strings(details.get("evidence"))],
              constraints=details.get("constraints"))
        for endpoint in _strings(details.get("endpoints")):
            _mark(caps, name, state=state, endpoint=endpoint)


def _local_schema(schema, document):
    seen = set()
    while isinstance(schema, Mapping) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if not ref.startswith("#/") or ref in seen or len(seen) >= 8:
            return {}
        seen.add(ref)
        schema = document
        for part in ref[2:].split("/"):
            schema = schema.get(part.replace("~1", "/").replace("~0", "~"), {}) if isinstance(schema, Mapping) else {}
    return schema if isinstance(schema, Mapping) else {}


def _openapi_capabilities(caps, document):
    if not isinstance(document, Mapping) or not isinstance(document.get("paths"), Mapping):
        return
    for route, methods in document["paths"].items():
        if not isinstance(methods, Mapping):
            continue
        for method, operation in methods.items():
            if not isinstance(method, str):
                continue
            name = _ROUTES.get((method.lower(), route))
            if name:
                _mark(caps, name, evidence=["startup OpenAPI route"], endpoint=route)
            if not isinstance(operation, Mapping):
                continue
            if method.lower() == "post" and route in _DIALECTS:
                body = _local_schema(operation.get("requestBody", {}), document)
                content = body.get("content", {})
                json_body = content.get("application/json", {}) if isinstance(content, Mapping) else {}
                schema = _local_schema(json_body.get("schema", {}) if isinstance(json_body, Mapping) else {}, document)
                props = schema.get("properties", {})
                if isinstance(props, Mapping):
                    for field, feature in (("tools", "tools"), ("stream", "streaming"),
                                           ("reasoning_effort", "reasoning"), ("response_format", "structured_output")):
                        if field in props:
                            _mark(caps, feature, evidence=[f"startup OpenAPI request field: {field}"])


def _snapshot(model_id, profile_revision, *, models=None, status=None, openapi=None, declared=None, verified=None,
              native_context_length=None, effective_context_length=None, served_model_name=None, discovery=None):
    caps = {name: {"status": "unknown", "evidence": [], "endpoints": [], "constraints": {}}
            for name in CAPABILITY_NAMES}
    # Profile hints are lowest precedence. The current server's explicit disabled
    # state must not be replaced by a remembered declaration of availability.
    _declarations(caps, declared, "explicit profile declaration")
    _openapi_capabilities(caps, openapi)
    entries = models.get("data", []) if isinstance(models, Mapping) else models if isinstance(models, list) else []
    entries = [entry for entry in entries if isinstance(entry, Mapping)] if isinstance(entries, list) else []
    wanted = served_model_name or model_id
    selected = next((entry for entry in entries if entry.get("id") == wanted), None)
    if selected is None and len(entries) == 1:
        selected = entries[0]
    selected = selected or {}
    status = status if isinstance(status, Mapping) else {}
    for route in status.get("dialects", []) if isinstance(status.get("dialects"), list) else []:
        if isinstance(route, str) and route in _DIALECTS:
            _mark(caps, _DIALECTS[route], evidence=["startup /v1/status dialect"], endpoint=route)
    for source, document in (("startup /v1/models capabilities", selected), ("startup /v1/status capabilities", status)):
        if isinstance(document.get("capabilities"), Mapping):
            _declarations(caps, document["capabilities"], source)
    structured = status.get("structured_output")
    if isinstance(structured, Mapping) and structured.get("formats"):
        _mark(caps, "structured_output", evidence=["startup /v1/status structured_output"],
              constraints={key: structured[key] for key in ("formats", "method", "constrained_decoding", "stream_buffered")
                           if key in structured})
    vision = status.get("vision")
    if isinstance(vision, Mapping):
        available = vision.get("available", vision.get("enabled"))
        if isinstance(available, bool):
            enabled = vision.get("enabled")
            _mark(caps, "vision", state="unavailable" if not available or enabled is False else "advertised",
                  evidence=["startup /v1/status vision"],
                  constraints={key: vision[key] for key in ("enabled", "available") if key in vision})
    architecture = selected.get("architecture")
    if caps["vision"]["status"] == "unknown" and isinstance(architecture, Mapping):
        modalities = architecture.get("input_modalities", [])
        if isinstance(modalities, list) and "image" in modalities:
            _mark(caps, "vision", evidence=["startup /v1/models image modality"])
    _declarations(caps, verified, "explicit acceptance verification", verified=True)
    # Even strict provider schemas do not replace local executor-side validation.
    caps["tools"]["constraints"]["local_argument_validation_required"] = True
    caps["structured_output"]["constraints"]["local_result_validation_required"] = True
    meta = selected.get("meta", {})
    context = status.get("context", {})
    effective = (_positive(effective_context_length) or _positive(status.get("cache_max_tokens")) or
                 _positive(context.get("max_positions") if isinstance(context, Mapping) else None) or
                 _positive(meta.get("n_ctx") if isinstance(meta, Mapping) else None))
    # Strata's context.native is the configured window, not independent training metadata.
    native = _positive(native_context_length) or _positive(status.get("native_context_length"))
    return CapabilitySnapshot(model_id, profile_revision, datetime.now(timezone.utc).isoformat(), caps,
                              {"effective_context_length": effective, "native_context_length": native}, discovery or {})


class CapabilityRegistry:
    """Memoized startup snapshots; callers invalidate changed profiles or new loads."""

    def __init__(self):
        self._cache: dict[tuple[str, str], CapabilitySnapshot] = {}
        self._latest: dict[str, tuple[str, str]] = {}
        self._inflight: dict[tuple[str, str], asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        self._epochs: dict[str, int] = {}

    def get(self, model_id: str, profile_revision: str | None = None) -> CapabilitySnapshot | None:
        key = (model_id, profile_revision) if profile_revision is not None else self._latest.get(model_id)
        return self._cache.get(key) if key else None

    def invalidate(self, model_id: str) -> None:
        self._epochs[model_id] = self._epochs.get(model_id, 0) + 1
        self._latest.pop(model_id, None)
        for key in [key for key in self._cache if key[0] == model_id]:
            self._cache.pop(key)
        for key in [key for key in self._inflight if key[0] == model_id]:
            task = self._inflight.pop(key)
            if not task.done():
                task.cancel()

    async def cancel_model(self, model_id: str) -> None:
        """Cancel and reap this model's discovery before its load/client ends."""
        tasks = [task for key, task in self._inflight.items() if key[0] == model_id]
        self.invalidate(model_id)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """End owned metadata requests before the shared HTTP client closes."""
        tasks = list(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()

    def record(self, model_id: str, profile_revision: str, **metadata) -> CapabilitySnapshot:
        existing = self.get(model_id, profile_revision)
        if existing is not None:
            return existing
        snapshot = _snapshot(model_id, profile_revision, **metadata)
        key = (model_id, profile_revision)
        self._cache[key] = snapshot
        self._latest[model_id] = key
        return snapshot

    async def discover(self, model_id: str, profile_revision: str, *, client: httpx.AsyncClient,
                       base_url: str, headers=None, models=None, status=None, declared=None, verified=None,
                       include_openapi: bool = False, openapi=None, native_context_length=None,
                       effective_context_length=None, served_model_name=None) -> CapabilitySnapshot:
        existing = self.get(model_id, profile_revision)
        if existing is not None:
            return existing
        key = (model_id, profile_revision)
        task = self._inflight.get(key)
        if task is None:
            epoch = self._epochs.get(model_id, 0)

            async def collect():
                api = base_url.rstrip("/")
                root = api[:-3] if api.endswith("/v1") else api
                api = api if api.endswith("/v1") else api + "/v1"
                documents = {"models": models, "status": status, "openapi": openapi}
                discovery = {name: {"status": "injected"} for name, value in documents.items() if value is not None}

                async def fetch(name, url):
                    try:
                        response = await client.get(url, headers=headers, timeout=5)
                        if response.is_error:
                            discovery[name] = {"status": "unavailable", "http_status": response.status_code}
                        elif len(response.content) > _MAX_METADATA_BYTES:
                            discovery[name] = {"status": "error", "reason": "metadata_too_large"}
                        else:
                            documents[name] = response.json()
                            discovery[name] = {"status": "ok", "http_status": response.status_code}
                    except (httpx.HTTPError, ValueError) as exc:
                        discovery[name] = {"status": "error", "reason": type(exc).__name__}

                jobs = [fetch(name, url) for name, url in (("models", api + "/models"), ("status", api + "/status"))
                        if documents[name] is None]
                if include_openapi and openapi is None:
                    jobs.append(fetch("openapi", root + "/openapi.json"))
                await asyncio.gather(*jobs)
                snapshot = _snapshot(model_id, profile_revision, **documents, declared=declared, verified=verified,
                                     native_context_length=native_context_length,
                                     effective_context_length=effective_context_length,
                                     served_model_name=served_model_name, discovery=discovery)
                if self._epochs.get(model_id, 0) == epoch:
                    self._cache[key] = snapshot
                    self._latest[model_id] = key
                return snapshot

            task = asyncio.create_task(collect())
            self._inflight[key] = task
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._inflight.get(key) is task:
                self._inflight.pop(key, None)
