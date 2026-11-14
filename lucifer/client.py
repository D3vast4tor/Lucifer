"""Small asynchronous API client, shared by the terminal and headless checks."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

import httpx


class APIError(RuntimeError):
    """An actionable backend or transport error."""


class APIClient:
    def __init__(self, api_url: str = "http://127.0.0.1:8000", api_token: str | None = None,
                 *, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=api_url.rstrip("/"), timeout=httpx.Timeout(600, connect=5),
            headers={"Authorization": f"Bearer {api_token}"} if api_token else {},
            transport=transport,
        )

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            raise APIError(f"{response.status_code}: {detail}")

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self.http.request(method, path, **kwargs)
            self._check(response)
            return response.json() if response.content else None
        except httpx.HTTPError as exc:
            raise APIError(f"Cannot reach backend: {exc}") from exc

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, data: dict | None = None) -> Any:
        return await self.request("POST", path, json=data or {})

    async def patch(self, path: str, data: dict) -> Any:
        return await self.request("PATCH", path, json=data)

    async def bootstrap(self) -> dict:
        """Fetch structural UI state once, or after a registry revision changes."""
        return await self.get("/system/bootstrap")

    async def telemetry(self) -> dict:
        """Read the backend's cached live sample; this does not rescan models."""
        return await self.get("/system/telemetry")

    async def load_model(self, model_id: str) -> dict:
        """Let the backend's profile startup timeout govern long model loads."""
        return await self.request("POST", "/models/load", json={"model_id": model_id},
                                  timeout=httpx.Timeout(None, connect=5))

    async def agent_history(self) -> list[dict]:
        """Fetch saved/retained mission metadata only when history is opened."""
        return await self.get("/agent/history")

    async def attach_knowledge(self, conversation_id: str, source: str, *, upload=False, video_frames=8):
        """Upload local bytes, or explicitly link a backend-local path/public URL."""
        from pathlib import Path
        from urllib.parse import urlsplit
        from .document_extract import MAX_BYTES
        payload = {"video_frames": video_frames}
        if urlsplit(source).scheme in {"http", "https"}:
            if upload:
                raise ValueError("Use /link for a public URL, /attach for a local upload")
            payload["url"] = source
        elif upload:
            import asyncio
            import base64
            selected = Path(source).expanduser()
            def read():
                with selected.open("rb") as stream:
                    return stream.read(MAX_BYTES + 1)
            raw = await asyncio.to_thread(read)
            if not raw or len(raw) > MAX_BYTES:
                raise ValueError("Attach a nonempty file no larger than 128 MiB")
            payload.update(filename=selected.name, content_base64=base64.b64encode(raw).decode())
        else:
            payload["path"] = str(Path(source).expanduser().absolute())
        return await self.request("POST", f"/conversations/{conversation_id}/knowledge", json=payload,
                                  timeout=httpx.Timeout(None, connect=5))

    async def agent_session(self, run_id: str) -> dict:
        """Read public transcript/availability without restoring or executing it."""
        return await self.get(f"/agent/runs/{run_id}/session")

    async def start_sleep(self, *, dry_run: bool = False, dynamic: bool = True, context_examples: bool = True,
                          schedule: dict | None = None, generation_thinking: str = "off",
                          generation_model: str | None = None) -> dict:
        """Start the whole pipeline; dynamic selection is owned by the backend."""
        payload = {"dynamic": dynamic, "dry_run": dry_run, "context_examples": context_examples}
        if schedule is not None:
            payload["schedule"] = schedule
        if generation_thinking != "off":
            payload["generation_thinking"] = generation_thinking
        if generation_model is not None:
            payload["generation_model"] = generation_model
        return await self.post("/sleep/start", payload)

    async def resume_sleep(self, run_id: str, *, idempotency_key: str | None = None) -> dict:
        """Resume one explicit job; manual retries must retain the same key."""
        key = uuid4().hex if idempotency_key is None else idempotency_key
        return await self.request("POST", f"/sleep/runs/{run_id}/resume", json={"idempotency_key": key},
                                  timeout=httpx.Timeout(None, connect=5))

    async def sleep_examples(self, status: str | None = "pending") -> list[dict]:
        return await self.get("/sleep/examples", params={"status": status} if status else {})

    async def review_sleep_example(self, candidate_id: str, *, decision: str, content_hash: str,
                                   correction: str | None = None, reason: str | None = None) -> dict:
        """Review the exact displayed content; do not retry with a newer hash."""
        payload = {"decision": decision, "content_hash": content_hash}
        if correction is not None:
            payload["correction"] = correction
        if reason is not None:
            payload["reason"] = reason
        return await self.post(f"/sleep/examples/{candidate_id}/review", payload)

    async def cancel_sleep(self, run_id: str) -> dict:
        return await self.request("POST", "/sleep/cancel", json={"run_id": run_id},
                                  timeout=httpx.Timeout(None, connect=5))

    async def start_meditation(self, *, conversation_ids: list[str] | None = None,
                               base_model: str | None = None, seed: int | None = None,
                               mode: str = "supervised", dry_run: bool = False,
                               pilot_updates: int | None = None,
                               idempotency_key: str | None = None) -> dict:
        """Start one native-head run; retain the caller key for an explicit retry."""
        if mode not in {"supervised", "unsupervised"}:
            raise ValueError("Choose supervised or unsupervised Meditation")
        payload = {"mode": mode, "dry_run": dry_run, "idempotency_key": idempotency_key or uuid4().hex}
        for name, value in (("conversation_ids", conversation_ids), ("base_model", base_model), ("seed", seed), ("pilot_updates", pilot_updates)):
            if value is not None:
                payload[name] = value
        return await self.request("POST", "/meditation/start", json=payload,
                                  timeout=httpx.Timeout(None, connect=5))

    async def meditation_action(self, run_id: str, action: str, *, idempotency_key: str | None = None,
                                conversation_ids: list[str] | None = None, source_roles: dict[str, str] | None = None) -> dict:
        from urllib.parse import quote
        if action not in {"resume", "cancel"}:
            raise ValueError("Choose resume or cancel")
        scope = {name: value for name, value in (("conversation_ids", conversation_ids), ("source_roles", source_roles))
                 if value is not None}
        if scope and action != "resume":
            raise ValueError("Source scope can only be extended on resume")
        return await self.request("POST", f"/meditation/runs/{quote(run_id, safe='')}/{action}",
                                  json={"idempotency_key": idempotency_key or uuid4().hex, **scope},
                                  timeout=httpx.Timeout(None, connect=5))

    async def activate_base(self, base_id: str, *, idempotency_key: str | None = None) -> dict:
        from urllib.parse import quote
        return await self.request("POST", f"/meditation/bases/{quote(base_id, safe='')}/activate",
                                  json={"idempotency_key": idempotency_key or uuid4().hex},
                                  timeout=httpx.Timeout(None, connect=5))

    async def rollback_base(self, base_id: str | None = None, *, idempotency_key: str | None = None) -> dict:
        payload = {"idempotency_key": idempotency_key or uuid4().hex}
        if base_id is not None:
            payload["base_id"] = base_id
        return await self.request("POST", "/meditation/rollback", json=payload,
                                  timeout=httpx.Timeout(None, connect=5))

    async def stream_chat(self, payload: dict) -> AsyncIterator[tuple[str, dict]]:
        """Parse SSE frames, including multiline data and final unterminated frames."""
        async with aclosing(self.stream_events("POST", "/chat/stream", payload=payload)) as events:
            async for event in events:
                yield event

    async def start_agent(self, payload: dict) -> dict:
        return await self.request("POST", "/agent/runs", json=payload, timeout=httpx.Timeout(None, connect=5))

    async def agent_input(self, run_id: str, content: str, *, interrupt: bool = True,
                          options: dict | None = None, idempotency_key: str | None = None) -> dict:
        """Submit one turn; retain a caller key when manually retrying that intent."""
        key = uuid4().hex if idempotency_key is None else idempotency_key
        return await self.request("POST", f"/agent/runs/{run_id}/input",
                                  json={"content": content, "interrupt": interrupt, "options": options or {},
                                        "idempotency_key": key},
                                  timeout=httpx.Timeout(None, connect=5))

    async def resume_agent(self, run_id: str, *, instruction: str | None = None,
                           limits: dict | None = None, idempotency_key: str | None = None) -> dict:
        """Resume once; a retry must reuse its original key and unchanged payload."""
        key = uuid4().hex if idempotency_key is None else idempotency_key
        return await self.request("POST", f"/agent/runs/{run_id}/resume",
                                  json={"instruction": instruction, "limits": limits or {}, "idempotency_key": key},
                                  timeout=httpx.Timeout(None, connect=5))

    async def stop_agent(self, run_id: str) -> dict:
        return await self.request("POST", f"/agent/runs/{run_id}/stop", json={},
                                  timeout=httpx.Timeout(None, connect=5))

    async def stream_agent(self, run_id: str, *, after: int = 0) -> AsyncIterator[tuple[str, dict]]:
        """Attach to an existing run. Reconnects never submit its goal again."""
        async with aclosing(self.stream_events("GET", f"/agent/runs/{run_id}/events", params={"after": after}, read_timeout=None)) as events:
            async for event in events:
                yield event

    async def stream_events(self, method: str, path: str, *, payload: dict | None = None,
                            params: dict | None = None, read_timeout: float | None = 600) -> AsyncIterator[tuple[str, dict]]:
        event, data = "message", []
        try:
            kwargs = {"params": params} if params is not None else {}
            kwargs["timeout"] = httpx.Timeout(read_timeout, connect=5)
            if payload is not None:
                kwargs["json"] = payload
            async with self.http.stream(method, path, **kwargs) as response:
                if response.is_error:
                    await response.aread()
                    self._check(response)
                async for line in response.aiter_lines():
                    if not line:
                        if data:
                            yield self._decode(event, data)
                        event, data = "message", []
                    elif line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                if data:
                    yield self._decode(event, data)
        except httpx.HTTPError as exc:
            raise APIError(f"Stream connection failed: {exc}") from exc

    @staticmethod
    def _decode(event: str, data: list[str]) -> tuple[str, dict]:
        raw = "\n".join(data)
        if raw == "[DONE]":
            return "done", {}
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise APIError("Backend returned an invalid stream frame") from exc
        if not isinstance(payload, dict):
            raise APIError("Backend returned an invalid stream payload")
        return event, payload

    async def close(self) -> None:
        await self.http.aclose()
