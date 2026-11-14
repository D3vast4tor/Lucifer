"""Explicitly configured Streamable-HTTP MCP tools, negotiated once at startup."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from urllib.parse import urlsplit

import httpx
from .. import __version__

from .workspace import ToolError, ToolPolicyError


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    enabled: bool = False
    allowed_tools: tuple[str, ...] | list[str] = ()
    allow_mutations: bool = False
    headers_env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 15.0

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,23}", self.name):
            raise ToolError("MCP server names must be short ASCII identifiers")
        parsed = urlsplit(self.url)
        if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
            raise ToolError("MCP URLs cannot contain credentials or fragments")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise ToolPolicyError("Remote MCP transport requires HTTPS; HTTP is restricted to loopback")
        if not 1 <= self.timeout_seconds <= 60:
            raise ToolError("MCP timeout must be between one and sixty seconds")
        if not isinstance(self.allowed_tools, (list, tuple)) or not all(isinstance(item, str) for item in self.allowed_tools):
            raise ToolError("MCP allowed_tools must be an explicit list of tool names")
        if self.enabled and not self.allowed_tools:
            raise ToolPolicyError("Enable a server with an explicit allowed_tools list, or ['*']")
        if not isinstance(self.headers_env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.headers_env.items()):
            raise ToolError("MCP header mappings must reference environment variable names")


class MCPClient:
    def __init__(self, config: MCPServerConfig, *, transport=None):
        self.config = config
        self.sequence = 0
        self.session_id = None
        self.initialized = False
        self._specs = []
        self._headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-03-26"}
        for header, variable in config.headers_env.items():
            value = os.environ.get(variable)
            if not value:
                raise ToolPolicyError("MCP authentication environment is missing: " + variable)
            self._headers[header] = value
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds, follow_redirects=False, transport=transport)

    async def _rpc(self, method, parameters=None, *, notification=False, output_limit=2 * 1024 * 1024):
        self.sequence += 1
        body = {"jsonrpc": "2.0", "method": method}
        if parameters is not None:
            body["params"] = parameters
        if not notification:
            body["id"] = self.sequence
        headers = dict(self._headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        async with self.client.stream("POST", self.config.url, json=body, headers=headers) as response:
            response.raise_for_status()
            session = response.headers.get("Mcp-Session-Id")
            if session and len(session) <= 256:
                self.session_id = session
            if notification:
                return {}
            try:
                if "text/event-stream" in response.headers.get("content-type", ""):
                    document, data, size = None, [], 0
                    async for line in response.aiter_lines():
                        size += len(line.encode("utf-8")) + 1
                        if size > output_limit:
                            raise ToolError("MCP response exceeded the bounded result size")
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif not line and data:
                            value = json.loads("\n".join(data))
                            data.clear()
                            if isinstance(value, dict) and value.get("id") == body["id"]:
                                document = value
                                break
                    if document is None:
                        raise ToolError("MCP SSE ended without the matching response")
                else:
                    raw = bytearray()
                    async for part in response.aiter_bytes():
                        raw.extend(part)
                        if len(raw) > output_limit:
                            raise ToolError("MCP response exceeded the bounded result size")
                    document = json.loads(raw)
            except (ValueError, UnicodeDecodeError, StopIteration) as exc:
                raise ToolError("MCP response was not a matching JSON-RPC message") from exc
            if not isinstance(document, dict) or document.get("jsonrpc") != "2.0" or document.get("id") != body["id"]:
                raise ToolError("MCP response ID did not match this request")
            if document.get("error") is not None:
                error = document["error"]
                message = error.get("message", "Remote MCP error") if isinstance(error, dict) else "Remote MCP error"
                raise ToolError(str(message)[:1000])
            if not isinstance(document.get("result"), dict):
                raise ToolError("MCP response must contain an object result")
            return document["result"]

    async def initialize(self):
        from . import ToolSpec
        if self.initialized:
            return list(self._specs)
        if not self.config.enabled:
            raise ToolPolicyError("MCP server was not explicitly enabled")
        result = await self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                               "clientInfo": {"name": "Lucifer", "version": __version__}})
        if not isinstance(result.get("capabilities", {}).get("tools"), dict):
            raise ToolError("Configured MCP server did not advertise tool support")
        await self._rpc("notifications/initialized", notification=True)
        specs = []
        cursor = None
        for page in range(4):
            listing = await self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(listing.get("tools"), list):
                raise ToolError("MCP tools/list returned an invalid tool list")
            for tool in listing["tools"]:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not isinstance(tool.get("inputSchema"), dict):
                    raise ToolError("MCP returned a malformed tool schema")
                name = tool["name"]
                if len(name) > 128 or not name:
                    raise ToolError("MCP tool name exceeds its limit")
                if "*" not in self.config.allowed_tools and name not in self.config.allowed_tools:
                    continue
                annotations = tool.get("annotations", {})
                readonly = isinstance(annotations, dict) and annotations.get("readOnlyHint") is True
                if not readonly and not self.config.allow_mutations:
                    continue
                local = "mcp_" + self.config.name + "_" + re.sub(r"[^A-Za-z0-9_-]", "_", name)
                if len(local) > 64:
                    local = local[:53] + "_" + hashlib.sha256(name.encode()).hexdigest()[:10]
                schema = tool["inputSchema"]
                if schema.get("type") != "object":
                    raise ToolError("MCP tool input schemas must have an object root")
                description = f"Configured remote tool on MCP server {self.config.name}; requires approval. " + str(tool.get("description", ""))[:2000]
                specs.append(ToolSpec(local, description, schema, "network", self.config.name, name))
                if len(specs) > 128:
                    raise ToolError("MCP server exceeds the 128-tool startup limit")
            cursor = listing.get("nextCursor")
            if not cursor:
                self.initialized = True
                self._specs = specs
                return list(specs)
            if not isinstance(cursor, str) or len(cursor) > 2048:
                raise ToolError("MCP pagination cursor is invalid")
        raise ToolError("MCP tool discovery exceeded four startup pages")

    async def call_tool(self, name, arguments, context):
        if not self.initialized or self.config.name not in context.allowed_mcp_servers or not context.approved:
            raise ToolPolicyError("Remote MCP tool is not in this run's approved server scope")
        context.check_cancel()
        import asyncio
        task = asyncio.create_task(self._rpc("tools/call", {"name": name, "arguments": arguments},
                                             output_limit=max(65536, context.max_output_bytes * 2)))
        cancel = asyncio.create_task(context.cancel.wait()) if context.cancel else None
        try:
            if cancel:
                completed, _ = await asyncio.wait([task, cancel], return_when=asyncio.FIRST_COMPLETED)
                if cancel in completed:
                    raise asyncio.CancelledError("MCP call cancelled")
            result = await task
            # MCP output is data, never a new tool instruction or authorization.
            return {"server": self.config.name, "remote_tool": name, "untrusted_result": result}
        finally:
            for item in (task, cancel):
                if item and not item.done():
                    item.cancel()
            await asyncio.gather(*(item for item in (task, cancel) if item), return_exceptions=True)

    async def close(self):
        await self.client.aclose()
