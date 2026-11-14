"""Typed, stateless model protocols used by Lucifer's agent runner.

Adapters are selected from a model's startup capability snapshot. They never
probe routes while handling a turn. Tool fragments stay private until the entire
provider response has completed successfully; transport errors cannot expose an
executable half-call. Provider reasoning is a separate, transient event channel.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
import asyncio
import base64
import copy
import json
import re
from typing import Any
from urllib.parse import urlsplit

import httpx


DIALECTS = ("chat_completions", "anthropic_messages", "responses")
_ENDPOINTS = {"chat_completions": "/chat/completions", "anthropic_messages": "/messages", "responses": "/responses"}
_MAX_EVENT_CHARS = 2 * 1024 * 1024
_MAX_ARGUMENT_CHARS = 1024 * 1024
_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.:-]{0,127}$")


class ProviderError(RuntimeError):
    """An invalid, incomplete or rejected provider turn; no tool is executable."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": copy.deepcopy(self.arguments)}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def strict_json(raw: str):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def _failure(error=None) -> ProviderError:
    """Do not turn a provider's echoed prompt/reasoning into durable errors."""
    details = []
    if isinstance(error, Mapping):
        for key in ("type", "code"):
            value = error.get(key)
            if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", value):
                details.append(f"{key}={value}")
    return ProviderError("Provider stream failed" + (" (" + ", ".join(details) + ")" if details else ""))


def _text(value, label: str, *, optional=False) -> str:
    if value is None and optional:
        return ""
    if not isinstance(value, str):
        raise ProviderError(f"{label} must be text")
    return value


def _arguments(raw) -> dict:
    if isinstance(raw, str):
        if len(raw) > _MAX_ARGUMENT_CHARS:
            raise ProviderError("Tool arguments exceed the response limit")
        try:
            raw = strict_json(raw)
        except (ValueError, RecursionError) as exc:
            raise ProviderError("Tool arguments must be one complete JSON object") from exc
    if not isinstance(raw, dict):
        raise ProviderError("Tool arguments must be a JSON object")
    # Also reject Python callers constructing NaN or unsupported objects directly.
    try:
        strict_json(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProviderError("Tool arguments contain invalid JSON") from exc
    return raw


def canonical_call(call: Mapping) -> dict:
    if not isinstance(call, Mapping):
        raise ProviderError("Tool calls must be objects")
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise ProviderError("Tool function must be an object")
    ident, name = call.get("id"), function.get("name")
    if not isinstance(ident, str) or not ident.strip() or len(ident) > 256:
        raise ProviderError("Tool call ID is missing or invalid")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ProviderError("Tool function name is missing or invalid")
    arguments = _arguments(function.get("arguments", {}))
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False)}}


def canonical_tools(tools) -> list[dict]:
    result, names = [], set()
    for tool in tools or []:
        if not isinstance(tool, Mapping) or tool.get("type", "function") != "function":
            raise ProviderError("Only explicitly registered function tools are supported")
        definition = tool.get("function", tool)
        if not isinstance(definition, Mapping):
            raise ProviderError("Tool definition must be an object")
        name = definition.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ProviderError("Tool names must be valid and unique")
        parameters = definition.get("parameters", definition.get("input_schema", {"type": "object", "properties": {}}))
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise ProviderError("Function tool parameters must be an object schema")
        names.add(name)
        function = {"name": name, "description": str(definition.get("description", "")), "parameters": copy.deepcopy(dict(parameters))}
        if "strict" in definition:
            function["strict"] = bool(definition["strict"])
        result.append({"type": "function", "function": function})
    return result


def canonical_content(content, *, allow_none=False):
    if content is None and allow_none:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ProviderError("Message content must be text or typed text/image parts")
    parts = []
    for part in content:
        if not isinstance(part, Mapping):
            raise ProviderError("Message content parts must be objects")
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": _text(part.get("text"), "Message text")})
        elif kind in ("image_url", "input_image"):
            image = part.get("image_url")
            image = image if isinstance(image, str) else image.get("url") if isinstance(image, Mapping) else None
            if not isinstance(image, str) or not image:
                raise ProviderError("Image parts require a data or HTTP(S) URL")
            parsed = urlsplit(image)
            if parsed.scheme not in ("data", "https", "http"):
                raise ProviderError("Local file URLs are not accepted by the provider adapter")
            if parsed.scheme == "data" and not re.match(r"^data:image/(?:png|jpeg|webp|gif);base64,", image):
                raise ProviderError("Inline images require an image MIME type and base64 data")
            # Fetching policy and file scope belong to the agent service. Never
            # read a file or fetch a remote image from this normalization path.
            parts.append({"type": "image_url", "image_url": {"url": image}})
        else:
            raise ProviderError(f"Unsupported message content part: {kind}")
    return parts


def canonical_messages(messages) -> list[dict]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ProviderError("A provider turn requires messages")
    result, pending, used = [], set(), set()
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ProviderError("Unsupported message role")
        role = message["role"]
        if pending and role != "tool":
            raise ProviderError("Outstanding tool calls require results before the next message")
        out = {"role": role, "content": canonical_content(message.get("content"), allow_none=role == "assistant")}
        if role == "assistant" and message.get("reasoning_content") is not None:
            out["reasoning_content"] = _text(message["reasoning_content"], "Assistant reasoning history")
        if role == "assistant" and message.get("tool_calls"):
            calls = [canonical_call(call) for call in message["tool_calls"]]
            for call in calls:
                if call["id"] in pending or call["id"] in used:
                    raise ProviderError("Historical tool call IDs must be unique")
                pending.add(call["id"])
                used.add(call["id"])
            out["tool_calls"] = calls
        if role == "tool":
            ident = message.get("tool_call_id")
            if not isinstance(ident, str) or ident not in pending:
                raise ProviderError("Tool results must link an outstanding assistant tool call")
            pending.remove(ident)
            out["tool_call_id"] = ident
        elif pending and role != "assistant":
            raise ProviderError("Every assistant tool call requires a linked result before the next message")
        result.append(out)
    if pending:
        raise ProviderError("A provider request cannot contain unanswered historical tool calls")
    return result


def _anthropic_content(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    parts = []
    for part in content or []:
        if part["type"] == "text":
            parts.append(dict(part))
        else:
            image = part["image_url"]["url"]
            if image.startswith("data:"):
                header, data = image.split(",", 1)
                # Validate framing here without decoding large images twice.
                try:
                    base64.b64decode(data, validate=True)
                except ValueError as exc:
                    raise ProviderError("Inline image data is not valid base64") from exc
                source = {"type": "base64", "media_type": header[5:].split(";", 1)[0], "data": data}
            else:
                source = {"type": "url", "url": image}
            parts.append({"type": "image", "source": source})
    return parts


def anthropic_messages(messages, *, include_reasoning=False) -> tuple[str, list[dict]]:
    system, output = [], []
    for message in messages:
        role, content = message["role"], message.get("content")
        if role == "system":
            if not isinstance(content, str):
                raise ProviderError("Anthropic system messages must be text")
            system.append(content)
            continue
        if role == "tool":
            role = "user"
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": _anthropic_content(content)}]
        else:
            blocks = _anthropic_content(content)
            if include_reasoning and role == "assistant" and message.get("reasoning_content"):
                blocks.insert(0, {"type": "thinking", "thinking": message["reasoning_content"]})
            blocks.extend({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                           "input": _arguments(call["function"]["arguments"])} for call in message.get("tool_calls", []))
        if output and output[-1]["role"] == role:
            output[-1]["content"].extend(blocks)
        else:
            output.append({"role": role, "content": blocks})
    return "\n\n".join(system), output


def responses_messages(messages) -> list[dict]:
    output = []
    for message in messages:
        role, content = message["role"], message.get("content")
        if role == "tool":
            output.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
            continue
        if content:
            if isinstance(content, str):
                parts = [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
            else:
                parts = [{"type": "input_image", "image_url": part["image_url"]["url"]} if part["type"] == "image_url"
                         else {"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]} for part in content]
            output.append({"type": "message", "role": role, "content": parts})
        for call in message.get("tool_calls", []):
            output.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                           "arguments": call["function"]["arguments"]})
    return output


@lru_cache(maxsize=32)
def _format_validator(encoded: str):
    """Compile each stable format once; schemas cannot fetch external data."""
    response_format = strict_json(encoded)
    kind = response_format.get("type")
    if kind in ("text", "json_object"):
        return kind, None
    if kind != "json_schema":
        raise ProviderError("Unsupported structured output format")
    wrapper = response_format.get("json_schema", {})
    schema = wrapper.get("schema") if isinstance(wrapper, Mapping) else None
    if not isinstance(schema, Mapping):
        raise ProviderError("json_schema requires a schema object")
    def local_only(node, depth=0):
        if depth > 64:
            raise ProviderError("JSON schema nesting exceeds the limit")
        if isinstance(node, Mapping):
            for key, item in node.items():
                if key in ("$ref", "$dynamicRef") and (not isinstance(item, str) or not item.startswith("#")):
                    raise ProviderError("Only local JSON schema references are allowed")
                local_only(item, depth + 1)
        elif isinstance(node, list):
            for item in node:
                local_only(item, depth + 1)
    local_only(schema)
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.validators import validator_for
        validator = validator_for(schema, default=None) if "$schema" in schema else Draft202012Validator
        if validator is None:
            raise ProviderError("Unsupported JSON Schema dialect")
        validator.check_schema(schema)
        return kind, validator(schema, format_checker=validator.FORMAT_CHECKER)
    except ImportError as exc:
        raise ProviderError("Full schema validation requires Lucifer's jsonschema dependency") from exc
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError("Invalid local JSON Schema format") from exc


def prepare_format(response_format):
    if not isinstance(response_format, Mapping):
        raise ProviderError("response_format must be an object")
    try:
        encoded = json.dumps(response_format, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProviderError("response_format must contain valid JSON") from exc
    if len(encoded) > 262144:
        raise ProviderError("Structured format exceeds the schema size limit")
    return _format_validator(encoded)


def validate_structured(text: str, response_format: dict) -> Any:
    """Validate the complete result locally; remote schema references never fetch."""
    kind, validator = prepare_format(response_format)
    if kind == "text":
        return text
    try:
        value = strict_json(text)
    except (ValueError, RecursionError) as exc:
        raise ProviderError("The model did not return valid structured JSON") from exc
    if not isinstance(value, dict):
        raise ProviderError("Structured output must be a JSON object")
    if validator is not None:
        try:
            validator.validate(value)
        except Exception as exc:
            raise ProviderError("Structured output failed local JSON Schema validation") from exc
    return value


async def sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, dict | str]]:
    event, lines, size = "message", [], 0
    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue
        if not line:
            if lines:
                raw = "\n".join(lines)
                if raw == "[DONE]":
                    yield event, raw
                else:
                    try:
                        value = strict_json(raw)
                    except (ValueError, RecursionError) as exc:
                        raise ProviderError("Provider SSE frame contains invalid JSON") from exc
                    if not isinstance(value, dict):
                        raise ProviderError("Provider SSE frames must contain objects")
                    yield event, value
            event, lines, size = "message", [], 0
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > _MAX_EVENT_CHARS:
                raise ProviderError("Provider SSE frame exceeds the limit")
            lines.append(value)
    if lines:
        raw = "\n".join(lines)
        if raw == "[DONE]":
            yield event, raw
        else:
            try:
                value = strict_json(raw)
            except (ValueError, RecursionError) as exc:
                raise ProviderError("Provider SSE frame contains invalid JSON") from exc
            if not isinstance(value, dict):
                raise ProviderError("Provider SSE frames must contain objects")
            yield event, value


class _Calls:
    def __init__(self):
        self.calls: dict[int, dict] = {}

    def start(self, index, *, ident=None, name=None, arguments=None):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ProviderError("Tool fragment index is invalid")
        if index not in self.calls and len(self.calls) >= 64:
            raise ProviderError("Provider exceeded the per-turn function-call limit")
        state = self.calls.setdefault(index, {"id": "", "name": "", "raw": "", "complete": False})
        if ident is not None:
            ident = _text(ident, "Tool call ID")
            if state["id"] and state["id"] != ident:
                raise ProviderError("A tool call changed ID while streaming")
            state["id"] = ident
        if name is not None:
            name = _text(name, "Tool function name")
            if state["name"] != name:
                state["name"] += name
        if arguments is not None:
            state["raw"] += _text(arguments, "Tool argument fragment")
            if len(state["raw"]) > _MAX_ARGUMENT_CHARS:
                raise ProviderError("Tool arguments exceed the response limit")
        return state

    def finish(self, reason: str, offered: set[str], *, require_blocks=False) -> list[ToolCall]:
        if not self.calls:
            if reason == "tool_calls":
                raise ProviderError("Provider completed a tool turn without any tool calls")
            return []
        if reason != "tool_calls":
            raise ProviderError("Incomplete tool turn: provider did not finish with tool calls")
        output, ids = [], set()
        for index, state in sorted(self.calls.items()):
            if require_blocks and not state["complete"]:
                raise ProviderError("A tool block did not complete")
            call = canonical_call({"id": state["id"], "name": state["name"], "arguments": state["raw"]})
            if call["id"] in ids:
                raise ProviderError("Provider returned duplicate tool call IDs")
            ids.add(call["id"])
            name = call["function"]["name"]
            if name not in offered:
                raise ProviderError(f"Provider requested an unknown or withdrawn tool: {name}")
            output.append(ToolCall(call["id"], name, _arguments(call["function"]["arguments"])))
        return output


class ProviderAdapter:
    """A prepared dialect; no runtime metadata or filesystem access per turn."""
    def __init__(self, dialect: str, snapshot):
        if dialect not in DIALECTS:
            raise ValueError(f"Unknown provider dialect: {dialect}")
        self.dialect = dialect
        self.snapshot = snapshot
        self.profile_revision = snapshot.profile_revision if snapshot else None

    def payload(self, *, model: str, messages: list[dict], tools: list[dict], max_tokens: int,
                temperature: float, thinking: str, thinking_options: dict | None = None,
                tool_choice="auto", response_format=None, compatibility=False) -> dict:
        messages, tools = canonical_messages(messages), canonical_tools(tools)
        history = self.snapshot.capabilities.get("reasoning", {}).get("constraints", {}) if self.snapshot else {}
        if history.get("history_field") != "reasoning_content":
            # Template compatibility does not establish a reasoning-history
            # protocol. Only the immutable startup contract permits forwarding.
            for message in messages:
                message.pop("reasoning_content", None)
        if response_format:
            prepare_format(response_format)
        if self.snapshot:
            if tools and not self.snapshot.supports("tools"):
                raise ProviderError("The loaded model does not advertise function tools")
            images = any(isinstance(m.get("content"), list) and any(p["type"] == "image_url" for p in m["content"]) for m in messages)
            if images and not self.snapshot.supports("vision"):
                raise ProviderError("The loaded model does not advertise enabled vision")
            if thinking != "off" and not self.snapshot.supports("reasoning"):
                raise ProviderError("The loaded model does not advertise selectable reasoning")
            if response_format and not self.snapshot.supports("structured_output"):
                raise ProviderError("The loaded model does not advertise structured output")
        if response_format and self.dialect == "anthropic_messages":
            raise ProviderError("Structured response_format is not supported by this Anthropic Messages adapter")
        if response_format and tools and self.dialect == "chat_completions" and self.snapshot:
            constraints = self.snapshot.capabilities.get("structured_output", {}).get("constraints", {})
            if constraints.get("chat_completions_response_format_with_tools") is False:
                raise ProviderError("This runtime does not support response_format combined with tools")
        options = copy.deepcopy(thinking_options or {})
        effort = options.get("reasoning_effort", "none" if thinking == "off" else thinking)
        if isinstance(tool_choice, dict):
            name = (tool_choice.get("function") or {}).get("name", tool_choice.get("name"))
            if name not in {t["function"]["name"] for t in tools}:
                raise ProviderError("Forced tool choice must name an offered function")
        elif tool_choice not in ("auto", "none", "required"):
            raise ProviderError("Unsupported tool_choice")
        if not tools and tool_choice not in ("auto", "none"):
            raise ProviderError("Forced tool choice requires an offered function")
        payload = {"model": model, "stream": True, "temperature": temperature}
        if self.dialect == "chat_completions":
            payload.update(messages=messages, max_tokens=max_tokens)
            payload.update(options)
            if not compatibility:
                payload.pop("chat_template_kwargs", None)
            if tools:
                payload.update(tools=tools, tool_choice=tool_choice)
            if response_format:
                payload["response_format"] = copy.deepcopy(response_format)
        elif self.dialect == "anthropic_messages":
            system, items = anthropic_messages(messages, include_reasoning=compatibility)
            payload.update(messages=items, max_tokens=max_tokens,
                           thinking={"type": "disabled"} if thinking == "off" else {"type": "enabled"})
            if thinking != "off":
                payload["output_config"] = {"effort": effort}
                # The known compatibility runtime accepts the effort directly;
                # generic Anthropic servers also require a thinking budget.
                payload["thinking"]["budget_tokens"] = min(max_tokens - 1, {"low": 1024, "medium": 2048, "high": 8192, "max": 16384}.get(thinking, 2048))
                if max_tokens < 1025 and not compatibility:
                    raise ProviderError("Anthropic thinking needs an output budget greater than 1024 tokens")
            if system:
                payload["system"] = system
            if tools:
                payload["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in tools]
                payload["tool_choice"] = ({"type": "tool", "name": name} if isinstance(tool_choice, dict)
                                          else {"type": {"auto": "auto", "required": "any", "none": "none"}[tool_choice]})
        else:
            payload.update(input=responses_messages(messages), store=False, max_output_tokens=max_tokens,
                           reasoning={"effort": effort})
            if compatibility and options.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = options["chat_template_kwargs"]
            if tools:
                payload["tools"] = [{"type": "function", **t["function"]} for t in tools]
                payload["tool_choice"] = {"type": "function", "name": name} if isinstance(tool_choice, dict) else tool_choice
            if response_format:
                kind = response_format.get("type")
                fmt = {"type": kind}
                if kind == "json_schema":
                    fmt.update(copy.deepcopy(response_format.get("json_schema", {})))
                payload["text"] = {"format": fmt}
        return payload

    @staticmethod
    def _retry_delay(value: str | None) -> float:
        if not value:
            return 1.0
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                raise ProviderError("Provider rate limit supplied an invalid Retry-After") from None
        if not 0 <= delay <= 10:
            raise ProviderError("Provider rate limited the turn; Retry-After exceeds the retry window")
        return delay

    async def stream(self, client: httpx.AsyncClient, base_url: str, headers: dict, payload: dict,
                     *, response_format=None) -> AsyncIterator[tuple[str, dict]]:
        headers = dict(headers)
        if self.dialect == "anthropic_messages":
            headers.setdefault("anthropic-version", "2023-06-01")
        for attempt in range(3):
            retry = None
            async with client.stream("POST", base_url.rstrip("/") + _ENDPOINTS[self.dialect], headers=headers, json=payload,
                                     timeout=httpx.Timeout(None, connect=10)) as response:
                if response.status_code == 429 and attempt < 2:
                    retry = self._retry_delay(response.headers.get("retry-after"))
                elif response.is_error:
                    raise ProviderError(f"Provider rejected {self.dialect} turn (HTTP {response.status_code})")
                else:
                    parser = {"chat_completions": self._chat, "anthropic_messages": self._anthropic, "responses": self._responses}[self.dialect]
                    offered = {t.get("function", t)["name"] for t in payload.get("tools", [])}
                    buffered, pending_calls = [], []
                    structured = response_format and response_format.get("type") != "text"
                    async with aclosing(parser(response, offered)) as events:
                        async for event, data in events:
                            if event == "tool_call":
                                pending_calls.append(data)
                            elif structured and event == "delta":
                                buffered.append(data["content"])
                            elif event == "finish":
                                if structured:
                                    reason, text = data.get("finish_reason"), "".join(buffered)
                                    if reason == "stop":
                                        validate_structured(text, response_format)
                                    elif reason != "tool_calls":
                                        raise ProviderError("Structured output did not finish successfully")
                                    # A format applies to the final answer, not
                                    # prose beside a complete function call.
                                    if text:
                                        yield "delta", {"content": text}
                                for call in pending_calls:
                                    yield "tool_call", call
                                yield event, data
                            else:
                                yield event, data
                    return
            # Only an explicit HTTP 429 before any stream content can retry.
            # Closing the response first prevents holding a connection while idle.
            if retry is not None:
                await asyncio.sleep(retry)

    async def _chat(self, response, offered):
        calls, finished, reason, terminated = _Calls(), False, None, False
        async for _, item in sse_events(response):
            if item == "[DONE]":
                terminated = True
                break
            if "error" in item:
                raise _failure(item["error"])
            choices = item.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise ProviderError("Agent turns require one response choice")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise ProviderError("Provider response choice is invalid")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ProviderError("Provider delta is invalid")
                if finished and any(delta.get(key) for key in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    raise ProviderError("Provider content arrived after completion")
                for field, event in (("content", "delta"), ("reasoning_content", "reasoning")):
                    value = delta.get(field)
                    if field == "reasoning_content" and not value:
                        value = delta.get("reasoning")
                    if value is not None:
                        value = _text(value, field)
                        if value:
                            yield event, {"content": value}
                fragments = delta.get("tool_calls", [])
                if not isinstance(fragments, list):
                    raise ProviderError("Tool fragments must be an array")
                if finished and fragments:
                    raise ProviderError("Tool fragments arrived after completion")
                for fragment in fragments:
                    if not isinstance(fragment, dict) or fragment.get("type", "function") != "function":
                        raise ProviderError("Invalid function-call fragment")
                    function = fragment.get("function", {})
                    if not isinstance(function, dict):
                        raise ProviderError("Function fragment must be an object")
                    calls.start(fragment.get("index"), ident=fragment.get("id"), name=function.get("name"), arguments=function.get("arguments"))
                if choice.get("finish_reason") is not None:
                    value = _text(choice["finish_reason"], "Finish reason")
                    if finished and value != reason:
                        raise ProviderError("Provider changed its finish reason")
                    reason, finished = value, True
            if item.get("usage") is not None:
                if not isinstance(item["usage"], dict):
                    raise ProviderError("Provider usage must be an object")
                yield "usage", item["usage"]
        if not finished and not terminated:
            raise ProviderError("Provider stream ended without a completion marker")
        reason = reason or "stop"
        for call in calls.finish(reason, offered):
            yield "tool_call", call.as_dict()
        yield "finish", {"finish_reason": reason}

    async def _anthropic(self, response, offered):
        calls, stopped, reason, blocks = _Calls(), False, None, {}
        async for name, item in sse_events(response):
            if item == "[DONE]":
                continue
            kind = item.get("type", name)
            if kind == "error" or "error" in item:
                raise _failure(item.get("error"))
            if kind == "content_block_start":
                index, block = item.get("index"), item.get("content_block")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0 or not isinstance(block, dict) or index in blocks:
                    raise ProviderError("Provider content block is invalid or repeated")
                blocks[index] = block.get("type")
                if block.get("type") == "tool_use":
                    initial = block.get("input", {})
                    state = calls.start(index, ident=block.get("id"), name=block.get("name"))
                    state["initial"] = initial
                elif block.get("type") == "text" and block.get("text"):
                    yield "delta", {"content": _text(block["text"], "Initial text block")}
                elif block.get("type") == "thinking" and block.get("thinking"):
                    yield "reasoning", {"content": _text(block["thinking"], "Initial thinking block")}
            elif kind == "content_block_delta":
                index, delta = item.get("index"), item.get("delta")
                if not isinstance(index, int) or index not in blocks or blocks[index] == "closed" or not isinstance(delta, dict):
                    raise ProviderError("Provider sent a delta outside a content block")
                if delta.get("type") == "text_delta":
                    if blocks[index] != "text":
                        raise ProviderError("Text delta arrived in another content block")
                    yield "delta", {"content": _text(delta.get("text"), "Text delta")}
                elif delta.get("type") == "thinking_delta":
                    if blocks[index] != "thinking":
                        raise ProviderError("Thinking delta arrived in another content block")
                    yield "reasoning", {"content": _text(delta.get("thinking"), "Thinking delta")}
                elif delta.get("type") == "input_json_delta":
                    if blocks[index] != "tool_use":
                        raise ProviderError("Tool arguments arrived in another content block")
                    calls.start(index, arguments=delta.get("partial_json"))
            elif kind == "content_block_stop":
                index = item.get("index")
                if not isinstance(index, int) or index not in blocks or blocks[index] == "closed":
                    raise ProviderError("Provider closed an unknown content block")
                if index in calls.calls:
                    state = calls.calls[index]
                    if state["raw"] and state.get("initial"):
                        raise ProviderError("Provider supplied conflicting initial and streamed tool arguments")
                    if not state["raw"]:
                        state["raw"] = json.dumps(_arguments(state.get("initial", {})), ensure_ascii=False)
                    state["complete"] = True
                blocks[index] = "closed"
            elif kind == "message_delta":
                delta = item.get("delta", {})
                if not isinstance(delta, dict):
                    raise ProviderError("Provider message delta is invalid")
                if delta.get("stop_reason"):
                    reason = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(delta["stop_reason"], delta["stop_reason"])
                if item.get("usage") is not None:
                    yield "usage", dict(item["usage"])
            elif kind == "message_start":
                message = item.get("message", {})
                if not isinstance(message, dict):
                    raise ProviderError("Provider message start is invalid")
                if isinstance(message.get("usage"), dict):
                    yield "usage", message["usage"]
            elif kind == "message_stop":
                stopped = True
                break
        if not stopped or not reason:
            raise ProviderError("Anthropic stream ended without message completion")
        for call in calls.finish(reason, offered, require_blocks=True):
            yield "tool_call", call.as_dict()
        yield "finish", {"finish_reason": reason}

    async def _responses(self, response, offered):
        calls, completed, reason, items = _Calls(), False, None, {}
        async for name, item in sse_events(response):
            if item == "[DONE]":
                continue
            kind = item.get("type", name)
            if completed:
                raise ProviderError("Responses events arrived after the terminal response")
            if kind in ("error", "response.failed") or "error" in item and item["error"]:
                raise _failure(item.get("error", (item.get("response") or {}).get("error")))
            if kind == "response.output_item.added":
                index, value = item.get("output_index"), item.get("item")
                if not isinstance(value, dict) or index in items:
                    raise ProviderError("Responses item is invalid or repeated")
                items[index] = value
                if value.get("type") == "function_call":
                    calls.start(index, ident=value.get("call_id"), name=value.get("name"), arguments=value.get("arguments", ""))
                elif value.get("type") in ("custom_tool_call", "mcp_call", "computer_call", "local_shell_call"):
                    raise ProviderError("Only registered function calls can be executed")
            elif kind in ("response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                yield "delta" if kind == "response.output_text.delta" else "reasoning", {"content": _text(item.get("delta"), "Responses text delta")}
            elif kind == "response.function_call_arguments.delta":
                index = item.get("output_index")
                if index not in calls.calls:
                    raise ProviderError("Responses tool arguments arrived before the call")
                calls.start(index, arguments=item.get("delta"))
            elif kind == "response.function_call_arguments.done":
                index = item.get("output_index")
                if index not in calls.calls:
                    raise ProviderError("Responses tool completion has no call")
                state = calls.calls[index]
                value = _text(item.get("arguments"), "Complete tool arguments")
                if state["raw"] and state["raw"] != value:
                    raise ProviderError("Responses tool arguments changed at completion")
                state["raw"] = value
            elif kind == "response.output_item.done":
                index, value = item.get("output_index"), item.get("item")
                if not isinstance(value, dict) or index not in items:
                    raise ProviderError("Responses closed an unknown item")
                if value.get("type") == "function_call":
                    state = calls.calls.get(index)
                    if state is None or value.get("status") != "completed":
                        raise ProviderError("Responses returned an incomplete function call")
                    if state["id"] != value.get("call_id") or state["name"] != value.get("name"):
                        raise ProviderError("Responses function identity changed at completion")
                    arguments = _text(value.get("arguments"), "Complete tool arguments")
                    if state["raw"] and arguments != state["raw"]:
                        raise ProviderError("Responses function arguments changed at completion")
                    state.update(raw=arguments, complete=True)
            elif kind in ("response.completed", "response.incomplete"):
                value = item.get("response")
                if not isinstance(value, dict):
                    raise ProviderError("Responses completion must contain a response object")
                if value.get("error"):
                    raise _failure(value["error"])
                if completed or value.get("status") not in ("completed", "incomplete"):
                    raise ProviderError("Responses terminal state is invalid or repeated")
                completed = True
                reason = "length" if kind == "response.incomplete" or value.get("status") == "incomplete" else "tool_calls" if calls.calls else "stop"
                if isinstance(value.get("usage"), dict):
                    yield "usage", value["usage"]
        if not completed:
            raise ProviderError("Responses stream ended without a completion marker")
        for call in calls.finish(reason, offered, require_blocks=True):
            yield "tool_call", call.as_dict()
        yield "finish", {"finish_reason": reason}


def prepare_adapters(snapshot) -> dict[str, ProviderAdapter]:
    return {dialect: ProviderAdapter(dialect, snapshot) for dialect in DIALECTS if snapshot and snapshot.supports(dialect)}


def conservative_count(messages: list[dict], tools: list[dict], response_format=None) -> dict:
    """Estimate text/template cost separately from a per-image patch reserve.

    Image URLs and base64 are transport data, not text tokens. The fixed 65,536
    token image reserve is deliberately conservative for this local profile, but
    is not a verified universal bound for unknown vision encoders. The provider
    remains authoritative for image expansion and context-overflow rejection.
    Native exact counting is used only for an advertised matching dialect.
    """
    normalized = canonical_messages(messages)
    images = 0
    # canonical_messages constructs fresh content dictionaries. Replace only
    # that private copy; callers still need the actual image bytes for inference.
    for message in normalized:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part["type"] == "image_url":
                    images += 1
                    part["image_url"] = {"url": "<image-payload>"}
    encoded = json.dumps({"messages": normalized, "tools": canonical_tools(tools), "response_format": response_format}, ensure_ascii=False, allow_nan=False)
    return {"input_tokens": len(encoded.encode("utf-8")) + 1024 + len(normalized) * 32 + images * 65536,
            "method": "conservative_estimate", "estimated": True}
