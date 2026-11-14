# Agent interface findings

Research snapshot: 2026-11-12. This document retains the protocol audit and its
historical boundaries. Lucifer 0.2 implemented the agent, all three typed provider
adapters, scoped tools and user-selected Direct PC access. Version 0.3 adds
multi-turn corrections/pause/resume, session-memory reasoning retention, dynamic
output allowances, continuation and conversation-derived experts. Current
implementation and workflow are in [AGENT_PLAN.md](AGENT_PLAN.md) and
[SESSIONS_AND_EXPERTS.md](SESSIONS_AND_EXPERTS.md); measured release evidence is in
[VALIDATION.md](VALIDATION.md). Historical gaps and proposals below are labeled.

## Confirmed installation and live checks

The installed Strata Control is `0.1.59.1`; the DeepSeek engine is `0.1.59`, distributed with Strata `0.2.13`. Public application/model name: **Lucifer**. The technical checkpoint identity and license remain in configuration and provenance records.

At the protocol-probe snapshot, the generated configuration at `var/strata/data/strata/install/strata-deepseek-v4.1-flash-ist-das-2.00bpw.json` selected an **8,192-token** context, GPU 0, vision on, DSpark auto and a 900 MiB GPU reserve. The prepared pack's `ds4.txt` records `max_position_embeddings 1048576`; this is a native capability, not proof of allocation. The subsequent application requirement is **the maximum context allowed by the loaded model**, resolved dynamically, rather than a fixed 8K or 100K setting. Use `effective_context_length` for the observed active window, `native_context_length` for independent model metadata, and `context_length: auto` for the requested policy. Agent prompts, tool schemas, tool results and output must share the observed active window; the 8K probes do not establish that a 1M allocation succeeds on this hardware.

Three serialized, synthetic, local API probes completed after the runtime was ready. No real tool, filesystem operation or external network operation was executed by the model:

| Probe | Observed result |
| --- | --- |
| Forced `lookup_public_fact` function, streamed | Returned a call ID, function name and JSON argument fragments. Assembly produced `{"topic":"capital_of_italy"}` and final `finish_reason: "tool_calls"`; 29 completion tokens, 14.850 seconds. |
| Tool-result continuation | An assistant message with `content: null` and `tool_calls`, followed by `role: "tool"` with the matching `tool_call_id` and a static `{"answer":"Rome"}` result, produced a final answer naming Rome; `finish_reason: "stop"`, 4.043 seconds. |
| JSON Schema response | With a required `answer`, enum `Rome`, and `additionalProperties: false`, returned `{"answer":"Rome"}`; independently valid, 7.114 seconds. This does not prove server schema enforcement. |

All used `reasoning_effort: "none"`, `chat_template_kwargs.enable_thinking: false`, small output budgets and temperature 0. Captured public synthetic responses are in `var/validation/lucifer-live/agent-probes.json`.

**Important:** the bundled Strata Python has no `jsonschema` package (`importlib.util.find_spec("jsonschema") is None`). Its JSON Schema implementation therefore falls back to checking only that the answer is one JSON object. A valid example is evidence of model compliance on that example, not a schema guarantee.

## Confirmed Strata protocol

The source paths below are under `var/strata/data/terminal/0.1.59.1/`.

| Capability | Confirmed contract and limits | Evidence |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions`, streaming and nonstreaming. Function tools use `tools[].type=function`, `function.name`, `description`, `parameters`. Tool calls and results are separate protocol messages. | `serve/frontend.py:495`, `:577`; `serve/server.py:8558`; live probes above. |
| Streamed tool calls | Accumulate by `tool_calls[].index`; ID/name can appear at the start and argument strings arrive in separate fragments. Execute only complete validated calls after successful completion, never fragments. Complete calls finish as `tool_calls`; truncation can finish as `length`. | `serve/server.py:7100`, `:7194`. |
| Tool choice | Accepts `auto`, `none`, `required` and a named function object. Forcing is a prompt-prefix mechanism, not constrained decoding. Unsupported choices or absent names can fall back to `auto`. | `serve/frontend.py:896`, `:919`. |
| Tool schemas | Schemas are supplied to the template/parser. The parser performs conversions and recovery; neither `strict: true` nor declared tool names replace executor-side validation and policy. Historical tool names can be reintroduced by the server, even after omission from a later request. | `serve/frontend.py:495`, `:1118`; `serve/server.py:9764`. |
| Anthropic compatibility | `POST /v1/messages` supports `tool_use`, `tool_result`, thinking/text blocks and streamed `input_json_delta` fragments. `POST /v1/messages/count_tokens` renders/tokenizes without inference. Source-confirmed; not live-probed in this audit. | `serve/frontend.py:677`; `serve/server.py:7343`, `:8822`, `:8831`. |
| Responses compatibility | A route exists and is advertised, but it is stateless: stored response retrieve/delete/cancel routes return errors. A typed adapter may use current-request output and tool events; it must not assume a remote durable agent. | `serve/server.py:8199`, `:8672`. |
| Reasoning | OpenAI output separates `delta.reasoning_content` from `delta.content`. Reasoning effort supports none/low/medium/high/max or numeric DeepSeek effort. Provider-emitted tokens can be displayed transiently in their own collapsible view, separate from final answers and durable records. | `serve/frontend.py:154`, `:577`; `serve/server.py:7140`. |
| Structured JSON | `response_format` supports `json_object` and object-root `json_schema`. It prompts and postvalidates; there is no grammar decoder. Structured streams are buffered. Chat Completions rejects structured `response_format` combined with tools/MCP. Full schema checking requires optional `jsonschema`, absent here. | `serve/structured.py:1`, `:75`, `:123`, `:237`; `serve/server.py:7306`, `:8567`. |
| Context and usage | Prompt overflow is rejected rather than silently truncating; output may be shortened to remaining context by default. Final OpenAI chunks carry prompt/completion/total/cached token usage and timings. Use token counts, not visible text length, for budgets. | `serve/server.py:5833`, `:7162`. |
| Cancellation and errors | Closing the HTTP request is observed by a socket watcher, including before the first output token, and requests engine cancellation. Midstream errors can arrive as SSE error objects. A transport ending without a completion marker is not success. Queue pressure can return 429 with `Retry-After`. | `serve/server.py:7646`, `:8180`, `:8618`. |
| Vision | The installed pack and config enable the native DeepSeek image path. Chat Completions accepts image content parts; Anthropic accepts image blocks. This audit confirmed configuration/status, not an image inference. Remote-image fetching and local-file access require a separate application policy. | `serve/frontend.py:181`, `:577`; `serve/server.py:5833`; `serve/ds_pictures.py`. |
| Tokenization | The prepared pack contains tokenizer assets and the Messages count route renders/tokenizes without inference. Standalone HTTP `/tokenize` or `/detokenize` routes were not found in this pinned server; their availability remains unknown rather than assumed from another backend. | `serve/server.py:8822`; installed pack and route inspection. |
| MCP | The pinned server has MCP integration and `/mcp` status. It requires configured MCP servers and a per-request `strata_mcp` opt-in. No MCP servers are configured in the current Lucifer profile. An advertised integration is not an enabled tool permission. | `serve/server.py:6835`, `:8038`, `:8593`. |

### DeepSeek template and reasoning caveats

The downloaded pack contains the older DSML spelling. The installed server checks it and substitutes its V4.1 fixture when necessary (`serve/dsml.py:226`, `:234`; `serve/server.py:10185`). V4.1 uses `<｜DSML｜ calls>` with a space. The server translates this into standard tool-call objects. A Lucifer agent should consume those objects rather than parse DSML or use text-pattern execution.

The 2-bit output parser also has a fallback: if an answer stops without a closing thinking delimiter, without visible content and without a tool call, it can promote accumulated reasoning into content (`serve/dsml.py:166`). Merely discarding the `reasoning_content` field cannot prove that all visible content is free of internal analysis. Thinking defaults to off and remains selectable. Test this parser boundary at each supported setting; a transient provider-reasoning view is not a guarantee that every visible answer is free of analysis. Do not claim a universal guarantee from the three short probes.

## Implemented startup capability recognition

`lucifer/capabilities.py` now provides an immutable `CapabilitySnapshot` and a
`CapabilityRegistry`. A snapshot is cached by model ID and stable profile revision;
changed registration and a new model load invalidate only that model. The manager
passes its existing readiness payloads from `/v1/models` and `/v1/status`, plus
independently read native context metadata and the allocated context limit. This
avoids duplicate metadata calls and never treats Strata's `context.native` label
as independent evidence of the model's training limit.

The registry optionally reads OpenAPI once, resolves only local schema references,
and never sends a generation request or executes a tool. Ordinary chat and status
reads use the cached snapshot. Failed or canceled startup discovery is canceled
and reaped before its HTTP client closes. Tests cover concurrent deduplication,
selective invalidation, no per-turn I/O, immutable copies and unknown providers.

The snapshot records `unknown`, `unavailable`, `advertised` or `verified`, with
evidence, endpoints and constraints for all interfaces in the table. Explicit
profile annotations are the lowest-priority evidence, OpenAPI follows, current
startup metadata overrides both, and a separate explicit acceptance ledger can
establish verified behavior. A provider or profile calling itself verified is
still only an advertisement. An actual disabled vision tower wins over a source
annotation of vision support. Unrecognized feature names and route aliases are
not silently mapped onto a known interface.

`configs/models.yaml` supplies pinned-source annotations for the installed Strata
profile, including all three model dialects, streaming tools, reasoning, vision,
structured JSON, usage, request cancellation, token counting and conditional MCP.
The annotations preserve restrictions: stateless Responses, prompt/postvalidation
JSON, no configured MCP servers and no advertised standalone tokenizer endpoint.
They do not upgrade earlier synthetic probes to verification of a changed profile
revision. The project API exposes the cached snapshot through
`GET /models/{id}/capabilities`, model metadata and the startup bootstrap response.

Availability is not authorization. Tool argument and result validation remain
local requirements even for a provider that advertises strict schemas. The agent
adapters and executor were proposed at that point and are implemented in 0.2.

## Application gaps at audit time — superseded by 0.2

These are application limitations, distinct from native model/server capability:

- `lucifer/schemas.py:29`: messages permit only system/user/assistant roles and nonempty string content. They cannot represent `role: tool`, `tool_call_id`, null assistant content with calls, or image parts.
- `lucifer/schemas.py`, `ChatRequest`: still needs tool schemas, tool choice and structured response formats for an agent protocol; it requires the last message to be a user message. Thinking modes are implemented for ordinary chat; this does not add tool-message support.
- `lucifer/inference.py`, `ModelManager.stream_events`: now preserves separate text/reasoning, usage and finish events. Its request is still a Chat Completions text payload and it does not assemble or expose tool-call fragments. `stream` remains an intentionally text-only convenience path. Agent adapters must add typed tool contracts and the other dialect transports without changing that compatibility helper.
- `lucifer/chat.py:106`: reconstructs history with only role/content. A tool-only assistant response is considered an empty failure. It commits an ordinary completed text turn, not an agent's sequence of calls/results.
- `lucifer/api.py`, `/chat` and `/chat/stream`: now expose separate reasoning, usage and finish events alongside text/meta/done/error. They still lack agent run lifecycle, scope decisions, tool events and explicit run cancellation.
- `lucifer/client.py`, `APIClient.stream_chat`: transports named SSE events. The TUI has a persistent thinking selector and total-generation budget controls, and renders actual provider reasoning in a separate collapsible transient panel. It still has no agent progress or scope-decision surface. `ChatService` records only the final answer, not the raw reasoning channel.
- `lucifer/store.py`: already provides transactional SQLite document records and an audit ledger. `ChatService` already separates transient and consented durable conversations. These are reusable foundations, not an existing agent state machine.
- `lucifer/learning.py:39`, `:49`: training uses reviewed, consented user/final-answer pairs and strips tagged thinking. Untagged parser fallback text cannot be identified by that regex. Agent tool traces and reasoning must not become training examples automatically.

## Privacy settings confirmed from source

Strata independently defaults to keeping displaced conversations across runs. This is separate from Lucifer's logging consent:

- `serve/server.py:4250` defaults `STRATA_DS4_SLOT_KEEP=1`, with a 2 GB slot-folder cap and a folder beside the run log.
- Upstream `src/dsv41/ds4_main.cpp:561` parses the flag numerically. `STRATA_DS4_SLOT_KEEP=0` sets `slot_max=0`; disk-folder startup/read (`:875`) and shutdown flush (`:1986`) are both guarded by `slot_max > 0`. This disables that persistent conversation/KV mechanism. It does not remove pre-existing slot files.
- `STRATA_DS4_AUTO_PIN` controls the RAM-only shared system/tool-prefix cache (`serve/server.py:6007`; upstream `ds4_main.cpp:791`). It need not be disabled to prevent disk persistence.
- **Unset `STRATA_DEBUG`**, rather than setting it to `0`: Python treats any nonempty value as enabled and can dump raw generated tokens (`serve/server.py:6644`, `:6821`).
- API content capture defaults off; keep config `api_monitor` false/absent and do not pass `--api-monitor` (`serve/server.py:4782`, `:10278`). A later debug mode must explicitly disclose content capture and retention.

The runtime owner was given these facts for the combined restart. Application of settings and cleanup verification belong in the runtime validation record; this document does not assert they have already been applied.

Final application validation subsequently confirmed those privacy flags, no
recreated slot directory, default autoload, live registry operations, and all five
thinking modes. The runtime allocated the source pack's full 1,048,576-token limit
and answered short prompts. No full-window prefill was benchmarked. See
`var/validation/lucifer-live/default-verification.json` and
`thinking-context-verification.json` for the final records.

## Original proposed boundary — implemented and extended in 0.2

Use a small asynchronous agent loop with typed Chat Completions, Anthropic Messages and stateless Responses adapters selected from the cached startup snapshot. Keep tool schemas, policy decisions and execution in Lucifer; leave model prompt encoding and DSML translation to Strata. Begin with workspace-confined read-only tools (bounded file listing, file reading and text search), one model request at a time, bounded rounds/time/tokens/output, and no arbitrary shell or external network tool. Context budgets must follow the loaded model's observed limit. Thinking off/low/medium/high/max is a per-request preference; thinking and final output share the generation budget. Actual provider-emitted reasoning may be shown in a collapsible transient view and remains outside durable history, exports and training targets.

For each completed response: assemble all fragments; validate call IDs, current registry membership, JSON object arguments and the full local schema; resolve workspace paths and reject traversal/symlink escape; check policy; run approved calls under time/size limits; append linked tool results; request the next turn. Treat tool output as untrusted data. A call is not executable because it appeared in text, was recalled from history, or passed the model's parser. Record idempotency so retries cannot repeat a completed mutation.

Add writes/execution only in a later reviewed phase, with explicit policy and review of concrete arguments, paths and effects when further approval is needed. Suggested application interfaces: create/get/cancel agent runs, list typed tools, and resolve a specific pending scope decision; an SSE run event stream for phase, tool start/result, required decision, final answer, usage and terminal error. Persist minimal run/audit state under existing consent rules; never persist the raw reasoning channel. TUI status should show observable phase and tool activity, with only actual provider tokens in the optional reasoning panel.

Acceptance should cover real two-step read-only tasks, fragmented/multiple calls, wrong argument types and extra keys, unknown/withdrawn tools, malformed/truncated calls, duplicate call IDs, workspace escapes, prompt injection in file/tool output, context exhaustion, cancellation before first token and during a tool, disconnect/restart recovery, denied/expired approvals, error/usage propagation, and absence of reasoning in logs/exports/training artifacts. Current probes establish protocol plumbing only; they do not establish general agent competence.

## Public primary sources

- [Strata 0.2.13 DeepSeek interface and context documentation](https://github.com/Niko1221/Strata/blob/v0.2.13/docs/DEEPSEEK41.md)
- [Strata server source at the released tag](https://github.com/Niko1221/Strata/blob/v0.2.13/serve/server.py)
- [Strata DeepSeek DSML adapter](https://github.com/Niko1221/Strata/blob/v0.2.13/serve/dsml.py)
- [Strata native conversation-slot implementation](https://github.com/Niko1221/Strata/blob/v0.2.13/src/dsv41/ds4_main.cpp)
- [Official DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): native context and the maintained encoding/reference toolkit; model inference, HTTP transport and tool execution remain caller responsibilities.
- [Official DeepSeek recipe toolkit](https://github.com/deepseek-ai/deepseek-recipe), linked by that model card. This audit did not install or evaluate the toolkit; the existing Strata adapter already supplies the relevant wire protocol.
