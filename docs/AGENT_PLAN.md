# Lucifer agent implementation and release plan

Status: initial agent delivered in **0.2.0**, extended with multi-turn sessions
and conversation-derived experts in **0.3.0**, followed by the context-aware
curriculum and scoped memory pipeline in **0.4.0**. The user's instruction to implement
this plan approved the agent work. Their subsequent request for direct PC
interaction adds an explicit `computer` access mode alongside confined workspace
modes. The session extension retains emitted thinking in RAM across pauses and
turns, as the user explicitly selected.

## Delivered behavior

Lucifer completes model → typed tool → model missions inside a selected workspace
or, when explicitly selected, directly on the host PC. Missions have no default
time, round, tool-call-count or cumulative-token limit. Native context and bounded
tool inputs, outputs and process lifetime still apply. The Cyberpunk 2077-inspired
terminal exposes goal entry, scope, thinking, dialect, live tool activity, concrete
approvals, pause/resume, corrections and final answers. F3 opens the
interactive adapter cortex inspired by the user's lowercase Colibri checkout.

The assistant identity remains Lucifer. Checkpoint sources, runtime identifiers
and licenses remain accurate in technical provenance. The model supplied at load
time determines native context and advertised interfaces; no checkpoint name is
built into agent execution.

## Milestones and implementation evidence

| Original milestone | Implementation | Evidence |
| --- | --- | --- |
| 1. Startup negotiation and typed adapters | Immutable cached capability snapshots; startup-prepared Chat Completions, Anthropic Messages and stateless Responses adapters; complete tool calls, reasoning, usage and completion/error events | `capabilities.py`, `providers.py`, `inference.py`; capability/provider/inference tests; real two-step tasks through all three dialects |
| 2. Read-only agent loop | Scoped list, read, literal search and optional image tools; linked tool-result continuation; context enforcement and optional explicit rounds/calls/total-token/time bounds | `agent.py`, `tools/`; agent/tool/API tests; actual source discovery with file/line citations |
| 3. Terminal run controls | Multi-turn mission mode, thinking/budget controls, tool timeline, stream cursor/reconnect, concrete approval dialog, correction while busy, stop/resume and final artifact links | `tui.py`, `client.py`; headless integration and wide/compact visual checks |
| 4. Scoped edits and execution | Atomic hashed create/edit/delete, mkdir, isolated commands, explicit Direct PC host commands/files, pre-effect ledger and result reuse | `tools/workspace.py`, `tools/sandbox.py`, `agent.py`; real workspace edit, real bubblewrap execution, approved host command outside the workspace, replay/cancellation tests |
| 5. Memory and consolidation | Consented completed goal/final-answer pairs use existing reviewed-feedback/sleep pipeline; explicit approved-summary retention; source deletion revokes agent exports and datasets | `agent.py`, `chat.py`, existing learning service; actual saved agent result → positive review → scoped sleep dry run → deletion/revocation |
| 6. Broader provider acceptance | Known/unknown feature gates, optional local image input, startup-configured MCP tools using the same registry and policy | Actual native image response; provider/tool contract tests; MCP startup/transport/allowlist/lifecycle tests and disabled-by-default example config |
| 7. Session and expert extension | Same-ID follow-ups/corrections, RAM reasoning retention, optional automatic goal stop, output continuation, content-derived expert proposals, verified fresh LoRA training and multi-parent source replay | `agent.py`, `experts.py`, `learning.py`, `train_worker.py`, TUI/client/cortex and API contracts; session, privacy, artifact and lineage tests |
| 8. Context curriculum and memory | Task/context recovery, bounded draft proposals, explicit verification, grouped training/held-out partitions, scoped memory reconciliation, hash-bound human review and minimum-data gates | `curriculum.py`, `memory.py`, learning/API integration; TUI review/hash/race and heartbeat tests |

Implementation files are under `lucifer/`. Tests are under `tests/`. Runtime
acceptance records are in `var/validation/agent-live/`; they are local validation
artifacts and excluded from distribution. Detailed historical protocol research
remains in `AGENT_INTERFACE_FINDINGS.md`.

## Startup and provider contracts

Model initialization reuses readiness `/v1/models` and `/v1/status` metadata and
explicit profile declarations. Actual disabled metadata overrides profile hints.
Unknown features stay unknown. Optional OpenAPI inspection happens once when
requested. A model/profile load invalidates its own snapshot and prepares its
advertised dialect adapters. No ordinary turn discovers capabilities, reloads
metadata files or enumerates GPUs.

Provider adapters normalize system/user/assistant/tool history, nullable assistant
content, call IDs, argument fragments, image parts, usage and finish reasons.
Calls are emitted only after a complete successful response. Malformed JSON,
duplicate argument keys, incomplete calls, truncation and a transport ending
without completion cannot produce executable fragments. Historical call IDs
reused by a model get fresh wire aliases while the executor preserves original
operation identity. Pre-stream 429 retries are bounded and respect Retry-After;
a partly emitted response is never blindly replayed.

All three advertised dialects use one tool registry and execution policy. Strata
translates DSML; Lucifer consumes typed calls, not command-looking prose. Responses
is stateless: the agent does not depend on stored remote response IDs, retrieval
or remote cancellation APIs. Closing the owned HTTP request signals cancellation.

Structured JSON uses full local validation with the declared JSON Schema dialect,
format checks, strict JSON parsing and local references only. Strata's optional
vendor validator is not trusted. Tool rounds omit response_format; if needed a
final no-tools formatting turn is validated locally. Unsupported format/dialect
combinations are rejected before any effects. The current Anthropic compatibility
interface does not advertise a structured response_format field.

Native image input is opt-in, restricted to bounded local PNG/JPEG/WebP data under
the selected file policy. Encoded image bytes remain transient. No remote image is
fetched implicitly. Current primary-model acceptance identified the red fixture
image correctly. Other supplied models still require their own advertised vision
support and acceptance evidence.

A dedicated `capture_screen` tool is available only in Direct PC mode, only for
image-enabled runs, and only when a desktop screenshot utility is present. It
requires explicit approval and returns a transient PNG; it does not write a
durable screenshot file.

## Context, output and optional run bounds

The loaded model's observed active context governs each complete request, including
system text, tool schemas, tool results, media and reserved output. Native and
allocated context remain separate metadata. The primary model has a native and
allocated window of 1,048,576 tokens; short tasks, not a full-window prefill, were
validated. No fixed 8K or 100K context limit was introduced.

Anthropic token counting uses the advertised native count route with its actual
rendered request. Other dialects use a clearly labeled conservative estimate;
base64 transport bytes are not mistaken for text tokens. Images receive an
explicit patch reserve, whose adequacy still depends on the provider's image
encoding. Backend overflow rejection remains authoritative. Counting changed
requests is necessary; capability recognition and stable schema compilation are
not repeated.

The initialized model record also carries `max_output_tokens` and
`output_limit_source`. Runtime output metadata takes precedence, then an explicit
profile allowance, then active context as the fallback. A profile allowance is
not a claim of a known native output ceiling. The TUI offers `/tokens N|auto|max`
without a global 32,768-token ceiling. Automatic agent budgets increase with the
selected thinking level; every actual request reserves output inside its current
active context.

When enabled, `continue_output` preserves an answer truncated at the generation
budget and requests the next chunk with the retained model/tool history. Thinking
and final output share the request budget, and each chunk consumes remaining
context. Truncated tool calls and structured output are not blindly continued.
The optional `max_continuations` can bound continuation count.

When the active context is full, the agent automatically compacts older messages
into a bounded model-generated working summary. The summary preserves the current
goal and constraints, working project state, completed tool results, fundamental
repository/docs facts and unresolved next steps; transient chatter, raw reasoning
and redundant tool output are omitted. Compaction is repeated in bounded chunks
until the next request fits, and the summary is retained in the session and
checkpoint. `context_compaction: false` can disable it explicitly.

Each run pins model ID, profile revision and active adapter. Pins are checked
before model turns, tools and effects after approval waits. Round, call,
cumulative-token and wall-clock bounds default to `null`; API clients can request
explicit limits. File bytes/lines, search results, command output and command
process lifetime retain their separate limits. One serving slot handles one
active agent segment at a time; conflicting model, chat and learning operations
are blocked while a segment is active. Paused and awaiting-input sessions release
that slot while retaining their pins and history. A changed profile or adapter
requires a new mission before execution can continue.

## Multi-turn session controls

`POST /agent/runs/ID/input` adds a follow-up or correction to the same mission.
With `interrupt: true`, the active request is stopped and cleanup completes
before the next segment starts. Completed tool results are preserved; pending
calls receive explicit interrupted/uncertain results without invoking tools.
`POST /agent/runs/ID/stop` pauses owned work. `POST /agent/runs/ID/resume` continues
an unfinished goal, optionally with guidance and updated limits. A final answer
can be followed by another input under the same run ID.

`stop_when_done: true` is the default and marks a final answer `completed`.
Setting it false marks the session `awaiting_input`; this is quiescent, not an
automatic generation loop. Both remain resumable through a new user message.
The TUI has separate AUTO STOP ON GOAL and AUTO CONTINUE OUTPUT controls, a busy
CORRECT input, STOP/Esc and RESUME. History / F5 lists retained sessions, saved
checkpoints, public transcripts and unavailable private runs on demand.
`/agent open ID` reads the available session without executing it. A valid
checkpoint can continue only through explicit input/resume; prior concrete
approvals expire and completed results are reused. `/session new` selects a
fresh conversation and mission.

Event sequences increase across segments and carry `segment` and `session_turn`.
The UI retains one thinking/answer panel per segment and ignores the old segment's
completion when a correction already selected a newer one. Workspace, access,
dialect, image/MCP scope and command grant stay pinned; changing the visible scope
starts a new mission with conversation history instead of silently widening an
existing session. Thinking, output and stop-policy options can change per turn.

## Tool and execution boundaries

Tools use full JSON argument validation, current-registry membership and explicit
effects. Normal scopes use pinned directory descriptors and no-follow traversal;
symlink/hardlink escapes, outside paths and parent traversal are rejected. Absolute
paths lexically inside the workspace are normalized to the same identity as their
relative aliases. Cache keys do not resolve away symlink checks.

READ inspects the workspace. WRITE adds file changes with required current SHA-256
preconditions and atomic replacement. SANDBOX (`execute`) adds actual Linux
bubblewrap isolation with bounded processes, files, environment and output; there
is no silent fallback when namespace setup is unavailable. DIRECT PC (`computer`)
was added at the user's request: it permits host paths and direct host argv with
the backend user's OS access. The TUI clearly identifies that mode. A model or
untrusted tool result cannot change access mode.

Commands normally require approval of concrete arguments and effects. The user
can explicitly select AUTO CMDS for automatic commands within that run's selected
mode. This does not silently authorize MCP effects. Cancellation terminates only
owned process groups, including descendants that ignore TERM, and bounds pipe
draining. Previously running user applications are not killed.

MCP is disabled unless a server is explicitly configured at startup using
`LUCIFER_MCP_CONFIG`. Transport, tool schemas and allowlists are negotiated once;
credentials refer to environment variables. A run must select ready server names.
Converted MCP tools share argument validation, concrete approval, bounds and replay
protection with local tools. Source-advertised native MCP support alone grants no
network or tool access.

## Repetition, recovery and privacy

HTTP idempotency keys map a repeated submission to the same run. Stream cursors
resume viewing without resubmitting a goal. Before an effect, an opaque operation
journal is written. Completed results are looked up before checking the now-changed
edit/delete precondition, so retries cannot repeat successful mutations. Reads
reuse results only while pinned-source metadata and the run's write revision are
unchanged. Metadata snapshots detect ordinary external edits; they are not a
cryptographic proof against deliberately preserved filesystem timestamps. Actual
mutation preconditions always hash current bytes.

Input and resume controls also accept an idempotency key scoped to the retained
session and control kind. The client generates a UUID per logical control call.
Matching concurrent or delayed retries reuse the original transition; a retry
after a later manual stop cannot restart work. A key reused with different
content, options or limits is rejected. Control receipts remain in session RAM
and the client adds no automatic transport retry.

A restarted backend marks unfinished runs interrupted and unfinished operations
uncertain. It never automatically resumes those effects. Raw provider reasoning,
image bytes, tool arguments and tool result bodies are not stored in durable run
or event records. Private goals and answers stay in memory; consented completed
pairs enter the existing chat store. Actual reasoning and full event/history
content remain in session RAM across pauses, corrections, follow-ups and UI
reconnects. A provider's initialized history contract controls whether its native
reasoning field can be forwarded on a later request. Reasoning is never converted
into a fabricated narrative. The collapsible UI retains every segment until the
user selects another conversation. It is excluded from exports, durable logs,
memory summaries, feedback and training targets. Restarting the backend discards
that RAM content. Minimal opaque lifecycle/replay metadata remains durable.

Feedback and training require saved conversation content and the existing
consent/review gates. An explicit remember action retains a user-approved final
summary. Deleting the conversation removes its agent content surface, memory and
associated training artifacts. The primary streamed profile remains inference
only; real LoRA jobs use the separate trainable profile. Vendor disk KV slots stay
disabled and raw debug capture stays unset.

## Conversation-derived experts

The expert service proposes transparent lexical topics from consented, reviewed
examples not already covered by recorded experts. Proposals are metadata and do
not train or activate weights. A requested expert build enters the existing sleep
pipeline and uses a compatible registered trainable base. When only one such
profile exists it can be selected automatically; otherwise the caller selects it.
The primary streamed model remains inference-only.

Real training commits an expert only after validating the PEFT configuration,
LoRA safetensors, finite worker metrics, nonzero optimizer steps and trained
parameter count. The checkpoint SHA-256 and tensor count are recorded. Training
status is separate from measured quality and existing promotion eligibility.
An actual trained expert can fail evaluation and remain ineligible for promotion.

Consolidation resolves selected expert/PEFT adapter IDs, verifies their artifacts,
requires the same registered base/checkpoint and current source consent, then
replays their reviewed source conversations plus any newly selected conversations
into a fresh adapter from the original base. It does not average tensors or
convert adapters across bases. All source parents are recorded in `source_expert_ids`
and drawn in the cortex. New conversations need consent and reviewed feedback
before a subsequent proposal/build can use them. No raw reasoning is a training
target. See `SESSIONS_AND_EXPERTS.md` for the full command and API workflow.

Sleep now defaults to `dynamic: true` and all new eligible reviewed information.
One request selects a compatible base, content-derived topics and compatible
source adapters, then freezes the evaluation suite before training. Configured
regression anchors are supplemented with independent exact task probes for the
implemented question families. Unsupported topics retain partial/unknown coverage;
they cannot acquire fabricated quality evidence or qualify for promotion. The
run may finish `quality_unknown` with a real trained artifact or
`no_new_information` without creating another one. Explicit API jobs can opt out
with `dynamic: false`.

F4 exposes a single SLEEP / CONSOLIDATE action, the actual current stage,
optimizer/evaluation response counts and candidate review. Job progress uses the
existing cached telemetry heartbeat. Known stage fractions are labeled as stage
progress; unknown overall work remains indeterminate. Terminal state and overall
100% are withheld until original inference restoration finishes. A failed restore
remains visible with no 100% claim. Cancellation retains the same cleanup path;
promotion stays manual. Reviewing the candidate reveals its real `E##` cortex
node without a new model load or activation.

Version 0.4 defaults to `context_examples: true`, `min_verified_examples: 3` and
`max_generated_examples: 32`. It recovers consented task context, preserves active
constraints/supersession, drafts bounded proposals and admits only verified
material to grouped partitions. Pending, rejected, duplicate and obsolete
examples stay outside training. Canonical memory keeps source-bound scoped
evidence and withholds conflicted records. Provider reasoning remains excluded.
Consented USER-only literal arithmetic, worked/error-correction tasks and quoted
configuration constraints can seed verified material without assistant feedback;
unreviewed assistant replies never become gold. The default gate requires three
useful verified examples and two distinct semantic task IDs. Literal USER
`Scope: project` declarations set scope; otherwise it stays conversation-bound.
Restricted pure-function tests use source-approved expected cases and grant no
host tool permission.

The same F4 action exposes context recovery, drafting, validation and memory
stages, with verified/train/held-out/pending/rejected/source-group counters.
`needs_review` and `insufficient_verified_data` are explicit stopped outcomes.
The on-demand review dialog displays literal targets and the full content hash;
human decisions send that exact hash and optional complete correction. Stale or
superseded sources fail rather than approving changed content. Reviews create
immutable replacement records; obsolete versions cannot be approved by the UI.
There is no per-heartbeat queue poll or automatic approval. Details and API bounds
are in `CONTEXT_CURRICULUM.md`; `context_examples: false` keeps the 0.3 path.

## Terminal and adapter cortex

AGENT is the single conversation view, with separate CORTEX and SLEEP views and
an on-demand HISTORY menu. Models without advertised tools retain a response-only
transport inside that same view. Markdown answers, literal user
text, transient thinking, live GPU counters and cached registry panels remain.
The old face has been replaced by an original minimal wing/halo monogram.

The cortex projects actual model/adapter nodes and recorded base/parent edges onto
an illustrative folded brain. Rotate, tilt, zoom, pan and select nodes to inspect
status, rank, source job and evaluation. Activation is a separate explicit action
subject to existing adapter gates. Empty/dense registries and compact screens are
supported. No fake activations or measured neural routing are claimed. See
`ADAPTER_MAP.md` for the data and interaction contract.

## Release gate

Acceptance includes actual model→tool→model tasks in all three dialects, scoped
writes, Direct PC approval/execution, local image input, strict structured output,
agent-to-learning consent and deletion, plus controlled cases for malformed or
multiple calls, wrong/extra arguments, unknown tools, duplicate IDs, external file
changes, context compaction and exhaustion, cancellation before output/during tools/approval,
429 handling, partial streams, restarts, denied/expired approvals, escapes,
oversized output, process descendants and absence of raw reasoning in durable
records. Unit tests are complemented by actual OS tools and live-model evidence;
no claim of general agent competence or full-window performance is inferred from
these integration checks.
### Screen capture recovery (14 November 2026)

The interrupted tool-development run added `capture_screen`. It now supports
GNOME/Wayland through the desktop Screenshot portal, in addition to installed
command-line screenshot utilities. Discovery happens once at tool initialization
and does not take a screenshot. This host already provides the portal, system
Python, D-Bus and GLib bindings; no additional package is required.

Capture requires Direct PC scope, image input enabled, and concrete tool approval.
The portal may also ask for desktop consent; denial does not fall back to another
backend. Cursor selection is controlled by the portal. The helper subscribes
before requesting a capture, runs a GLib event loop, and closes only its own
request on timeout/cancellation. Image payloads remain transient and the tool's
private temporary copy is deleted. Any original file returned by the portal is
managed by the desktop; the tool does not delete arbitrary user images.

