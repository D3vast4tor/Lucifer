# Validation — 13 November 2026

## Relocated checkout repair — 13 November 2026

Validation ran from `/home/Lucifer/Soul` after relocating the checkout. The
primary model paths, runtime symlinks, generated Strata profiles and weight
manifest destinations still referenced the former directory. Configuration now
resolves explicit local paths from its own directory. The Strata launcher
repairs local generated paths and links on initialization, and the vLLM wrapper
uses its relocated interpreter directly instead of a stale console-script
shebang. `doctor` reports both local inference runtimes available. The eight-file
offline preparation check reports no pending files; no weights were downloaded.

The repaired source passed **849 core tests**, with one optional Torch/PEFT module
skipped in the core environment and the existing Starlette/httpx deprecation
warning, in **92.36 seconds**. Ruff, Python compilation, dependency checks and
wheel/source builds passed. A separate wheel import from outside the checkout
loaded packaged configuration and passed demo chat and bootstrap checks. A real
PTY demo launched and exited with Ctrl+Q, shutting down its owned backend.

Regression fixes cover model loading beyond the ordinary client read timeout,
malformed approval input without a frontend crash, expert activation during
Sleep, escaped/Unicode tool output byte limits, file-targeted text searches,
partial-read truncation reporting and relocation independent of the current
working directory.

The real primary model initialized its native **1,048,576-token** context and
answered `2+2` with `4`. A private read-only agent retrieved and cited a synthetic
file, and a saved write agent changed exactly its requested synthetic file.
HTTP retries returned the same completed runs without changing the edited file's
timestamp. Saved transcript/checkpoint recovery and private-content exclusion
were verified. Thinking was off for these live checks; reasoning-channel privacy
is also covered by the automated suite. Validation receipts exclude raw
reasoning and user transcripts.

An isolated `.train-venv` was restored without modifying the shared vLLM
environment. Its dependency check and 16 training-specific tests passed. Actual
CPU LoRA completed four optimizer steps; adaptive training resumed an immutable
checkpoint from four to eight successful updates. Real GPU QLoRA on the cached
Qwen3.5-4B checkpoint completed one optimizer step with **5,308,416 trainable
parameters**, finite loss and verified adapter safetensors. No smoke adapter was
registered or promoted. These integration checks establish execution, not model
quality improvement.

The relocated vLLM profile also loaded successfully, advertised its configured
4,096-token context, answered `2+2` with `4`, and unloaded cleanly. All temporary
inference and training workers released their GPU memory after validation.

Evidence is retained under `var/validation/repair-20261113/`, including
`live-agent.json`, `training-result.json`, `adaptive-training-result.json` and
`qwen-training-result.json`, plus `vllm-inference.json`. Historical validation below describes earlier
states and remains subject to its stated hardware limits.

### Existing backend attachment follow-up

The exact command `./lucifer run --color always`, launched from the checkout's
`scripts` directory, initially exposed an additional failure: an older backend
on port 8000 retained its former file-based model descriptor. Attachment now
refreshes a changed, unloaded file profile using a fully resolved snapshot
accepted by that older API. Current backends also validate the snapshot against
the file before retaining its file provenance. Loaded models and explicit runtime
registrations keep their settings. The legacy compatibility bridge registers its
snapshot as a runtime profile until the next explicit model-file reload.

The exact command subsequently opened the real terminal UI successfully and
exited with Ctrl+Q, preserving the existing backend. Lucifer remained loaded from
`/home/Lucifer/Soul/scripts/serve-strata` and answered `2+2` with `4`. Five CLI
regressions and two API regressions cover stale snapshots, legacy API payloads,
authentication, startup timeouts, failed loads, runtime overrides and file
provenance. Evidence: `existing-backend-attachment.json` in the repair directory.

## Earlier environment

The test environment has Python 3.13.5, an NVIDIA RTX 5070 with 12,227 MiB VRAM,
and the existing vLLM 0.33.0 environment. Core dependencies include Textual 8.2.8,
FastAPI 0.162.5, and httpx 0.28.1. Training was tested with Torch 2.14,
Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, and bitsandbytes 0.50.2.

## Initial validation (historical)

The following initial checks and the 0.2 release record describe earlier
milestones. The later release results supersede their test and registry counts.

- 239 automated tests passed. They cover actual FastAPI↔TUI integration,
  persistence across restart, logging and training consent, corrections,
  redaction, interrupted streams, adapter gates, cancellation, shutdown,
  source revocation, and GPU placement configuration.
- Ruff and Python compilation checks passed.
- A wheel and source distribution built successfully.
- The installed wheel was loaded from a separate environment outside the source
  checkout; configuration initialization, doctor, and the demo API passed.
- The combined `lucifer demo` launcher ran in a real PTY and exited with Ctrl+Q,
  shutting down its backend cleanly.
- Wide and compact (88×30) terminal layouts were rendered and inspected.
  Demo previews and a separate live primary-model preview are retained. The
  live preview shows an actual Markdown table and highlighted Python response.

## Initial real model inference (historical)

Using the cached `Qwen/Qwen3.5-4B` checkpoint, the application launched a real
vLLM server, streamed an answer through its real HTTP API, and saved the
conversation. A one-word capital-of-France question returned **Paris**. The
saved conversation contained the user and assistant messages.

The tested profile used a 2,048-token context, 80% GPU-memory utilization,
LoRA enabled (one adapter, maximum rank 32), eager mode, text-only serving,
and the Torch fallback sampler. The model occupied approximately 8.11 GiB
for weights. Qwen thinking was disabled through `chat_template_kwargs` so
short response budgets produce the final answer rather than exhausting tokens
on reasoning.

## Initial real training and adapter serving (historical)

- A local tiny model completed actual CPU PEFT LoRA training and wrote
  `adapter_config.json` and `adapter_model.safetensors`.
- A two-process CPU `torchrun` job completed actual distributed LoRA training,
  with the artifacts saved by rank zero.
- Cached Qwen3.5-4B completed an actual one-step GPU QLoRA run using a short
  public example, rank 4, batch size 1, and sequence limit 128. It trained
  5,308,416 parameters; runtime was 12.56 seconds and training loss was
  0.00626068. This is an integration check, not evidence of improved capability.
- The resulting real PEFT adapter loaded into vLLM through
  `/v1/load_lora_adapter`. A real generation routed to that adapter returned
  **Paris**. No test adapter was promoted.
- Owned inference and training processes stopped and released GPU memory.

The complete real `/sleep/start` API flow was then exercised independently:
saved chat → reviewed correction → one approved example plus three replay
examples → inference unload → real QLoRA worker → candidate registration →
inference restore → four baseline and four candidate generations → evaluation.
Both task scores were 1.0, with zero regressions and no safety-case failures.
The measured improvement was 0.0, below the 0.02 policy threshold, so the run
correctly finished `rejected`. The active adapter remained unset. This confirms
the orchestration and promotion gate without claiming a quality improvement.

Local evidence is retained under `var/validation/` (excluded from distribution
and source control), including `api-real/result.json`, the training worker log,
training configuration, and PEFT checkpoint. The separate training environment
reuses the existing vLLM environment's packages without modifying it.

## Limits of these checks

Only one physical GPU was attached. Multi-GPU device selection, masks,
CUDA/ROCm compatibility, tensor/pipeline parallel flags, llama.cpp splitting,
and distributed launch commands were verified with controlled inventories.
Distributed lifecycle was exercised on two CPU processes. Multi-GPU CUDA,
ROCm, Vulkan, Metal, and real GGUF serving require validation on their respective
hardware and backend builds. `llama-server` was not installed on this machine.

Those initial runs used a small deterministic regression suite.
It checks actual baseline/candidate responses and fails closed on missing
suites or failed gates; it is not a broad safety benchmark. Automated tests
exercise promotion and rollback with controlled model responses. No real
candidate was claimed to improve or pass promotion policy.

One upstream Starlette warning appears in tests: using httpx with TestClient
is deprecated in favor of httpx2. Current tests pass; the runtime client uses
httpx directly and is unaffected.

## Primary streamed profile integration

The project default is now the public `Lucifer` profile. Configuration defaults,
public model aliases, terminal branding, managed Strata process ownership, and
readiness are covered by automated tests. Strata must advertise both the expected
model alias and `loaded: true` before the application marks it ready. A borrowed
external server is never stopped by the model manager.

The primary weights are acquired with Axel 2.17.14 at 100 connections. The local
Axel build expands URL buffers to handle Hugging Face's long signed CDN links;
TLS verification remains enabled. A local HTTP Range fixture verified import of
the prior downloader's completed chunks into Axel's resume state, and the first
complete model shard subsequently passed its pinned SHA-256 hash.

All eight files completed hash verification (257,789,086,989 bytes). The primary
model was packed locally without further acquisition and loaded through Lucifer's
managed Strata backend. The application was started without a `--load` argument;
`default_model: lucifer` selected it automatically. A real project chat answered
with Lucifer's name and Paris, and a streamed arithmetic request returned `4`.

Live registration of the unchanged descriptor, inline-spec loading, and YAML
reload preserved the running model process. Controlled tests also exercise new
arbitrary descriptors, persisted registrations, changed sources, and failure
atomicity. An immediate unload/load regression exposed a TIME_WAIT port probe
issue; the availability probe now supports rapid reuse while rejecting a live
listener.

The primary profile now uses `context_length: auto`. The launch helper reads
`max_position_embeddings` from the selected pack, and the runtime allocated and
advertised 1,048,576 tokens. Both `native_context_length` (source metadata) and
`effective_context_length` (observed runtime) equal that value. Short prompts were
verified at this setting; a million-token prefill, throughput, and quality test
was not performed.

All five selectable thinking levels—off, low, medium, high and max—completed live
requests through the project API and returned the expected answer `323` for
17×19. Each response reported the requested level; the profile default remained
off. The terminal has a persistent mode selector and `/thinking` and `/tokens`
commands. Headless tests cover mode persistence, compact visibility, budgets,
and suppression of the separate reasoning channel when thinking is off.

The wrapper disables Strata's persistent conversation slots and unsets its raw
debug mode. Old synthetic slot files were removed during the controlled restart;
subsequent verification found no slot directory recreated. At that integration
checkpoint, the service was loaded on loopback ports 8000 (application) and 8089
(runtime).

Evidence: `var/validation/lucifer-live/default-verification.json`,
`thinking-context-verification.json`, `agent-probes.json`, and startup/runtime
logs. The earlier real QLoRA evidence refers to the smaller trainable profile;
the primary streamed model is used for inference.

## Response formatting, thinking and live hardware

The updated backend was restarted with the existing primary model as its default.
A real high-thinking request for 17×19 streamed **19 reasoning events** (74
characters), then the final answer **323**. The saved synthetic conversation and
its export contained only the user message and final answer. The synthetic saved
conversation was deleted after the check. Reasoning text was not written to the
validation record. Thinking off also returned **323**.

During that request, the shared hardware sampler produced seven distinct sample
timestamps. GPU utilization changed through **2%, 11%, 9%, 24%, 13%, 98%, 100%**;
VRAM, temperature and power counters were present. The unchanged model/adapter
revision did not trigger another bootstrap. The capability snapshot before and
after generation was identical, including its discovery timestamp.

Automated checks cover cached startup discovery, profile invalidation, metadata
precedence, cancellation of pending discovery, stream cancellation, reasoning
privacy, stale/unsupported GPU counters, changing samples, and revision-based UI
snapshots. Hardware inventory is shared with inference and training placement.
Changing metrics still require periodic sampling; static discovery does not.

Evidence: `var/validation/lucifer-live/tui-backend-verification.json`.

The real-model Textual session received a separate thinking channel and toggled
its panel open and closed successfully. The exported previews keep it collapsed
so reasoning text does not enter documentation assets. Wide layout shows the
original haloed portrait and live GPU meters; compact layout retains thinking
selection and a short GPU summary. The response table and highlighted code render
as Markdown, while the user prompt remains literal. These earlier checks preceded
the agent implementation recorded below.

## Lucifer 0.2 agent and Cyberpunk release

The 0.2 suite passed **239 tests** in 30.78 seconds. Ruff and Python compilation
passed. The runtime package now requires jsonschema; version 4.26.0 is installed.
The new tests cover all three typed dialects, complete/multiple/malformed calls,
strict JSON and schema formats, safe 429 retries, model/context pins, image budget
accounting, idempotency, external-file changes, denied/expired approvals, command
cleanup, consent/deletion, MCP configuration/transport and the interactive cortex.

Actual primary-model runs completed the following tasks through the agent API:

| Task | Observed result |
| --- | --- |
| Workspace investigation | Located and read `amplify`, explained `value * 3 + 1`, and cited `signal_engine.py:1` and line 3 |
| Anthropic Messages | A real tool/result continuation read the source and returned the correct explanation |
| Stateless Responses | A real tool/result continuation read the source and returned the correct explanation |
| Scoped write | Read the current SHA-256 and changed only the multiplier from 3 to 4 using `edit_file`; the fixture file was verified |
| Direct PC | Proposed the exact host argv, waited for concrete approval, executed outside the workspace, and wrote the expected host test artifact |
| Structured output | Returned `{"multiplier":4}`, validated against an object-root schema with a required constant and no extra keys |
| Vision | Read the explicit local image input and answered `Red` correctly |

The capability snapshot before and after those runs was identical. Initial runs
exposed an inefficient repair round for absolute paths inside the workspace.
Absolute/relative aliases are now normalized lexically while preserving all
escape/symlink guards; tests verify the fix. No capability probing was added to
turns. Source metadata invalidates read caches after ordinary external edits;
completed edits/deletes are reused before live precondition checks.

A later real Textual mission used high thinking, streamed 92 characters of emitted
reasoning into the collapsible view, read the updated source, and returned the
correct `value * 4 + 1` answer with file/line citations. Expansion/collapse worked.
Wide and compact screenshots were inspected; hidden reasoning was absent from
exported SVGs. At the 0.2 validation checkpoint, the live adapter registry held no
trained adapters, so its cortex showed the loaded base core and zero shards. This
historical zero count does not describe the later 0.3 expert study. Dense
adapter lineage/status interaction is additionally covered with controlled
registries and explicitly synthetic previews.

A completed real agent answer was saved with explicit logging/training consent,
reviewed positively, retained once as an explicitly approved memory, and included
in a scoped sleep dry run. Only the user/final-answer pair was used. Deleting that
synthetic conversation removed its agent-export content, memory and dataset.
No primary-model weights were trained or promoted in these agent checks.

A real mission was cancelled before output. The model stayed loaded and the next
ordinary chat returned `323` for 17×19; cancellation plus follow-up took 3.15
seconds. Actual bubblewrap and host subprocess tests also verify time/output
bounds and cleanup of descendants that ignore TERM, including closed pipes.

MCP remains disabled because no server was configured for this project. Its
startup negotiation, allowlists, explicit run scope, approval, result bounds,
JSON/SSE transport, rollback and cancellation are covered with controlled servers.
Enabling a real external MCP service requires its explicit configuration and
service-specific acceptance; native capability advertisement is not permission.

The 0.2 backend was restarted on that release's source. Health reported 0.2.0,
Lucifer was loaded as the default, context was 1,048,576, GPU telemetry was fresh,
and no agent run was left active. Runtime persistent conversation slots and raw
vendor debug capture were disabled. These are integration checks, not a claim
of general agent competence or full-window model performance.

Local evidence: `var/validation/agent-live/result.json`, `tui-result.json`,
`learning-bridge.json`, `cancellation.json`, plus owned backend/runtime logs.
Release previews: `docs/assets/terminal.svg`, `terminal-compact.svg`, `cortex.svg`
and `cortex-compact.svg`. The updated implementation matrix is in `AGENT_PLAN.md`.

## Resumable sessions, dynamic sleep and Cortex (0.3)

The final 0.3 source passed **546 automated tests**, plus Ruff and Python
compilation. One optional Torch-dependent module was skipped in the core
environment; its **three real Torch/PEFT CPU tests** passed separately in the
training environment without initializing a GPU. The remaining warning is the
upstream Starlette/httpx TestClient deprecation described above.
Coverage includes unbounded default missions, same-run follow-ups, corrections
during thinking, pause/resume, output continuation, retry-safe input/resume,
expired approvals, interrupted image batches, and actual HTTP shutdown cleanup.
Dynamic sleep checks cover consented reviewed-data selection, compatible source
replay, frozen run-local evaluation artifacts, independent oracle variants,
strict JSON types, incomplete-response handling, and restoration before final
progress completion. Alternating native base/LoRA requests preserve the model
registry revision and cached capability objects; explicit adapter selection
publishes a change once.

### Verified real sessions and output budgets

A real primary-model session spanned **2,591.34 seconds (43 minutes 11 seconds)**
and included a pause during actually emitted thinking. The same run retained its
session reasoning in RAM, recalled `cobalt-29`, accepted a correction, and ended
with `copper-71` on session turn 4. Its deadline remained unset and its final
status was `awaiting_input`. The export contained no provider reasoning.

The freshly loaded 0.3 application selected Lucifer by default and reported both
native and active context as **1,048,576 tokens**. A real request accepted an
output budget of **262,144 tokens** and returned the short answer `READY` with a
normal stop. That limit is a profile output budget; the runtime's native output
limit remains unknown. This verifies budget routing with a short response,
without establishing full-length output or million-token-prefill performance.

Evidence: `var/validation/session-live/result.json`,
`var/validation/reload-0.3.json`, and `var/validation/output-0.3.json`.

### Twenty-conversation expert study and data preflight

Twenty actual primary-model conversations covered different complex topics at
MAX thinking. All twenty finished normally with `finish_reason: stop`; their
emitted provider reasoning totalled **344,521 characters**. Only counts were
retained in the receipts. Raw thinking stayed in session memory and was excluded
from saved answers, exports, feedback and training targets.

All twenty public final answers received independent, hash-bound reviews;
eleven used complete reviewed replacement targets. Corrections addressed actual
mathematical or protocol errors before training eligibility. A strict CPU-only
preflight verified every conversation, original/replacement hash, target identity,
and exact trainable-4B chat-template encoding. All twenty prompts and reviewed
targets survived redaction unchanged. The longest complete training pair was
**1,532 tokens**, fitting the selected **4,096-token** sequence limit without
truncating any final target. Preflight accepted all twenty examples with no errors
and `ready_for_feedback_and_training: true`.

The dynamic pipeline freezes its configured regression anchors, source-derived
independent probes, unchanged promotion policy, profile identity and source
fingerprints before training. Its related-family smoke tests have explicit
coverage limits; incomplete or unsupported evidence remains unknown, and
anchor-only gains cannot establish expert-domain improvement. Promotion remains
an explicit action after the measured gates pass.

The real 0.3 candidate `lucifer-reasoning-generalist20-memory2` completed GPU
QLoRA on the pinned 4B training base: **18 optimizer steps**, **23 examples**
over three epochs, **10,616,832 trainable parameters**, **83.2739 seconds** of
training and **0.93507572 training loss**. Its checkpoint contains **256 LoRA
tensors** and **42,504,600 bytes**. A separate CPU audit found every tensor
finite and all **128 LoRA-B tensors nonzero**. These facts establish trained
weights, not an improvement in held-out capability.

An earlier attempt ran out of GPU memory during backward; its failed job and
confirmed effect receipt were preserved. The fresh `memory2` attempt reused the
same reviews and frozen evaluation plan. Restoring the frozen, tied embedding
to its original BF16 storage saved **1,271,398,400 bytes**, while FP32
normalization preparation remained intact. The single-worker QLoRA allocator
default uses expandable segments only when the user supplied no allocator
configuration; distributed workers keep their existing policy. This follows
the documented [PyTorch 2.14 allocator options](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).
Complete reviewed targets were retained throughout.

The first candidate's evaluation correctly remained **unknown**: the cached
vLLM startup snapshot had no advertised transport, so all 34 pairs were unscored.
No model responses or improvement were fabricated. Startup now reads the
provider's origin OpenAPI schema once for compatible servers; repeated native
turns reuse the same immutable snapshot. A separate consolidation check verifies
the corrected live progress and evaluation paths below.

The explicit consolidation `lucifer-consolidated-generalist20` then replayed
the same twenty reviewed conversations from the verified first expert, plus
three bundled examples. It completed **six optimizer steps** in **28.7047
seconds**, with **1.13666331 training loss**, and registered a distinct checkpoint
with the correct parent edge. Its 256 tensors were independently checked on CPU:
all finite, all 128 LoRA-B tensors nonzero, and the registered SHA-256 matched.
This is source-data replay on the compatible base, not a merge of adapter tensors.
No feedback or dataset-preview POST was repeated for this consolidation.

Actual cached training progress was observed at **1/6** before completion and
rendered in wide and 88×30 Sleep views. Evaluation progressed through **31/68**
and **54/68** attempted responses. Both trained experts and their parent edge
were visible in the real Cortex; Review selected the child while preserving
camera orientation and zoom. The capture client used GET requests only and
contained no thinking widgets. Its six structural bootstraps did not grow with
each evaluation response; the normal telemetry heartbeat carried progress.

The consolidation's first evaluation completed **66 of 68** final outputs.
The missing pair was rejected locally: its conservative 2,098-token estimate
exceeded the small profile's 2,048-token serving window before generation. The
other 33 pairs found zero task or domain gain. A final acceptance evaluation
uses the startup-advertised native chat tokenizer with these exact same frozen
prompts; its results are recorded separately, preserving both historical jobs.

The final acceptance run evaluated **both existing checkpoints** without further
training or promotion. Each completed **34/34 scored pairs**, or **68 native
final responses**; all twenty dynamic probes were scored. Both baseline and
candidate task scores were **4/33 (12.12%)**, excluding the safety case. Both
task and dynamic-domain deltas were **0.0**, with zero regressions and no
safety-case failures. The first candidate's measured latency delta was
**−7.04%**; the consolidated candidate's was **+39.31%**. Both therefore remain
**not_passed**, and neither was promoted. These small transfer smoke checks
establish no broad expertise or capability improvement.

The previously blocked adapter-replay question had **297 native input tokens**,
not the conservative estimate of 2,098. All final completion receipts record
the advertised `vllm_tokenize_chat` method and `estimated: false`; one count was
shared by each baseline/candidate pair. Frozen questions, their digests, source
reviews and promotion gates were unchanged. Finalization adopted the two
confirmed result receipts read-only after a report guard rejected a
capability-name metadata key; no model request was repeated.

After evaluation, Lucifer 0.3 restored primary `lucifer` as the loaded/default
profile with native and active context **1,048,576**, active adapter unset and
no active agent or learning job. The large primary remains the inference
default; both new LoRA experts require the pinned compatible 4B training base.

Evidence: `var/validation/expert20/collection.json`, `review.json`,
`preflight.json`, `training-result-proof-memory2.json`,
`tensor-proof-memory2.json`, and the twenty saved public prompt/final-answer
records. The failed attempt remains in `training-result-proof.json`.
Consolidation evidence is in `consolidation-result-proof.json`,
`tensor-proof-consolidation.json`, and `var/validation/sleep-tui-live/result.json`.
Final complete measurements and model-restoration receipts are in
`reevaluation-result-proof.json` and `reevaluation-effect-receipts.json`.

### Cortex rendering and one-start sleep UI

The unprofiled Cortex benchmark rendered **100 frames** of a **129-node synthetic
graph** at **160×50** terminal cells. Median render time was **4.81 ms** and p95
was **5.51 ms**. These measurements describe the rendering fixture, not a count
of trained experts or a full-application frame-rate benchmark. Evidence:
`var/validation/cortex-performance/unprofiled.json`.

Wide and compact session and sleep controls were rendered under
`var/validation/session-tui/` and `var/validation/sleep-tui/`. Layout fixtures use
labelled synthetic thinking and expert data. The F4 sleep view retains the
cyberpunk palette and exposes one start action, stage/optimizer/evaluation
progress, cancellation and candidate review. Overall completion reaches 100%
only after the original inference profile is restored; a restoration error
remains visible without a false completed progress bar.


## Context curriculum 0.4 — real GPU acceptance

The new context-aware pipeline ran against three fresh, consented fictional
project conversations and four actual primary-model final responses. **No
assistant feedback was submitted**. Targets came from independent literal
arithmetic, quoted USER constraints and deterministic worked/error-correction
exercises; unreviewed assistant answers were not gold targets. A multi-turn
Python/8089 → Rust/9090 correction retained the current explicit constraints.

The first job was cancelled when the application closed during teacher prompt
prefill, before training. Its immutable teacher receipt remains `uncertain` with
`CancelledError`. A new job adopted the saved chats and independent candidates;
it did not reset or repeat that teacher request. Thus this particular run proves
the independent generator and uncertain-request recovery path, not successful
live model-drafted example generation. Bounded draft parsing, cache reuse and
mandatory independent/human validation also have automated coverage.

Run `sleep_a96843c6209f4d21` verified **19 candidates**, split into **14 training**
and **5 held-out** examples with disjoint source groups. The training diversity
gate counted **two source tasks**; direct/worked/correction versions of one task
cannot inflate that count. Three bundled replay examples produced **17 training
rows**. All fresh and held-out candidate evidence was independently revalidated
from a read-only SQLite connection after completion. Memory reconciliation
created **19 bound records** with no source conflicts.

Real QLoRA on the pinned compatible 4B base completed **five optimizer steps** in
**5.3742 seconds**, loss **0.87977816**, and **10,616,832 trainable parameters**.
Adapter `lucifer-context-curriculum-v1` has **256 tensors**, **42,504,600 bytes**,
and SHA-256 `d7da5e3da8565491ec35bbcd03cc3db643d949903e7b7c8c6594fbb81a778e4c`.
A separate CPU audit found all tensors finite and all **128 LoRA-B tensors
nonzero**, without initializing CUDA.

Evaluation completed **20/20 baseline/candidate pairs**, or **40 final responses**,
using native `vllm_tokenize_chat` counts. Baseline and candidate task scores were
both **7/19 (36.84%)**, with zero task/domain gain, zero regressions and no safety
failures. Candidate latency increased **17.69%**. Coverage remained partial,
including just one supported dynamic family probe, so quality is **unknown** and
promotion remains disabled. Real training alone establishes no expertise gain.

The GET-only TUI acceptance captured actual optimizer progress, evaluation and
restoration in wide and 88×30 views. Cortex displayed the new trained adapter and
its unknown quality; Review revealed it while preserving camera orientation and
zoom. Hidden progress animations stopped, and no thinking widgets were captured.
The normal heartbeat supplied updates; no model/tool requests were made by the
capture client. Primary Lucifer returned as default/loaded with **1,048,576**
context, **262,144** output allowance and no active adapter or learning job.

Evidence: `var/validation/curriculum-0.4/real-training-proof.json`,
`tensor-proof.json`, `live-effects.json`, `resume-effect.json`, and
`var/validation/context-curriculum-live/run-sleep_a96843c6209f4d21/result.json`.

The finite pure-code validator checks only USER-supplied cases inside a restricted
expression interpreter; it does not run host code or prove general correctness.
Deduplication uses normalized semantic/task identities, not a universal semantic
equivalence oracle. Sleep does not automatically reopen every file/URL previously
read by an agent. File access is not training consent or independent documentary
evidence; a claim requires its permitted source, version binding and validation.


### Agent history and current resource gauges

The missing user session was traced read-only to conversation
`conv_49e6aa5ccf3b4d1b`: logging and training were both off and there were zero
durable messages. The paused run retained 81 completed tool-operation receipts,
which contain replay identities but not a recoverable conversation. Restart
therefore removed its RAM-only public transcript as well as thinking. The update
lists that session honestly as unavailable; it does not reconstruct invented text.

The Agent view replaces the separate Chat navigation. F5/HISTORY opens an
on-demand list with public transcript availability, retention and pinned scope.
OPEN only reads. Future conversations with logging explicitly enabled can retain
versioned public checkpoints and completed result receipts; restart continuation
is an explicit action under the same profile, workspace, access and tool scope.
Concrete pending approvals expire. Logging-off and deletion purge durable public
checkpoints/results while retaining opaque replay guards. Provider reasoning
remains session-only. Oversized checkpoints are withheld whole rather than
silently truncated.

Resource gauges show current GPU utilization and VRAM percentages with themed
proportional bars. Unknown values use a distinct dotted gauge, measured zero uses
an empty gauge, and stale samples are labelled. No trend history is collected by
the TUI. Rendering consumes the existing telemetry cache and adds no discovery or
sampling calls. Temperature and power retain their native units as secondary
readings; percentages are not invented without a meaningful capacity limit.


### Final 0.4 verification

After the final checkpoint and UI changes, **663 core tests passed**, with one
optional Torch/PEFT test module skipped in the core-only environment and one
existing Starlette/httpx deprecation warning. Runtime was **69.44 seconds**.
Ruff and Python compilation also passed. Optional training behavior was verified
by the real GPU run and independent CPU tensor audit described above.

The persistence regressions include restart with and without logging, explicit
SAVE of paused RAM context, whole-unit checkpoint bounds, source/profile/tool
scope changes, stale completed-result recovery, expired approvals, and
pre-effect proposal durability. An input accepted while cancellation was in
progress is conservatively marked interrupted on restart when its application
cannot be confirmed; the UI displays the notice and never replays it silently.
Separate tests cover on-demand History reads, Agent-only navigation, current-only
resource gauges and unchanged telemetry samples without extra widget writes.


The final source was activated on the live backend and native Agent frontend.
A GET-only live UI check listed **15 historical runs**, including the exact old
private run with logging off and no recoverable transcript. Both wide and compact
captures show Agent-only navigation and current GPU/VRAM bars. Capture requests
were one bootstrap, one History list, one selected-session read and two normal
telemetry heartbeats; there were no mutations, model streams or thinking widgets.
Evidence: `var/validation/agent-history-tui/native-live/result.json`,
`native-agent-wide.png`, and `native-history-private-compact.png`.
Restart continuation of logged checkpoints is covered by isolated lifecycle and
crash-boundary tests; the live private run was not misrepresented as resumable.


## Saved defaults and frontend recovery — 12 November 2026

The recent frontend process exited while its backend mission
`agent_2dd778c2bb124d2d` continued running. The original exit cause was not
recorded. The terminal profile closes a window when its child exits; no kernel
or user-cgroup OOM kills were recorded, and GNOME Terminal itself remained alive.
The real session replayed successfully, so no rendering failure was fabricated.

The mission was briefly paused and its existing conversation was explicitly
saved through the API. A complete public recovery checkpoint retained both user
inputs and **49 completed tool results**. Thinking remained in backend RAM and
was never copied into the checkpoint, diagnostics or validation artifacts.
A GET-only UI replay consumed **12,134 events**, including **11,735 reasoning
events**, without invoking a tool or resubmitting the task. Wide/compact resizing
also succeeded. The native frontend then reopened that same mission with
`tui --session ID`; the user continued it with its original MAX reasoning mode.
The backend was not restarted, preserving session thinking as requested.

New TUI conversations explicitly save by default, with training consent off.
The schema, backend creation default and bootstrap declaration have the same
saved/on, training/off defaults in source. The already-running backend keeps its
old omitted-field API default until its next normal restart, as the user chose to
retain current RAM thinking. Existing explicit private-mode choices stay private.
A new monotonic saving path allows logging to be enabled during an active agent
without cancelling its provider request; other privacy/scope changes remain
subject to the existing idle guard.

Ctrl+C now pauses owned work rather than silently quitting; Ctrl+Q explicitly
quits. Fatal Textual diagnostics contain only exception types and frame locations,
excluding exception messages, locals, source lines and provider content. Diagnostic
files are written atomically with mode0600. The CLI propagates frontend failures,
keeps a native fatal window visible until Enter, and preserves its owned backend
on frontend failure; normal explicit exit still cleans up its owned backend.

Evidence: `var/validation/chat-recovery-20261112/save-current.json`,
`reopen-ui.json`, `var/validation/frontend-recovery/result.json`, and
`var/validation/frontend-recovery-os.json`. These receipts contain counts and
lifecycle metadata, not raw reasoning or copies of the user's transcript.


The final recovery/defaults source passed **676 tests**, with one optional
Torch/PEFT module skipped and the existing Starlette/httpx deprecation warning,
in **134.31 seconds** under the ongoing live inference load. Ruff and compilation
passed. Regression coverage also caught and fixed a queued thinking-picker event
race: an older OFF event can no longer overwrite a newer MAX selection. The
correction test injects that stale event and waits for the corrected segment's
actual final Markdown flush, preserving both emitted thought streams.
