# Agent sessions and conversation-derived experts

Lucifer keeps an agent mission open across corrections, pauses and follow-ups.
Version 0.4 adds a context-aware curriculum before offline training. The default
sleep action recovers consented tasks, verifies examples and reconciles scoped
memory; pending proposals require explicit review. See
[the context curriculum workflow](CONTEXT_CURRICULUM.md).

## Continuing and steering a mission

1. Select **AGENT** with F2. Choose the workspace and visible access policy.
   READ and WRITE stay inside the workspace; SANDBOX adds confined commands;
   DIRECT PC permits host paths and commands with the backend user's OS access.
   AUTO CMDS is an explicit run choice. It does not authorize MCP effects.
2. Select the thinking level and send the first goal. Expand **Model thinking**
   to inspect the separate reasoning channel actually emitted by the provider.
   That panel is available while generation is running.
3. To change direction during generation, type in the **CORRECT** input and press
   its button. For example: `Use the new configuration; keep the source findings
   but do not edit files yet.` This interrupts the current segment, adds the
   correction to the same mission and starts its next segment. Earlier panels,
   conversation context, completed tool results and reasoning remain available.
4. Use **STOP**, Esc or Ctrl+C to pause; Ctrl+Q explicitly quits the terminal UI.
   Resume with the **RESUME** button or `/resume`.
   Add guidance with `/resume Inspect the uncertain command result before editing`.
   Only owned model requests and command process groups are stopped. A command
   interrupted after starting may have effects; the operation journal marks that
   uncertainty and the agent must inspect it before another mutation.
5. After a final answer, send the next message normally. It continues the same
   mission ID with retained history. Completed operations keep their identities
   and results. Reattaching the event stream never resubmits the goal.

Equivalent commands include:

```text
/agent workspace /absolute/project/path
/agent access read
/thinking max
Investigate the configuration and cite the relevant files.
/correct Check the production configuration instead of the example.
/agent stop
/resume
/resume Continue with the revised requirement.
/agent status
/agent list
/agent open AGENT_RUN_ID
```

**HISTORY / F5**, `/history` and `/agent list` open the on-demand mission list.
The list distinguishes backend session memory, saved checkpoints, public
transcripts and unavailable private runs. Select an entry to preview its public
messages and logging/training status. **OPEN SESSION** reads the available
content without starting work; **RESUME** or a new input is the explicit action
that continues a recoverable checkpoint. Checkpoint restoration validates the
loaded model/profile/adapter and recorded scope, expires concrete approvals, and
reuses recorded completed results rather than replaying their effects.

`/agent open ID` replays retained session events from the running backend, or
loads a saved checkpoint/transcript after restart, without sending a new goal.
It restores that mission's visible workspace, access, dialect,
image/MCP scope, command choice, thinking and output settings. Resume preserves
the mission's pinned model/profile/adapter. Changing its workspace, access policy,
dialect, images, MCP servers or command grant requires a new mission; sending a
new goal under that selection creates one with the conversation history.
`/agent new` selects a new mission within the conversation. `/session new` or Ctrl+N
explicitly selects a new conversation and mission.

New conversations save automatically, while training consent stays OFF. Use
`/privacy logging off` for a private conversation; that explicit setting remains
honored. `/session save` enables saving again. When an older private session has no saved transcript,
History still lists its metadata and explains why its text is unavailable.
Operation hashes and status receipts cannot reconstruct the missing text.
For startup attachment, `lucifer tui --session AGENT_RUN_ID` reads the selected
session after connecting, without resuming execution. Frontend crash diagnostics
contain only exception types and code locations, never messages, locals or
provider content. Ctrl+C pauses owned work; Ctrl+Q is the explicit exit action.

### Stopping automatically or waiting for input

**AUTO STOP ON GOAL** defaults to on. After a final answer, the mission is marked
`completed`; a follow-up can still continue that same session. Turn it off to
mark the mission `awaiting_input` after its answer. Both states are idle and
release the serving slot. Turning it off does not make the model generate
meaningless additional turns. Use a new message for more work after an answer;
plain `/resume` is for an unfinished paused goal.

```text
/agent auto-stop on
/agent auto-stop off
```

There is no default agent wall-clock deadline, round limit, tool-call count limit
or cumulative token limit. API clients can still request those bounds. Native
context, per-request output allowance, file/search/output bounds, command
lifetimes, approvals and process cleanup remain enforced. Removing a mission
deadline does not remove the selected tool policy.

### What reasoning retention means

Actual provider reasoning stays in the TUI and backend session RAM across pauses,
corrections, model/tool turns and UI reconnects. Each turn has its own expandable
panel. A supported provider history field can carry that emitted reasoning into
later requests under the loaded profile's startup contract; unsupported providers
do not receive a fabricated reasoning transcript.

Reasoning is never saved to chat exports, durable run logs, feedback, memory
summaries or training datasets. A backend restart discards session reasoning.
With logging enabled, a final-only checkpoint and public transcript preserve
eligible mission context and recorded completed results. Reopening after restart
cannot recover old reasoning and never automatically replays uncertain effects.
Private sessions with logging off retain no recoverable message text after
restart. Keep a user-approved final summary
with `/agent remember SUMMARY` when durable task knowledge is useful.

## Output budget and context

The model's startup snapshot determines native/active context and the output
allowance. The API exposes `max_output_tokens` and `output_limit_source`. The
allowance is bounded by advertised output metadata, the configured profile budget
and active context; the smallest known bound applies. An allowance from profile
configuration is not a claim that the model's native output maximum is known.

```text
/thinking off
/thinking low
/thinking medium
/thinking high
/thinking max
/tokens 65536
/tokens max
/tokens auto
/agent continuation on
/agent continuation off
```

The requested token count must fit the initialized profile. There is no global
32,768-token output ceiling. Agent automatic defaults are 2,048/4,096/8,192/16,384/
32,768 for off/low/medium/high/max respectively, bounded by available context.
Chat defaults remain 512 with thinking off and 4,096 with thinking on. An explicit
`/tokens N` persists until changed; `/tokens auto` restores the mode defaults.
Thinking and final answer consume the same request budget.

**AUTO CONTINUE OUTPUT**, on by default, can issue another model turn when an
agent's text response reaches its output limit. It retains the task, partial
answer, actual reasoning and completed tool results, and requests continuation.
This increases the total answer possible over multiple requests. It consumes the
remaining native context and cannot create a larger context window. The provider
can also finish before the requested budget. Incomplete tool calls never execute;
structured responses are not blindly stitched across output truncations.

When the active context is full, the agent automatically compacts older history
into a bounded model-generated working summary. It preserves the current goal,
working project state, completed tool results, fundamental repository/docs facts
and unresolved next steps, while omitting transient chatter, raw reasoning and
redundant tool output. Compaction repeats in bounded chunks until the next request
fits; the summary is retained in the session and checkpoint. `context_compaction:
false` disables it explicitly. Long output settings are capacity requests, not a
guarantee of response length or quality.

## Creating and updating experts

An expert is a named, trained PEFT LoRA adapter with recorded source conversations,
topics, training base and verified checkpoint artifacts. It is not a renamed
model entry or a claim of measured expertise. Artifact verification and evaluation
quality are separate facts.

### Start the whole sleep pipeline

After collecting consented, reviewed answers as described below, press F4 and
select **SLEEP / CONSOLIDATE**. `/sleep start` performs the same action. There is
no need to manually switch away from the primary model or specify conversation
IDs. The default pipeline first recovers context, drafts and validates examples,
and reconciles memory. It continues when the verification minimum and source-group
partition requirements are met. It then selects a compatible trainable base,
topics and source adapters and freezes its evaluation plan before training.
It trains, verifies artifacts, measures baseline/candidate
responses and restores the original inference profile and adapter.

The yellow overall bar stays indeterminate while total pipeline work is unknown.
The cyan stage bar shows a percentage only when that stage has a known total.
Optimizer steps and baseline/candidate evaluation response counts come from the
worker and evaluator; the chosen evaluation case count is shown separately.
Before evaluation starts, a known frozen case count can supply a labeled planned
response total; missing completed counts remain unknown. Indeterminate bars only
animate while this view is visible and work is active. This view uses the existing telemetry heartbeat;
opening it does not add polling or repeat capability discovery.

Overall completion reaches 100% only after restoration and cleanup finish.
**CANCEL JOB** cancels the selected owned job while preserving restoration.
**REVIEW CANDIDATE** selects and reveals the actual registered expert in Cortex.
It does not promote it. The cortex keeps all rotations available, including
mouse drag and Q/E roll; progress-only updates do not rebuild its graph.

A run can end with `needs_review` or `insufficient_verified_data`, a candidate
awaiting review, a rejected evaluation, unknown quality, no new information, a
preview, cancellation or failure. REVIEW EXAMPLES fetches pending canonical
questions/answers on demand. Each decision binds the full displayed content hash;
a corrected target creates a new version. The UI keeps corrections after a stale
hash/source conflict and requires explicit reload before reviewing new content.
Independent
topic probes currently cover the implemented exact task families; unsupported
questions are reported as partial/unknown coverage. The pipeline never invents
ground truth or broad expert competence from an unsupported topic. A trained
artifact with incomplete evidence remains `quality_unknown` and cannot pass the
promotion gate. A restoration failure is visible and never displays 100%.

`/sleep start --dry-run` previews the curriculum and available dataset/evaluation
plan without training. `/sleep start --legacy` or `context_examples: false` keeps
the 0.3 example path; `dynamic: false` preserves explicit expert selection.
There is no automatic schedule or automatic promotion.

### Collect and review a conversation

1. Ask a question or complete an agent goal. New conversations save by default;
   training consent remains off. For an explicitly private conversation, use
   `/session save` before enabling training, then
   `/privacy training on` to permit learning from that conversation.
2. Review the completed final answer. Use `/feedback good` for an answer you
   verified, or `/feedback correct THE COMPLETE CORRECT ANSWER` to supply the
   target answer. Negative-only and unreviewed answers are excluded. A steering
   correction during thinking changes the mission; it is not training feedback.
   Consented USER-only literal tasks/configuration can also seed independently
   verified examples without approving an assistant answer. The default threshold
   needs three useful verified examples across at least two semantic task IDs.
3. Continue collecting verified examples on the topic. Use `/history` to find
   source conversation IDs. Consent is per conversation; a new conversation
   starts with saving on and training off. Only the latest feedback decision for an answer applies.
4. Preview with `/sleep start --dry-run`, inspect the verified/pending/source-group
   counts, review pending examples and inspect any reported private JSONL before
   training. That command uses the ordinary sleep selection;
   use the scoped API example below to preview exact expert sources.

```text
/session save
/privacy training on
/feedback good
/sleep start --dry-run
/sleep status
```

The default curriculum recovers source context and selects verified canonical
user/final-answer pairs, applies redaction and deduplication, preserves grouped
training/held-out partitions, and includes the configured replay examples. The training
worker masks user/system tokens and trains on the assistant answer. Reasoning
and tool traces are excluded throughout.

### Propose and train an expert

```text
/experts propose
/experts propose CONVERSATION_ID_1,CONVERSATION_ID_2
/experts build "Lucifer configuration procedures" --base lucifer-train --topics python,configuration --conversations CONVERSATION_ID_1,CONVERSATION_ID_2 --description "Reviewed configuration tasks"
/sleep status
/experts list
/adapters
/eval results
```

Without IDs, TUI `propose` and `build` use the current conversation when present.
Proposals are deterministic lexical topic groups from eligible reviewed content;
they exclude examples already covered by recorded experts on the selected base.
Their selection is whole conversations, so inspect the proposed source IDs.
A proposal does not train weights or switch the current model.

`build` starts a real sleep job and returns its run ID. It selects the single
compatible trainable profile automatically when unambiguous; `--base PROFILE`
makes that choice explicit. On this machine `lucifer-train` is the trainable
profile. The primary streamed profile is for inference. Inference is released
for the worker and restored by the learning service, then the candidate is
evaluated against the current baseline using held-out cases.

A trained expert must have a PEFT configuration, valid LoRA safetensors, finite
worker metrics and at least one optimizer step. The registry records the tensor
count and checkpoint SHA-256. `expert_training_status: trained` establishes real
weights; `expert_quality_status` reports `not_evaluated`, `passed`, `not_passed` or `unknown`
separately. A trained candidate can fail the promotion gate. Review evaluation
results before `/promote ADAPTER_ID`; promotion remains explicit and gated.

The adapter is compatible with its recorded base and checkpoint. A 4B expert
cannot be loaded into the primary streamed model. Loading its compatible base
and selecting an eligible adapter is a separate serving action.

### Add new information or consolidate experts

After new consented conversations have been reviewed, run `propose` again or build
a named expert from the new source IDs. The latest reviewed correction replaces
the target for that answer. Keep original source conversations and current
consent available when you want to replay them later.

To combine existing experts or trained PEFT adapters, select their actual adapter
IDs and optionally include new conversation IDs:

```text
/experts consolidate "Lucifer combined procedures" ADAPTER_ID_1,ADAPTER_ID_2 --conversations NEW_CONVERSATION_ID --topics python,configuration
/sleep status
/experts list
```

Consolidation verifies the selected source artifacts and requires the same
registered training base/checkpoint, replayable sources and current consent. It
unions those sources with the new conversations, rebuilds the reviewed dataset,
and trains a fresh LoRA from the original base. This is data-level consolidation:
no tensor averaging, adapter weight merge or cross-base conversion is performed.
Source adapters remain available and the new record links every source parent.
F3 shows those lineage edges, training state, quality state and verification data.

Turning training consent off excludes the conversation from future datasets and
blocks affected promotion; deleting or revoking source content prevents later
consolidation from it. External exports and already-trained weights are separate
artifacts and cannot be erased by changing a consent flag.

## API examples

Agent session controls preserve one run ID. SSE cursors are monotonic across
segments; `segment` and `session_turn` distinguish interrupted work from the new
turn. `GET /agent/runs/ID` reports `resumable` and `reasoning_retained`.

| Route | Body or purpose |
| --- | --- |
| `GET /agent/history` | On-demand mission metadata with transcript/session/checkpoint availability and logging/training state |
| `GET /agent/runs/ID/session` | Public transcript and recorded scope without restoring or executing the mission |
| `POST /agent/runs` | A goal, explicit workspace/access, optional `stop_when_done`, `continue_output` and limits |
| `POST /agent/runs/ID/input` | `{"content":"Use the new requirement","interrupt":true,"options":{"thinking":"max","max_tokens":65536}}` |
| `POST /agent/runs/ID/stop` | `{}` pauses owned work |
| `POST /agent/runs/ID/resume` | `{"limits":{"timeout_seconds":null}}`, optionally with `instruction` |
| `GET /agent/runs/ID/events?after=N` | View events after a cursor without repeating execution |
| `GET /experts` | Verified trained expert records |
| `POST /experts/propose` | `{"conversation_ids":["CONVERSATION_ID"],"base_model":"lucifer-train"}` |
| `POST /experts/build` | Start a scoped expert sleep job using the body below |
| `POST /sleep/start` | `{"dynamic":true}` starts the full dynamic pipeline over new eligible reviewed conversations |
| `GET /sleep/examples?status=pending` | Fetch canonical examples for explicit review |
| `POST /sleep/examples/ID/review` | `decision`, displayed `content_hash`, optional complete `correction` and `reason` |

Use `timeout_seconds: null`, `max_rounds: null`, `max_tool_calls: null` and
`max_total_tokens: null` for unbounded defaults. They can be supplied as explicit
bounds in the initial request or updated through input `options` and resume
`limits`. `stop_when_done` and `continue_output` default to true;
`max_continuations` defaults to null. Scope and model pins cannot be widened
through turn options.

Input and resume bodies also accept `idempotency_key`. The TUI client supplies
one UUID for each logical control request and does not retry it automatically.
An API caller retransmitting a request must retain the same key and unchanged
content/instruction, options/limits and interrupt choice. Matching duplicates
reuse the recorded transition instead of creating another turn; a delayed retry
after STOP does not restart the paused work. Reusing a key with another payload
is rejected. Input and resume receipts use separate namespaces. Private-session
receipts live in RAM; logging-enabled checkpoints preserve their receipts with
the recorded scope and completed results. Disabling logging removes the durable
checkpoint. If a restart interrupted an accepted control before its payload was
retained, History displays that notice; the application does not invent or
automatically repeat the missing instruction.

Scoped dry-run expert request:

```json
{
  "name": "Lucifer configuration procedures",
  "base_model": "lucifer-train",
  "topics": ["python", "configuration"],
  "conversation_ids": ["CONVERSATION_ID_1", "CONVERSATION_ID_2"],
  "dry_run": true
}
```

Inspect its curriculum and any eligible dataset, then submit the same selection with `dry_run: false` for
real training. To consolidate, add `source_expert_ids` with compatible adapter
IDs. The API also accepts `expert_auto: true` to select a content-derived proposal
when the requested job starts; it uses only eligible new reviewed information.
This is an explicit job option, not a background schedule or automatic promotion.

`name`, `description`, `topics` and `conversation_ids` are API aliases for
`expert_name`, `expert_description`, `expert_topics` and `dataset_ids`.
The normal training fields also apply. Expert jobs return ordinary sleep records;
monitor `GET /sleep/status` or `GET /sleep/runs/ID`. A dry run only plans an expert
and prepares data; it never appears as a verified trained expert.
