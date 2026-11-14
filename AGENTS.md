# Lucifer project conventions

- Public assistant/model identity is Lucifer. Keep technical checkpoint sources,
  runtime identifiers and license attribution accurate in provenance metadata.
- Discover model interfaces and native context during model initialization.
  Cache an immutable snapshot for that loaded profile revision. Invalidate it
  only on explicit model/profile initialization or change. Never probe features,
  reload metadata files or enumerate hardware from chat/token/render paths.
- Collect hardware inventory once at backend startup. Use the shared background
  telemetry sampler for changing GPU counters. HTTP reads consume its cached
  snapshot; unknown and stale counters must remain distinguishable.
- Bootstrap TUI registries once, then invalidate them on structural revisions.
  Job progress and GPU counters travel through one telemetry heartbeat. Update
  widgets only when their rendered data changes, and batch streaming Markdown.
- Keep provider answer, reasoning, tool-call, usage and completion channels
  distinct. Show only reasoning actually emitted by the provider. Reasoning is
  transient UI data and must not enter chat exports, durable logs, feedback or
  training datasets. User messages are literal; assistant answers use Markdown.
- When the active context is full, compact older history into a bounded
  model-generated working summary. Preserve the current goal, working project
  state, completed tool results and fundamental repository/docs facts; omit
  transient chatter, raw reasoning and redundant tool output.
- Future agent execution must record call IDs and operation fingerprints before
  effects. Reconnects and retries must reuse completed results instead of
  repeating completed operations. Retry a mutation only under an explicit,
  documented idempotency contract. Capability recognition is not permission to
  execute a tool.
- Runtime access modes are explicit: read/write/execute are workspace-confined;
  computer is the user-requested Direct PC mode with host paths and commands.
  Never silently switch into computer access. Preserve concrete approvals and
  the explicit AUTO CMDS choice. Cancel only owned processes.
- The user has authorized implementation of docs/AGENT_PLAN.md. Capability
  discovery does not authorize individual runtime tool effects; each run's
  workspace, access policy and concrete approval decisions govern execution.
