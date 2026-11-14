# Lucifer adaptive Sleep

Adaptive Sleep uses primary Lucifer as a teacher and a separate compact Lucifer
profile as the student. The primary model prepares learning proposals with
maximum supported thinking; an isolated LoRA/QLoRA worker trains the compact
student. The primary SSD-streamed model remains inference-only.

New Sleep jobs default to the shared `behavioral_v1` outcome protocol, including
one-shot and adaptive runs. Each checkpoint is compared with its original
student baseline on fresh owned rule-inference, hidden-test coding, tool-workflow
and failure-recovery episodes. Tools and budgets match; independent verifiers
score final answers, code and state. Old likelihood/topic smoke scores are not
relabeled as behavioral evidence. Fixed one-shot safety/completion controls and
artifact/source checks remain in effect.

The shared [training master prompt](../lucifer/resources/training_master_prompt.txt)
guides the loaded teacher, while independent program/source checks decide which
targets may train. Unsupported drafts are excluded automatically; an empty
verified dataset is reported honestly rather than waiting for Codex review.
The Sleep page shows task outcomes and a responsive legend. No test proves
unrestricted intelligence, and ties or incomplete measurements cannot pass
admission. Sleep trains adapters; [Meditation](MEDITATION.md) changes only the
primary model's existing output head.

Lucifer is the public assistant identity. The student remains an adapter for its
original architecture and checkpoint. Technical source names, revisions,
runtime identifiers and license attribution remain in provenance; renaming an
adapter does not make it compatible with another model. See
[model provenance](MODEL_PROVENANCE.md).

## Starting and controlling a run

Save the source conversations and enable training consent for the material you
want to use. Reviewed targets and independently verified source-derived examples
can be admitted to the curriculum. Pending drafts, conflicts and unsupported
claims require evidence or review before training.

Pause any active mission before starting Sleep. For the profiles configured in
this checkout, the primary teacher is `lucifer` and the compact training profile
is `lucifer-train`. Check `/models` for the IDs in your installation.

```text
/sleep start --adaptive --thinking max --generation-model lucifer --seed 19
/sleep status
/sleep cancel SLEEP_RUN_ID
/sleep resume SLEEP_RUN_ID
```

The seed controls schedule randomness. Ordinary `/sleep start` keeps its existing
one-shot behavior. Add `--dry-run` to prepare the eligible dataset and evaluation
plan without training weights. Explicit API clients can select the compact
student with `base_model`:

```json
{
  "base_model": "lucifer-train",
  "generation_model": "lucifer",
  "generation_thinking": "max",
  "schedule": {
    "mode": "until_admitted",
    "refresh_optimizer_steps": 50,
    "initial_update_budget": 50,
    "seed": 19
  }
}
```

Send this request to `POST /sleep/start`. Resume uses
`POST /sleep/runs/{run_id}/resume` with an `idempotency_key`; reuse the same key
when retrying an unacknowledged resume request. The terminal retains that key.
An explicit resume can continue committed work, but cannot bypass changed source
consent, missing evidence or an uncertain operation without a safe recovery.

## Adaptive rounds and growing random bounds

A round requests a number of **successful optimizer updates**, rather than a
number of complete dataset passes. Gradient-accumulation microbatches and
optimizer steps skipped by mixed-precision overflow do not consume that budget.

The initial state is adaptive epoch 1, round budget 50, and integer random bounds
`1..7`. After a complete round, compare the new checkpoint with the previous
round checkpoint on the same development-suite revision:

- If the score improved, draw `r` from the current bounds and add `3 + r` to the
  next round's budget.
- If the score worsened, draw from the same bounds and subtract `3 + r`.
- A tie leaves the budget and random-generator state unchanged.

Every completed round advances both random bounds by one, including worsening
rounds and ties. Thus the initial adjustment is `4..10` updates; later ranges can
produce larger adjustments. For example, a first draw of 4 adds 7 updates, making
the next budget 57 and the next bounds `2..8`. A following tie preserves 57 and
advances the bounds to `3..9` without drawing another random number.

If a proposed budget is **10 or below, or 90 or above**, begin the next adaptive
epoch with budget 50 and bounds `1..7`. Both threshold directions reset. An
adaptive epoch records a budget regime; dataset-pass counts are separate
telemetry. Randomness changes only the update budget, not the learning rate or
admission rules.

## Checkpoints, evaluation and data refresh

The outer loop has no arbitrary round or update cap. Each isolated worker segment
is finite: it stops at the earliest complete dataset-pass boundary, global
50-update refresh boundary, or remaining round-budget boundary. The controller
assesses the resulting checkpoint. Coincident boundaries use the same committed
checkpoint.

Data refresh is assessed at cumulative successful-update counts 50, 100, 150,
and so on. Round and adaptive-epoch resets do not reset this counter. If a
57-update round begins at cumulative update 50, its refresh at update 100 leaves
seven updates to finish that round. Refresh and resume preserve the remainder
and do not advance the random bounds.

Refresh admits only material passing the existing consent, independent
verification, source-conflict, partition and overlap checks. Completed proposals
and generation receipts are reused. New data is committed as a separate revision;
verified replay remains available when no new material qualifies. Missing
independent evidence pauses the run rather than weakening its gates.

Every 50-update refresh measures the latest checkpoint on the separate frozen
development panel, including boundaries in the middle of a round. Failed
development families select practice targets even when source history is
unchanged. Generation receives family tags only; admission questions, expected
answers, case parameters and failed admission responses are excluded. The
refresh accepts at most 32 novel, independently solved proposals from consented
training groups. Previously reserved admission and development groups are
excluded before the teacher is called. Verified replay retains mastered skills.
If no new verified training material qualifies, both the existing dataset and
development revision remain unchanged. Unsupported or incomplete development
evidence pauses generation for review.

Development comparisons use the previous round and current checkpoint on the
same development revision. Admission uses a separate frozen suite and pinned
student baseline. Complete evidence, supported topic coverage, safety,
regression, measured improvement and latency gates remain mandatory. A baseline
already at the improvement ceiling produces `unreachable_gate`; it does not
cause easier tests or reduced thresholds. Supported task-family smoke tests do
not establish unrestricted reasoning ability or general domain mastery.

The first checkpoint passing admission stops the loop before another budget
adjustment. It becomes **admitted** and available for inspection in Sleep/Cortex.
Admission does not load or activate it. Primary Lucifer is restored, and a paused
mission remains paused. To use the new adapter, explicitly load its compatible
compact profile and select the admitted adapter. Selection does not silently
switch the base model.

## Resume, privacy and source withdrawal

Committed checkpoints contain the adapter, optimizer, constant learning-rate
scheduler, native Trainer/RNG state, optional mixed-precision scaler, sampler
cursor metadata and adaptive state. Hashes and a manifest published after the
checkpoint files bind the result. Resume preserves successful-update counters,
remaining budget, random bounds and random-generator state. A new dataset
revision resets its data cursor without resetting the optimizer or accumulated
successful updates. Changes in distributed rank topology require a committed
round boundary.

Validation calls record stable identities before execution. Completed numerical
receipts, answer hashes and verdicts are reused on reconnect; uncertain requests
pause for inspection instead of being silently resent. Validation does not
execute tools. Explicit resume of a stopped run may authorize a new attempt for
an uncertain tools-disabled validation call. The unknown attempt remains in its
journal, and the replacement call identity is committed before execution. This
contract does not authorize retrying training, generation or tool effects;
completed validation receipts remain unchanged. Teacher reasoning is transient provider data and never enters
training targets, saved validation evidence or durable logs. Learning artifacts
contain verified final targets and their source bindings.

Deleting a source conversation, withdrawing training consent or superseding its
evidence invalidates dependent datasets, checkpoints and adapters, including
already admitted descendants. Such artifacts cannot become valid merely by
removing the source from a later refresh. A continuation must retain valid
lineage for every revision it consumed.

The Sleep view receives actual job progress through the shared telemetry
heartbeat. Overall work remains indeterminate while the uncapped loop is active;
a job is not reported restored until its original inference profile has been
restored successfully. See [the interface reference](ADAPTIVE_SLEEP_INTERFACE.md)
and [the context curriculum](CONTEXT_CURRICULUM.md) for commands and source-review
controls.

Optimizer progress reports successful updates from the owned adaptive worker;
native Trainer steps are separate and cannot advance the schedule. The dashboard
distinguishes live updates from the last sealed checkpoint. Counts remain stable
while loading models, generating examples or evaluating. Adaptive epoch 1 can
span multiple dataset passes and rounds until a budget reset occurs.

The first-expert workflow is to resume the same source-valid checkpoint, assess
each completed boundary, refresh practice from independent development failures,
and stop at the first checkpoint passing every frozen admission gate. The result
must be an admitted, source-valid adapter available for explicit loading on its
matching compact profile. A better training loss alone does not complete this
workflow.
