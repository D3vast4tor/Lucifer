# Meditation

Open **Meditation / F6**, or launch `lucifer tui --view meditation`.
Choose a mode, source scope and Standard or small-pilot budget, then press **Start**. Opening the page does not
start training. There is no mandatory human review step.

| Mode | Objective and eligible material |
| --- | --- |
| Supervised | Predict final answer targets from saved training chats. Provided corrections take precedence; explicitly bad answers without a correction are excluded. Source excerpts can provide answer targets. |
| Unsupervised | Predict the next token in saved public chat and document text. No answer labels or approval are required. Long sources use bounded windows without dropping new target tokens. |

Both modes update only the primary Lucifer model's existing output head:
661,913,600 weights. They preserve the backbone, Engram, input embedding, norm,
vision and draft tensors. Sleep remains the separate compact-model adapter
workflow. Meditation does not substitute that compact model.

New runs compute the head with F32 activations/logits at exactly its BF16 saved
weight values. CPU F32 gradient accumulation keeps successive backward passes
within the pilot VRAM budget. The stored model remains BF16. Older checkpoints
retain their recorded arithmetic; changing precision while resuming is rejected.

For checked question/answer material in documents, use the explicit
[problem/solution JSONL format](MEDITATION_DATA_FORMAT.md). Supervised mode uses
its provided answers as targets rather than learning to reproduce the surrounding
JSON. Ordinary document excerpts retain their existing literal-source behavior.

## Sources

The source scope can select the current chat or all eligible chats. A chat must
already have saving and training enabled. Neither mode changes those choices.
The page shows selected chats, consumed chats, source counts and partitions.
Empty selected chats contribute no targets.

Attached files and linked public documents use the existing knowledge extractor.
Successful web reads can save the bytes already downloaded into the chat's
knowledge collection, retaining source hashes and the chunks actually returned.
No second fetch is needed merely to create training evidence. A private chat
keeps sources transient. Quoted document instructions remain source data.

For a document downloaded with a terminal command, use `import_document` on its
authorized local path, then `search_knowledge` or `read_knowledge`. Importing makes
the document searchable; only chunks with recorded interactions enter Meditation.
Successful bounded text-file reads also import the exact bytes already read and
record the returned passage. Neither path broadens workspace access or enables
training for a chat. Visual extraction through the agent requires image input
enabled for that run.

Public user text, completed final answers and document text have distinct source
authority. Provider reasoning, raw tool traces and unfinished answers are
excluded. Image/video descriptions retain their model-derived authority;
Meditation trains their text representation, not the visual encoder.

Duplicate documents and connected chats stay in the same source group. Training,
development and heldout groups remain disjoint. New eligible sources can be
incorporated through an immutable source revision; previously consumed sources
must still be valid. Withdrawing consent or deleting/changing sources blocks
derived candidates and clears managed source/feature caches. It cannot undo
knowledge already present in a trained weight tensor.

## Adaptive training and saved bases

New jobs default to `behavioral_v1`, shared with Sleep. The controller creates
owned task resources and independent program oracles automatically: unfamiliar
rule/path problems, code repairs with hidden tests, inventory workflows and
recovery from interrupted payments or changed tool arguments. Baseline and
candidate receive the same tasks, tools, completion rules and per-case budgets.
Different correct code implementations and minimum-cost paths are accepted.
The task protocol still imposes a bounded output/tool contract; it is not an
unrestricted intelligence examination.

The page reports solved tasks by category, gains, regressions, incomplete
measurements and an explanatory legend. Loss measures target-token prediction;
KL measures sampled distribution drift. Neither measures general intelligence.
If source chats do not supply numerical development/heldout partitions, trusted
generators supply separate numerical probes. They never enter training and are
not counted as independent human-source groups. Behavioral episodes use their
own separate seeds and hidden tests.

The shared [master prompt](../lucifer/resources/training_master_prompt.txt) guides
loaded-model teaching proposals. The native controller checks its source-order
proposal against the actual available authority labels; unsupported suggestions
fall back to enforced rules. Sleep uses the same contract while drafting
independently verified examples. No teacher can certify its own answer, change
consent or expose admission oracles to training.

Select **Small pilot / 1 update**, or pass `--pilot-updates 1`, to run a bounded
retained segment after the separate disposable rate probes. A pilot can finish
as `pilot_complete` with an honestly rejected or inconclusive candidate and
verified restoration. This is a successful workflow test, not a quality pass.
Standard jobs keep adaptive segments until admission or a development plateau;
each segment receives a fresh sealed outcome draw. The sequential confidence
budget tightens across attempts. Interrupted measurements remain unknown and
do not authorize replay of optimizer effects.

The outer search has no default time, round or cumulative optimizer-update cap.
Each worker segment is finite and checkpointed. Development measurements guide
learning rate, segment length and rollback to the best development checkpoint.
An initial search tests three conservative rates with five updates each, using
training features and development scores only. Each trial starts from the same
untrained BF16 head. Numerical receipts are sealed before disposable probe
weights are removed; a partial probe resumes its own checkpoint. The selected
rate starts the retained optimizer afresh. Probe updates are shown separately
and do not count as retained training updates or quality admission evidence.
Resource limits, source withdrawal, cancellation and insufficient evidence can
stop or pause work. Repeating a request with its idempotency key does not repeat
completed training operations.

Recoverable CUDA allocation/VRAM-limit failures permit at most three microbatch
reductions per finite trial. A retry requires the owning worker to have exited,
unchanged head/data/optimizer input hashes, and no committed result. Only then
are its private uncommitted files discarded. Smaller microbatches increase
accumulation to preserve the effective example count, including sampler wrap at
dataset boundaries. Committed or uncertain work is never replayed on this basis.
Fixed host-state or disk-capacity failures stop rather than enter this fallback.

A verified VRAM-limit failure in read-only development evaluation can offload
that evaluation to CPU. Its exact worker source and input hashes must match,
the failed worker must have exited, and no evaluation result may be committed.
The first CPU evaluation checks overall and per-group development baseline
losses against the committed GPU baseline within 1e-5. Completed training steps
are preserved; CPU evaluation writes a separate result. Failed parity or a host
RAM limit stops the run, and the resource caps stay unchanged.

The dashboard shows the actual parameters, successful updates, head deltas,
measured losses, current stage and restoration state. Overall progress stays
indeterminate during the open-ended search. An iteration does not count as an
improvement merely because its training loss fell.

Development retention includes forward KL from the untrained BF16 head, capped
at mean 0.005 and maximum 0.02 over the fixed development target features. These
are sampled distribution limits, not a guarantee about every possible prompt.
Missing or mismatched receipts cannot support admission. The dashboard and
Results show precision and distribution retention. Three consecutive failed
regression/retention segments restore the best checkpoint and pause for new
evidence; a meaningful valid improvement resets that streak.

The first fully passing checkpoint is saved automatically as an immutable base
revision. Its sidecar contains the changed head; manifests share the unchanged
large model shards. The original pack is preserved.

Automatic gates include:

1. Byte-identical head sidecar loading and separate BF16 conversion parity.
2. Correct native features, labels and real changes in exported BF16 weights.
3. For new jobs: complete paired behavioral outcomes, positive task improvement,
   no task regressions and the exact discordant-episode sign test. Numerical
   retention remains mandatory. A tied or incomplete pilot cannot admit a base.
   **Legacy jobs only:** native-engine heldout improvement against both the original model and the untrained BF16
   conversion baseline, using fresh independent groups and a sequential testing
   budget. The rule needs at least eight fresh heldout groups per admission
   attempt, increasing the panel as the sequential budget tightens. Too little
   evidence produces `awaiting_evidence`.
4. Real loading of the changed head, changed native logits and matching feature
   projection, with unchanged original head hashes.
5. Fixed instruction, JSON/tool-contract, source-instruction and repetition
   controls, followed by verified return to the parent model.

There is no manual example or candidate review requirement. **Activate** remains
a separate explicit action once all automatic gates pass. **Rollback** restores
the previously activated base. Both revalidate artifact and source identity.
Saving a candidate alone does not switch the active model.

Native capture and candidate evaluation pin a reproducible execution plan: CPU
routed experts, no GPU expert cache, 12 CPU threads, six routed experts, and the
engine's verified 16-token feed window. The original and BF16 heads must produce
identical frozen backbone features under this plan. This can run more slowly
than the usual serving profile; the saved candidate carries the evaluated plan.
Numerical improvement uses this controlled native parent as its reference, and
serving controls also check the ordinary parent profile. Development tuning may
use cached feature projections; new behavioral admission generates actual model
answers/actions, while legacy heldout admission uses native source-window scores.
A passing pending admission is reconciled before
more optimizer work after an interruption.

The optimizer is capped at 6 GiB VRAM and 24 GiB RAM, with a 50 GB run/cache
budget. These optimizer limits do not describe the frozen native backbone's
normal inference memory: feature capture still loads the large original model
and uses its host-memory cache. Supervised long answers use bounded windows with
recent task context and every assistant target scored once; earlier public turns
are context, never extra answer targets.

Incompatible experts disappear from the active Cortex and selectors; their
files, registry and history remain. Compatibility uses the runtime's actual
adapter support and exact base revision where recorded. Rolling back restores
the corresponding view. Current Strata profiles do not serve LoRA adapters.

## Commands and API

```text
/meditation
/meditation start --mode supervised
/meditation start --mode supervised --current --pilot-updates 1
/meditation start --mode unsupervised
/meditation start --mode unsupervised --dry-run
/meditation status
/meditation sources
/meditation results
/meditation cancel RUN_ID
/meditation resume RUN_ID --idempotency-key RESUME_KEY
/meditation bases
/meditation activate BASE_ID --idempotency-key ACTIVATE_KEY
/meditation rollback --idempotency-key ROLLBACK_KEY
```

Source and result inspection is optional and does not affect eligibility.

`POST /meditation/start` takes `mode`, an `idempotency_key`, optional
`conversation_ids`, `base_model`, `seed`, `dry_run` and `max_seq_length`.
An optional `source_roles` map precommits consumed source components to training,
development or heldout roles; existing reservations cannot be relabeled and
related source references cannot cross roles. See the data-format guide.
The API exposes `/meditation/runs`, `/meditation/runs/{id}`,
`/meditation/runs/{id}/cancel`, `/meditation/runs/{id}/resume`,
`/meditation/bases`, `/meditation/bases/{id}/activate` and
`/meditation/rollback`. Mutations require an idempotency key. The existing
bootstrap/telemetry heartbeat carries Meditation progress.

Model loading and training are serialized. The original inference profile is
restored on completion, failure or cancellation; the job remains busy until that
restoration finishes. Resume uses committed optimizer state and source lineage.
An uncertain uncommitted effect is not silently retried.

## Validation status

The first checked Supervised pilot completed 75 retained updates and changed
24,883,699 BF16 elements from its untrained baseline. Development loss worsened
from 0.7665509462 to 0.7671482044, so it paused without admission. Native loading
and exact parent rollback passed; the eight heldout groups were not consumed by
an admission attempt. The precision diagnostic and complete receipts live in
`var/knowledge/agent-llm-curriculum/pilot-v1/`. The subsequent precision and KL
correction still requires its own actual-hardware acceptance evidence.

The latest calibrated application-started Unsupervised run completed 15
disposable rate-probe updates and 75 retained updates. It changed 275,303,019
BF16 elements relative to the untrained export, loaded the changed head and
restored the original parent exactly. Development loss worsened from
2.9670177928 to 2.9954966914, so the run paused on a plateau. It is not currently
training; no new base was admitted or activated. Two fresh heldout groups remain
available against the required eight. Weight changes alone are not improvement.
The receipts are `calibrated-native-result.json` and
`calibrated-total-head-delta.json` under
`var/validation/meditation-native-20261113/`.

The user's request for better agent/LLM material is addressed in
[the curriculum plan](MEDITATION_CURRICULUM_PLAN.md). It recommends verified
problem/solution tasks and public worked explanations for Supervised mode,
with selected versioned documents for knowledge retrieval or Unsupervised
adaptation. The user still chooses and starts a mode. The original model passed
11 of 12 narrow diagnostic cases; these cases are excluded from training and
admission. The source list is a preparation plan, not an imported training corpus
or proof of eight independent fresh evaluation families.

The full-sized native pilot `meditation_3707832fd56a4ad0` completed 25 successful
updates on the installed `[129280,5120]` head. Its exported BF16 tensor changed
65,713,997 elements (L2 delta 1.6263993, maximum absolute delta 0.00048828125).
The native engine loaded that head and produced changed logits; restoring the
original parent reproduced its logits bit for bit, with the original head hash
unchanged. Peak optimizer allocations were 5,387,568,640 bytes VRAM and
11,878,277,120 bytes host RSS.

Cached-feature development loss moved from 2.96701779 to 2.96655722. This is a
development measurement, not proof of general improvement. The pilot restored
the normal model and stopped at `awaiting_evidence`: its three source groups do
not supply the eight independent heldout groups needed for quality admission.
No new base was admitted or activated. The receipt is
`var/validation/meditation-native-20261113/native-head-acceptance-v5.json`.
Automatic admission/activation paths have fixture tests; a statistically admitted
native base still requires sufficient independent real sources and a passing
native heldout result.

See [the implementation plan](MEDITATION_PLAN.md) and
[the original hardware assessment](BASE_MODEL_TRAINING_FEASIBILITY.md).
