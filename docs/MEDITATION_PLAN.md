# Meditation: native output-head training and new base revisions

## Superseding user requirements — 14 November 2026

The user explicitly replaced the previous goal and its stop condition. This
section takes precedence over the historical progress/admission objective below.
The goal service refused to replace its unfinished objective record; that is an
accounting limitation, not permission to keep the superseded stop condition.

Implement the shared design in [TRAINING_EVALUATION_PLAN.md](TRAINING_EVALUATION_PLAN.md):
default outcome-based evaluation for **both Sleep and Meditation**, automatic
preparation of consented learning material and independent validation resources,
an enforced shared master prompt with mode-specific rules, and understandable
page metrics with a legend. Preserve Sleep's adapter/expert consolidation and
Meditation's primary-head/base training, source/privacy rules, explicit activation
and verified restoration. Routine jobs must not depend on Codex/manual curation.

**New stop condition:** validate a small actual-model run end to end, including
generated independent behavioral tasks, baseline/candidate comparison, real
weight updates, a justified acceptance or rejection and verified restoration.
An admitted improving base is no longer required to finish this implementation
goal. Rejecting an inadequate small run is valid evidence; fabricating or
weakening a pass is not. The broader suite does not claim universal intelligence.

**Replacement goal complete:** see the completion evidence in
[TRAINING_EVALUATION_PLAN.md](TRAINING_EVALUATION_PLAN.md) and
`var/knowledge/agent-llm-curriculum/behavioral-pilot/completion-audit.json`.
The actual small native run changed weights, completed paired behavioral tests,
was correctly not admitted and restored the original model. The verifier's
`set.add` defect was fixed and both saved code artifacts rescored without
replaying training or inference. No positive quality/admission result is claimed.

---

User-authorized implementation plan, 13 November 2026, updated 14 November.
Status: implementation and full-sized weight-update/loading/rollback tests are
complete; positive statistical quality admission remains unproven. The progress
entries below record successive stages; the latest results supersede earlier
pending-stage descriptions.

Current active run: `meditation_70107a862cea40a0`, started through the real
application in Supervised mode after a source-only precommit. Its expanded
material retains the 48 checked worked solutions and adds 58 source-verified
technical continuation exercises, totaling 5,886 native assistant-target tokens.
Six new owned public training chats reuse the original ten evaluation chats;
their complete evaluation group records are unchanged. Partition groups remain
train=1, development=2, heldout=8. No source group count is inflated by new
documents. Intake staging and 23 focused checks passed. The source pack and exact
runtime snapshot are in `var/knowledge/agent-llm-curriculum/pilot-v2/`. Native
preparation and calibration are complete; main training is active. Initial
full-sized updates are proven, while quality admission remains unproven.
The original-head training baseline completed all 14,829 teacher-forced rows,
covering 5,886 assistant targets. It then completed BF16-head frozen-feature
capture; `pilot-v2/capture-progress-receipt.json` records that intermediate phase.
`pilot-v2/native-intake-agreement.json` also verifies that the native intake
contains exactly the staged 106 training examples and 5,886 assistant targets,
with source/dataset hashes bound and provider reasoning/tool traces excluded.
Training-feature extraction committed all 5,886 target rows across 12 bounded
shards; F32 shape, byte sizes and feature/label hashes match. Development and
heldout preparation then completed and calibration began. The first five-update
probe changed 2,572,699 BF16 head entries within the unchanged resource caps.
Verified GPU-budget failure automatically moved its read-only development
evaluation to CPU, preserving the completed probe without a manual resume;
baseline parity passed. The first development result moved only microscopically
and is not a quality admission. Matching receipts are
`pilot-v2/training-feature-receipt.json`, `preparation-complete-receipt.json` and
`first-probe-and-automatic-cpu-recovery.json`. All three probes lowered development
loss slightly (0.7665472570, 0.7665369935, 0.7664997432); the controller selected
2e-5 and started main training. Retained training still needs its quality/rollback
gates; no new base is yet admitted or activated.
The first 25-update retained segment improved development loss from 0.7665509462
to 0.7646511233 and passed native head loading plus bit-identical parent rollback.
Its first native admission failed: only 3/8 groups improved by more than 0.001
against each baseline, with sign-test p=0.85546875 against alpha=0.025. Mean
improvement was 0.0003465355 versus untrained BF16 and 0.0006007232 versus original
Q5_K. The controller correctly accepted no base and continued training. Those
eight admission groups are now consumed; a later candidate needs a fresh panel.
Evidence: `pilot-v2/first-native-admission-result.json` and
`first-retained-native-verification.json`. Do not reuse the failed panel or
describe the development improvement as passing independent admission.
The retained 5,886 tokens are below the original proposed sizing range; this is
a measured expanded pilot, not a claim of optimal or sufficient coverage.

**API ownership deployment completed:** auditing the active run found that the
Meditation resume endpoint omitted the plain-chat guard present on start. Sleep
resume also omitted plain-chat and Meditation ownership checks. Regression tests
reproduced both omissions; the API-only correction passed 70 focused checks,
including same-Sleep receipt replay, and a package build. After the pilot stopped
and restored the original model, idle API/native checks and owned-worker exit
were verified before backend restart. A real native, training-disabled chat then
confirmed that both resume endpoints return 409 while its turn is active. That
owned temporary chat was deleted after completion; no learning job started and
Lucifer remains restored. Numeric and bound controller modules were unchanged.
Receipts are in `pilot-v2/pending-resume-ownership-fix/`, including the original
staging record, `restart-receipt.json` and `deployment-verified.json`.

Latest completed run: `meditation_c57925a4c8484495` (Supervised, native F32
arithmetic with BF16 saved weights). It completed 15 disposable calibration
updates and 75 retained updates, then stopped `awaiting_evidence` on a development
plateau. Loss worsened from 0.7665509462 to 0.7671532498. Native loading, exact
parent rollback and KL retention passed; the original head was independently
rehashed unchanged and normal Lucifer serving restored. No base was admitted or
activated and no admission attempt consumed the heldout panel. CPU evaluation
recovery reused the completed probe and original frozen features; baseline parity
passed (maximum loss delta 0.00000249027, tolerance 0.00001). Total exported BF16
changes were 24,887,010 entries, L2 delta 1.1110354270. The arithmetic correction
therefore did not resolve this corpus's quality plateau. Broader checked
training material is required before another optimizer attempt. Evidence is in
`var/validation/meditation-native-f32-20261114/status.json`.

**Latest measured result and curriculum amendment (14 November):** the calibrated
application-started run `meditation_869f6f1591644398` completed three disposable
five-update rate probes and 75 retained updates. Total BF16 changes from the
untrained export were 275,303,019 elements, L2 delta 13.9729053385. Native loading
and exact parent rollback passed; original weights were preserved and normal
`lucifer` serving restored. Development loss worsened from 2.9670177928 to
2.9954966914. The run paused `awaiting_evidence` on a development plateau; it
must not be described as improving, admitted, activated or still running.
Only two independent fresh heldout groups currently exist against eight needed.
Receipts: `calibrated-native-result.json` and `calibrated-total-head-delta.json`
under `var/validation/meditation-native-20261113/`.

The user subsequently requested deducing the best material for reasoning and
agents/LLMs instead of supplying more private chat paths. The original model
passed 11/12 narrow diagnostic content checks. Prioritize verified recovery,
debugging and tool-selection tasks with public worked explanations, plus
versioned technical documents. Follow [MEDITATION_CURRICULUM_PLAN.md](MEDITATION_CURRICULUM_PLAN.md)
for source choices, bounded pilot preparation, source-family separation and
measurement. This is new data-selection work, not permission to fabricate eight
independent groups or silently start the user's Supervised mode. No private
provider reasoning is retained or used for training.

Combined verification after the latest control fixes: 1,089 core tests passed
(four skipped); 46 native numeric/capture tests passed separately. Storage
preflight, document-origin lineage and plain-chat ownership guards are deployed.
The shell/agent busy checks include ordinary chat activity before model control.
The screenshot helper integration is complete and its portal capability is
available on this host; actual desktop capture still follows runtime consent.

**Supervised curriculum continuation:** the user chose Supervised for the next
checked problem/solution pilot. Explicit inert JSONL source records now supply
question/answer targets instead of passage-copying prompts. Source references
and canonical repository/paper families merge dependency groups; lineage changes
invalidate source snapshots. An optional start-request `source_roles` map
precommits whole consumed groups and refuses conflicting roles or relabeling
existing reservations. These controls do not execute document verifiers, confer
truth on supplied answers, start training automatically or require review.
The pilot source selection reserves six train, two development and eight
heldout families before examples, scores or optimization. Source/material
preparation is still in progress; the native quality gate remains unproven.

**Supervised pilot started:** `meditation_55a9bb1e732d45fe` is now running from
the real application's terminal input, scoped to 16 newly prepared public-source
study chats. Its 48 checked training examples share one authored-dataset lineage
and therefore one training group. Four literal development passages use two
other repositories; 16 literal heldout passages use eight further repositories.
An initial batch of 30 locally authored nontraining exercises was excluded from
admission rather than using new citations to inflate independence. Original
reference files and selected workbooks are searchable (32 attached documents);
only interacted workbook records enter this pilot. The live source-only
precommit (`meditation_6c2c1c11b6f84933`) confirmed all roles without optimization.
The no-op, BF16 conversion and frozen-feature preprocessing gates passed; native
preparation is still active, so training/improvement/admission remain unproven
for this run. The heldout score evaluates literal technical text, not broader
reasoning ability. Evidence: `var/knowledge/agent-llm-curriculum/pilot-v1/`.
The deployed intake changes passed 1,099 core tests (four skipped) and 46 native
numeric/capture tests. No worker/support code is being edited during this run.

The current requirement-by-requirement audit is
`var/knowledge/agent-llm-curriculum/pilot-v1/completion-audit.json` (29 entries,
with evidence paths/symbols checked). It distinguishes implementation tests,
previous full-sized native proof and still-missing positive admission evidence.
The active Supervised run must supply matching update, statistical admission,
immutable autosave, serving-control and parent-restoration receipts before the
goal can be declared complete. Native text-likelihood gains alone do not prove
broader reasoning gains.

**Supervised pilot result and numerical correction:**
`meditation_55a9bb1e732d45fe` completed 15 disposable probe updates and 75
retained updates, then paused on a development plateau. Development loss moved
from 0.7665509462 to 0.7671482044 (worse). Total exported BF16 changes from its
untrained baseline were 24,883,699 elements, L2 delta 1.1106500940. Native loading
and exact parent restoration passed; the original Q5_K head hash matched before
and after. No admission attempt consumed the eight heldout groups and no base
was admitted or activated. Matching receipts are `supervised-native-result.json`
and `total-head-delta.json` in the pilot folder. Its exact executed worker and
package were archived under `executed-runtime/` before numerical changes.

A read-only CPU diagnostic on 32 train and 32 development target rows compared
native-style F32 activations/logits to the trainer's BF16-rounded arithmetic.
The latter changed sampled development loss by +0.0011176448, larger than the
meaningful-improvement threshold. This is evidence of an objective mismatch, not
proof that precision caused the plateau. New runs now use a F32 compute shadow
containing exactly the BF16 values that serving reads. Gradients accumulate in
CPU F32 RAM so that repeated backward passes do not require a second full GPU
gradient. Exported weights remain BF16 and the original backbone remains frozen.
Checkpoint precision is explicit; changing it on a resume is rejected.

New runs also measure forward KL from the untrained BF16 head on fixed development
target features: mean <= 0.005 and per-token maximum <= 0.02. Missing, nonfinite,
excessive or mismatched measurements cannot support admission. The receipt binds
both head hashes and the contract; candidate identity and activation validation
retain that binding. Calibration can use this development retention evidence,
never heldout scores. Three consecutive unproductive regression/retention
segments now request fresh evidence after restoring the best checkpoint,
preventing repeated best-reset trials from looping forever. Meaningful valid
improvement resets that streak; no global update limit is introduced.
Hardware verification of the updated arithmetic remains required before claiming
the new default's measured resource peaks or quality result.

**Updated native run started:** `meditation_c57925a4c8484495` was started through
the real terminal application's prompt in Supervised mode, using the same 16
source chats and untouched heldout reservations. It records `native_f32` and the
fixed KL contract, with CPU F32 gradient accumulation and BF16 saved weights.
The updated code passed 1,108 core tests (four skipped), 53 native numeric tests
and six focused UI tests. Its exact deployed package/modules are archived in
`var/validation/meditation-native-f32-20261114/executed-runtime/`. Native
preparation is active; do not change bound worker/support code or claim its new
hardware peaks, improvement, admission or activation before matching receipts.

**Evaluation offload recovery:** the first FP32 probe completed five actual
updates, changing 2,211,819 exported BF16 entries. Peak optimization VRAM was
5,396,170,240 bytes and peak host RSS 14,339,153,920 bytes, inside the unchanged
6 GiB / 24 GiB caps. Its two-head KL evaluation then raised a VRAM-limit error;
the original model was restored. The checkpoint and five successful updates are
preserved. The unchanged worker source, exact VRAM raise frame, pinned input
hashes, missing committed evaluation result and absence of a matching live
worker authorize a read-only CPU evaluation recovery. A new output directory
preserves the failed GPU receipt; no optimizer step is replayed.

CPU evaluation must first reproduce the committed GPU development baseline's
overall and per-group losses within 1e-5 on the same head/features. The original
baseline remains the calibration reference, so a completed partial probe keeps
its input identity and optimizer count. The CPU parity receipt is retained with
any eventual admission. A failed CPU parity or fixed host-memory limit stops
rather than oscillating between devices. The numerical worker is unchanged;
only the controller/evaluation recovery and frontend source-scope display are
updated. Actual CPU baseline parity and candidate evaluation have passed. The
first retained 25-update segment also passed native loading and exact parent
rollback; development loss worsened slightly to 0.7666152375, so this is no quality
admission. Candidate CPU evaluation used peak RSS 7,859,847,168 bytes and no GPU
allocation. The run subsequently stopped after three development plateaus and
75 retained updates, with the original model restored. Positive end-to-end
admission remains unproven; the first-segment evidence is
`first-retained-segment-result.json` in the current validation folder.
The complete result is `native-f32-result.json`, with an independent final-to-
untrained weight comparison in `total-head-delta.json` in that same folder.
Evidence: `evaluation-failure-result.json` and `evaluation-recovery-preflight.json`
under `var/validation/meditation-native-f32-20261114/`.

**Capture-cache resume correction:** the first CPU resume revealed that changing
calibration policy metadata also changed the frozen-feature cache identity. Its
redundant read-only capture was cancelled and the parent restored before any new
training. A sealed completed capture can now be reused when every arithmetic,
model, tokenizer, dataset and execution-plan dependency is identical; optimizer
policy source hashes remain provenance rather than capture dependencies. Its
original complete configuration and feature contract are retained, while the
current controller's native binding records the updated policy source. A dead
superseded read-only capture can be reconciled only against a sealed matching
completed capture; its partial files and failed receipts remain untouched.
Unknown optimizer effects still block resume. The original five-update training
operation ID, start timestamp and result hash are unchanged. The real application
resumed the same run after 47 focused tests and a successful package build.
Receipts: `capture-reuse-preflight.json`, `capture-reuse-resume-receipt.json`,
and `capture-reuse-qa.txt` in the current validation folder. No new base is yet
admitted or activated.

**Recovery amendment (14 November):** after hibernation, the user authorized
finishing Lucifer's interrupted tool implementation as well as continuing this
goal. The saved tool run is `interrupted`, and no model/backend process survives;
the half-hour waiting phase has ended. Its explicit final task was to implement
vision/screen capture and stop. Preserve its additive tool/tests/docs changes and
finish the missing GNOME portal backend rather than replaying completed commands.
The former backend service was transient and is no longer registered after the
restart; recreate an owned backend only after combined verification.

The recovery audit also requires actual native heldout scores for admission,
immutable tokenizer/module binding, public supervised task history, and a final
source-cache purge after owned workers exit. These corrections are implementation
work, not evidence that a full-size optimizer update or quality gate has passed.

Recovery progress: the portal backend and document import/read interaction hooks
are implemented. Native admission now reserves a complete source panel before
effects, scores the actual candidate engine, and reconciles that measurement
before further training after interruption. Tokenizer files and reference modules
are bound to cache and candidate identities. Supervised examples preserve bounded
public preceding context and window long assistant answers without duplicated or
dropped targets. Revoked source caches are purged again after owned workers and
controller cleanup finish. Focused core checks passed (77), native numeric/capture
checks passed (42), and a later native-admission recovery test passed separately.

Native validation v5 is `meditation_3707832fd56a4ad0`, using the three existing
consented chats in the precommitted pilot selection. Its real native no-op gate
passed, parent/BF16 backbone features were bit-identical, and BF16 conversion
maximum KL was 0.000171942 (limit 0.02). These are preprocessing gates; at this
checkpoint an optimizer update and activation are still unproven. Full native
capture is running. Three source groups cannot satisfy final quality admission's
independent heldout-panel requirement, and that rule will not be lowered.

**Native v5 result:** 25 successful updates changed 65,713,997 exported BF16
elements, L2 delta 1.6263993094 and maximum delta 0.00048828125. Peak optimizer
VRAM was 5,387,568,640 bytes; peak RSS was 11,878,277,120 bytes. The updated head
loaded in the native engine, changed 2,973,431 diagnostic logits, and the restored
parent was bit-identical (maximum logit difference zero). Original head hashes
before/after matched. The original serving model `lucifer` was restored. The
immutable receipt is `var/validation/meditation-native-20261113/native-head-acceptance-v5.json`.
The small development projection improvement (2.96701779 -> 2.96655722) is not
statistical quality admission. The run stopped `awaiting_evidence`; no base was
admitted or activated. Remaining acceptance work includes starting Unsupervised
through the real application, combined final QA/deployment, and honest handling
of the currently insufficient independent source panel. Do not claim a passing
native autosave/admission path from fixtures or this three-group diagnostic.

**Application-started all-chat run:** `meditation_d8e993f9c6d948eb` was started by
selecting Unsupervised/all eligible chats and clicking Start in the real
`LuciferApp` connected to the live backend. A dedicated GNOME terminal was opened.
It consumed six independent groups from seven selected chats (one empty), scored
9,490 heldout targets, completed 25 updates, passed native loading/rollback and
restored the normal model. It paused honestly with two fresh heldout groups
against the required eight. No base was admitted or activated.

**Recovery/calibration audit correction:** memory fallback and initial rate
probes are now implemented, with further native validation pending. A typed
failure receipt must prove unchanged immutable inputs and no committed result;
the controller must observe its own worker exit before discarding private trial
files. At most three microbatch reductions preserve the actual effective batch
through increased accumulation and full sampler wrap. CUDA/VRAM failures alone
qualify; uncertain, committed, fixed RAM-state or disk effects do not.
Initial calibration tests three rates (0.5x, 1x, 2x the conservative starting rate,
within pinned bounds), five updates each, solely on training/development data.
Probe state and receipts support interruption, partial continuation and disposal;
the retained optimizer starts from the original BF16 baseline at the selected
rate. Probe updates are separate from retained update counters. Partial probe
checkpoints roll forward after durable replacement to bound storage. Native
worker/support-module hashes bind these changes to subsequent runs; earlier
native receipts do not imply these later paths were tested on the full head.

**Latest user amendment:** Meditation has two explicitly started UI modes,
Supervised and Unsupervised, and no mandatory human review step. This supersedes
references below to requiring reviewed Meditation examples. Supervised mode uses
provided answers/corrections and eligible final-answer targets. Unsupervised mode
uses causal next-token prediction on consented chat/document text without answer
labels or approval requirements. Both preserve consent, source identity, heldout
evaluation, resource limits and verified rollback. Optional source/result
inspection is informational. Activation remains a separate explicit action after
automatic gates, with no manual review workflow.

**Integration amendment:** the user has an active Lucifer chat developing a
tool. Wait for that chat to finish; do not pause, cancel or restart it. Then
inspect its completed tool/code artifacts and integrate them with Meditation,
reconciling shared files and testing the combined result before native acceptance.
Do not replace its completed work with an earlier source snapshot.

**Timed continuation and start authorization:** the user requested that active
work stop between 30-minute checks of the running Lucifer tool chat. Keep that
chat untouched. After it finishes, integrate both implementations, complete
combined/native verification and start an unreviewed **Unsupervised** Meditation
session by interacting with the application. No additional review approval is
required. Automatic integrity, evaluation and rollback gates remain in force.
That waiting phase ended after hibernation and the subsequent recovery
authorization. The requested application-started run has now been performed;
its plateau does not justify repeatedly starting identical training.

## Outcome and completion criteria

Add a **Meditation** terminal page alongside Agent, Cortex, Sleep and History.
Meditation trains the existing output head of the primary Lucifer model using
consented training chats and source-bound documents used within those chats.
It must never redirect a native-head job to the separate compact training model.
The backbone, Engram, input embedding, final norm, vision and draft remain frozen.

A run has no default time, round or optimizer-update limit. It consists of finite,
checkpointed segments whose parameters respond to development measurements.
The first checkpoint passing the fixed improvement and retention contract is
automatically saved as a new immutable **base revision** and ends the run.
Saving is separate from activation. Before activation is offered, evaluate the
candidate with the real loader and rehearse restoring its original parent.
Activation remains an explicit action after these gates; rollback remains
available. A run is allowed to wait for evidence or fail honestly instead of
claiming that indefinite training guarantees improvement.

Completion requires: a real update to the installed 661,913,600-element head;
before/after tensor hashes and numerical deltas; measured optimizer steps;
source and partition receipts; heldout/regression evaluation against the exact
parent; candidate load and rollback receipts; and a usable terminal page/API.
Unit tests or a miniature fixture alone do not establish native-model success.
Preserve original weights and every existing expert artifact.

## Existing integration and gaps

- `lucifer/tui.py` uses CSS-class pages; add Meditation/F6 without changing
  F2 Agent, F3 Cortex, F4 Sleep or F5 History. `lucifer/cli.py` selects initial
  pages, and `lucifer/client.py` owns HTTP calls.
- `lucifer/learning.py`, `adaptive_learning.py`, `curriculum.py`,
  `adaptive_validation.py`, `knowledge.py` already provide consented sources,
  reviewed targets, source partitions and model lifecycle patterns. Reuse their
  contracts without changing Sleep's current LoRA worker.
- `lucifer/api.py` owns shared control locking, bootstrap revision caching and
  telemetry. Meditation progress must use this heartbeat, with no extra polling
  loop or hardware probes in render/chat paths.
- `lucifer/inference.py` owns profile initialization and process lifetimes.
  Current Strata profiles reject adapters, and expert identity currently lacks
  an immutable base-weight revision. A renamed model ID is not compatibility.
- `lucifer/adapter_map.py` currently renders all registered experts, including
  incompatible other-base experts. Filter the Cortex projection and pickers
  using shared compatibility decisions; retain the full historical registry.
- `KnowledgeService` persists attached documents. Web reads are presently
  transient tool results, so downloaded/read public sources need immutable
  conversation-bound ingestion receipts before they can train.
- Native hidden-feature export and head optimization require implementation
  and numerical validation. The installed engine's debug dumps may provide the
  necessary final tensors without rebuilding it; this must pass the same
  reconstruction/logit tests as a new exporter. If unavailable, obtain complete
  matching source and build an isolated engine. Never substitute earlier hidden
  streams or draft features.

The hardware/capacity and source audit is in
[BASE_MODEL_TRAINING_FEASIBILITY.md](BASE_MODEL_TRAINING_FEASIBILITY.md).

## Data eligibility and lineage

1. Default selection is saved chats with both logging and training enabled.
   Accept an explicit conversation-ID selection. Keep that selection visible.
   Snapshot source content and consent at preparation, each refresh, resume,
   evaluation, autosave and activation. Deletion, supersession or withdrawal
   invalidates every dependent descendant and blocks future promotion.
2. In Supervised mode, use provided corrections/answers and saved public final
   answers as assistant targets. Prefer corrections and positive feedback when
   present; omit an explicitly rejected final without a correction. This does
   not require a separate Meditation review action. In Unsupervised mode, use
   saved public user/final text and extracted document text for genuine causal
   next-token prediction without answer labels. Record authority explicitly:
   user text/correction, reviewed final, public model final, literal document or
   model-derived visual description. Source identity does not establish factual
   truth. Never save or train on provider reasoning, tool traces or failed partial
   output. Document instructions are quoted source data; they cannot alter the
   run, consent, hyperparameters or gates.
3. Attachments, linked files and public pages/documents actually read/downloaded
   in a selected chat receive a document SHA, chunk hashes, original URL/path,
   content type, conversation ID and interaction receipt. Record which chunks
   were returned to the model, cited or explicitly read. Do not mark every
   attachment as interacted merely because it exists in the chat.
4. Reuse existing readable-document extraction for PDF, CSV/Excel, Word and other
   supported text formats. Images/video use extracted text/captions with their
   model-derived authority/provenance visible; they do not trigger a mandatory
   review gate. Initial head training uses text examples from those sources;
   do not imply visual encoder training.
5. Deduplicate by content hash across URLs/chats and partition by independent
   document/conversation/source group before tokenization. Keep train, development
   and admission source groups disjoint across refreshes. Block prompt/target
   duplicates and variants crossing reserved groups. Reuse valid replay data.
6. Tokenize using the primary model's pinned tokenizer. Supervised examples use
   its chat template and assistant-only masks. Unsupervised examples tokenize
   raw public text with an explicit special-token policy and train all valid
   next-token positions. Stream long documents through bounded overlapping
   windows without dropping targets or counting overlap targets twice. Save
   next-token alignment (feature at t, target at t+1) and the exact masks.
   Bind each feature shard to the frozen model, norm, tokenizer/template,
   exporter, source, positions, masks and consent revision.

Source ingestion should occur as part of a successful owned read operation and
reuse that operation's identity on retries. Do not refetch a URL to manufacture
a training receipt or silently broaden runtime file/network access. Private-chat
sources remain transient. With training off, reading does not opt a chat in.

## Native head artifacts and engine gates

The installed independent head is Q5_K `[129280,5120]`, 455,065,600 bytes. The
BF16 training representation is 1,323,827,200 bytes. Store a revision as a small
head sidecar plus a complete manifest referencing the unchanged parent shards;
do not copy 258 GB or overwrite the original checkpoint. A new base revision
retains the same architecture and parameter count.

Implement native-pack helpers for bounded tensor reads, checksums, Q5_K decoding,
row-major BF16 export and safe manifest replacement. Paths resolve relative to
the pack's original `model_dir`. Replace the old `head.weight` row rather than
adding a duplicate, and refuse ambiguous names, ranges or unsupported formats.

Before optimization:

1. Load a byte-identical Q5_K head sidecar and compare real logits to the parent.
2. Verify engine-consistent dequantization and unchanged BF16-head logits/loss.
   Quantify conversion drift separately from any improvement due to training.
3. Capture the exact post-mHC/final-normalization input to the output head.
   Force effective chunk size zero (`--prefill-chunk 0`, `DS4_CHUNK=0`), disable
   draft loading/speculation, and use teacher-forced tokens. Reconstructing from
   existing debug dumps is allowed only with numerical parity evidence.
4. Check masks/positions on independently inspected tiny examples. Features
   reproduced by another architecture or synthetic examples cannot pass this
   native gate.

## Finite segments within an indefinite run

Persist a versioned controller state: seed/RNG, exact parent, data revision,
successful updates, optimizer state, sampler cursor, learning rate, microbatch,
accumulation, segment budget, best development checkpoint and numerical metrics.
No default global update/time cap. Cancel/pause and source/resource failures
remain effective at segment boundaries and owned process cancellation points.

Use an explicit CPU FP32 master/head optimizer with a GPU F32 compute shadow of
the BF16 serving head, F32 activations/logits and CPU F32 gradient accumulation.
The earlier BF16 arithmetic path remains explicit for matching legacy checkpoints;
it is not the default for newly created runs after the numerical correction.
Begin with a small microbatch (32 feature rows or the measured smaller fit),
gradient accumulation and a conservative learning-rate trial on training data.
Calibrate candidate rates on disposable train/development probes, never on
admission responses. Record the chosen values rather than claiming universal
optimal settings. Persistent mixed state is about 10.591 GB; allow for temporary
allocations and an extra CPU gradient copy.

Initial pilot admission caps: 6 GiB VRAM, 24 GiB RAM, 50 GB additional disk,
with a free-space reserve. These are limits to measure/enforce, not proven peaks.
Serialize capture, unload, optimize and evaluation. Do not train concurrently
with the streamed inference cache. Restore the prior model/profile/adapter on
every exit, and hold the job busy until restoration actually finishes.

Dynamic policy (version and persist every decision):

- Start with short segments (initially 25 successful updates) and evaluate on the
  same frozen development revision after each segment. Skipped/nonfinite steps
  do not count as successful updates.
- If development loss improves meaningfully and retention/KL checks stay inside
  their fixed envelope, keep the rate and grow the next finite budget gradually
  (for example ×1.25, within a bounded 10–100 update segment range).
- On regression, nonfinite gradients or excessive drift, restore the best valid
  checkpoint, reduce the rate (for example ×0.5 down to a recorded floor), and
  shorten the segment. OOM recovery reduces microbatch while preserving an
  explicitly recorded effective batch; no unbounded retry loop. Three consecutive
  unproductive regression/retention segments pause for fresh evidence as well.
- On a plateau, reduce the rate and refresh only independently verified practice
  from eligible training groups. If no useful data/evidence exists, report
  `awaiting_evidence`; never lower improvement thresholds or train on holdout.
- Keep disk use bounded through rolling unadmitted checkpoint retention. Preserve
  parent, best checkpoint, resume state and admitted revision. A long run must
  not grow its checkpoint/cache storage without limit.

The effective rate, batch, segment length, reason for each change and actual
successful updates are visible in Meditation. The run's overall progress remains
indeterminate; only known finite stages receive percentages.

## Improvement, autosave and rollback

Pin both the original Q5_K parent and an untrained BF16-head baseline. Train loss
movement or a changed checksum is evidence of updating, not evidence of better
model performance. Use group-level heldout next-token loss plus independently
scored target tasks, tool formatting, instruction following and repetition/loop
controls. Require complete measurements and predeclared non-regression limits.

Use development results for all tuning. Admission evidence stays sealed from
generation, optimizer and parameter selection. To avoid indefinite repeated
testing against a convenient final panel, precommit fresh disjoint admission
groups per eligible attempt, with a sequential error budget such as
`alpha_k = 0.05 / (k * (k + 1))`. A statistical improvement rule must use a
valid paired/group-level bound, not label arbitrary margins as significance.
If too few independent groups are available, pause for evidence. Never reuse a
failed final panel as training or turn a statistically unknown result into pass.

At the first passing checkpoint:

1. Atomically seal the candidate head, full manifest, provenance, lineage,
   measured tensor delta, optimizer-step and evaluation receipts.
2. Load it with the real primary engine under a temporary evaluation profile.
   Reserve the additional BF16 serving memory (about 0.869 GB above current head)
   by reducing expert cache/reserve as measured; reject an OOM configuration.
3. Run end-to-end evaluation. Restore the parent/profile/adapter and compare a
   fixed rollback probe to the pinned parent result. Record verified restoration.
4. Only after successful restoration mark the immutable base revision
   `admitted`/saved. End the indefinite loop. A failing loader, evaluation or
   restore cannot produce an activatable revision.
5. Offer explicit Activate and Rollback controls. Activation revalidates source
   consent, artifact hashes, parent identity and measured gates. Journal intent
   before effects and reconcile an uncertain crash instead of repeating it.

Maintain a saved base-revision registry separate from LoRA adapters; keep an
atomic active/previous revision pointer and append-only ancestry. Persist profiles
without making a YAML edit silently change a running model. Restart must recover
the last committed revision and report incomplete restoration truthfully.

## Expert compatibility and Cortex behavior

Introduce one shared compatibility result used by backend activation, Cortex
nodes and dropdowns. Bind experts to base model, architecture/tokenizer and exact
weight revision; unknown compatibility does not become true through renaming.
For Strata, retain its actual adapter-support restriction. A head-only revision
must not automatically inherit old compatibility just because its backbone is
shared; require an explicit compatible-artifact contract and validation.

After activating a new base, remove incompatible experts from the default Cortex
visual and selection controls. Preserve their records, files, lineage and history;
do not delete trained weights. Saving a candidate alone must not hide experts
belonging to the still-active parent. Rollback restores the corresponding view.
Show a small count/reason for hidden incompatible experts and retain an explicit
history view. Base candidates themselves appear as base revisions, not experts.

## Interface and work ownership

Planned HTTP contract:

- `POST /meditation/start` with optional `conversation_ids`, `base_model`, `seed`,
  `dry_run` and an idempotency key; no global cap by default.
- `GET /meditation/runs`, `GET /meditation/runs/{id}`;
  `POST /meditation/runs/{id}/cancel`, `/resume` (idempotency key).
- `GET /meditation/bases`; `POST /meditation/bases/{id}/activate`;
  `POST /meditation/rollback`. Mutation receipts are idempotent and scoped.
- Bootstrap contains `jobs.meditation`, base revisions and compatibility metadata;
  existing telemetry carries job progress without extra registry enumeration.

Terminal `/meditation start|status|cancel|resume|bases|activate|rollback` and F6
open the Meditation page. Actions: Start, Pause/Cancel, Resume, Sources,
Results, Activate and Rollback. Results are available for unadmitted runs too.
Show source eligibility, native head
scope, stage, measured loss/delta, dynamic parameters, admitted revision and
restoration status. Training is not preselected for chats that have it off.

Implementation areas:

| Area | Files/modules |
| --- | --- |
| Native tensor I/O, feature contract and optimizer worker | `lucifer/head_pack.py`, `lucifer/head_capture.py`, `lucifer/meditation_worker.py` |
| Controller, source lineage, base candidates, lifecycle | `lucifer/meditation.py`, `meditation_sources.py`, `meditation_policy.py`, manager locking |
| Calibration, bounded recovery and storage preflight | `lucifer/meditation_calibration.py`, `meditation_recovery.py`, `meditation_storage.py` |
| Source interactions and ingestion | `knowledge.py`, `agent.py`, `tools/web.py`, chat source receipts |
| Compatibility and profiles | `inference.py`, shared compatibility helper, candidate profile registry |
| API and telemetry | `schemas.py`, `api.py`, shared bootstrap/cache revisions |
| Terminal | `client.py`, `tui.py`, `cli.py`, `adapter_map.py` |
| Documentation/evidence | README, provenance, new Meditation guide and validation receipts |

## Delivery sequence and verification

1. **Plan and persistent goal:** save this plan, start a goal with these acceptance
   criteria, then implement without requiring the user to restate authorization.
2. **Native gates:** prove no-op sidecar loading and exact features; implement
   bounded head I/O and worker, with a real optimizer test and resume hashes.
3. **Orchestration/data:** add consented/interacted source collection, partitions,
   adaptive finite segments, sealed evaluation and candidate/base registry.
4. **UI/compatibility:** add Meditation page/API/commands and base-aware Cortex
   filtering; preserve Sleep behavior and expert history.
5. **Failure and integration tests:** source revocation, duplicate operations,
   checkpoint corruption, cancellation/OOM, busy model exclusion, profile restore,
   activation rejection without evidence and rollback across restart.
6. **Integrate Lucifer's tool:** after its active development chat finishes,
   inspect the saved public task results and actual changed files, determine the
   tool's entry points, dependencies and permission/source contracts, then merge
   its completed implementation with these changes. Preserve both features,
   resolve any shared-file edits and run focused plus combined regressions.
7. **Native acceptance run:** use actual eligible training data, measure changed
   head values/hash, unchanged frozen weights, heldout performance and resource
   peaks; load candidate and restore parent before any activation. A small-model
   fixture is useful test coverage but cannot replace this evidence. Do not
   interrupt an unrelated active chat to seize the GPU; queue the acceptance
   stages until the model slot is idle.
8. **Finish:** document measured capability and limits, retain the saved original
   and candidate receipts, run appropriate regressions and mark the goal complete
   only when the requested native feature and its acceptance evidence exist.

## Progress

- Persistent goal active. Supervised and Unsupervised modes, F6 page, commands,
  API, shared telemetry, native-head worker, source ingestion/lineage, immutable
  revisions and base-aware Cortex filtering are implemented.
- Dynamic finite segments, source refresh, development-panel reset, first-passing
  admission recovery, immutable candidate sealing, explicit activation/rollback,
  consent withdrawal and ancestor checks have focused automated coverage.
- Native helpers passed CPU optimizer/resume, decoder, manifest, tokenizer and
  capture-contract tests. Actual primary head export completed without changing
  original bytes. These are not yet full-model optimizer acceptance evidence.
- Native attempts exposed real deployment differences before optimization: the
  tokenizer directory contains a Strata configuration rather than a Hugging Face
  tokenizer, and GPU-resident MoE returns bypass inner residual debug hooks.
  Capture now uses the runtime's exact tokenizer and the outer `DS4_DUMP_H`
  trajectory with final `f_pre`, checking native logits at unchanged tolerances.
  Failed attempts restored the original inference profile.
- Full-sized native feature/optimizer/loading/rollback tests passed, including
  the calibrated run. Positive quality admission remains unproven. Evidence lives under
  `/home/Lucifer/Soul/var/validation/meditation-native-20261113/` and the individual
  managed Meditation run directories. Current source data has fewer than eight
  independent heldout groups; no statistically admitted base or activation is
  claimed merely from a successful update test.
