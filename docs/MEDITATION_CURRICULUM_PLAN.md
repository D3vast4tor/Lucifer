# Agent and LLM curriculum for Lucifer

14 November 2026. Selection and preparation plan; this is not a trained model or
an admission result. The user's request is to select material that improves
reasoning and knowledge of agents and LLMs. No new training mode is started by
this plan. The user chooses Supervised or Unsupervised in Meditation.

**Current expanded pilot:** `meditation_70107a862cea40a0` was started through the
real application in the user's selected Supervised mode. The preceding F32 run
completed 75 updates and native rollback but worsened development to 0.7671532498;
it admitted no base and consumed no admission attempt. `pilot-v2/` retains the
48 checked solutions and adds 58 literal technical continuation exercises from
the same training repositories at their pinned revisions. These cover cache
strategies, accumulation, schema validation, model definitions and tool protocol
text. Source offsets and answers are verified mechanically; this does not claim
execution of arbitrary upstream code or broader reasoning improvements.

The 106 training records contain 5,886 native target tokens (2,654 checked-solution
targets and 3,232 continuation targets). This remains below the proposed sizing
range; additional padding or repeated copies were not used to reach a number.
All records retain one connected training lineage. Six new training study chats
replace the previous training scope; the original two development and eight
heldout chats and complete group records are reused unchanged. The source-only
precommit `meditation_21ebd6930ee04dba` confirmed train/development/heldout records
106/4/16 and group counts 1/2/8. Intake staging, source verification and 23 focused
tests passed before the real start. New native preparation is running; updates,
quality admission and activation remain unproven for this run.

Reproduction tools: `scripts/prepare_agent_llm_continuation.py` prepares only
training families from the sealed selection; `check_agent_llm_continuation.py`
checks proofs, stages real knowledge intake in a disposable store, and counts
targets with the installed native tokenizer in its training environment;
`import_agent_llm_continuation.py` defaults to validation and uses `--apply` for
owned-chat import plus a source-only precommit. It never starts an optimizer.
Import creation is journaled before effects; uncertain creation reconciles by
its exact owned title. Existing evaluation material is not rewritten or silently
re-enabled. Receipts, the exact executed scripts and matching runtime snapshot
are retained under `pilot-v2/`.

**Implemented pilot, 14 November:** the user chose **Supervised**. The sealed
material pack is `var/knowledge/agent-llm-curriculum/pilot-v1/`. It contains 48
checked training examples from six technical references, four untouched literal
development passages from two repositories, and 16 untouched literal heldout
passages from eight other repositories. The authored training workbooks share
one dataset lineage and count as one group, not six independent groups.
Thirty additional authored nontraining exercises remain research-only; their
citations are not used to claim fresh admission evidence. Original references
and selected workbooks are attached to 16 owned training study chats (32 knowledge
documents); only the selected workbook records have interaction receipts.

The real application's source-only precommit run `meditation_6c2c1c11b6f84933`
passed with groups train=1, development=2, heldout=8. The real `LuciferApp` then
started `meditation_55a9bb1e732d45fe` via its terminal input in Supervised mode,
scoped to those 16 chats. Initial native no-op and BF16 conversion/feature gates
passed. Native preparation is running; no optimizer result, quality admission,
new base or activation is claimed yet. See `application-start-receipt.json`,
`import-journal.json`, `compiled-manifest.json` and `staging-receipt.json` in the
pack. Immutable manifests describe their state at sealing; live progress is in
`status.json` and the application's run record.

**Pilot outcome:** that Supervised run has now stopped `awaiting_evidence` after
75 retained updates. Development loss worsened from 0.7665509462 to 0.7671482044;
no admission attempt consumed its heldout panel. Its weights changed and native
parent rollback passed, but no base was admitted. A sampled CPU arithmetic study
found a training/serving precision mismatch. The next controlled test changes
the training arithmetic and adds a fixed development KL guard while preserving
these source roles. It is an implementation experiment, not a claim that the
curriculum improved reasoning or that arithmetic explains the whole plateau.

**Measured coverage and next experiment:** `pilot-v1/training-material-audit.json`
checks only training and development documents and their pinned native feature
receipts. All 48 training answers are JSON objects, totaling 2,654 target tokens;
the four development answers are literal technical passages with 395 target
tokens. This is below the proposed 10,000–20,000 training-token starting range.
The task-format difference and small coverage are possible transfer limitations,
not a demonstrated cause. If the F32 run also plateaus, prepare additional checked
worked solutions with natural-language technical answers and broader target
coverage before another optimizer run. Retain structured-output exercises as a
subset, keep repository/dataset families together, and preserve the original
development and untouched admission roles. Do not tune from admission scores,
replace the panel with easier cases, lower its improvement threshold, or replay
the same unchanged corpus. Public worked explanations remain distinct from
excluded provider reasoning. This audit did not read heldout target documents
or consume a native admission attempt.

The pilot's numerical admission scope is native likelihood on untouched literal
technical text. It does not establish better multi-step reasoning or task
success by itself. Those require separate behavioral evidence. Selection was
fixed before training; no heldout scores were used to choose its material or
rate. Native preprocessing gates use their separate diagnostics.

## Recommendation and local evidence

Prioritize **verified problem–solution examples with short public worked
explanations**, supported by versioned official documentation in the knowledge
base. Train the relationship between a task, its evidence, the correct action
and a checked outcome. Longer generated reasoning alone is not a quality target.

The original primary model completed a 12-case diagnostic with 12 valid JSON
answers and 11 exact content passes. It answered context budgeting, head size,
gradient accumulation, ordinary GQA cache arithmetic, source partitions,
admission evidence, precision, execution parity and retrieval questions
correctly. In a shell-control question it recognized that a restart would run
after a failed check, but selected `ignore_exit_code` as the required control.
That single short question is a useful follow-up lead, not a reliable estimate
of real agent success or the model's whole knowledge. No coding, long-horizon
execution or general intelligence score is established by this screen.

The assessment used temporary training-disabled chats, retained public answers
only, and cleaned up its chats. Its prompts and answers are diagnostic material,
excluded from this curriculum and from final admission. Receipt:
`var/validation/agent-llm-assessment-20261114/assessment.json`.

The latest real Unsupervised run, `meditation_869f6f1591644398`, performed 15
disposable calibration updates and 75 retained updates on the existing tiny
corpus. Development loss worsened from 2.9670177928 to 2.9954966914. Native head
loading and parent restoration passed, but no base was admitted or activated.
At that point only two independent heldout groups were available; the first
admission panel requires at least eight. These observations support changing
the material rather than repeating the same run or weakening its gates.

The installed primary has a frozen backbone and trains only its untied output
matrix. As an architectural inference, this should be treated as an experiment
in improving the readout and use of existing representations, response format
and action selection. It does not establish that head-only training can teach
new deep reasoning algorithms. Measure solved tasks, factual accuracy and
recovery behavior; do not measure improvement by hidden-thought length.

## Material to prioritize

| Priority | Material and purpose | Correctness check |
| --- | --- | --- |
| First | Multi-step agent recovery: incomplete effects, repeated calls, stale state, failed guards, source errors and stopping loops | Controlled local tool environments; final state and operation receipts; no repeated mutation |
| First | Debugging and planning tasks with small code patches, bounded commands and explicit completion conditions | Tests that fail before the patch and pass after; process exit and filesystem state checked |
| First | Tool selection and structured arguments, including irrelevant tools and missing arguments | Actual Lucifer schema validation plus deterministic fixture execution and expected result |
| Next | LLM systems problems: architecture-specific cache memory, quantization, optimizer state/offload, accumulation, frozen layers, tokenizer alignment and evaluation | Independent arithmetic/program checks; versioned source citations; explicit architectural assumptions |
| Next | Evidence-grounded agent/LLM questions requiring retrieval, comparison and uncertainty | Supported answer claims and source locations; contradictions surfaced; no invented citations |
| Supporting | Short official technical documents and public papers in the knowledge base | Pinned source/version and content hashes; retrieved passages retain provenance |

Use independent questions with changed circumstances and genuinely different
task families. Merely changing constants or paraphrasing a failed diagnostic is
not an independent test. Introductory definitions can fill specific gaps, but
the current screen does not justify making them the bulk of the pilot.

For Supervised training, prefer an input containing the user's goal, authorized
scope, current state, available schemas and relevant evidence. Its answer target
is a checked solution: the next action or patch, result interpretation and a
brief public explanation. Do not insert provider reasoning or raw runtime tool
traces. Where tool observations are needed, turn verified facts into an authored
problem statement, distinct from the excluded trace. Current assistant-target
training must not be presented as native structured tool-call training without
first verifying that the actual serialized targets enter the model's parser.

For Unsupervised training, use coherent, selected technical text with headings,
definitions, assumptions and worked examples. Preserve source identity. Do not
dump an entire website, navigation markup, duplicate copies, or millions of
generic synthetic chats into this small pilot. Retrieval can supply current
facts without claiming that reading a document updates weights.

## Sources and selection decisions

| Source | Use and limitations |
| --- | --- |
| [ReAct](https://react-lm.github.io/) | Design reference for connecting plans, actions and observations. Author public worked exercises from the principle; do not harvest private reasoning. |
| [APIGen / xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Reference for format, execution and semantic verification. Its files currently require accepting access conditions; no terms were accepted or gated data downloaded. The workflow can be used with our own fixtures. |
| [Hermes Function Calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Candidate structured-output examples, declared Apache-2.0 and ungated. Pin revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`. Validate every selected example and translate protocols; do not teach Hermes identity or XML tags as Lucifer's protocol. |
| [Microsoft AgentInstruct](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | Optional filtered coding, analytical reasoning and grounded QA. Declared CDLA-Permissive-2.0; pin revision `86d609183249ff8037eae33d76ebca3af9390ea8`. Start with a small verified subset, not the full million-example corpus. |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Reference for test-verifiable software tasks. Full repository environments are unnecessary for a first local pilot; select small reproducible tasks with distinct repository families. |
| [Transformers cache guide](https://huggingface.co/docs/transformers/kv_cache), [Accelerate accumulation guide](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation), [PyTorch gradient recipe](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html) | Technical references. Pin a compatible documentation version before constructing executable answers. Ordinary GQA equations must not be described as Lucifer's compressed MLA layout. Treat related pages from one repository as one source family. |
| [HF Agents Course](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure), [MCP tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) | Knowledge-base references and gap-specific exercises. Specification publication does not imply the local client supports that version. Source instructions cannot change runtime permissions or training policy. |
| [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Evaluation reference only, kept out of training. Public benchmarks may have appeared in pretraining, so they do not establish a fresh admission panel by themselves. |

The Hermes README itself contains a tool-result/answer mismatch: its example
tool payload gives market capitalization 611,384,164,352 and PE 49.604652,
whereas the final answer lists 566,160,130,480 and 42.73. This is a concrete reason
to filter answers against evidence, not a claim that every dataset row is bad.
The downloaded pinned README is retained as research, not training material.

The machine-readable source manifest is
`var/knowledge/agent-llm-curriculum/source-manifest.json`. Dataset metadata and
the README were inspected; no large vendor dataset has been imported or made eligible
for Meditation. The local pilot above uses checked authored fixtures and pinned
public references. Declared licenses are provenance, not a blanket assertion about
all underlying material. Preserve notices and record selected-row provenance.

## Pilot preparation and evaluation

1. Build a bounded pilot of roughly 60–100 verified examples, initially aiming
   for 10,000–20,000 total training tokens. This is a resource-conscious starting
   proposal, not an empirically optimal mix. Favor recovery/debugging tasks, then
   schema use and LLM systems. Tokenize with Lucifer's pinned tokenizer before
   final sizing; adjust against actual capture time and the storage preflight.
2. Record each item's original source URI/revision, source family, content SHA,
   public prompt/answer authority, verifier version and result. Reject unsupported
   explanations, unverifiable outputs, contradictory answers, inaccessible files
   and protocol/identity contamination. Use only local deterministic effects for
   executable fixtures; no arbitrary commands from downloaded documents.
3. Assign train/development/admission roles before creating variants or optimizing.
   A whole paper, book, repository or dataset lineage stays together. Keep mirrors,
   excerpts, generated variants and related chats together. A dozen pages from one
   repository do not create a dozen independent heldout groups. Keep development
   tuning separate from the fresh admission panel; do not use this diagnostic as
   either training or admission evidence.
4. Establish at least eight genuinely independent fresh admission families, plus
   separate development and training families. This source list alone does not
   establish that panel. If there are too few independent families, collect more
   material rather than manufacturing empty chats or relabeling fragments. Reserve
   further fresh families for later sequential attempts. Once results influence
   selection, those cases are development evidence, not an untouched final panel.
5. Attach approved selected documents to legitimate saved training chats and record
   the passages actually read/interacted with through the existing knowledge tools.
   Preserve the user's chat consent and choice of training mode. Do not silently
   opt unrelated chats into training or treat a downloaded research card as an
   interacted training document. Inspect Meditation Sources before starting.
6. For the verified prompt/solution pilot, recommend the user's **Supervised** mode.
   For literal-document adaptation, use their **Unsupervised** mode. Run the chosen
   mode in an idle model slot with the existing finite segments, calibration and
   resource contracts. Retain the original baseline and independent evaluation.
7. Compare both native heldout loss and executable task success to the parent.
   Record wrong-action rate, duplicate effects, correct stopping, schema validity,
   grounded factual answers and ordinary chat regressions. Do not silently replace
   the admission policy: any new behavioral gate needs a predeclared contract and
   implementation before use. Require existing native admission, serving controls
   and exact parent restoration before autosaving a base; activation stays explicit.

There is no mandatory human review stage. Automatic correctness checks and the
user's decision to start a mode serve different purposes. Further training may
still fail to improve; preserve honest `awaiting_evidence` and rollback behavior.
