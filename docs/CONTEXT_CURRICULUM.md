# Context-aware sleep and example review

Lucifer 0.4 keeps one **SLEEP / CONSOLIDATE** action in F4. The default job turns
consented task context into a verified curriculum before it trains an expert.
Provider reasoning is never a curriculum source, memory record or training target.

## Start and follow the pipeline

1. New conversations save automatically, with training off. Enable training
   consent for the conversations you want to use with `/privacy training on`.
   If you explicitly disabled saving, use `/session save` first. Feedback or corrections remain useful source
   signals; a model answer or a positive feedback label does not become an
   independently verified fact merely by being present.
   Consented USER-only material can also seed the curriculum: literal arithmetic,
   worked/error-correction tasks and quoted configuration constraints can be
   verified without feedback on an assistant answer. Unreviewed assistant replies
   never become gold targets.
2. Press F4 and select **SLEEP / CONSOLIDATE**, or enter `/sleep start`.
   No manual conversation ID, model switch or dataset-editing sequence is needed
   for the default pipeline.
3. The job recovers task context and active user constraints, drafts bounded
   proposals when a generation model is available, and validates the results.
   Brief follow-ups keep their source task context. Explicit corrections and
   superseded assumptions are recorded rather than silently mixed into targets.
4. Verified examples enter source-group partitions for training and held-out
   evaluation. Variants from one source group stay together. Pending, rejected,
   obsolete and duplicate material is withheld. Scoped memory is reconciled from
   verified evidence; conflicting records are not treated as active facts.
5. Once the configured useful-example minimum, source-task diversity and
   partition requirements are met,
   the same job selects a compatible training base, trains, verifies artifacts,
   evaluates the candidate and restores inference. Promotion stays explicit.

The default threshold is three useful verified examples from at least two
distinct source task IDs. More generally, the task minimum is
`min(2, min_verified_examples)`. Several variants of one task cannot alone satisfy
that diversity gate. The dashboard shows verified-task counts separately from
the number of raw source tasks.

Scope comes only from a literal USER declaration such as `Scope: project`; without
one, the conversation ID provides the scope. Quoted USER configuration records
what the user declared, rather than claiming that the host PC was inspected or
changed. Prior permission to access a file is not training consent. A remembered
file path or a past access claim is not documentary evidence for its contents;
claims need their current, consented source bindings and validation. Code
validation is limited to restricted pure-function tests bound to
source-approved expected cases. It grants no host tool, file or network permission.

The current stage and its counters travel through the existing telemetry
heartbeat. The view includes verified, train, held-out, pending, rejected,
duplicate, obsolete, drafted, verified-task and source-group counts. A stage percentage is shown
only when its total is known. Unknown work stays indeterminate; the overall bar
cannot reach 100% before restoration and cleanup finish. Hidden or idle bars have
no animation timers, and progress updates do not rebuild the Cortex registry.

Stages include `recovering_context`, `drafting_examples`, `validating_examples`,
`reconciling_memory`, `preparing_evaluation`, `training`, `registering`,
`evaluating` and `restoring`. The exact stages depend on the available material.

## Review pending examples

Use **REVIEW EXAMPLES** in F4 or `/sleep examples pending`. In compact terminals,
the button is labeled **EXAMPLES**. This fetch happens on demand; the heartbeat
does not fetch the review queue.

Select an example and read its literal question and complete answer. The full
64-character content hash is shown with that content. Expand **SOURCE / VALIDATOR**
to inspect its source group, scope, task, message IDs, source snapshot hashes,
active constraints, supersession count and validation method. The panel does not
dump unreviewed conversation context or provider thinking.

- **APPROVE DISPLAYED** approves that exact version. An optional correction
  replaces the complete answer; the editor supports multiline text and code.
- **REJECT DISPLAYED** rejects the displayed target. An optional reason records
  the review decision.
- **RELOAD** explicitly fetches current candidates. Filters include pending,
  verified, rejected, obsolete and all records.

Approval is an explicit human attestation validator. It is distinct from the
ordinary conversation feedback label. Reviews create immutable versions; the
prior candidate becomes obsolete when a replacement is created. The panel selects
the returned version, and obsolete entries cannot be approved from the TUI.

The request carries the exact hash that was displayed. If the candidate or its
source binding changed, the backend returns a conflict. The correction remains in
the editor. Reload explicitly, inspect the current content and decide again;
there is no automatic retry against a newer hash. Duplicate clicks do not cancel
or repeat an in-flight decision.

Command equivalents:

```text
/sleep examples pending
/sleep examples verified
/sleep examples all
/sleep approve EXAMPLE_ID
/sleep approve EXAMPLE_ID THE COMPLETE CORRECTED ANSWER
/sleep reject EXAMPLE_ID
```

An ID must first have its question, answer and hash displayed in this TUI session.
Fetching a list does not authorize approvals of unseen entries. For multiline
corrections, the review editor preserves the complete text. A review never starts
training or promotes a model by itself.

## Outcomes and memory

`needs_review` and `insufficient_verified_data` are normal stopped outcomes when
the job cannot establish enough eligible verified material. The UI makes the
review requirement visible; 100% means that the current pipeline and restoration
ended, not that an expert was trained. Review pending examples or add suitable
consented context, then start sleep again.

`completed` reports a completed job, while `no_new_information` reports a no-op.
Training, evaluation and artifact fields still determine what actually happened.
An expert can be genuinely trained yet fail the promotion gate or retain unknown
quality. The registry records those facts separately. **REVIEW CANDIDATE** reveals
an actual adapter in Cortex without activating it.

Canonical memory retains verified evidence with scope/task/source bindings.
Explicit supersession can retire an older scoped record; unresolved conflicts
remain conflicted and are withheld from training/retrieval as active knowledge.
Revoking source consent or changing/deleting bound content invalidates derived
material and affected promotion eligibility. Already-trained weights are separate
artifacts; changing consent does not erase prior learning.

## API controls

`POST /sleep/start` defaults to:

```json
{
  "context_examples": true,
  "dynamic": true,
  "min_verified_examples": 3,
  "max_generated_examples": 32,
  "generation_model": null,
  "scope": "all"
}
```

`min_verified_examples` accepts 1–5,000. `max_generated_examples` accepts 0–256;
zero disables model-generated drafts while allowing other eligible sources.
`generation_model` is an optional identifier for the currently loaded model.
Load another teacher before starting Sleep if you want to select it. The backend
does not switch or rediscover interfaces for every draft. Context and output allowances
come from that model's initialized snapshot.

`dry_run: true` previews preparation without training. `context_examples: false`
retains the 0.3 dataset path; `/sleep start --legacy` is its TUI equivalent.
`dynamic: false` preserves explicit expert/base selection. Existing source,
episode, resource and training bounds still apply. Expert build requests inherit
these fields.

| Route | Purpose |
| --- | --- |
| `GET /sleep/examples?status=pending` | Fetch pending canonical candidates on demand |
| `GET /sleep/examples` | Fetch candidates without a status filter |
| `POST /sleep/examples/ID/review` | Submit `decision`, the displayed `content_hash`, optional complete `correction`, and optional `reason` |

The review decision is `approve` or `reject`. A content hash must be 64 hexadecimal
characters. Corrections allow up to 100,000 characters and reasons up to 2,000.
Always use the current hash returned with the candidate; copying a stale hash
cannot approve a newer version. Source authorization and binding checks remain
authoritative even when the text hash matches.
