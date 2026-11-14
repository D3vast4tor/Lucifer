# Problem/solution documents for Meditation

To supply actual question/answer targets in a readable document, attach a UTF-8
`.jsonl` file with one `lucifer.problem-solution.v1` record per line. Import it
through the existing knowledge upload/path interface or `import_document`, then
read/search its desired chunks in the training chat. Consent and interaction
requirements are identical to other documents. Plain JSONL remains literal text.

Each record has exactly these fields:

```json
{"format":"lucifer.problem-solution.v1","id":"example-1","question":"A completed operation receipt exists after reconnecting. What should the agent do?","answer":"Reuse the completed result and continue with the next unfinished step.","sources":[{"url":"https://example.org/document","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"verification":{"method":"provided_answer","receipt_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}
```

The example digests above are placeholders; a material builder must use actual
source and receipt hashes. Each record is at most 2,400 characters including JSON
syntax, its ID at most 160 characters, and it has one to eight public source
references. IDs are unique within a file. Split longer examples into independent
complete problems rather than truncating answers. Ordinary larger documents
remain supported through their existing extraction/windowing paths.

The optional `lineage` field is a list of one to eight dataset SHA-256
identifiers. All fragments, variants and derivative workbooks of one generated
dataset must carry the same identifier. This merges their dependency components
even when their citations name different repositories. Citation variety does not
make one authored dataset into independent training and test datasets.

In Supervised mode the question is context and the answer is the assistant
target. Only interacted records are included. The target authority is
`document_provided_solution`; a declared verification method or receipt digest
does **not** make its answer true or confer permissions. The importer never
executes checkers, commands, source instructions or URLs from these records.
Build and run trusted verifiers separately, recording their results and source
identities. Provider reasoning and raw runtime traces must not be placed in the
public question or answer. Undocumented fields such as `reasoning_content` and
executable verifier bodies are rejected.

In Unsupervised mode each record remains literal source text. The format does
not change the user's mode selection or start training. Deleting or altering a
source, withdrawing consent, or changing its lineage invalidates the original
source binding. No manual example review is required.

## Precommitting source roles

The optional `source_roles` map on `POST /meditation/start` can reserve eligible,
consumed conversations as `train`, `development` or `heldout` before optimization:

```json
{"mode":"supervised","conversation_ids":["TRAIN_CHAT","DEV_CHAT","TEST_CHAT"],"source_roles":{"TRAIN_CHAT":"train","DEV_CHAT":"development","TEST_CHAT":"heldout"},"dry_run":true,"idempotency_key":"my-source-precommit-v1"}
```

This three-chat illustration does not satisfy the eight-group admission minimum.
Roles apply to whole connected source components, including exact document
copies, supplied source hashes and conservatively canonicalized source URLs.
Related pages from one repository, declared dataset lineages and recognized raw/hosted mirrors merge.
Unknown sites conservatively merge by origin. Shared references can merge more
than one chat; separate IDs or documents alone do not establish independence.

Existing reservations cannot be relabeled. Conflicting roles for connected
sources are rejected before reservations are written. Unspecified fresh groups
retain the normal score-independent assignment. A dry run reserves roles and
collects source evidence without optimizer updates or model loading. It cannot
reveal heldout scores. This API is an advanced preparation control; ordinary UI
users can continue choosing a mode and source scope normally.
