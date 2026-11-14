# Lucifer reliability, tools and knowledge plan

Requested 13 November 2026. Canonical installation root: `/home/Lucifer`.

1. **Audit and migrate installation paths.** Inspect source, model profiles,
   launchers, virtual-environment entry points, runtime model registrations and
   training/restore configuration for references to the former installation.
   Point operational paths at `/home/Lucifer/Soul` and `/home/Lucifer/vLLM`;
   consolidate owned model and application data under `/home/Lucifer`. Preserve
   immutable historical evidence instead of changing the meaning of old runs.
   Validate training evaluation/restore entry points as well as normal startup.
2. **Repair automatic context compaction.** Use the currently loaded, pinned
   model/profile/adapter to summarize older completed exchanges, including tool
   history within a single mission. Reserve generation headroom, summarize all
   removed source text in bounded chunks, preserve goals/corrections and recent
   complete tool exchanges, and retain durable working summaries without raw
   provider reasoning. Activate the updated implementation after validation.
3. **Detect stalled progress.** Reuse completed operations; detect repeated
   unchanged results/errors, steer once, then pause recoverably when progress
   does not resume. Keep normal long missions free of arbitrary overall caps.
4. **Improve tools and autonomous web research.** Correct advertised versus
   actual command timeouts, preserve structured metadata when output is bounded,
   improve search filtering/paging, and supply no-key web search plus public-page
   reading. Teach Lucifer when to select these tools without per-turn reminders;
   return source URLs and explicit service failures.
5. **Add terminal knowledge attachments.** Support upload and local/public-URL
   links, searchable/readable document chunks with source/page/sheet/frame
   references, common text/PDF/Office/spreadsheet formats, images and sampled
   video frames through the loaded model's vision capability. Distinguish literal
   document text from model-derived visual descriptions and show coverage and
   extraction limitations. Documents are source data, never user instructions.
6. **Connect knowledge to sleep consent.** Include attached sources from chats
   with training enabled in curriculum preparation. Bind examples to source
   hashes and current consent, retain human review for model-derived targets,
   and invalidate derived datasets when attachments or consent are removed.
   Logging-off sessions keep their attachments in memory only.
7. **Validate and activate.** Exercise a long single-goal mission, repeated-work
   recovery, tools, real public search, representative document/media imports,
   terminal/API flows, sleep source eligibility and privacy deletion. Restart
   only the owned idle backend after preserving checkpoints; report actual
   checks, remaining format limitations and any restart impact.

## Implementation and validation

Phases 1–6 are implemented. The backend was restarted with the new code and
canonical paths. The offline migration moved application data and the Qwen model,
repaired 59 launcher/configuration files and 31 operational database records,
and removed four dangling service links. A consistent database and changed-file
backup is at `var/state/migration-backups/20261113T172712.871755Z`. A repeat
migration report contains zero remaining moves, file repairs or record changes.
The auxiliary Strata settings and legacy runtime launcher also use their current
installation root. Immutable historical receipts and transcripts stay unchanged.

Compaction uses the pinned loaded model and preserves complete recent tool
exchanges, literal goals/corrections and the operation ledger. Source text is
visited in finite bounded chunks, with generation headroom reserved before the
context is exhausted. A summary failure leaves the original context available
and pauses recoverably. Repeated unchanged evidence is steered, then paused if
the same work continues. Command receipts are versioned across explicit source
edits, avoiding reuse of an earlier test result after a code change.

The full suite passed with **962 tests, one skipped**; final focused checks cover
subsequent web-link navigation and attachment/sleep integration. Ruff passes.
Headless terminal tests cover upload/link/retrieval/removal, consent messaging,
argument errors and sleep readiness. Representative format fixtures include
real PDF rendering, Word/Excel/ODF/EPUB/presentation content, images, video frames
and embedded subtitles. An isolated complete sleep dry run consumes attached
sources under training consent; no real training or promotion was triggered.

The live loaded model read an image's red square and “42”, and answered a CSV
question correctly with the attachment cited. A real model-generated summary
reduced a synthetic 12K-limit context from 12,418 to 7,193 estimated tokens while
preserving the goal, key file fact, pending input, recent tool pair and saved
summary. This was a simulated threshold; the actual 1,048,576-token allocation
was unchanged. Receipts are in `var/validation/reliability-knowledge-20261113`.

The final autonomous web check **completed** in six model rounds and five tool
calls. The user prompt did not name any tools. Lucifer selected search itself,
followed actual documentation links, read the official Search API and settings
pages, and returned the correct formats and JSON-enablement instructions with
official citations. General search was unavailable for that niche query, so the
tool explicitly disclosed its repository/documentation discovery fallback.
Phase 7 is complete; the canonical backend is active and the temporary live
acceptance conversations were removed. Reopen the terminal frontend with
`/home/Lucifer/lucifer tui` to load its new attachment commands.

Public engines can challenge requests or return irrelevant results. Search
rejects unrelated responses and discloses repository-only fallback coverage.
Page reads expose actual links with filtering/pagination. Video vision is
sampled; audio without subtitles needs a transcript. Model-derived visual
descriptions remain reviewable candidates rather than automatic truth labels.
