# Attached knowledge and public web research

Lucifer chooses `web_search` and `read_web_page` when the task benefits from
current facts or documentation. No API key or paid provider is required. Search
uses DuckDuckGo and Bing with relevance checks, and can use a public SearXNG instance via
`LUCIFER_SEARXNG_URL` with JSON output enabled. External engines can rate-limit or
be unavailable; the tool reports this explicitly. Search queries are sent to the
chosen engine. Page reads are unauthenticated, bounded, and restricted to public
HTTP(S) destinations. Source URLs accompany results.
When general engines are unavailable or unrelated, a keyless GitHub fallback
can discover public software repositories and their published documentation
homepages. Its results explicitly identify this narrower coverage. Queries with
no relevant source return an unavailable result rather than unrelated pages.

Completed command results are reused for the same tracked file revision.
An explicit successful file edit creates a new command-input revision, allowing
the same test command to run against the changed source with its normal approval
policy. Reconnects and repeated proposals at the same revision reuse the result.
Arbitrary external edits and opaque shell effects are outside this revision
contract; inspect those sources explicitly before relying on an earlier check.

## Terminal commands

```text
/attach "/path/to/document.pdf"
/attach "/path/to/video.mp4" --frames 12
/link /absolute/backend/path/workbook.xlsx
/link https://example.org/document.pdf
/knowledge list
/knowledge search precise topic or phrase
/knowledge read knowledge_ID 0
/knowledge remove knowledge_ID
```

`/attach` sends the selected local file's bytes from the terminal client.
`/link` explicitly imports a backend-local path or a public URL. Import takes an
immutable copy: edits to the original need a new import. Identical bytes already
attached to this chat are reused. An attachment belongs to the selected chat;
opening another chat does not expose it.

Lucifer gets a document catalogue and bounded relevant passages automatically.
`search_knowledge` and `read_knowledge` expose the remaining chunks with document,
page, sheet, block or frame references. Document text and visible instructions
are untrusted source data; attaching a document does not tell Lucifer to execute
its instructions. The plain chat endpoint receives bounded retrieved excerpts;
Agent mode can inspect additional chunks with tools.

## Formats and coverage

Readers support UTF-8/Unicode text and source files, CSV/TSV, PDF, DOCX, XLS/XLSX,
ODT/ODS/ODP, PPTX, EPUB, RTF and HTML. Legacy DOC/PPT use the installed LibreOffice
converter in a separate temporary profile with macros disabled. Office macros and
spreadsheet formulas are never executed. Encrypted, corrupt and unsupported
documents produce an actionable import error.

Images use the loaded model's advertised vision interface. Image-only PDF pages
are rendered through `pdftoppm` and read through that same model. Videos are
sampled across their duration (eight frames by default, configurable from one to
24), with timestamps; supported embedded text subtitles are also extracted.
Video coverage is sampled, and audio is not speech-transcribed. Attach a
transcript or subtitles when spoken content matters. Visual text/descriptions
are model interpretations, not verified literal text.

Imports are bounded to 128 MiB locally, 64 MiB for public links, eight MiB of
extracted text, 1,000 PDF pages and 24 visual inputs. Oversized sources need
splitting. All retained text is chunked without silently dropping its middle or
tail. Retrieval and source reads are paged instead of filling the model context
with an entire document.

## Saving, training and deletion

With logging on, the immutable source and extracted chunks are saved with the
chat. With logging off, attachments remain in backend session memory. Turning
logging off removes saved source/chunk content and derived training evidence;
turning saving back on explicitly persists the currently retained sources.
Deleting a chat removes its attached knowledge too.

When that chat's training setting is on, sleep curriculum preparation includes
source-qualified literal excerpts. Current source hashes and consent bind the
examples, and the entire document shares one training/holdout group. Later sleep
cycles skip covered chunks and can advance to new ones. Visual descriptions and
unverified model drafts need explicit example review before becoming targets.
Training consent makes sources eligible; normal minimum-data, verification,
evaluation and promotion gates still apply. Import does not start a sleep job.

Removing an attachment invalidates its derived training evidence and datasets;
turning training off excludes it from later sleep cycles. Raw provider reasoning
and image transport bytes do not become supervised training targets.
