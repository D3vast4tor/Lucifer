# Terminal art direction — Cyberpunk operations

The current 0.4 interface extends the 0.3 sleep controls and 0.2 layout that superseded the earlier
DedSec theme. Its reference is
[Anna Kuźmińska's Cyberpunk 2077 UI art showcase](https://www.behance.net/gallery/127222375/Cyberpunk-2077-UI-Art-and-UI-Art-Concepts),
including the in-world red/cyan neural-analysis panels. Lucifer uses an original
dark navy, signal yellow, warning red and cyan system with angular rails, dense
telemetry, mission controls and a clean wing/halo monogram. Game images are
research references and are not application assets.

The adapter cortex translates Colibri's projected, inspectable brain navigation
into a terminal Braille surface. Real registry nodes and lineage are drawn over
that illustrative shape. Readable response text, Markdown and visible access
controls take priority over decorative animation.

Agent is the single conversation view; the separate Chat navigation is removed.
History / F5 opens a cyan-framed list and public transcript preview on demand.
Availability labels distinguish RAM, checkpoint, transcript and unavailable
private content. A yellow OPEN SESSION action reads the selection; resuming work
stays explicit. New conversations show saving ON and training OFF, with an
explicit privacy command to disable logging. Logging and training states remain visible. Sidebar and compact
session telemetry use current GPU/VRAM percentage bars, without trend animation.

The mission view keeps the stop policy and output continuation as separate
controls. While a segment runs, a CORRECT input takes the normal composer space
so steering remains available in compact terminals. STOP/Esc pauses owned work;
RESUME and normal follow-ups retain the existing turn panels. Emitted thinking
can be expanded or collapsed per segment and remains in session memory only.
The generation row displays the initialized output allowance and active context;
mission status identifies the absence of a default deadline.

F4 opens the sleep pipeline with one yellow SLEEP / CONSOLIDATE action, cyan
stage progress and a pink cancellation control. Overall progress remains
indeterminate until the pipeline can establish completion; a stage percentage
never stands in for a whole-job estimate. The inference restoration phase stays
visible and controls remain locked until cleanup finishes. Candidate review
reveals its recorded expert node in Cortex; promotion stays separate.
Indeterminate bars stop their timers while idle or outside the Sleep view, so
Cortex interaction has no hidden progress animation work. Returning to Sleep
resumes animation from the cached job state.

The example review dialog uses the same yellow approval, cyan evidence and pink
rejection accents. Canonical questions and answers remain literal text; their
full content hash is shown directly. Source/validation details expand separately.
The complete correction editor supports multiline text. Queue fetches happen only
on opening, filtering or explicit reload; stale conflicts keep the draft intact.

The previous reference research is retained below as design history.

The requested search phrase was **dedsec art style**. Image results and
[Lieke Schnater's *my DED_SEC*](https://www.artstation.com/artwork/n16Y4)
were visually inspected as references. The artist describes the work as inspired
by Watch Dogs. The reference image uses black and white cutouts, magenta fields,
acid lime accents, halftone texture, diagonal slashes and hand-drawn marks.

Lucifer translates those motifs into original terminal graphics: vivid panel
accents, stencil-like labels, slash marks, pixel borders and high contrast text.
Conversation backgrounds stay dark to support sustained reading. The upper-left
portrait is an original terminal illustration of Lucifer before the fall:
halo, wings, a composed face and gold light. It is separate from the skull motif
in many DedSec images.

Reference images inform the design and are not packaged as application assets.
The interface uses Unicode/Rich graphics and supports ordinary terminals without
an image protocol. The saved previews under `docs/assets` show the implemented
layout rather than a design mockup.
