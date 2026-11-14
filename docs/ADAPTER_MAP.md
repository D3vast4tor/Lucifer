# Adapter cortex implementation

The reference is the user's local Colibri checkout at
`/home/d3vast4tor/colibri`, specifically `web/src/BrainWorkspace.tsx` and
`web/src/lib/cortex.ts`. Its projected cortical surface, clustered regions,
rotation/zoom and selection inspector informed an original terminal design.
No Colibri source code or game assets are copied into Lucifer.

`lucifer/adapter_map.py` turns model, adapter and learning-run records into stable
nodes and lineage edges. A base model is a core; an adapter links to its recorded
parent adapter or its base. A consolidated expert links to every available
`source_expert_ids` parent as well as any recorded adapter parent, with duplicate
edges removed. Missing references stay in metadata; no missing trained node is
invented, and the base edge is retained when no parent node is available. Status
colors and evaluation metadata come from the registry. Coordinates are
deterministic and the folded two-lobe surface
uses Unicode Braille with Rich colors.

Experts use `E##` labels; ordinary adapters use `A##`. Sleep's REVIEW CANDIDATE
action selects and reveals the real registered node while preserving the current
rotation and zoom. Adapter registration changes the structural registry revision;
the next heartbeat triggers one bootstrap. Optimizer, evaluation and log updates
alone do not rebuild the cortex graph.

The view refreshes after a structural registry change, not for every GPU sample or
model token. Rendering is driven by resize, selection and camera interaction. It
does not launch inference, scan model weights or infer hidden activations. A node
selection inspects metadata; activation requires the explicit existing adapter
selection API and its evaluation gates.

Expert inspectors show the expert name/topics, training state, separate quality
state, source lineage, optimizer steps and verified tensor count when recorded.
`trained` means the worker produced a verified checkpoint; it does not imply an
evaluation improvement or promotion. Expert creation and consolidation use the
normal sleep job flow. Consolidation replays currently consented reviewed source
examples into a fresh LoRA on the same compatible base. The graph records data
lineage, not an adapter tensor merge or learned neural routing. See
[the session and expert workflow](SESSIONS_AND_EXPERTS.md).

Controls: F3 or `/brain [ADAPTER_ID]`; arrows or mouse drag rotate/tilt; Q/E roll;
Shift+arrows pan; plus/minus or mouse wheel zoom; N/P cycle nodes; Enter inspects;
R resets. Yaw, pitch and roll support full rotations. Mouse selection is
supported. Compact terminals put the inspector below the surface when requested.
Use a terminal font with Braille fallback; exported SVG previews include fallback
font metadata.

Acceptance covers empty/single/dense registries, deterministic coordinates,
recorded parent/base and multiple expert-parent lineage, training/quality details,
active/rejected/candidate styling, selection,
rotation/zoom/pan, compact inspection and a separate activation action. Tests are
in `tests/test_adapter_map.py` and `tests/test_tui.py`.
