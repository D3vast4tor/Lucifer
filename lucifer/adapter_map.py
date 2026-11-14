"""An adapter atlas projected into terminal Braille cells.

The folded cortex is an illustrative navigation surface. Nodes and lineage edges
come from Lucifer's registry; it does not invent model activations or routing.
Inspired by Colibri's separate exploratory cortex, implemented independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from typing import Any

from rich.text import Text
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

YELLOW = "#fcee0a"
CYAN = "#54d9e8"
RED = "#ff515b"
MUTED = "#596a7b"
STATUS_COLORS = {"active": YELLOW, "loaded": YELLOW, "candidate": CYAN, "admitted": "#9de39b",
                 "evaluated": CYAN, "archived": "#879bc0", "rejected": RED,
                 "failed": RED, "unavailable": RED}


@dataclass(frozen=True)
class AtlasNode:
    id: str
    label: str
    kind: str
    status: str
    position: tuple[float, float, float]
    data: dict[str, Any] = field(compare=True, repr=False)


@dataclass(frozen=True)
class AtlasEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class Atlas:
    nodes: tuple[AtlasNode, ...]
    edges: tuple[AtlasEdge, ...]
    active_adapter: str | None

    def node(self, ident: str | None) -> AtlasNode | None:
        return next((node for node in self.nodes if node.id == ident), None)


def build_atlas(models: list[dict], adapters: list[dict], *, current: dict | None = None,
                active_adapter: str | None = None, jobs: dict | None = None) -> Atlas:
    """Build registry topology. An absent parent is never fabricated as an edge."""
    # Compatibility is computed by the backend at structural revisions. Retain
    # legacy/unknown records until that metadata exists; never delete artifacts.
    adapters = [adapter for adapter in adapters if adapter.get("compatible") is not False]
    current = current or next((m for m in models if m.get("status") in {"loaded", "serving"}), None)
    known = {str(model["id"]): model for model in models if model.get("id")}
    base_ids = {str(adapter.get("base_model_id")) for adapter in adapters if adapter.get("base_model_id")}
    if current and current.get("id"):
        base_ids.add(str(current["id"]))
    if not base_ids and known:
        base_ids.add(next(iter(known)))
    nodes = []
    for index, ident in enumerate(sorted(base_ids)):
        model = known.get(ident)
        loaded = bool(current and current.get("id") == ident)
        data = {key: (model or {}).get(key) for key in
                ("native_context_length", "effective_context_length", "default_thinking_mode", "status", "backend")}
        data["profile_id"] = ident
        data["available"] = model is not None
        position = ((index - (len(base_ids) - 1) / 2) * .38, .12, 0.0)
        nodes.append(AtlasNode("model:" + ident, "CORE" if loaded else f"BASE {index + 1}", "model",
                               "loaded" if loaded else "ready" if model else "unavailable", position, data))
    adapter_ids = {str(adapter["id"]) for adapter in adapters if adapter.get("id")}
    edges = []
    centers = {"active": (-.55, -.10, .40), "admitted": (-.35, -.12, .35), "candidate": (.60, -.12, .45),
               "evaluated": (.65, -.12, .40), "archived": (-.45, .25, -.45),
               "failed": (.5, .28, -.5), "rejected": (.5, .28, -.5)}
    runs = [run for entries in (jobs or {}).values() if isinstance(entries, list) for run in entries if isinstance(run, dict)]
    grouped: dict[str, int] = {}
    for index, adapter in enumerate(sorted(adapters, key=lambda item: str(item.get("id", "")))):
        if not adapter.get("id"):
            continue
        ident = str(adapter["id"])
        state = "active" if active_adapter == ident else str(adapter.get("status", "candidate"))
        ordinal = grouped.get(state, 0)
        grouped[state] = ordinal + 1
        center = centers.get(state, (.60, -.12, .45))
        seed = int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)
        angle = ordinal * 2.399963 + (seed % 1000) / 1000
        radius = min(.35, .13 + .035 * math.sqrt(ordinal))
        position = (center[0] + radius * math.cos(angle), center[1] + .6 * radius * math.sin(angle),
                    center[2] + radius * math.sin(angle))
        data = {key: adapter.get(key) for key in ("id", "status", "base_model_id", "parent_adapter_id", "base_adapter_id",
                "rank", "metrics", "source_revoked", "created_at", "training_run_id", "sleep_run_id", "training_metrics",
                "expert_name", "expert_description", "expert_topics", "source_expert_ids", "is_expert",
                "expert_training_status", "expert_quality_status", "artifact_verification", "source_invalidated",
                "admission_report", "admitted_at", "compatible", "compatibility_reason", "base_revision")}
        source = next((run for run in runs if ident in {run.get("candidate_id"), run.get("candidate_adapter_id")}), None)
        if source:
            data["source_job"] = {key: source.get(key) for key in ("id", "status", "total_examples", "replay_examples")}
        prefix = "E" if adapter.get("is_expert") else "A"
        nodes.append(AtlasNode("adapter:" + ident, f"{prefix}{index + 1:02}", "adapter", state, position, data))
        parents = list(adapter.get("source_expert_ids") or [])
        parent = adapter.get("parent_adapter_id") or adapter.get("base_adapter_id")
        if parent:
            parents.append(parent)
        recorded = [str(parent) for parent in dict.fromkeys(parents) if str(parent) in adapter_ids and str(parent) != ident]
        for parent in recorded:
            edges.append(AtlasEdge("adapter:" + parent, "adapter:" + ident, "lineage"))
        if not recorded and adapter.get("base_model_id"):
            edges.append(AtlasEdge("model:" + str(adapter["base_model_id"]), "adapter:" + ident, "base"))
    return Atlas(tuple(nodes), tuple(edges), active_adapter)


def admission_gate_lines(report: dict | None) -> list[str]:
    """Render only recorded admission results; missing gates stay unknown."""
    if not isinstance(report, dict) or not isinstance(report.get("gates"), dict):
        return []
    lines = []
    reasons = report.get("reasons") or {}
    for name, gate in report["gates"].items():
        passed = gate.get("passed") if isinstance(gate, dict) else gate
        state = "PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN"
        reason = gate.get("reason") if isinstance(gate, dict) else reasons.get(name) if isinstance(reasons, dict) else None
        lines.append(f"{str(name).replace('_', ' ')}: {state}" + (f" — {reason}" if reason else ""))
    return lines


def node_details(node: AtlasNode | None, atlas: Atlas) -> str:
    if node is None:
        return "## No registry nodes\n\nLoad a model or register an adapter to populate the atlas."
    data = node.data
    if node.kind == "model":
        rows = ["## Lucifer / base core", f"**State:** {node.status.upper()}"]
        for key, label in (("effective_context_length", "Active context"), ("native_context_length", "Native context")):
            if isinstance(data.get(key), int):
                rows.append(f"**{label}:** {data[key]:,} tokens")
        descendants = sum(edge.source == node.id for edge in atlas.edges)
        rows.extend([f"**Connected adapters:** {descendants}",
                     "The core represents a registered base profile. The cortex outline is an illustrative navigation surface."])
        if not data.get("available"):
            rows.append("This adapter's base profile is absent from the current registry.")
        return "\n\n".join(rows)
    rows = [f"## {node.label} / {'expert' if data.get('is_expert') else 'adapter'}", f"**ID:** `{data.get('id', '')}`", f"**State:** {node.status.upper()}"]
    if data.get("expert_name"):
        rows.append(f"**Expert:** {data['expert_name']}")
    if data.get("expert_topics"):
        rows.append("**Topics:** " + ", ".join(str(topic) for topic in data["expert_topics"]))
    if data.get("expert_training_status"):
        rows.append(f"**Expert training:** {data['expert_training_status']}")
    if data.get("expert_quality_status"):
        rows.append(f"**Quality gate:** {data['expert_quality_status']}")
    proof = data.get("artifact_verification")
    if isinstance(proof, dict):
        if proof.get("optimizer_steps") is not None:
            rows.append(f"**Optimizer steps:** {proof['optimizer_steps']}")
        if proof.get("tensor_count") is not None:
            rows.append(f"**Verified adapter tensors:** {proof['tensor_count']}")
    if data.get("source_revoked"):
        rows.append("**Source consent revoked.** This artifact is not eligible for reuse.")
    elif data.get("source_invalidated"):
        rows.append("**Source evidence invalidated.** This artifact is not eligible for reuse.")
    if data.get("rank") is not None:
        rows.append(f"**LoRA rank:** {data['rank']}")
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key, label in (("passed", "Evaluation passed"), ("task_success", "Task success"),
                           ("improvement", "Improvement"), ("regression_rate", "Regression rate")):
            if key in metrics:
                rows.append(f"**{label}:** {metrics[key]}")
        evaluation = metrics.get("evaluation") or {}
        for key, label in (("baseline_score", "Baseline task success"), ("candidate_score", "Candidate task success"),
                           ("task_success_delta", "Task success change"), ("dynamic_task_success_delta", "Domain success change"),
                           ("regression_rate", "Regression rate"), ("latency_delta", "Latency change")):
            value = evaluation.get(key) if isinstance(evaluation, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                rows.append(f"**{label}:** {value:+.2%}" if key.endswith("delta") else f"**{label}:** {value:.2%}")
        if isinstance(evaluation, dict) and evaluation.get("unscored_pairs") is not None:
            rows.append(f"**Incomplete evaluation pairs:** {evaluation['unscored_pairs']}")
    gates = admission_gate_lines(data.get("admission_report"))
    if gates:
        rows.append("**Admission gates:**\n\n" + "\n".join(f"- {line}" for line in gates))
    if node.status == "admitted":
        rows.append("**Admitted:** available for explicit selection on its compatible Lucifer profile; not automatically active.")
    parents = [atlas.node(edge.source) for edge in atlas.edges if edge.target == node.id]
    rows.append(f"**Lineage:** {', '.join(parent.label for parent in parents if parent) or 'not recorded'} → {node.label}")
    job = data.get("source_job")
    if isinstance(job, dict):
        rows.append(f"**Source run:** `{job.get('id')}` / {job.get('status', 'unknown')}")
        if job.get("total_examples") is not None:
            rows.append(f"**Training examples:** {job['total_examples']}")
    rows.append("Selecting a node only inspects it. Activate an evaluated adapter with the explicit button below.")
    return "\n\n".join(rows)


class BrailleCanvas:
    """Two by four subpixels per character, with an independent text overlay."""
    MASKS = ((1, 8), (2, 16), (4, 32), (64, 128))

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.cells: dict[tuple[int, int], tuple[int, str]] = {}
        self.labels: dict[tuple[int, int], tuple[str, str]] = {}

    def dot(self, x: float, y: float, color: str):
        px, py = round(x), round(y)
        if 0 <= px < self.width * 2 and 0 <= py < self.height * 4:
            key = px // 2, py // 4
            mask, _ = self.cells.get(key, (0, color))
            self.cells[key] = mask | self.MASKS[py % 4][px % 2], color

    def line(self, a: tuple[float, float], b: tuple[float, float], color: str):
        x0, y0, x1, y1 = round(a[0]), round(a[1]), round(b[0]), round(b[1])
        xmax, ymax = self.width * 2 - 1, self.height * 4 - 1
        if max(x0, x1) < 0 or min(x0, x1) > xmax or max(y0, y1) < 0 or min(y0, y1) > ymax:
            return
        if not (0 <= x0 <= xmax and 0 <= x1 <= xmax and 0 <= y0 <= ymax and 0 <= y1 <= ymax):
            # Clip before rasterization: zooming/panning must not spend time
            # sampling thousands of invisible pixels or skip a visible crossing.
            dx, dy = x1 - x0, y1 - y0
            start, end = 0.0, 1.0
            for p, q in ((-dx, x0), (dx, xmax - x0), (-dy, y0), (dy, ymax - y0)):
                if p == 0:
                    if q < 0:
                        return
                elif p < 0:
                    start = max(start, q / p)
                else:
                    end = min(end, q / p)
                if start > end:
                    return
            x0, y0, x1, y1 = round(x0 + dx * start), round(y0 + dy * start), round(x0 + dx * end), round(y0 + dy * end)
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx + dy
        cells, masks = self.cells, self.MASKS
        while True:
            key = x0 // 2, y0 // 4
            bits, _ = cells.get(key, (0, color))
            cells[key] = bits | masks[y0 % 4][x0 % 2], color
            if x0 == x1 and y0 == y1:
                break
            twice = error * 2
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def label(self, x: int, y: int, text: str, color: str):
        for offset, character in enumerate(text):
            if 0 <= x + offset < self.width and 0 <= y < self.height:
                self.labels[x + offset, y] = character, color

    def render(self) -> Text:
        text = Text(no_wrap=True, overflow="crop")
        for y in range(self.height):
            run, previous_style = [], None
            for x in range(self.width):
                key = x, y
                label = self.labels.get(key)
                if label is not None:
                    character, style = label
                else:
                    bits, style = self.cells.get(key, (0, ""))
                    character = chr(0x2800 + bits) if bits else " "
                if style != previous_style and run:
                    text.append("".join(run), style=_style(previous_style) if previous_style else None)
                    run.clear()
                previous_style = style
                run.append(character)
            if run:
                text.append("".join(run), style=_style(previous_style) if previous_style else None)
            if y + 1 < self.height:
                text.append("\n")
        return text


@lru_cache(maxsize=64)
def _style(value: str) -> Style:
    return Style.parse(value)


def _projector(width, height, yaw, pitch, roll, zoom, pan_x, pan_y):
    """Build one camera matrix per frame, rather than trigonometry per point."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    m00, m01, m02 = cr * cy - sr * sy * sp, -sr * cp, cr * sy + sr * cy * sp
    m10, m11, m12 = sr * cy + cr * sy * sp, cr * cp, sr * sy - cr * cy * sp
    m20, m21, m22 = -sy * cp, sp, cy * cp
    scale = min(width * .71, height * 1.65) * zoom
    ox, oy = width + pan_x * width * 2, height * 1.9 + pan_y * height * 4
    def project(point):
        x, y, z = point
        rx, ry = m00 * x + m01 * y + m02 * z, m10 * x + m11 * y + m12 * z
        depth = m20 * x + m21 * y + m22 * z
        factor = scale * 3 / max(.8, 3 - depth)
        return ox + rx * factor, oy + ry * factor, depth
    return project


def cortex_lines() -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """An original folded two-lobe shell; geometry is a visual metaphor."""
    lines = []
    for side in (-1, 1):
        for latitude in range(1, 15):
            a = math.pi * latitude / 15
            points = []
            for longitude in range(41):
                b = -math.pi / 2 + math.pi * longitude / 40
                fold = .038 * math.sin(a * 9 + math.sin(b * 4))
                points.append((side * (.07 + (.92 + fold) * math.sin(a) * math.cos(b)),
                               (.64 + fold) * math.sin(a) * math.sin(b), 1.03 * math.cos(a)))
            lines.append(tuple(points))
        for longitude in range(1, 10):
            b = -math.pi / 2 + math.pi * longitude / 10
            points = []
            for latitude in range(31):
                a = .03 + (math.pi - .06) * latitude / 30
                fold = .038 * math.sin(a * 9 + math.sin(b * 4))
                points.append((side * (.07 + (.92 + fold) * math.sin(a) * math.cos(b)),
                               (.64 + fold) * math.sin(a) * math.sin(b), 1.03 * math.cos(a)))
            lines.append(tuple(points))
    return tuple(lines)


CORTEX_LINES = cortex_lines()


class AdapterBrain(Widget, can_focus=True):
    """Interactive, event-driven atlas. No timers, network calls or fake pulses."""
    BINDINGS = [
        Binding("left", "rotate(-1,0)", "Rotate", show=False),
        Binding("right", "rotate(1,0)", "Rotate", show=False),
        Binding("up", "rotate(0,-1)", "Tilt", show=False),
        Binding("down", "rotate(0,1)", "Tilt", show=False),
        Binding("q", "roll(-1)", "Roll", show=False),
        Binding("e", "roll(1)", "Roll", show=False),
        Binding("shift+left", "pan(-1,0)", "Pan", show=False),
        Binding("shift+right", "pan(1,0)", "Pan", show=False),
        Binding("shift+up", "pan(0,-1)", "Pan", show=False),
        Binding("shift+down", "pan(0,1)", "Pan", show=False),
        Binding("plus,equals_sign", "zoom(1)", "Zoom", show=False),
        Binding("minus", "zoom(-1)", "Zoom", show=False),
        Binding("n", "next_node(1)", "Next node", show=False),
        Binding("p", "next_node(-1)", "Previous node", show=False),
        Binding("enter", "inspect", "Inspect", show=False),
        Binding("r", "reset", "Reset", show=False),
    ]

    class Selected(Message):
        def __init__(self, node: AtlasNode | None):
            self.node = node
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atlas = Atlas((), (), None)
        self.selected_id: str | None = None
        self.yaw, self.pitch, self.zoom_level = .1, .82, 1.0
        self.roll_angle = 0.0
        self.pan_x = self.pan_y = 0.0
        self._projected: dict[str, tuple[int, int]] = {}
        self._render_key = None
        self._rendered = Text("")
        self._shell_key = None
        self._shell_cells = {}
        self._drag_position = None
        self._dragged = self._suppress_click = False
        self.data_revision = 0

    def set_data(self, models: list[dict], adapters: list[dict], *, current=None, active_adapter=None, jobs=None):
        atlas = build_atlas(models, adapters, current=current, active_adapter=active_adapter, jobs=jobs)
        if atlas != self.atlas:
            self.atlas = atlas
            self.data_revision += 1
            if atlas.node(self.selected_id) is None:
                active = "adapter:" + str(active_adapter) if active_adapter else None
                self.selected_id = active if atlas.node(active) else atlas.nodes[0].id if atlas.nodes else None
            self._render_key = None
            self.post_message(self.Selected(atlas.node(self.selected_id)))
            self.refresh()

    def selected(self) -> AtlasNode | None:
        return self.atlas.node(self.selected_id)

    def action_rotate(self, x: int, y: int):
        self.yaw = math.remainder(self.yaw + x * .16, math.tau)
        self.pitch = math.remainder(self.pitch + y * .12, math.tau)
        self.refresh()

    def action_roll(self, direction: int):
        self.roll_angle = math.remainder(self.roll_angle + direction * .16, math.tau)
        self.refresh()

    def action_pan(self, x: int, y: int):
        self.pan_x += x * .08
        self.pan_y += y * .08
        self.refresh()

    def action_zoom(self, direction: int):
        self.zoom_level = max(.55, min(2.4, self.zoom_level * (1.13 if direction > 0 else 1 / 1.13)))
        self.refresh()

    def action_next_node(self, direction: int):
        if self.atlas.nodes:
            ids = [node.id for node in self.atlas.nodes]
            index = ids.index(self.selected_id) if self.selected_id in ids else 0
            self.selected_id = ids[(index + direction) % len(ids)]
            self.action_inspect()
            self.refresh()

    def action_inspect(self):
        self.post_message(self.Selected(self.selected()))

    def select_node(self, ident: str, *, reveal: bool = False) -> bool:
        node = self.atlas.node(ident)
        if node is None:
            return False
        self.selected_id = ident
        if reveal and self.content_size.width and self.content_size.height:
            width, height = self.content_size.width, self.content_size.height
            project = _projector(width, height, self.yaw, self.pitch, self.roll_angle,
                                 self.zoom_level, self.pan_x, self.pan_y)
            x, y, _ = project(node.position)
            self.pan_x += (width - x) / (width * 2)
            self.pan_y += (height * 2 - y) / (height * 4)
        self.action_inspect()
        self.refresh()
        return True

    def action_reset(self):
        self.yaw, self.pitch, self.zoom_level = .1, .82, 1.0
        self.roll_angle = 0.0
        self.pan_x = self.pan_y = 0.0
        self.refresh()

    def on_click(self, event: events.Click):
        self.focus()
        if self._suppress_click:
            self._suppress_click = False
            event.stop()
            return
        candidates = [(math.hypot(event.x - x, event.y - y), ident) for ident, (x, y) in self._projected.items()]
        if candidates:
            distance, ident = min(candidates)
            if distance <= 5:
                self.selected_id = ident
                self.action_inspect()
                self.refresh()

    def on_mouse_down(self, event: events.MouseDown):
        if event.button == 1:
            self.focus()
            self.capture_mouse()
            self._drag_position = event.screen_x, event.screen_y
            self._dragged = False
            event.stop()

    def on_mouse_move(self, event: events.MouseMove):
        if self._drag_position is not None:
            x, y = self._drag_position
            dx, dy = event.screen_x - x, event.screen_y - y
            self._drag_position = event.screen_x, event.screen_y
            if dx or dy:
                self._dragged = True
                self.yaw = math.remainder(self.yaw + dx * .055, math.tau)
                self.pitch = math.remainder(self.pitch + dy * .11, math.tau)
                self.refresh()
            event.stop()

    def on_mouse_up(self, event: events.MouseUp):
        if self._drag_position is not None and event.button == 1:
            self._drag_position = None
            self._suppress_click = self._dragged
            self.release_mouse()
            event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp):
        event.stop()
        self.action_zoom(1)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown):
        event.stop()
        self.action_zoom(-1)

    def render(self) -> Text:
        width, height = max(1, self.content_size.width), max(1, self.content_size.height)
        view = (width, height, self.yaw, self.pitch, self.roll_angle, self.zoom_level, self.pan_x, self.pan_y)
        key = (*view, self.selected_id, self.data_revision)
        if key == self._render_key:
            return self._rendered
        canvas = BrailleCanvas(width, height)
        project = _projector(*view)
        if view == self._shell_key:
            canvas.cells = self._shell_cells.copy()
        else:
            for line in CORTEX_LINES:
                previous = None
                for point in line:
                    x, y, depth = project(point)
                    color = "#344c5c" if depth < -.1 else "#5c4654" if depth < .3 else "#9d5964"
                    if previous:
                        canvas.line(previous, (x, y), color)
                    previous = (x, y)
            self._shell_key, self._shell_cells = view, canvas.cells.copy()
        projected = {node.id: project(node.position) for node in self.atlas.nodes}
        for edge in self.atlas.edges:
            if edge.source in projected and edge.target in projected:
                a, b = projected[edge.source], projected[edge.target]
                bright = self.selected_id in {edge.source, edge.target}
                canvas.line(a[:2], b[:2], CYAN if bright else "#416978")
        self._projected = {}
        for node in sorted(self.atlas.nodes, key=lambda item: projected[item.id][2]):
            x, y, _ = projected[node.id]
            cx, cy = round(x / 2), round(y / 4)
            if 0 <= cx < width and 0 <= cy < height:
                self._projected[node.id] = cx, cy
            color = STATUS_COLORS.get(node.status, CYAN)
            selected = node.id == self.selected_id
            canvas.label(cx - 1, cy, "[◆]" if selected else " ○ ", "bold " + color)
            canvas.label(cx - len(node.label) // 2, cy + 1, node.label, "bold " + color)
        if not any(node.kind == "adapter" for node in self.atlas.nodes):
            caption = "NO ADAPTERS / BASE CORE ONLY"
            canvas.label(max(0, (width - len(caption)) // 2), height - 2, caption, CYAN)
        elif any(node.data.get("is_expert") for node in self.atlas.nodes):
            canvas.label(max(0, width - 21), height - 1, "E EXPERT / A ADAPTER", MUTED)
        canvas.label(1, 0, "ADAPTER ATLAS / TOPOLOGY", "bold " + YELLOW)
        self._render_key, self._rendered = key, canvas.render()
        return self._rendered

    def snapshot_signature(self) -> str:
        return hashlib.sha256(json.dumps({"nodes": [(n.id, n.status, n.data) for n in self.atlas.nodes],
                                          "edges": [(e.source, e.target, e.kind) for e in self.atlas.edges]},
                                         sort_keys=True, default=str).encode()).hexdigest()
