"""Textual terminal workbench. No browser or cloud account is required."""
from __future__ import annotations

import asyncio
import json
import re
import shlex
import tempfile
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.renderables.bar import Bar as BarRenderable
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.worker import WorkerFailed
from textual.widgets import Button, Checkbox, Collapsible, Footer, Header, Input, Label, Markdown, ProgressBar, RichLog, Select, Static, TextArea

from lucifer.client import APIClient, APIError
from lucifer.adapter_map import AdapterBrain, admission_gate_lines, node_details
from lucifer.training_readings import LEGEND, outcome_lines, retained_updates

HELP = """/agent run [--workspace PATH --access read|write|execute|computer] GOAL
/agent status|stop|resume|open ID|approve ID|deny ID
/correct YOUR CORRECTION  ·  /resume [GUIDANCE]  ·  /agent auto-stop on|off
/agent continuation on|off  ·  /experts list|propose|build|consolidate
/agent workspace PATH  ·  /agent access read|write|execute|computer
/agent dialect chat_completions|anthropic_messages|responses
/agent image PATH|clear  ·  /agent vision on|off  ·  /agent mcp NAME1,NAME2|none
/agent remember [APPROVED SUMMARY]
/history  ·  F5 history  ·  F2 agent  ·  F3 adapter cortex
/model list | load PROFILE | unload | current
/model register PROFILE.json|PROFILE.yaml | reload
/thinking off|low|medium|high|max  ·  /tokens N|auto|max
/session new | save | export [PATH] | clear
/privacy logging on|off  ·  /privacy training on|off
/feedback good | bad | correct ANSWER
/train start [--dry-run] | status | cancel [ID]
/sleep start [--dry-run] [--legacy] [--adaptive] [--seed N] [--thinking MODE] [--generation-model PROFILE]
/sleep resume ID [--idempotency-key KEY] | view | status | queue | cancel [ID]  ·  F4 sleep
/sleep examples [pending|verified|rejected|obsolete|all]
/sleep approve EXAMPLE_ID [COMPLETE CORRECT ANSWER]  ·  /sleep reject EXAMPLE_ID
/meditation start [--mode supervised|unsupervised] [--current|--conversations ID,ID] [--base PROFILE] [--seed N] [--pilot-updates 1..100] [--dry-run]
/meditation status|bases|sources|results [ID]  ·  F6 Meditation
/meditation resume|cancel ID [--idempotency-key KEY]  ·  /meditation activate ID|rollback [ID]
/attach LOCAL_FILE [--frames N]  ·  /link LOCAL_PATH_OR_PUBLIC_URL [--frames N]
/knowledge list|search QUERY|read DOCUMENT_ID [CHUNK]|remove DOCUMENT_ID
/adapters  ·  /adapter select ID|base
/eval run ID | results  ·  /promote ID  ·  /rollback [ID]
/gpu list  ·  /memory QUERY  ·  /help  ·  /quit

New conversations are saved by default. Training stays OFF until you opt in.
Use /privacy logging off for a private conversation; /privacy training on opts in to learning.
Dry runs prepare datasets; they do not train an adapter.
Agent missions have no time deadline. STOP/Esc pauses the mission and retains
its thinking in session memory. Continue or correct the same mission normally.
History shows saved transcripts, retained sessions and unavailable private runs.
Logging OFF means private content will not survive a backend restart.
Ctrl+C pauses/interrupts owned work. Ctrl+Q quits.
Ctrl+N explicitly starts a new conversation and mission."""

THINKING_MODES = ("off", "low", "medium", "high", "max")
CONTEXT_RESERVE = 256
STREAM_INTERVAL = 0.05
TERMINAL_LEARNING_STATES = {"pilot_complete", "completed", "failed", "cancelled", "dry_run", "interrupted", "awaiting_promotion", "rejected", "no_new_information", "quality_unknown", "insufficient_verified_data", "needs_review", "admitted", "awaiting_evidence", "unreachable_gate", "paused", "blocked"}
SLEEP_PHASES = (
    ("CONTEXT", {"queued", "collecting", "recovering_context", "refreshing", "refreshing_data"}),
    ("EXAMPLES", {"drafting_examples", "validating_examples", "filtering"}),
    ("MEMORY", {"reconciling_memory", "consolidating"}),
    ("DATA", {"dataset_ready", "preparing_evaluation"}),
    ("TRAIN", {"unloading", "training"}),
    ("VERIFY", {"registering"}),
    ("EVALUATE", {"evaluating", "assessing_development"}),
    ("RESTORE", {"restoring"}),
    ("REVIEW", {"finished"}),
)


def sleep_start_options(args: list[str]) -> dict:
    """Parse explicit Sleep options while retaining the one-shot defaults."""
    options = {"dry_run": False, "context_examples": True}
    adaptive, seed, seen, cursor = False, None, set(), 0
    while cursor < len(args):
        flag = args[cursor]
        if flag in seen:
            raise ValueError(f"Duplicate Sleep option: {flag}")
        seen.add(flag)
        if flag in {"--dry-run", "--legacy", "--adaptive"}:
            if flag == "--dry-run":
                options["dry_run"] = True
            elif flag == "--legacy":
                options["context_examples"] = False
            else:
                adaptive = True
            cursor += 1
            continue
        if flag not in {"--seed", "--thinking", "--generation-model"} or cursor + 1 >= len(args):
            raise ValueError("Sleep options: --dry-run, --legacy, --adaptive, --seed N, --thinking MODE, --generation-model PROFILE")
        value = args[cursor + 1]
        if flag == "--seed":
            try:
                seed = int(value)
            except ValueError:
                raise ValueError("--seed needs an integer") from None
        elif flag == "--thinking":
            if value not in THINKING_MODES:
                raise ValueError("--thinking needs off, low, medium, high or max")
            options["generation_thinking"] = value
        elif not value.strip() or value.startswith("--"):
            raise ValueError("--generation-model needs a registered profile")
        else:
            options["generation_model"] = value
        cursor += 2
    if seed is not None and not adaptive:
        raise ValueError("--seed requires --adaptive")
    if adaptive:
        if not options["context_examples"]:
            raise ValueError("Adaptive Sleep requires context examples; remove --legacy")
        options["schedule"] = {"mode": "until_admitted", "refresh_optimizer_steps": 50, "initial_update_budget": 50}
        if seed is not None:
            options["schedule"]["seed"] = seed
        if options.get("generation_thinking", "off") not in {"off", "max"}:
            raise ValueError("Adaptive Sleep requires maximum teacher thinking")
        options["generation_thinking"] = "max"
    return options


def lucifer_mark() -> Text:
    """A compact original wing-and-halo monogram, without an ASCII face."""
    mark = Text("      ◇\n   ╱  │  ╲\n  ╱   L   ╲\n      ━\n L U C I F E R", style="bold #fcee0a")
    mark.stylize("#54d9e8", 8, 32)
    return mark


def angel_portrait() -> Text:
    # Existing integrations may import the old illustration function.
    return lucifer_mark()


def example_messages(record: dict) -> list[dict]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict)
            and message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str) for message in messages):
        return []
    return messages


def sanitized_exception(error: BaseException) -> dict:
    """Keep code locations only; never format messages, values or source lines."""
    pending, seen, failures = [error], set(), []
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        frames, cursor = [], current.__traceback__
        while cursor is not None:
            code = cursor.tb_frame.f_code
            frames.append({"file": code.co_filename, "function": code.co_name, "line": cursor.tb_lineno})
            cursor = cursor.tb_next
        failures.append({"type": type(current).__module__ + "." + type(current).__qualname__, "frames": frames})
        if isinstance(current, WorkerFailed):
            pending.append(current.error)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            pending.append(cause)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "failures": failures}


class ApprovalScreen(ModalScreen[bool]):
    """A concrete tool decision. Opening this screen never grants permission."""
    BINDINGS = [Binding("escape", "deny", "Deny")]

    def __init__(self, request: dict):
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("EXECUTION AUTHORIZATION / USER DECISION", id="approval-title")
            with VerticalScroll(id="approval-review"):
                name = self.request.get("name", self.request.get("tool", "Requested operation"))
                summary = self.request.get("summary", self.request.get("effect", "Review the exact arguments before approving."))
                yield Static(f"{name}\n{summary}", markup=False)
                yield Static(json.dumps(self.request, indent=2, ensure_ascii=False), markup=False)
            with Horizontal(id="approval-actions"):
                yield Button("DENY", id="deny-operation")
                yield Button("APPROVE THIS OPERATION", id="approve-operation")

    def on_mount(self):
        self.query_one("#deny-operation", Button).focus()

    @on(Button.Pressed, "#deny-operation")
    def denied(self):
        self.dismiss(False)

    @on(Button.Pressed, "#approve-operation")
    def approved(self):
        self.dismiss(True)

    def action_deny(self):
        self.dismiss(False)


class AgentHistoryScreen(ModalScreen[None]):
    """On-demand mission history. Viewing it never restores a checkpoint."""

    BINDINGS = [Binding("escape", "close_history", "Close", priority=True)]

    def __init__(self, owner, records: list[dict]):
        super().__init__()
        self.owner = owner
        self.records = {str(record.get("run_id") or record.get("id")): record for record in records
                        if record.get("run_id") or record.get("id")}
        self.sessions: dict[str, dict] = {}
        self.fetching = False
        self.selected_id: str | None = None

    def options(self):
        rows = []
        for ident, row in self.records.items():
            availability = ("RAM" if row.get("session_available") else
                            "CHECKPOINT" if row.get("restore_available") else
                            "CHECKPOINT BLOCKED" if row.get("checkpoint_available") else
                            "TRANSCRIPT" if row.get("transcript_available") else "UNAVAILABLE")
            title = str(row.get("title") or ident).replace("\n", " ")
            rows.append((f"{availability} / {title[:90]}", ident))
        return rows

    def compose(self) -> ComposeResult:
        with Vertical(id="history-dialog"):
            yield Label("AGENT / HISTORY", id="history-heading")
            yield Static("Open a retained session or saved transcript. Viewing history never resumes work.", id="history-help", markup=False)
            with Horizontal(id="history-selectors"):
                yield Select(self.options(), value=next(iter(self.records), Select.BLANK), prompt="No agent sessions", id="history-picker")
                yield Button("RELOAD", id="history-reload")
            with VerticalScroll(id="history-content"):
                yield Static("", id="history-metadata", markup=False)
                yield Static("", id="history-preview", markup=False)
            yield Static("", id="history-status", markup=False)
            with Horizontal(id="history-actions"):
                yield Button("OPEN SESSION", id="history-open")
                yield Button("NEW SESSION", id="history-new")
                yield Button("CLOSE", id="history-close")

    def on_mount(self) -> None:
        self.select_session()
        self.query_one("#history-picker", Select).focus()

    def controls(self) -> None:
        if not self.is_mounted or not self.owner.is_running:
            return
        blocked = self.fetching or self.owner.busy
        self.query_one("#history-open", Button).disabled = blocked or not self.selected_id
        self.query_one("#history-new", Button).disabled = blocked
        for selector in ("#history-picker", "#history-reload"):
            self.query_one(selector).disabled = self.fetching
        if self.owner.busy and not self.fetching:
            self.query_one("#history-status", Static).update("Pause the current mission before opening another session. You can inspect this history meanwhile.")

    @on(Select.Changed, "#history-picker")
    def selected(self, event: Select.Changed) -> None:
        if not self.fetching and str(event.value) != self.selected_id:
            self.select_session()

    def select_session(self) -> None:
        selected = self.query_one("#history-picker", Select).value
        self.selected_id = str(selected) if selected is not Select.BLANK else None
        row = self.records.get(self.selected_id or "")
        if not row:
            self.query_one("#history-metadata", Static).update("No agent sessions are listed yet. New conversations save by default; training stays OFF.")
            self.query_one("#history-preview", Static).update("")
            self.controls()
            return
        self.render_session(row, self.sessions.get(self.selected_id))
        if self.selected_id not in self.sessions:
            self.fetching = True
            self.controls()
            self.run_worker(self.fetch_session(self.selected_id), group="history-read")

    def render_session(self, row: dict, session: dict | None) -> None:
        run = (session or {}).get("run") or row
        conversation = (session or {}).get("conversation") or {}
        logging = conversation.get("logging_enabled", row.get("logging_enabled")) is True
        training = conversation.get("training_enabled", row.get("training_enabled")) is True
        lines = [f"MISSION // {row.get('run_id') or row.get('id')}",
                 f"STATUS // {run.get('status') or row.get('status') or 'unknown'}",
                 f"LOGGING // {'ON' if logging else 'OFF'} // TRAINING // {'ON' if training else 'OFF'}"]
        conversation_id = conversation.get("id") or run.get("conversation_id") or row.get("conversation_id")
        if conversation_id:
            lines.append("CONVERSATION // " + str(conversation_id))
        if run.get("session_available", row.get("session_available")):
            lines.append("AVAILABLE // backend session memory; emitted thinking can be replayed while this backend stays alive")
        elif run.get("checkpoint_available", row.get("checkpoint_available")):
            lines.append("AVAILABLE // saved checkpoint; OPEN reads it, RESUME explicitly continues under the recorded scope")
            lines.append("THINKING // session memory ended; provider reasoning was never saved")
        elif run.get("transcript_available", row.get("transcript_available")):
            lines.append("AVAILABLE // public transcript only; runtime session cannot currently resume")
            lines.append("THINKING // session memory ended; provider reasoning was never saved")
        else:
            reason = (session or {}).get("transcript_unavailable_reason") or run.get("transcript_unavailable_reason") or row.get("transcript_unavailable_reason") or "no_saved_transcript"
            lines.append("UNAVAILABLE // " + str(reason).replace("_", " "))
            if not logging:
                lines.append("Logging was OFF. Private message text cannot be reconstructed from operation receipts.")
        if run.get("restore_unavailable_reason") or row.get("restore_unavailable_reason"):
            lines.append("RESTORE // " + str(run.get("restore_unavailable_reason") or row["restore_unavailable_reason"]))
        control_notice = run.get("control_notice") or row.get("control_notice")
        if isinstance(control_notice, str) and control_notice:
            lines.append("CONTROL NOTICE // " + control_notice)
        for key in ("workspace", "access"):
            if run.get(key) or row.get(key):
                lines.append(key.upper() + " // " + str(run.get(key) or row[key]))
        self.query_one("#history-metadata", Static).update(Text("\n".join(lines)))
        messages = (session or {}).get("messages") or []
        public = [message for message in messages if isinstance(message, dict)
                  and message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)]
        preview = "\n\n".join(f"{message['role'].upper()}" + (" / " + message["status"].upper() if message.get("status") in {"partial", "in_progress"} else "")
                                + " // " + message["content"][:2500] for message in public[-6:])
        self.query_one("#history-preview", Static).update(Text(preview or ("No saved public messages." if session is not None else "Loading public transcript…")))
        self.query_one("#history-status", Static).update("Preview shows the latest public turns. OPEN loads the available session; it does not run tools.")
        self.controls()

    async def fetch_session(self, ident: str) -> None:
        try:
            session = await self.owner.client.agent_session(ident)
            if not self.is_mounted or not self.owner.is_running:
                return
            self.sessions[ident] = session
            if ident == self.selected_id:
                self.render_session(self.records[ident], session)
        except APIError as error:
            if self.is_mounted and self.owner.is_running:
                self.query_one("#history-status", Static).update(f"History unavailable: {error}")
        finally:
            self.fetching = False
            self.controls()

    @on(Button.Pressed, "#history-reload")
    def reload_clicked(self) -> None:
        if not self.fetching:
            self.fetching = True
            self.controls()
            self.run_worker(self.reload_history(), group="history-read")

    async def reload_history(self) -> None:
        try:
            records = await self.owner.client.agent_history()
            if not self.is_mounted or not self.owner.is_running:
                return
            self.records = {str(row.get("run_id") or row.get("id")): row for row in records if row.get("run_id") or row.get("id")}
            self.sessions.clear()
            picker = self.query_one("#history-picker", Select)
            with picker.prevent(Select.Changed):
                picker.set_options(self.options())
                picker.value = self.selected_id if self.selected_id in self.records else next(iter(self.records), Select.BLANK)
            self.fetching = False
            self.select_session()
        except APIError as error:
            if self.is_mounted and self.owner.is_running:
                self.query_one("#history-status", Static).update(f"History unavailable: {error}")
                self.fetching = False
                self.controls()

    @on(Button.Pressed, "#history-open")
    def open_clicked(self) -> None:
        if not self.fetching and not self.owner.busy and self.selected_id:
            ident = self.selected_id
            self.dismiss()
            self.owner.call_after_refresh(self.owner.submit, "/agent open " + shlex.quote(ident))

    @on(Button.Pressed, "#history-new")
    def new_clicked(self) -> None:
        if not self.fetching and not self.owner.busy:
            self.dismiss()
            self.owner.call_after_refresh(self.owner.submit, "/session new")

    @on(Button.Pressed, "#history-close")
    def action_close_history(self) -> None:
        self.dismiss()
        self.owner.call_after_refresh(self.owner.update_readiness)


class ExampleReviewScreen(ModalScreen[None]):
    """Explicit review of canonical examples; content remains literal text."""

    BINDINGS = [Binding("escape", "close_review", "Close", priority=True)]

    def __init__(self, owner, records: list[dict], *, status: str = "pending"):
        super().__init__()
        self.owner = owner
        self.records = {str(record["id"]): record for record in records if record.get("id")}
        self.filter_status = status
        self.displayed: dict | None = None
        self.review_busy = False
        self.drafts: dict[str, tuple[str, str]] = {}

    def options(self):
        return [(f"{'STALE DISPLAY' if record.get('_ui_stale') else str(record.get('status') or 'unknown').upper()} / {ident}", ident)
                for ident, record in self.records.items()]

    def compose(self) -> ComposeResult:
        with Vertical(id="example-dialog"):
            yield Label("SLEEP / REVIEW EXAMPLES", id="example-heading")
            yield Static("Review the exact question and answer below. A correction replaces the complete answer.", id="example-help", markup=False)
            with Horizontal(id="example-selectors"):
                yield Select([(value.upper(), value) for value in ("pending", "verified", "rejected", "obsolete", "all")],
                             value=self.filter_status, allow_blank=False, id="example-filter")
                yield Select(self.options(), value=next(iter(self.records), Select.BLANK), prompt="No examples", id="example-picker")
                yield Button("RELOAD", id="example-reload")
            with VerticalScroll(id="example-content"):
                yield Static("", id="example-hash", markup=False)
                yield Static("", id="example-question", markup=False)
                yield Static("", id="example-answer", markup=False)
                yield Collapsible(Static("", id="example-metadata", markup=False), title="SOURCE / VALIDATION",
                                  collapsed=True, id="example-provenance")
            yield Label("COMPLETE CORRECTION / optional", id="example-correction-label")
            yield TextArea("", id="example-correction", soft_wrap=True)
            yield Input(placeholder="Optional review reason", id="example-reason")
            yield Static("", id="example-status", markup=False)
            with Horizontal(id="example-actions"):
                yield Button("APPROVE DISPLAYED", id="example-approve")
                yield Button("REJECT DISPLAYED", id="example-reject")
                yield Button("CLOSE", id="example-close")

    def on_mount(self) -> None:
        self.render_candidate()
        self.query_one("#example-picker", Select).focus()

    def controls(self) -> None:
        if not self.is_mounted or not self.owner.is_running:
            return
        messages = example_messages(self.displayed or {})
        complete = all(any(message.get("role") == role and str(message.get("content", "")).strip()
                           for message in messages) for role in ("user", "assistant")) and all(
                               message.get("role") in {"user", "assistant"} for message in messages)
        valid = bool(complete and self.displayed and not self.displayed.get("_ui_stale") and self.displayed.get("status") != "obsolete"
                     and re.fullmatch(r"[0-9a-fA-F]{64}", str(self.displayed.get("content_hash", ""))))
        for selector in ("#example-approve", "#example-reject"):
            self.query_one(selector, Button).disabled = self.review_busy or not valid
        for selector in ("#example-filter", "#example-picker", "#example-reload", "#example-correction", "#example-reason", "#example-close"):
            self.query_one(selector).disabled = self.review_busy

    def render_candidate(self, *, reset_draft: bool = False) -> None:
        if self.displayed:
            ident = str(self.displayed["id"])
            if reset_draft:
                self.drafts.pop(ident, None)
            else:
                self.drafts[ident] = (self.query_one("#example-correction", TextArea).text,
                                      self.query_one("#example-reason", Input).value)
        selected = self.query_one("#example-picker", Select).value
        record = self.records.get(str(selected))
        self.displayed = dict(record) if record else None
        correction, reason = self.drafts.get(str(selected), ("", ""))
        self.query_one("#example-correction", TextArea).load_text(correction)
        self.query_one("#example-reason", Input).value = reason
        if not record:
            self.query_one("#example-hash", Static).update("No examples for this filter. Nothing has been approved.")
            self.query_one("#example-metadata", Static).update("No examples for this filter. Nothing has been approved.")
            self.query_one("#example-question", Static).update("")
            self.query_one("#example-answer", Static).update("")
            self.controls()
            return
        validation = record.get("validation") or {}
        if not isinstance(validation, dict):
            validation = {"method": validation} if isinstance(validation, str) else {}
        validators = validation.get("validators") or validation.get("validator") or validation.get("method") or "not recorded"
        if isinstance(validators, list):
            validators = ", ".join(str(item) for item in validators)
        if validation.get("human_reviewed") is True and "human_reviewed" not in str(validators):
            validators = str(validators) + ", human_reviewed"
        if validation.get("method") == "explicit_human_attestation":
            validators = "human_reviewed / explicit_human_attestation"
        self.query_one("#example-provenance", Collapsible).title = "SOURCE / VALIDATOR // " + str(validators)
        self.query_one("#example-hash", Static).update(Text("CONTENT HASH // " + str(record.get("content_hash") or "missing; reload before review")))
        metadata = (f"ID // {record['id']}\nSTATUS // {record.get('status', 'unknown')} / PARTITION // {record.get('partition') or 'unassigned'}\n"
                    f"SOURCE GROUP // {record.get('source_group_id') or 'unknown'}\nVALIDATOR // {validators}")
        if record.get("_ui_stale"):
            metadata = "STALE DISPLAY // absent from the reloaded filter; choose a current record\n" + metadata
        for key in ("scope_id", "task_id", "source_message_id"):
            if record.get(key):
                metadata += f"\n{key.upper().replace('_', ' ')} // {record[key]}"
        for binding in record.get("source_bindings") or []:
            if isinstance(binding, dict) and binding.get("snapshot_sha256"):
                metadata += "\nSOURCE SNAPSHOT // " + str(binding["snapshot_sha256"])
        provenance = record.get("provenance") or {}
        if isinstance(provenance, dict):
            for key in ("conversation_id", "source_conversation_id", "source_task_id", "recovery_method"):
                if provenance.get(key):
                    metadata += f"\n{key.upper().replace('_', ' ')} // {provenance[key]}"
            constraints = provenance.get("active_constraints") or {}
            if isinstance(constraints, dict):
                for key, value in constraints.items():
                    value = value.get("value") if isinstance(value, dict) else value
                    metadata += f"\nACTIVE CONSTRAINT // {key}: {value}"
            changes = provenance.get("supersedes") or []
            if isinstance(changes, list) and changes:
                metadata += f"\nSUPERSEDES // {len(changes)} earlier constraints"
        if validation.get("status"):
            metadata += "\nVALIDATION STATUS // " + str(validation["status"])
        self.query_one("#example-metadata", Static).update(Text(metadata))
        messages = example_messages(record)
        questions = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
        answers = [str(message.get("content", "")) for message in messages if message.get("role") == "assistant"]
        self.query_one("#example-question", Static).update(Text("QUESTION\n" + "\n\n".join(questions)))
        self.query_one("#example-answer", Static).update(Text("ANSWER\n" + "\n\n".join(answers)))
        self.owner.record_displayed_example(record)
        self.controls()

    @on(Select.Changed, "#example-picker")
    def candidate_selected(self) -> None:
        if self.is_mounted:
            self.render_candidate()

    @on(Select.Changed, "#example-filter")
    def filter_selected(self, event: Select.Changed) -> None:
        if self.review_busy:
            with self.query_one("#example-filter", Select).prevent(Select.Changed):
                self.query_one("#example-filter", Select).value = self.filter_status
            return
        if event.value is not Select.BLANK and event.value != self.filter_status:
            self.filter_status = str(event.value)
            self.run_worker(self.reload_examples(), group="example-review-control")

    @on(Button.Pressed, "#example-reload")
    def reload_clicked(self) -> None:
        if not self.review_busy:
            self.run_worker(self.reload_examples(), group="example-review-control")

    async def reload_examples(self) -> None:
        if self.review_busy:
            return
        self.review_busy = True
        self.controls()
        try:
            prior = dict(self.displayed) if self.displayed else None
            draft = (self.query_one("#example-correction", TextArea).text, self.query_one("#example-reason", Input).value)
            records = await self.owner.client.sleep_examples(None if self.filter_status == "all" else self.filter_status)
            if not self.owner.is_running or not self.is_mounted:
                return
            self.records = {str(record["id"]): record for record in records if record.get("id")}
            retained = bool(prior and str(prior["id"]) not in self.records and any(draft))
            if retained:
                prior["_ui_stale"] = True
                self.records[str(prior["id"])] = prior
            self.owner.cache_examples(records)
            picker = self.query_one("#example-picker", Select)
            selected = picker.value
            with picker.prevent(Select.Changed):
                picker.set_options(self.options())
                picker.value = selected if str(selected) in self.records else next(iter(self.records), Select.BLANK)
            self.render_candidate()
            self.query_one("#example-status", Static).update(
                "Previous example left this filter. Draft retained for copying; choose a current record." if retained else
                "Reloaded. Review the currently displayed content before deciding.")
        except APIError as exc:
            if self.owner.is_running and self.is_mounted:
                self.query_one("#example-status", Static).update(Text(str(exc), style="#ff636c"))
        finally:
            self.review_busy = False
            self.controls()

    @on(Button.Pressed, "#example-approve")
    def approve_clicked(self) -> None:
        if not self.review_busy:
            self.run_worker(self.review("approve"), group="example-review-control")

    @on(Button.Pressed, "#example-reject")
    def reject_clicked(self) -> None:
        if not self.review_busy:
            self.run_worker(self.review("reject"), group="example-review-control")

    async def review(self, decision: str) -> None:
        if self.review_busy or not self.displayed:
            return
        correction = self.query_one("#example-correction", TextArea).text if decision == "approve" else ""
        reason = self.query_one("#example-reason", Input).value.strip()
        if len(correction) > 100000 or len(reason) > 2000:
            self.query_one("#example-status", Static).update("Corrections allow 100,000 characters; reasons allow 2,000.")
            return
        self.review_busy = True
        self.controls()
        try:
            updated = await self.owner.submit_example_review(str(self.displayed["id"]), decision,
                correction=correction if correction.strip() else None, reason=reason or None,
                content_hash=str(self.displayed.get("content_hash", "")))
            if not self.owner.is_running or not self.is_mounted:
                return
            previous = str(self.displayed["id"])
            if str(updated["id"]) != previous:
                self.records[previous] = {**self.records[previous], "status": "obsolete", "superseded_by": updated["id"]}
            self.records[str(updated["id"])] = updated
            picker = self.query_one("#example-picker", Select)
            with picker.prevent(Select.Changed):
                picker.set_options(self.options())
                picker.value = str(updated["id"])
            self.render_candidate(reset_draft=True)
            self.query_one("#example-status", Static).update(f"Review saved. Current status: {updated.get('status', 'unknown')}.")
        except (APIError, ValueError) as exc:
            if self.owner.is_running and self.is_mounted:
                self.query_one("#example-status", Static).update(Text(f"{exc}\nYour correction is retained. Reload explicitly if the content changed.", style="#ff636c"))
        finally:
            self.review_busy = False
            self.controls()

    @on(Button.Pressed, "#example-close")
    def close_clicked(self) -> None:
        self.action_close_review()

    def action_close_review(self) -> None:
        if not self.review_busy:
            self.dismiss(None)
            self.owner.call_after_refresh(self.owner.update_readiness)


class MarkdownBuffer:
    """Coalesce tiny network fragments before incrementally parsing Markdown."""

    def __init__(self, widget: Markdown):
        self.widget = widget
        self.source = ""
        self.pending: list[str] = []
        self.task: asyncio.Task | None = None

    def write(self, fragment: str) -> None:
        if not fragment:
            return
        self.source += fragment
        self.pending.append(fragment)
        if self.task is None:
            self.task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while self.pending:
                await asyncio.sleep(STREAM_INTERVAL)
                fragment = "".join(self.pending)
                self.pending.clear()
                await self.widget.append(fragment)
        finally:
            self.task = None

    async def finish(self) -> None:
        if self.task is not None:
            await self.task
        if self.pending:
            fragment = "".join(self.pending)
            self.pending.clear()
            await self.widget.append(fragment)
        # Settle the final document once. Textual 8's streamed fence retains
        # stale code metadata until a full update; never do this per token.
        if self.source:
            await self.widget.update(self.source)


class AssistantMessage(Vertical):
    """Final Markdown and the actual, ephemeral upstream thinking channel."""

    def __init__(self, content: str = "", *, thinking_mode: str | None = None, **kwargs):
        self.initial_content = content
        self.thinking_mode = thinking_mode
        self.answer = Markdown(content, classes="answer")
        self.thinking = Markdown("", classes="thinking-content")
        self.thinking_panel = Collapsible(self.thinking, title="Model thinking", collapsed=True, classes="thinking")
        if thinking_mode == "off":
            self.thinking_panel.title = "Model thinking · off"
        elif content:
            self.thinking_panel.title = "Model thinking · not retained"
        self.answer_buffer = MarkdownBuffer(self.answer)
        self.thinking_buffer = MarkdownBuffer(self.thinking)
        self.tool_log = RichLog(classes="tool-timeline", wrap=True, markup=False, highlight=False, max_lines=100)
        super().__init__(classes="message assistant", **kwargs)
        self.border_title = " LUCIFER // RX "

    def compose(self) -> ComposeResult:
        yield self.tool_log
        yield self.thinking_panel
        yield self.answer
        yield Static("" if self.initial_content else "Waiting for Lucifer…", classes="reply-state", markup=False)

    def append_answer(self, fragment: str) -> None:
        self.query_one(".reply-state", Static).update("")
        self.answer_buffer.write(fragment)

    def append_thinking(self, fragment: str) -> None:
        self.thinking_panel.title = "Model thinking · streaming"
        self.query_one(".reply-state", Static).update("Thinking…")
        self.thinking_buffer.write(fragment)

    def append_tool(self, event: str, data: dict) -> None:
        self.tool_log.display = True
        name = data.get("name", data.get("tool", "tool"))
        summary = data.get("summary", data.get("status", ""))
        line = Text(f"{event.upper().replace('_', ' ')} / {name}", style="bold #54d9e8")
        if summary:
            line.append(f" / {summary}", style="#91a8bb")
        self.tool_log.write(line)

    async def finish(self, *, partial: bool = False, paused: bool = False) -> None:
        await asyncio.gather(self.answer_buffer.finish(), self.thinking_buffer.finish())
        self.thinking_panel.title = (
            "Model thinking" if self.thinking_buffer.source else
            "Model thinking · off" if self.thinking_mode == "off" else "Model thinking · not streamed"
        )
        self.query_one(".reply-state", Static).update(
            "Paused · thinking and partial response retained in session memory" if paused else
            "Interrupted · partial response" if partial else "" if self.answer_buffer.source or self.initial_content
            else "No final answer was returned."
        )
        if partial:
            self.add_class("partial")
            self.border_subtitle = " PAUSED // resumable session " if paused else " PARTIAL // not saved as a finished reply "


class Message(Static):
    def __init__(self, role: str, content: str = "", **kwargs):
        super().__init__(Text(content), classes=f"message {role}", **kwargs)
        self.border_title = {"user": " YOU // TX ", "assistant": " LUCIFER // RX ", "system": " CONSOLE "}.get(role, role.upper())
        if role == "system" and content.startswith("Error:"):
            self.add_class("error")


class SleepMeter(Widget, can_focus=False):
    """A bar whose reactive value can never enable hidden animation."""

    COMPONENT_CLASSES = {"bar--bar", "bar--complete", "bar--indeterminate"}
    DEFAULT_CSS = "SleepMeter { width: 1fr; height: 1; }"
    percentage: reactive[float | None] = reactive(None)

    def __init__(self, **kwargs):
        self.pulsing = False
        super().__init__(**kwargs)

    def watch_percentage(self, value: float | None) -> None:
        self.set_pulsing(self.pulsing)

    def set_pulsing(self, enabled: bool) -> None:
        self.pulsing = enabled
        interval = 1 / 15 if enabled and self.percentage is None and self.app.animation_level != "none" else None
        if self.auto_refresh != interval:
            self.auto_refresh = interval

    def render(self):
        width = self.size.width
        if self.percentage is None:
            component = "bar--indeterminate"
            span = width * .25
            distance = width + span
            start = (30 * monotonic()) % (2 * distance) if distance else 0
            if start > distance:
                start = 2 * distance - start
            start -= span
            bounds = (max(0, start), min(width, start + span)) if self.app.animation_level != "none" else (0, width)
        else:
            component = "bar--complete" if self.percentage >= 1 else "bar--bar"
            bounds = (0, width * self.percentage)
        style = self.get_component_rich_style(component)
        return BarRenderable(highlight_range=bounds, highlight_style=Style.from_color(style.color),
                             background_style=Style.from_color(style.bgcolor))


class SleepProgressBar(ProgressBar):
    """Heartbeat-driven progress; animate only visible, active unknown work."""

    def on_mount(self, event: events.Mount) -> None:
        # The native mount handler installs an ETA timer even when ETA is hidden.
        # Sleep has no ETA and receives all progress through the shared heartbeat.
        event.prevent_default()
        self.update()
        self.set_pulsing(False)

    def compose(self) -> ComposeResult:
        yield SleepMeter(id="bar").data_bind(ProgressBar.percentage)

    def set_pulsing(self, enabled: bool) -> None:
        self.query_one("#bar", SleepMeter).set_pulsing(enabled)


class DeleteExpertScreen(ModalScreen):
    CSS = """DeleteExpertScreen { align: center middle; }
    #delete-expert-dialog { width: 75%; height: auto; padding: 1 2; border: heavy #ff515b; background: #0b1521; }
    #delete-expert-dialog Button { margin: 1; }"""

    def __init__(self, ident):
        super().__init__()
        self.ident = ident

    def compose(self):
        with Vertical(id="delete-expert-dialog"):
            yield Static(f"Delete expert {self.ident}?\nOwned checkpoint files move to Trash. Audit history remains. External files are preserved.", markup=False)
            yield Button("KEEP EXPERT", id="keep-expert")
            yield Button("DELETE TO TRASH", id="confirm-delete-expert")

    @on(Button.Pressed)
    def decide(self, event):
        self.dismiss(event.button.id == 'confirm-delete-expert')


class MeditationReviewScreen(ModalScreen[None]):
    """Inspection only: saving a revision never activates it."""
    BINDINGS = [Binding("escape", "close_review", "Close", priority=True)]
    CSS = """MeditationReviewScreen { align: center middle; }
    #meditation-review-dialog { width: 90%; height: 90%; padding: 1 2; border: heavy #54d9e8; background: #0b1521; }
    #meditation-review-body { height: 1fr; }
    #meditation-review-close { height: 3; }"""

    def __init__(self, title: str, record: dict, owner):
        super().__init__()
        self.title_text, self.record = title, record
        self.owner = owner

    def compose(self):
        with Vertical(id="meditation-review-dialog"):
            yield Static(self.title_text, markup=False)
            with VerticalScroll(id="meditation-review-body"):
                # A literal Static prevents source text from acting as markup.
                yield Static(json.dumps(LuciferApp.public_result(self.record), indent=2, ensure_ascii=False), markup=False)
                yield Static("Evaluation and rollback checks gate activation automatically. This inspection is optional. Saving a base leaves the current model unchanged.", markup=False)
            yield Button("CLOSE", id="meditation-review-close")

    @on(Button.Pressed, "#meditation-review-close")
    def close_clicked(self):
        self.dismiss(None)

    def action_close_review(self):
        self.dismiss(None)

    def on_unmount(self):
        if self.owner.is_running:
            self.owner.call_after_refresh(self.owner.update_readiness)


class LuciferApp(App):
    TITLE = "LUCIFER / NEURAL OPERATIONS"
    SUB_TITLE = "NEURAL LINK // YOUR SIGNAL. YOUR CONTROL."
    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New session"),
        Binding("ctrl+l", "clear_chat", "Clear view"),
        Binding("f1", "help", "Commands"),
        Binding("f2", "toggle_agent", "Agent"),
        Binding("f3", "toggle_brain", "Cortex"),
        Binding("f4", "toggle_sleep", "Sleep"),
        Binding("f5", "history", "History"),
        Binding("f6", "toggle_meditation", "Meditation"),
        Binding("escape", "cancel_request", "Interrupt"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "cancel_request", "Pause", show=False, priority=True),
    ]
    CSS = """
    Screen { background: #080c14; color: #e4e9ef; scrollbar-color: #ff515b; scrollbar-background: #111a27; scrollbar-color-hover: #fcee0a; scrollbar-size: 1 1; }
    Header { height: 1; background: #090e18; color: #fcee0a; text-style: bold; }
    Header .header--title { text-style: bold; }
    Footer { background: #080c14; color: #8fa7ba; }
    FooterKey > .footer-key--key { background: #ff515b; color: #080c14; }
    FooterKey > .footer-key--description { background: #101824; color: #d0dae4; }
    #masthead { height: 3; padding: 0 2; background: #fcee0a; color: #080c14; border-bottom: heavy #ff515b; }
    #workspace { height: 1fr; }
    #sidebar { width: 30; min-width: 25; background: #0c121d; padding: 0 1; border-right: solid #ff515b; }
    #skull { height: 7; margin-top: 1; content-align: center middle; border-bottom: solid #573640; border-title-color: #fcee0a; }
    .panel-title { color: #ff636c; text-style: bold; margin-top: 1; height: 1; padding: 0; }
    #models, #adapters, #status, #gpus { height: auto; margin-top: 0; }
    #models { max-height: 5; }
    #adapters { max-height: 4; }
    Select { height: 3; margin-top: 0; }
    Select > SelectCurrent { background: #111b29; border: tall #34525d; color: #54d9e8; }
    Select:focus > SelectCurrent { border: tall #fcee0a; color: #fcee0a; }
    #jobs { height: auto; min-height: 2; margin-bottom: 1; }
    #chat-pane { width: 1fr; }
    #view-nav { height: 3; padding: 0 1; background: #080c14; border-bottom: solid #433340; }
    #view-nav Button { min-width: 12; width: 1fr; height: 2; border: none; background: #101824; color: #ff636c; margin-right: 1; }
    #view-nav Button.selected { background: #ff515b; color: #080c14; text-style: bold; }
    #session { height: 2; padding: 0 2; color: #8fa7ba; background: #0a111c; border-bottom: solid #203845; }
    #thinking-controls { height: 3; padding: 0 1; background: #0a111c; }
    #thinking-label { width: 10; height: 3; content-align: left middle; color: #ff636c; }
    #thinking-picker { width: 17; }
    #generation-settings { width: 1fr; height: 3; padding: 0 1; content-align: left middle; color: #839eae; }
    #agent-controls { display: none; height: 3; padding: 0 1; background: #0c1722; }
    .agent-mode #agent-controls { display: block; }
    #workspace-label { width: 6; height: 3; content-align: left middle; color: #ff636c; }
    #agent-workspace { width: 1fr; min-width: 12; background: #0c1722; border: tall #34525d; }
    #agent-access { width: 15; margin-left: 1; }
    #agent-dialect { width: 23; margin-left: 1; }
    #agent-auto { display: none; height: 3; width: 18; padding: 0 1; background: #0c1722; color: #ff636c; }
    .command-access #agent-auto { display: block; }
    #agent-options { display: none; height: 3; padding: 0 1; background: #0c1722; }
    .agent-mode #agent-options { display: block; }
    #agent-stop { width: 28; height: 3; padding: 0 1; color: #54d9e8; }
    #agent-continue { width: 31; height: 3; padding: 0 1; color: #54d9e8; }
    #resume-mission { width: 14; min-width: 12; height: 3; margin-left: 1; background: #142635; color: #fcee0a; border: tall #34525d; }
    #agent-limits { width: 1fr; height: 3; content-align: left middle; color: #839eae; padding: 0 1; }
    #mission-status { height: auto; max-height: 3; padding: 0 2; color: #54d9e8; background: #0b1722; }
    .agent-mode #mission-status { display: block; }
    #messages { height: 1fr; padding: 0 2 1 2; }
    .message { height: auto; margin-top: 1; padding: 1 2; border-left: solid #3e5767; background: #0c131f; }
    .user { border-left: heavy #ff515b; border-title-color: #ff636c; background: #14121c; }
    .assistant { border-left: heavy #fcee0a; border-bottom: solid #45513a; border-title-color: #fcee0a; background: #0b141c; }
    .assistant.partial { border-left: heavy #ff515b; border-subtitle-color: #ff515b; }
    .answer, .thinking-content { height: auto; padding: 0; margin: 0; background: transparent; }
    .thinking { height: auto; padding: 0; margin: 0 0 1 0; border: none; background: #0e1a26; }
    .thinking > CollapsibleTitle { color: #54d9e8; padding: 0; }
    .thinking > Contents { padding: 0 1; }
    .thinking-content { color: #91b5c6; }
    .reply-state { height: auto; color: #7c9aaa; text-style: italic; }
    MarkdownH1, MarkdownH2, MarkdownH3 { color: #fcee0a; background: transparent; }
    MarkdownFence { background: #101b29; margin: 1 0; }
    MarkdownTable { margin: 1 0; }
    .system { border-left: solid #34525d; border-title-color: #54d9e8; color: #9bb0bf; padding: 0 1; background: #0a111c; }
    .system.error { border-left: heavy #ff515b; color: #ff9da2; }
    .tool-event { border-left: solid #54d9e8; background: #0e1a26; color: #92becd; }
    .tool-timeline { display: none; height: auto; min-height: 2; max-height: 6; margin-bottom: 1; background: #0e1a26; color: #54d9e8; }
    #input-row { height: 3; padding: 0 1; }
    #correction-row { display: none; height: 3; padding: 0 1; }
    .agent-running #correction-row { display: block; }
    .agent-running #input-row { display: none; }
    #correction { width: 1fr; background: #0d1723; border: tall #54d9e8; color: #e4e9ef; }
    #correct-mission { width: 16; min-width: 16; height: 3; margin-left: 1; background: #54d9e8; color: #080c14; border: tall #54d9e8; }
    #prompt { width: 1fr; background: #0d1723; border: tall #ff515b; color: #e4e9ef; }
    #prompt:focus { border: tall #fcee0a; }
    #send { height: 3; content-align: center middle; padding: 0 1; width: 14; min-width: 14; margin-left: 1; background: #fcee0a; color: #080c14; text-style: bold; border: tall #fcee0a; }
    #send:hover, #send:focus { background: #54d9e8; border: tall #54d9e8; }
    #send:disabled { background: #28313a; color: #667d8d; border: tall #28313a; }
    #activity { height: 4; border-top: solid #ff515b; border-title-color: #ff636c; background: #080e17; padding: 0 1; }
    #busy { height: 1; padding: 0 2; color: #fcee0a; }
    #brain-view { display: none; height: 1fr; background: #0a111c; }
    .brain-view #brain-view { display: block; }
    .brain-view #messages, .brain-view #thinking-controls, .brain-view #agent-controls, .brain-view #agent-options, .brain-view #mission-status, .brain-view #session { display: none; }
    #brain-heading { height: 3; padding: 0 2; color: #fcee0a; border-bottom: solid #ff515b; }
    #brain-body { height: 1fr; }
    #brain-canvas { width: 1fr; height: 1fr; min-height: 8; }
    #brain-canvas:focus { background: #0d1621; }
    #brain-inspector { width: 31; height: 1fr; border-left: solid #ff515b; padding: 0 1; }
    #brain-details-scroll { height: 1fr; }
    #brain-details { padding: 0; margin: 0; background: transparent; }
    #brain-select { height: 3; width: 100%; min-width: 0; background: #162534; color: #54d9e8; border: tall #34525d; }
    #brain-legend { height: 2; padding: 0 1; color: #8fa7ba; background: #101a25; }
    #sleep-view { display: none; height: 1fr; background: #0b1521; padding: 0 2; }
    .sleep-view #sleep-view { display: block; }
    .sleep-view #messages, .sleep-view #thinking-controls, .sleep-view #agent-controls, .sleep-view #agent-options, .sleep-view #mission-status, .sleep-view #session, .sleep-view #input-row, .sleep-view #correction-row, .sleep-view #busy { display: none; }
    #sleep-heading { height: 2; content-align: left middle; color: #fcee0a; text-style: bold; border-bottom: solid #ff515b; }
    #sleep-actions { height: 3; margin-top: 1; }
    #sleep-actions Button { height: 3; width: 1fr; min-width: 12; margin-right: 1; background: #142635; color: #54d9e8; border: tall #34525d; }
    #sleep-actions #sleep-start { width: 28; background: #fcee0a; color: #080c14; border: tall #fcee0a; text-style: bold; }
    #sleep-actions #sleep-cancel { color: #ff636c; border: tall #773d4b; }
    #sleep-actions Button:disabled { background: #15212b; color: #667d8d; border: tall #283b47; }
    #sleep-details { height: 1fr; width: 1fr; padding: 1 0; }
    .training-body { height: 1fr; }
    .training-legend-pane { width: 32; padding: 1; margin-left: 1; border-left: solid #34525d; }
    .training-legend { height: auto; color: #91a8bb; }
    .training-legend-inline { display: none; height: auto; color: #91a8bb; margin-bottom: 1; }
    .training-outcomes { height: auto; margin-bottom: 1; padding: 0 1; border-left: solid #fcee0a; color: #e4e9ef; }
    .compact .training-legend-pane { display: none; }
    .compact .training-legend-inline { display: block; }
    #sleep-intro, #sleep-plan, #sleep-result { height: auto; color: #91a8bb; margin-bottom: 1; }
    #sleep-run { height: auto; color: #54d9e8; }
    #sleep-journey { height: auto; margin: 1 0; }
    #sleep-overall-label, #sleep-stage, #sleep-stage-count, #sleep-counters { height: auto; }
    #sleep-overall-label { color: #fcee0a; }
    #sleep-stage, #sleep-stage-count, #sleep-counters { color: #54d9e8; }
    #sleep-overall, #sleep-stage-bar { width: 1fr; margin-bottom: 1; }
    #sleep-overall SleepMeter, #sleep-stage-bar SleepMeter { width: 1fr; }
    #sleep-overall .bar--bar, #sleep-overall .bar--complete, #sleep-overall .bar--indeterminate { color: #fcee0a; background: #14232d; }
    #sleep-stage-bar .bar--bar, #sleep-stage-bar .bar--complete, #sleep-stage-bar .bar--indeterminate { color: #54d9e8; background: #14232d; }
    #sleep-counters { margin: 0 0 1 0; padding: 0 1; border-left: solid #54d9e8; background: #0e1d29; }
    #sleep-result { padding: 0 1; border-left: solid #ff515b; }
    #approval-dialog { width: 85%; height: 85%; padding: 1 2; border: heavy #ff515b; background: #0b1521; }
    ApprovalScreen { align: center middle; background: #000000b0; }
    #approval-title { height: 2; color: #fcee0a; text-style: bold; }
    #approval-review { height: 1fr; }
    #approval-actions { height: 3; }
    #approval-actions Button { width: 1fr; height: 3; background: #142331; color: #54d9e8; border: tall #34525d; margin: 0 1; }
    #approval-actions #approve-operation { background: #fcee0a; color: #080c14; border: tall #fcee0a; }
    ExampleReviewScreen { align: center middle; background: #000000b0; }
    #example-dialog { width: 94%; height: 94%; padding: 0 1; border: heavy #54d9e8; background: #0b1521; }
    #example-heading { height: 2; color: #fcee0a; text-style: bold; }
    #example-help { height: auto; max-height: 3; color: #91a8bb; }
    #example-selectors { height: 3; }
    #example-filter { width: 16; }
    #example-picker { width: 1fr; margin-left: 1; }
    #example-reload { width: 10; min-width: 10; height: 3; margin-left: 1; color: #54d9e8; background: #142635; border: tall #34525d; }
    #example-content { height: 1fr; padding: 0 1; border-left: solid #34525d; background: #0d1a26; }
    #example-hash, #example-metadata, #example-question, #example-answer { height: auto; margin-bottom: 1; }
    #example-hash { color: #54d9e8; }
    #example-provenance { height: auto; border: none; background: #0d1a26; padding: 0; }
    #example-provenance > Contents { padding: 0; }
    #example-metadata { color: #91a8bb; }
    #example-question { color: #fcee0a; }
    #example-answer { color: #e4e9ef; }
    #example-correction-label { height: 1; color: #54d9e8; }
    #example-correction { height: 6; background: #0d1723; border: tall #34525d; color: #e4e9ef; }
    #example-reason { height: 3; background: #0d1723; border: tall #34525d; }
    #example-status { height: auto; max-height: 3; color: #54d9e8; }
    #example-actions { height: 3; }
    #example-actions Button { width: 1fr; min-width: 10; height: 3; background: #142635; color: #54d9e8; border: tall #34525d; margin-right: 1; }
    #example-actions #example-approve { background: #fcee0a; color: #080c14; border: tall #fcee0a; }
    #example-actions #example-reject { color: #ff636c; border: tall #773d4b; }
    AgentHistoryScreen { align: center middle; background: #000000b0; }
    #history-dialog { width: 94%; height: 90%; padding: 0 1; border: heavy #54d9e8; background: #0b1521; }
    #history-heading { height: 2; color: #fcee0a; text-style: bold; }
    #history-help { height: auto; max-height: 3; color: #91a8bb; }
    #history-selectors { height: 3; }
    #history-picker { width: 1fr; }
    #history-reload { width: 10; min-width: 10; height: 3; margin-left: 1; color: #54d9e8; background: #142635; border: tall #34525d; }
    #history-content { height: 1fr; padding: 0 1; background: #0d1a26; border-left: solid #34525d; }
    #history-metadata, #history-preview { height: auto; margin-bottom: 1; }
    #history-metadata { color: #54d9e8; }
    #history-preview { color: #e4e9ef; }
    #history-status { height: auto; max-height: 3; color: #91a8bb; }
    #history-actions { height: 3; }
    #history-actions Button { width: 1fr; min-width: 10; height: 3; margin-right: 1; background: #142635; color: #54d9e8; border: tall #34525d; }
    #history-actions #history-open { background: #fcee0a; color: #080c14; border: tall #fcee0a; }
    .compact #sidebar { display: none; }
    .compact #masthead { display: none; }
    .compact #session { height: 3; padding: 0 1; }
    .compact .message { padding: 0 1; }
    .compact #thinking-label { width: 8; }
    .compact #thinking-picker { width: 13; }
    .compact #activity { height: 3; }
    .compact #view-nav { height: 3; }
    .compact #workspace-label { display: none; }
    .compact #agent-dialect { width: 20; }
    .compact #agent-stop { width: 26; padding: 0; }
    .compact #agent-continue { width: 29; padding: 0; }
    .compact #resume-mission { width: 12; margin-left: 0; }
    .compact #brain-body { layout: vertical; }
    .compact #brain-inspector { display: none; width: 1fr; height: 8; border-left: none; border-top: solid #ff515b; }
    .compact.brain-inspecting #brain-inspector { display: block; }
    .compact #brain-heading { height: 2; }
    .compact #view-nav Button { min-width: 10; margin-right: 0; }
    .compact #sleep-view { padding: 0 1; }
    .compact #sleep-actions { margin-top: 0; }
    .compact #sleep-actions #sleep-start { width: 26; }
    .compact #sleep-actions Button { min-width: 10; padding: 0; margin-right: 0; }
    #meditation-view { display: none; height: 1fr; background: #0b1521; padding: 0 2; }
    .meditation-view #meditation-view { display: block; }
    .meditation-view #messages, .meditation-view #thinking-controls, .meditation-view #agent-controls, .meditation-view #agent-options, .meditation-view #mission-status, .meditation-view #session, .meditation-view #input-row, .meditation-view #correction-row, .meditation-view #busy { display: none; }
    #meditation-heading { height: 2; content-align: left middle; color: #fcee0a; text-style: bold; border-bottom: solid #ff515b; }
    .meditation-actions { height: 3; margin-top: 1; }
    #meditation-config { height: 3; }
    #meditation-config Select { width: 1fr; }
    .meditation-actions Button { height: 3; width: 1fr; min-width: 9; margin-right: 1; background: #142635; color: #54d9e8; border: tall #34525d; }
    #meditation-start { background: #fcee0a; color: #080c14; text-style: bold; }
    #meditation-activate { color: #fcee0a; }
    #meditation-cancel { color: #ff636c; }
    .meditation-actions Button:disabled { background: #15212b; color: #667d8d; border: tall #283b47; }
    #meditation-scope { width: 1fr; }
    #meditation-details { height: 1fr; width: 1fr; padding: 1 0; }
    #meditation-intro, #meditation-run, #meditation-stage, #meditation-counters, #meditation-plan, #meditation-result, #meditation-active { height: auto; margin-bottom: 1; }
    #meditation-run, #meditation-counters { color: #54d9e8; }
    #meditation-counters { padding: 0 1; border-left: solid #54d9e8; background: #0e1d29; }
    #meditation-result { color: #91a8bb; }
    #meditation-stage-bar { width: 1fr; margin-bottom: 1; }
    .compact #meditation-view { padding: 0 1; }
    .compact .meditation-actions { margin-top: 0; }
    .compact .meditation-actions Button { padding: 0; margin-right: 0; }
    .compact #example-dialog { width: 100%; height: 100%; }
    .compact #example-help { max-height: 2; }
    .compact #example-correction { height: 4; }
    .compact #example-actions Button { padding: 0; }
    .compact #history-dialog { width: 100%; height: 100%; }
    .compact #history-actions Button { padding: 0; }
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000", api_token: str | None = None,
                 *, client: APIClient | None = None, initial_agent_run_id: str | None = None,
                 initial_view: str | None = None, crash_report_path: Path | str | None = None, **kwargs):
        if initial_view is not None and initial_view not in {"agent", "chat", "brain", "sleep", "meditation"}:
            raise ValueError("Choose agent, chat, brain, sleep or meditation as the initial view")
        super().__init__(**kwargs)
        self.api_url = api_url
        self.client = client or APIClient(api_url, api_token)
        self.initial_agent_run_id = initial_agent_run_id
        self.initial_view = initial_view
        self._initial_view_pending = initial_view is not None
        self.crash_report_path = Path(crash_report_path).expanduser() if crash_report_path is not None else None
        self.crash_report: dict | None = None
        self.conversation: dict | None = None
        self.last_message_id: str | None = None
        self.latest_train: str | None = None
        self.latest_sleep: str | None = None
        self.latest_meditation: str | None = None
        self.request_worker = None
        self.busy = False
        self._refreshing = False
        self._model_options = None
        self._model_profiles: list[str] = []
        self._adapter_options = None
        self.thinking_mode = "off"
        self.output_tokens_override: int | None = None
        self._current_model: dict | None = None
        self._bootstrapped = False
        self._structure_dirty = False
        self._revision: int | None = None
        self._models: list[dict] = []
        self._adapters: list[dict] = []
        self._status: dict = {}
        self._jobs: dict = {"train": [], "sleep": [], "meditation": []}
        self._base_candidates: list[dict] = []
        self._meditation_pending_job: dict | None = None
        self._meditation_error: str | None = None
        self._meditation_scope_options = (("All eligible training chats", "all"), ("Current chat only", "current"))
        self._gpu: dict = {}
        self._gpu_summary = ""
        self._last_connection_error: str | None = None
        self.capabilities: dict = {}
        self._panel_values: dict[str, Text] = {}
        self.input_mode = "agent"
        self.active_view = "agent"
        self._mode_chosen = False
        self.agent_workspace = str(Path.cwd())
        self.agent_access = "read"
        self.agent_dialect = "chat_completions"
        self.agent_auto_approve = False
        self.agent_stop_when_done = True
        self.agent_continue_output = True
        self.agent_images: list[str] = []
        self.agent_allow_images = False
        self.agent_mcp_servers: list[str] = []
        self.agent_run_id: str | None = None
        self._agent_streaming = False
        self.agent_runs: list[dict] = []
        self._agent_sequence = 0
        self._agent_target_segment = 0
        self._agent_replies: dict[int, AssistantMessage] = {}
        self._agent_finished_segments: set[int] = set()
        self._pending_agent_attachment: tuple[dict, AssistantMessage] | None = None
        self._agent_bound_scope: tuple | None = None
        self._agent_status: str | None = None
        self._agent_replaying = False
        self._agent_replay_approval_id: str | None = None
        self._agent_retargeting = False
        self._agent_retargeted = asyncio.Event()
        self._agent_retargeted.set()
        self._pending_approval: dict | None = None
        self._brain_signature = None
        self._main_screen = None
        self._sleep_pending_job: dict | None = None
        self._sleep_start_error: str | None = None
        self._example_records: dict[str, dict] = {}
        self._example_displayed_hashes: dict[str, str] = {}
        self._history_loading = False

    def compose(self) -> ComposeResult:
        yield Header(icon="◈")
        brand = Text()
        brand.append("L U C I F E R   /   NEURAL OPERATIONS", style="bold #080c14")
        brand.append("\nLOCAL INTELLIGENCE   /   YOUR SIGNAL. YOUR CONTROL.   //  01", style="#080c14")
        yield Static(brand, id="masthead")
        with Horizontal(id="workspace"):
            with VerticalScroll(id="sidebar"):
                portrait = Static(angel_portrait(), id="skull")
                portrait.border_title = " LCF // LOCAL ID "
                yield portrait
                yield Label("01 / CYBERWARE TELEMETRY", classes="panel-title")
                yield Static("Waiting for GPU telemetry…", id="gpus", markup=False)
                yield Label("02 / NEURAL CORE", classes="panel-title")
                yield Static("Connecting…", id="models", markup=False)
                yield Select([], prompt="Load Lucifer…", id="model-picker")
                yield Label("03 / ADAPTER SHARDS", classes="panel-title")
                yield Static("Lucifer · base", id="adapters", markup=False)
                yield Select([], prompt="Select adapter…", id="adapter-picker")
                yield Label("04 / SYSTEM LINK", classes="panel-title")
                yield Static("Backend: connecting", id="status", markup=False)
                yield Label("05 / CONSOLIDATION", classes="panel-title")
                yield Static("IDLE // no offline jobs", id="jobs", markup=False)
            with Vertical(id="chat-pane"):
                with Horizontal(id="view-nav"):
                    yield Button("AGENT / F2", id="view-agent", classes="selected")
                    yield Button("CORTEX / F3", id="view-brain")
                    yield Button("SLEEP / F4", id="view-sleep")
                    yield Button("HISTORY / F5", id="view-history")
                    yield Button("MEDITATE / F6", id="view-meditation")
                    yield Button("STOP", id="stop-mission", disabled=True)
                yield Static("CHANNEL: NEW  //  LOG: ON  //  TRAIN: OFF", id="session", markup=False)
                with Horizontal(id="thinking-controls"):
                    yield Label("THINKING", id="thinking-label")
                    yield Select([(mode.upper(), mode) for mode in THINKING_MODES], value=self.thinking_mode,
                                 allow_blank=False, id="thinking-picker")
                    yield Static("OUT 512 // CTX checking", id="generation-settings", markup=False)
                with Horizontal(id="agent-controls"):
                    yield Label("ROOT", id="workspace-label")
                    yield Input(value=self.agent_workspace, placeholder="Workspace root", id="agent-workspace")
                    yield Select([("READ ONLY", "read"), ("EDIT FILES", "write"), ("SANDBOX", "execute"), ("DIRECT PC", "computer")],
                                 value="read", allow_blank=False, id="agent-access")
                    yield Select([("CHAT COMPLETIONS", "chat_completions"), ("MESSAGES", "anthropic_messages"),
                                  ("RESPONSES", "responses")], value="chat_completions", allow_blank=False,
                                 id="agent-dialect")
                    yield Checkbox("AUTO CMDS", value=False, id="agent-auto")
                with Horizontal(id="agent-options"):
                    yield Checkbox("AUTO STOP ON GOAL", value=True, id="agent-stop")
                    yield Checkbox("AUTO CONTINUE OUTPUT", value=True, id="agent-continue")
                    yield Button("RESUME", id="resume-mission", disabled=True)
                    yield Static("NO DEADLINE", id="agent-limits", markup=False)
                yield Static("READY / Bounded workspace tools. Your scope controls access.", id="mission-status", markup=False)
                with VerticalScroll(id="messages"):
                    yield Message("system", "LUCIFER / LOCAL NEURAL LINK\nYour conversation and tools share this Agent workspace. HISTORY lists saved and retained sessions.\nF1 commands / F2 agent / F3 cortex / F4 sleep / F5 history\nNew conversations save automatically. Training stays OFF until you opt in. Use /privacy logging off for a private conversation. Provider thinking stays in session memory only.")
                with Vertical(id="brain-view"):
                    yield Static("ADAPTER CORTEX  /  EXPLORE YOUR NEURAL SHARDS\nRegistry topology. The brain surface is illustrative, not model activations.",
                                 id="brain-heading", markup=False)
                    with Horizontal(id="brain-body"):
                        yield AdapterBrain(id="brain-canvas")
                        with Vertical(id="brain-inspector"):
                            with VerticalScroll(id="brain-details-scroll"):
                                yield Markdown("## Neural core\n\nSelect a node to inspect its registry metadata.", id="brain-details")
                            yield Button("ACTIVATE ADAPTER", id="brain-select", disabled=True)
                            yield Button("DELETE EXPERT", id="brain-delete", disabled=True)
                    yield Static("Arrows/drag rotate · Q/E roll · +/- or wheel zoom · Shift+arrows pan\nN/P node · Enter inspect · R reset · F3 return · E expert / A adapter",
                                 id="brain-legend", markup=False)
                with Vertical(id="sleep-view"):
                    yield Static("SLEEP // LOCAL CONSOLIDATION", id="sleep-heading", markup=False)
                    with Horizontal(id="sleep-actions"):
                        yield Button("SLEEP / CONSOLIDATE", id="sleep-start")
                        yield Button("CANCEL JOB", id="sleep-cancel", disabled=True)
                        yield Button("RESULTS", id="sleep-review", disabled=True)
                        yield Button("EXAMPLES", id="sleep-review-examples")
                    with Horizontal(classes="training-body"):
                        with VerticalScroll(id="sleep-details"):
                            yield Static("Start once. Lucifer recovers consented task context, drafts and validates examples, reconciles memory, then trains and evaluates when enough verified data exists.\nUnverified proposals stay excluded. Verified preparation is automatic; activation remains explicit.", id="sleep-intro", markup=False)
                            yield Static("No sleep job selected", id="sleep-run", markup=False)
                            yield Static("Behavioral evaluation has not run yet.", id="sleep-outcomes", classes="training-outcomes", markup=False)
                            yield Static("COLLECT → DATA → TRAIN → VERIFY → EVALUATE → RESTORE → REVIEW", id="sleep-journey", markup=False)
                            yield Static("OVERALL // waiting to start", id="sleep-overall-label", markup=False)
                            yield SleepProgressBar(total=None, show_percentage=False, show_eta=False, id="sleep-overall")
                            yield Static("STAGE // waiting to start", id="sleep-stage", markup=False)
                            yield SleepProgressBar(total=None, show_percentage=False, show_eta=False, id="sleep-stage-bar")
                            yield Static("", id="sleep-stage-count", markup=False)
                            yield Static("", id="sleep-counters", markup=False)
                            yield Static("The compatible base, topics, sources and evaluation plan appear here when selected.", id="sleep-plan", markup=False)
                            yield Static("Behavioral and numerical gates decide qualification; activation remains explicit.", id="sleep-result", markup=False)
                            yield Static(LEGEND, id="sleep-legend-inline", classes="training-legend-inline", markup=False)
                        with VerticalScroll(classes="training-legend-pane"):
                            yield Static(LEGEND, id="sleep-legend", classes="training-legend", markup=False)
                with Vertical(id="meditation-view"):
                    yield Static("MEDITATION // PRIMARY MODEL / OUTPUT HEAD", id="meditation-heading", markup=False)
                    with Horizontal(id="meditation-config"):
                        yield Select([("SUPERVISED / assistant targets", "supervised"),
                                      ("UNSUPERVISED / source text", "unsupervised")],
                                     value="supervised", allow_blank=False, id="meditation-mode")
                        yield Select([("All eligible training chats", "all"), ("Current chat only", "current")],
                                     value="all", allow_blank=False, id="meditation-scope")
                        yield Select([("Standard segment", "standard"), ("Small pilot / 1 update", "1"),
                                      ("Pilot / 5 updates", "5"), ("Pilot / 10 updates", "10")],
                                     value="standard", allow_blank=False, id="meditation-pilot")
                    with Horizontal(classes="meditation-actions"):
                        yield Button("START", id="meditation-start")
                        yield Button("PAUSE", id="meditation-cancel", disabled=True)
                        yield Button("RESUME", id="meditation-resume", disabled=True)
                        yield Button("SOURCES", id="meditation-sources")
                    with Horizontal(classes="meditation-actions"):
                        yield Button("RESULTS", id="meditation-review", disabled=True)
                        yield Button("ACTIVATE", id="meditation-activate", disabled=True)
                        yield Button("ROLLBACK", id="meditation-rollback", disabled=True)
                    with Horizontal(classes="training-body"):
                        with VerticalScroll(id="meditation-details"):
                            yield Static("Updates 661.9M existing output-head weights. The backbone remains frozen. Choose supervised assistant targets or unsupervised source text, then START. Both use consented training chats and interacted documents. The first checkpoint passing automated evaluation and rollback checks is saved as a new base; activation is explicit. Source/result inspection is optional.", id="meditation-intro", markup=False)
                            yield Static("No Meditation run selected", id="meditation-run", markup=False)
                            yield Static("Behavioral evaluation has not run yet.", id="meditation-outcomes", classes="training-outcomes", markup=False)
                            yield Static("STAGE // waiting to start", id="meditation-stage", markup=False)
                            yield SleepProgressBar(total=None, show_percentage=False, show_eta=False, id="meditation-stage-bar")
                            yield Static("No global time or update limit. Finite segments continue until admission or pause.", id="meditation-counters", markup=False)
                            yield Static("Selected sources and measured evaluation appear here.", id="meditation-plan", markup=False)
                            yield Static("ACTIVE BASE // current model", id="meditation-active", markup=False)
                            yield Static("Training consent remains off until enabled for each chat.", id="meditation-result", markup=False)
                            yield Static(LEGEND, id="meditation-legend-inline", classes="training-legend-inline", markup=False)
                        with VerticalScroll(classes="training-legend-pane"):
                            yield Static(LEGEND, id="meditation-legend", classes="training-legend", markup=False)
                yield Static("", id="busy", markup=False)
                with Horizontal(id="correction-row"):
                    yield Input(placeholder="Correct the running mission; earlier thinking stays available", id="correction")
                    yield Button("CORRECT >>", id="correct-mission")
                with Horizontal(id="input-row"):
                    yield Input(placeholder=">_ Message or /help", id="prompt")
                    yield Button("SEND >>", id="send")
        yield RichLog(id="activity", markup=False, highlight=False, max_lines=200, wrap=True)
        yield Footer()

    async def on_mount(self) -> None:
        self._main_screen = self.screen
        self.update_sleep_action_labels()
        self.main_query("#prompt", Input).focus()
        self.main_query("#activity", RichLog).border_title = " EVENT FEED // LOCAL "
        self.activity("LOCAL LINK READY / F2 agent / F3 cortex / F4 sleep / F5 history / F1 commands")
        self.set_interval(1, self.refresh_panels)
        await self.refresh_panels()
        if self.initial_agent_run_id:
            if self._bootstrapped:
                self.call_after_refresh(self.submit, "/agent open " + shlex.quote(self.initial_agent_run_id))
            else:
                await self.message("system", "The backend is unavailable. Your requested mission has not been resumed or changed; use History to reopen it after reconnecting.")
                self._show_initial_view()
        else:
            self._show_initial_view()

    def _show_initial_view(self) -> None:
        if self._initial_view_pending:
            self._initial_view_pending = False
            self.set_view(self.initial_view)

    def _handle_exception(self, error: Exception) -> None:
        # Textual's default fatal traceback includes locals. Those locals can
        # contain actual provider reasoning, so keep the exit path metadata-only.
        self.crash_report = sanitized_exception(error)
        if self.crash_report_path is not None:
            temporary = None
            try:
                self.crash_report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.crash_report_path.parent,
                                                 prefix=".tui-crash-", suffix=".json", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(self.crash_report, handle, indent=2)
                temporary.replace(self.crash_report_path)
            except OSError:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        self._return_code = 1
        if self._exception is None:
            self._exception = error
            self._exception_event.set()
        self.panic(Text("Lucifer terminal stopped. Diagnostic code locations only:\n"
                        + json.dumps(self.crash_report, indent=2), style="#ff636c"))

    def main_query(self, selector, expect_type=None):
        screen = self._main_screen or self.screen
        return screen.query_one(selector, expect_type) if expect_type is not None else screen.query_one(selector)

    def on_resize(self, event) -> None:
        self.set_class(event.size.width < 100, "compact")
        if self._main_screen:
            self.update_sleep_action_labels(event.size.width)
        messages = self.query("#messages")
        if messages and messages.first().is_vertical_scroll_end:
            self.call_after_refresh(self.scroll_chat_end)

    def update_sleep_action_labels(self, width: int | None = None) -> None:
        compact = (self.size.width if width is None else width) < 100
        self.main_query("#sleep-review", Button).label = "CANDIDATE" if compact else "REVIEW CANDIDATE"
        self.main_query("#sleep-review-examples", Button).label = "EXAMPLES" if compact else "REVIEW EXAMPLES"

    def set_view(self, view: str, *, explicit: bool = True) -> None:
        if view not in {"chat", "agent", "brain", "sleep", "meditation"}:
            raise ValueError("Choose chat, agent, brain, sleep or meditation")
        if view in {"chat", "agent"}:
            self.input_mode = view
        if explicit:
            self._mode_chosen = True
        self.active_view = "agent" if view == "chat" else view
        self.set_class(self.input_mode == "agent", "agent-mode")
        self.update_generation_settings()
        self.set_class(view == "brain", "brain-view")
        self.set_class(view == "sleep", "sleep-view")
        self.set_class(view == "meditation", "meditation-view")
        self.remove_class("brain-inspecting")
        for name in ("agent", "brain", "sleep", "meditation"):
            self.main_query("#view-" + name, Button).set_class(self.active_view == name, "selected")
        self.update_readiness()
        self.main_query("#send", Button).label = "EXECUTE >>" if self.input_mode == "agent" else "SEND >>"
        if view == "brain":
            self.update_brain()
            self.main_query("#brain-canvas", AdapterBrain).focus()
        elif view == "sleep":
            self.main_query("#sleep-start", Button).focus()
        elif view == "meditation":
            self.main_query("#meditation-start", Button).focus()
        else:
            if self.screen is self._main_screen:
                self.main_query("#prompt", Input).focus()
            if not self.agent_run_id:
                self.update_mission_preferences()

    @on(Button.Pressed, "#view-agent")
    def view_agent(self) -> None:
        self.action_toggle_agent()

    @on(Button.Pressed, "#view-history")
    def view_history(self) -> None:
        self.action_history()

    @on(Button.Pressed, "#view-brain")
    def view_brain(self) -> None:
        self.action_toggle_brain()

    @on(Button.Pressed, "#view-sleep")
    def view_sleep(self) -> None:
        self.action_toggle_sleep()

    @on(Button.Pressed, "#view-meditation")
    def view_meditation(self) -> None:
        self.action_toggle_meditation()

    @on(Button.Pressed, "#meditation-start")
    def start_meditation_clicked(self) -> None:
        suffix = " --current" if self.main_query("#meditation-scope", Select).value == "current" else ""
        mode = str(self.main_query("#meditation-mode", Select).value)
        pilot = str(self.main_query("#meditation-pilot", Select).value)
        if pilot != "standard":
            suffix += " --pilot-updates " + pilot
        self.submit("/meditation start --mode " + mode + suffix)

    @on(Button.Pressed, "#meditation-cancel")
    def cancel_meditation_clicked(self) -> None:
        job = self.selected_meditation_job()
        if job:
            self.submit("/meditation cancel " + shlex.quote(job["id"]))

    @on(Button.Pressed, "#meditation-resume")
    def resume_meditation_clicked(self) -> None:
        job = self.selected_meditation_job()
        if job:
            self.submit("/meditation resume " + shlex.quote(job["id"]))

    @on(Button.Pressed, "#meditation-sources")
    def meditation_sources_clicked(self) -> None:
        self.submit("/meditation sources")

    @on(Button.Pressed, "#meditation-review")
    def meditation_review_clicked(self) -> None:
        self.submit("/meditation results")

    @on(Button.Pressed, "#meditation-activate")
    def meditation_activate_clicked(self) -> None:
        candidate = self.meditation_candidate(self.selected_meditation_job())
        if candidate:
            self.submit("/meditation activate " + shlex.quote(candidate))

    @on(Button.Pressed, "#meditation-rollback")
    def meditation_rollback_clicked(self) -> None:
        self.submit("/meditation rollback")

    @on(Button.Pressed, "#sleep-start")
    def start_sleep_clicked(self) -> None:
        self.set_view("sleep")
        self.submit("/sleep start")

    @on(Button.Pressed, "#sleep-review-examples")
    def review_examples_clicked(self) -> None:
        self.submit("/sleep examples pending")

    @on(Button.Pressed, "#sleep-cancel")
    def cancel_sleep_clicked(self) -> None:
        selected = self.selected_learning_job()
        if selected:
            kind, job = selected
            self.submit(f"/{kind} cancel {job['id']}")

    @on(Button.Pressed, "#sleep-review")
    def review_sleep_candidate(self) -> None:
        selected = self.selected_learning_job()
        candidate = self.learning_candidate(selected[1]) if selected else None
        brain = self.main_query("#brain-canvas", AdapterBrain)
        if candidate and brain.atlas.node("adapter:" + candidate):
            self.set_view("brain")
            self.call_after_refresh(brain.select_node, "adapter:" + candidate, reveal=True)
        else:
            self.activity("Candidate metadata is not available yet; waiting for the registry heartbeat.")

    @on(Button.Pressed, "#stop-mission")
    def stop_mission(self) -> None:
        self.action_cancel_request()

    @on(Select.Changed, "#agent-access")
    def access_selected(self, event: Select.Changed) -> None:
        if event.value in {"read", "write", "execute", "computer"}:
            self.agent_access = str(event.value)
            self.set_class(self.agent_access in {"execute", "computer"}, "command-access")
            label = "DIRECT PC / HOST ACCESS" if self.agent_access == "computer" else self.agent_access.upper()
            self.update_panel("#mission-status", Text(f"READY / {label} / {self.agent_workspace}", style="#ff515b" if self.agent_access == "computer" else "#54d9e8"))

    @on(Checkbox.Changed, "#agent-auto")
    def auto_commands_changed(self, event: Checkbox.Changed) -> None:
        self.agent_auto_approve = event.value

    @on(Checkbox.Changed, "#agent-stop")
    def auto_stop_changed(self, event: Checkbox.Changed) -> None:
        self.agent_stop_when_done = event.value
        self.update_mission_preferences()

    @on(Checkbox.Changed, "#agent-continue")
    def continue_output_changed(self, event: Checkbox.Changed) -> None:
        self.agent_continue_output = event.value
        self.update_mission_preferences()

    @on(Button.Pressed, "#resume-mission")
    def resume_mission(self) -> None:
        self.submit("/agent resume")

    @on(Input.Submitted, "#correction")
    def correction_submitted(self, event: Input.Submitted) -> None:
        self.queue_correction(event.value)

    @on(Button.Pressed, "#correct-mission")
    def correct_mission(self) -> None:
        self.queue_correction(self.main_query("#correction", Input).value)

    def update_mission_preferences(self, *, force: bool = False) -> None:
        if self.busy and not force:
            return
        if self.learning_busy:
            self.update_panel("#mission-status", Text(f"MISSION / WAITING / {self.learning_notice().upper()}", style="#fcee0a"))
            return
        if self.input_mode != "agent":
            self.update_panel("#mission-status", Text("CONVERSATION / Response mode; tools are unavailable or not selected for this model.", style="#54d9e8"))
            return
        access = "DIRECT PC / HOST ACCESS" if self.agent_access == "computer" else self.agent_access.upper()
        attachments = f" / IMAGES {len(self.agent_images)}" if self.agent_images else ""
        vision = " / VISION ON" if self.agent_allow_images else ""
        mcp = f" / MCP {','.join(self.agent_mcp_servers)}" if self.agent_mcp_servers else ""
        stopping = "AUTO STOP ON GOAL" if self.agent_stop_when_done else "WAIT FOR YOUR NEXT TURN"
        continuation = " / OUTPUT CONTINUATION ON" if self.agent_continue_output else ""
        self.update_panel("#mission-status", Text(f"READY / {access}{attachments}{vision}{mcp} / {stopping}{continuation}", style="#ff515b" if self.agent_access == "computer" else "#54d9e8"))

    @on(Select.Changed, "#agent-dialect")
    def dialect_selected(self, event: Select.Changed) -> None:
        if event.value in {"chat_completions", "anthropic_messages", "responses"}:
            self.agent_dialect = str(event.value)

    @on(Input.Changed, "#agent-workspace")
    def workspace_changed(self, event: Input.Changed) -> None:
        self.agent_workspace = event.value

    def update_brain(self) -> None:
        current = self._current_model
        active = self._status.get("active_adapter")
        if isinstance(active, dict):
            active = active.get("id")
        jobs = {kind: [{key: run.get(key) for key in ("id", "status", "candidate_id", "candidate_adapter_id", "total_examples", "replay_examples")}
                       for run in records] for kind, records in self._jobs.items()}
        signature = json.dumps([self._models, self._adapters, current, active, jobs], sort_keys=True, default=str)
        if signature == self._brain_signature:
            return
        brain = self.main_query("#brain-canvas", AdapterBrain)
        brain.set_data(self._models, self._adapters, current=current, active_adapter=active, jobs=jobs)
        self._brain_signature = signature
        count = sum(node.kind == "adapter" for node in brain.atlas.nodes)
        hidden = sum(adapter.get("compatible") is False for adapter in self._adapters)
        self.update_panel("#brain-heading", Text(
            f"ADAPTER CORTEX / {count} SHARDS / {len(brain.atlas.edges)} LINEAGE LINKS"
            + (f" / {hidden} INCOMPATIBLE HIDDEN" if hidden else "") + "\n"
            "Registry topology. The brain surface is illustrative, not model activations.", style="#fcee0a"
        ))

    def can_activate_adapter(self, node) -> bool:
        metrics = node.data.get("metrics") if node else None
        eligible = bool(node and (node.status == "archived" and isinstance(metrics, dict) and metrics.get("passed")
                        or (node.status == "admitted" or node.status == "archived" and node.data.get("admitted_at"))
                        and any(model.get('id') == node.data.get('base_model_id') for model in self._models)))
        return bool(node and node.kind == "adapter" and eligible
                    and node.data.get("compatible") is not False
                    and not node.data.get("source_revoked") and not node.data.get("source_invalidated")
                    and not self.busy and not self.learning_busy)

    @on(AdapterBrain.Selected)
    async def brain_selected(self, event: AdapterBrain.Selected) -> None:
        # A queued selection may arrive after the screen's children unmount.
        # It only paints local details, so there is nothing left to update.
        try:
            brain = self.main_query("#brain-canvas", AdapterBrain)
            details = self.main_query("#brain-details", Markdown)
            button = self.main_query("#brain-select", Button)
            delete_button = self.main_query('#brain-delete', Button)
        except NoMatches:
            return
        await details.update(node_details(event.node, brain.atlas))
        node = event.node
        button.disabled = not self.can_activate_adapter(node)
        switch = bool(node and node.data.get('admitted_at') and node.data.get('base_model_id') != (self._current_model or {}).get('id'))
        button.label = "ACTIVE ADAPTER" if node and node.status == "active" else "LOAD BASE & ACTIVATE" if switch else "ACTIVATE ADAPTER"
        delete_button.disabled = not self.can_delete_adapter(node)
        if self.active_view == "brain" and brain.has_focus:
            self.add_class("brain-inspecting")

    @on(Button.Pressed, "#brain-select")
    def brain_activate(self) -> None:
        node = self.main_query("#brain-canvas", AdapterBrain).selected()
        if node and node.kind == "adapter":
            action = 'activate' if node.data.get('admitted_at') else 'select'
            self.submit("/adapter " + action + " " + shlex.quote(str(node.data["id"])))

    def can_delete_adapter(self, node):
        active = self._status.get('active_adapter')
        if isinstance(active, dict):
            active = active.get('id')
        return bool(node and node.kind == 'adapter' and node.status != 'active'
                    and node.data.get('id') != active and not self.busy and not self.learning_busy)

    @on(Button.Pressed, '#brain-delete')
    def brain_delete(self):
        node = self.main_query('#brain-canvas', AdapterBrain).selected()
        if not self.can_delete_adapter(node):
            return
        ident = node.data['id']
        self.push_screen(DeleteExpertScreen(ident), lambda confirmed: self.submit('/adapter delete ' + shlex.quote(ident)) if confirmed else None)

    def scroll_chat_end(self) -> None:
        self.main_query("#messages", VerticalScroll).anchor()

    @on(Select.Changed, "#model-picker")
    def model_selected(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.submit(f"/model load {shlex.quote(str(event.value))}")

    @on(Select.Changed, "#adapter-picker")
    def adapter_selected(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.submit(f"/adapter select {shlex.quote(str(event.value))}")

    @on(Select.Changed, "#thinking-picker")
    def thinking_selected(self, event: Select.Changed) -> None:
        # Changed messages are queued. A prior selection must not overwrite a
        # newer visible value or feed another obsolete change back into Select.
        if event.value in THINKING_MODES and event.value == event.select.value:
            self.thinking_mode = str(event.value)
            self.update_generation_settings()

    def context_limits(self) -> tuple[int | None, int | None]:
        current = self._current_model or {}

        def positive(value):
            return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

        effective = positive(current.get("effective_context_length")) or positive(current.get("context_length"))
        native = positive(current.get("native_context_length"))
        return effective, native

    def output_limit(self) -> int | None:
        current = self._current_model or {}
        value = current.get("max_output_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        effective, _ = self.context_limits()
        return max(1, effective - CONTEXT_RESERVE) if effective else None

    def output_budget(self) -> int:
        if self.input_mode == "agent":
            default = {"off": 2048, "low": 4096, "medium": 8192, "high": 16384, "max": 32768}[self.thinking_mode]
        else:
            default = 512 if self.thinking_mode == "off" else 4096
        budget = self.output_tokens_override or default
        limit = self.output_limit()
        if limit:
            budget = min(budget, limit)
        effective, _ = self.context_limits()
        if effective:
            budget = min(budget, max(1, effective - CONTEXT_RESERVE))
        return budget

    def update_generation_settings(self) -> None:
        effective, native = self.context_limits()
        context = f"{effective:,} active" if effective else "checking"
        if native:
            context += f" / {native:,} native"
        limit = self.output_limit()
        ceiling = f" / MAX {limit:,}" if limit else ""
        self.update_panel("#generation-settings", Text(f"OUT {self.output_budget():,}{ceiling} // CTX {context}"))
        picker = self.main_query("#thinking-picker", Select)
        if picker.value != self.thinking_mode:
            picker.value = self.thinking_mode

    async def on_unmount(self) -> None:
        await self.client.close()

    def activity(self, content: str) -> None:
        line = Text(f">> {datetime.now():%H:%M:%S} ", style="#fcee0a")
        line.append(content, style="#ff747d" if content.startswith("Error:") else "#91a8bb")
        self.main_query("#activity", RichLog).write(line)

    def update_panel(self, selector: str, value: Text) -> None:
        if value != self._panel_values.get(selector):
            self.main_query(selector, Static).update(value)
            self._panel_values[selector] = value.copy()

    async def message(self, role: str, content: str = "", *, thinking_mode: str | None = None) -> Message | AssistantMessage:
        item = AssistantMessage(content, thinking_mode=thinking_mode) if role == "assistant" else Message(role, content)
        viewport = self.main_query("#messages", VerticalScroll)
        await viewport.mount(item)
        viewport.anchor()
        return item

    def update_session(self) -> None:
        conv = self.conversation if self.conversation is not None else {"logging_enabled": True}
        ident = conv.get("id", "New session")
        logging = "on" if conv.get("logging_enabled") else "off"
        training = "on" if conv.get("training_enabled") else "off"
        line = Text(f"{ident}  //  ", style="#8fa7ba")
        line.append(f"LOG: {logging.upper()}", style="#fcee0a")
        line.append("  //  ", style="#34525d")
        line.append(f"TRAIN: {training.upper()}", style="#ff515b")
        if conv.get("knowledge_count"):
            line.append(f"  //  KNOWLEDGE: {conv['knowledge_count']}", style="#54d9e8")
        if self._gpu_summary:
            line.append("\n" + self._gpu_summary, style="#8fa7ba")
        self.update_panel("#session", line)

    async def refresh_panels(self, *, structural: bool = False) -> None:
        self._structure_dirty = self._structure_dirty or structural
        if self._refreshing or not self.is_running:
            return
        self._refreshing = True
        try:
            if not self._bootstrapped or self._structure_dirty:
                snapshot = await self.client.bootstrap()
                if not self.is_running:
                    return
                self.apply_bootstrap(snapshot)
            else:
                snapshot = await self.client.telemetry()
                if not self.is_running:
                    return
                if snapshot.get("revision") != self._revision:
                    snapshot = await self.client.bootstrap()
                    if not self.is_running:
                        return
                    self.apply_bootstrap(snapshot)
                else:
                    self.apply_telemetry(snapshot)
            if self._last_connection_error is not None:
                self._last_connection_error = None
                self.update_sleep_view()
                self.update_meditation_view()
        except (APIError, KeyError, TypeError, ValueError) as exc:
            if not self.is_running:
                return
            self._bootstrapped = False
            self.update_panel("#status", Text(f"Backend unavailable\n{self.api_url}\nStart: lucifer serve"))
            self.render_gpu(connection_error=str(exc))
            if str(exc) != self._last_connection_error:
                self.activity(str(exc))
                self._last_connection_error = str(exc)
            self.update_sleep_view()
            self.update_meditation_view()
        finally:
            self._refreshing = False

    def apply_bootstrap(self, snapshot: dict) -> None:
        self._models = snapshot["models"]
        self._adapters = snapshot["adapters"]
        self._base_candidates = snapshot.get("base_candidates", self._base_candidates)
        self.capabilities = snapshot.get("capabilities", {})
        self._revision = snapshot.get("revision")
        self._structure_dirty = False
        self._bootstrapped = True
        self._current_model = snapshot.get("status", {}).get("model") or next(
            (model for model in self._models if model.get("status") in {"loaded", "serving"}), None)
        self.render_registry()
        self.apply_telemetry(snapshot)
        if not self._mode_chosen:
            tools = self.capabilities.get("capabilities", {}).get("tools", {})
            supported = tools.get("status") in {"advertised", "verified"}
            demo = self._current_model and self._current_model.get("backend") == "demo"
            self.set_view("agent" if supported and not demo else "chat", explicit=False)
        self.update_brain()

    def render_registry(self) -> None:
        models = self._models
        adapters = [adapter for adapter in self._adapters if adapter.get("compatible") is not False]
        self._model_profiles = [m["id"] for m in models]
        model_options = [(self.model_label(m), m["id"]) for m in models]
        adapter_options = [("Lucifer · base", "base")] + [
            (a["id"], a["id"]) for a in adapters if a.get("status") in ("active", "archived")
            or (a.get("status") == "admitted" and not a.get("source_revoked") and not a.get("source_invalidated")
                and self._current_model and a.get("base_model_id") == self._current_model.get("id"))
        ]
        for selector, options, previous in (
            ("#model-picker", model_options, self._model_options),
            ("#adapter-picker", adapter_options, self._adapter_options),
        ):
            if options != previous:
                picker = self.main_query(selector, Select)
                selected = picker.value
                with picker.prevent(Select.Changed):
                    picker.set_options(options)
                    if selected in {value for _, value in options}:
                        picker.value = selected
        self._model_options, self._adapter_options = model_options, adapter_options
        model_text = "\n".join(
            f"{'●' if m.get('status') in ('loaded', 'serving') else '○'} {self.model_label(m)}\n  {m.get('status', 'ready')}"
            for m in models
        )
        self.update_panel("#models", Text(model_text or "Lucifer is not configured"))
        adapter_text = "\n".join(f"{a['id']} · {a.get('status', 'candidate')}" for a in adapters)
        self.update_panel("#adapters", Text(adapter_text or "Lucifer · base"))

    @property
    def learning_busy(self) -> bool:
        return self._status.get("learning_busy") is True or self._status.get("meditation_busy") is True

    def learning_notice(self) -> str:
        active = self._status.get("jobs")
        if active is None:
            active = [run for records in self._jobs.values() for run in records
                      if run.get("status") not in TERMINAL_LEARNING_STATES]
        return "Offline learning in progress" if active else "Restoring inference"

    def selected_learning_job(self) -> tuple[str, dict] | None:
        records = [(kind, job) for kind, jobs in self._jobs.items() if kind != "meditation" for job in jobs]
        if self._sleep_pending_job:
            # A resume acknowledgment can arrive before the next heartbeat for
            # the same older record. Show the acknowledged state immediately.
            records = [(kind, job) for kind, job in records if job.get("id") != self._sleep_pending_job.get("id")]
            records.append(("sleep", self._sleep_pending_job))
        active = [(kind, job) for kind, job in records if job.get("status") not in TERMINAL_LEARNING_STATES]
        restoring = [(kind, job) for kind, job in records if self.learning_busy
                     and job.get("progress", {}).get("terminal") is False
                     and job.get("progress", {}).get("restored") is not True]
        selected = active or restoring or [(kind, job) for kind, job in records if job.get("id") == self.latest_sleep] or records
        return max(enumerate(selected), key=lambda pair: (str(pair[1][1].get("started_at", "")), pair[0]))[1] if selected else None

    def selected_meditation_job(self) -> dict:
        records = list(self._jobs.get("meditation", []))
        pending = self._meditation_pending_job
        if pending:
            records = [record for record in records if record.get("id") != pending.get("id")] + [pending]
        active = [record for record in records if record.get("status") not in TERMINAL_LEARNING_STATES]
        selected = active or [record for record in records if record.get("id") == self.latest_meditation] or records
        return max(enumerate(selected), key=lambda pair: (str(pair[1].get("started_at", pair[1].get("created_at", ""))), pair[0]))[1] if selected else {}

    @staticmethod
    def meditation_candidate(job: dict | None) -> str | None:
        job = job or {}
        return job.get("saved_base_id") or job.get("candidate_base_id") or job.get("base_revision_id")

    def selected_meditation_base(self) -> dict:
        ident = self.meditation_candidate(self.selected_meditation_job())
        return next((record for record in self._base_candidates if record.get("id") == ident), {})

    def update_meditation_view(self) -> None:
        bar = self.main_query("#meditation-stage-bar", SleepProgressBar)
        if self.active_view != "meditation" or self.screen is not self._main_screen:
            bar.set_pulsing(False)
            return
        job = self.selected_meditation_job()
        self.update_panel("#meditation-outcomes", Text("\n".join(outcome_lines(job))))
        progress = job.get("progress") or {}
        controller = job.get("policy") or job.get("controller") or job.get("adaptive") or {}
        if isinstance(controller.get("state"), dict):
            controller = {**controller, **controller["state"]}
        if isinstance(progress.get("policy"), dict):
            controller = {**progress["policy"], **controller}
        values = {**controller, **progress}
        if isinstance((job.get('development_metrics') or {}).get('loss'), (int, float)):
            values['development_loss'] = job['development_metrics']['loss']
        status = job.get("status", "idle")
        active = bool(job) and (status not in TERMINAL_LEARNING_STATES or self._status.get("meditation_busy") is True)
        request = job.get("request") or {}
        selected_ids = request.get("conversation_ids")
        scope = (f"{len(selected_ids)} selected training chat" + ("s" if len(selected_ids) != 1 else "")
                 if selected_ids is not None else "All eligible training chats")
        blocked = self.busy or self.learning_busy or bool(self._status.get("agent_busy") or self._status.get("chat_busy"))
        candidate = self.selected_meditation_base()
        report, receipt = candidate.get("admission_report") or {}, candidate.get("rollback_receipt") or {}
        admission = candidate.get("status") in {"admitted", "active", "archived"} and report.get("passed") is True and receipt.get("verified") is True
        self.main_query("#meditation-start", Button).disabled = blocked or active
        self.main_query("#meditation-cancel", Button).disabled = self.busy or not active
        self.main_query("#meditation-resume", Button).disabled = blocked or not job or not job.get("resumable")
        self.main_query("#meditation-sources", Button).disabled = self.busy
        self.main_query("#meditation-review", Button).disabled = self.busy or not bool(job)
        self.main_query("#meditation-activate", Button).disabled = blocked or not admission or candidate.get("status") == "active"
        self.main_query("#meditation-rollback", Button).disabled = blocked or not bool(
            self._status.get("previous_base_revision") or self._status.get("previous_base_id")
            or self._status.get("active_base_revision") or any(base.get("status") == "active" for base in self._base_candidates))
        scope_picker = self.main_query("#meditation-scope", Select)
        scope_picker.disabled = blocked or active
        scope_options = ((scope, "all"), (scope, "current")) if active else (
            ("All eligible training chats", "all"), ("Current chat only", "current"))
        if scope_options != self._meditation_scope_options:
            selected = scope_picker.value
            with scope_picker.prevent(Select.Changed):
                scope_picker.set_options(scope_options)
                scope_picker.value = selected
                # A label-only change keeps the same value; explicitly refresh
                # its rendered caption without emitting a user selection event.
                scope_picker.mutate_reactive(Select.value)
            self._meditation_scope_options = scope_options
        mode_picker = self.main_query("#meditation-mode", Select)
        mode_picker.disabled = blocked or active
        self.main_query("#meditation-pilot", Select).disabled = blocked or active
        run_mode = (job.get("request") or {}).get("mode")
        if active and run_mode in {"supervised", "unsupervised"} and mode_picker.value != run_mode:
            mode_picker.value = run_mode
        self.main_query("#meditation-intro", Static).display = not bool(job)
        connection = " // DISCONNECTED — LAST RECEIVED COUNTERS" if self._last_connection_error else ""
        self.update_panel("#meditation-run", Text(f"MEDITATION / {job.get('id', 'no run selected')} / {status.upper()}{connection}", style="#54d9e8"))
        stage = progress.get("stage") or status
        label = progress.get("label") or str(stage).replace("_", " ")
        self.update_panel("#meditation-stage", Text(f"STAGE // {str(stage).upper()} // {label}"))
        fraction = self.metric(progress, "fraction", "stage_fraction")
        completed, total = self.metric(progress, "completed"), self.metric(progress, "total")
        if fraction is None and completed is not None and total is not None and total > 0:
            fraction = completed / total
        if fraction is not None and not 0 <= fraction <= 1:
            fraction = None
        bar.display = active or fraction is not None
        bar_total, bar_progress = (100, fraction * 100) if fraction is not None else (None, 0)
        if (bar.total, bar.progress) != (bar_total, bar_progress):
            bar.update(total=bar_total, progress=bar_progress)
        bar.set_pulsing(active and self.screen is self._main_screen)
        counters = ["RUN // " + (f"pilot capped at {request['pilot_updates']} segment updates; calibration separate"
            if request.get('pilot_updates') else 'bounded segment, then outcome evaluation'
            if request.get('evaluation_protocol') == 'behavioral_v1' else 'no global time or optimizer-update limit')]
        retained = retained_updates(job)
        counters.append("CHECKPOINT UPDATES RETAINED // " + (str(retained) if retained is not None else "unknown"))
        for title, keys in (
            ("SUCCESSFUL UPDATES", ("successful_updates", "global_successful_updates", "optimizer_steps", "optimizer_step")),
            ("SEGMENT", ("segment", "segment_index", "round_index")),
            ("SEGMENT BUDGET", ("segment_updates", "segment_budget", "round_budget")),
            ("LEARNING RATE", ("learning_rate", "lr")),
            ("MICROBATCH", ("microbatch", "microbatch_size", "batch_size")),
            ("GRADIENT ACCUMULATION", ("gradient_accumulation", "accumulation")),
            ("DEVELOPMENT LOSS", ("development_loss", "dev_loss")),
            ("BEST DEVELOPMENT LOSS", ("best_development_loss",)),
            ("VALIDATION LOSS", ("validation_loss",)),
        ):
            if title == 'VALIDATION LOSS' and request.get('evaluation_protocol') == 'behavioral_v1':
                continue
            value = self.metric(values, *keys)
            if title == 'SEGMENT BUDGET' and request.get('pilot_updates') and value is not None:
                value = min(value, request['pilot_updates'])
            counters.append(title + " // " + (f"{value:g}" if value is not None else "unknown"))
        reason = (job.get("parameter_decision") or {}).get("reason") or values.get("decision_reason") or values.get("reason")
        if reason:
            counters.append("PARAMETER DECISION // " + str(reason))
        calibration = job.get("learning_rate_calibration") or {}
        if calibration:
            counters.append("CALIBRATION UPDATES // " + str(calibration.get("total_successful_updates", 0)) + " / disposable probes")
        if job.get("discarded_optimizer_updates"):
            counters.append("DISCARDED TRIAL UPDATES // " + str(job["discarded_optimizer_updates"]))
        self.update_panel("#meditation-counters", Text("\n".join(counters)))
        curriculum = job.get("dataset_metrics") or job.get("curriculum") or {}
        lines = ["MODE // " + str(request.get("mode", "supervised")).upper(), "SOURCES // " + scope,
                 "PRECISION // " + ({"native_f32": "F32 compute / BF16 saved", "bf16": "BF16 compute / BF16 saved"}.get(
                     job.get("training_precision", "bf16"), "unknown")),
                 "TRAINABLE // existing output head / 661,913,600 parameters",
                 "ADMITTED BASE // " + (job.get('saved_base_id') or "none")]
        drift = job.get("distribution_retention") or {}
        automatic = curriculum.get('automatic_numerical_validation') or {}
        if automatic:
            lines.append(f"AUTOMATIC NUMERICAL PROBES // development {automatic.get('development', 0)} / heldout {automatic.get('holdout', 0)}")
            lines.append("PROBE ORIGIN // trusted generators; separate from source chats and behavioral tasks")
        if drift:
            lines.append("DEVELOPMENT KL // " + str(drift.get("mean_forward_kl", "unknown"))
                         + " / " + ("PASS" if drift.get("passed") is True else "FAILED"))
        for key, aliases in (("train_examples", ("train_examples", "train")),
                             ("development_examples", ("development_examples", "development")),
                             ("heldout_examples", ("heldout_examples", "heldout")),
                             ("source_conversations", ("source_conversations",)),
                             ("source_groups", ("source_groups", "independent_groups")),
                             ("source_interactions", ("source_interactions",)), ("pending", ("pending",))):
            value = self.metric(curriculum, *aliases)
            if value is not None:
                lines.append(key.replace("_", " ").upper() + f" // {value:g}")
        update = job.get("weight_update_receipt") or candidate.get("weight_update_receipt") or {}
        changed = self.metric(update, "changed_elements", "changed_weight_count")
        if changed is not None:
            lines.append(f"CHANGED HEAD WEIGHTS // {changed:g}")
        for key in ("delta_l2", "delta_max_abs"):
            value = self.metric(update, key)
            if value is not None:
                lines.append(key.replace("_", " ").upper() + f" // {value:g}")
        lines.extend("ADMISSION // " + line for line in admission_gate_lines(report or job.get("admission_report")))
        run_admission = job.get("admission_report") or {}
        if run_admission.get("status") == "awaiting_evidence" and run_admission.get("required_groups"):
            lines.append(f"FRESH HELDOUT GROUPS // {run_admission.get('fresh_groups', 0)} / {run_admission['required_groups']}")
        if candidate:
            lines.append("CANDIDATE // " + candidate['id'] + ' / ' + str(candidate.get('status', 'unknown')))
            lines.append("ADMISSION CONTROL ROLLBACK // " + ("VERIFIED" if receipt.get("verified") is True else "not run / candidate not admitted"))
            if candidate.get("head_sha256"):
                lines.append("HEAD SHA-256 // " + str(candidate["head_sha256"]))
        native_rollback = job.get('native_update_receipt') or {}
        if native_rollback:
            lines.append("NATIVE HEAD ROLLBACK // " + ('VERIFIED' if native_rollback.get('rollback_probe_passed') is True
                and native_rollback.get('original_head_unchanged') is True else 'unknown'))
        lines.append("SERVING PROFILE RESTORED // " + ('YES' if progress.get('terminal') and progress.get('restored') is True
            else 'pending' if active else 'unknown'))
        self.update_panel("#meditation-plan", Text("\n".join(lines)))
        active_id = self._status.get("active_base_revision") or self._status.get("active_base_id")
        if not active_id:
            active_id = next((base["id"] for base in self._base_candidates if base.get("status") == "active"), None)
        self.update_panel("#meditation-active", Text("ACTIVE BASE // " + str(active_id or "original selected profile"), style="#fcee0a"))
        error = self._meditation_error or job.get("restoration_error") or job.get("error")
        if error:
            result = str(error)
        elif admission:
            result = "New base saved with evaluation and verified rollback. Activation remains explicit." if candidate.get("status") != "active" else "Saved base is active. ROLLBACK restores the previous model revision."
        elif active:
            result = "Searching for improvement in finite segments. The original profile is restored before admission completes."
        elif job:
            result = ("Pilot finished; candidate was not admitted. Original serving profile restored."
                if status == 'pilot_complete' and progress.get('restored') is True else
                job.get("note") or "Automated evaluation and source checks determine when the run can resume.")
        else:
            result = "Start with eligible training chats. Training consent remains unchanged; source documents are data, never instructions."
        self.update_panel("#meditation-result", Text(result, style="#ff636c" if error else "#91a8bb"))

    def cache_examples(self, records: list[dict]) -> None:
        for record in records:
            if record.get("id"):
                self._example_records[str(record["id"])] = dict(record)

    def record_displayed_example(self, record: dict) -> None:
        self.cache_examples([record])
        ident, content_hash = str(record["id"]), str(record.get("content_hash", ""))
        messages = example_messages(record)
        complete = all(any(message.get("role") == role and str(message.get("content", "")).strip()
                           for message in messages) for role in ("user", "assistant"))
        if complete and not record.get("_ui_stale") and record.get("status") != "obsolete" and all(message.get("role") in {"user", "assistant"} for message in messages) and re.fullmatch(r"[0-9a-fA-F]{64}", content_hash):
            self._example_displayed_hashes[ident] = content_hash
        else:
            self._example_displayed_hashes.pop(ident, None)

    async def submit_example_review(self, ident: str, decision: str, *, correction: str | None = None,
                                    reason: str | None = None, content_hash: str | None = None) -> dict:
        displayed_hash = self._example_displayed_hashes.get(ident)
        if not displayed_hash:
            raise ValueError("Display this example with /sleep examples and review its question and answer before deciding.")
        if content_hash is not None and content_hash != displayed_hash:
            raise ValueError("The displayed example changed. Reload and inspect its current content before deciding.")
        if decision not in {"approve", "reject"} or correction is not None and len(correction) > 100000 or reason is not None and len(reason) > 2000:
            raise ValueError("Review requires approve/reject, a complete correction up to 100,000 characters and a reason up to 2,000.")
        updated = await self.client.review_sleep_example(ident, decision=decision, content_hash=displayed_hash,
                                                         correction=correction, reason=reason)
        if str(updated.get("id")) != ident and ident in self._example_records:
            self._example_records[ident] = {**self._example_records[ident], "status": "obsolete", "superseded_by": updated.get("id")}
        self.cache_examples([updated])
        self._example_displayed_hashes.pop(ident, None)
        return updated

    async def open_example_review(self, status: str = "pending") -> None:
        if status not in {"pending", "verified", "rejected", "obsolete", "all"}:
            raise ValueError("Example status must be pending, verified, rejected, obsolete or all")
        records = await self.client.sleep_examples(None if status == "all" else status)
        self.cache_examples(records)
        await self.push_screen(ExampleReviewScreen(self, records, status=status))

    @staticmethod
    def learning_candidate(job: dict) -> str | None:
        return job.get("candidate_adapter_id") or job.get("candidate_id")

    def sleep_progress_bar(self, selector: str, fraction: float | None, *, visible: bool = True, pulsing: bool = False) -> None:
        bar = self.main_query(selector, SleepProgressBar)
        bar.display = visible
        total, progress = (100, fraction * 100) if fraction is not None else (None, 0)
        if (bar.total, bar.progress) != (total, progress):
            bar.update(total=total, progress=progress)
        bar.set_pulsing(visible and pulsing and self.active_view == "sleep")

    def update_sleep_view(self) -> None:
        if self.active_view != "sleep" or self.screen is not self._main_screen:
            # The heartbeat keeps job counters in _jobs; paint them on return.
            # Hidden widgets neither animate nor receive progress layout work.
            for selector in ("#sleep-overall", "#sleep-stage-bar"):
                self.main_query(selector, SleepProgressBar).set_pulsing(False)
            return
        selected = self.selected_learning_job()
        job = selected[1] if selected else {}
        self.update_panel("#sleep-outcomes", Text("\n".join(outcome_lines(job))))
        self.main_query("#sleep-intro", Static).display = not bool(job)
        progress = job.get("progress", {})
        adaptive = job.get("adaptive") or {}
        adaptive_job = bool(job.get("request", {}).get("schedule") or adaptive)
        stage = progress.get("stage") or job.get("status", "queued")
        terminal = progress.get("terminal") is True
        restored = progress.get("restored") is True
        finished = terminal and restored and not self.learning_busy
        active = bool(job) and (job.get("status") not in TERMINAL_LEARNING_STATES or self.learning_busy)
        blocked = self.busy or self.learning_busy or active or bool(self._status.get("agent_busy"))
        self.main_query("#sleep-start", Button).disabled = blocked
        self.main_query("#sleep-cancel", Button).disabled = self.busy or not active
        self.main_query("#sleep-review-examples", Button).disabled = self.busy
        candidate = self.learning_candidate(job)
        self.main_query("#sleep-review", Button).disabled = self.busy or not candidate or not finished
        if not job:
            waiting = self.learning_notice() if self.learning_busy else "No sleep job selected"
            self.update_panel("#sleep-run", Text(waiting, style="#54d9e8"))
            self.update_panel("#sleep-overall-label", Text("OVERALL // total work unknown" if self.learning_busy else "OVERALL // waiting to start", style="#fcee0a"))
            self.update_panel("#sleep-stage", Text("STAGE // waiting for job metadata" if self.learning_busy else "STAGE // waiting to start"))
            for selector in ("#sleep-overall", "#sleep-stage-bar"):
                self.sleep_progress_bar(selector, None, visible=self.learning_busy, pulsing=self.learning_busy)
            self.update_panel("#sleep-stage-count", Text())
            self.update_panel("#sleep-counters", Text())
            self.update_panel("#sleep-plan", Text("Lucifer will select compatible sources, a training base and evaluation cases."))
            result = self._sleep_start_error or ("An agent mission is active. Pause it before starting sleep." if self._status.get("agent_busy") else
                                                "Start once, then wait here. Your current model will be restored before completion.")
            self.update_panel("#sleep-result", Text(result, style="#ff636c" if self._sleep_start_error else "#91a8bb"))
            return
        connection = " // DISCONNECTED — LAST RECEIVED COUNTERS" if self._last_connection_error else ""
        displayed_status = job.get("status", "queued")
        if displayed_status == "queued" and not terminal and stage not in {"queued", "finished"}:
            displayed_status = stage
        self.update_panel("#sleep-run", Text(f"{selected[0].upper()} / {job.get('id', '?')} / {str(displayed_status).upper()}{connection}", style="#54d9e8"))
        journey = Text()
        for index, (label, stages) in enumerate(SLEEP_PHASES):
            if index:
                journey.append(" → ", style="#516a7b")
            journey.append(label, style="bold #fcee0a" if stage in stages else "#6e899b")
        self.update_panel("#sleep-journey", journey)
        overall = self.metric(progress, "overall_fraction")
        if overall is not None and (not 0 <= overall <= 1 or overall == 1 and not finished):
            overall = None
        admission_unfinished = adaptive_job and finished and job.get("status") not in {"admitted", "dry_run"}
        if admission_unfinished:
            overall = None
        elif finished:
            overall = 1.0
        summary = (f"OVERALL // {overall:.0%}" + (" // REVIEW REQUIRED" if job.get("status") in {"needs_review", "insufficient_verified_data"} else "") if overall is not None else
                   "OVERALL // in progress; total work unknown" if active else
                   "OVERALL // inference restoration needs attention" if terminal and not restored else
                   "OVERALL // progress details unavailable")
        if admission_unfinished:
            outcome = "PAUSED" if job.get("status") in {"cancelled", "interrupted", "paused"} else str(job.get("status", "stopped")).replace("_", " ").upper()
            summary = f"OVERALL // {outcome} // admission not reached"
        self.update_panel("#sleep-overall-label", Text(summary, style="#fcee0a"))
        self.sleep_progress_bar("#sleep-overall", overall, visible=overall is not None or active, pulsing=active)
        fraction = self.metric(progress, "fraction", "stage_fraction")
        if fraction is not None and not 0 <= fraction <= 1:
            fraction = None
        completed, total = self.metric(progress, "completed"), self.metric(progress, "total")
        if fraction is None and completed is not None and total is not None and total > 0:
            fraction = max(0, min(1, completed / total))
        label = progress.get("label") or str(stage).replace("_", " ").capitalize()
        if self.learning_busy and job.get("status") in TERMINAL_LEARNING_STATES and not terminal:
            label = progress.get("label") or "Restoring inference"
        self.update_panel("#sleep-stage", Text(f"STAGE // {str(stage).upper()} // {label}"))
        self.sleep_progress_bar("#sleep-stage-bar", fraction, visible=fraction is not None or active, pulsing=active)
        count = f"STAGE PROGRESS // {fraction:.0%}" if fraction is not None else "STAGE PROGRESS // total unknown"
        if completed is not None:
            count += f" // {completed:g}" + (f" / {total:g}" if total is not None else " completed")
        if progress.get("evaluating"):
            count += " // " + str(progress["evaluating"]).upper()
        self.update_panel("#sleep-stage-count", Text(count))
        self.main_query("#sleep-stage-count", Static).display = not terminal or completed is not None or fraction is not None
        counters = []
        plan = job.get("dynamic_plan", {})
        curriculum = job.get("curriculum") or {}
        early_curriculum = bool(curriculum) and (stage in {"queued", "collecting", "recovering_context", "drafting_examples", "validating_examples", "reconciling_memory", "filtering", "consolidating"}
            or finished and not candidate and job.get("status") in {"needs_review", "insufficient_verified_data", "dry_run", "no_new_information", "completed"})
        for title, done_key, total_key in (("OPTIMIZER STEPS", "optimizer_step", "optimizer_steps_total"),
                                          ("EVALUATION RESPONSES", "eval_completed", "eval_total")):
            if adaptive_job and done_key == "optimizer_step":
                continue
            done, maximum = self.metric(progress, done_key), self.metric(progress, total_key)
            if done_key == "eval_completed" and done is not None and stage not in {"evaluating", "assessing_development"}:
                title = "LAST " + title
            planned = False
            cases = self.metric(plan, "evaluation_cases")
            if total_key == "eval_total" and maximum is None and cases is not None and cases > 0 and not plan.get("preview_only"):
                maximum, planned = cases * 2, True
            if not early_curriculum or done is not None or maximum is not None:
                counters.append(f"{title} // " + (f"{done:g}" if done is not None else "unknown")
                                + (f" / {maximum:g}" + (" planned" if planned else "") if maximum is not None else " / total unknown"))
        if adaptive_job:
            saved = adaptive.get("state") or {}
            adaptive_progress = {**saved, **adaptive,
                "cumulative_optimizer_steps": saved.get("global_successful_updates"),
                "next_refresh_optimizer_step": saved.get("next_refresh_at"),
                "data_passes": saved.get("dataset_passes_completed"), **progress}
            for title, keys in (("ADAPTIVE EPOCH", ("adaptive_epoch",)),
                                ("ROUND UPDATE BUDGET", ("round_budget",)),
                                ("ROUND UPDATES LEFT", ("round_remaining",)),
                                ("TOTAL OPTIMIZER UPDATES", ("cumulative_optimizer_steps",)),
                                ("DATA PASSES", ("data_passes",)),
                                ("NEXT DATA REFRESH AT UPDATE", ("next_refresh_optimizer_step",))):
                value = self.metric(adaptive_progress, *keys)
                meaning = " // budget reset cycle" if title == "ADAPTIVE EPOCH" else ""
                counters.append(f"{title} // " + (f"{value:g}" if value is not None else "unknown") + meaning)
            checkpointed = self.metric(saved, "global_successful_updates")
            counters.append("CHECKPOINTED UPDATES // " + (f"{checkpointed:g}" if checkpointed is not None else "unknown"))
            round_index = self.metric(adaptive_progress, "round_index", "round")
            if round_index is not None:
                counters.append(f"ROUND INDEX // {round_index:g}")
            low, high = self.metric(adaptive_progress, "randint_min"), self.metric(adaptive_progress, "randint_max")
            bounds = [f"{value:g}" if value is not None else "unknown" for value in (low, high)]
            counters.append(f"ROUND ADJUSTMENT // 3 + RANDINT({bounds[0]}, {bounds[1]})")
        if curriculum:
            def amount(key):
                value = self.metric(curriculum, key)
                return f"{value:g}" if value is not None else "unknown"
            curriculum_rows = [
                f"VERIFIED // {amount('verified')} / MIN {amount('min_verified_examples')} // DRAFTED {amount('drafted')}",
                f"VERIFIED TASKS // {amount('verified_tasks')} / MIN {amount('min_verified_tasks')}",
                f"TRAIN // {amount('train_examples')} // HELD OUT {amount('heldout_examples')} // PENDING {amount('pending')}",
                f"REJECTED // {amount('rejected')} // DUPLICATES {amount('duplicates')} // OBSOLETE {amount('obsolete')}",
                f"SOURCE GROUPS // {amount('source_groups')} // TASKS {amount('source_tasks')} // CONVERSATIONS {amount('source_conversations')}",
            ]
            if early_curriculum:
                counters = curriculum_rows + counters
            else:
                counters.extend(curriculum_rows)
        metrics = job.get("dataset_metrics", {})
        reviewed = metrics.get("examples")
        dataset = f"REVIEWED EXAMPLES // {reviewed if reviewed is not None else 'selecting'}"
        if job.get("replay_examples") is not None:
            dataset += f" // REPLAY {job['replay_examples']}"
        counters.append(dataset)
        epoch, loss = self.metric(progress, "native_epoch", "epoch"), self.metric(progress, "loss")
        training_metrics = []
        if epoch is not None:
            training_metrics.append(f"TRAINER DATA EPOCH {epoch:g}")
        if loss is not None:
            training_metrics.append(f"LOSS {loss:.5g}")
        if training_metrics:
            counters.append(" // ".join(training_metrics))
        self.update_panel("#sleep-counters", Text("\n".join(counters)))
        expert, request = job.get("expert", {}), job.get("request", {})
        base = plan.get("base") or plan.get("base_model") or expert.get("training_base_model_id") or request.get("base_model")
        name = plan.get("expert_name") or expert.get("expert_name") or request.get("expert_name")
        topics = plan.get("topics") or plan.get("expert_topics") or expert.get("expert_topics") or request.get("expert_topics") or []
        parents = plan.get("source_expert_ids") or expert.get("source_expert_ids") or request.get("source_expert_ids") or []
        sources = job.get("source_conversation_ids") or []
        lines = [f"BASE // {base or ('not selected' if finished else 'selecting a compatible training profile')}",
                 f"{'EXPERT' if candidate else 'EXPERT PLAN'} // {name or candidate or ('no candidate created' if finished else 'discovering reviewed topics')}"]
        if topics:
            lines.append("TOPICS // " + ", ".join(str(topic) for topic in topics))
        lines.append(f"SOURCES // {len(sources)} conversations // {len(parents)} parent adapters" if "source_conversation_ids" in job else
                     "SOURCES // selecting reviewed conversations")
        evaluation_plan = job.get("evaluation_plan") or plan.get("evaluation") or {}
        if isinstance(evaluation_plan, dict):
            evaluation_label = evaluation_plan.get("label") or evaluation_plan.get("method") or evaluation_plan.get("suite")
            if evaluation_label:
                lines.append("EVALUATION // " + str(evaluation_label))
        if plan.get("evaluation_cases") is not None:
            lines.append(f"FROZEN EVALUATION // {plan['evaluation_cases']} cases // baseline + candidate")
        coverage = plan.get("coverage", {})
        if coverage.get("status"):
            lines.append("PROBE COVERAGE // " + str(coverage["status"]).upper())
            supported, source_count = coverage.get("supported_question_count"), coverage.get("source_question_count")
            if supported is not None and source_count is not None:
                lines.append(f"SUPPORTED SOURCE QUESTIONS // {supported} / {source_count}")
        if plan.get("preview_only"):
            lines.append("PREVIEW ONLY // no trained weights or measured evaluation")
        development = adaptive.get("development_assessment") or {}
        if "target_families" in development:
            targets = ", ".join(str(family) for family in development["target_families"]) or "none"
            lines.append("PRACTICE TARGETS // " + targets + " // independent development")
        refresh = adaptive.get("last_refresh") or {}
        refresh_step, new_verified = self.metric(refresh, "step"), self.metric(refresh, "new_verified_examples")
        if refresh_step is not None:
            lines.append(f"LAST DATA REFRESH AT UPDATE // {refresh_step:g}")
        if new_verified is not None:
            lines.append(f"NEW VERIFIED EXAMPLES // {new_verified:g}")
        admission = job.get("admission_report") or progress.get("admission_report") or {}
        if admission:
            evaluation_id = admission.get("evaluation_id") or job.get("evaluation_id")
            evaluated = next((adapter for adapter in self._adapters
                              if evaluation_id and adapter.get("evaluation_id") == evaluation_id), {})
            evaluated_id = admission.get("adapter_id") or evaluated.get("id")
            lines.append("ADMISSION CHECKPOINT // " + (evaluated_id or "unavailable") + " // last completed evaluation")
            if evaluation_id:
                lines.append("ADMISSION EVALUATION // " + str(evaluation_id))
        lines.extend("ADMISSION // " + line for line in admission_gate_lines(admission))
        self.update_panel("#sleep-plan", Text("\n".join(lines)))
        if self._sleep_start_error:
            result = self._sleep_start_error
        elif job.get("restoration_error"):
            result = "Could not restore inference: " + str(job["restoration_error"])
        elif job.get("error"):
            result = str(job["error"])
        elif finished:
            if job.get("status") == "admitted":
                result = (f"Adapter {candidate} admitted. Lucifer is restored. Select the adapter explicitly on its compatible profile."
                          if candidate else "Admission passed. Lucifer is restored; adapter activation remains explicit.")
            elif job.get("status") in {"awaiting_evidence", "unreachable_gate"}:
                result = job.get("note") or "Admission needs additional verified evidence. Inspect the failed gates before resuming."
            elif job.get("status") in {"interrupted", "paused", "blocked"} and job.get("request", {}).get("schedule"):
                result = job.get("note") or f"Adaptive Sleep stopped. Resume this checkpoint with /sleep resume {job['id']}."
            elif job.get("status") == "needs_review":
                result = "NEEDS REVIEW // " + (job.get("note") or "Open REVIEW EXAMPLES to approve, correct or reject pending targets, then start sleep again.")
            elif job.get("status") == "insufficient_verified_data":
                result = "INSUFFICIENT VERIFIED DATA // " + (job.get("note") or "Review pending examples or collect more consented task context before training.")
            elif job.get("status") == "no_new_information":
                result = job.get("note") or "No new eligible information was found."
            elif candidate and job.get("status") == "quality_unknown":
                result = f"Candidate {candidate} trained. Evaluation coverage is incomplete; review it in Cortex. Promotion remains gated."
            else:
                result = (f"Candidate {candidate}. Review its measured evaluation in Cortex; promotion remains manual." if candidate else
                          job.get("note") or label)
        elif terminal and restored and self.learning_busy:
            result = "Inference restored. Finishing pipeline cleanup."
        elif stage == "restoring" or self.learning_busy and job.get("status") in TERMINAL_LEARNING_STATES:
            result = "Restoring your inference model. The pipeline is still running."
        else:
            result = "Learning is running. Training, evaluation and restoration are managed together."
        color = "#ff636c" if job.get("status") in {"failed", "rejected", "cancelled", "quality_unknown", "needs_review", "insufficient_verified_data", "awaiting_evidence", "unreachable_gate"} or self._sleep_start_error else "#91a8bb"
        self.update_panel("#sleep-result", Text(result, style=color))

    def update_readiness(self) -> None:
        unavailable = self.busy or self.learning_busy
        for selector in ("#send", "#model-picker", "#adapter-picker"):
            self.main_query(selector).disabled = unavailable
        self.main_query("#resume-mission", Button).disabled = unavailable or not bool(self.agent_run_id)
        brain = self.main_query("#brain-canvas", AdapterBrain)
        self.main_query("#brain-select", Button).disabled = not self.can_activate_adapter(brain.selected())
        self.main_query("#brain-delete", Button).disabled = not self.can_delete_adapter(brain.selected())
        self.main_query("#prompt", Input).placeholder = (
            self.learning_notice() + " // commands remain available" if self.learning_busy else
            "Mission objective or /help" if self.input_mode == "agent" else "Message Lucifer or /help"
        )
        if not self.busy:
            text = f">> {self.learning_notice().upper()}_ // waiting for backend" if self.learning_busy else ""
            busy = self.main_query("#busy", Static)
            if busy.render().plain != text:
                busy.update(text)
        self.update_sleep_view()
        self.update_meditation_view()
        if isinstance(self.screen, AgentHistoryScreen):
            self.screen.controls()

    def apply_telemetry(self, snapshot: dict) -> None:
        was_learning_busy = self.learning_busy
        self._status = snapshot.get("status", self._status)
        jobs = snapshot.get("jobs", self._jobs)
        jobs_changed = jobs != self._jobs
        self._jobs = jobs
        if self._sleep_pending_job and any(job.get("id") == self._sleep_pending_job.get("id") for records in jobs.values() for job in records):
            self._sleep_pending_job = None
        if self._meditation_pending_job and any(job.get("id") == self._meditation_pending_job.get("id") for job in jobs.get("meditation", [])):
            self._meditation_pending_job = None
        self._gpu = snapshot.get("gpu", self._gpu)
        self.agent_runs = snapshot.get("agent_runs", self.agent_runs)
        current = self._status.get("model")
        self._current_model = current or next(
            (model for model in self._models if model.get("status") in ("loaded", "serving")), None
        )
        self.update_generation_settings()
        active = self._status.get("active_adapter") or "base"
        if isinstance(active, dict):
            active = active.get("id", "base")
        state = Text(self.learning_notice().upper() if self.learning_busy else "LUCIFER ONLINE" if current else "ONLINE",
                     style="bold #fcee0a")
        state.append(f"  //  Adapter: {active}\n{self.api_url}", style="#8fa7ba")
        self.update_panel("#status", state)
        trains, sleeps = self._jobs.get("train", []), self._jobs.get("sleep", [])
        jobs = [("Train", x) for x in trains[-3:]] + [("Sleep", x) for x in sleeps[-3:]]
        meditations = self._jobs.get("meditation", [])
        jobs.extend(("Meditation", job) for job in meditations[-3:])
        job_text = "\n".join(f"{kind.upper()} // {run.get('status', '?')}\n{run['id']}" for kind, run in jobs)
        if self.learning_busy:
            job_text += ("\n" if job_text else "") + self.learning_notice().upper()
        self.update_panel("#jobs", Text(job_text or "IDLE // no offline jobs"))
        if trains:
            self.latest_train = trains[-1]["id"]
        if sleeps:
            self.latest_sleep = sleeps[-1]["id"]
        if meditations:
            self.latest_meditation = meditations[-1]["id"]
        if jobs_changed:
            self.update_brain()
        self.update_readiness()
        if not self.busy and (self.learning_busy or was_learning_busy):
            self.update_mission_preferences()
        self.render_gpu()

    @staticmethod
    def metric(card: dict, *keys: str) -> float | None:
        for key in keys:
            value = card.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf"):
                return float(value)
        return None

    @staticmethod
    def meter(percent: float | None, width: int = 10) -> str:
        if percent is None or isinstance(percent, bool) or percent != percent or abs(percent) == float("inf"):
            return "·" * width
        filled = round(max(0, min(100, percent)) / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def render_gpu(self, *, connection_error: str | None = None) -> None:
        gpu = self._gpu
        sampled = gpu.get("sampled_at")
        stamp = "sample time unavailable"
        time_known = False
        if sampled:
            try:
                moment = datetime.fromtimestamp(sampled) if isinstance(sampled, (int, float)) else datetime.fromisoformat(str(sampled).replace("Z", "+00:00")).astimezone()
                stamp = moment.strftime("%H:%M:%S")
                time_known = True
            except (ValueError, TypeError, OverflowError):
                pass
        stale = bool(gpu.get("stale")) or connection_error is not None
        lines = Text()
        summaries = []
        for position, card in enumerate(gpu.get("gpus", [])):
            if position:
                lines.append("\n\n")
            index = card.get("index", card.get("id", position))
            lines.append(f"[{index}] ", style="bold #fcee0a")
            lines.append(str(card.get("name", "Unknown device")), style="#e4e9ef")
            utilization = self.metric(card, "utilization_pct")
            if utilization is not None and not 0 <= utilization <= 100:
                utilization = None
            gpu_percent = f"{utilization:.0f}%" if utilization is not None else "--%"
            lines.append(f"\nGPU  {gpu_percent} {self.meter(utilization)}", style="#fcee0a" if utilization is not None else "#78909e")
            total = self.metric(card, "total_mb", "memory_total_mb")
            used = self.metric(card, "memory_used_mb", "used_mb")
            free = self.metric(card, "memory_free_mb", "free_mb")
            if used is None and total is not None and total > 0 and free is not None and 0 <= free <= total:
                used = total - free
            memory = used / total * 100 if total is not None and total > 0 and used is not None and 0 <= used <= total else None
            memory_percent = f"{memory:.0f}%" if memory is not None else "--%"
            lines.append(f"\nVRAM {memory_percent} {self.meter(memory)}", style="#ff515b" if memory is not None else "#78909e")
            if memory is not None:
                lines.append(f"\n    {used:,.0f} / {total:,.0f} MiB", style="#9bb0bf")
            if utilization is None or memory is None:
                lines.append("\n--% = unavailable", style="#78909e")
            temperature = self.metric(card, "temp_c", "temperature_c")
            power = self.metric(card, "power_w")
            values = []
            if temperature is not None:
                values.append(f"{temperature:.0f}°C")
            if power is not None:
                values.append(f"{power:.0f} W")
            if values:
                lines.append("\n" + " // ".join(values), style="#9bb0bf")
            summaries.append(f"GPU {gpu_percent} {self.meter(utilization, 5)} // VRAM {memory_percent} {self.meter(memory, 5)}")
        if not lines.plain:
            lines.append("No GPU telemetry available\nCPU inference remains available", style="#78909e")
        status = "STALE" if stale else "LIVE" if time_known else "UNKNOWN"
        lines.append(f"\n{status} // {stamp}", style="#ff747d" if stale else "#8fa7ba")
        errors = [connection_error] if connection_error else gpu.get("errors", [])
        if errors:
            lines.append("\n" + "; ".join(str(error) for error in errors), style="#ff747d")
        self.update_panel("#gpus", lines)
        if summaries:
            self._gpu_summary = summaries[0] + f" // {status}"
        else:
            self._gpu_summary = "GPU telemetry unavailable" + (" // STALE" if stale else "")
        self.update_session()

    @on(Input.Submitted, "#prompt")
    def submitted(self, event: Input.Submitted) -> None:
        self.submit(event.value)

    @on(Button.Pressed, "#send")
    def send_clicked(self) -> None:
        self.submit(self.main_query("#prompt", Input).value)

    @staticmethod
    def needs_ready_inference(text: str) -> bool:
        if not text.startswith("/"):
            return True
        words = text.split(maxsplit=2)
        command, sub = words[0], words[1] if len(words) > 1 else ""
        return (command in {"/correct", "/resume", "/promote", "/rollback"}
                or command == "/agent" and sub in {"run", "resume", "attach"}
                or command == "/model" and sub in {"load", "unload", "register", "reload"}
                or command == "/adapter" and sub in {"select", "activate"}
                or command == "/eval" and sub == "run"
                or command in {"/train", "/sleep"} and sub == "start"
                or command == "/sleep" and sub == "resume"
                or command == "/meditation" and sub in {"start", "resume", "activate", "rollback"}
                or command == "/experts" and sub in {"build", "consolidate"}
                or command in {"/attach", "/link"}
                or command == "/knowledge" and sub == "remove")

    def submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.busy:
            if text in {"/agent cancel", "/agent stop", "/stop"}:
                self.action_cancel_request()
            elif text.startswith("/correct "):
                self.queue_correction(text[len("/correct "):])
            elif self.input_mode == "agent" and not text.startswith("/"):
                self.queue_correction(text)
            elif text.startswith(("/agent approve ", "/agent deny ")):
                try:
                    parts = shlex.split(text)
                except ValueError:
                    self.activity("Error: Usage: /agent approve|deny APPROVAL_ID; close any quoted argument.")
                    return
                if len(parts) == 3:
                    self.run_worker(self.resolve_agent_approval(parts[2], parts[1] == "approve"), group="agent-control")
            return
        if self.learning_busy and self.needs_ready_inference(text):
            self.activity(f"{self.learning_notice()}. Waiting for the backend; your draft is retained.")
            return
        self.main_query("#prompt", Input).value = ""
        self.busy = True
        self.update_readiness()
        self.request_worker = self.run_worker(self.handle_submission(text), group="request", exclusive=True)

    async def handle_submission(self, text: str) -> None:
        self.busy = True
        self.main_query("#send", Button).disabled = True
        self.main_query("#model-picker", Select).disabled = True
        self.main_query("#adapter-picker", Select).disabled = True
        self.main_query("#agent-workspace", Input).disabled = True
        self.main_query("#agent-access", Select).disabled = True
        self.main_query("#agent-dialect", Select).disabled = True
        self.main_query("#agent-auto", Checkbox).disabled = True
        self.main_query("#agent-stop", Checkbox).disabled = True
        self.main_query("#agent-continue", Checkbox).disabled = True
        self.main_query("#resume-mission", Button).disabled = True
        self.main_query("#stop-mission", Button).disabled = False
        self.main_query("#brain-select", Button).disabled = True
        self.main_query("#busy", Static).update(">> PROCESSING_  //  Esc to interrupt")
        try:
            if text.startswith("/"):
                await self.command(text)
            elif self.input_mode == "agent":
                await self.agent_goal(text)
            else:
                await self.chat(text)
        except asyncio.CancelledError:
            self.activity("Request interrupted. Partial replies are not eligible for feedback.")
            self.last_message_id = None
            raise
        except (APIError, ValueError, OSError, KeyError) as exc:
            self.activity(f"Error: {exc}")
            await self.message("system", f"Error: {exc}")
            if text.startswith("/sleep start"):
                self._sleep_start_error = str(exc)
                self.update_sleep_view()
            if text.startswith("/meditation"):
                self._meditation_error = str(exc)
                self.update_meditation_view()
        finally:
            self.busy = False
            self._agent_streaming = False
            self.remove_class("agent-running")
            self.main_query("#send", Button).disabled = False
            self.main_query("#model-picker", Select).disabled = False
            self.main_query("#adapter-picker", Select).disabled = False
            self.main_query("#agent-workspace", Input).disabled = False
            self.main_query("#agent-access", Select).disabled = False
            self.main_query("#agent-dialect", Select).disabled = False
            self.main_query("#agent-auto", Checkbox).disabled = False
            self.main_query("#agent-stop", Checkbox).disabled = False
            self.main_query("#agent-continue", Checkbox).disabled = False
            self.main_query("#resume-mission", Button).disabled = not bool(self.agent_run_id)
            self.main_query("#stop-mission", Button).disabled = True
            self.main_query("#busy", Static).update("")
            if self.screen is self._main_screen:
                self.main_query("#prompt", Input).focus()
            self.update_readiness()
            structural = text.split(maxsplit=1)[0] in {"/model", "/adapter", "/adapters", "/promote", "/rollback"}
            await self.refresh_panels(structural=structural)
            if self.initial_agent_run_id and text == "/agent open " + shlex.quote(self.initial_agent_run_id):
                self._show_initial_view()
            brain = self.main_query("#brain-canvas", AdapterBrain)
            brain.action_inspect()
            if self._pending_agent_attachment and not self._agent_retargeting:
                self.call_after_refresh(self.submit, "/agent attach")

    async def ensure_conversation(self) -> None:
        if self.conversation is None:
            self.conversation = await self.client.post("/conversations", {"logging_enabled": True, "training_enabled": False})
            self.update_session()

    async def chat(self, text: str) -> None:
        thinking, budget = self.thinking_mode, self.output_budget()
        self.main_query("#busy", Static).update(f">> PROCESSING_ // THINK {thinking.upper()} // Esc to interrupt")
        await self.ensure_conversation()
        await self.message("user", text)
        reply = await self.message("assistant", thinking_mode=thinking)
        assert isinstance(reply, AssistantMessage)
        done = {}
        completed = False
        succeeded = False
        self.last_message_id = None
        payload = {"conversation_id": self.conversation["id"], "messages": [{"role": "user", "content": text}],
                   "thinking": thinking, "max_tokens": budget}
        try:
            async for event, data in self.client.stream_chat(payload):
                if event == "reasoning":
                    fragment = data.get("content", "")
                    if isinstance(fragment, str) and thinking != "off":
                        reply.append_thinking(fragment)
                elif event == "delta":
                    fragment = data.get("content", "")
                    if isinstance(fragment, str):
                        reply.append_answer(fragment)
                elif event == "done":
                    completed, done = True, data
                elif event == "error":
                    raise APIError(data.get("detail", "Inference failed"))
            if not completed:
                raise APIError("Response stream ended before completion. Try again.")
            succeeded = True
            if reply.answer_buffer.source:
                self.last_message_id = done.get("message_id")
        finally:
            await reply.finish(partial=not succeeded)
        self.activity(f"Reply complete // THINK {thinking.upper()} // /feedback good | bad | correct <answer>")

    def agent_turn_options(self) -> dict:
        return {"thinking": self.thinking_mode, "max_tokens": self.output_budget(),
                "stop_when_done": self.agent_stop_when_done, "continue_output": self.agent_continue_output,
                "timeout_seconds": None, "max_rounds": None, "max_tool_calls": None, "max_total_tokens": None}

    def reset_agent_session(self) -> None:
        self.agent_run_id = None
        self._agent_sequence = 0
        self._agent_target_segment = 0
        self._agent_replies = {}
        self._agent_finished_segments = set()
        self._agent_bound_scope = None
        self._agent_status = None
        self._agent_replay_approval_id = None
        self._pending_agent_attachment = None
        self._agent_retargeting = False
        self._agent_retargeted.set()
        self.main_query("#resume-mission", Button).disabled = True

    async def open_agent_session(self, ident: str) -> None:
        session = await self.client.agent_session(ident)
        run = session["run"]
        retained = run.get("session_available")
        if retained is None:
            retained = run.get("resumable") is not False and run.get("content_retained") is not False
        checkpoint = run.get("checkpoint_available") is True
        self.reset_agent_session()
        self.last_message_id = None
        self.agent_run_id = ident if retained or checkpoint else None
        self._agent_status = run.get("status")
        self.agent_workspace = run.get("workspace", self.agent_workspace)
        self.main_query("#agent-workspace", Input).value = self.agent_workspace
        self.agent_access = run.get("access", "read")
        self.main_query("#agent-access", Select).value = self.agent_access
        self.agent_dialect = run.get("dialect", "chat_completions")
        self.main_query("#agent-dialect", Select).value = self.agent_dialect
        self.agent_auto_approve = bool(run.get("auto_approve_commands", False))
        self.main_query("#agent-auto", Checkbox).value = self.agent_auto_approve
        self.agent_stop_when_done = bool(run.get("stop_when_done", True))
        self.main_query("#agent-stop", Checkbox).value = self.agent_stop_when_done
        self.agent_continue_output = bool(run.get("continue_output", True))
        self.main_query("#agent-continue", Checkbox).value = self.agent_continue_output
        self.agent_allow_images = bool(run.get("allow_images", False))
        self.agent_images = list(run.get("images", []))
        self.agent_mcp_servers = list(run.get("mcp_servers", []))
        self.thinking_mode = run.get("thinking", self.thinking_mode)
        if isinstance(run.get("max_tokens"), int):
            self.output_tokens_override = run["max_tokens"]
        self.conversation = session.get("conversation")
        if not self.conversation and run.get("conversation_id"):
            self.conversation = await self.client.get(f"/conversations/{run['conversation_id']}")
        self.update_session()
        self._agent_bound_scope = self.agent_scope()
        self._agent_target_segment = int(run.get("segment", 0))
        self.set_view("agent")
        if ident == self.initial_agent_run_id:
            self._show_initial_view()
        await self.main_query("#messages", VerticalScroll).remove_children()
        if isinstance(run.get("control_notice"), str) and run["control_notice"]:
            await self.message("system", "CONTROL NOTICE // " + run["control_notice"])
        if not retained:
            public = [message for message in session.get("messages", []) if isinstance(message, dict)
                      and message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)]
            for message in public:
                item = await self.message(message["role"], message["content"])
                if isinstance(item, AssistantMessage):
                    partial = bool(message.get("partial")) or message.get("status") == "partial"
                    await item.finish(partial=partial)
                    item.thinking_panel.title = "Model thinking · not retained"
                    if not partial:
                        self.last_message_id = message.get("id")
            self._agent_sequence = int(run.get("last_sequence", 0))
            if checkpoint:
                availability = ("RESUME explicitly continues the recorded mission." if run.get("restore_available") else
                                "Restoration is currently unavailable: " + str(run.get("restore_unavailable_reason") or "check the loaded model and recorded scope"))
                notice = f"Saved mission {ident} loaded. {availability} Provider thinking was session memory and was not saved."
            elif public:
                notice = f"Saved transcript for {ident} loaded. Its runtime session is unavailable; your next message starts a new mission. Provider thinking was not saved."
            else:
                reason = session.get("transcript_unavailable_reason") or run.get("transcript_unavailable_reason") or "no_saved_transcript"
                notice = (f"Mission {ident} is listed, but its message text is unavailable ({str(reason).replace('_', ' ')}). "
                          "Private text cannot be reconstructed from operation receipts. New conversations now save by default; training consent stays separate.")
            await self.message("system", notice)
            self.update_panel("#mission-status", Text("HISTORY / " + ("SAVED CHECKPOINT / EXPLICIT RESUME REQUIRED" if checkpoint else "TRANSCRIPT ONLY" if public else "PRIVATE CONTENT UNAVAILABLE"), style="#fcee0a"))
            return
        await self.message("system", f"Reopening mission {ident}. Reasoning is replayed from backend session memory.")
        if run.get("goal"):
            await self.message("user", run["goal"])
        self._agent_replaying = True
        self._agent_replay_approval_id = run.get("pending_approval") or (run.get("approval") or {}).get("approval_id")
        self._agent_streaming = True
        self.add_class("agent-running")
        try:
            await self.stream_agent_session()
        finally:
            self._agent_replaying = False
            self._agent_replay_approval_id = None

    def agent_scope(self, payload: dict | None = None) -> tuple:
        value = payload or {"workspace": self.agent_workspace.strip(), "access": self.agent_access,
                            "dialect": self.agent_dialect, "auto_approve_commands": self.agent_auto_approve,
                            "allow_images": self.agent_allow_images, "images": self.agent_images,
                            "mcp_servers": self.agent_mcp_servers}
        return (str(Path(value["workspace"]).expanduser()), value["access"], value["dialect"],
                bool(value.get("auto_approve_commands")) and value["access"] in {"execute", "computer"}, bool(value.get("allow_images")),
                tuple(value.get("mcp_servers", [])), tuple(value.get("images", [])))

    async def agent_goal(self, goal: str, *, options: dict | None = None, resume: bool = False,
                         attachment: tuple[dict, AssistantMessage] | None = None) -> None:
        """Continue one mission across user turns; only an explicit reset loses its identity."""
        if not str((options or {}).get("workspace", self.agent_workspace)).strip():
            raise ValueError("Set a workspace root before starting an agent mission.")
        self.set_view("agent")
        await self.ensure_conversation()
        if attachment:
            run, reply = attachment
        else:
            if goal:
                await self.message("user", goal)
            elif resume:
                await self.message("system", "Resuming this mission from its available context.")
            reply = await self.message("assistant", thinking_mode=self.thinking_mode)
            assert isinstance(reply, AssistantMessage)
            self.last_message_id = None
            payload = {"goal": goal, "workspace": self.agent_workspace.strip(), "access": self.agent_access,
                       "dialect": self.agent_dialect, "conversation_id": self.conversation["id"],
                       "auto_approve_commands": self.agent_auto_approve and self.agent_access in {"execute", "computer"},
                       "images": list(self.agent_images), "allow_images": self.agent_allow_images,
                       "allow_mcp": bool(self.agent_mcp_servers), "mcp_servers": list(self.agent_mcp_servers),
                       **self.agent_turn_options()}
            payload.update(options or {})
            same_scope = self.agent_run_id is not None and self.agent_scope(payload) == self._agent_bound_scope
            if resume and not self.agent_run_id:
                await reply.finish(partial=True)
                raise ValueError("There is no mission to resume")
            if resume and not same_scope:
                await reply.finish(partial=True)
                raise ValueError("The visible scope differs from this mission. Restore it or send a new goal under the selected scope.")
            if same_scope or resume:
                creation = asyncio.create_task(
                    self.client.resume_agent(self.agent_run_id, instruction=goal or None, limits=self.agent_turn_options())
                    if resume else self.client.agent_input(self.agent_run_id, goal, options=self.agent_turn_options())
                )
            else:
                self._agent_sequence = 0
                self._agent_replies = {}
                self._agent_finished_segments = set()
                creation = asyncio.create_task(self.client.start_agent(payload))
            try:
                run = await asyncio.shield(creation)
            except asyncio.CancelledError:
                run = await creation
                await self.client.stop_agent(run["id"])
                await reply.finish(partial=True, paused=True)
                raise
            except (APIError, ValueError, OSError):
                await reply.finish(partial=True)
                raise
            self._agent_bound_scope = self.agent_scope(payload)
        self.agent_run_id = run["id"]
        self._agent_status = run.get("status", "running")
        segment = int(run.get("segment", self._agent_target_segment + 1 if self._agent_replies else 0))
        self._agent_target_segment = segment
        self._agent_replies[segment] = reply
        reply.border_title = f" LUCIFER / MISSION TURN {run.get('session_turn', len(self._agent_replies))} "
        self._agent_streaming = True
        self.add_class("agent-running")
        self.last_message_id = None
        self.activity(f"MISSION {self.agent_run_id} / {self.agent_access.upper()} / NO DEADLINE")
        await self.stream_agent_session()

    async def finish_agent_segment(self, segment: int, *, partial: bool, paused: bool = False) -> None:
        reply = self._agent_replies.get(segment)
        if reply is not None and segment not in self._agent_finished_segments:
            await reply.finish(partial=partial, paused=paused)
            self._agent_finished_segments.add(segment)

    async def stream_agent_session(self) -> None:
        complete, succeeded, paused = False, False, False
        ending_segment = self._agent_target_segment
        retries = 0
        try:
            while not complete:
                attached_target = self._agent_target_segment
                try:
                    async with aclosing(self.client.stream_agent(self.agent_run_id, after=self._agent_sequence)) as events:
                        async for event, data in events:
                            event_segment = int(data.get("segment", self._agent_target_segment))
                            if self._agent_retargeting and (event == "done" or event_segment > self._agent_target_segment):
                                await self._agent_retargeted.wait()
                            sequence = data.get("sequence")
                            if isinstance(sequence, int):
                                if sequence <= self._agent_sequence:
                                    continue
                                self._agent_sequence = sequence
                            ending_segment = event_segment
                            if self._agent_replaying and event in {"input", "correction"} and data.get("content"):
                                await self.message("user", str(data["content"]))
                            reply = self._agent_replies.get(event_segment)
                            if reply is None:
                                if event_segment != self._agent_target_segment and not self._agent_replaying:
                                    continue
                                reply = await self.message("assistant", thinking_mode=None if self._agent_replaying else self.thinking_mode)
                                assert isinstance(reply, AssistantMessage)
                                reply.border_title = f" LUCIFER / RETAINED TURN {data.get('session_turn', event_segment)} "
                                self._agent_replies[event_segment] = reply
                            if event_segment >= self._agent_target_segment:
                                self._pending_agent_attachment = None
                            if event == "reasoning":
                                if reply.thinking_mode != "off":
                                    reply.append_thinking(str(data.get("content", "")))
                            elif event == "delta":
                                reply.append_answer(str(data.get("content", "")))
                            elif event in {"phase", "input", "correction"}:
                                phase = data.get("phase", "correction applied" if event == "correction" else "running")
                                status = f"MISSION / {phase.upper()} / NO DEADLINE"
                                if data.get("round") is not None:
                                    status += f" / ROUND {data['round']}"
                                self.update_panel("#mission-status", Text(status, style="#54d9e8"))
                            elif event in {"tool_requested", "tool_started", "tool_result"}:
                                reply.append_tool(event, data)
                                self.activity(f"{event.upper()} / {data.get('name', data.get('tool', 'tool'))}")
                            elif event == "approval_required":
                                reply.append_tool(event, data)
                                approval_id = data.get("approval_id", data.get("id"))
                                if not self._agent_replaying or self._agent_replay_approval_id is not None and approval_id == self._agent_replay_approval_id:
                                    self.show_agent_approval(data)
                            elif event == "usage":
                                total = data.get("total_tokens")
                                if total is not None:
                                    self.update_panel("#mission-status", Text(f"MISSION / {total:,} tokens reported / NO DEADLINE", style="#54d9e8"))
                            elif event == "context_compacted":
                                removed = data.get("removed_messages", 0)
                                after = data.get("after_tokens", 0)
                                self.activity(f"CONTEXT COMPACTED / {removed} messages / {after:,} tokens")
                                self.update_panel("#mission-status", Text(f"MISSION / CONTEXT COMPACTED / {after:,} TOKENS", style="#54d9e8"))
                            elif event == "progress_warning":
                                notice = data.get("message") or "Repeated unchanged work detected; Lucifer is being steered toward a new step."
                                self.activity("PROGRESS CHECK / " + notice)
                                self.update_panel("#mission-status", Text("MISSION / CHECKING REPEATED WORK", style="#fcee0a"))
                            elif event == "final":
                                content = data.get("content", "")
                                if content and content != reply.answer_buffer.source:
                                    await reply.answer_buffer.finish()
                                    reply.answer_buffer.source = content
                                    await reply.answer.update(content)
                                self.last_message_id = data.get("message_id")
                                if data.get("conversation_id"):
                                    self.conversation["id"] = data["conversation_id"]
                                if data.get("artifacts"):
                                    reply.append_tool("artifacts", {"name": "workspace outputs", "summary": ", ".join(str(a) for a in data["artifacts"])})
                            elif event == "error":
                                if event_segment < self._agent_target_segment:
                                    continue
                                self._agent_status = "failed"
                                complete = True
                                raise APIError(data.get("detail", data.get("error", "Agent mission failed")))
                            elif event in {"paused", "cancelled"}:
                                if event_segment == self._agent_target_segment:
                                    paused = True
                                    if data.get("detail"):
                                        self.activity("MISSION PAUSED / " + data["detail"])
                                        await self.message("system", data["detail"])
                            elif event == "done":
                                status = data.get("status", "finished")
                                segment_success = status in {"completed", "succeeded", "awaiting_input"}
                                await self.finish_agent_segment(event_segment, partial=not segment_success,
                                                                paused=status in {"paused", "cancelled", "interrupted"})
                                if event_segment < self._agent_target_segment:
                                    continue
                                complete, succeeded = True, segment_success
                                paused = status in {"paused", "cancelled", "interrupted"}
                                self._agent_status = status
                                label = "AWAITING YOUR NEXT TURN" if status == "awaiting_input" else status.upper()
                                self.update_panel("#mission-status", Text(f"MISSION / {label} / THINKING RETAINED IN SESSION", style="#fcee0a" if succeeded else "#54d9e8"))
                    if not complete:
                        if self._agent_retargeting:
                            await self._agent_retargeted.wait()
                        if attached_target < self._agent_target_segment:
                            continue
                        raise APIError("Mission event stream disconnected")
                except APIError:
                    if complete or retries >= 2:
                        raise
                    retries += 1
                    self.activity(f"MISSION LINK / reconnecting from event {self._agent_sequence}")
                    await asyncio.sleep(.25 * retries)
            if not succeeded:
                self.last_message_id = None
        finally:
            await self.finish_agent_segment(ending_segment, partial=not succeeded, paused=paused)
            if self._pending_approval:
                self._pending_approval = None
                if isinstance(self.screen, ApprovalScreen):
                    self.screen.dismiss(False)
        self.activity("MISSION READY FOR NEXT TURN" if succeeded else "MISSION PAUSED / context and thinking retained")

    def queue_correction(self, content: str) -> None:
        content = content.strip()
        if not content:
            return
        if not self.agent_run_id:
            self.activity("Start a mission before sending a correction")
            return
        if self._agent_retargeting:
            self.activity("A correction is already being applied; your draft remains in the input")
            return
        self.main_query("#correction", Input).value = ""
        if not self._agent_streaming:
            self.submit("/correct " + content)
            return
        self._agent_retargeting = True
        self._agent_retargeted.clear()
        self.run_worker(self.correct_agent(content), group="agent-control")

    async def correct_agent(self, content: str) -> None:
        try:
            run = await self.client.agent_input(self.agent_run_id, content, interrupt=True, options=self.agent_turn_options())
            await self.message("user", "Correction: " + content)
            reply = await self.message("assistant", thinking_mode=self.thinking_mode)
            assert isinstance(reply, AssistantMessage)
            segment = int(run.get("segment", self._agent_target_segment + 1))
            self._agent_target_segment = segment
            self._agent_replies[segment] = reply
            reply.border_title = f" LUCIFER / CORRECTED TURN {run.get('session_turn', len(self._agent_replies))} "
            self._pending_agent_attachment = (run, reply)
            self._agent_status = run.get("status", "running")
            self.activity("CORRECTION APPLIED / same mission, prior thinking retained")
        except (APIError, ValueError, KeyError) as exc:
            self.activity(f"Error: {exc}")
            await self.message("system", f"Correction could not be applied: {exc}")
        finally:
            self._agent_retargeting = False
            self._agent_retargeted.set()
            if not self._agent_streaming and self._pending_agent_attachment:
                self.call_after_refresh(self.submit, "/agent attach")

    def show_agent_approval(self, request: dict) -> None:
        self._pending_approval = request
        approval_id = request.get("approval_id", request.get("id"))
        if not approval_id:
            raise APIError("Agent supplied an approval without an ID")
        self.update_panel("#mission-status", Text(f"AWAITING YOUR DECISION / {approval_id}", style="#fcee0a"))
        def decision(approved: bool | None):
            if self._pending_approval is not None:
                self.run_worker(self.resolve_agent_approval(str(approval_id), bool(approved)), group="agent-control")
        self.push_screen(ApprovalScreen(request), decision)

    async def resolve_agent_approval(self, approval_id: str, approved: bool) -> None:
        if not self.agent_run_id:
            raise ValueError("No agent run is selected")
        try:
            await self.client.post(f"/agent/runs/{self.agent_run_id}/approve", {"approval_id": approval_id, "approved": approved})
            self.activity(f"OPERATION {'APPROVED' if approved else 'DENIED'} / {approval_id}")
            self._pending_approval = None
        except APIError as exc:
            self.activity(f"Error: {exc}")

    async def cancel_agent(self) -> None:
        if self.agent_run_id:
            try:
                await self.client.stop_agent(self.agent_run_id)
            except APIError as exc:
                self.activity(f"Error: {exc}")
                if self.request_worker and self.busy:
                    self.request_worker.cancel()

    async def show_result(self, data) -> None:
        public = self.public_result(data)
        await self.message("system", json.dumps(public, indent=2, ensure_ascii=False) if not isinstance(public, str) else public)

    @staticmethod
    def public_result(data):
        """Present application records without changing their stored provenance."""
        if isinstance(data, list):
            return [LuciferApp.public_result(item) for item in data]
        if not isinstance(data, dict):
            return data
        hidden = {"spec", "model_spec", "path", "model_path", "dataset_path", "output_dir", "request", "logs",
                  "hf_repo", "hf_filename", "repo_id", "download_url", "run_config", "args", "env", "cwd"}
        branded = {"model", "model_id", "model_name", "display_name", "base_model", "base_model_id", "training_model"}
        return {key: "Lucifer" if key in branded and value is not None else LuciferApp.public_result(value)
                for key, value in data.items() if key not in hidden}

    def model_label(self, model: dict) -> str:
        """Keep the product name stable while profile IDs remain loader details."""
        if len(self._model_profiles) > 1 and model.get("id") in self._model_profiles:
            return f"Lucifer · profile {self._model_profiles.index(model['id']) + 1}"
        return "Lucifer"

    async def show_model_result(self, data) -> None:
        if isinstance(data, list):
            self._model_profiles = [model["id"] for model in data]
            rows = [f"{self.model_label(model)} // {model.get('status', 'configured')}" for model in data]
            if len(data) > 1:
                rows.append("Load a profile with /model load NUMBER or the selector.")
            await self.message("system", "\n".join(rows) or "Lucifer is not configured.")
        elif isinstance(data, dict):
            lines = [f"{self.model_label(data)} // {data.get('status', 'loaded')}"]
            for field, label in (("effective_context_length", "Active context"), ("native_context_length", "Native context")):
                value = data.get(field)
                if isinstance(value, int) and value > 0:
                    lines.append(f"{label}: {value:,} tokens")
            await self.message("system", "\n".join(lines))
        else:
            await self.message("system", "Lucifer is not loaded.")

    async def show_job_result(self, kind: str, data) -> None:
        jobs = data if isinstance(data, list) else [data]
        rows = []
        for job in jobs:
            lines = [f"Lucifer // {kind.upper()} // {job.get('status', 'unknown')}"]
            if job.get("id"):
                lines.append(f"Run: {job['id']}")
            if job.get("dry_run"):
                lines.append("Dry run: dataset preparation only")
            if job.get("total_examples") is not None:
                lines.append(f"Examples: {job['total_examples']}")
            if job.get("candidate_id"):
                lines.append(f"Adapter: {job['candidate_id']}")
            if job.get("error"):
                lines.append("The job failed. Details are available in its local logs.")
            rows.append("\n".join(lines))
        await self.message("system", "\n\n".join(rows) or f"No Lucifer {kind} jobs.")

    async def command(self, text: str) -> None:
        review = text.split(None, 3)
        if len(review) >= 3 and review[0].lower() == "/sleep" and review[1].lower() in {"approve", "reject"}:
            decision, ident = review[1].lower(), review[2]
            if decision == "reject" and len(review) > 3:
                raise ValueError("Usage: /sleep reject EXAMPLE_ID; enter a reason in the review panel if needed")
            updated = await self.submit_example_review(ident, decision, correction=review[3] if len(review) > 3 else None)
            await self.message("system", f"Example {ident}: review saved; current status {updated.get('status', 'unknown')}.")
            return
        if text.startswith("/correct "):
            await self.agent_goal(text[len("/correct "):].strip())
            return
        if text == "/resume" or text.startswith("/resume "):
            await self.agent_goal(text[len("/resume"):].strip(), resume=True)
            return
        if text.startswith("/agent run "):
            options, goal = self.parse_agent_goal(text[len("/agent run "):])
            await self.agent_goal(goal, options=options)
            return
        if text.startswith("/agent remember "):
            await self.remember_agent(text[len("/agent remember "):].strip())
            return
        raw = text.split(None, 2)
        if len(raw) == 3 and raw[0].lower() == "/feedback" and raw[1].lower() == "correct":
            command, args = "/feedback", ["correct", raw[2]]
        elif raw[0].lower() == "/memory":
            command, args = "/memory", [text.partition(" ")[2]]
        else:
            parts = shlex.split(text)
            command, args = parts[0].lower(), parts[1:]
        sub = args[0] if args else ""
        if command in ("/quit", "/exit"):
            self.exit()
        elif command == "/help":
            await self.message("system", HELP)
        elif command in {"/brain", "/cortex"}:
            self.set_view("brain")
            if args:
                brain = self.main_query("#brain-canvas", AdapterBrain)
                node = brain.atlas.node("adapter:" + args[0]) or brain.atlas.node(args[0])
                if not node:
                    raise ValueError("No atlas node matches that adapter ID")
                brain.selected_id = node.id
                brain.action_inspect()
                brain.refresh()
        elif command == "/agent":
            await self.agent_command(args)
        elif command in {"/attach", "/link"}:
            if len(args) not in {1, 3} or len(args) == 3 and (args[1] != "--frames" or not args[2].isdecimal()):
                raise ValueError(f"Usage: {command} FILE_OR_URL [--frames 1..24]; quote paths containing spaces")
            frames = int(args[2]) if len(args) == 3 else 8
            if not 1 <= frames <= 24:
                raise ValueError("Video frame count must be 1..24")
            await self.ensure_conversation()
            self.activity("Reading attachment; images and video use the loaded model's vision.")
            result = await self.client.attach_knowledge(self.conversation["id"], args[0], upload=command == "/attach", video_frames=frames)
            documents = await self.client.get(f"/conversations/{self.conversation['id']}/knowledge")
            self.conversation["knowledge_count"] = len(documents)
            self.update_session()
            lines = [f"Attached {result['name']} // {result['chunks']} searchable chunks // {result['id']}",
                     "Lucifer can search this source in this conversation. " + (
                         "Included as a source for sleep; model-derived visual targets need review." if self.conversation.get("training_enabled") else
                         "Training is OFF for this conversation; enable it with /privacy training on to include this source in sleep.")]
            lines.extend(result.get("warnings", []))
            await self.message("system", "\n".join(lines))
        elif command == "/knowledge":
            await self.knowledge_command(args)
        elif command == "/thinking":
            if len(args) != 1 or sub not in THINKING_MODES:
                raise ValueError("Usage: /thinking off|low|medium|high|max")
            self.thinking_mode = sub
            self.update_generation_settings()
            await self.message("system", f"Thinking {sub.upper()}. Output budget: {self.output_budget():,} tokens for thinking and the answer.")
        elif command == "/tokens":
            limit = self.output_limit()
            if len(args) != 1 or (sub not in {"auto", "max"} and (not sub.isdecimal() or int(sub) < 1 or (limit and int(sub) > limit))):
                ceiling = f"1..{limit:,}" if limit else "a positive integer"
                raise ValueError(f"Usage: /tokens {ceiling} | auto | max")
            if sub == "max" and limit is None:
                raise ValueError("The loaded model has not advertised an output/context limit yet")
            self.output_tokens_override = None if sub == "auto" else limit if sub == "max" else int(sub)
            self.update_generation_settings()
            await self.message("system", f"Output budget: {self.output_budget():,} tokens for thinking and the answer. The backend fits it to the remaining context.")
        elif command == "/model":
            if sub in ("list", "", "status"):
                await self.show_model_result(await self.client.get("/models"))
            elif sub == "current":
                await self.show_model_result(await self.client.get("/models/current"))
            elif sub == "register" and len(args) == 2:
                descriptor = Path(args[1]).expanduser()
                import yaml
                try:
                    spec = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
                except yaml.YAMLError as exc:
                    raise ValueError(f"Invalid model descriptor: {exc}") from exc
                if not isinstance(spec, dict):
                    raise ValueError("A model descriptor must contain one profile object")
                result = await self.client.post("/models/register", spec)
                await self.show_model_result(result)
                await self.message("system", "Lucifer profile registered. Select it to load; the current profile is unchanged.")
            elif sub == "reload" and len(args) == 1:
                await self.client.post("/models/reload")
                await self.show_model_result(await self.client.get("/models"))
            elif sub == "load" and len(args) == 2:
                model_id = args[1]
                if model_id.isdecimal() and 1 <= int(model_id) <= len(self._model_profiles):
                    model_id = self._model_profiles[int(model_id) - 1]
                await self.show_model_result(await self.client.load_model(model_id))
                self.last_message_id = None
            elif sub == "unload":
                await self.show_model_result(await self.client.post("/models/unload", {"model_id": args[1]} if len(args) > 1 else {}))
            else:
                raise ValueError("Usage: /model list|load PROFILE|unload|current|register FILE|reload")
        elif command in {"/history", "/sessions"}:
            await self.open_history()
        elif command in {"/chat", "/session"}:
            await self.chat_command(args)
        elif command in {"/experts", "/expert"}:
            await self.expert_command(args)
        elif command == "/privacy":
            if len(args) != 2 or sub not in ("logging", "training") or args[1] not in ("on", "off"):
                raise ValueError("Usage: /privacy logging|training on|off")
            await self.ensure_conversation()
            self.conversation = await self.client.patch(f"/conversations/{self.conversation['id']}", {f"{sub}_enabled": args[1] == "on"})
            self.update_session()
            await self.message("system", f"{sub.title()} {args[1]} for this conversation.")
        elif command == "/feedback":
            if not self.last_message_id or not self.conversation:
                raise ValueError("Finish an assistant reply before submitting feedback.")
            if sub not in ("good", "bad", "correct") or (sub == "correct" and len(args) < 2):
                raise ValueError("Usage: /feedback good|bad|correct ANSWER")
            payload = {"conversation_id": self.conversation["id"], "message_id": self.last_message_id, "label": sub}
            if sub == "correct":
                payload["correction"] = " ".join(args[1:])
            await self.client.post("/feedback", payload)
            await self.message("system", "Feedback recorded. Learning also requires logging and training consent.")
        elif command in ("/train", "/sleep"):
            await self.job_command(command[1:], args)
        elif command == "/meditation":
            await self.meditation_command(args)
        elif command == "/adapters":
            await self.show_result(await self.client.get("/adapters"))
        elif command == "/adapter" and len(args) == 2 and sub == "select":
            await self.show_result(await self.client.post("/adapters/select", {"adapter_id": None if args[1] == "base" else args[1]}))
        elif command == '/adapter' and len(args) == 2 and sub == 'activate':
            await self.show_result(await self.client.post('/adapters/activate', {'adapter_id': args[1]}))
        elif command == '/adapter' and len(args) == 2 and sub == 'delete':
            from urllib.parse import quote
            await self.show_result(await self.client.request('DELETE', '/adapters/' + quote(args[1], safe='')))
        elif command in ("/promote", "/rollback"):
            if command == "/promote" and len(args) != 1:
                raise ValueError("Usage: /promote ADAPTER_ID")
            await self.show_result(await self.client.post(f"/adapters/{command[1:]}", {"adapter_id": args[0]} if args else {}))
        elif command == "/eval":
            if sub == "run" and len(args) == 2:
                await self.show_result(await self.client.post("/eval/run", {"adapter_id": args[1]}))
            elif sub == "results":
                await self.show_result(await self.client.get("/eval/results"))
            else:
                raise ValueError("Usage: /eval run ADAPTER_ID|results")
        elif command == "/gpu" and sub in ("", "list"):
            await self.show_result(await self.client.get("/system/gpu"))
        elif command == "/memory":
            await self.show_result(await self.client.get("/memory", params={"q": " ".join(args)}))
        else:
            raise ValueError("Unknown command. Press F1 or type /help.")

    async def meditation_command(self, args: list[str]) -> None:
        sub = args[0] if args else "view"
        if sub in {"start", "resume", "cancel", "activate", "rollback"}:
            self._meditation_error = None
        if sub == "view" and len(args) <= 1:
            self.set_view("meditation")
            return
        if sub == "start":
            options = {}
            tokens, index = args[1:], 0
            while index < len(tokens):
                flag = tokens[index]
                if flag == "--dry-run":
                    options["dry_run"] = True
                elif flag == "--current":
                    if "conversation_ids" in options:
                        raise ValueError("Choose --current or --conversations, not both")
                    if not self.conversation:
                        raise ValueError("Open a saved training chat before selecting --current")
                    options["conversation_ids"] = [self.conversation["id"]]
                elif flag in {"--mode", "--conversations", "--base", "--seed", "--pilot-updates", "--idempotency-key"}:
                    index += 1
                    if index >= len(tokens):
                        raise ValueError(f"{flag} requires a value")
                    value = tokens[index]
                    if flag == "--mode":
                        if value not in {"supervised", "unsupervised"}:
                            raise ValueError("--mode requires supervised or unsupervised")
                        options["mode"] = value
                    elif flag == "--conversations":
                        if "conversation_ids" in options:
                            raise ValueError("Choose one conversation scope")
                        ids = list(dict.fromkeys(value.split(",")))
                        if any(not ident.strip() for ident in ids):
                            raise ValueError("--conversations requires comma-separated nonempty IDs")
                        options["conversation_ids"] = ids
                    elif flag == "--seed":
                        try:
                            options["seed"] = int(value)
                        except ValueError:
                            raise ValueError("--seed requires an integer") from None
                    elif flag == "--pilot-updates":
                        try:
                            updates = int(value)
                        except ValueError:
                            raise ValueError("--pilot-updates requires an integer from 1 to 100") from None
                        if not 1 <= updates <= 100:
                            raise ValueError("--pilot-updates requires an integer from 1 to 100")
                        options['pilot_updates'] = updates
                    else:
                        options["base_model" if flag == "--base" else "idempotency_key"] = value
                else:
                    raise ValueError("Unknown Meditation option: " + flag)
                index += 1
            self._meditation_error = None
            try:
                record = await self.client.start_meditation(**options)
            except (APIError, ValueError) as error:
                self._meditation_error = str(error)
                self.set_view("meditation")
                raise
            self.latest_meditation = record.get("id")
            self._meditation_pending_job = record
            self.set_view("meditation")
            self.update_meditation_view()
            self.activity("MEDITATION STARTED / " + str(record.get("id", "queued")))
            return
        if sub in {"resume", "cancel"}:
            tokens = args[1:]
            scope = {}
            if sub == "resume" and "--conversations" in tokens:
                flag = tokens.index("--conversations")
                if tokens.count("--conversations") != 1 or flag + 1 >= len(tokens):
                    raise ValueError("--conversations requires the complete expanded scope")
                ids = tokens[flag + 1].split(",")
                if any(not ident.strip() for ident in ids) or len(set(ids)) != len(ids):
                    raise ValueError("--conversations requires unique nonempty conversation IDs")
                scope["conversation_ids"] = ids
                tokens = tokens[:flag] + tokens[flag + 2:]
            key = None
            if "--idempotency-key" in tokens:
                flag = tokens.index("--idempotency-key")
                if flag + 1 >= len(tokens) or flag + 2 != len(tokens):
                    raise ValueError("End the command with --idempotency-key KEY")
                key, tokens = tokens[flag + 1], tokens[:flag]
            if len(tokens) > 1:
                raise ValueError(f"Usage: /meditation {sub} [ID] [--idempotency-key KEY]")
            ident = tokens[0] if tokens else self.selected_meditation_job().get("id")
            if not ident:
                raise ValueError("No Meditation run selected")
            record = await self.client.meditation_action(ident, sub, idempotency_key=key, **scope)
            self.latest_meditation = record.get("id", ident)
            self._meditation_pending_job = record
            self._meditation_error = None
            self.set_view("meditation")
            self.update_meditation_view()
            return
        if sub == "bases" and len(args) == 1:
            self._base_candidates = await self.client.get("/meditation/bases")
            await self.show_result(self._base_candidates)
            return
        if sub == "status" and len(args) <= 2:
            if len(args) == 2:
                from urllib.parse import quote
                record = await self.client.get("/meditation/runs/" + quote(args[1], safe=""))
                self.latest_meditation = record.get("id")
                self._meditation_pending_job = record
            self.set_view("meditation")
            self.update_meditation_view()
            return
        if sub in {"sources", "review", "results"} and len(args) <= 2:
            job = self.selected_meditation_job()
            if sub == "sources":
                record = {"conversation_ids": (job.get("request") or {}).get("conversation_ids", "all eligible training chats"),
                          **{key: job[key] for key in ("sources", "source_conversation_ids", "source_documents", "source_manifest", "dataset_metrics", "curriculum", "source_receipts") if key in job}}
                await self.push_screen(MeditationReviewScreen("MEDITATION / SOURCE ELIGIBILITY", record, self))
                return
            ident = args[1] if len(args) == 2 else self.meditation_candidate(job)
            if not ident:
                if not job:
                    raise ValueError("No Meditation run is available for inspection")
                record = {key: job[key] for key in ("id", "status", "policy", "progress", "dataset_metrics",
                    "parameter_decision", "admission_report", "weight_update_receipt", "native_update_receipt",
                    "learning_rate_calibration", "training_precision", "distribution_retention",
                    "discarded_optimizer_updates", "error", "restoration_error") if key in job}
                await self.push_screen(MeditationReviewScreen("MEDITATION / RUN RESULTS", record, self))
                return
            record = next((base for base in self._base_candidates if base.get("id") == ident), None)
            if record is None:
                self._base_candidates = await self.client.get("/meditation/bases")
                record = next((base for base in self._base_candidates if base.get("id") == ident), None)
            if record is None:
                raise ValueError("Base candidate metadata is not available yet")
            await self.push_screen(MeditationReviewScreen("MEDITATION / SAVED BASE RESULTS", record, self))
            return
        if sub in {"activate", "rollback"}:
            tokens, key = args[1:], None
            if "--idempotency-key" in tokens:
                flag = tokens.index("--idempotency-key")
                if flag + 1 >= len(tokens) or flag + 2 != len(tokens):
                    raise ValueError("End the command with --idempotency-key KEY")
                key, tokens = tokens[flag + 1], tokens[:flag]
            if len(tokens) > 1 or sub == "activate" and not tokens:
                raise ValueError(f"Usage: /meditation {sub} " + ("ID" if sub == "activate" else "[ID]") + " [--idempotency-key KEY]")
            if sub == "activate":
                result = await self.client.activate_base(tokens[0], idempotency_key=key)
            else:
                result = await self.client.rollback_base(tokens[0] if tokens else None, idempotency_key=key)
            self._structure_dirty = True
            await self.show_result(result)
            return
        raise ValueError("Usage: /meditation start|status|sources|results|bases|cancel|resume|activate ID|rollback [ID]")

    async def knowledge_command(self, args: list[str]) -> None:
        await self.ensure_conversation()
        base = f"/conversations/{self.conversation['id']}/knowledge"
        sub = args[0] if args else "list"
        if sub == "list" and len(args) <= 1:
            documents = await self.client.get(base)
            lines = [f"{doc['id']} // {doc['name']} // {doc['chunks']} chunks // {doc['retention']} // TRAIN {'ON' if doc['training_eligible'] else 'OFF'}" for doc in documents]
            await self.message("system", "\n".join(lines) or "No knowledge attached. Use /attach FILE or /link PATH_OR_URL.")
        elif sub == "search" and len(args) >= 2:
            result = await self.client.get(base + "/search", params={"query": " ".join(args[1:]), "limit": 5})
            await self.message("system", "\n\n".join(f"{item['citation']} // {item['name']}\n{item['text']}" for item in result["matches"]) or "No matching passages.")
        elif sub == "read" and len(args) in {2, 3}:
            if len(args) == 3 and not args[2].isdecimal():
                raise ValueError("Chunk must be a nonnegative integer")
            result = await self.client.get(base + "/" + args[1], params={"chunk": int(args[2]) if len(args) == 3 else 0})
            await self.message("system", "\n\n".join(f"{args[1]} chunk {item['chunk']} ({item['label']}) // {item['authority']}\n{item['text']}" for item in result["chunks"]) +
                               (f"\nNext: /knowledge read {args[1]} {result['next_chunk']}" if result["next_chunk"] is not None else ""))
        elif sub == "remove" and len(args) == 2:
            await self.client.request("DELETE", base + "/" + args[1])
            documents = await self.client.get(base)
            self.conversation["knowledge_count"] = len(documents)
            self.update_session()
            await self.message("system", "Attachment removed; derived training evidence was invalidated.")
        else:
            raise ValueError("Usage: /knowledge list|search QUERY|read DOCUMENT_ID [CHUNK]|remove DOCUMENT_ID")

    async def chat_command(self, args: list[str]) -> None:
        sub = args[0] if args else ""
        if sub == "new":
            self.conversation = None
            self.last_message_id = None
            self.reset_agent_session()
            await self.main_query("#messages", VerticalScroll).remove_children()
            self.update_session()
            self.action_toggle_agent()
            await self.message("system", "New conversation. Saving is ON; training is OFF. Use /privacy logging off if you want this conversation to stay private. Provider thinking is never saved.")
        elif sub == "clear":
            if self.agent_run_id:
                await self.message("system", "Mission thinking stays in this view until /session new explicitly resets it. Collapse a panel to reduce clutter.")
            else:
                await self.main_query("#messages", VerticalScroll).remove_children()
                await self.message("system", "View cleared. Conversation history remains available; /session new resets the context.")
        elif sub == "list":
            await self.open_history()
        elif sub == "open" and len(args) == 2:
            self.conversation = await self.client.get(f"/conversations/{args[1]}")
            self.last_message_id = None
            self.reset_agent_session()
            await self.main_query("#messages", VerticalScroll).remove_children()
            for message in self.conversation.get("messages", []):
                await self.message(message["role"], message["content"])
                if message["role"] == "assistant":
                    self.last_message_id = message["id"]
            self.update_session()
        elif sub == "save":
            await self.ensure_conversation()
            self.conversation = await self.client.patch(f"/conversations/{self.conversation['id']}", {"logging_enabled": True})
            self.update_session()
            await self.message("system", "Conversation saved. Logging is now on; training consent remains separate.")
        elif sub == "export":
            if not self.conversation:
                raise ValueError("There is no conversation to export.")
            data = await self.client.get(f"/conversations/{self.conversation['id']}/export")
            if len(args) > 1:
                target = Path(args[1]).expanduser()
            else:
                from .config import default_data_dir
                target = default_data_dir() / "exports" / f"{self.conversation['id']}-{datetime.now():%Y%m%d-%H%M%S}.json"
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Never silently overwrite a previous export.
            with target.open("x", encoding="utf-8") as handle:
                target.chmod(0o600)
                json.dump(data, handle, indent=2, ensure_ascii=False)
            await self.message("system", f"Exported to {target.resolve()}")
        else:
            raise ValueError("Usage: /session new|save|export [PATH]|clear; /history opens saved missions")

    async def job_command(self, kind: str, args: list[str]) -> None:
        sub = args[0] if args else "status"
        if kind == "sleep" and sub == "examples":
            if len(args) > 2:
                raise ValueError("Usage: /sleep examples [pending|verified|rejected|obsolete|all]")
            await self.open_example_review(args[1] if len(args) > 1 else "pending")
            return
        if kind == "sleep" and sub in {"approve", "reject"}:
            raise ValueError(f"Usage: /sleep {sub} EXAMPLE_ID" + (" [COMPLETE CORRECT ANSWER]" if sub == "approve" else ""))
        if kind == "sleep" and (not args or sub == "view"):
            self.set_view("sleep")
            return
        if sub == "start":
            if kind == "sleep":
                options = sleep_start_options(args[1:])
                self._sleep_start_error = None
                self.set_view("sleep")
                result = await self.client.start_sleep(**options)
                self._sleep_pending_job = result
            else:
                if any(arg != "--dry-run" for arg in args[1:]):
                    raise ValueError("Usage: /train start [--dry-run]")
                result = await self.client.post("/train/start", {"dry_run": "--dry-run" in args})
            setattr(self, f"latest_{kind}", result["id"])
            self.update_sleep_view()
            await self.show_job_result(kind, result)
        elif kind == "sleep" and sub == "resume":
            if len(args) not in {2, 4} or len(args) == 4 and args[2] != "--idempotency-key":
                raise ValueError("Usage: /sleep resume ID [--idempotency-key KEY]")
            ident = args[1]
            # Retain an unacknowledged intent so a manual retry after a lost
            # response reuses the service's receipt instead of training twice.
            pending = getattr(self, "_sleep_resume_keys", {})
            key = args[3] if len(args) == 4 else pending.get(ident) or uuid4().hex
            pending[ident] = key
            self._sleep_resume_keys = pending
            result = await self.client.resume_sleep(ident, idempotency_key=key)
            pending.pop(ident, None)
            self.latest_sleep = result["id"]
            self._sleep_pending_job = result
            self._sleep_start_error = None
            self.set_view("sleep")
            self.update_sleep_view()
            await self.show_job_result(kind, result)
        elif sub == "cancel":
            ident = args[1] if len(args) > 1 else getattr(self, f"latest_{kind}")
            if not ident:
                raise ValueError(f"No {kind} run selected. Provide a run ID.")
            if kind == "train":
                result = await self.client.post(f"/train/{ident}/cancel")
            else:
                result = await self.client.cancel_sleep(ident)
                if self._sleep_pending_job and self._sleep_pending_job.get("id") == result.get("id"):
                    self._sleep_pending_job = result
            self.update_sleep_view()
            await self.show_job_result(kind, result)
        elif sub in ("status", "queue"):
            path = "/train" if kind == "train" else f"/sleep/{sub}"
            await self.show_job_result(kind, await self.client.get(path))
        else:
            raise ValueError(f"Usage: /{kind} start [--dry-run]|view|status|cancel [ID]")

    async def expert_command(self, args: list[str]) -> None:
        sub = args[0] if args else "list"
        if sub == "list":
            await self.show_result(await self.client.get("/experts"))
            return
        if sub == "propose":
            values = {"conversation_ids": [self.conversation["id"]]} if self.conversation else {}
            if len(args) > 1:
                values["conversation_ids"] = [ident for ident in args[1].split(",") if ident]
            await self.show_result(await self.client.post("/experts/propose", values))
            return
        if sub not in {"build", "consolidate"} or len(args) < 2:
            raise ValueError("Usage: /experts list|propose [CONVERSATIONS]|build NAME [--base PROFILE --topics CSV --conversations CSV --sources CSV]|consolidate NAME EXPERT1,EXPERT2")
        values = {"name": args[1]}
        cursor = 2
        if sub == "consolidate":
            if len(args) < 3 or args[2].startswith("--"):
                raise ValueError("Provide source expert IDs after the consolidated expert name")
            values["source_expert_ids"] = list(dict.fromkeys(ident for ident in args[2].split(",") if ident))
            cursor = 3
        elif self.conversation:
            values["conversation_ids"] = [self.conversation["id"]]
        flags = {"--base": "base_model", "--topics": "topics", "--conversations": "conversation_ids",
                 "--sources": "source_expert_ids", "--description": "description"}
        while cursor < len(args):
            flag = args[cursor]
            if flag not in flags or cursor + 1 >= len(args):
                raise ValueError("Expert options require --base, --topics, --conversations, --sources or --description followed by a value")
            key, value = flags[flag], args[cursor + 1]
            values[key] = list(dict.fromkeys(item for item in value.split(",") if item)) if key in {"topics", "conversation_ids", "source_expert_ids"} else value
            cursor += 2
        result = await self.client.post("/experts/build", values)
        ident = result.get("id", "")
        if ident.startswith("sleep_"):
            self.latest_sleep = ident
            self._sleep_pending_job = result
        elif ident:
            self.latest_train = ident
        await self.show_job_result("expert", result)
        self.set_view("sleep")
        await self.message("system", "Expert learning is running. The cortex lists it after real adapter weights are verified; promotion remains a separate evaluation decision.")

    def action_new_chat(self) -> None:
        self.submit("/session new")

    def action_clear_chat(self) -> None:
        self.submit("/session clear")

    def action_help(self) -> None:
        self.submit("/help")

    def action_toggle_agent(self) -> None:
        tools = self.capabilities.get("capabilities", {}).get("tools", {})
        self.set_view("agent" if tools.get("status") in {"advertised", "verified"} else "chat")

    def action_history(self) -> None:
        if not self._history_loading and not isinstance(self.screen, AgentHistoryScreen):
            self.run_worker(self.open_history(), group="history-menu")

    async def open_history(self) -> None:
        if self._history_loading or isinstance(self.screen, AgentHistoryScreen):
            return
        self._history_loading = True
        try:
            records = await self.client.agent_history()
            if self.is_running:
                await self.push_screen(AgentHistoryScreen(self, records))
                self.update_sleep_view()
        except APIError as error:
            if self.is_running:
                self.activity(f"History unavailable: {error}")
        finally:
            self._history_loading = False

    def action_toggle_brain(self) -> None:
        self.set_view(self.input_mode if self.active_view == "brain" else "brain")

    def action_toggle_sleep(self) -> None:
        self.set_view(self.input_mode if self.active_view == "sleep" else "sleep")

    def action_toggle_meditation(self) -> None:
        self.set_view(self.input_mode if self.active_view == "meditation" else "meditation")

    @staticmethod
    def parse_agent_goal(text: str) -> tuple[dict, str]:
        """Leading flags are parsed; the remaining natural-language goal is literal."""
        options = {}
        while text.startswith("--"):
            match = re.match(r"^(--[a-z-]+)(?:\s+|$)", text)
            if not match:
                raise ValueError("Invalid agent option")
            flag = match[1]
            text = text[match.end():].lstrip()
            if flag == "--auto-approve-commands":
                options["auto_approve_commands"] = True
                continue
            key = {"--workspace": "workspace", "--access": "access", "--dialect": "dialect"}.get(flag)
            if key is None:
                raise ValueError(f"Unknown agent option: {flag}")
            value = re.match(r'''^("[^"]*"|'[^']*'|\S+)(?:\s+|$)''', text)
            if not value:
                raise ValueError(f"{flag} needs a value")
            options[key] = shlex.split(value[1])[0]
            text = text[value.end():].lstrip()
        if not text.strip():
            raise ValueError("Provide a goal after /agent run and any leading options")
        return options, text

    async def agent_command(self, args: list[str]) -> None:
        sub = args[0] if args else ""
        if sub == "mode" and len(args) == 2 and args[1] in {"chat", "agent"}:
            self.set_view(args[1])
        elif sub == "attach":
            if not self._pending_agent_attachment:
                raise ValueError("There is no pending mission stream to attach")
            attachment, self._pending_agent_attachment = self._pending_agent_attachment, None
            await self.agent_goal("", attachment=attachment)
        elif sub == "new":
            self.reset_agent_session()
            await self.message("system", "New mission selected. Existing conversation history remains available; /session new also resets the conversation.")
        elif sub == "resume":
            await self.agent_goal(" ".join(args[1:]), resume=True)
        elif sub == "open" and len(args) == 2:
            await self.open_agent_session(args[1])
        elif sub == "auto-stop" and len(args) == 2 and args[1] in {"on", "off"}:
            self.main_query("#agent-stop", Checkbox).value = args[1] == "on"
            self.agent_stop_when_done = args[1] == "on"
            self.update_mission_preferences(force=True)
        elif sub == "continuation" and len(args) == 2 and args[1] in {"on", "off"}:
            self.main_query("#agent-continue", Checkbox).value = args[1] == "on"
            self.agent_continue_output = args[1] == "on"
            self.update_mission_preferences(force=True)
        elif sub == "workspace" and len(args) == 2:
            self.main_query("#agent-workspace", Input).value = str(Path(args[1]).expanduser())
            self.agent_workspace = str(Path(args[1]).expanduser())
            await self.message("system", f"Mission root: {self.agent_workspace}")
        elif sub == "access" and len(args) == 2 and args[1] in {"read", "write", "execute", "computer"}:
            self.main_query("#agent-access", Select).value = args[1]
            self.agent_access = args[1]
            await self.message("system", "DIRECT PC / HOST ACCESS selected." if args[1] == "computer" else f"Mission access: {args[1]}")
        elif sub == "dialect" and len(args) == 2 and args[1] in {"chat_completions", "anthropic_messages", "responses"}:
            self.main_query("#agent-dialect", Select).value = args[1]
            self.agent_dialect = args[1]
        elif sub == "list":
            await self.open_history()
        elif sub == "status":
            result = await self.client.get(f"/agent/runs/{self.agent_run_id}") if self.agent_run_id and sub == "status" else await self.client.get("/agent/runs")
            await self.show_result(result)
        elif sub in {"cancel", "stop"}:
            await self.cancel_agent()
        elif sub in {"approve", "deny"} and len(args) == 2:
            await self.resolve_agent_approval(args[1], sub == "approve")
        elif sub == "tools":
            await self.show_result(await self.client.get("/agent/tools"))
        elif sub == "image" and len(args) == 2:
            if args[1] == "clear":
                self.agent_images.clear()
            elif len(self.agent_images) >= 4:
                raise ValueError("An agent mission supports at most four image attachments")
            elif args[1] not in self.agent_images:
                self.agent_images.append(args[1])
                self.agent_allow_images = True
            self.update_mission_preferences(force=True)
            await self.message("system", "Mission images: " + (", ".join(self.agent_images) or "none"))
        elif sub == "vision" and len(args) == 2 and args[1] in {"on", "off"}:
            self.agent_allow_images = args[1] == "on"
            if not self.agent_allow_images:
                self.agent_images.clear()
            self.update_mission_preferences(force=True)
        elif sub == "mcp" and len(args) == 2:
            self.agent_mcp_servers = [] if args[1] == "none" else list(dict.fromkeys(name.strip() for name in args[1].split(",") if name.strip()))
            self.update_mission_preferences(force=True)
            await self.message("system", "MCP selection: " + (", ".join(self.agent_mcp_servers) or "none") + ". Only configured ready servers can be used.")
        elif sub == "remember":
            await self.remember_agent()
        else:
            raise ValueError("Usage: /agent run [--workspace PATH --access read|write|execute|computer] GOAL; mode, workspace, access, dialect, status, cancel, tools")

    async def remember_agent(self, summary: str | None = None) -> None:
        if not self.agent_run_id:
            raise ValueError("Complete an agent mission before saving its approved summary")
        result = await self.client.post(f"/agent/runs/{self.agent_run_id}/remember", {"summary": summary} if summary else {})
        await self.show_result(result)

    def action_cancel_request(self) -> None:
        if self._agent_streaming and self.agent_run_id and self.busy:
            self.run_worker(self.cancel_agent(), group="agent-control")
        elif self.request_worker and self.busy:
            self.request_worker.cancel()
