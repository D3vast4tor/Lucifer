"""Evidence-bound memory reconciliation, without provider or tool execution.

Verification here means that an assertion has current consented source evidence,
not that arbitrary user statements have become independently proven world facts.
Ambiguous assertions sharing an explicit scope/key are withheld from retrieval.
"""
from __future__ import annotations

import hashlib
import json
import re

from .store import now

_CORRECTION = re.compile(r"^\s*(?:actually\b|correction\b|instead\b|replace\b|change\b|revise\b|switch\b|from now on\b|use\b.+\binstead of\b)", re.I)
_NEW_TASK = re.compile(r"^\s*(?:new (?:task|topic)\s*:|start (?:a )?new (?:task|topic)\b)", re.I)
_FIELDS = re.compile(r"(?m)^\s*([A-Za-z][\w -]{0,24})\s*[:=]\s*([^\n;]{1,200})\s*$")
_ALIASES = {"city": "destination", "output_format": "format", "programming_language": "language"}


class SourceEvidenceError(ValueError):
    def __init__(self, message, *, conversation_id=None, message_ids=None, reason="source_snapshot_changed"):
        super().__init__(message)
        self.conversation_id, self.message_ids, self.reason = conversation_id, message_ids, reason


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scrub(value):
    from .learning import scrub
    return scrub(value or "")


def _text_hash(value):
    return hashlib.sha256(_scrub(value).encode()).hexdigest()


def _normal(value):
    return " ".join(_scrub(value).casefold().split())


def capture_source_snapshot(store, conversation_id, message_ids=None, *, require_training=True):
    """Bind selected public messages and current review targets to consent."""
    conversation = store.get("conversations", conversation_id)
    if not conversation.get("logging_enabled") or (require_training and not conversation.get("training_enabled")):
        reason = "logging_consent_revoked" if not conversation.get("logging_enabled") else "training_consent_revoked"
        raise SourceEvidenceError("Source lacks current logging/training consent", conversation_id=conversation_id, reason=reason)
    messages = store.list("messages", conversation_id=conversation_id)
    selected = set(message_ids) if message_ids is not None else None
    missing = selected - {item["id"] for item in messages} if selected is not None else set()
    if missing:
        raise SourceEvidenceError("A bound source message no longer exists", conversation_id=conversation_id, message_ids=sorted(missing))
    reviews = {item.get("message_id"): item for item in store.list("feedback", conversation_id=conversation_id)}
    evidence = []
    for message in messages:
        if selected is not None and message["id"] not in selected:
            continue
        role, state, target = message.get("role"), "context_only", None
        if role == "user":
            state = "user"
        elif role == "assistant":
            review = reviews.get(message["id"], {})
            correction = _scrub(review.get("correction"))
            approved = bool(correction) or review.get("label") in {"good", "user_liked"}
            state = "approved" if approved else "rejected" if review else "unreviewed"
            if approved:
                target = _text_hash(correction or message.get("content"))
        evidence.append({"id": message["id"], "role": role, "learning_revision": message.get("learning_revision", 0),
                         "content_sha256": _text_hash(message.get("content")),
                         "review_state": state, "target_sha256": target})
    snapshot = {"schema_version": 1, "conversation_id": conversation_id, "require_training": bool(require_training),
                "learning_revision": conversation.get("learning_revision", 0), "logging_revision": conversation.get("logging_revision", 0),
                "message_ids": [item["id"] for item in evidence], "messages": evidence}
    knowledge_ids = []
    for message in messages:
        if selected is not None and message["id"] not in selected:
            continue
        if message.get("knowledge_id"):
            knowledge_ids.append(message["knowledge_id"])
        knowledge_ids.extend(message.get("knowledge_source_ids", []))
    if knowledge_ids:
        from .knowledge import document_binding
        try:
            snapshot["knowledge"] = [document_binding(store, ident) for ident in dict.fromkeys(knowledge_ids)]
        except KeyError:
            raise SourceEvidenceError("A bound knowledge source was removed", conversation_id=conversation_id,
                                      reason="knowledge_removed") from None
    if require_training and not evidence:
        raise ValueError("Training evidence must bind at least one public source message")
    return {**snapshot, "snapshot_sha256": _digest(snapshot)}


def validate_source_snapshot(store, snapshot, *, require_training=True, allow_superseded=False):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("Unknown source evidence schema")
    content = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if _digest(content) != snapshot.get("snapshot_sha256"):
        raise ValueError("Source evidence snapshot changed")
    if require_training and not snapshot.get("require_training"):
        raise ValueError("Logging-only evidence cannot authorize training")
    try:
        current = capture_source_snapshot(store, snapshot["conversation_id"], snapshot["message_ids"],
                                          require_training=bool(snapshot.get("require_training") or require_training))
    except KeyError:
        raise SourceEvidenceError("A bound source conversation was deleted", conversation_id=snapshot["conversation_id"], reason="conversation_deleted") from None
    expected, observed = snapshot, current
    if not require_training and not snapshot.get("require_training"):
        # Training consent changes do not revoke separately approved, logging-
        # only notes. Logging and message epochs still protect those notes.
        expected = {key: value for key, value in snapshot.items() if key not in {"learning_revision", "snapshot_sha256"}}
        observed = {key: value for key, value in current.items() if key not in {"learning_revision", "snapshot_sha256"}}
    if observed != expected:
        entries = {item["id"]: item for item in current["messages"]}
        changed = [item["id"] for item in snapshot["messages"] if entries.get(item["id"]) != item]
        raise SourceEvidenceError("Source content or current review changed", conversation_id=snapshot["conversation_id"], message_ids=changed)
    if not allow_superseded and source_context_superseded(store, snapshot):
        raise SourceEvidenceError("A later explicit user correction superseded this source task", conversation_id=snapshot["conversation_id"],
                                  message_ids=snapshot["message_ids"][-1:], reason="later_user_correction")
    return True


def source_context_superseded(store, snapshot):
    """Later corrections affect the current task, stopping at an explicit new task."""
    if snapshot.get("knowledge"):
        bound_messages = [message for message in store.list("messages", conversation_id=snapshot["conversation_id"])
                          if message["id"] in snapshot.get("message_ids", [])]
        if bound_messages and all(message.get("knowledge_id") for message in bound_messages):
            # An immutable attachment is replaced/removed explicitly. Ordinary
            # corrections to a later chat task do not edit its source bytes.
            return False
    bound = set(snapshot.get("message_ids", []))
    if not bound:
        return False
    messages = store.list("messages", conversation_id=snapshot["conversation_id"])
    end = max((index for index, item in enumerate(messages) if item["id"] in bound), default=-1)
    constraints = {}
    for item in messages[:end + 1]:
        if item["id"] in bound and item.get("role") == "user":
            for match in _FIELDS.finditer(item.get("content", "")):
                key = match[1].casefold().replace(" ", "_")
                constraints[_ALIASES.get(key, key)] = match[2].strip()
    for item in messages[end + 1:]:
        if item.get("role") != "user":
            continue
        text = _scrub(item.get("content"))
        if _NEW_TASK.search(text):
            break
        if _CORRECTION.search(text):
            return True
        for match in _FIELDS.finditer(text):
            key = match[1].casefold().replace(" ", "_")
            key = _ALIASES.get(key, key)
            old = constraints.get(key, snapshot["conversation_id"] if key == "scope" else None)
            if old is not None and old != match[2].strip():
                return True
    return False


def _source_matches(record, conversation_id, message_id):
    bindings = record.get("source_bindings", [])
    matching = [binding for binding in bindings if binding.get("conversation_id") == conversation_id]
    if matching:
        bound_ids = {ident for binding in matching for ident in binding.get("message_ids", [])}
        bound_ids.update(record.get("source_message_ids", []))
        return message_id is None or not bound_ids or message_id in bound_ids
    sources = set(record.get("source_conversation_ids") or [])
    if record.get("conversation_id"):
        sources.add(record["conversation_id"])
    if conversation_id in sources:
        ids = record.get("source_message_ids") or ([record["source_message_id"]] if record.get("source_message_id") else [])
        return message_id is None or not ids or message_id in ids
    # Adaptive jobs retain cumulative source bindings below their lineage.
    # Match metadata recursively without consulting dataset or model files.
    for value in record.values():
        if isinstance(value, dict) and _source_matches(value, conversation_id, message_id):
            return True
        if isinstance(value, list) and any(isinstance(item, dict) and _source_matches(item, conversation_id, message_id)
                                           for item in value):
            return True
    return False


def _candidate_matches(record, candidate_id):
    return (candidate_id in record.get("curriculum_manifest", {}).get("candidate_ids", [])
            or any(ref.get("candidate_id") == candidate_id
                   for key in ("example_refs", "holdout_refs", "curriculum_example_refs", "curriculum_holdout_refs")
                   for ref in record.get(key, [])))


def _invalidate_adaptive_records(store, *, adapter_ids, run_ids, reason, source_matches=None, candidate_id=None):
    """Revoke cached learning evidence without loading files or running models."""
    for kind in ("adaptive_datasets", "adaptive_development_suites", "admission_receipts"):
        for item in store.list(kind):
            affected = (item.get("adapter_id") in adapter_ids or item.get("run_id") in run_ids
                        or source_matches is not None and source_matches(item)
                        or candidate_id is not None and _candidate_matches(item, candidate_id))
            if not affected:
                continue
            changes = {"artifact_revoked": True, "source_invalidated": True, "source_invalidation_reason": reason}
            if kind == "admission_receipts":
                changes["status"] = "revoked"
            if any(item.get(key) != value for key, value in changes.items()):
                store.update(kind, item["id"], **changes)


def invalidate_memory_sources(store, conversation_id, message_id=None, *, reason="source_changed"):
    """Invalidate linked evidence and descendants without runtime execution."""
    from .meditation_sources import invalidate_meditation_sources
    invalidate_meditation_sources(store, conversation_id=conversation_id, message_id=message_id, reason=reason)
    counts = {"memories_invalidated": 0, "candidates_obsoleted": 0, "adapters_invalidated": 0, "runs_invalidated": 0}
    with store.transaction():
        for item in store.list("memory_items"):
            if reason == "training_consent_revoked" and item.get("source") == "approved_agent_summary":
                continue
            if _source_matches(item, conversation_id, message_id) and item.get("status") != "invalidated":
                store.update("memory_items", item["id"], status="invalidated", active=False, verified=False,
                             invalidation_reason=reason, invalidated_at=now())
                counts["memories_invalidated"] += 1
        for item in store.list("curriculum_candidates"):
            if _source_matches(item, conversation_id, message_id) and item.get("status") != "obsolete":
                store.update("curriculum_candidates", item["id"], status="obsolete", obsolete_reason=reason, obsolete_at=now())
                counts["candidates_obsoleted"] += 1
        adapters = store.list("adapters")
        revisions = [item for item in store.list("adaptive_datasets") if _source_matches(item, conversation_id, message_id)]
        revision_ids = {item["id"] for item in revisions}
        invalid = {item["id"] for item in adapters if _source_matches(item, conversation_id, message_id)
                   or revision_ids.intersection(item.get("adaptive_lineage", {}).get("revision_ids", []))}
        while True:
            descendants = {item["id"] for item in adapters if set(item.get("source_expert_ids", [])) & invalid
                           or item.get("parent_adapter_id") in invalid}
            if descendants <= invalid:
                break
            invalid |= descendants
        for item in adapters:
            if item["id"] in invalid and not item.get("source_invalidated"):
                changes = {"source_invalidated": True, "source_revoked": True, "source_invalidation_reason": reason,
                           "source_invalidation_conversation_id": conversation_id, "source_invalidation_message_id": message_id}
                if item.get("status") not in {"active", "archived"}:
                    changes["status"] = "failed"
                store.update("adapters", item["id"], **changes)
                counts["adapters_invalidated"] += 1
        invalid_runs = {item.get("run_id") for item in revisions if item.get("run_id")}
        for kind in ("sleep_runs", "training_runs"):
            for item in store.list(kind):
                if (_source_matches(item, conversation_id, message_id) or item.get("candidate_adapter_id") in invalid
                        or item["id"] in invalid_runs):
                    invalid_runs.add(item["id"])
                    if not item.get("artifact_revoked"):
                        store.update(kind, item["id"], artifact_revoked=True, source_invalidated=True,
                                     source_invalidation_reason=reason)
                        counts["runs_invalidated"] += 1
        _invalidate_adaptive_records(store, adapter_ids=invalid, run_ids=invalid_runs, reason=reason,
                                     source_matches=lambda item: _source_matches(item, conversation_id, message_id))
    return counts


def register_approved_memory(store, record, *, source_message_id=None):
    """Bind an explicitly user-approved agent note under logging-only consent."""
    conversation_id = record["conversation_id"]
    snapshot = capture_source_snapshot(store, conversation_id, [source_message_id] if source_message_id else [], require_training=False)
    content = _scrub(record.get("content"))
    if not content:
        raise ValueError("Approved memory has no public content")
    if (record.get("schema_version") == 1 and record.get("status") == "active"
            and record.get("source_bindings") == [snapshot] and record.get("value_sha256") == _text_hash(content)):
        return record
    return store.put("memory_items", {**record, "schema_version": 1, "source": "approved_agent_summary", "content": content,
        "scope_id": conversation_id, "fact_id": "fact_" + _digest([conversation_id, record["id"]])[:24],
        "task_id": "manual_" + record["id"], "version": record.get("version", 1), "status": "active", "active": True,
        "verified": True, "validation": {"status": "verified", "method": "explicit_user_approval"},
        "source_bindings": [snapshot], "source_conversation_ids": [conversation_id],
        "source_message_ids": snapshot["message_ids"], "value_sha256": _text_hash(content)})


def initialize_memory(store):
    """One startup migration: preserve approved notes, withhold legacy copies."""
    counts = {"manual_preserved": 0, "legacy_invalidated": 0}
    with store.transaction():
        for item in store.list("memory_items"):
            if item.get("schema_version") == 1:
                continue
            if item.get("source") == "approved_agent_summary":
                try:
                    register_approved_memory(store, item)
                    counts["manual_preserved"] += 1
                    continue
                except (ValueError, KeyError):
                    store.delete("memory_items", item["id"])
            else:
                if item.get("status") != "invalidated" or item.get("invalidation_reason") != "legacy_unbound_memory":
                    store.update("memory_items", item["id"], status="invalidated", active=False, verified=False,
                                 invalidation_reason="legacy_unbound_memory")
                    counts["legacy_invalidated"] += 1
    return counts


def invalidate_candidate_dependents(store, candidate_id):
    from .meditation_sources import invalidate_meditation_sources
    invalidate_meditation_sources(store, candidate_id=candidate_id, reason="candidate_review_changed")
    with store.transaction():
        return _invalidate_candidate_dependents(store, candidate_id)


def _invalidate_candidate_dependents(store, candidate_id):
    """A draft review invalidates its consumers, without choosing a fact winner."""
    adapters = store.list("adapters")
    revisions = [item for item in store.list("adaptive_datasets") if _candidate_matches(item, candidate_id)]
    revision_ids = {item["id"] for item in revisions}
    invalid = {item["id"] for item in adapters if _candidate_matches(item, candidate_id)
               or revision_ids.intersection(item.get("adaptive_lineage", {}).get("revision_ids", []))}
    while True:
        descendants = {item["id"] for item in adapters if set(item.get("source_expert_ids", [])) & invalid or item.get("parent_adapter_id") in invalid}
        if descendants <= invalid:
            break
        invalid |= descendants
    count = 0
    for item in adapters:
        if item["id"] in invalid and not item.get("source_invalidated"):
            store.update("adapters", item["id"], source_invalidated=True, source_revoked=True,
                source_invalidation_reason="candidate_review_changed", status=item["status"] if item.get("status") in {"active", "archived"} else "failed")
            count += 1
    invalid_runs = {item.get("run_id") for item in revisions if item.get("run_id")}
    for kind in ("sleep_runs", "training_runs"):
        for item in store.list(kind):
            if item.get("candidate_adapter_id") in invalid or item["id"] in invalid_runs:
                invalid_runs.add(item["id"])
                if not item.get("artifact_revoked"):
                    store.update(kind, item["id"], artifact_revoked=True, source_invalidated=True, source_invalidation_reason="candidate_review_changed")
    _invalidate_adaptive_records(store, adapter_ids=invalid, run_ids=invalid_runs,
                                 reason="candidate_review_changed", candidate_id=candidate_id)
    return {"adapters_invalidated": count}


def _reviewed_examples(store, conversation_ids):
    from .learning import build_dataset
    from .curriculum import _context
    selected = store.list("conversations") if conversation_ids is None else conversation_ids
    examples, _ = build_dataset(store, conversation_ids, max_episodes=max(1, len(selected)))
    recovered = []
    for item in examples:
        messages = store.list("messages", conversation_id=item["conversation_id"])
        task_rows = []
        for message in messages:
            if message.get("role") == "user" and _NEW_TASK.search(message.get("content", "")):
                task_rows = []
            if message.get("role") in {"user", "assistant"}:
                task_rows.append(message)
            if message["id"] == item["source_message_id"]:
                break
        try:
            clean_rows = [{**row, "content": _scrub(row.get("content"))} for row in task_rows]
            question, _, constraints, _ = _context(clean_rows, item["source_message_id"], 262144)
        except ValueError:
            continue
        ids = [row["id"] for row in task_rows]
        task_id = "task_" + _digest([item["conversation_id"], task_rows[0]["id"]])[:24]
        item["messages"][0]["content"] = question
        item.update(source_bindings=[capture_source_snapshot(store, item["conversation_id"], ids)],
                    source_message_ids=ids, task_id=task_id, fact_key="task:" + task_id,
                    scope_id=constraints.get("scope", {}).get("value") or item["conversation_id"],
                    validation={"status": "verified", "method": "explicit_reviewed_final"})
        recovered.append(item)
    return recovered


def _reconcile_source_conflicts(store):
    """Cache current immutable conflict witnesses at explicit reconciliation.

    Retrieval consumes these hash-only markers without walking candidate or
    source messages. A rejected exercise does not retract its source facts.
    """
    current = {}
    for witness in store.list("curriculum_candidates"):
        if witness.get("validation", {}).get("method") != "scoped_fact_conflict":
            continue
        bindings = witness.get("source_bindings", [])
        cid, mid = witness.get("conversation_id"), witness.get("source_message_id")
        if not bindings or not any(binding.get("conversation_id") == cid and mid in binding.get("message_ids", []) for binding in bindings):
            continue
        try:
            for binding in bindings:
                validate_source_snapshot(store, binding)
        except (ValueError, KeyError):
            continue
        ident = "source_conflict_" + _digest([cid, mid, [binding["snapshot_sha256"] for binding in bindings]])[:24]
        marker = {"id": ident, "conversation_id": cid, "source_message_id": mid,
                  "source_bindings": bindings, "active": True, "status": "conflicted"}
        current[ident] = marker
    previous = {item["id"]: item for item in store.list("memory_source_conflicts")}
    for ident, marker in current.items():
        if ident not in previous or any(previous[ident].get(key) != value for key, value in marker.items()):
            store.put("memory_source_conflicts", marker)
    for ident, marker in previous.items():
        if ident not in current and marker.get("active"):
            store.update("memory_source_conflicts", ident, active=False, status="source_changed")
    return {(item["conversation_id"], item["source_message_id"]) for item in current.values()}


def _matches_conflicted_source(item, conflicts):
    return any((binding.get("conversation_id"), mid) in conflicts
               for binding in item.get("source_bindings", []) for mid in binding.get("message_ids", []))


def reconcile_memory(store, conversation_ids=None, *, examples=None):
    counts = {"created": 0, "unchanged": 0, "superseded": 0, "conflicted": 0, "invalidated": 0, "rejected_examples": 0}
    with store.transaction():
        counts.update(initialize_memory(store))
        source_conflicts = _reconcile_source_conflicts(store)
        selected = set(conversation_ids) if conversation_ids is not None else None
        existing = store.list("memory_items")
        touched = {item.get("fact_id") for item in existing if item.get("fact_id") and
                   (selected is None or selected.intersection(item.get("source_conversation_ids", [item.get("conversation_id")]))) }
        for item in existing:
            if item.get("schema_version") != 1 or item.get("status") not in {"active", "conflicted"}:
                continue
            if selected is not None and not selected.intersection(item.get("source_conversation_ids", [item.get("conversation_id")])):
                continue
            try:
                for binding in item.get("source_bindings", []):
                    validate_source_snapshot(store, binding, require_training=item.get("source") != "approved_agent_summary")
                candidate_id = item.get("candidate_id")
                if candidate_id:
                    try:
                        candidate = store.get("curriculum_candidates", candidate_id)
                    except KeyError:
                        candidate = {}
                    if candidate.get("status") != "verified" or candidate.get("content_hash") != item.get("candidate_content_hash"):
                        invalidate_candidate_dependents(store, candidate_id)
                        if item.get("status") == "conflicted":
                            if not item.get("candidate_invalidated"):
                                store.update("memory_items", item["id"], candidate_invalidated=True, active=False, verified=False)
                        else:
                            store.update("memory_items", item["id"], status="invalidated", active=False, verified=False,
                                         invalidation_reason="candidate_review_changed")
                            counts["invalidated"] += 1
            except ValueError as error:
                mids = getattr(error, "message_ids", None)
                if mids is None and not isinstance(error, SourceEvidenceError):
                    mids = binding.get("message_ids") or None
                for mid in mids or [None]:
                    invalidated = invalidate_memory_sources(store, binding["conversation_id"], mid,
                        reason=getattr(error, "reason", "source_snapshot_changed"))
                    counts["invalidated"] += invalidated["memories_invalidated"]
        material = _reviewed_examples(store, conversation_ids) if examples is None else examples
        groups = {}
        for item in store.list("memory_items"):
            groups.setdefault(item.get("fact_id"), []).append(item)
            if _matches_conflicted_source(item, source_conflicts) or item.get("source_conflicted"):
                touched.add(item.get("fact_id"))
        for example in material:
            try:
                bindings = example.get("source_bindings", [])
                if example.get("validation", {}).get("status") != "verified" or not bindings:
                    raise ValueError("Memory requires independently validated source evidence")
                for binding in bindings:
                    validate_source_snapshot(store, binding)
                if _matches_conflicted_source(example, source_conflicts):
                    raise ValueError("A source fact remains ambiguous")
                sources = sorted({binding["conversation_id"] for binding in bindings})
                if selected is not None and not selected.intersection(sources):
                    raise ValueError("Memory example lies outside the selected scope")
                messages = example["messages"]
                if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
                    raise ValueError("Memory requires a public user/final-answer pair")
                question, answer = _scrub(messages[0]["content"]), _scrub(messages[-1]["content"])
                if not question or not answer:
                    raise ValueError("Memory has no public question/final target")
            except (ValueError, KeyError, TypeError, IndexError):
                counts["rejected_examples"] += 1
                continue
            scope = str(example.get("scope_id") or sources[0])
            key = str(example.get("fact_key") or "question_" + _digest(_normal(question)))
            fact_id = "fact_" + _digest([scope, key])[:24]
            task_id = str(example.get("task_id") or "task_" + _digest([sources, question])[:24])
            value_hash = _digest(" ".join(answer.split()))
            binding_hash = _digest([binding["snapshot_sha256"] for binding in bindings])
            final_id = example.get("source_message_id") or example.get("provenance", {}).get("source_final_message_id")
            if not final_id:
                final_id = next((message["id"] for binding in reversed(bindings) for message in reversed(binding["messages"])
                                 if message.get("role") == "assistant"), None)
            versions = groups.setdefault(fact_id, [])
            same = next((item for item in versions if item.get("task_id") == task_id and item.get("value_sha256") == value_hash
                         and item.get("source_binding_sha256") == binding_hash and item.get("status") in {"active", "conflicted"}), None)
            touched.add(fact_id)
            if same:
                counts["unchanged"] += 1
                continue
            version = max((item.get("version", 0) for item in versions), default=0) + 1
            ident = "memory_" + fact_id[5:] + f"_v{version}"
            explicit_user = any(message.get("role") == "user" and re.search(r"\b(?:correction|supersed\w*|replace|instead|update|actually)\b",
                _scrub(store.get("messages", message["id"]).get("content")), re.I)
                for binding in bindings for message in binding["messages"])
            supersedes = (set(example.get("supersedes", [])) if explicit_user and
                          example.get("validation", {}).get("explicit_supersession") is True else set())
            for old in versions:
                same_task = old.get("task_id") == task_id
                changed_source = old.get("source_message_id") == final_id and old.get("source_binding_sha256") != binding_hash
                explicit = old["id"] in supersedes or old.get("task_id") in supersedes
                if ((same_task and changed_source) or explicit) and old.get("status") != "superseded":
                    store.update("memory_items", old["id"], status="superseded", active=False, superseded_by=ident)
                    old.update(status="superseded", active=False, superseded_by=ident)
                    counts["superseded"] += 1
            fingerprints = example.get("source_fingerprints") or ([example["fingerprint"]] if example.get("fingerprint") else [])
            record = store.put("memory_items", {"id": ident, "schema_version": 1, "type": "reviewed_task", "scope_id": scope,
                "fact_id": fact_id, "fact_key": key, "task_id": task_id, "version": version, "status": "active", "active": True, "verified": True,
                "content": f"Task: {question}\nVerified answer: {answer}", "question": question, "answer": answer,
                "value_sha256": value_hash, "conversation_id": sources[0], "source_conversation_ids": sources,
                "source_bindings": bindings, "source_binding_sha256": binding_hash, "source_message_ids": example.get("source_message_ids", []),
                "source_message_id": final_id,
                "candidate_id": example.get("candidate_id"), "candidate_content_hash": example.get("content_hash"),
                "source_fingerprints": fingerprints, "validation": example["validation"], "retrieval": "lexical"})
            versions.append(record)
            counts["created"] += 1
        for fact_id in touched:
            live = [item for item in groups.get(fact_id, []) if item.get("status") in {"active", "conflicted"}]
            value_conflict = len({item["value_sha256"] for item in live}) > 1
            for item in live:
                source_conflicted = _matches_conflicted_source(item, source_conflicts)
                conflict = value_conflict or source_conflicted
                status = "conflicted" if conflict else "active"
                if item["status"] != status or bool(item.get("source_conflicted")) != source_conflicted:
                    store.update("memory_items", item["id"], status=status, active=not conflict,
                                 source_conflicted=source_conflicted)
                    counts["conflicted"] += int(conflict)
        counts["active"] = sum(item.get("status") == "active" and item.get("verified") is True for item in store.list("memory_items"))
    return counts


def reconcile_after_turn(store, conversation_id):
    """Explicit message-commit hook; never invokes a provider or trusts a reply."""
    return reconcile_memory(store, [conversation_id], examples=[])


def conflicted_source_messages(store, *, records=None):
    """Training must withhold every candidate anchored to a conflicted source."""
    result = {(item["conversation_id"], item["source_message_id"]) for item in store.list("memory_source_conflicts")
              if item.get("active") and item.get("status") == "conflicted"}
    for item in store.list("memory_items") if records is None else records:
        if item.get("status") != "conflicted":
            continue
        final_id = item.get("source_message_id")
        for binding in item.get("source_bindings", []):
            if final_id and final_id in binding.get("message_ids", []):
                result.add((binding["conversation_id"], final_id))
            elif not final_id:
                final = next((message["id"] for message in reversed(binding.get("messages", []))
                              if message.get("role") == "assistant"), None)
                if final:
                    result.add((binding["conversation_id"], final))
    return result
