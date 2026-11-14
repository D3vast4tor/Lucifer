"""Conversation-derived LoRA experts and replay-based consolidation.

An expert is a trained adapter with verified artifacts and provenance. Topic
proposals are deterministic metadata; proposing a topic never trains weights.
Consolidation replays reviewed source examples into a fresh compatible adapter,
rather than averaging arbitrary adapter tensors.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from .store import now

_STOP = set("""
the and for with that this from your you are what how which when where why
should would could please explain answer only return value function source
about into have has was were will can use using more than not its these those
each given then also show find does let must make determine calculate consider
assume step steps reasoning lucifer
complex expert learning study configured think max level carefully self
contained final worked solution assumptions main derivation results concise
verification private system instructions reveal solve derive prove provide
""".split())
_WORD_BUDGET = re.compile(r"\b(?:under|within|maximum|max|limit)\s*\d+\s*(?:words|tokens)\b", re.I)
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:60] or "generalist"


def infer_topics(examples: list[dict], maximum: int = 6) -> list[str]:
    counts = Counter()
    for example in examples:
        # Formatting budgets are instructions, including compact forms such as
        # "under800words". Do not let a shared wrapper become an expert topic.
        question = _WORD_BUDGET.sub(" ", example["messages"][0]["content"])
        words = re.findall(r"[a-z][a-z0-9_]{3,29}", question.casefold())
        # Count a term once per reviewed task so one long prompt cannot dominate.
        counts.update(set(word for word in words if word not in _STOP))
    return [word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:maximum]]


def verify_adapter_artifacts(path: str | Path) -> dict:
    """Verify a real worker checkpoint without importing Torch or pickle."""
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Expert checkpoint is not a directory")
    config_path, weights_path = directory / "adapter_config.json", directory / "adapter_model.safetensors"
    metrics_path = directory / "training_metrics.json"
    for artifact in (config_path, weights_path, metrics_path):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("A trained expert requires regular config, safetensors weights and worker metrics")
    configuration = json.loads(config_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if configuration.get("peft_type") != "LORA" or not configuration.get("base_model_name_or_path"):
        raise ValueError("Expert artifact must be a PEFT LoRA checkpoint with a recorded base")
    steps = metrics.get("optimizer_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Expert metrics must prove at least one completed optimizer step")
    parameters = metrics.get("trainable_parameters")
    if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters < 1:
        raise ValueError("Expert metrics must record trained adapter parameters")
    if not isinstance(metrics.get("train_loss"), (int, float)) or not math.isfinite(metrics["train_loss"]):
        raise ValueError("Expert training loss is absent or non-finite")
    size = weights_path.stat().st_size
    with weights_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Expert safetensors header is incomplete")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("Expert safetensors header length is invalid")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("Expert safetensors header is invalid")
    tensors, tensor_bytes, intervals = 0, 0, []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or "lora_" not in name:
            raise ValueError("Expert weights must contain LoRA tensors")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if not isinstance(shape, list) or not shape or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape):
            raise ValueError("Expert tensor shape is invalid")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(n, int) or isinstance(n, bool) for n in offsets):
            raise ValueError("Expert tensor offsets are invalid")
        begin, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES.get(value.get("dtype"), 0)
        if expected < 1 or not 0 <= begin < end <= size - 8 - header_size or end - begin != expected:
            raise ValueError("Expert tensor byte range is invalid")
        tensors += 1
        tensor_bytes += expected
        intervals.append((begin, end))
    if tensors < 1:
        raise ValueError("Expert checkpoint contains no adapter tensors")
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != size - 8 - header_size or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Expert tensor byte ranges overlap or contain unaccounted data")
    weight_digest = hashlib.sha256()
    with weights_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            weight_digest.update(chunk)
    return {"verified_at": now(), "format": "safetensors", "sha256": weight_digest.hexdigest(),
            "bytes": size, "tensor_bytes": tensor_bytes, "tensor_count": tensors,
            "optimizer_steps": steps, "trainable_parameters": parameters,
            "base_model": configuration["base_model_name_or_path"],
            **({"memory_policy": metrics["memory_policy"]} if isinstance(metrics.get("memory_policy"), dict) else {})}


class ExpertService:
    def __init__(self, store):
        self.store = store

    def list(self) -> list[dict]:
        # Verified at training commit or explicit consolidation initialization.
        # No file/model scans are performed by a telemetry/list request.
        return [adapter for adapter in self.store.list("adapters") if adapter.get("is_expert")
                and adapter.get("artifact_verification", {}).get("optimizer_steps", 0) > 0
                and adapter.get("artifact_verification", {}).get("tensor_count", 0) > 0]

    def automatic_sources(self, examples: list[dict], *, base_model_id: str, training_model: str, maximum=2,
                          allowed_conversation_ids=None) -> list[str]:
        """Choose bounded compatible replay parents by topic or source lineage.

        This reads only current registry/chat data during explicit job planning;
        resolve_sources performs the checkpoint/consent verification afterward.
        A proposal never authorizes mixing arbitrary adapter tensors.
        """
        topics = set(infer_topics(examples, 40))
        new_sources = {item["conversation_id"] for item in examples}
        candidates = []
        for adapter in self.store.list("adapters"):
            proof = adapter.get("artifact_verification", {})
            if (adapter.get("format") != "peft" or adapter.get("source_revoked")
                    or adapter.get("base_model_id") != base_model_id or adapter.get("training_model") != training_model
                    or proof.get("optimizer_steps", 0) < 1 or proof.get("tensor_count", 0) < 1):
                continue
            source_ids = adapter.get("source_conversation_ids", [])
            if allowed_conversation_ids is not None and not set(source_ids) <= set(allowed_conversation_ids):
                continue
            parent_examples = []
            valid = bool(source_ids)
            for ident in source_ids:
                try:
                    conversation = self.store.get("conversations", ident)
                except KeyError:
                    valid = False
                    break
                if not conversation.get("logging_enabled") or not conversation.get("training_enabled"):
                    valid = False
                    break
                parent_examples.extend({"messages": [message]} for message in self.store.list("messages", conversation_id=ident)
                                       if message.get("role") == "user")
            if not valid:
                continue
            parent_topics = set(adapter.get("expert_topics", [])) | set(infer_topics(parent_examples, 40))
            score = len(topics & parent_topics) + 2 * len(new_sources & set(source_ids))
            if score:
                candidates.append((score, adapter["id"]))
        return [ident for _, ident in sorted(candidates, key=lambda item: (-item[0], item[1]))[:maximum]]

    def proposals(self, examples: list[dict], *, base_model: str | None = None) -> list[dict]:
        covered = {fingerprint for expert in self.list()
                   if not expert.get("source_revoked") and (base_model is None or expert.get("base_model_id") == base_model)
                   for fingerprint in expert.get("source_example_fingerprints", [])}
        examples = [example for example in examples if example["fingerprint"] not in covered]
        if not examples:
            return []
        def proposal(selected, anchor=None):
            topics = infer_topics(selected)
            if anchor:
                topics = [anchor, *(topic for topic in topics if topic != anchor)]
            label = "Lucifer " + (" / ".join(topics[:2]) if topics else "generalist")
            fingerprints = sorted(example["fingerprint"] for example in selected)
            identity = hashlib.sha256(json.dumps([base_model, fingerprints, topics]).encode()).hexdigest()[:20]
            source_ids = sorted({item["conversation_id"] for item in selected})
            return {"id": "proposal_" + identity, "expert_name": label, "expert_topics": topics,
                    "base_model": base_model, "source_conversation_ids": source_ids,
                    "example_count": len(selected), "source_fingerprints": fingerprints,
                    "kind": "new_information", "selection": "whole_conversations",
                    "note": "Proposal only. Training includes reviewed turns of selected conversations; no weights changed."}
        result = [proposal(examples)]
        # Offer narrower recurring topics as well as the generalist proposal.
        # These are transparent lexical groups, not an invented semantic model.
        groups = {}
        for item in examples:
            for topic in infer_topics([item], 12):
                groups.setdefault(topic, set()).add(item["conversation_id"])
        all_ids = {item["conversation_id"] for item in examples}
        selections = {frozenset(all_ids)}
        for topic, ids in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            selection = frozenset(ids)
            if len(ids) < 2 or selection in selections:
                continue
            selections.add(selection)
            result.append(proposal([item for item in examples if item["conversation_id"] in ids], topic))
            if len(result) >= 6:
                break
        return result

    def resolve_sources(self, expert_ids: list[str], *, base_model_id: str, training_model: str) -> tuple[list[str], list[dict]]:
        conversations, parents = set(), []
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("source_expert_ids must not contain duplicates")
        for ident in expert_ids:
            adapter = self.store.get("adapters", ident)
            if adapter.get("format") != "peft" or not adapter.get("path"):
                raise ValueError("Consolidation sources must be trained PEFT adapters or experts")
            if adapter.get("source_revoked"):
                raise ValueError("A selected expert's source consent was revoked")
            if adapter.get("base_model_id") != base_model_id or adapter.get("training_model") != training_model:
                raise ValueError("Consolidated experts must use the same registered base and checkpoint")
            verification = verify_adapter_artifacts(adapter["path"])
            recorded = adapter.get("artifact_verification")
            if recorded and verification["sha256"] != recorded.get("sha256"):
                raise ValueError("A selected expert's checkpoint changed after registration")
            source_ids = adapter.get("source_conversation_ids", [])
            if not source_ids:
                raise ValueError("Expert has no replayable reviewed source conversations")
            for source in source_ids:
                try:
                    conversation = self.store.get("conversations", source)
                except KeyError:
                    raise ValueError("A selected expert's source conversation was deleted") from None
                if not conversation.get("logging_enabled") or not conversation.get("training_enabled"):
                    raise ValueError("A selected expert's source conversation lacks current training consent")
                conversations.add(source)
            if not recorded:
                adapter = self.store.update("adapters", ident, artifact_verification=verification)
            parents.append(adapter)
        return sorted(conversations), parents

    def metadata(self, request: dict, examples: list[dict], parents: list[dict]) -> dict:
        topics = list(dict.fromkeys(request.get("expert_topics") or infer_topics(examples)))
        name = (request.get("expert_name") or ("Lucifer " + " / ".join(topics[:2]) if topics else "Lucifer generalist")).strip()
        if not name or len(name) > 120 or len(topics) > 20 or any(not isinstance(topic, str) or not topic.strip() or len(topic) > 80 for topic in topics):
            raise ValueError("Expert name/topics exceed their bounds")
        return {"is_expert": True, "expert_name": name, "expert_description": str(request.get("expert_description", ""))[:2000],
                "expert_topics": topics, "source_expert_ids": [parent["id"] for parent in parents],
                "expert_kind": "consolidation" if parents else "new_information",
                "expert_training_status": "planned", "expert_quality_status": "not_evaluated",
                "consolidation_method": "reviewed_source_replay" if parents else None,
                "source_example_fingerprints": sorted(example["fingerprint"] for example in examples)}
