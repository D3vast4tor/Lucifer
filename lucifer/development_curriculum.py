"""Development-only scalar weakness signals for independently solved training.

Admission evidence never selects training examples. Reports expose family tags
and numeric outcomes; question text, reference answers and provider channels
are deliberately absent. Their digest checks integrity, not authorization: the
controller must obtain the suite from its registered frozen development binding.
"""
from __future__ import annotations

import copy
import re

from .evaluation import (
    _PATTERNS, _SOURCE_FAMILIES, _digest, source_family, validate_frozen_suite,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _report(suite_digest, revision, scalar_rows):
    known = set(_PATTERNS) | set(_SOURCE_FAMILIES.values())
    groups, unsupported, incomplete, identities = {}, [], [], set()
    for row in scalar_rows:
        _require(isinstance(row, dict) and set(row) == {"id", "repetition", "family", "candidate_pass"},
                 "Development outcome contains unexpected channels")
        ident, repetition, family, passed = (row[key] for key in ("id", "repetition", "family", "candidate_pass"))
        _require(isinstance(ident, str) and bool(ident) and type(repetition) is int and repetition >= 0,
                 "Development outcome identity is invalid")
        _require((ident, repetition) not in identities, "Development outcomes contain duplicate calls")
        identities.add((ident, repetition))
        _require(family is None or isinstance(family, str) and family in known, "Unsupported development family")
        _require(passed is None or type(passed) is bool, "Development outcomes require boolean scores")
        if passed is None:
            incomplete.append(ident)
        if family is None:
            if passed is False:
                unsupported.append(ident)
            continue
        metric = groups.setdefault(family, {"family": family, "attempts": 0, "passes": 0, "failures": 0, "incomplete": 0})
        metric["attempts"] += 1
        metric["passes"] += passed is True
        metric["failures"] += passed is False
        metric["incomplete"] += passed is None
    metrics = sorted(groups.values(), key=lambda item: item["family"])
    ranked = sorted((item for item in metrics if item["failures"]),
                    key=lambda item: (-item["failures"] / item["attempts"], -item["failures"], item["family"]))
    status = "awaiting_evidence" if incomplete or unsupported or not scalar_rows else "targeted" if ranked else "sufficient"
    value = {"schema_version": 1, "evidence_source": "development_only", "development_freeze_digest": suite_digest,
        "dataset_revision": revision, "scalar_rows": copy.deepcopy(scalar_rows), "family_metrics": metrics,
        "target_families": [item["family"] for item in ranked],
        "unsupported_failed_case_ids": sorted(set(unsupported)), "incomplete_case_ids": sorted(set(incomplete)),
        "status": status}
    value["report_sha256"] = _digest(value)
    return value


def summarize_development(development_suite, measured_rows):
    """Reduce a verified *development* panel to scalar family-level weaknesses."""
    _require(isinstance(development_suite, dict), "A frozen development suite is required")
    meta = development_suite.get("metadata", {})
    _require(meta.get("suite_purpose") == "development", "Only development evidence may target training; admission is excluded")
    validate_frozen_suite(development_suite, base_model_id=meta["base_model_id"], training_model=meta["training_model"],
        source_fingerprints=meta.get("evaluation_source_fingerprints", meta["source_fingerprints"]),
        profile_revision=meta.get("profile_revision"))
    _require(isinstance(measured_rows, list), "Development outcomes must be a list")
    families = {record["case_id"]: record["family"] for record in meta.get("dynamic_cases", [])}
    cases = {case["id"]: case for case in development_suite["cases"]}
    for ident, case in cases.items():
        if ident not in families:
            # A trusted anchor can select a known narrow family from its USER
            # question. Its reference answer is never a source of new labels.
            families[ident] = source_family({"messages": [{"role": "user", "content": case["prompt"]}]})
    scalar, seen = [], set()
    for row in measured_rows:
        _require(isinstance(row, dict) and row.get("id") in cases, "Development outcome does not belong to this suite")
        _require("candidate_pass" in row, "Development needs the latest candidate's own outcome")
        ident = row["id"]
        scalar.append({"id": ident, "repetition": row.get("repetition", 0),
                       "family": families[ident], "candidate_pass": row["candidate_pass"]})
        seen.add(ident)
    for ident in sorted(cases.keys() - seen):
        scalar.append({"id": ident, "repetition": 0, "family": families[ident], "candidate_pass": None})
    return _report(meta["freeze_digest"], meta.get("dataset_revision"), scalar)


def validate_development_targets(report):
    """Verify a bounded scalar report before it influences any generation effect."""
    _require(isinstance(report, dict) and report.get("schema_version") == 1
             and type(report.get("schema_version")) is int and report.get("evidence_source") == "development_only",
             "Only development-derived training targets are allowed")
    suite_digest = report.get("development_freeze_digest")
    _require(isinstance(suite_digest, str) and re.fullmatch(r"[0-9a-f]{64}", suite_digest),
             "Development targets lack a frozen suite identity")
    revision = report.get("dataset_revision")
    _require(revision is None or isinstance(revision, str) and bool(revision), "Development dataset identity is invalid")
    _require(isinstance(report.get("scalar_rows"), list), "Development targets lack scalar evidence")
    expected = _report(suite_digest, revision, report["scalar_rows"])
    _require(report == expected, "Development target report changed or contains unexpected channels")
    _require(report["status"] != "awaiting_evidence", "Development evidence is incomplete or needs independent review")
    return tuple(report["target_families"])
