"""Inert, explicit question/answer document records; never executable checkers."""
from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlsplit


FORMAT = "lucifer.problem-solution.v1"
MAX_RECORD_CHARS = 2400


def parse_record(text):
    try:
        row = json.loads(text)
    except (TypeError, ValueError):
        raise ValueError("Problem/solution record must be a JSON object") from None
    if not isinstance(row, dict) or row.get("format") != FORMAT:
        raise ValueError("Unknown problem/solution document format")
    fields = {"format", "id", "question", "answer", "sources", "verification"}
    if not fields <= set(row) or set(row) - fields - {"lineage"}:
        raise ValueError("Problem/solution records require only the documented public fields")
    if ("lineage" in row and (not isinstance(row["lineage"], list) or not 1 <= len(row["lineage"]) <= 8
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in row["lineage"]))):
        raise ValueError("Problem/solution lineage is 1..8 dataset SHA-256 identifiers")
    for key in ("id", "question", "answer"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError("Problem/solution id, question and answer must be nonempty strings")
    if len(row["id"]) > 160 or len(text) > MAX_RECORD_CHARS:
        raise ValueError("Problem/solution record exceeds its 2400-character chunk bound")
    if not isinstance(row["sources"], list) or not 1 <= len(row["sources"]) <= 8:
        raise ValueError("Problem/solution records require 1..8 public source references")
    for source in row["sources"]:
        if (not isinstance(source, dict) or set(source) != {"url", "sha256"}
                or not isinstance(source["url"], str)
                or urlsplit(source["url"]).scheme not in {"http", "https"}
                or not urlsplit(source["url"]).hostname
                or not isinstance(source["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("Problem/solution source requires a public URL and SHA-256")
    receipt = row["verification"]
    if (not isinstance(receipt, dict) or set(receipt) != {"method", "receipt_sha256"}
            or not isinstance(receipt["method"], str) or not 1 <= len(receipt["method"]) <= 120
            or not isinstance(receipt["receipt_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["receipt_sha256"])):
        raise ValueError("Problem/solution verification is a method and receipt digest, never a program")
    # A supplied receipt is provenance, not proof of correctness or authority.
    return row


def source_family(uri):
    """Conservative merge keys; URL parameters cannot create independent sources.

    These keys only merge dependency components. They never grant training
    consent, truth, network access, a partition role, or evaluation independence.
    """
    try:
        parsed = urlsplit(uri)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if parsed.scheme not in {"http", "https"} or not host:
            return None
        if host in {"github.com", "raw.githubusercontent.com"} and len(parts) >= 2:
            return "repository:github.com/" + "/".join(parts[:2]).lower().removesuffix(".git")
        if host == "huggingface.co" and len(parts) >= 3 and parts[0] == "datasets":
            return "dataset:huggingface.co/" + "/".join(parts[1:3]).lower()
        if host == "huggingface.co" and len(parts) >= 2 and parts[0] in {"docs", "learn"}:
            if parts[0] == "docs" and parts[1].lower() in {"transformers", "accelerate", "peft", "trl", "datasets", "smolagents"}:
                return "repository:github.com/huggingface/" + parts[1].lower()
            if parts[:2] == ["learn", "agents-course"]:
                return "repository:github.com/huggingface/agents-course"
            return "documentation:huggingface.co/" + "/".join(parts[:2]).lower()
        if host in {"arxiv.org", "export.arxiv.org"} and len(parts) >= 2 and parts[0] in {"abs", "pdf", "html"}:
            ident = "/".join(parts[1:]).removesuffix(".pdf")
            return "paper:arxiv.org/" + re.sub(r"v\d+$", "", ident)
        # Unknown sites stay together across paths, fragments and tracking URLs.
        return "origin:" + host
    except (TypeError, ValueError):
        return None
