#!/usr/bin/env python3
"""Fetch the predeclared public curriculum references, with bounded pinned bytes.

This does not import chats, assign training consent, run downloaded code or
start training. Existing receipts are checked and reused on continuation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


LIMIT = 2 * 1024 * 1024
TECHNICAL_DOCUMENTS = {
    "modelcontextprotocol/modelcontextprotocol": (
        "0175d20f8b2c49e0402c37c37ed39ae2be47277e",
        "docs/specification/2026-07-28/server/tools.mdx"),
    "sgl-project/sglang": (
        "2456e0beab40a2c130f56df4858976db91e2f735",
        "docs/docs/advanced_features/structured_outputs.mdx"),
}


def fetch(url):
    request = Request(url, headers={"User-Agent": "Lucifer-local-curriculum/1", "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=35) as response:
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("Public source exceeds the 2 MiB preparation bound")
        return raw, response.url


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(root, selected, selection_sha):
    folder = root / "references" / selected["id"]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "receipt.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["selection_sha256"] != selection_sha or sha((folder / result["filename"]).read_bytes()) != result["sha256"]:
            raise ValueError("Existing source receipt no longer matches selection or bytes")
        return result
    revision = None
    filename = "source.html"
    license_record = None
    if selected["repository"]:
        repo = selected["repository"]
        if repo in TECHNICAL_DOCUMENTS:
            revision, document_path = TECHNICAL_DOCUMENTS[repo]
            candidates = [document_path]
        else:
            commit_raw, _ = fetch(f"https://api.github.com/repos/{repo}/commits/HEAD")
            revision = json.loads(commit_raw)["sha"]
            candidates = ["README.md", "README.rst", "README"]
        for document_path in candidates:
            try:
                raw, url = fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{document_path}")
                filename = Path(document_path).name
                break
            except HTTPError as exc:
                if exc.code != 404:
                    raise
        else:
            raise ValueError("No readable source README found")
        # Preserve available license notices alongside research bytes. No code
        # from a repository is imported into the local Python environment.
        for license_name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            try:
                license_raw, license_url = fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{license_name}")
                (folder / "LICENSE.source").write_bytes(license_raw)
                license_record = {"url": license_url, "sha256": sha(license_raw), "filename": "LICENSE.source"}
                break
            except HTTPError as exc:
                if exc.code != 404:
                    raise
    else:
        raw, url = fetch(selected["url"])
    text = raw.decode("utf-8", errors="strict")
    terms = [term for term in selected["theme_terms"] if term.casefold() in text.casefold()]
    if not terms:
        raise ValueError("Reference does not contain any predeclared topic markers")
    (folder / filename).write_bytes(raw)
    result = {"id": selected["id"], "role": selected["role"], "repository": selected["repository"],
              "revision": revision, "url": url, "filename": filename, "sha256": sha(raw), "bytes": len(raw),
              "theme_terms_found": terms, "license_notice": license_record,
              "selection_sha256": selection_sha, "downloaded_at": datetime.now(timezone.utc).isoformat(),
              "training_eligible": False, "theme_check_proves_answer_correctness": False}
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    selection_raw = (root / "selection.json").read_bytes()
    selection = json.loads(selection_raw)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(row["id"], pool.submit(prepare, root, row, sha(selection_raw))) for row in selection["sources"]]
        failures = []
        for ident, future in futures:
            try:
                result = future.result()
                print(ident, result["role"], result["bytes"], "bytes", flush=True)
            except Exception as exc:
                failures.append(ident)
                print(ident, "failed:", type(exc).__name__, str(exc), flush=True)
    if failures:
        raise SystemExit("Incomplete references; continue the same selection after inspecting failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
