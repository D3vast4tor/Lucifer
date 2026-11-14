#!/usr/bin/env python3
"""Stage or import the sealed public material pack; never start optimization."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import httpx

from lucifer.chat import ChatService
from lucifer.meditation_sources import prepare_meditation_sources, revalidate_meditation_sources
from lucifer.problem_solution import parse_record
from lucifer.store import Store


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def load(root):
    manifest_raw = (root / "compiled-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert sha((root / "selection.json").read_bytes()) == manifest["source_selection_sha256"]
    original = json.loads((root / "materials-manifest.json").read_text())
    assert sha((root / "materials-manifest.json").read_bytes()) == manifest["original_materials_manifest_sha256"]
    assert sha((root / "executed-builder.py").read_bytes()) == original["builder_sha256"]
    loaded = []
    for item in manifest["materials"]:
        raw = (root / item["path"]).read_bytes()
        assert sha(raw) == item["sha256"]
        reference = json.loads((root / "references" / item["id"] / "receipt.json").read_text())
        source_raw = (root / "references" / item["id"] / reference["filename"]).read_bytes()
        assert sha(source_raw) == reference["sha256"]
        for line in raw.decode().splitlines():
            row = parse_record(line)
            assert row["sources"] == [{"url": reference["url"], "sha256": reference["sha256"]}]
            index = row["id"].rsplit("-", 1)[-1]
            receipt_path = (root / "materials" / item["id"] if item["role"] == "train" else
                            root / "compiled" / item["id"]) / f"{index}.receipt.json"
            receipt_raw = receipt_path.read_bytes()
            assert sha(receipt_raw) == row["verification"]["receipt_sha256"]
            receipt = json.loads(receipt_raw)
            assert receipt["source_sha256"] == reference["sha256"]
            assert receipt["answer_sha256"] == sha(row["answer"].encode())
            if item["role"] == "train":
                assert row["lineage"] == [item["dataset_lineage"]]
                assert receipt["builder_sha256"] == original["builder_sha256"]
                assert receipt["question_sha256"] == sha(row["question"].encode())
            else:
                assert "lineage" not in row
                source_text = source_raw.decode("utf-8")
                assert row["answer"] == source_text[receipt["start"]:receipt["end"]]
        loaded.append((item, raw, reference, source_raw))
    return sha(manifest_raw), loaded


async def stage(root, loaded, manifest_sha):
    with tempfile.TemporaryDirectory(prefix="lucifer-pilot-stage-") as folder:
        store = Store(Path(folder) / "state.db")
        chat = ChatService(store, SimpleNamespace())
        roles = {}
        try:
            for item, raw, reference, source_raw in loaded:
                cid = chat.create(title="Curriculum source: " + item["id"], training_enabled=True)["id"]
                await chat.knowledge.attach(cid, filename=reference["filename"],
                    content_base64=base64.b64encode(source_raw).decode())
                doc = await chat.knowledge.attach(cid, filename=item["id"] + ".jsonl",
                    content_base64=base64.b64encode(raw).decode())
                for chunk in range(0, doc["chunks"], 5):
                    chat.knowledge.read(cid, doc["id"], chunk, 5)
                roles[cid] = item["role"]
            bundle = await prepare_meditation_sources(store, {"mode": "supervised", "source_roles": roles})
            assert revalidate_meditation_sources(store, bundle)
            groups = bundle["manifest"]["groups"]
            counts = {role: sum(row["partition"] == role for row in groups) for role in ("train", "development", "heldout")}
            assert counts == {"train": 1, "development": 2, "heldout": 8}, counts
            assert len(bundle["examples"]) == 48
            assert len(bundle["development_examples"]) == 4 and len(bundle["holdout_examples"]) == 16
            report = {"compiled_manifest_sha256": manifest_sha, "staged": True, "live_imported": False,
                      "groups": counts, "metrics": bundle["metrics"], "knowledge_documents": len(store.list("knowledge_documents")),
                      "raw_source_documents_interacted": False, "only_selected_workbook_records_interacted": True}
            atomic(root / "staging-receipt.json", report)
            print(json.dumps(report, indent=2))
        finally:
            store.close()


async def apply(root, loaded, manifest_sha, base_url):
    path = root / "import-journal.json"
    journal = json.loads(path.read_text()) if path.exists() else {"compiled_manifest_sha256": manifest_sha, "sources": {}}
    assert journal["compiled_manifest_sha256"] == manifest_sha
    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        async def call(method, endpoint, **kwargs):
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        status = await call("GET", "/system/status")
        if any(status.get(key) for key in ("agent_busy", "chat_busy", "learning_busy", "meditation_busy")):
            raise SystemExit("Model/application busy; wait without interrupting its work")
        for item, raw, reference, source_raw in loaded:
            ident = item["id"]
            title = f"Agent/LLM curriculum — {ident} [{manifest_sha[:12]}]"
            record = journal["sources"].setdefault(ident, {"title": title, "role": item["role"]})
            if "conversation_id" not in record:
                if record.get("create_pending"):
                    matches = [row for row in await call("GET", "/conversations") if row["title"] == title]
                    if len(matches) != 1:
                        raise SystemExit("Uncertain conversation creation: inspect the existing pending journal; do not repeat the effect")
                    record["conversation_id"] = matches[0]["id"]
                else:
                    record["create_pending"] = True
                    atomic(path, journal)
                    created = await call("POST", "/conversations", json={"title": title, "logging_enabled": True, "training_enabled": True})
                    record["conversation_id"] = created["id"]
                record.pop("create_pending", None)
                atomic(path, journal)
            cid = record["conversation_id"]
            conversation = await call("GET", "/conversations/" + cid)
            if not conversation.get("logging_enabled") or not conversation.get("training_enabled"):
                raise SystemExit("Source consent changed; do not opt the chat back into training")
            # The attach API has a content-hash deduplication contract within
            # this owned conversation; a completed attachment is not repeated.
            original = await call("POST", f"/conversations/{cid}/knowledge", json={
                "filename": reference["filename"], "content_base64": base64.b64encode(source_raw).decode()})
            document = await call("POST", f"/conversations/{cid}/knowledge", json={
                "filename": ident + ".jsonl", "content_base64": base64.b64encode(raw).decode()})
            assert original["sha256"] == reference["sha256"] and document["sha256"] == item["sha256"]
            assert document["chunks"] == item["examples"]
            for chunk in range(0, document["chunks"], 5):
                read = await call("GET", f"/conversations/{cid}/knowledge/{document['id']}", params={"chunk": chunk, "max_chunks": 5})
                for entry in read["chunks"]:
                    parse_record(entry["text"])
            record.update(original_document_id=original["id"], document_id=document["id"], interacted=True)
            atomic(path, journal)
            print(ident, cid, item["examples"], "interacted examples", flush=True)
        roles = {row["conversation_id"]: row["role"] for row in journal["sources"].values()}
        request = {"mode": "supervised", "conversation_ids": list(roles), "source_roles": roles,
                   "dry_run": True, "max_seq_length": 512,
                   "idempotency_key": "agent-llm-source-precommit-v1-" + manifest_sha[:20]}
        atomic(root / "precommit-request.json", request)
        journal["imported"] = True
        journal["optimization_started"] = False
        atomic(path, journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Import into the running local application; default stages in a disposable database")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest_sha, loaded = load(root)
    asyncio.run(apply(root, loaded, manifest_sha, args.base_url) if args.apply else stage(root, loaded, manifest_sha))


if __name__ == "__main__":
    main()
