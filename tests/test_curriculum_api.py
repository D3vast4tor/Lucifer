"""History-to-curriculum acceptance checks through the real local API."""
import asyncio
from importlib.resources import files
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings
from lucifer.curriculum import prepare_curriculum, revalidate_example


def application(tmp_path):
    evaluation = tmp_path / "evals.yaml"
    evaluation.write_bytes(files("lucifer").joinpath("resources/evals.yaml").read_bytes())
    (tmp_path / "replay.jsonl").write_bytes(files("lucifer").joinpath("resources/replay.jsonl").read_bytes())
    return create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml", eval_file=evaluation), autoload="demo")


def source(client, store, number, question=None, target=None):
    conversation = client.post("/conversations", json={"logging_enabled": True, "training_enabled": True}).json()
    question = question or f"Arithmetic: compute ({13 + number} * 17) + 19."
    target = target or str((13 + number) * 17 + 19)
    store.put("messages", {"id": f"q{number}", "conversation_id": conversation["id"], "role": "user", "content": question})
    store.put("messages", {"id": f"a{number}", "conversation_id": conversation["id"], "role": "assistant", "content": target})
    assert client.post("/feedback", json={"conversation_id": conversation["id"], "message_id": f"a{number}", "label": "good"}).status_code == 201
    return conversation["id"]


def finished(client, ident):
    for _ in range(200):
        job = client.get("/sleep/runs/" + ident).json()
        if job.get("progress", {}).get("terminal"):
            return job
        time.sleep(.01)
    raise AssertionError(job)


def test_one_start_prepares_grounded_training_and_disjoint_holdouts(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        assert client.post("/models/register", json={"id": "trainable", "backend": "vllm", "path": "local/test", "context_length": 2048}).status_code == 200
        for index in range(12):
            source(client, app.state.store, index)
        submitted = client.post("/sleep/start", json={"dry_run": True, "max_generated_examples": 8})
        assert submitted.status_code == 202
        job = finished(client, submitted.json()["id"])
        assert job["status"] == "dry_run", job.get("error")
        assert job["curriculum"]["verified"] >= 3
        assert job["progress"]["restored"]
        partitions = job["curriculum_partitions"]
        assert set(partitions["train"]).isdisjoint(partitions["heldout"])
        dataset = [json.loads(line) for line in Path(job["dataset_path"]).read_text().splitlines()]
        rows = [row for row in dataset if row.get("source") != "bundled_replay"]
        assert rows and all(revalidate_example(app.state.store, row) for row in rows)
        assert all(row["partition"] == "train" for row in rows)
        assert all(row["candidate_id"] in partitions["train"] for row in rows)
        assert any(row["source"] == "generated_oracle" for row in rows)
        assert all("source_bindings" in row and row["content_hash"] for row in rows)
        assert client.get("/adapters").json() == []
        assert client.get("/system/status").json()["model"]["id"] == "demo"


def test_model_draft_requires_displayed_hash_review_and_current_source(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        cid = source(client, app.state.store, 1, "Describe our local service configuration.", "port: 8089")
        calls = []
        async def draft(messages, **kwargs):
            calls.append(messages)
            group = json.loads(messages[-1]["content"])["sources"][0]["source_group_id"]
            return json.dumps({"candidates": [{"source_group_id": group, "question": "Which port does this service use?", "answer": "8089", "kind": "rewrite"}]})
        prepared = asyncio.run(prepare_curriculum(app.state.store, [cid], draft=draft, options={"max_generated_examples": 4}))
        assert len(calls) == 1
        pending = client.get("/sleep/examples", params={"status": "pending"}).json()
        assert pending and prepared["pending"]
        candidate = pending[0]
        assert candidate["validation"]["status"] != "verified"
        review = {"decision": "approve", "content_hash": "0" * 64, "correction": "8089"}
        assert client.post(f"/sleep/examples/{candidate['id']}/review", json=review).status_code == 409
        review["content_hash"] = candidate["content_hash"]
        response = client.post(f"/sleep/examples/{candidate['id']}/review", json=review)
        assert response.status_code == 200, response.text
        approved = response.json()
        assert approved["status"] == "verified" and approved["validation"]["method"] == "explicit_human_attestation"
        assert revalidate_example(app.state.store, approved)
        again = client.post(f"/sleep/examples/{candidate['id']}/review", json=review)
        assert again.status_code == 200 and again.json()["id"] == approved["id"]
        assert client.post("/feedback", json={"conversation_id": cid, "message_id": "a1", "label": "bad"}).status_code == 201
        assert app.state.store.get("curriculum_candidates", approved["id"])["status"] == "obsolete"
        assert client.post(f"/sleep/examples/{approved['id']}/review", json={"decision": "approve", "content_hash": approved["content_hash"]}).status_code == 409


def test_insufficient_evidence_does_not_launch_training(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        client.post("/models/register", json={"id": "trainable", "backend": "vllm", "path": "local/test"})
        source(client, app.state.store, 2, "Explain this particular project decision.", "A reviewed project-specific answer.")
        async def forbidden(*args, **kwargs):
            raise AssertionError("No worker or model process may start without enough verified examples")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
        job = finished(client, client.post("/sleep/start", json={"max_generated_examples": 0, "min_verified_examples": 20}).json()["id"])
        assert job["status"] in {"insufficient_verified_data", "no_new_information"}
        assert job["progress"]["restored"] and job["progress"]["terminal"]
        assert client.get("/adapters").json() == []


def test_presentations_and_oracle_variants_do_not_inflate_source_task_diversity(tmp_path, monkeypatch):
    app = application(tmp_path)
    with TestClient(app) as client:
        client.post("/models/register", json={"id": "trainable", "backend": "vllm", "path": "local/test"})
        cid = client.post("/conversations", json={"logging_enabled": True, "training_enabled": True}).json()["id"]
        # Deliberately select a train-partition source. No assistant answer or
        # feedback is required for the exact, independently computed target.
        from lucifer.curriculum import _context, _partition
        for number in range(100):
            question = f"Arithmetic: compute ({13 + number} * 17) + 19."
            unit = {"id": "only-user-task", "role": "user", "content": question}
            if _partition(_context([unit], unit["id"], 100000)[1]) == "train":
                break
        app.state.store.put("messages", {**unit, "conversation_id": cid})
        async def forbidden(*args, **kwargs):
            raise AssertionError("One source task must not launch a training worker")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
        submitted = client.post("/sleep/start", json={"max_generated_examples": 16, "min_verified_examples": 3})
        job = finished(client, submitted.json()["id"])
        assert job["status"] == "insufficient_verified_data", job.get("error")
        assert job["curriculum"]["train_examples"] >= 3
        assert job["curriculum"]["verified_tasks"] == 1
        assert job["curriculum"]["min_verified_tasks"] == 2
        assert client.get("/adapters").json() == []
