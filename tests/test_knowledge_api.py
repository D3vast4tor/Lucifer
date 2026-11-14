"""Knowledge ingestion, retrieval, consent and removal through the local API."""
import base64
import copy
import time

import pytest
from fastapi.testclient import TestClient

from lucifer.api import create_app
from lucifer.config import Settings
from lucifer.tools.web import FetchResult


@pytest.fixture
def running(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml",
                              eval_file=tmp_path / "evals.yaml"), autoload="demo")
    with TestClient(app) as client:
        yield client, app


def attachment(text, name="source.txt"):
    raw = text.encode() if isinstance(text, str) else text
    return {"filename": name, "content_base64": base64.b64encode(raw).decode()}


def create(client, **values):
    response = client.post("/conversations", json=values)
    assert response.status_code == 201
    return response.json()["id"]


def test_attached_sources_flow_through_complete_sleep_dry_run(running):
    client, app = running
    cid = create(client, training_enabled=True)
    for number in range(5):
        response = client.post(f"/conversations/{cid}/knowledge", json=attachment(
            f"Source record {number}: orbit marker {number * 7 + 19} is a literal observation.", f"orbit-{number}.txt"))
        assert response.status_code == 201, response.text
    response = client.post("/sleep/start", json={"dry_run": True, "dynamic": True,
        "context_examples": True, "dataset_ids": [cid], "max_generated_examples": 0, "min_verified_examples": 1})
    assert response.status_code == 202, response.text
    ident = response.json()["id"]
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        job = client.get(f"/sleep/runs/{ident}").json()
        if job["status"] in {"dry_run", "failed", "no_new_information", "needs_review"}:
            break
        time.sleep(.01)
    assert job["status"] == "dry_run", job
    assert job["curriculum"]["train_examples"] > 0
    assert any(item["source"] == "attached_knowledge" for item in app.state.store.list("curriculum_candidates"))
    assert job["source_conversation_ids"] == [cid]


def test_upload_local_file_and_public_link_have_full_retrieval_routes(running, tmp_path, monkeypatch):
    client, _ = running
    cid = create(client)
    prefix = f"/conversations/{cid}/knowledge"
    submitted = client.post(prefix, json=attachment("Start of the source. " + "cobalt " * 900 + " FINAL_CITATION"))
    assert submitted.status_code == 201, submitted.text
    doc = submitted.json()
    assert doc["chunks"] >= 3 and doc["instructions_trusted"] is False
    assert client.get(f"/conversations/{cid}").json()["knowledge_count"] == 1
    listed = client.get(prefix).json()
    assert listed[0]["retention"] == "saved" and not listed[0]["training_eligible"]
    page = client.get(prefix + "/" + doc["id"], params={"max_chunks": 2}).json()
    assert page["next_chunk"] == 2 and len(page["chunks"]) == 2
    tail = client.get(prefix + "/" + doc["id"], params={"chunk": 2, "max_chunks": 5}).json()
    assert tail["next_chunk"] is None and "FINAL_CITATION" in tail["chunks"][-1]["text"]
    assert client.get(prefix + "/search", params={"query": "FINAL_CITATION"}).json()["matches"]
    local = tmp_path / "local.csv"
    local.write_text("item,count\nquartz,27\n")
    linked = client.post(prefix, json={"path": str(local)})
    assert linked.status_code == 201 and linked.json()["source"] == str(local)
    requested = []
    async def fetch(url, **kwargs):
        requested.append(url)
        return FetchResult(b"item,count\nopal,31\n", "text/csv", "https://example.org/final.csv")
    monkeypatch.setattr("lucifer.tools.web.fetch_public", fetch)
    remote = client.post(prefix, json={"url": "https://example.org/table.csv"})
    assert remote.status_code == 201, remote.text
    assert requested == ["https://example.org/table.csv"]
    assert remote.json()["source"] == "https://example.org/final.csv"
    assert client.get(prefix + "/search", params={"query": "opal"}).json()["matches"][0]["text"].endswith("opal,31\n")


def test_private_attachment_remains_in_memory_until_save_and_is_removed_on_delete(running):
    client, app = running
    cid = create(client, logging_enabled=False)
    prefix = f"/conversations/{cid}/knowledge"
    text = "PRIVATE_KNOWLEDGE_API_MARKER_9975"
    response = client.post(prefix, json=attachment(text))
    assert response.status_code == 201
    doc = response.json()
    assert client.get(prefix).json()[0]["retention"] == "session"
    assert app.state.store.list("knowledge_documents") == []
    assert not (app.state.store.path.parent / "knowledge").exists()
    assert client.patch(f"/conversations/{cid}", json={"training_enabled": True}).status_code == 400
    assert client.patch(f"/conversations/{cid}", json={"logging_enabled": True}).status_code == 200
    assert client.get(prefix).json()[0]["retention"] == "saved"
    source = app.state.store.path.parent / "knowledge" / cid / (doc["id"] + ".source")
    assert source.read_text() == text
    assert client.patch(f"/conversations/{cid}", json={"training_enabled": True}).status_code == 200
    assert client.get(prefix).json()[0]["training_eligible"]
    assert client.patch(f"/conversations/{cid}", json={"logging_enabled": False}).status_code == 200
    assert not source.exists()
    assert client.get(prefix + "/" + doc["id"]).json()["chunks"][0]["text"] == text
    assert client.delete(prefix + "/" + doc["id"]).status_code == 200
    assert client.get(prefix).json() == []
    assert client.get(prefix + "/" + doc["id"]).status_code == 404


def test_plain_chat_receives_generic_attachment_context_and_does_not_export_reasoning(running, monkeypatch):
    client, app = running
    cid = create(client)
    text = "The amethyst reference value is 735."
    assert client.post(f"/conversations/{cid}/knowledge", json=attachment(text)).status_code == 201
    contexts = []
    async def stream(messages, **kwargs):
        contexts.append(copy.deepcopy(messages))
        yield "reasoning", {"content": "PRIVATE_VISION_OR_CHAT_REASONING"}
        yield "delta", {"content": "The attached reference states 735."}
        yield "finish", {"finish_reason": "stop"}
    monkeypatch.setattr(app.state.manager, "stream_events", stream)
    response = client.post("/chat", json={"conversation_id": cid,
        "messages": [{"role": "user", "content": "Summarize the attachment."}]})
    assert response.status_code == 200, response.text
    relevant = [message for message in contexts[0] if text in message["content"]]
    assert relevant and relevant[0]["role"] == "user" and "untrusted" in relevant[0]["content"]
    assert "PRIVATE_VISION_OR_CHAT_REASONING" not in client.get(f"/conversations/{cid}/export").text
    assert contexts[0][-1]["content"] == "Summarize the attachment."


def test_attachment_validation_and_cross_conversation_scope(running):
    client, _ = running
    cid, other = create(client), create(client)
    prefix = f"/conversations/{cid}/knowledge"
    response = client.post(prefix, json=attachment("Only this conversation may retrieve this passage."))
    doc = response.json()
    assert client.get(f"/conversations/{other}/knowledge/{doc['id']}").status_code == 404
    assert client.delete(f"/conversations/{other}/knowledge/{doc['id']}").status_code == 404
    assert client.post(prefix, json={}).status_code == 422
    assert client.post(prefix, json={"path": "/tmp/example", **attachment("ambiguous")} ).status_code == 422
    assert client.post(prefix, json={"filename": "broken.txt", "content_base64": "!invalid!"}).status_code == 400
    assert client.post(prefix, json=attachment(b"\x00\x01\x02", "binary.dat")).status_code == 400
    assert client.get(prefix + "/search", params={"limit": 0}).status_code == 400
    assert client.get(prefix + "/" + doc["id"], params={"chunk": -1}).status_code == 400
    assert client.get(prefix + "/" + doc["id"], params={"max_chunks": 6}).status_code == 400
    assert client.delete(f"/conversations/{cid}").status_code == 200
    assert client.get(prefix).status_code == 404
