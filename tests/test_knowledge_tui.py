"""Knowledge commands exercise upload bytes, citations, consent and readiness."""
import base64
import json

import httpx
import pytest
from textual.widgets import Input

from lucifer.client import APIClient
from lucifer.tui import LuciferApp
from test_tui import Backend, submit


class KnowledgeBackend(Backend):
    def __init__(self):
        super().__init__()
        self.documents = []
        self.queries = []

    def handle(self, request):
        path = request.url.path
        base = "/conversations/conv_test/knowledge"
        if path.startswith(base):
            payload = json.loads(request.content) if request.content else {}
            self.requests.append((request.method, path, payload))
            self.queries.append((path, dict(request.url.params)))
            if path == base:
                if request.method == "POST":
                    document = {"id": f"knowledge_{len(self.documents) + 1}",
                                "name": payload.get("filename", payload.get("url", payload.get("path"))),
                                "chunks": 2, "warnings": []}
                    self.documents.append(document)
                    return httpx.Response(201, json=document)
                return httpx.Response(200, json=[{**doc,
                    "training_eligible": self.conv["logging_enabled"] and self.conv["training_enabled"],
                    "retention": "saved" if self.conv["logging_enabled"] else "session"} for doc in self.documents])
            if path == base + "/search":
                return httpx.Response(200, json={"matches": [{"citation": "knowledge_1 chunk 0 (page 1)",
                    "name": "My source notes.txt", "text": "The calibrated value is 42."}]})
            if request.method == "DELETE":
                self.documents = [doc for doc in self.documents if doc["id"] != path.rsplit("/", 1)[-1]]
                return httpx.Response(200, json={"deleted": True})
            chunk = int(request.url.params.get("chunk", "0"))
            return httpx.Response(200, json={"chunks": [{"chunk": chunk, "label": f"page {chunk + 1}",
                "authority": "source_text", "text": "The calibrated value is 42."}],
                "next_chunk": 1 if chunk == 0 else None})
        response = super().handle(request)
        if path == "/conversations/conv_test" and request.method == "PATCH" and not self.conv["logging_enabled"]:
            self.conv["training_enabled"] = False
            return httpx.Response(200, json=self.conv)
        return response


def last_notice(app):
    return list(app.query(".system"))[-1].render().plain


@pytest.mark.asyncio
async def test_knowledge_upload_link_retrieval_removal_and_consent_in_one_session(tmp_path):
    source = tmp_path / "My source notes.txt"
    source_bytes = "The calibrated value is 42.\nLiteral [red]source[/red].\n".encode()
    source.write_bytes(source_bytes)
    backend = KnowledgeBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        await submit(app, pilot, f'/attach "{source}"')
        base = "/conversations/conv_test/knowledge"
        upload = next(payload for method, path, payload in backend.requests if method == "POST" and path == base)
        assert upload["filename"] == source.name
        assert base64.b64decode(upload["content_base64"]) == source_bytes
        assert "path" not in upload and upload["video_frames"] == 8
        assert app.conversation["knowledge_count"] == 1
        assert "Training is OFF" in last_notice(app)
        assert "KNOWLEDGE: 1" in app.query_one("#session").render().plain

        # An attachment is source material; it cannot masquerade as a completed
        # assistant answer eligible for feedback.
        await submit(app, pilot, "/feedback good")
        assert "Finish an assistant reply" in last_notice(app)
        assert not any(path == "/feedback" for _, path, _ in backend.requests)

        await submit(app, pilot, "/privacy training on")
        await submit(app, pilot, '/link "https://example.org/reference?q=calibrated%20value" --frames 3')
        linked = [payload for method, path, payload in backend.requests if method == "POST" and path == base][-1]
        assert linked == {"url": "https://example.org/reference?q=calibrated%20value", "video_frames": 3}
        assert "Included as a source for sleep" in last_notice(app)
        assert app.conversation["knowledge_count"] == 2

        await submit(app, pilot, "/knowledge list")
        assert "knowledge_1" in last_notice(app) and "TRAIN ON" in last_notice(app)
        await submit(app, pilot, '/knowledge search "calibrated value"')
        assert (base + "/search", {"query": "calibrated value", "limit": "5"}) in backend.queries
        assert "knowledge_1 chunk 0 (page 1)" in last_notice(app)
        await submit(app, pilot, "/knowledge read knowledge_1")
        assert "Next: /knowledge read knowledge_1 1" in last_notice(app)
        await submit(app, pilot, "/knowledge read knowledge_1 1")
        assert (base + "/knowledge_1", {"chunk": "1"}) in backend.queries
        assert "source_text" in last_notice(app) and "Next:" not in last_notice(app)

        await submit(app, pilot, "/privacy logging off")
        await submit(app, pilot, "/knowledge list")
        assert not app.conversation["logging_enabled"] and not app.conversation["training_enabled"]
        assert "session // TRAIN OFF" in last_notice(app)
        await submit(app, pilot, "/knowledge remove knowledge_1")
        assert ("DELETE", base + "/knowledge_1", {}) in backend.requests
        assert app.conversation["knowledge_count"] == 1
        assert "derived training evidence was invalidated" in last_notice(app)
        assert app.crash_report is None and not app.busy


@pytest.mark.asyncio
async def test_knowledge_argument_errors_and_sleep_readiness_keep_drafts_and_read_commands(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source")
    backend = KnowledgeBackend()
    app = LuciferApp(client=APIClient(transport=httpx.MockTransport(backend.handle)))
    async with app.run_test(size=(120, 42)) as pilot:
        for command in (f'/attach "{source}" --frames 25',
                        "/link https://example.org --frames wrong",
                        "/link https://example.org --frames 0",
                        "/attach 'unfinished path"):
            await submit(app, pilot, command)
            assert "Error:" in last_notice(app)
            assert not app.busy and app.crash_report is None
        assert not any("/knowledge" in path for _, path, _ in backend.requests)
        await submit(app, pilot, f'/attach "{source}"')
        base = "/conversations/conv_test/knowledge"
        count = len([1 for _, path, _ in backend.requests if path.startswith(base)])
        await submit(app, pilot, "/knowledge read knowledge_1 -1")
        assert "nonnegative integer" in last_notice(app)
        assert len([1 for _, path, _ in backend.requests if path.startswith(base)]) == count

        backend.status["learning_busy"] = True
        await app.refresh_panels()
        assert app.learning_busy
        mutations = [(method, path) for method, path, _ in backend.requests if method in {"POST", "DELETE"} and path.startswith(base)]
        for draft in (f'/attach "{source}"', "/link https://example.org/new", "/knowledge remove knowledge_1"):
            app.query_one("#prompt", Input).value = draft
            await submit(app, pilot, draft)
            assert app.query_one("#prompt", Input).value == draft
            assert not app.busy
        assert [(method, path) for method, path, _ in backend.requests if method in {"POST", "DELETE"} and path.startswith(base)] == mutations
        await submit(app, pilot, "/knowledge list")
        assert "knowledge_1" in last_notice(app)
        await submit(app, pilot, "/knowledge search source")
        await submit(app, pilot, "/knowledge read knowledge_1")
        assert "The calibrated value is 42." in last_notice(app)
        assert app.learning_busy and app.crash_report is None
