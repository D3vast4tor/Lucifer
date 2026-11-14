"""Consent withdrawal cancels affected Meditation work through restoration."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from test_meditation_api import application


@pytest.mark.parametrize("action", ["training_off", "logging_off", "delete_chat", "delete_document"])
def test_source_withdrawal_cancels_owned_meditation_and_awaits_restoration(tmp_path, action):
    app = application(tmp_path)
    with TestClient(app) as client:
        cid = client.post("/conversations", json={"training_enabled": True}).json()["id"]
        document = None
        if action == "delete_document":
            document = client.post(f"/conversations/{cid}/knowledge", json={"filename": "source.txt",
                "content_base64": base64.b64encode(b"Consented source material.").decode()}).json()
        run_id = "meditation_owned_privacy"
        store = app.state.store
        store.put("meditation_runs", {"id": run_id, "status": "running", "source_conversation_ids": [cid]})
        store.put("adapters", {"id": "old_expert", "status": "active", "base_model_id": "demo",
                               "source_conversation_ids": [cid]})
        store.set_setting("active_adapter", "old_expert")
        cancelled, release_restore = Event(), Event()
        async def owned():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert store.get("meditation_runs", run_id)["source_invalidated"]
                store.update("meditation_runs", run_id, status="cancelled",
                    progress={"stage": "restoring", "terminal": True, "restored": False})
                cancelled.set()
                while not release_restore.is_set():
                    await asyncio.sleep(.005)
                store.update("meditation_runs", run_id,
                    progress={"stage": "cancelled", "terminal": True, "restored": True})
                raise
        async def launch():
            app.state.meditation.tasks[run_id] = asyncio.create_task(owned())
            await asyncio.sleep(0)
        client.portal.call(launch)
        def withdraw():
            if action == "delete_chat":
                return client.delete(f"/conversations/{cid}")
            if action == "delete_document":
                return client.delete(f"/conversations/{cid}/knowledge/{document['id']}")
            return client.patch(f"/conversations/{cid}", json={
                "training_enabled" if action == "training_off" else "logging_enabled": False})
        with ThreadPoolExecutor(max_workers=1) as executor:
            response_future = executor.submit(withdraw)
            try:
                assert cancelled.wait(2), "Consent withdrawal did not cancel its affected owned Meditation task"
                assert not response_future.done(), "Withdrawal returned before the model restoration finished"
                status = client.get("/system/status").json()
                assert status["meditation_busy"] is True and status["learning_busy"] is True
                assert store.setting("active_adapter") == "old_expert"
            finally:
                release_restore.set()
            response = response_future.result(timeout=3)
        assert response.status_code == 200, response.text
        assert client.get("/system/status").json()["meditation_busy"] is False
        assert store.get("meditation_runs", run_id)["progress"]["restored"]
        assert store.setting("active_adapter") is None
        assert store.get("adapters", "old_expert")["source_revoked"]


def test_unrelated_consent_withdrawal_does_not_cancel_meditation(tmp_path):
    app = application(tmp_path)
    with TestClient(app) as client:
        source = client.post("/conversations", json={"training_enabled": True}).json()["id"]
        unrelated = client.post("/conversations", json={"training_enabled": True}).json()["id"]
        ident, cancelled = "meditation_unrelated", Event()
        app.state.store.put("meditation_runs", {"id": ident, "status": "running", "source_conversation_ids": [source]})
        async def owned():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def launch():
            app.state.meditation.tasks[ident] = asyncio.create_task(owned())
            await asyncio.sleep(0)
        client.portal.call(launch)
        response = client.patch(f"/conversations/{unrelated}", json={"training_enabled": False})
        assert response.status_code == 200
        assert not cancelled.is_set() and app.state.meditation.busy
        assert not app.state.store.get("meditation_runs", ident).get("source_invalidated")
        async def cleanup():
            task = app.state.meditation.tasks[ident]
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        client.portal.call(cleanup)


@pytest.mark.parametrize("ordinary_busy", ["agent", "sleep_task", "sleep_record"])
def test_source_withdrawal_preserves_agent_and_ordinary_sleep_busy_guards(tmp_path, monkeypatch, ordinary_busy):
    app = application(tmp_path)
    with TestClient(app) as client:
        cid = client.post("/conversations", json={"training_enabled": True}).json()["id"]
        monkeypatch.setattr(app.state, "meditation", SimpleNamespace(busy=True))
        if ordinary_busy == "agent":
            monkeypatch.setattr(app.state, "agent", SimpleNamespace(busy=True))
        elif ordinary_busy == "sleep_task":
            monkeypatch.setattr(app.state, "learning", SimpleNamespace(busy=True))
        else:
            app.state.store.put("sleep_runs", {"id": "ordinary_sleep", "status": "training"})
        response = client.patch(f"/conversations/{cid}", json={"training_enabled": False})
        assert response.status_code == 409
        assert app.state.store.get("conversations", cid)["training_enabled"]


@pytest.mark.parametrize("changes", [{"title": "rename"}, {"training_enabled": True},
    {"training_enabled": False, "title": "rename"}])
def test_revocation_exception_does_not_allow_other_chat_mutations(tmp_path, monkeypatch, changes):
    app = application(tmp_path)
    with TestClient(app) as client:
        cid = client.post("/conversations", json={"training_enabled": True}).json()["id"]
        monkeypatch.setattr(app.state, "meditation", SimpleNamespace(busy=True))
        response = client.patch(f"/conversations/{cid}", json=changes)
        assert response.status_code == 409
        assert app.state.store.get("conversations", cid)["training_enabled"]
