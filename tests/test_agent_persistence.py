"""Explicit restoration preserves public context and effect receipts, never thinking."""
import asyncio
import json
import sys

import pytest
from fastapi.testclient import TestClient

from lucifer.agent import AgentService
from lucifer.api import create_app
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.schemas import AgentRequest
from lucifer.store import Store
from test_agent import Manager, call, turn
from test_agent_sessions import SessionManager


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "calc.py").write_text("answer = 42\n")
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "missing.yaml", agent_workspace=workspace)
    store = Store(settings.data_dir / "state.db")
    yield settings, store, workspace
    store.close()


async def _logged(setup, turns, **options):
    settings, store, workspace = setup
    manager = SessionManager(turns)
    chat = ChatService(store, manager)
    conversation = chat.create(title="Saved mission", logging_enabled=True)
    service = AgentService(settings, store, manager, chat)
    run = await service.start(AgentRequest(goal="Inspect the project", conversation_id=conversation["id"],
                                          workspace=str(workspace), **options))
    return service, manager, chat, run


@pytest.mark.asyncio
async def test_logged_paused_session_restores_once_without_repeating_completed_write(setup):
    action = {"path": "created.py", "content": "answer = 42\n", "expected_sha256": None}
    service, manager, _, run = await _logged(setup, [turn(call("write", "write_file", **action)),
        {"reasoning": "PRIVATE_REASONING_NEVER_SAVED_7281", "partial": "Public partial answer"}], access="write", thinking="high")
    await manager.entered.wait()
    await service.stop(run["id"])
    await service.close()
    assert (setup[2] / "created.py").read_text() == "answer = 42\n"
    checkpoint = setup[1].get("agent_checkpoints", run["id"])
    assert "PRIVATE_REASONING_NEVER_SAVED" not in json.dumps(checkpoint)
    assert "PRIVATE_REASONING_NEVER_SAVED" not in setup[1].path.read_bytes().decode(errors="ignore")
    replacement = Manager([turn(call("different-wire-id", "write_file", **action)), turn(answer="Finished without rewriting.")])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        assert not restored.states and not restored.tasks and not replacement.inputs
        history = restored.history()
        assert history[0]["restore_available"] and not history[0]["resumable"]
        session = restored.session(run["id"])
        assert any(message["content"] == "Public partial answer" for message in session["messages"])
        assert not restored.states and not replacement.inputs  # GET history never rehydrates or executes.
        # A follow-up explicitly abandons the partial answer and proposes the
        # same write. Its completed receipt must be reused after restart.
        await restored.submit_input(run["id"], "Finish and check the existing created file.")
        await restored.tasks[run["id"]]
        assert restored.get(run["id"])["status"] == "completed"
        assert len(setup[1].list("agent_operations", run_id=run["id"])) == 1
        assert any(event["event"] == "tool_result" and event["data"].get("reused") for event in restored.states[run["id"]].events)
        assert not any("reasoning_content" in message for message in replacement.inputs[0][0])
        assert restored.get(run["id"])["reasoning_restored"] is False
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_private_session_history_explains_unavailable_transcript_after_restart(setup):
    manager = Manager([turn(answer="Private result")])
    chat = ChatService(setup[1], manager)
    service = AgentService(setup[0], setup[1], manager, chat)
    conversation = chat.create(logging_enabled=False)
    run = await service.start(AgentRequest(goal="Private mission", workspace=str(setup[2]), conversation_id=conversation["id"]))
    await service.tasks[run["id"]]
    await service.close()
    restored = AgentService(setup[0], setup[1], manager, ChatService(setup[1], manager))
    try:
        row = restored.history()[0]
        assert row["conversation_id"] == run["conversation_id"]
        assert not row["logging_enabled"] and not row["transcript_available"] and not row["restore_available"]
        assert row["transcript_unavailable_reason"] == "logging_disabled"
        assert restored.session(run["id"])["messages"] == []
        assert setup[1].list("agent_checkpoints") == []
        with pytest.raises(ValueError, match="logging_disabled"):
            await restored.resume(run["id"])
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_checkpoint_only_history_exposes_original_direct_pc_auto_image_and_mcp_scope(setup):
    manager = SessionManager([{"reasoning": "Transient planning"}])
    chat = ChatService(setup[1], manager)
    conversation = chat.create(logging_enabled=True)
    service = AgentService(setup[0], setup[1], manager, chat)
    service.tools.mcp_status["selected-docs"] = {"status": "ready"}
    run = await service.start(AgentRequest(goal="Inspect using the explicitly selected scope", workspace=str(setup[2]),
        conversation_id=conversation["id"], access="computer", auto_approve_commands=True, allow_images=True,
        allow_mcp=True, mcp_servers=["selected-docs"], thinking="max", max_tokens=12345, stop_when_done=False,
        continue_output=False))
    await manager.entered.wait()
    await service.stop(run["id"])
    await service.close()
    replacement = Manager([])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        shown = restored.session(run["id"])["run"]
        assert shown["access"] == "computer" and shown["auto_approve_commands"] is True
        assert shown["allow_images"] is True and shown["images"] == []
        assert shown["allow_mcp"] is True and shown["mcp_servers"] == ["selected-docs"]
        assert shown["thinking"] == "max" and shown["max_tokens"] == 12345
        assert shown["stop_when_done"] is False and shown["continue_output"] is False
        assert shown["require_approval"] is False and shown["workspace"] == str(setup[2])
        assert not restored.states and not replacement.inputs
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_logging_revocation_purges_checkpoint_results_but_preserves_ram(setup):
    service, manager, chat, run = await _logged(setup, [turn(call("read", "read_file", path="calc.py")), turn(answer="Read complete.")])
    try:
        await service.tasks[run["id"]]
        assert setup[1].list("agent_checkpoints")
        assert setup[1].list("agent_operations")[0].get("result")
        chat.patch(run["conversation_id"], {"logging_enabled": False})
        assert setup[1].list("agent_checkpoints") == []
        assert not setup[1].list("agent_operations")[0].get("result")
        assert service.get(run["id"])["session_available"]
        assert service.session(run["id"])["messages"][-1]["content"] == "Read complete."
        assert sum(item["content"] == "Read complete." for item in service.session(run["id"])["messages"]) == 1
        assert run["id"] in service.states
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_explicit_save_checkpoints_an_already_paused_private_session(setup):
    manager = SessionManager([{"partial": "Public saved fragment", "reasoning": "NEVER_SAVE_THIS_THINKING"}])
    chat = ChatService(setup[1], manager)
    service = AgentService(setup[0], setup[1], manager, chat)
    conversation = chat.create(logging_enabled=False)
    run = await service.start(AgentRequest(goal="Private until saved", workspace=str(setup[2]), thinking="high", conversation_id=conversation["id"]))
    try:
        await manager.entered.wait()
        await service.stop(run["id"])
        assert setup[1].list("agent_checkpoints") == []
        chat.patch(run["conversation_id"], {"logging_enabled": True})
        checkpoint = setup[1].get("agent_checkpoints", run["id"])
        assert "Public saved fragment" in json.dumps(checkpoint)
        assert "NEVER_SAVE_THIS_THINKING" not in json.dumps(checkpoint)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unconfirmed_control_claim_after_restart_is_reported_interrupted_without_reapplying(setup):
    from lucifer.agent import digest
    service, manager, _, run = await _logged(setup, [{"reasoning": "Planning"}])
    await manager.entered.wait()
    await service.stop(run["id"])
    await service.close()
    checkpoint = setup[1].get("agent_checkpoints", run["id"])
    address = digest({"kind": "input", "key": "crashed-correction"})
    checkpoint["checkpoint"]["state"]["control_requests"][address] = {
        "fingerprint": digest({"content": "Uncommitted correction", "interrupt": True, "options": {}}), "status": "accepted"}
    checkpoint["sha256"] = digest(checkpoint["checkpoint"])
    setup[1].put("agent_checkpoints", checkpoint)
    replacement = Manager([])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        assert "application was not confirmed" in restored.session(run["id"])["run"]["control_notice"]
        replay = await restored.submit_input(run["id"], "Uncommitted correction", idempotency_key="crashed-correction")
        assert replay["idempotent_replay"] and replay["control_status"] == "interrupted"
        assert "application was not confirmed" in replay["control_notice"]
        assert not replacement.inputs and not restored.tasks
        assert not any(message.get("content") == "Uncommitted correction" for message in restored.states[run["id"]].messages)
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_result", [True, False])
async def test_crash_after_completed_receipt_imports_result_or_marks_unknown_never_replays(setup, retain_result):
    action = {"path": "crash-created.py", "content": "original\n", "expected_sha256": None}
    service, manager, _, run = await _logged(setup, [turn(call("write", "write_file", **action)), {"reasoning": "Waiting after write"}], access="write")
    execute = service.tools.execute
    before_effect = []
    async def checked_execute(name, arguments, context):
        prepared = setup[1].get("agent_checkpoints", run["id"])["checkpoint"]["state"]
        assert prepared["pending_calls"][0]["id"] == "write"
        assert "write" in prepared["operation_ids"]
        assert not (setup[2] / "crash-created.py").exists()
        before_effect.append(True)
        return await execute(name, arguments, context)
    service.tools.execute = checked_execute
    await manager.entered.wait()
    assert before_effect == [True]
    await service.stop(run["id"])
    await service.close()
    checkpoint = setup[1].get("agent_checkpoints", run["id"])
    body = checkpoint["checkpoint"]
    saved_call = {"id": "write", "wire_id": "write", "name": "write_file", "arguments": action}
    body["state"].update(calls={}, results={}, workspace_revision=0, pending_calls=[saved_call], pending_index=0,
                          operation_ids={"write": setup[1].list("agent_operations")[0]["id"]})
    # Emulate the durable checkpoint immediately before the effect, followed by
    # its completed journal receipt, with no later checkpoint committed.
    body["state"]["messages"] = [message for message in body["state"]["messages"] if message.get("role") != "tool"]
    from lucifer.agent import digest
    checkpoint["sha256"] = digest(body)
    setup[1].put("agent_checkpoints", checkpoint)
    operation = setup[1].list("agent_operations")[0]
    if not retain_result:
        setup[1].put("agent_operations", {key: value for key, value in operation.items() if key not in {"result", "result_sha256"}})
    replacement = Manager([turn(call("new-wire-id", "write_file", **action)), turn(answer="No repeated effect.")])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        await restored.submit_input(run["id"], "Finish using the existing receipt.")
        await restored.tasks[run["id"]]
        state = restored.states[run["id"]]
        assert len(setup[1].list("agent_operations")) == 1
        assert (setup[2] / "crash-created.py").read_text() == "original\n"
        assert state.workspace_revision == 1
        if retain_result:
            assert any(event["event"] == "tool_result" and event["data"].get("reused") for event in state.events)
        else:
            assert any("uncertain" in message.get("content", "") for message in replacement.inputs[0][0] if message.get("role") == "tool")
        assert "write" in state.wire_ids and "new-wire-id" in state.wire_ids
        assert restored.get(run["id"])["last_sequence"] > body["state"]["sequence"]
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_restore_expires_concrete_approval_and_new_command_needs_new_decision(setup):
    target = setup[2].parent / "never-approved.txt"
    argv = [sys.executable, "-c", f"from pathlib import Path;Path({str(target)!r}).write_text('bad')"]
    service, _, _, run = await _logged(setup, [turn(call("first", "run_command", argv=argv))], access="computer")
    async with asyncio.timeout(3):
        while service.get(run["id"])["status"] != "awaiting_approval":
            await asyncio.sleep(.01)
    old = service.get(run["id"])["pending_approval"]
    await service.stop(run["id"])
    await service.close()
    replacement = Manager([turn(call("second", "run_command", argv=argv)), turn(answer="New decision respected.")])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        await restored.resume(run["id"])
        async with asyncio.timeout(3):
            while restored.get(run["id"])["status"] != "awaiting_approval":
                await asyncio.sleep(.01)
        current = restored.get(run["id"])["pending_approval"]
        assert current != old and not target.exists()
        assert restored.get(run["id"])["access"] == "computer"
        with pytest.raises(ValueError, match="expired"):
            await restored.approve(run["id"], old, True)
        await restored.approve(run["id"], current, False)
        await restored.tasks[run["id"]]
        assert not target.exists()
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "profile", "scope", "tool_schema", "digest"])
async def test_checkpoint_restore_fails_closed_when_binding_changes(setup, change):
    service, manager, _, run = await _logged(setup, [{"reasoning": "RAM-only"}])
    await manager.entered.wait()
    await service.stop(run["id"])
    await service.close()
    replacement = Manager([turn(answer="Must not generate")])
    restored = AgentService(setup[0], setup[1], replacement, ChatService(setup[1], replacement))
    try:
        if change == "model":
            replacement.loaded["id"] = "other"
        elif change == "profile":
            replacement.capabilities.record("model", "changed-revision", declared={"chat_completions": True, "tools": True})
        elif change == "scope":
            setup[1].update("agent_runs", run["id"], access="computer")
        elif change == "tool_schema":
            original = restored.tools.specs
            restored.tools.specs = lambda context: original(context)[:-1]
        else:
            setup[1].update("agent_checkpoints", run["id"], sha256="wrong")
        with pytest.raises(ValueError):
            await restored.resume(run["id"])
        assert not restored.tasks and not replacement.inputs
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_checkpoint_size_limit_discards_whole_unit_without_truncation(setup, monkeypatch):
    import lucifer.agent as module
    monkeypatch.setattr(module, "MAX_CHECKPOINT_BYTES", 64)
    service, _, _, run = await _logged(setup, [turn(answer="A public final answer.")])
    try:
        await service.tasks[run["id"]]
        assert setup[1].list("agent_checkpoints") == []
        assert setup[1].get("agent_runs", run["id"])["checkpoint_unavailable_reason"] == "checkpoint_size_limit"
        assert service.get(run["id"])["answer"] == "A public final answer."
    finally:
        await service.close()


def test_history_api_lists_private_old_run_and_saved_transcript_without_effects(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "missing.yaml")
    app = create_app(settings, autoload="demo")
    with TestClient(app) as client:
        private = app.state.chat.create(title="Unavailable old mission", logging_enabled=False)
        app.state.store.put("agent_runs", {"id": "old-private", "conversation_id": private["id"], "status": "paused"})
        saved = app.state.chat.create(title="Saved mission", logging_enabled=True)
        app.state.store.put("messages", {"id": "saved-user", "conversation_id": saved["id"], "role": "user", "content": "Saved public request"})
        app.state.store.put("agent_runs", {"id": "old-saved", "conversation_id": saved["id"], "status": "interrupted"})
        app.state.store.put("agent_operations", {"id": "image-result", "run_id": "old-saved", "status": "completed",
            "result": {"content": {"path": "preview.png"}, "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE_IMAGE_BYTES"}}]}})
        rows = {row["id"]: row for row in client.get("/agent/history").json()}
        assert rows["old-private"]["transcript_unavailable_reason"] == "logging_disabled"
        assert rows["old-saved"]["transcript_available"] and not rows["old-saved"]["restore_available"]
        transcript = client.get("/agent/runs/old-saved/session").json()
        assert transcript["messages"] == [{"id": "saved-user", "role": "user", "content": "Saved public request"}]
        assert transcript["operations"][0]["result"] == {"content": {"path": "preview.png"}}
        assert "PRIVATE_IMAGE_BYTES" not in json.dumps(transcript)
        assert app.state.store.get("agent_operations", "image-result")["result"]["images"]  # Internal replay payload is untouched.
        assert not app.state.agent.states and not app.state.agent.tasks
