"""Completed commands are reused only across the same tracked source revision."""
import asyncio
import hashlib
import sys

import pytest

from lucifer.agent import AgentService, digest
from lucifer.chat import ChatService
from test_agent import Manager, call, service, turn
from test_agent import setup as agent_setup


@pytest.fixture
def setup(tmp_path):
    yield from agent_setup.__wrapped__(tmp_path)


def command_and_edit(setup):
    original = (setup[2] / "calc.py").read_text()
    arguments = {"argv": [sys.executable, "-c",
        "from pathlib import Path; p=Path('command-count.txt'); "
        "p.write_text((p.read_text() if p.exists() else '')+'run\\n'); "
        "print(Path('calc.py').read_text())"]}
    edit = {"path": "calc.py", "old_text": "* 2", "new_text": "* 3",
            "expected_sha256": hashlib.sha256(original.encode()).hexdigest()}
    return arguments, edit


@pytest.mark.asyncio
async def test_command_reuses_same_inputs_but_runs_fresh_after_explicit_file_edit(setup):
    command, edit = command_and_edit(setup)
    agent, _, chat, run = await service(setup, [
        turn(call("before", "run_command", **command)),
        turn(call("same-before", "run_command", **command)),
        turn(call("edit", "edit_file", **edit)),
        turn(call("after", "run_command", **command)),
        turn(call("same-after", "run_command", **command)),
        turn(answer="Validated the updated source with a fresh command result."),
    ], access="computer", auto_approve_commands=True)
    chat.patch(run["conversation_id"], {"logging_enabled": True})
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        state = agent.states[run["id"]]
        assert "* 2" in state.calls["before"][1].content["stdout"]
        assert "* 3" in state.calls["after"][1].content["stdout"]
        assert state.calls["before"][1] is state.calls["same-before"][1]
        assert state.calls["after"][1] is state.calls["same-after"][1]
        assert (setup[2] / "command-count.txt").read_text().splitlines() == ["run", "run"]
        operations = setup[1].list("agent_operations", run_id=run["id"])
        commands = [operation for operation in operations if operation["effect"] == "command"]
        assert len(operations) == 3 and {operation["command_input_revision"] for operation in commands} == {0, 1}
        assert len({operation["fingerprint"] for operation in commands}) == 2
        assert state.command_input_revision == 1
        checkpoint = setup[1].get("agent_checkpoints", run["id"])["checkpoint"]
        assert checkpoint["state"]["command_input_revision"] == 1
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_new_command_source_revision_requires_a_new_concrete_approval(setup):
    command, edit = command_and_edit(setup)
    agent, _, _, run = await service(setup, [
        turn(call("before", "run_command", **command)),
        turn(call("edit", "edit_file", **edit)),
        turn(call("after", "run_command", **command)),
        turn(answer="Both source revisions were explicitly approved."),
    ], access="computer")
    approved = []
    async def decide():
        while not agent.tasks[run["id"]].done():
            shown = agent.get(run["id"])
            if shown["status"] == "awaiting_approval":
                approval = shown["approval"]
                assert approval["approval_id"] not in approved
                assert approval["arguments"]["argv"] == command["argv"]
                approved.append(approval["approval_id"])
                await agent.approve(run["id"], approval["approval_id"], True)
            await asyncio.sleep(.01)
        await agent.tasks[run["id"]]
    try:
        await asyncio.wait_for(decide(), 5)
        assert agent.get(run["id"])["status"] == "completed"
        assert len(approved) == 2
        assert (setup[2] / "command-count.txt").read_text().splitlines() == ["run", "run"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_restored_revision_and_completion_receipts_do_not_reexecute_commands(setup):
    command, edit = command_and_edit(setup)
    agent, _, chat, run = await service(setup, [
        turn(call("edit", "edit_file", **edit)),
        turn(call("validated", "run_command", **command)),
        turn(answer="Completed command receipt is saved."),
    ], access="computer", auto_approve_commands=True)
    chat.patch(run["conversation_id"], {"logging_enabled": True})
    await agent.tasks[run["id"]]
    await agent.close()
    # Simulate a checkpoint taken before its final counter was published. The
    # durable completed-operation record must recover the input revision too.
    record = setup[1].get("agent_checkpoints", run["id"])
    checkpoint = record["checkpoint"]
    checkpoint["state"]["command_input_revision"] = 0
    setup[1].put("agent_checkpoints", {"id": run["id"], "checkpoint": checkpoint, "sha256": digest(checkpoint)})
    manager = Manager([])
    restored = AgentService(setup[0], setup[1], manager, ChatService(setup[1], manager))
    try:
        state = restored._restore_state(setup[1].get("agent_runs", run["id"]))
        assert state.command_input_revision == 1
        cached = await restored._tool(run["id"], {"id": "new-wire-id", "name": "run_command", "arguments": command}, {"run_command"})
        assert "* 3" in cached.content["stdout"]
        assert (setup[2] / "command-count.txt").read_text().splitlines() == ["run"]
        assert len(setup[1].list("agent_operations", run_id=run["id"])) == 2
        assert manager.inputs == []
    finally:
        await restored.close()
