"""Repeated work is steered once, then paused without losing the mission."""
import json
import os

import pytest

from lucifer.providers import canonical_messages
from test_agent import call, service, turn
from test_agent import setup as agent_setup


@pytest.fixture
def setup(tmp_path):
    yield from agent_setup.__wrapped__(tmp_path)


@pytest.mark.asyncio
async def test_repeated_cached_reads_warn_then_pause_with_retained_results(setup):
    agent, manager, _, run = await service(setup, [
        *[turn(call(f"read-{n}", "read_file", path="calc.py")) for n in range(6)],
        turn(answer="This must not be reached while stalled."),
    ])
    try:
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "paused" and result["stop_reason"] == "no_progress"
        assert result["resumable"] and result["no_progress_rounds"] == 5
        assert len(manager.inputs) == 6 and len(setup[1].list("agent_operations")) == 1
        state = agent.states[run["id"]]
        canonical_messages(state.messages)
        assert len(state.calls) == 6 and state.turn_inputs
        warnings = [event for event in state.events if event["event"] == "progress_warning"]
        assert len(warnings) == 1
        assert any("Progress check:" in str(message.get("content")) for message in manager.inputs[4][0])
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_training_knowledge_ingestion_does_not_reset_repeated_read_detection(setup):
    from lucifer.chat import ChatService
    from types import SimpleNamespace
    chat = ChatService(setup[1], SimpleNamespace())
    cid = chat.create(training_enabled=True)["id"]
    agent, _, _, run = await service(setup, [
        *[turn(call(f"read-{n}", "read_file", path="calc.py")) for n in range(6)],
        turn(answer="This must not be reached while stalled."),
    ], conversation_id=cid)
    try:
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "paused" and result["stop_reason"] == "no_progress"
        assert len(setup[1].list("agent_operations")) == 1
        assert len(setup[1].list("knowledge_documents")) == 1
        assert len(setup[1].list("knowledge_interactions")) == 1
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_new_evidence_after_warning_clears_stall_and_finishes(setup):
    (setup[2] / "new.py").write_text("new_fact = 42\n")
    agent, _, _, run = await service(setup, [
        *[turn(call(f"read-{n}", "read_file", path="calc.py")) for n in range(4)],
        turn(call("fresh", "read_file", path="new.py")),
        turn(answer="Used the new evidence and completed the goal."),
    ])
    try:
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "completed"
        assert result["no_progress_rounds"] == 0 and not result["progress_warning"]
        assert sum(event["event"] == "progress_warning" for event in agent.states[run["id"]].events) == 1
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_external_file_changes_and_fresh_results_are_progress(setup):
    def changed(number):
        def emit():
            (setup[2] / "calc.py").write_text(f"def compute(value):\n    return value * {number}\n")
            # Explicit versions avoid timestamp coalescing on coarse CI filesystems.
            os.utime(setup[2] / "calc.py", ns=(number + 1, number + 1))
            return turn(call(f"read-{number}", "read_file", path="calc.py"))
        return emit
    agent, _, _, run = await service(setup, [*[changed(n) for n in range(10)], turn(answer="Observed all ten versions.")])
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        assert len(setup[1].list("agent_operations")) == 10
        assert not any(event["event"] == "progress_warning" for event in agent.states[run["id"]].events)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_alternating_repeated_errors_do_not_evade_guard(setup):
    agent, manager, _, run = await service(setup, [
        turn(call(f"missing-{n}", "read_file", path=f"missing-{n % 2}.py")) for n in range(9)
    ])
    try:
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "paused" and result["stop_reason"] == "no_progress"
        assert len(manager.inputs) == 7
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_repeated_reasoning_only_output_pauses_without_persisting_reasoning(setup):
    private = "ACTUAL_PRIVATE_REPEATING_PROVIDER_REASONING"
    limited = [("reasoning", {"content": private}), ("finish", {"finish_reason": "length"})]
    agent, _, chat, run = await service(setup, [limited for _ in range(6)], thinking="high")
    chat.patch(run["conversation_id"], {"logging_enabled": True})
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "paused"
        assert agent.get(run["id"])["reasoning_characters"] == len(private) * 6
        checkpoint = setup[1].get("agent_checkpoints", run["id"])
        assert private not in json.dumps(checkpoint)
        assert "progress_seen" not in checkpoint["checkpoint"]["state"]
        assert private not in json.dumps(setup[1].list("agent_events"))
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_explicit_correction_recovers_a_paused_mission(setup):
    agent, _, _, run = await service(setup, [
        *[turn(call(f"read-{n}", "read_file", path="calc.py")) for n in range(6)],
        turn(answer="The corrected goal is complete."),
    ])
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "paused"
        await agent.submit_input(run["id"], "Use the existing result and finish now.")
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        assert len(setup[1].list("agent_operations")) == 1
    finally:
        await agent.close()
