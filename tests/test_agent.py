import asyncio
import json

import pytest

from lucifer.agent import AgentService
from lucifer.capabilities import CapabilityRegistry
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.learning import build_dataset
from lucifer.providers import canonical_messages
from lucifer.schemas import AgentRequest
from lucifer.store import Store


def call(ident, name, **arguments):
    return "tool_call", {"id": ident, "name": name, "arguments": arguments}


def turn(*calls, answer=None, reasoning=None):
    events = []
    if reasoning:
        events.append(("reasoning", {"content": reasoning}))
    events.extend(calls)
    if answer:
        events.append(("delta", {"content": answer}))
    events.extend([("usage", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}),
                   ("finish", {"finish_reason": "tool_calls" if calls else "stop"})])
    return events


class Manager:
    def __init__(self, turns):
        self.turns = turns
        self.inputs = []
        self.counts = []
        self.capabilities = CapabilityRegistry()
        self.capabilities.record("model", "revision", declared={"chat_completions": True,
            "anthropic_messages": True, "responses": True, "tools": True, "vision": True,
            "reasoning": True, "structured_output": True})
        self.loaded = {"id": "model", "display_name": "Lucifer", "effective_context_length": 65536}
        self.count = 100
        self.cancelled = False

    def current(self):
        return self.loaded

    async def count_tokens(self, messages, tools, **kwargs):
        self.counts.append((messages.copy(), tools, kwargs))
        return {"input_tokens": self.count, "method": "conservative_estimate", "estimated": True}

    async def agent_events(self, messages, tools, **kwargs):
        canonical_messages(messages)
        index = len(self.inputs)
        self.inputs.append((messages.copy(), tools, kwargs))
        selected = self.turns[index]
        if callable(selected):
            selected = selected()
        if selected == "block":
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        for item in selected:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text("def compute(value):\n    return value * 2\n")
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml", agent_workspace=root)
    store = Store(settings.data_dir / "state.db")
    yield settings, store, root
    store.close()


async def service(setup, turns, **request):
    settings, store, root = setup
    manager = Manager(turns)
    chat = ChatService(store, manager)
    agent = AgentService(settings, store, manager, chat)
    if "conversation_id" not in request:
        request["conversation_id"] = chat.create(logging_enabled=False)["id"]
    run = await agent.start(AgentRequest(goal="Find compute and cite its file/line", workspace=str(root), **request))
    return agent, manager, chat, run


@pytest.mark.asyncio
async def test_real_tools_multistep_final_and_no_reasoning_or_traces_in_learning(setup):
    agent, manager, chat, run = await service(setup, [
        turn(call("find", "search_text", query="def compute"), reasoning="PRIVATE_REASONING_3918"),
        turn(call("read", "read_file", path="calc.py")),
        turn(answer="`compute` is defined at calc.py:1 and doubles its input."),
    ], thinking="high")
    try:
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "completed" and "calc.py:1" in result["answer"]
        assert len(manager.inputs) == 3
        assert any(m.get("tool_call_id") == "find" for m in manager.inputs[1][0])
        transcript = chat.get(run["conversation_id"])["messages"]
        assert len(transcript) == 2 and transcript[-1]["source"] == "agent_final"
        assert setup[1].list("messages") == []
        assert "PRIVATE_REASONING_3918" not in json.dumps(setup[1].list("agent_events"))
        assert "return value" not in json.dumps(setup[1].list("agent_operations"))
        assert "Find compute" not in setup[1].path.read_bytes().decode(errors="ignore")
        chat.patch(run["conversation_id"], {"logging_enabled": True, "training_enabled": True})
        assert build_dataset(setup[1])[0] == []
        setup[1].put("feedback", {"conversation_id": run["conversation_id"], "message_id": result["message_id"], "label": "good"})
        examples = build_dataset(setup[1])[0]
        assert len(examples) == 1 and "calc.py:1" in examples[0]["messages"][-1]["content"]
        assert "PRIVATE_REASONING" not in json.dumps(examples) and "tool_call_id" not in json.dumps(examples)
        memory = agent.remember(run["id"], "compute doubles its input; calc.py:1")
        assert agent.remember(run["id"], memory["content"])["id"] == memory["id"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_writes_and_duplicate_call_results_execute_once(setup):
    action = {"path": "created.py", "content": "answer = 42\n", "expected_sha256": None}
    agent, _, _, run = await service(setup, [
        turn(call("create", "write_file", **action)), turn(call("duplicate", "write_file", **action)),
        turn(call("create", "write_file", **action)), turn(answer="Created created.py."),
    ], access="write")
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        assert (setup[2] / "created.py").read_text() == "answer = 42\n"
        assert len(setup[1].list("agent_operations")) == 1
        results = [e for e in agent.states[run["id"]].events if e["event"] == "tool_result"]
        assert sum(e["data"].get("reused", False) for e in results) == 2
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_partial_provider_failure_never_executes_proposed_tool(setup):
    agent, _, chat, run = await service(setup, [[call("bad", "write_file", path="never", content="x", expected_sha256=None), RuntimeError("broken stream")]], access="write")
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "failed"
        assert not (setup[2] / "never").exists()
        assert chat.get(run["conversation_id"])["messages"] == []
        assert setup[1].list("agent_operations") == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_cancel_before_first_token_releases_provider_and_does_not_save(setup):
    agent, manager, chat, run = await service(setup, ["block"])
    try:
        while not manager.inputs:
            await asyncio.sleep(0)
        result = await agent.cancel(run["id"])
        assert result["status"] == "cancelled" and manager.cancelled
        assert chat.get(run["conversation_id"])["messages"] == []
        events = [event async for event in agent.events(run["id"])]
        assert events[-1][0] == "done"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_context_budget_stops_before_generation(setup):
    settings, store, root = setup
    manager = Manager([])
    manager.count = 65536
    agent = AgentService(settings, store, manager, ChatService(store, manager))
    try:
        run = await agent.start(AgentRequest(goal="Read project", workspace=str(root)))
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "paused"
        assert agent.get(run["id"])["stop_reason"] == "context_compaction_failed"
        assert manager.inputs == [] and store.list("agent_operations") == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_schema_and_withdrawn_tool_errors_are_linked_without_execution(setup):
    agent, _, _, run = await service(setup, [
        turn(call("unknown", "shell_in_text", command="anything"), call("extra", "read_file", path="calc.py", surprise=True)),
        turn(answer="The unsupported requests were rejected."),
    ])
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        assert setup[1].list("agent_operations") == []
        results = [e for e in agent.states[run["id"]].events if e["event"] == "tool_result"]
        assert len(results) == 2 and all(e["data"]["content"]["rejected"] for e in results)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_direct_pc_host_command_requires_decision_and_can_write_outside_workspace(setup):
    outside = setup[2].parent / "host-result.txt"
    argv = ["/usr/bin/python3", "-c", f"from pathlib import Path;Path({str(outside)!r}).write_text('host works')"]
    agent, _, _, run = await service(setup, [turn(call("host", "run_command", argv=argv)), turn(answer="Host operation completed.")], access="computer")
    try:
        async with asyncio.timeout(5):
            while agent.get(run["id"])["status"] != "awaiting_approval":
                assert not agent.tasks[run["id"]].done(), agent.get(run["id"])
                await asyncio.sleep(.01)
        assert not outside.exists()
        pending = agent.get(run["id"])["approval"]
        assert pending["arguments"]["argv"] == argv
        await agent.approve(run["id"], pending["approval_id"], True)
        await agent.tasks[run["id"]]
        assert outside.read_text() == "host works"
        assert agent.get(run["id"])["status"] == "completed"
        with pytest.raises(ValueError, match="resolved|expired"):
            await agent.approve(run["id"], pending["approval_id"], False)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_http_idempotency_and_event_replay_do_not_rerun_tools(setup):
    agent, manager, _, run = await service(setup, [turn(answer="Done.")], idempotency_key="one-task")
    try:
        await agent.tasks[run["id"]]
        same = await agent.start(AgentRequest(goal="Find compute and cite its file/line", workspace=str(setup[2]),
            conversation_id=run["conversation_id"], idempotency_key="one-task"))
        assert same["id"] == run["id"]
        first = [event async for event in agent.events(run["id"])]
        second = [event async for event in agent.events(run["id"], first[-2][1]["sequence"])]
        assert second == first[-1:] and len(manager.inputs) == 1
        with pytest.raises(RuntimeError, match="different"):
            await agent.start(AgentRequest(goal="Different task", idempotency_key="one-task"))
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_private_content_expires_on_restart_and_pending_runs_are_not_resumed(setup):
    agent, manager, chat, run = await service(setup, [turn(answer="Transient final")])
    await agent.tasks[run["id"]]
    await agent.close()
    setup[1].put("agent_runs", {"id": "agent_interrupted", "status": "running"})
    setup[1].put("agent_operations", {"id": "uncertain", "status": "started"})
    restarted = AgentService(setup[0], setup[1], manager, ChatService(setup[1], manager))
    try:
        assert restarted.get(run["id"])["content_retained"] is False
        assert restarted.get("agent_interrupted")["status"] == "interrupted"
        assert setup[1].get("agent_operations", "uncertain")["status"] == "uncertain"
        assert not restarted.tasks and len(manager.inputs) == 1
    finally:
        await restarted.close()


def test_invalid_schema_is_rejected_before_run_creation():
    with pytest.raises(ValueError):
        AgentRequest(goal="x", response_format={"type": "json_schema", "json_schema": {"schema": {"type": "wrong"}}})
    with pytest.raises(ValueError, match="local"):
        AgentRequest(goal="x", response_format={"type": "json_schema", "json_schema": {"schema": {"$ref": "https://example.com/schema"}}})


@pytest.mark.asyncio
async def test_edit_and_delete_replays_reuse_results_before_live_preconditions(setup):
    target = setup[2] / "calc.py"
    import hashlib
    before = target.read_text()
    original = hashlib.sha256(before.encode()).hexdigest()
    changed = before.replace("* 2", "* 3")
    edited = hashlib.sha256(changed.encode()).hexdigest()
    edit = {"path": "calc.py", "old_text": "* 2", "new_text": "* 3", "expected_sha256": original}
    delete = {"path": "calc.py", "expected_sha256": edited}
    agent, _, _, run = await service(setup, [turn(call("a", "edit_file", **edit)),
        turn(call("b", "edit_file", **edit)), turn(call("c", "delete_file", **delete)),
        turn(call("d", "delete_file", **delete)), turn(answer="Completed once each.")], access="write")
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed" and not target.exists()
        operations = setup[1].list("agent_operations")
        assert len(operations) == 2 and all(o["status"] == "completed" for o in operations)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_external_file_change_invalidates_read_result_cache(setup):
    import os
    target = setup[2] / "calc.py"
    def external_edit():
        info = target.stat()
        target.write_text("def compute(value):\n    return value * 9\n")
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        return turn(call("second", "read_file", path="calc.py"))
    agent, _, _, run = await service(setup, [turn(call("first", "read_file", path="calc.py")), external_edit, turn(answer="Read updated source.")])
    try:
        await agent.tasks[run["id"]]
        state = agent.states[run["id"]]
        assert "* 2" in json.dumps(state.calls["first"][1].content)
        assert "* 9" in json.dumps(state.calls["second"][1].content)
        assert len(setup[1].list("agent_operations")) == 2
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_structured_final_uses_strict_validation_and_supported_startup_gate(setup):
    schema = {"type": "json_schema", "json_schema": {"name": "result", "schema": {
        "type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}}}
    agent, manager, _, run = await service(setup, [turn(answer='{"answer":1,"answer":2}'), turn(answer='{"answer":2}')], response_format=schema)
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["answer"] == '{"answer":2}'
        assert len(manager.inputs) == 2 and manager.inputs[-1][1] == []
        with pytest.raises(ValueError, match="structured"):
            await agent.start(AgentRequest(goal="JSON", dialect="anthropic_messages", response_format=schema))
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_denied_command_and_changed_model_after_approval_never_execute(setup):
    action = call("pc", "run_command", argv=["/usr/bin/printf", "approved output"])
    for change_model in (False, True):
        agent, manager, _, run = await service(setup, [turn(action), turn(answer="The operation was not executed.")], access="computer")
        try:
            async with asyncio.timeout(3):
                while agent.get(run["id"])["status"] != "awaiting_approval":
                    await asyncio.sleep(.01)
            pending = agent.get(run["id"])["approval"]
            if change_model:
                manager.loaded = None
            await agent.approve(run["id"], pending["approval_id"], change_model)
            await agent.tasks[run["id"]]
            started = [e for e in agent.states[run["id"]].events if e["event"] == "tool_started"]
            assert started == []
            assert agent.get(run["id"])["status"] == ("failed" if change_model else "completed")
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_approval_expires_with_run_deadline_and_reports_reason(setup):
    agent, _, _, run = await service(setup, [turn(call("pc", "run_command", argv=["/usr/bin/true"]))], access="computer", timeout_seconds=1)
    try:
        async with asyncio.timeout(3):
            while agent.get(run["id"])["status"] != "awaiting_approval":
                await asyncio.sleep(.01)
        pending = agent.get(run["id"])["approval"]
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "failed"
        assert any(e["event"] == "error" and "time limit" in e["data"]["detail"] for e in agent.states[run["id"]].events)
        with pytest.raises(ValueError, match="expired"):
            await agent.approve(run["id"], pending["approval_id"], True)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_explicit_auto_commands_execute_on_host_without_repeated_prompt(setup):
    agent, _, _, run = await service(setup, [turn(call("pc", "run_command", argv=["/usr/bin/printf", "pc works"])), turn(answer="PC command succeeded.")], access="computer", auto_approve_commands=True)
    try:
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "completed"
        assert not any(e["event"] == "approval_required" for e in agent.states[run["id"]].events)
        assert agent.states[run["id"]].calls["pc"][1].content["stdout"] == "pc works"
    finally:
        await agent.close()
