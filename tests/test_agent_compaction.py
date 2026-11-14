"""Automatic context compaction keeps a bounded working project summary."""
import asyncio
import copy
import json

import pytest

from lucifer.agent import AgentCompactionError, AgentService, RunState, COMPACTION_SYSTEM
from lucifer.capabilities import CapabilityRegistry
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.providers import canonical_messages, conservative_count
from lucifer.schemas import AgentRequest
from lucifer.store import Store
from lucifer.tools import ToolContext, ToolResult
from test_agent import turn


class CompactManager:
    def __init__(self, turns, capacity=19000):
        self.turns = turns
        self.inputs = []
        self.counts = []
        self.capabilities = CapabilityRegistry()
        self.capabilities.record("model", "revision", declared={
            "chat_completions": True, "anthropic_messages": True, "responses": True,
            "tools": True, "vision": True, "reasoning": True, "structured_output": True,
        })
        self.loaded = {"id": "model", "display_name": "Lucifer", "effective_context_length": capacity}

    def current(self):
        return self.loaded

    async def count_tokens(self, messages, tools, **kwargs):
        self.counts.append((copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        return conservative_count(messages, list(tools), kwargs.get("response_format"))

    async def agent_events(self, messages, tools, **kwargs):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        selected = self.turns[len(self.inputs) - 1]
        if callable(selected):
            selected = selected()
        for item in selected:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text("def compute(value):\n    return value * 2\n")
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml",
                        agent_workspace=root)
    store = Store(settings.data_dir / "state.db")
    yield settings, store, root
    store.close()


def history(conversation_id):
    rows = []
    for index in range(12):
        rows.append({"id": f"u{index}", "role": "user",
                     "content": f"Earlier project detail {index}: " + "x" * 800})
        rows.append({"id": f"a{index}", "role": "assistant",
                     "content": f"Earlier answer {index}: " + "y" * 800})
    return rows


@pytest.mark.asyncio
async def test_auto_compaction_preserves_working_project_and_docs(setup):
    settings, store, root = setup
    summary = ("Compacted mission context: working project calc.py; fundamental repo docs; "
               "next step inspect the retained files.")
    manager = CompactManager([
        turn(answer=summary),
        turn(answer=summary),
        turn(answer="The project and docs are retained."),
    ], capacity=19000)
    chat = ChatService(store, manager)
    conversation = chat.create(logging_enabled=False)
    chat.transient[conversation["id"]] = history(conversation["id"])
    agent = AgentService(settings, store, manager, chat)
    try:
        run = await agent.start(AgentRequest(
            goal="Continue the project", workspace=str(root),
            conversation_id=conversation["id"], max_tokens=2048,
        ))
        await agent.tasks[run["id"]]
        result = agent.get(run["id"])
        assert result["status"] == "completed"
        assert len(manager.inputs) == 3
        assert manager.inputs[0][1] == [] and manager.inputs[1][1] == []
        assert manager.inputs[2][1] != []
        summary_message = manager.inputs[2][0][1]
        assert summary_message["role"] == "user"
        assert summary_message["content"].startswith("Compacted mission context")
        assert "working project" in summary_message["content"]
        assert "fundamental repo docs" in summary_message["content"]
        assert agent.states[run["id"]].compactions == 1
        events = [event for event in agent.states[run["id"]].events
                  if event["event"] == "context_compacted"]
        assert events
        assert events[0]["data"]["removed_messages"] > 0
        assert events[0]["data"]["after_tokens"] < events[0]["data"]["before_tokens"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_compaction_summary_is_checkpointed(setup):
    settings, store, root = setup
    summary = ("Compacted mission context: working project calc.py; fundamental repo docs; "
               "next step inspect the retained files.")
    manager = CompactManager([
        turn(answer=summary),
        turn(answer=summary),
        turn(answer="The project and docs are retained."),
    ], capacity=19000)
    chat = ChatService(store, manager)
    conversation = chat.create(logging_enabled=True)
    rows = []
    for index in range(12):
        rows.append({"id": f"u{index}", "role": "user", "conversation_id": conversation["id"],
                     "content": f"Earlier project detail {index}: " + "x" * 800})
        rows.append({"id": f"a{index}", "role": "assistant", "conversation_id": conversation["id"],
                     "content": f"Earlier answer {index}: " + "y" * 800})
    store.put_many("messages", rows)
    agent = AgentService(settings, store, manager, chat)
    try:
        run = await agent.start(AgentRequest(
            goal="Continue the project", workspace=str(root),
            conversation_id=conversation["id"], max_tokens=2048,
        ))
        await agent.tasks[run["id"]]
        checkpoint = store.get("agent_checkpoints", run["id"])["checkpoint"]
        assert checkpoint["state"]["compactions"] == 1
        assert any(message["content"].startswith("Compacted mission context")
                   for message in checkpoint["state"]["messages"])
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_compaction_can_be_disabled_explicitly(setup):
    settings, store, root = setup
    manager = CompactManager([turn(answer="unused")], capacity=19000)
    chat = ChatService(store, manager)
    conversation = chat.create(logging_enabled=False)
    chat.transient[conversation["id"]] = history(conversation["id"])
    agent = AgentService(settings, store, manager, chat)
    try:
        run = await agent.start(AgentRequest(
            goal="Continue the project", workspace=str(root),
            conversation_id=conversation["id"], max_tokens=2048,
            context_compaction=False,
        ))
        await agent.tasks[run["id"]]
        assert agent.get(run["id"])["status"] == "failed"
        assert manager.inputs == []
    finally:
        await agent.close()


def exchange(number, content):
    ident = f"call-{number}"
    return [
        {"role": "assistant", "content": None, "reasoning_content": "PRIVATE_THINKING_NEVER_SUMMARIZED",
         "tool_calls": [{"id": ident, "type": "function", "function": {
             "name": "read_file", "arguments": json.dumps({"path": f"file-{number}.py"})}}]},
        {"role": "tool", "tool_call_id": ident, "content": content},
    ]


def retained_agent(setup, messages, *, capacity=19000, turns=None, dialect="chat_completions"):
    settings, store, root = setup
    summary = "Working project: calc.py. Fundamental repo docs retained. Reuse completed reads; inspect the next source."
    manager = CompactManager(turns or [turn(answer=summary) for _ in range(100)], capacity=capacity)
    chat = ChatService(store, manager)
    conversation = chat.create(logging_enabled=True)
    agent = AgentService(settings, store, manager, chat)
    request = AgentRequest(goal="Implement the project", workspace=str(root), max_tokens=2048,
                           dialect=dialect, conversation_id=conversation["id"])
    context = ToolContext(root, access="read", cancel=asyncio.Event(),
                          knowledge=getattr(chat, "knowledge", None), conversation_id=conversation["id"])
    state = RunState(request, context, "model", "revision", None, conversation["id"],
                     messages=messages, turn_inputs=[request.goal])
    state.tools = agent.tools.specs(context)
    agent.states["test-run"] = state
    store.put("agent_runs", {"id": "test-run", "status": "paused", "conversation_id": conversation["id"],
        "model_id": "model", "profile_revision": "revision", "adapter_id": None,
        "workspace": str(root), "access": "read", "dialect": dialect})
    return agent, manager, state


def mission_history():
    return [{"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Implement the project"},
            *[message for n in range(8) for message in exchange(n, f"Receipt {n}: " + "x" * 3000)]]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["chat_completions", "anthropic_messages", "responses"])
async def test_single_mission_compacts_old_tool_exchanges_with_pinned_provider(setup, dialect):
    agent, manager, state = retained_agent(setup, mission_history(), dialect=dialect)
    state.adapter = {"id": "active-adapter", "base_model_id": "model"}
    setup[1].set_setting("active_adapter", "active-adapter")
    state.calls["completed-effect"] = ("signature", ToolResult({"already_written": True}))
    state.operation_ids["completed-effect"] = "operation-retained"
    try:
        before = conservative_count(state.messages, [])["input_tokens"]
        result = await agent._compact_context("test-run", 19000, [], dialect, None, before)
        assert result and result["after_tokens"] + state.request.max_tokens + 32 <= 19000
        canonical_messages(state.messages)  # No orphan tool results or partial tool batches.
        assert state.messages[2] == {"role": "user", "content": "Implement the project"}
        assert any(message.get("tool_call_id") == "call-7" for message in state.messages)
        assert state.operation_ids["completed-effect"] == "operation-retained"
        assert state.calls["completed-effect"][1].content == {"already_written": True}
        for messages, tools, options in manager.inputs:
            assert messages[0]["content"] == COMPACTION_SYSTEM and tools == []
            assert options["model_id"] == "model" and options["profile_revision"] == "revision"
            assert options["adapter"] == state.adapter and options["dialect"] == dialect
            assert options["thinking"] == "off"
        assert "PRIVATE_THINKING" not in json.dumps(manager.inputs)
        checkpoint = setup[1].get("agent_checkpoints", "test-run")
        assert "PRIVATE_THINKING" not in json.dumps(checkpoint)
        assert any(message["content"].startswith("Compacted mission context")
                   for message in checkpoint["checkpoint"]["state"]["messages"])
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_oversized_message_and_arguments_are_fully_covered_by_summary_parts(setup):
    large = "FIRST_RECEIPT" + "a" * 6000 + "MIDDLE_RECEIPT" + "b" * 17000 + "FINAL_RECEIPT"
    original = [{"role": "system", "content": "System instructions"},
                {"role": "user", "content": "Implement the project"},
                *exchange(0, large), *exchange(1, "recent result"), *exchange(2, "latest result")]
    original[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "x" * 1500 + "ARGUMENT_TAIL"})
    agent, manager, state = retained_agent(setup, original)
    try:
        await agent._compact_context("test-run", 19000, [], "chat_completions", None, 50000)
        parts = [messages[1]["content"].split("Messages to compact:\n", 1)[1].rsplit(
            "\n\nReturn the updated compact summary.", 1)[0] for messages, _, _ in manager.inputs]
        assert "".join(parts) == agent._serialize_context(original[2:4])
        assert all(marker in "".join(parts) for marker in ["FIRST_RECEIPT", "MIDDLE_RECEIPT", "FINAL_RECEIPT", "ARGUMENT_TAIL"])
        assert len(parts) >= 3
        assert "PRIVATE_THINKING" not in "".join(parts)
        assert state.messages[-1]["content"] == "latest result"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_summary_fit_loop_terminates_and_retains_original_context(setup):
    agent, manager, state = retained_agent(setup, mission_history(), capacity=100)
    original = copy.deepcopy(state.messages)
    calls = 0
    async def unfittable(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls < 12
        return {"input_tokens": 1000}
    manager.count_tokens = unfittable
    try:
        with pytest.raises(AgentCompactionError, match="original context is retained"):
            await asyncio.wait_for(agent._compact_context("test-run", 100, [], "chat_completions", None, 50000), 1)
        assert state.messages == original and state.compactions == 0
        assert manager.inputs == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_failed_later_summary_rolls_back_all_context_replacement(setup):
    agent, manager, state = retained_agent(setup, mission_history(), turns=[
        turn(answer="A valid first working summary."),
        [("delta", {"content": "Incomplete summary"}), ("finish", {"finish_reason": "length"})],
    ])
    original = copy.deepcopy(state.messages)
    try:
        with pytest.raises(AgentCompactionError, match="did not finish"):
            await agent._compact_context("test-run", 19000, [], "chat_completions", None, 50000)
        assert state.messages == original and state.compactions == 0
        assert not any(event["event"] == "context_compacted" for event in state.events)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_compaction_respects_explicit_total_token_budget(setup):
    agent, manager, state = retained_agent(setup, mission_history())
    state.request = state.request.model_copy(update={"max_total_tokens": 100})
    original = copy.deepcopy(state.messages)
    try:
        with pytest.raises(AgentCompactionError, match="total token budget"):
            await agent._compact_context("test-run", 19000, [], "chat_completions", None, 50000)
        assert state.messages == original and manager.inputs == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_compaction_runs_before_context_exhaustion_to_reserve_output(setup):
    agent, manager, state = retained_agent(setup, mission_history())
    before = conservative_count(state.messages, [])["input_tokens"]
    manager.loaded["effective_context_length"] = before + 1000
    try:
        await agent._loop("test-run", [])
        assert setup[1].get("agent_runs", "test-run")["status"] == "completed"
        assert manager.inputs[0][0][0]["content"] == COMPACTION_SYSTEM
        assert state.compactions == 1
        phases = [event["data"] for event in state.events if event["event"] == "phase"]
        assert phases[-1]["output_budget"] == state.request.max_tokens
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_literal_user_corrections_and_compacted_checkpoint_survive_restore(setup):
    messages = mission_history()
    correction = {"role": "user", "content": "Use Python only; keep the existing API."}
    messages.insert(8, correction)
    agent, _, state = retained_agent(setup, messages)
    state.turn_inputs.append(correction["content"])
    try:
        await agent._compact_context("test-run", 19000, [], "chat_completions", None, 50000)
        assert correction in state.messages
        compacted = agent._public_messages(state.messages)
        agent.states.pop("test-run")
        restored = agent._restore_state(setup[1].get("agent_runs", "test-run"))
        assert restored.messages == compacted
        assert restored.compactions == 1 and correction in restored.messages
        assert restored.reasoning_turns == []
    finally:
        await agent.close()
