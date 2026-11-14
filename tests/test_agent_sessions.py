"""Resumable mission conversations keep context and emitted thinking in RAM."""
import asyncio
import copy
import json
import sys

import pytest

from lucifer.agent import AgentService
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.schemas import AgentRequest
from lucifer.store import Store
from test_agent import Manager, call, turn


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "calc.py").write_text("def compute(value):\n    return value * 2\n")
    settings = Settings(data_dir=tmp_path / "data", models_file=tmp_path / "models.yaml", agent_workspace=workspace)
    store = Store(settings.data_dir / "state.db")
    yield settings, store, workspace
    store.close()


class SessionManager(Manager):
    def __init__(self, turns):
        super().__init__(turns)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def agent_events(self, messages, tools, **kwargs):
        from lucifer.providers import canonical_messages
        canonical_messages(messages)
        index = len(self.inputs)
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        selected = self.turns[index]
        if isinstance(selected, dict):
            if selected.get("reasoning"):
                yield "reasoning", {"content": selected["reasoning"]}
            if selected.get("partial"):
                yield "delta", {"content": selected["partial"]}
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.cancelled = True
            selected = selected.get("after", turn(answer="Uncorrected answer"))
        for event in selected:
            yield event


async def create(setup, turns, **options):
    settings, store, workspace = setup
    manager = SessionManager(turns)
    chat = ChatService(store, manager)
    service = AgentService(settings, store, manager, chat)
    if "conversation_id" not in options:
        options["conversation_id"] = chat.create(logging_enabled=False)["id"]
    run = await service.start(AgentRequest(goal="Inspect compute", workspace=str(workspace), **options))
    return service, manager, chat, run


@pytest.mark.asyncio
async def test_defaults_remove_deadline_and_implicit_round_limit(setup):
    request = AgentRequest(goal="Long mission")
    assert request.timeout_seconds is None and request.max_rounds is None
    assert request.max_tool_calls is None and request.max_total_tokens is None
    for n in range(14):
        (setup[2] / f"part-{n}.txt").write_text(f"Independent evidence {n}")
    turns = [turn(call(f"read-{n}", "read_file", path=f"part-{n}.txt")) for n in range(14)]
    service, manager, _, run = await create(setup, [*turns, turn(answer="Reached the goal.")])
    try:
        await asyncio.wait_for(service.tasks[run["id"]], 3)
        result = service.get(run["id"])
        assert result["status"] == "completed" and result["round"] == 15
        assert result["limits"]["timeout_seconds"] is None
        assert len(manager.inputs) == 15 and len(setup[1].list("agent_operations")) == 14
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_same_run_followup_preserves_context_and_saves_distinct_pairs(setup):
    service, manager, chat, run = await create(setup, [turn(answer="The name is Ada."), turn(answer="The corrected name is Grace.")])
    try:
        await service.tasks[run["id"]]
        first_segment = service.get(run["id"])["segment"]
        response = await service.submit_input(run["id"], "Correct the name to Grace.", options={"stop_when_done": False, "max_tokens": 8192})
        assert response["id"] == run["id"] and response["session_turn"] == 2
        assert response["segment"] == first_segment + 1 and response["max_tokens"] == 8192
        await service.tasks[run["id"]]
        second_history = manager.inputs[-1][0]
        assert any(m["content"] == "The name is Ada." for m in second_history)
        assert second_history[-1] == {"role": "user", "content": "Correct the name to Grace."}
        transcript = chat.get(run["conversation_id"])["messages"]
        assert [m["content"] for m in transcript] == ["Inspect compute", "The name is Ada.", "Correct the name to Grace.", "The corrected name is Grace."]
        assert service.get(run["id"])["status"] == "awaiting_input" and not service.busy
        assert len(manager.inputs) == 2  # Manual session mode waits instead of spinning.
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_new_run_seeds_the_existing_chat_conversation(setup):
    settings, store, workspace = setup
    manager = SessionManager([turn(answer="I remember the previous conversation.")])
    chat = ChatService(store, manager)
    conv = chat.create(logging_enabled=False)
    chat.transient[conv["id"]] = [{"id": "prior-user", "role": "user", "content": "My selected name is Ada."},
                                  {"id": "prior-answer", "role": "assistant", "content": "I will use Ada."}]
    service = AgentService(settings, store, manager, chat)
    try:
        run = await service.start(AgentRequest(goal="Continue that discussion", workspace=str(workspace), conversation_id=conv["id"]))
        await service.tasks[run["id"]]
        assert [m["content"] for m in manager.inputs[0][0]][1:] == ["My selected name is Ada.", "I will use Ada.", "Continue that discussion"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_correction_during_thinking_interrupts_and_retains_actual_thoughts(setup):
    thought = "ACTUAL_PROVIDER_THOUGHT_482: inspect the wrong branch."
    service, manager, chat, run = await create(setup, [{"reasoning": thought}, turn(answer="Using the corrected branch.")], thinking="high")
    try:
        await manager.entered.wait()
        corrected = await service.submit_input(run["id"], "Use the stable branch instead.")
        assert corrected["segment"] == 2 and corrected["session_turn"] == 2
        await service.tasks[run["id"]]
        assert manager.cancelled
        result = service.get(run["id"])
        assert result["reasoning_retained"] and result["reasoning_characters"] == len(thought)
        assert any(e["event"] == "reasoning" and e["data"]["content"] == thought for e in service.states[run["id"]].events)
        history = manager.inputs[-1][0]
        assert history[-1]["content"] == "Use the stable branch instead."
        assert any(m.get("reasoning_content") == thought for m in history if m["role"] == "assistant")
        assert thought not in "\n".join(str(m.get("content")) for m in history)
        saved = chat.get(run["conversation_id"])["messages"]
        assert len(saved) == 2 and "stable branch" in saved[0]["content"]
        assert thought not in json.dumps(saved) and thought not in json.dumps(setup[1].list("agent_events"))
        assert thought not in json.dumps(result)
        done = [e["data"] for e in service.states[run["id"]].events if e["event"] == "done"]
        assert [(e["segment"], e["status"]) for e in done] == [(1, "paused"), (2, "completed")]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_manual_stop_and_resume_keep_thinking_and_original_goal(setup):
    service, manager, _, run = await create(setup, [{"reasoning": "Retained actual reasoning."}, turn(answer="Finished the original mission.")], thinking="max")
    try:
        await manager.entered.wait()
        paused = await service.stop(run["id"])
        assert paused["status"] == "paused" and paused["resumable"]
        assert not service.busy and paused["reasoning_retained"]
        resumed = await service.resume(run["id"])
        assert resumed["segment"] == 2 and resumed["session_turn"] == 1
        await service.tasks[run["id"]]
        history = manager.inputs[-1][0]
        assert any(m["content"] == "Inspect compute" for m in history)
        assert "Resume the same mission" in history[-1]["content"]
        assert service.get(run["id"])["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_queued_correction_applies_at_completed_provider_boundary(setup):
    service, manager, chat, run = await create(setup, [{"after": turn(answer="Superseded interim answer")}, turn(answer="Corrected final answer")])
    try:
        await manager.entered.wait()
        queued = await service.submit_input(run["id"], "Revise the target first.", interrupt=False)
        assert queued["queued"] and not manager.cancelled
        manager.release.set()
        await service.tasks[run["id"]]
        assert manager.inputs[-1][0][-1]["content"] == "Revise the target first."
        transcript = chat.get(run["conversation_id"])["messages"]
        assert len(transcript) == 2 and transcript[-1]["content"] == "Corrected final answer"
        assert "Superseded interim answer" not in json.dumps(transcript)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_queued_correction_keeps_thinking_only_native_history(setup):
    thought = "ACTUAL_PROVIDER_THINKING before the queued clarification."
    service, manager, _, run = await create(setup,
        [{"reasoning": thought, "after": turn()}, turn(answer="Clarification applied.")], thinking="high")
    try:
        await manager.entered.wait()
        await service.submit_input(run["id"], "Use the revised target.", interrupt=False)
        manager.release.set()
        await service.tasks[run["id"]]
        history = manager.inputs[-1][0]
        assert any(m.get("reasoning_content") == thought and m["content"] is None for m in history)
        assert history[-1]["content"] == "Use the revised target."
        assert service.get(run["id"])["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_debits_unknown_usage_without_bypassing_explicit_budget(setup):
    service, manager, _, run = await create(setup,
        [{"partial": "Incomplete answer"}, turn(answer=" continued.")], max_total_tokens=1100)
    try:
        await manager.entered.wait()
        paused = await service.stop(run["id"])
        assert paused["total_tokens"] == 1100
        assert paused["last_turn_usage"] == {"total_tokens": 1100, "estimated": True, "interrupted": True}
        await service.resume(run["id"])
        await service.tasks[run["id"]]
        assert service.get(run["id"])["status"] == "failed" and len(manager.inputs) == 1
        await service.resume(run["id"], limits={"max_total_tokens": 3000})
        await service.tasks[run["id"]]
        assert service.get(run["id"])["answer"] == "Incomplete answer continued."
        assert service.get(run["id"])["total_tokens"] == 1120
        assert len(manager.inputs) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_correction_withdraws_unapproved_batch_and_repairs_tool_history(setup):
    target = setup[2].parent / "must-not-exist.txt"
    command = [sys.executable, "-c", f"from pathlib import Path;Path({str(target)!r}).write_text('bad')"]
    service, manager, _, run = await create(setup, [turn(call("command", "run_command", argv=command), call("read", "read_file", path="calc.py")), turn(answer="No host command was executed.")], access="computer")
    try:
        async with asyncio.timeout(3):
            while service.get(run["id"])["status"] != "awaiting_approval":
                await asyncio.sleep(.01)
        old_approval = service.get(run["id"])["pending_approval"]
        await service.submit_input(run["id"], "Do not run that command; answer without changing files.")
        await service.tasks[run["id"]]
        assert not target.exists()
        history = manager.inputs[-1][0]
        tool_results = [json.loads(m["content"]) for m in history if m["role"] == "tool"]
        assert len(tool_results) == 2 and all(result["interrupted"] for result in tool_results)
        assert setup[1].list("agent_operations")[0]["status"] == "withdrawn"
        with pytest.raises(ValueError):
            await service.approve(run["id"], old_approval, True)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_paused_tool_batch_retains_completed_images_without_reading_again(setup):
    import base64
    image = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6ZQAAAABJRU5ErkJggg==')
    image_path = setup[2] / 'tiny.png'
    image_path.write_bytes(image)
    service, manager, _, run = await create(setup, [turn(call('image', 'read_image', path='tiny.png'),
        call('pending', 'run_command', argv=['/usr/bin/true'])), turn(answer='Retained the completed image.')],
        access='computer', allow_images=True)
    try:
        async with asyncio.timeout(3):
            while service.get(run['id'])['status'] != 'awaiting_approval':
                await asyncio.sleep(.01)
        image_path.unlink()  # A resumed history must use the completed result.
        await service.stop(run['id'])
        await service.resume(run['id'])
        await service.tasks[run['id']]
        history = manager.inputs[-1][0]
        pictures = [part for m in history if isinstance(m['content'], list) for part in m['content']
                    if part['type'] == 'image_url']
        assert len(pictures) == 1 and pictures[0]['image_url']['url'].endswith(base64.b64encode(image).decode())
        assert service.get(run['id'])['status'] == 'completed'
        assert not any(e['event'] == 'tool_started' and e['data']['call_id'] == 'pending'
                       for e in service.states[run['id']].events)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_previous_segment_approval_cannot_be_reused_during_followup(setup):
    service, manager, _, run = await create(setup, [turn(call('approved', 'run_command', argv=['/usr/bin/true'])),
        turn(answer='Approved command completed.'), {'reasoning': 'A later conversation turn.'}], access='computer')
    try:
        async with asyncio.timeout(3):
            while service.get(run['id'])['status'] != 'awaiting_approval':
                await asyncio.sleep(.01)
        approval = service.get(run['id'])['pending_approval']
        await service.approve(run['id'], approval, True)
        await service.tasks[run['id']]
        await service.submit_input(run['id'], 'A follow-up requiring thought.')
        await manager.entered.wait()
        with pytest.raises(ValueError, match='previous session segment'):
            await service.approve(run['id'], approval, True)
        assert len(setup[1].list('agent_operations')) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_input_lost_response_and_concurrent_retries_apply_one_transition(setup):
    update = 'TRANSPORT_UPDATE_918: use the corrected target.'
    service, manager, _, run = await create(setup, [turn(answer='Initial result.'),
        {'reasoning': 'Working on the corrected target.', 'after': turn(answer='Corrected result.')}])
    try:
        await service.tasks[run['id']]
        # Deliberately discard the first response, as a disconnected client does.
        await service.submit_input(run['id'], update, options={'max_tokens': 4096}, idempotency_key='lost-input')
        await manager.entered.wait()
        task = service.tasks[run['id']]
        replies = await asyncio.gather(*[service.submit_input(run['id'], update,
            options={'max_tokens': 4096}, idempotency_key='lost-input') for _ in range(8)])
        assert all(r['idempotent_replay'] and r['segment'] == 2 and r['session_turn'] == 2 for r in replies)
        assert service.tasks[run['id']] is task and not manager.cancelled
        for changed in ({'content': 'Conflicting target.'}, {'interrupt': False}, {'options': {'max_tokens': 2048}}):
            request = {'content': update, 'options': {'max_tokens': 4096}, **changed}
            with pytest.raises(ValueError, match='different input request'):
                await service.submit_input(run['id'], **request, idempotency_key='lost-input')
        assert service.get(run['id'])['max_tokens'] == 4096
        manager.release.set()
        await task
        replay = await service.submit_input(run['id'], update,
            options={'max_tokens': 4096}, idempotency_key='lost-input')
        assert replay['status'] == 'completed' and replay['segment'] == 2
        assert len(manager.inputs) == 2
        assert sum(m.get('content') == update for m in service.states[run['id']].messages) == 1
        ledger = json.dumps(service.states[run['id']].control_requests)
        assert update not in ledger and 'lost-input' not in ledger
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_queued_input_concurrent_retries_are_queued_only_once(setup):
    update = 'Apply this clarification once.'
    service, manager, chat, run = await create(setup,
        [{'reasoning': 'Original task.', 'after': turn(answer='Interim result.')}, turn(answer='Clarified result.')])
    try:
        await manager.entered.wait()
        replies = await asyncio.gather(*[service.submit_input(run['id'], update, interrupt=False,
            idempotency_key='queued-once') for _ in range(8)])
        assert sum(bool(reply.get('queued')) for reply in replies) == 1
        assert service.get(run['id'])['pending_inputs'] == 1
        manager.release.set()
        await service.tasks[run['id']]
        assert len(manager.inputs) == 2 and service.get(run['id'])['session_turn'] == 2
        transcript = chat.get(run['conversation_id'])['messages']
        assert transcript[0]['content'].count(update) == 1
        assert sum(e['event'] == 'input' for e in service.states[run['id']].events) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_resume_retry_does_not_undo_a_later_manual_pause(setup):
    service, manager, _, run = await create(setup, [{'reasoning': 'First attempt.'},
        {'reasoning': 'Resumed attempt.'}, turn(answer='Finally finished.')])
    try:
        await manager.entered.wait()
        await service.stop(run['id'])
        manager.entered.clear()
        await service.resume(run['id'], limits={'max_tokens': 4096}, idempotency_key='resume-once')
        await manager.entered.wait()
        task = service.tasks[run['id']]
        replies = await asyncio.gather(*[service.resume(run['id'], limits={'max_tokens': 4096},
            idempotency_key='resume-once') for _ in range(8)])
        assert all(reply['idempotent_replay'] and reply['segment'] == 2 for reply in replies)
        assert service.tasks[run['id']] is task
        with pytest.raises(ValueError, match='different resume request'):
            await service.resume(run['id'], limits={'max_tokens': 8192}, idempotency_key='resume-once')
        await service.stop(run['id'])
        paused = await service.resume(run['id'], limits={'max_tokens': 4096}, idempotency_key='resume-once')
        assert paused['status'] == 'paused' and paused['segment'] == 2 and not service.busy
        await service.resume(run['id'], idempotency_key='explicit-new-resume')
        await service.tasks[run['id']]
        assert service.get(run['id'])['status'] == 'completed' and len(manager.inputs) == 3
        controls = [m for m in manager.inputs[-1][0]
                    if isinstance(m['content'], str) and m['content'].startswith('Resume the same mission')]
        assert len(controls) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_resume_instruction_has_its_own_idempotency_namespace(setup):
    service, manager, _, run = await create(setup,
        [turn(answer='First.'), turn(answer='Second.'), turn(answer='Third.')])
    try:
        await service.tasks[run['id']]
        await service.submit_input(run['id'], 'Second goal.', idempotency_key='shared-key')
        await service.tasks[run['id']]
        await service.resume(run['id'], 'Third goal.', idempotency_key='shared-key')
        await service.tasks[run['id']]
        replay = await service.resume(run['id'], 'Third goal.', idempotency_key='shared-key')
        assert replay['idempotent_replay'] and replay['segment'] == 3 and len(manager.inputs) == 3
        with pytest.raises(ValueError, match='different resume request'):
            await service.resume(run['id'], 'Conflicting third goal.', idempotency_key='shared-key')
        assert len(service.states[run['id']].control_requests) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_interrupted_control_claim_is_not_reapplied_implicitly(setup, monkeypatch):
    service, manager, _, run = await create(setup, [turn(answer='Initial.'), turn(answer='Recovered.')])
    try:
        await service.tasks[run['id']]
        launch = service._launch
        async def fail_after_history(ident):
            assert service.states[ident].control_requests  # Claimed before transition effects.
            raise RuntimeError('Transport request interrupted before launch')
        monkeypatch.setattr(service, '_launch', fail_after_history)
        with pytest.raises(RuntimeError, match='interrupted before launch'):
            await service.submit_input(run['id'], 'Retained new goal.', idempotency_key='interrupted-control')
        replay = await service.submit_input(run['id'], 'Retained new goal.', idempotency_key='interrupted-control')
        assert replay['control_status'] == 'interrupted' and replay['session_turn'] == 2
        assert sum(m.get('content') == 'Retained new goal.' for m in service.states[run['id']].messages) == 1
        monkeypatch.setattr(service, '_launch', launch)
        await service.resume(run['id'], idempotency_key='explicit-recovery')
        await service.tasks[run['id']]
        assert service.get(run['id'])['answer'] == 'Recovered.' and len(manager.inputs) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_invalid_control_options_do_not_claim_the_retry_key(setup):
    service, manager, _, run = await create(setup, [turn(answer='Initial.'), turn(answer='Valid follow-up.')])
    try:
        await service.tasks[run['id']]
        with pytest.raises(ValueError, match='Only thinking'):
            await service.submit_input(run['id'], 'A valid goal.', options={'access': 'computer'},
                                       idempotency_key='valid-after-validation')
        assert service.states[run['id']].control_requests == {}
        await service.submit_input(run['id'], 'A valid goal.', idempotency_key='valid-after-validation')
        await service.tasks[run['id']]
        assert service.get(run['id'])['answer'] == 'Valid follow-up.' and len(manager.inputs) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_resuming_does_not_repeat_a_completed_write(setup):
    action = {"path": "once.txt", "content": "write once", "expected_sha256": None}
    service, manager, _, run = await create(setup, [turn(call("write", "write_file", **action)), {"reasoning": "Already wrote the file."}, turn(call("recalled", "write_file", **action)), turn(answer="No duplicate effect.")], access="write", thinking="high")
    try:
        await manager.entered.wait()
        before = (setup[2] / "once.txt").stat().st_mtime_ns
        await service.stop(run["id"])
        await service.resume(run["id"])
        await service.tasks[run["id"]]
        assert (setup[2] / "once.txt").stat().st_mtime_ns == before
        assert len(setup[1].list("agent_operations")) == 1
        assert service.get(run["id"])["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_explicit_budget_extension_resumes_same_history(setup):
    service, manager, _, run = await create(setup, [turn(call("read", "read_file", path="calc.py")), turn(answer="Completed after extending the selected budget.")], max_rounds=1)
    try:
        await service.tasks[run["id"]]
        assert service.get(run["id"])["status"] == "failed"
        await service.resume(run["id"], limits={"max_rounds": 3})
        await service.tasks[run["id"]]
        assert service.get(run["id"])["status"] == "completed" and len(manager.inputs) == 2
        assert any(m["role"] == "tool" for m in manager.inputs[-1][0])
        assert len(setup[1].list("agent_operations")) == 1
        with pytest.raises(ValueError, match="Only thinking"):
            await service.submit_input(run["id"], "Escalate", options={"access": "computer"})
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_length_limited_final_text_continues_without_losing_prefix(setup):
    first = [("reasoning", {"content": "Actual reasoning before the limit."}), ("delta", {"content": "First half;"}), ("finish", {"finish_reason": "length"})]
    service, manager, chat, run = await create(setup, [first, turn(answer=" second half.")], thinking="high")
    try:
        await service.tasks[run["id"]]
        assert service.get(run["id"])["answer"] == "First half; second half."
        assert manager.inputs[-1][1] == []
        assert any(m["role"] == "assistant" and m["content"] == "First half;" for m in manager.inputs[-1][0])
        assert len(chat.get(run["conversation_id"])["messages"]) == 2
        assert any(e["event"] == "continuation" for e in service.states[run["id"]].events)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_partial_tool_calls_are_never_continued_as_effects(setup):
    unfinished = [call("bad", "write_file", path="bad.txt", content="never", expected_sha256=None), ("finish", {"finish_reason": "length"})]
    service, _, _, run = await create(setup, [unfinished], access="write")
    try:
        await service.tasks[run["id"]]
        assert service.get(run["id"])["status"] == "failed"
        assert not (setup[2] / "bad.txt").exists() and setup[1].list("agent_operations") == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_thinking_only_length_retains_native_history_and_can_still_call_tools(setup):
    thought = "ACTUAL_NATIVE_THINKING: the source must be inspected next."
    limited = [("reasoning", {"content": thought}), ("finish", {"finish_reason": "length"})]
    service, manager, chat, run = await create(setup, [limited,
        turn(call("read", "read_file", path="calc.py")), turn(answer="Inspected the source.")], thinking="max")
    try:
        await service.tasks[run["id"]]
        continuation = manager.inputs[1]
        assert continuation[1]  # A thinking-only limit did not remove needed tools.
        assert any(m.get("reasoning_content") == thought and m["content"] is None for m in continuation[0])
        assert thought not in json.dumps(chat.get(run["conversation_id"])["messages"])
        assert service.get(run["id"])["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_reasoning_history_is_not_evicted_by_old_ring_or_session_count(setup):
    chunk = "PROVIDER_THINKING_RETAINED_" + "z" * 65536
    large = [("reasoning", {"content": chunk}) for _ in range(65)] + turn(answer="Retained.")
    service, manager, chat, run = await create(setup, [large, *[turn(answer="Other mission") for _ in range(33)]], thinking="high")
    try:
        await service.tasks[run["id"]]
        for _ in range(33):
            other = await service.start(AgentRequest(goal="Another mission", workspace=str(setup[2])))
            await service.tasks[other["id"]]
        state = service.states[run["id"]]
        assert state.event_bytes > 4 * 1024 * 1024
        assert sum(e["event"] == "reasoning" for e in state.events) == 65
        assert service.get(run["id"])["reasoning_characters"] == len(chunk) * 65
        assert "PROVIDER_THINKING_RETAINED" not in json.dumps(setup[1].list("agent_events"))
        chat.delete(run["conversation_id"])
        assert run["id"] not in service.states
        with pytest.raises(ValueError, match="no longer in memory"):
            await service.resume(run["id"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_before_initial_image_preparation_is_resumable(setup):
    import base64
    image = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6ZQAAAABJRU5ErkJggg==')
    (setup[2] / 'tiny.png').write_bytes(image)
    service, manager, _, run = await create(setup, [turn(answer='Image input retained.')], images=['tiny.png'])
    try:
        await service.stop(run['id'])
        await service.resume(run['id'])
        await service.tasks[run['id']]
        assert service.get(run['id'])['status'] == 'completed'
        assert any(isinstance(m['content'], list) for m in manager.inputs[-1][0])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stopping_host_effect_records_uncertainty_and_does_not_replay(setup):
    witness = setup[2] / 'command-once.txt'
    code = f"from pathlib import Path;import time;p=Path({str(witness)!r});p.write_text((p.read_text() if p.exists() else '')+'effect\\n');time.sleep(60)"
    argv = [sys.executable, '-c', code]
    service, manager, _, run = await create(setup, [turn(call('first-command', 'run_command', argv=argv)),
        turn(call('recalled-command', 'run_command', argv=argv)), turn(answer='The uncertain effect was not replayed.')], access='computer', auto_approve_commands=True)
    try:
        async with asyncio.timeout(4):
            while not witness.exists():
                assert not service.tasks[run['id']].done(), service.get(run['id'])
                await asyncio.sleep(.01)
        await service.stop(run['id'])
        assert setup[1].list('agent_operations')[0]['status'] == 'uncertain'
        await service.resume(run['id'], idempotency_key='resume-uncertain-once')
        await service.tasks[run['id']]
        replay = await service.resume(run['id'], idempotency_key='resume-uncertain-once')
        assert replay['idempotent_replay'] and replay['segment'] == 2
        assert witness.read_text() == 'effect\n'
        assert service.get(run['id'])['status'] == 'completed'
        results = [json.loads(m['content']) for m in manager.inputs[-1][0] if m['role'] == 'tool']
        assert results[0]['uncertain'] is True
        assert results[-1]['rejected'] is True
    finally:
        await service.close()
