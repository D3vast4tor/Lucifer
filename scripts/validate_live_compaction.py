#!/usr/bin/env python3
"""Exercise compaction with the currently loaded model and a simulated 12K limit.

The application's real model allocation is never changed. All mission history,
tool receipts and files are synthetic, held in a temporary validation store.
Only a small public receipt is retained; provider reasoning is never written.
Run this explicitly after other live model acceptance requests have finished.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import httpx

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lucifer.agent import AgentService, RunState, SUMMARY_PREFIX, digest  # noqa: E402
from lucifer.capabilities import CapabilitySnapshot  # noqa: E402
from lucifer.chat import ChatService  # noqa: E402
from lucifer.config import Settings  # noqa: E402
from lucifer.inference import ModelManager, ModelSpec  # noqa: E402
from lucifer.providers import ProviderAdapter, canonical_messages, conservative_count  # noqa: E402
from lucifer.schemas import AgentRequest  # noqa: E402
from lucifer.store import Store  # noqa: E402
from lucifer.tools import ToolContext, ToolResult  # noqa: E402


SIMULATED_CAPACITY = 12000


class LoadedModelProxy:
    """Use one cached startup contract without launching or probing a runtime."""
    def __init__(self, client, current, adapter, dialect):
        self.client, self.actual, self.adapter, self.dialect = client, current, adapter, dialect
        self.spec = ModelSpec.model_validate(current["spec"])
        saved = current["capabilities"]
        self.snapshot = CapabilitySnapshot(**{key: saved[key] for key in (
            "model_id", "profile_revision", "discovered_at", "capabilities", "context", "discovery")})
        if not self.snapshot.supports(dialect):
            raise ValueError(f"The cached model snapshot does not advertise {dialect}")
        self.capabilities = {self.snapshot.model_id: self.snapshot}
        self.simulated = {**current, "effective_context_length": SIMULATED_CAPACITY,
                          "max_output_tokens": min(1024, current.get("max_output_tokens") or 1024)}
        self.calls = []
        self.reasoning_events = 0

    def current(self):
        return self.simulated

    async def count_tokens(self, messages, tools, **kwargs):
        return conservative_count(messages, list(tools), kwargs.get("response_format"))

    async def agent_events(self, messages, tools, **kwargs):
        if (kwargs["model_id"] != self.snapshot.model_id
                or kwargs["profile_revision"] != self.snapshot.profile_revision
                or kwargs["adapter"] != self.adapter or kwargs["dialect"] != self.dialect):
            raise ValueError("Validation attempted to change the pinned provider")
        if tools or kwargs["thinking"] != "off" or kwargs["max_tokens"] > 1024:
            raise ValueError("Validation must use bounded summaries with tools and thinking disabled")
        provider = ProviderAdapter(self.dialect, self.snapshot)
        model = (self.adapter["id"] if self.adapter and (
            self.spec.backend == "vllm" or self.spec.adapter_backend == "vllm") else self.spec.model_name)
        payload = provider.payload(model=model, messages=messages, tools=[],
            max_tokens=kwargs["max_tokens"], temperature=0.0, thinking="off",
            thinking_options=ModelManager._thinking_options(self.spec, "off"),
            compatibility=self.spec.backend in {"strata", "vllm", "llama.cpp"})
        key = os.environ.get(self.spec.api_key_env, "") if self.spec.api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        receipt = {"max_output_tokens": kwargs["max_tokens"], "estimated_input_tokens": kwargs["input_tokens"]}
        self.calls.append(receipt)
        async with aclosing(provider.stream(self.client, self.spec.base_url, headers, payload)) as events:
            async for event, data in events:
                if event == "reasoning":
                    self.reasoning_events += 1
                    continue
                if event == "usage":
                    receipt["usage"] = {key: data[key] for key in (
                        "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
                        if isinstance(data.get(key), int) and not isinstance(data[key], bool)}
                if event == "finish":
                    receipt["finish_reason"] = data.get("finish_reason")
                yield event, data


def synthetic_history(goal):
    messages = [{"role": "system", "content": "Synthetic acceptance fixture. You are Lucifer; preserve the ORBIT project facts."},
                {"role": "user", "content": goal}]
    facts = ["ORBIT project file src/cache.py now records cache revision 7. Completed: inspect source and confirm digest. Pending: verify restart persistence.",
             "ORBIT documentation docs/CACHE.md describes immutable startup snapshots. Reuse completed reads; do not repeat the completed inspection.",
             "Recent preserved receipt: ORBIT cache revision 7 was observed in src/cache.py.",
             "Recent preserved receipt: the next step is a restart-persistence check; no commands are currently executing."]
    for number, fact in enumerate(facts):
        ident = f"synthetic-read-{number}"
        content = fact + "\n" + ("Synthetic previously observed cache metadata. " * (54 if number < 2 else 35))
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": ident, "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "src/cache.py" if number % 2 == 0 else "docs/CACHE.md"})}}]},
            {"role": "tool", "tool_call_id": ident, "content": content},
        ])
    while conservative_count(messages, [])["input_tokens"] < SIMULATED_CAPACITY + 256:
        messages[-1]["content"] += " Synthetic recent receipt padding." * 16
    canonical_messages(messages)
    return messages


async def validate(args):
    started = time.monotonic()
    token = os.environ.get("LUCIFER_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), trust_env=False) as client:
        current_response = await client.get(args.api_url.rstrip("/") + "/models/current", headers=headers)
        current_response.raise_for_status()
        actual = current_response.json()
        if not actual or not actual.get("capabilities"):
            raise ValueError("Load a model and its startup capability snapshot before validation")
        status_response = await client.get(args.api_url.rstrip("/") + "/system/status", headers=headers)
        status_response.raise_for_status()
        status = status_response.json()
        if not args.dry_run and (status.get("agent_busy") or status.get("jobs")):
            raise ValueError("Wait for the active mission or training/sleep jobs before this live check")
        adapter = None
        if status.get("active_adapter"):
            response = await client.get(args.api_url.rstrip("/") + "/adapters", headers=headers)
            response.raise_for_status()
            adapter = next(item for item in response.json() if item["id"] == status["active_adapter"])
        actual_capacity = actual.get("effective_context_length") or actual.get("context_length")
        if not isinstance(actual_capacity, int) or actual_capacity < SIMULATED_CAPACITY:
            raise ValueError("The real model context must exceed the simulated 12K fixture")
        proxy = LoadedModelProxy(client, actual, adapter, args.dialect)
        goal = "Complete the ORBIT cache change in src/cache.py. Preserve completed reads. Pending: verify restart persistence."
        messages = synthetic_history(goal)
        before = conservative_count(messages, [])["input_tokens"]
        with tempfile.TemporaryDirectory(prefix="lucifer-live-compaction-") as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = Store(root / "state" / "validation.sqlite3")
            chat = ChatService(store, proxy)
            conversation = chat.create(title="Synthetic live compaction acceptance", training_enabled=False)
            context = ToolContext(workspace, access="read", cancel=asyncio.Event(),
                                  knowledge=chat.knowledge, conversation_id=conversation["id"])
            request = AgentRequest(goal=goal, workspace=str(workspace), conversation_id=conversation["id"],
                                   dialect=args.dialect, max_tokens=512, thinking="off")
            if adapter:
                store.put("adapters", adapter)
                store.set_setting("active_adapter", adapter["id"])
            agent = AgentService(Settings(data_dir=store.path.parent, models_file=root / "unused-models.yaml",
                                           agent_workspace=workspace), store, proxy, chat)
            state = RunState(request, context, proxy.snapshot.model_id, proxy.snapshot.profile_revision,
                             adapter, conversation["id"], messages=messages, turn_inputs=[goal])
            state.tools = agent.tools.specs(context)
            state.pending_inputs = ["Preserve the ORBIT tag and the pending restart-persistence check."]
            state.calls["completed-fixture-read"] = ("synthetic-signature", ToolResult({"observed_revision": 7}))
            state.operation_ids["completed-fixture-read"] = "synthetic-operation-receipt"
            original_ledger = digest({"operations": state.operation_ids, "calls": {
                key: [signature, result.to_dict()] for key, (signature, result) in state.calls.items()}})
            pending = list(state.pending_inputs)
            agent.states["synthetic-validation"] = state
            store.put("agent_runs", {"id": "synthetic-validation", "status": "paused", "conversation_id": conversation["id"],
                "model_id": state.model_id, "profile_revision": state.profile_revision,
                "adapter_id": adapter["id"] if adapter else None, "workspace": str(workspace), "access": "read", "dialect": args.dialect})
            try:
                if args.dry_run:
                    return {"status": "prepared_only", "actual_model_id": actual["id"],
                        "actual_context_tokens": actual_capacity, "simulated_context_tokens": SIMULATED_CAPACITY,
                        "estimated_fixture_tokens": before, "inference_requests": 0}
                async with asyncio.timeout(args.timeout_seconds):
                    result = await agent._compact_context("synthetic-validation", SIMULATED_CAPACITY, [], args.dialect, None, before)
                summary = next(message["content"][len(SUMMARY_PREFIX):].strip() for message in state.messages
                               if isinstance(message.get("content"), str) and message["content"].startswith(SUMMARY_PREFIX))
                checkpoint = store.get("agent_checkpoints", "synthetic-validation")["checkpoint"]
                canonical_messages(state.messages)
                checks = {"one_compaction": state.compactions == 1,
                    "one_summary_request": len(proxy.calls) == 1,
                    "summary_preserves_orbit": "ORBIT" in summary.upper(),
                    "summary_preserves_file": "cache.py" in summary,
                    "literal_goal_preserved": {"role": "user", "content": goal} in state.messages,
                    "pending_input_preserved": state.pending_inputs == pending,
                    "recent_tool_pair_preserved": all(any(message.get("tool_call_id") == ident for message in state.messages)
                        for ident in ("synthetic-read-2", "synthetic-read-3")),
                    "output_headroom_preserved": result["after_tokens"] + request.max_tokens + 32 <= SIMULATED_CAPACITY,
                    "summary_checkpointed": any(message.get("content", "") == SUMMARY_PREFIX + "\n" + summary
                        for message in checkpoint["state"]["messages"]),
                    "raw_reasoning_not_checkpointed": all(value not in json.dumps(checkpoint)
                        for value in ('"reasoning_content"', '"reasoning_turns"')),
                    "operation_ledger_unchanged": original_ledger == digest({"operations": state.operation_ids, "calls": {
                        key: [signature, value.to_dict()] for key, (signature, value) in state.calls.items()}})}
                final_model_response = await client.get(args.api_url.rstrip("/") + "/models/current", headers=headers)
                final_model_response.raise_for_status()
                final_model = final_model_response.json()
                final_status_response = await client.get(args.api_url.rstrip("/") + "/system/status", headers=headers)
                final_status_response.raise_for_status()
                checks["loaded_profile_unchanged"] = (final_model["id"] == actual["id"]
                    and final_model["capabilities"]["profile_revision"] == proxy.snapshot.profile_revision
                    and final_status_response.json().get("active_adapter") == status.get("active_adapter")
                    and final_model.get("effective_context_length") == actual.get("effective_context_length"))
                receipt = {"status": "passed" if all(checks.values()) else "failed", "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_id": actual["id"], "profile_revision": proxy.snapshot.profile_revision,
                    "adapter_id": status.get("active_adapter"), "dialect": args.dialect,
                    "actual_context_tokens": actual_capacity, "simulated_context_tokens": SIMULATED_CAPACITY,
                    "simulation_notice": "Only the validation manager used a 12K limit; the live model allocation was unchanged.",
                    "before_estimated_tokens": before, "after_estimated_tokens": result["after_tokens"],
                    "counter_method": "conservative_estimate", "summary_characters": len(summary),
                    "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(), "checks": checks,
                    "provider_calls": proxy.calls, "raw_reasoning_persisted": False,
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
                args.output.chmod(0o600)
                return receipt
            finally:
                await agent.close()
                store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dialect", choices=("chat_completions", "anthropic_messages", "responses"), default="chat_completions")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, default=PROJECT / "var/validation/live-compaction-receipt.json")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and inspect the synthetic fixture without inference")
    args = parser.parse_args()
    receipt = asyncio.run(validate(args))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] in {"passed", "prepared_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
