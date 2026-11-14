"""Loopback-first HTTP API used by the terminal client."""
import asyncio
from contextlib import aclosing, asynccontextmanager
import hmac
import json

import httpx

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from . import __version__
from .chat import ChatService
from .config import Settings
from .inference import ModelSpec
from .schemas import (AdapterOptional, AdapterRequest, AgentApproval, AgentInput, AgentMemoryRequest, AgentRequest, AgentResume, CancelRequest, ChatRequest,
                      ConversationCreate, ConversationPatch, FeedbackRequest,
                      ExampleReview, ExpertBuild, ExpertProposal, KnowledgeAttachment, MeditationMutation, MeditationResume,
                      MeditationRequest, MeditationRollback, ModelLoad, ModelUnload, SleepRequest, SleepResumeRequest, TrainRequest)
from .store import Store
from .telemetry import TelemetryService

TERMINAL_STATES = {"pilot_complete", "completed", "failed", "cancelled", "interrupted", "dry_run", "awaiting_promotion", "rejected", "succeeded", "paused", "blocked", "admitted", "awaiting_evidence", "unreachable_gate", "awaiting_input", "quality_unknown", "no_new_information", "insufficient_verified_data", "needs_review"}


def create_app(settings: Settings | None = None, *, autoload: str | None = None):
    settings = settings or Settings()
    restore_base_selection = autoload is None
    autoload = autoload if autoload is not None else settings.default_model

    @asynccontextmanager
    async def lifespan(app):
        from .inference import ModelManager
        from .learning import LearningService
        import fcntl
        lockfile = (settings.data_dir / "backend.lock").open("a+")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lockfile.close()
            raise RuntimeError("Another Lucifer backend is using this data directory") from None
        store = Store(settings.data_dir / "lucifer.sqlite3")
        try:
            manager = ModelManager(settings, store)
        except BaseException:
            store.close()
            lockfile.close()
            raise
        learning = LearningService(settings, store, manager)
        from .meditation import MeditationService
        meditation = MeditationService(settings, store, manager, learning)
        app.state.store, app.state.manager = store, manager
        app.state.learning = learning
        app.state.meditation = meditation
        app.state.chat = ChatService(store, manager)
        from .memory import reconcile_memory
        reconcile_memory(store)
        app.state.control_lock = asyncio.Lock()
        app.state.ui_cache = None
        app.state.ui_revision = 0
        telemetry = None
        agent = None
        try:
            from .agent import AgentService
            agent = AgentService(settings, store, manager, app.state.chat)
            app.state.agent = agent
            if settings.agent_mcp_file is not None:
                import yaml
                configuration = yaml.safe_load(settings.agent_mcp_file.read_text()) or {}
                if not isinstance(configuration, dict) or not isinstance(configuration.get("servers"), list):
                    raise ValueError("MCP config must contain an explicit servers list")
                await agent.tools.connect_mcp(configuration["servers"])
            if autoload:
                selected = autoload
                if restore_base_selection and store.setting("active_base_revision"):
                    candidate = store.get("base_candidates", store.setting("active_base_revision"))
                    selected = candidate["model_id"]
                await manager.load(selected)
                await restore_selection(store, manager)
            telemetry = TelemetryService(getattr(manager, "gpu_inventory", None))
            await telemetry.start()
            manager.gpu_inventory = telemetry.inventory
            app.state.telemetry = telemetry
            ui_snapshot()
            yield
        finally:
            if agent is not None:
                await agent.close()
            if telemetry is not None:
                await telemetry.close()
            await meditation.close()
            await learning.close()
            await manager.close()
            store.close()
            lockfile.close()

    async def authorize(request: Request):
        if settings.api_token:
            actual = request.headers.get("authorization", "")
            if not hmac.compare_digest(actual, "Bearer " + settings.api_token):
                raise HTTPException(401, "Invalid API token")

    app = FastAPI(title="Lucifer", version=__version__, lifespan=lifespan,
                  dependencies=[Depends(authorize)])

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def not_installed(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(httpx.HTTPError)
    async def upstream_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": f"Inference backend request failed: {exc}"})

    def ui_snapshot():
        """Rebuild registry/status panels only after an actual store change."""
        store = app.state.store
        app.state.manager.check_process()
        revision = store.revision("models", "adapters", "training_runs", "sleep_runs", "settings", "agent_runs",
                                  "meditation_runs", "base_candidates")
        cached = app.state.ui_cache
        if cached is not None and cached[0] == revision:
            return cached[1]
        previous = cached[1] if cached else {}
        def changed(index):
            return cached is None or cached[0][index] != revision[index]
        jobs = {
            "train": store.list("training_runs") if changed(2) else previous["jobs"]["train"],
            "sleep": store.list("sleep_runs") if changed(3) else previous["jobs"]["sleep"],
            "meditation": store.list("meditation_runs") if changed(6) else previous["jobs"]["meditation"],
        }
        agent_runs = app.state.agent.summaries() if changed(5) else previous["agent_runs"]
        current = app.state.manager.current() if changed(0) else previous["status"]["model"]
        capability = None
        if changed(0):
            registry = getattr(app.state.manager, "capabilities", None)
            capability = registry.get(current["id"]) if registry and current else None
        from .model_compatibility import decorate_adapters
        adapters = decorate_adapters(current, store.list("adapters")) if changed(0) or changed(1) else previous["adapters"]
        snapshot = {
            "models": app.state.manager.list() if changed(0) else previous["models"],
            "adapters": adapters,
            "base_candidates": store.list("base_candidates") if changed(7) else previous["base_candidates"],
            "capabilities": (capability.as_dict() if capability else {}) if changed(0) else previous["capabilities"],
            "jobs": jobs,
            "agent_runs": agent_runs,
            "status": {"version": __version__, "model": current,
                       "default_model": settings.default_model,
                       "active_adapter": store.setting("active_adapter") if changed(4) else previous["status"]["active_adapter"],
                       "previous_adapter": store.setting("previous_adapter") if changed(4) else previous["status"]["previous_adapter"],
                       "active_base_revision": store.setting("active_base_revision"),
                       "previous_base_revision": store.setting("previous_base_revision"),
                       "previous_base_id": store.setting("previous_base_id"),
                       "agent_busy": any(run.get("status") not in TERMINAL_STATES for run in agent_runs),
                       "agent_workspace": str(settings.agent_workspace),
                       "jobs": [r for runs in jobs.values() for r in runs if r.get("status") not in TERMINAL_STATES],
                       "data_dir": str(settings.data_dir), "models_file": str(settings.models_file),
                       "privacy_defaults": {"logging_enabled": True, "training_enabled": False}},
        }
        # Job progress travels in telemetry itself. It must not cause another
        # bootstrap or rebuild the model and adapter selection controls.
        if any(changed(index) for index in (0, 1, 4, 7)):
            app.state.ui_revision += 1
        snapshot["revision"] = app.state.ui_revision
        app.state.ui_cache = (revision, snapshot)
        return snapshot

    def busy():
        # A terminal job record can precede its finally block's model restore.
        # Keep controls locked until the owned learning task actually exits.
        return app.state.learning.busy or app.state.meditation.busy or any(r.get("status") not in TERMINAL_STATES
            for runs in ui_snapshot()["jobs"].values() for r in runs)

    def runtime_status(snapshot):
        # Task flags change without a structural registry revision. Overlay
        # these cheap in-memory values on the cached status; never rediscover.
        return {**snapshot, "learning_busy": app.state.learning.busy or app.state.meditation.busy,
                "meditation_busy": app.state.meditation.busy,
                "chat_busy": app.state.chat.busy,
                "agent_busy": app.state.agent.busy}

    def require_idle(*, allow_chat=False):
        if app.state.agent.busy:
            raise HTTPException(409, "An agent run is active. Wait or cancel it first.")
        if not allow_chat and app.state.chat.busy:
            raise HTTPException(409, "A chat turn is active. Wait for it to finish.")
        if busy():
            raise HTTPException(409, "A learning job is running. Wait or cancel it first.")

    def require_source_revocation():
        """Consent withdrawal may cancel only affected owned Meditation work."""
        if not app.state.meditation.busy:
            require_idle(allow_chat=True)
            return
        if app.state.agent.busy:
            raise HTTPException(409, "An agent run is active. Wait or cancel it first.")
        ordinary = ui_snapshot()["jobs"]
        if app.state.learning.busy or any(run.get("status") not in TERMINAL_STATES
                for kind in ("train", "sleep") for run in ordinary[kind]):
            raise HTTPException(409, "A Sleep or training job is running. Wait or cancel it first.")

    async def detach_invalid_adapter():
        store = app.state.store
        ident = store.setting("active_adapter")
        if not ident:
            return
        adapter = store.get("adapters", ident)
        if not (adapter.get("source_revoked") or adapter.get("source_invalidated")):
            return
        store.set_setting("active_adapter", None)
        store.update("adapters", ident, status="failed")
        store.audit("invalidated_adapter_detached", adapter_id=ident)
        current = app.state.manager.current()
        if current and current.get("id") == adapter.get("base_model_id"):
            await app.state.manager.activate_adapter(None)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/system/status")
    async def system_status():
        return runtime_status(ui_snapshot()["status"])

    @app.get("/system/bootstrap")
    async def bootstrap():
        snapshot = ui_snapshot()
        return {**snapshot, "status": runtime_status(snapshot["status"]),
                "gpu": app.state.telemetry.snapshot()}

    @app.get("/system/telemetry")
    async def telemetry():
        snapshot = ui_snapshot()
        return {"revision": snapshot["revision"], "status": runtime_status(snapshot["status"]),
                "jobs": snapshot["jobs"], "agent_runs": snapshot["agent_runs"],
                "gpu": app.state.telemetry.snapshot()}

    @app.get("/agent/tools")
    async def agent_tools():
        return app.state.agent.tools.describe()

    @app.get("/agent/runs")
    async def agent_runs():
        return app.state.agent.summaries()

    @app.get("/agent/history")
    async def agent_history():
        return app.state.agent.history()

    @app.post("/agent/runs", status_code=202)
    async def start_agent(body: AgentRequest):
        async with app.state.control_lock:
            # Idempotent HTTP retries resolve to the same run, even while it is active.
            if busy():
                raise HTTPException(409, "A learning job is running. Wait or cancel it first.")
            return await app.state.agent.start(body)

    @app.get("/agent/runs/{run_id}")
    @app.get("/agent/runs/{run_id}/export")
    async def agent_run(run_id: str):
        return app.state.agent.get(run_id)

    @app.get("/agent/runs/{run_id}/session")
    async def agent_session(run_id: str):
        return app.state.agent.session(run_id)

    @app.post("/agent/runs/{run_id}/cancel")
    async def cancel_agent(run_id: str):
        return await app.state.agent.cancel(run_id)

    @app.post("/agent/runs/{run_id}/stop")
    async def stop_agent(run_id: str):
        return await app.state.agent.stop(run_id)

    @app.post("/agent/runs/{run_id}/input")
    async def input_agent(run_id: str, body: AgentInput):
        async with app.state.control_lock:
            if busy():
                raise HTTPException(409, "A learning job is running. Wait or cancel it first.")
            return await app.state.agent.submit_input(run_id, body.content,
                interrupt=body.interrupt, options=body.options, idempotency_key=body.idempotency_key)

    @app.post("/agent/runs/{run_id}/resume")
    async def resume_agent(run_id: str, body: AgentResume):
        async with app.state.control_lock:
            if busy():
                raise HTTPException(409, "A learning job is running. Wait or cancel it first.")
            return await app.state.agent.resume(run_id, body.instruction, limits=body.limits,
                                                idempotency_key=body.idempotency_key)

    @app.post("/agent/runs/{run_id}/approve")
    async def approve_agent(run_id: str, body: AgentApproval):
        return await app.state.agent.approve(run_id, body.approval_id, body.approved)

    @app.post("/agent/runs/{run_id}/remember")
    async def remember_agent(run_id: str, body: AgentMemoryRequest):
        return app.state.agent.remember(run_id, body.summary)

    @app.get("/agent/runs/{run_id}/events")
    async def agent_events(run_id: str, request: Request, after: int = 0):
        app.state.agent.get(run_id)
        try:
            cursor = max(0, int(request.headers.get("last-event-id", after)))
        except ValueError:
            raise HTTPException(400, "Event cursor must be a nonnegative integer") from None
        async def stream_events():
            async with aclosing(app.state.agent.events(run_id, cursor)) as events:
                async for event, data in events:
                    if await request.is_disconnected():
                        return
                    yield f"id: {data.get('sequence', 0)}\n" + sse(event, data)
        return StreamingResponse(stream_events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/system/gpu")
    async def gpu():
        return app.state.telemetry.snapshot()

    @app.get("/models")
    async def models():
        return app.state.manager.list()

    @app.get("/models/current")
    async def current():
        return app.state.manager.current()

    @app.get("/models/{model_id}/status")
    async def model_status(model_id: str):
        for model in app.state.manager.list():
            if model["id"] == model_id:
                return model
        raise KeyError(model_id)

    @app.get("/models/{model_id}/capabilities")
    async def model_capabilities(model_id: str):
        if model_id not in app.state.manager.specs:
            raise KeyError(model_id)
        registry = getattr(app.state.manager, "capabilities", None)
        snapshot = registry.get(model_id) if registry else None
        return snapshot.as_dict() if snapshot else {
            "model_id": model_id, "discovery_state": "not_initialized", "features": {},
        }

    @app.get("/models/{model_id}/logs")
    async def model_logs(model_id: str):
        if not any(m["id"] == model_id for m in app.state.manager.list()):
            raise KeyError(model_id)
        # IDs are validated by ModelSpec; no arbitrary server file reads.
        path = settings.data_dir / "logs" / f"model-{model_id}.log"
        if not path.exists():
            return {"model_id": model_id, "logs": "", "note": "No managed process logs"}
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            return {"model_id": model_id, "logs": handle.read().decode(errors="replace")}

    @app.post("/models/load")
    async def load(body: ModelLoad, request: Request):
        async with app.state.control_lock:
            require_idle()
            if body.spec is not None:
                # Older backends ignore this header and still accept the
                # resolved spec. Current backends retain file provenance after
                # checking that this snapshot still matches the actual file.
                source = "file" if request.headers.get("X-Lucifer-Profile-Source") == "file" else "runtime"
                app.state.manager.register(body.spec, source=source)
            await app.state.manager.load(body.model_id)
            await restore_selection(app.state.store, app.state.manager)
            return app.state.manager.current()

    @app.post("/models/register")
    async def register_model(body: ModelSpec):
        async with app.state.control_lock:
            require_idle()
            return app.state.manager.register(body)

    @app.post("/models/reload")
    async def reload_models():
        async with app.state.control_lock:
            require_idle()
            return await app.state.manager.reload()

    @app.post("/models/unload")
    async def unload(body: ModelUnload):
        async with app.state.control_lock:
            require_idle()
            await app.state.manager.unload(body.model_id)
            return {"status": "unloaded"}

    @app.post("/conversations", status_code=201)
    async def create_conversation(body: ConversationCreate):
        return app.state.chat.create(**body.model_dump())

    @app.get("/conversations")
    async def conversations():
        return app.state.store.list("conversations")

    @app.get("/conversations/{conversation_id}")
    @app.get("/conversations/{conversation_id}/export")
    async def conversation(conversation_id: str):
        return app.state.chat.get(conversation_id)

    @app.patch("/conversations/{conversation_id}")
    async def update_conversation(conversation_id: str, body: ConversationPatch):
        async with app.state.control_lock:
            changes = body.model_dump(exclude_none=True)
            # Explicitly saving public context only adds logging consent. It
            # neither cancels the owned request nor opts any data into training.
            revoking = bool(changes) and set(changes) <= {"logging_enabled", "training_enabled"} and all(
                value is False for value in changes.values())
            if revoking:
                require_source_revocation()
            elif changes != {"logging_enabled": True}:
                require_idle()
            async with app.state.chat.locks[conversation_id]:
                result = app.state.chat.patch(conversation_id, changes)
                if revoking:
                    await app.state.meditation.cancel_invalidated()
                await detach_invalid_adapter()
                return result

    @app.get("/conversations/{conversation_id}/knowledge")
    async def knowledge_documents(conversation_id: str):
        return app.state.chat.knowledge.list(conversation_id)

    @app.post("/conversations/{conversation_id}/knowledge", status_code=201)
    async def attach_knowledge(conversation_id: str, body: KnowledgeAttachment):
        async with app.state.control_lock:
            require_idle()
            async with app.state.chat.locks[conversation_id]:
                return await app.state.chat.knowledge.attach(conversation_id, **body.model_dump(exclude_none=True))

    @app.get("/conversations/{conversation_id}/knowledge/search")
    async def search_knowledge(conversation_id: str, query: str = "", limit: int = 5):
        if not 1 <= limit <= 20 or len(query) > 2000:
            raise ValueError("Search limit must be 1..20 and query no longer than 2000 characters")
        return app.state.chat.knowledge.search(conversation_id, query, limit)

    @app.get("/conversations/{conversation_id}/knowledge/{document_id}")
    async def read_knowledge(conversation_id: str, document_id: str, chunk: int = 0, max_chunks: int = 1):
        if chunk < 0 or not 1 <= max_chunks <= 5:
            raise ValueError("Chunk must be nonnegative and max_chunks 1..5")
        return app.state.chat.knowledge.read(conversation_id, document_id, chunk, max_chunks)

    @app.delete("/conversations/{conversation_id}/knowledge/{document_id}")
    async def remove_knowledge(conversation_id: str, document_id: str):
        async with app.state.control_lock:
            require_source_revocation()
            async with app.state.chat.locks[conversation_id]:
                result = app.state.chat.knowledge.delete(conversation_id, document_id)
                await app.state.meditation.cancel_invalidated()
                await detach_invalid_adapter()
                return result

    @app.delete("/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        async with app.state.control_lock:
            require_source_revocation()
            async with app.state.chat.locks[conversation_id]:
                app.state.chat.delete(conversation_id)
                await app.state.meditation.cancel_invalidated()
                await detach_invalid_adapter()
        return {"status": "deleted"}

    def prepare_chat(body):
        require_idle(allow_chat=True)
        if app.state.control_lock.locked():
            raise HTTPException(409, "Model control operation in progress")
        if not app.state.manager.current():
            raise HTTPException(409, "Load a model first with /model load <id>")
        if body.model:
            current = app.state.manager.current()
            if body.model not in {current["id"], current.get("display_name", "Lucifer")}:
                raise ValueError("Requested model is not loaded")
        if body.conversation_id:
            app.state.store.get("conversations", body.conversation_id)
            return body.conversation_id
        return app.state.chat.create()["id"]

    @app.post("/chat")
    async def chat(body: ChatRequest):
        conversation_id = prepare_chat(body)
        text, done = "", {}
        async with aclosing(app.state.chat.turn(body, conversation_id)) as turn:
            async for event, data in turn:
                if event == "delta":
                    text += data["content"]
                elif event == "done":
                    done = data
        return {**done, "choices": [{"message": {"role": "assistant", "content": text}}]}

    @app.post("/chat/stream")
    async def stream(body: ChatRequest, request: Request):
        conversation_id = prepare_chat(body)
        async def events():
            meta = {"conversation_id": conversation_id}
            if body.thinking is not None:
                meta["thinking"] = body.thinking
            yield sse("meta", meta)
            try:
                async with aclosing(app.state.chat.turn(body, conversation_id)) as turn:
                    async for event, data in turn:
                        if await request.is_disconnected():
                            return
                        yield sse(event, data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield sse("error", {"detail": str(exc)})
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/feedback", status_code=201)
    async def feedback(body: FeedbackRequest):
        async with app.state.control_lock:
            require_idle()
            async with app.state.chat.locks[body.conversation_id]:
                conv = app.state.store.get("conversations", body.conversation_id)
                if not conv["logging_enabled"]:
                    raise ValueError("Save this conversation or enable logging before storing feedback")
                msg = app.state.store.get("messages", body.message_id)
                if msg["conversation_id"] != body.conversation_id or msg["role"] != "assistant":
                    raise ValueError("Feedback must target an assistant message in this conversation")
                from .memory import invalidate_memory_sources, reconcile_memory
                from .learning import scrub
                reviews = app.state.store.list("feedback", conversation_id=body.conversation_id, message_id=body.message_id)
                def reviewed_target(review):
                    correction = scrub(review.get("correction") or "")
                    return scrub(correction or msg["content"]) if correction or review.get("label") in {"good", "user_liked"} else None
                if reviewed_target(reviews[-1] if reviews else {}) != reviewed_target(body.model_dump()):
                    invalidate_memory_sources(app.state.store, body.conversation_id, message_id=body.message_id, reason="feedback_changed")
                    app.state.store.update("messages", body.message_id, learning_revision=int(msg.get("learning_revision", 0)) + 1)
                result = app.state.store.put("feedback", body.model_dump())
                reconcile_memory(app.state.store, [body.conversation_id])
                await detach_invalid_adapter()
                app.state.store.audit("feedback_saved", conversation_id=body.conversation_id, message_id=body.message_id, label=body.label)
        return result

    @app.get("/sleep/examples")
    async def sleep_examples(status: str | None = None, limit: int = 200):
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between one and 1000")
        records = app.state.store.list("curriculum_candidates")
        if status is not None:
            records = [record for record in records if record.get("status") == status]
        return records[-limit:]

    @app.post("/sleep/examples/{candidate_id}/review")
    async def review_sleep_example(candidate_id: str, body: ExampleReview):
        async with app.state.control_lock:
            require_idle()
            from .curriculum import review_candidate
            try:
                result = review_candidate(app.state.store, candidate_id, **body.model_dump())
            except ValueError as exc:
                if any(word in str(exc).casefold() for word in ("changed", "deleted", "superseded", "obsolete", "different review")):
                    raise HTTPException(409, str(exc)) from None
                raise
            from .memory import invalidate_candidate_dependents, reconcile_memory
            if result.get("id") != candidate_id:
                invalidate_candidate_dependents(app.state.store, candidate_id)
                await detach_invalid_adapter()
            reconcile_memory(app.state.store, examples=[result] if result.get("status") == "verified" else [])
            app.state.store.audit("curriculum_reviewed", candidate_id=candidate_id, decision=body.decision,
                                  content_hash=body.content_hash)
            return result

    @app.get("/adapters")
    async def adapters():
        return app.state.store.list("adapters")

    @app.get("/experts")
    async def experts():
        return app.state.learning.list_experts()

    @app.post("/experts/propose")
    async def propose_experts(body: ExpertProposal):
        return app.state.learning.propose_experts(body.conversation_ids, body.base_model)

    @app.post("/experts/build", status_code=202)
    async def build_expert(body: ExpertBuild):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.start_expert(body.learning_request())

    @app.post("/adapters/promote")
    async def promote(body: AdapterRequest):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.promote(body.adapter_id)

    @app.post("/adapters/rollback")
    async def rollback(body: AdapterOptional):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.rollback(body.adapter_id)

    @app.post("/adapters/activate")
    async def activate_expert(body: AdapterRequest):
        async with app.state.control_lock:
            require_idle()
            from .adapter_lifecycle import activate_expert
            return await activate_expert(app.state.learning, body.adapter_id)

    @app.delete("/adapters/{adapter_id}")
    async def delete_adapter(adapter_id: str):
        async with app.state.control_lock:
            require_idle()
            from .adapter_lifecycle import delete_adapter
            return delete_adapter(app.state.learning, adapter_id)

    @app.post("/adapters/select")
    async def select(body: AdapterOptional):
        async with app.state.control_lock:
            require_idle()
            if body.adapter_id:
                adapter = app.state.store.get("adapters", body.adapter_id)
                if adapter.get("status") == "admitted" or (adapter.get("admitted_at") and adapter.get("status") in {"active", "archived"}):
                    return await app.state.learning.activate_admitted(body.adapter_id)
                if adapter.get("status") not in {"active", "archived"} or not adapter.get("metrics", {}).get("passed"):
                    raise ValueError("Select an admitted or previously promoted adapter, or promote an evaluated candidate")
                return await app.state.learning.rollback(body.adapter_id)
            old = app.state.store.setting("active_adapter")
            await app.state.manager.activate_adapter(None)
            if old:
                app.state.store.update("adapters", old, status="archived")
            app.state.store.set_setting("previous_adapter", old)
            app.state.store.set_setting("active_adapter", None)
            app.state.store.audit("base_selected")
            return {"active_adapter": None, "status": "base"}

    @app.post("/train/start", status_code=202)
    async def start_train(body: TrainRequest):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.start_train(body.model_dump(exclude_none=True))

    @app.get("/train")
    async def train_list():
        return app.state.store.list("training_runs")

    @app.get("/train/{run_id}")
    async def train_get(run_id: str):
        return app.state.store.get("training_runs", run_id)

    @app.get("/train/{run_id}/logs")
    async def train_logs(run_id: str):
        return app.state.store.get("training_runs", run_id).get("logs", [])

    @app.post("/train/{run_id}/cancel")
    async def train_cancel(run_id: str):
        return await app.state.learning.cancel(run_id)

    @app.post("/sleep/start", status_code=202)
    async def start_sleep(body: SleepRequest):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.start_sleep(body.model_dump(exclude_none=True))

    @app.get("/sleep/runs")
    async def sleep_runs():
        return app.state.store.list("sleep_runs")

    @app.post("/meditation/start", status_code=202)
    async def meditation_start(body: MeditationRequest):
        async with app.state.control_lock:
            if app.state.agent.busy:
                raise HTTPException(409, "An agent run is active. Wait or pause it first.")
            if app.state.chat.busy:
                raise HTTPException(409, "A chat turn is active. Wait for it to finish.")
            return await app.state.meditation.start(body.model_dump(exclude_none=True))

    @app.get("/meditation/runs")
    async def meditation_runs():
        return app.state.store.list("meditation_runs")

    @app.get("/meditation/runs/{run_id}")
    async def meditation_run(run_id: str):
        return app.state.store.get("meditation_runs", run_id)

    @app.post("/meditation/runs/{run_id}/resume", status_code=202)
    async def meditation_resume(run_id: str, body: MeditationResume):
        async with app.state.control_lock:
            if app.state.agent.busy:
                raise HTTPException(409, "An agent run is active. Wait or pause it first.")
            if app.state.chat.busy:
                raise HTTPException(409, "A chat turn is active. Wait for it to finish.")
            return await app.state.meditation.resume(run_id, body.idempotency_key,
                **body.model_dump(exclude_none=True, exclude={"idempotency_key"}))

    @app.post("/meditation/runs/{run_id}/cancel")
    async def meditation_cancel(run_id: str, body: MeditationMutation):
        return await app.state.meditation.cancel(run_id, body.idempotency_key)

    @app.get("/meditation/bases")
    async def meditation_bases():
        return app.state.store.list("base_candidates")

    @app.post("/meditation/bases/{base_id}/activate")
    async def meditation_activate(base_id: str, body: MeditationMutation):
        async with app.state.control_lock:
            require_idle()
            return await app.state.meditation.activate(base_id, body.idempotency_key)

    @app.post("/meditation/rollback")
    async def meditation_rollback(body: MeditationRollback):
        async with app.state.control_lock:
            require_idle()
            return await app.state.meditation.rollback(body.idempotency_key, body.base_id)

    @app.get("/sleep/runs/{run_id}")
    async def sleep_run(run_id: str):
        return app.state.store.get("sleep_runs", run_id)

    @app.post("/sleep/runs/{run_id}/resume", status_code=202)
    async def resume_sleep(run_id: str, body: SleepResumeRequest):
        async with app.state.control_lock:
            if app.state.agent.busy:
                raise HTTPException(409, "An agent run is active. Wait or pause it first.")
            if app.state.chat.busy:
                raise HTTPException(409, "A chat turn is active. Wait for it to finish.")
            if app.state.meditation.busy:
                raise HTTPException(409, "A Meditation job is running. Wait or pause it first.")
            # The service replays the durable receipt before checking whether
            # learning is busy; reconnecting never starts another training run.
            return await app.state.learning.resume_sleep(run_id, idempotency_key=body.idempotency_key)

    @app.get("/sleep/status")
    async def sleep_status():
        runs = app.state.store.list("sleep_runs")
        return runs[-1] if runs else {"status": "idle"}

    @app.get("/sleep/queue")
    async def sleep_queue():
        return [r for r in app.state.store.list("sleep_runs") if r.get("status") not in TERMINAL_STATES]

    @app.post("/sleep/cancel")
    async def sleep_cancel(body: CancelRequest):
        runs = app.state.store.list("sleep_runs")
        ident = body.run_id or next((r["id"] for r in reversed(runs) if r.get("status") not in TERMINAL_STATES), None)
        if not ident:
            raise ValueError("No sleep job is running")
        return await app.state.learning.cancel(ident)

    @app.post("/eval/run")
    async def evaluate(body: AdapterRequest):
        async with app.state.control_lock:
            require_idle()
            return await app.state.learning.evaluate(body.adapter_id)

    @app.get("/eval/results")
    async def eval_results():
        return app.state.store.list("evaluation_results")

    @app.get("/memory")
    async def memory(q: str = ""):
        return app.state.chat.memory(q)

    @app.get("/audit")
    async def audit():
        return app.state.store.list("audit")[-200:]

    return app


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def restore_selection(store, manager):
    """Keep the active pointer and displayed runtime selection consistent."""
    active = store.setting("active_adapter")
    if not active:
        return
    adapter = store.get("adapters", active)
    if adapter.get("source_revoked") or adapter.get("source_invalidated"):
        store.update("adapters", active, status="failed")
        store.set_setting("active_adapter", None)
        store.audit("invalidated_adapter_detached", adapter_id=active)
        return
    current = manager.current()
    if current and adapter["base_model_id"] == current["id"]:
        await manager.activate_adapter(adapter)
    else:
        store.update("adapters", active, status="archived")
        store.set_setting("previous_adapter", active)
        store.set_setting("active_adapter", None)
        store.audit("adapter_archived_on_model_switch", adapter_id=active)
