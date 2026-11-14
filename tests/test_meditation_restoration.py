"""Consent-aware restoration uses tiny artifacts and never starts inference.

Candidate admission fields are fixture data; these tests prove restoration and
registry behavior, not native training quality or successful head admission.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from lucifer.adaptive import atomic_json, file_sha256
from lucifer.chat import ChatService
from lucifer.config import Settings
from lucifer.inference import ModelSpec
from lucifer.meditation import MeditationService
from lucifer.meditation_sources import prepare_meditation_sources
from lucifer.store import Store


class RestorationManager:
    def __init__(self, original):
        self.specs = {original.id: original}
        self.selected = {"id": original.id, "adapter_id": None}
        self.loads, self.adapter_selections = [], []
        self.unloads = 0
        self.fail_load = None

    def current(self):
        return dict(self.selected) if self.selected else None

    async def load(self, ident):
        self.loads.append(ident)
        if self.fail_load == ident:
            self.selected = None
            raise RuntimeError("Owned fallback loader failed")
        self.native_base_validator(self.specs[ident])
        self.selected = {"id": ident, "adapter_id": None}
        return self.current()

    async def activate_adapter(self, adapter):
        if adapter and (adapter.get("source_revoked") or adapter.get("source_invalidated")):
            raise ValueError("Revoked expert must never be restored")
        ident = adapter["id"] if adapter else None
        self.adapter_selections.append(ident)
        if self.selected:
            self.selected["adapter_id"] = ident

    async def unload(self):
        self.unloads += 1
        self.selected = None


@pytest.fixture
async def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "missing.yaml")
    store = Store(settings.data_dir / "state.sqlite")
    original = ModelSpec(id="original", backend="strata", path=str(tmp_path / "original-pack"),
                         runtime_command=["/bin/true"])
    manager = RestorationManager(original)
    service = MeditationService(settings, store, manager, SimpleNamespace(lock=asyncio.Lock(), busy=False))
    chat = ChatService(store, manager)
    cid = chat.create(training_enabled=True)["id"]
    store.put("messages", {"id": "public-source", "conversation_id": cid, "role": "user",
                           "content": "A saved public text used only by restoration test fixtures."})
    bundle = await prepare_meditation_sources(store, {"mode": "unsupervised"})
    sources = settings.data_dir / "fixture-sources.json"
    atomic_json(sources, bundle)
    context = SimpleNamespace(settings=settings, store=store, original=original, manager=manager,
                              service=service, chat=chat, source_id=cid, source_path=sources)
    yield context
    await service.close()
    store.close()


def candidate(env, name, parent, *, parent_base=None, revoked=False, status="archived"):
    pack = env.settings.data_dir / "fixture-packs" / name
    pack.mkdir(parents=True)
    head = pack / "head.bin"
    head.write_bytes(("tiny restoration head " + name).encode())
    revision = file_sha256(head)
    spec = ModelSpec(id=name, backend="strata", path=str(pack), runtime_command=["/bin/true"],
                     base_revision=revision, parent_model_id=parent)
    env.manager.specs[name] = spec
    record = env.store.put("base_candidates", {"id": "base_" + name, "model_id": name,
        "parent_model_id": parent, "parent_base_id": parent_base, "status": status,
        "source_revoked": revoked, "source_invalidated": revoked, "head_sha256": revision,
        "source_cache_path": str(env.source_path), "source_bundle_sha256": file_sha256(env.source_path),
        "source_conversation_ids": [env.source_id], "spec": spec.model_dump(),
        "admission_report": {"passed": True}, "rollback_receipt": {"verified": True},
        "artifact_files": [{"path": str(head), "sha256": revision}]})
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["revoked", "deleted"])
async def test_dry_run_restores_without_revoked_or_deleted_previous_expert(env, state):
    if state == "revoked":
        env.store.put("adapters", {"id": "old_expert", "status": "active", "base_model_id": "original",
            "source_revoked": True, "source_invalidated": True})
    env.store.set_setting("active_adapter", "old_expert")
    env.manager.selected["adapter_id"] = "old_expert"
    run = await env.service.start({"base_model": "original", "mode": "unsupervised", "dry_run": True,
                                   "idempotency_key": "restore-prior-expert"})
    await env.service.tasks[run["id"]]
    result = env.store.get("meditation_runs", run["id"])
    assert result["status"] == "dry_run" and result["progress"]["restored"] is True
    assert env.manager.current() == {"id": "original", "adapter_id": None}
    assert env.manager.adapter_selections == [None]
    assert env.store.setting("active_adapter") is None
    audit = env.store.list("audit", action="meditation_restoration_source_fallback")
    assert len(audit) == 1 and audit[0]["requested_adapter_id"] == "old_expert"
    assert audit[0]["restored_adapter_id"] is None


@pytest.mark.asyncio
async def test_revoked_native_chain_skips_invalid_loads_and_restores_original_ancestor(env):
    first = candidate(env, "first", "original", revoked=True)
    second = candidate(env, "second", "first", parent_base=first["id"], revoked=True, status="active")
    env.manager.selected = {"id": "second", "adapter_id": None}
    env.store.set_setting("active_base_revision", second["id"])
    receipt = await env.service._restore(env.manager.current(), None)
    assert receipt == {"verified": True, "model_id": "original", "adapter_id": None,
                       "source_consent_overrode_original_selection": True}
    assert env.manager.loads == ["original"]
    assert env.store.setting("active_base_revision") is None
    assert env.store.get("base_candidates", second["id"])["status"] == "archived"
    assert Path(second["artifact_files"][0]["path"]).is_file()
    audit = env.store.list("audit", action="meditation_restoration_source_fallback")[-1]
    assert audit["requested_model_id"] == "second" and audit["restored_model_id"] == "original"


@pytest.mark.asyncio
async def test_revoked_child_restores_valid_native_parent_with_consistent_active_registry(env):
    first = candidate(env, "valid-parent", "original")
    second = candidate(env, "revoked-child", "valid-parent", parent_base=first["id"], revoked=True, status="active")
    env.service.validate_profile(env.manager.specs["valid-parent"])
    env.manager.selected = {"id": "revoked-child", "adapter_id": None}
    env.store.set_setting("active_base_revision", second["id"])
    receipt = await env.service._restore(env.manager.current(), None)
    assert receipt["verified"] and receipt["model_id"] == "valid-parent"
    assert env.manager.loads == ["valid-parent"]
    assert env.store.setting("active_base_revision") == first["id"]
    assert env.store.get("base_candidates", first["id"])["status"] == "active"
    assert env.store.get("base_candidates", second["id"])["status"] == "archived"


@pytest.mark.asyncio
async def test_idle_invalidated_active_base_is_restored_without_a_running_job(env):
    invalid = candidate(env, "invalid-active", "original", revoked=True, status="active")
    env.manager.selected = {"id": "invalid-active", "adapter_id": None}
    env.store.set_setting("active_base_revision", invalid["id"])
    assert not env.service.tasks and not env.service.busy
    await env.service.cancel_invalidated()
    assert env.manager.loads == ["original"]
    assert env.manager.current()["id"] == "original"
    assert env.store.setting("active_base_revision") is None
    assert env.store.get("base_candidates", invalid["id"])["status"] == "archived"
    assert env.store.list("audit", action="meditation_restoration_source_fallback")


@pytest.mark.asyncio
async def test_idle_valid_base_is_preserved(env):
    valid = candidate(env, "valid-active", "original", status="active")
    env.manager.selected = {"id": "valid-active", "adapter_id": None}
    env.store.set_setting("active_base_revision", valid["id"])
    await env.service.cancel_invalidated()
    assert env.manager.loads == []
    assert env.store.setting("active_base_revision") == valid["id"]
    assert env.store.get("base_candidates", valid["id"])["status"] == "active"


@pytest.mark.asyncio
async def test_failed_consent_fallback_is_not_reported_as_restored(env):
    invalid = candidate(env, "failed-parent", "original", revoked=True, status="active")
    env.manager.selected = {"id": "failed-parent", "adapter_id": None}
    env.manager.fail_load = "original"
    env.store.set_setting("active_base_revision", invalid["id"])
    run = await env.service.start({"base_model": "original", "mode": "unsupervised", "dry_run": True,
                                   "idempotency_key": "failed-consent-fallback"})
    await env.service.tasks[run["id"]]
    result = env.store.get("meditation_runs", run["id"])
    assert result["status"] == "failed" and result["progress"]["restored"] is False
    assert "restoration failed" in result["error"]
    assert env.manager.loads == ["original"] and env.manager.current() is None
    assert env.store.list("audit", action="meditation_restoration_source_fallback") == []


@pytest.mark.asyncio
async def test_missing_registered_ancestor_unloads_instead_of_loading_revoked_weights(env):
    invalid = candidate(env, "orphan", "missing-model", revoked=True, status="active")
    env.manager.selected = {"id": "orphan", "adapter_id": None}
    env.store.set_setting("active_base_revision", invalid["id"])
    result = await env.service._restore(env.manager.current(), None)
    assert result["verified"] and result["model_id"] is None
    assert env.manager.loads == [] and env.manager.unloads == 1
    assert env.manager.current() is None and env.store.setting("active_base_revision") is None
