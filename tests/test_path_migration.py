"""Relocation changes operational references without altering frozen evidence."""
import fcntl
import json
from pathlib import Path
import sqlite3

import pytest

from lucifer.path_migration import PathMigration, path_replacement


@pytest.fixture
def migration(tmp_path):
    legacy = tmp_path / "old-home"
    canonical = tmp_path / "Lucifer"
    project = canonical / "Soul"
    config = project / "configs/models.yaml"
    config.parent.mkdir(parents=True)
    source_data = legacy / ".local/share/lucifer"
    source_data.mkdir(parents=True)
    model = legacy / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B"
    (model / "blobs").mkdir(parents=True)
    (model / "blobs/weights").write_bytes(b"model-weights")
    snapshot = model / "snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to("../../blobs/weights")
    config.write_text(f"# Preserve this explanation.\nmodels:\n  - id: lucifer-train\n    path: {snapshot}\n    training_model: {snapshot}\n    executable: ../scripts/vllm\n")
    vllm = canonical / "vLLM/Lucifer/bin/vllm"
    vllm.parent.mkdir(parents=True)
    vllm.write_text(f"#!{legacy}/vLLM/Lucifer/bin/python3\nprint('entrypoint')\n")
    vllm.chmod(0o755)
    record = {"id": "lucifer-train", "backend": "vllm", "executable": str(legacy / "vLLM/Lucifer/bin/vllm"),
              "path": str(snapshot), "spec": {"id": "lucifer-train", "backend": "vllm", "executable": str(legacy / "vLLM/Lucifer/bin/vllm")},
              "error": f"historical failure {legacy}/vLLM/Lucifer/bin/vllm"}
    history = {"id": "checkpoint", "sha256": "unchanged-hash", "checkpoint": {"messages": [{"content": str(legacy / "Lucifer/private.py")}], "workspace": str(legacy / "Lucifer")}}
    with sqlite3.connect(source_data / "lucifer.sqlite3") as connection:
        connection.execute("CREATE TABLE documents (kind TEXT,id TEXT,data TEXT,PRIMARY KEY(kind,id))")
        for kind, value in [("models", record), ("agent_checkpoints", history), ("adapters", {"id": "adapter", "path": str(source_data / "adapters/adapter"), "artifact_verification": {"sha256": "keep", "base_model": str(snapshot)}})]:
            connection.execute("INSERT INTO documents VALUES (?,?,?)", (kind, value["id"], json.dumps(value)))
    return PathMigration(canonical_root=canonical, legacy_home=legacy)


def read_rows(database):
    with sqlite3.connect(database) as connection:
        return {(kind, ident): encoded for kind, ident, encoded in connection.execute("SELECT kind,id,data FROM documents")}


def test_dry_run_is_read_only_and_reports_operational_scope(migration):
    before = read_rows(migration.source_data / "lucifer.sqlite3")
    result = migration.plan()
    assert len(result["moves"]) == 2
    assert len(result["files"]) == 2
    assert result["database"]["preserved_historical_records"] == {"agent_checkpoints": 1}
    assert read_rows(migration.source_data / "lucifer.sqlite3") == before
    assert not migration.target_data.exists()
    assert not (migration.project / "var/state/migration-backups").exists()


def test_apply_moves_bytes_keeps_narrow_aliases_and_preserves_evidence(migration):
    before = read_rows(migration.source_data / "lucifer.sqlite3")
    result = migration.apply()
    assert migration.source_data.is_symlink()
    assert migration.source_data.resolve() == migration.target_data
    assert migration.source_model.is_symlink()
    assert (migration.target_model / "snapshots/revision/model.safetensors").read_bytes() == b"model-weights"
    after = read_rows(migration.target_data / "lucifer.sqlite3")
    assert after[("agent_checkpoints", "checkpoint")] == before[("agent_checkpoints", "checkpoint")]
    current = json.loads(after[("models", "lucifer-train")])
    assert current["executable"] == str(migration.project / "scripts/vllm")
    assert current["spec"]["executable"] == str(migration.project / "scripts/vllm")
    assert current["path"].startswith(str(migration.target_model))
    assert current["error"] == json.loads(before[("models", "lucifer-train")])["error"]
    adapter = json.loads(after[("adapters", "adapter")])
    assert adapter["path"].startswith(str(migration.target_data))
    assert adapter["artifact_verification"] == json.loads(before[("adapters", "adapter")])["artifact_verification"]
    assert read_rows(Path(result["backup"]) / "lucifer.sqlite3") == before
    assert not (migration.legacy_home / "Lucifer").exists()
    config = (migration.project / "configs/models.yaml").read_text()
    assert config.startswith("# Preserve this explanation.") and str(migration.target_model) in config
    entrypoint = migration.root / "vLLM/Lucifer/bin/vllm"
    assert entrypoint.read_text().splitlines()[0] == "#!" + str(migration.root / "vLLM/Lucifer/bin/python3")
    assert entrypoint.stat().st_mode & 0o777 == 0o755


def test_migration_is_repeatable_without_merging_or_rewriting_again(migration):
    migration.apply()
    again = migration.plan()
    assert again["moves"] == [] and again["files"] == []
    assert again["database"]["records"] == []


def test_apply_refuses_running_backend_lock_before_any_changes(migration):
    with (migration.source_data / "backend.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="backend"):
            migration.apply()
    assert migration.source_data.is_dir() and not migration.source_data.is_symlink()
    assert not migration.target_data.exists()


def test_existing_target_is_never_merged_or_overwritten(migration):
    migration.target_data.mkdir(parents=True)
    (migration.target_data / "user-file").write_text("keep")
    with pytest.raises(RuntimeError, match="existing migration target"):
        migration.plan()
    assert (migration.target_data / "user-file").read_text() == "keep"


def test_genuinely_empty_target_created_by_settings_is_accepted(migration):
    migration.target_data.mkdir(parents=True)
    assert migration.plan()["moves"][0]["remove_empty_target"]
    migration.apply()
    assert (migration.target_data / "lucifer.sqlite3").is_file()


def test_only_exact_dangling_old_repository_service_links_are_removed(migration):
    folder = migration.legacy_home / ".config/systemd/user"
    for parent in (folder, folder / "default.target.wants"):
        parent.mkdir(parents=True, exist_ok=True)
        for name in ("lucifer-local-search.service", "lucifer-post-training.service"):
            (parent / name).symlink_to(migration.legacy_home / "Lucifer/configs" / name)
    unrelated = folder / "another.service"
    unrelated.symlink_to(migration.legacy_home / "Lucifer/configs/another.service")
    plan = migration.plan()
    assert len(plan["remove_dangling_service_links"]) == 4
    assert all(Path(link["path"]).is_symlink() for link in plan["remove_dangling_service_links"])
    result = migration.apply()
    assert len(result["removed_service_links"]) == 4
    assert unrelated.is_symlink()
    saved = json.loads((Path(result["backup"]) / "plan.json").read_text())
    assert saved["remove_dangling_service_links"] == plan["remove_dangling_service_links"]


def test_path_replacement_uses_component_boundaries_and_ignores_prose():
    mapping = {"/old/repo": "/new/repo"}
    assert path_replacement("/old/repo/file", mapping) == "/new/repo/file"
    assert path_replacement("/old/repository/file", mapping) == "/old/repository/file"
    assert path_replacement("A command used /old/repo/file", mapping) == "A command used /old/repo/file"
