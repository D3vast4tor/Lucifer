"""Offline relocation of operational Lucifer paths, preserving frozen evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3


MODEL_PATH_FIELDS = {
    "path", "training_model", "executable", "runtime_command", "context_metadata_file",
    "native_context_source", "log_path", "extra_args",
}
RECORD_PATH_FIELDS = {
    "models": MODEL_PATH_FIELDS,
    "adapters": {"path", "training_model"},
    "adaptive_datasets": {"dataset_path"},
    "sleep_runs": {"dataset_path"},
    "training_runs": {"dataset_path"},
}
CONFIG_PATH_LINE = re.compile(r"^\s*(?:path|training_model|executable|runtime_command|context_metadata_file):")


def path_replacement(value, replacements):
    """Replace only complete absolute path prefixes, never embedded prose."""
    if isinstance(value, list):
        return [path_replacement(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old + "/"):
            return new + value[len(old):]
    return value


def operational_record(kind, record, replacements, launcher):
    result = copy.deepcopy(record)
    changed = []
    for field in RECORD_PATH_FIELDS.get(kind, ()):
        if field not in result:
            continue
        previous = result[field]
        result[field] = path_replacement(previous, replacements)
        if kind == "models" and field == "executable" and result.get("backend") == "vllm":
            if isinstance(previous, str) and previous.endswith("/vLLM/Lucifer/bin/vllm"):
                result[field] = str(launcher)
        if result[field] != previous:
            changed.append(field)
    if kind == "models" and isinstance(result.get("spec"), dict):
        spec, fields = operational_record("models", {key: value for key, value in result["spec"].items() if key != "spec"}, replacements, launcher)
        result["spec"] = spec
        changed.extend("spec." + field for field in fields)
    return result, sorted(changed)


def database_changes(database, replacements, launcher, *, apply=False):
    uri = "file:" + str(database) + ("?mode=rw" if apply else "?mode=ro")
    connection = sqlite3.connect(uri, uri=True)
    changes, preserved = [], {}
    try:
        if apply:
            connection.execute("BEGIN IMMEDIATE")
        for kind, ident, encoded in connection.execute("SELECT kind,id,data FROM documents").fetchall():
            record = json.loads(encoded)
            updated, fields = operational_record(kind, record, replacements, launcher)
            if fields:
                changes.append({"kind": kind, "id": ident, "fields": fields})
                if apply:
                    connection.execute("UPDATE documents SET data=? WHERE kind=? AND id=?",
                        (json.dumps(updated, ensure_ascii=False), kind, ident))
            if any(old in encoded for old in replacements) and kind not in RECORD_PATH_FIELDS:
                preserved[kind] = preserved.get(kind, 0) + 1
        if apply:
            connection.commit()
        return {"records": changes, "preserved_historical_records": preserved}
    except BaseException:
        if apply:
            connection.rollback()
        raise
    finally:
        connection.close()


def console_rewrite(path, replacements):
    """Repair executable interpreter paths and activation values, not bytecode."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        return None
    try:
        original = path.read_text()
    except (UnicodeError, OSError):
        return None
    lines = original.splitlines(keepends=True)
    if not lines:
        return None
    if lines[0].startswith("#!"):
        interpreter = lines[0][2:].rstrip("\r\n")
        rewritten = path_replacement(interpreter, replacements)
        if rewritten != interpreter:
            lines[0] = "#!" + rewritten + "\n"
    if path.name.startswith("activate"):
        for index, line in enumerate(lines):
            for old, new in replacements.items():
                line = line.replace(old, new)
            lines[index] = line
    updated = "".join(lines)
    return updated if updated != original else None


def config_rewrite(path, replacements):
    if not path.is_file():
        return None
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not CONFIG_PATH_LINE.match(line):
            continue
        for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
            line = line.replace(old, new)
        lines[index] = line
    updated = "".join(lines)
    return updated if updated != original else None


@contextmanager
def stopped_backend(data_dir):
    """The serving process owns this same advisory lock for its whole lifetime."""
    lock_path = data_dir / "backend.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Stop the Lucifer backend before applying path migration; its data lock is held") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class PathMigration:
    def __init__(self, *, canonical_root="/home/Lucifer", legacy_home="/home/d3vast4tor", source_data=None):
        self.root = Path(canonical_root).absolute()
        self.legacy_home = Path(legacy_home).absolute()
        self.project = self.root / "Soul"
        self.source_data = Path(source_data).absolute() if source_data else self.legacy_home / ".local/share/lucifer"
        self.target_data = self.project / "var/state/runtime"
        self.source_model = self.legacy_home / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B"
        self.target_model = self.root / "models/Qwen3.5-4B"
        self.replacements = {
            str(self.legacy_home / "Lucifer"): str(self.project),
            str(self.legacy_home / "vLLM"): str(self.root / "vLLM"),
            str(self.source_data): str(self.target_data),
            str(self.source_model): str(self.target_model),
        }

    def _active_data(self):
        if self.target_data.is_dir() and (not self.source_data.exists() or self.source_data.resolve() == self.target_data.resolve()):
            return self.target_data
        return self.source_data

    def file_changes(self):
        changes = {}
        for config in (self.project / "configs/models.yaml",):
            updated = config_rewrite(config, self.replacements)
            if updated is not None:
                changes[config] = updated
        for environment in (self.project / ".venv", self.project / ".train-venv", self.root / "vLLM/Lucifer"):
            folder = environment / "bin"
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                updated = console_rewrite(path, self.replacements)
                if updated is not None:
                    changes[path] = updated
        return changes

    def stale_service_links(self):
        folder = self.legacy_home / ".config/systemd/user"
        links = []
        for parent in (folder, folder / "default.target.wants"):
            for name in ("lucifer-local-search.service", "lucifer-post-training.service"):
                path = parent / name
                if not path.is_symlink() or path.exists():
                    continue
                target = str(path.readlink())
                if target.startswith(str(self.legacy_home / "Lucifer") + "/"):
                    links.append({"path": str(path), "target": target})
        return links

    def plan(self):
        active_data = self._active_data()
        database = active_data / "lucifer.sqlite3"
        if not database.is_file():
            raise FileNotFoundError("Lucifer database is unavailable: " + str(database))
        moves = []
        for source, target in ((self.source_data, self.target_data), (self.source_model, self.target_model)):
            if source.is_symlink() and source.resolve() != target.resolve():
                raise RuntimeError("Refusing to relocate an unrelated source symlink: " + str(source))
            if source.exists() and source.resolve() != target.resolve():
                empty_target = target.is_dir() and not target.is_symlink() and not any(target.iterdir())
                if target.is_symlink() or target.exists() and not empty_target:
                    raise RuntimeError("Refusing to merge an existing migration target: " + str(target))
                ancestor = target.parent
                while not ancestor.exists():
                    ancestor = ancestor.parent
                if source.stat().st_dev != ancestor.stat().st_dev:
                    raise RuntimeError("Cross-filesystem migration requires a separately verified copy; source was not removed")
                moves.append({"source": str(source), "target": str(target), "compatibility_link": True,
                              "remove_empty_target": empty_target})
        return {"canonical_root": str(self.root), "active_data": str(active_data), "moves": moves,
                "files": [str(path) for path in self.file_changes()],
                "remove_dangling_service_links": self.stale_service_links(),
                "database": database_changes(database, self.replacements, self.project / "scripts/vllm"),
                "historical_content": "Unchanged: transcripts, agent checkpoints, logs, evidence bindings, hashes and model artifacts"}

    @staticmethod
    def _move_with_alias(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise RuntimeError("Cross-filesystem migration requires a separately verified copy; source was not removed")
        if target.exists():
            # rmdir is atomic and refuses a target populated after planning.
            target.rmdir()
        os.rename(source, target)
        try:
            source.symlink_to(target, target_is_directory=True)
        except BaseException:
            os.rename(target, source)
            raise

    def apply(self):
        active_data = self._active_data()
        with stopped_backend(active_data):
            plan = self.plan()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            backup = self.project / "var/state/migration-backups" / stamp
            backup.mkdir(parents=True, mode=0o700)
            source = sqlite3.connect("file:" + str(active_data / "lucifer.sqlite3") + "?mode=ro", uri=True)
            destination = sqlite3.connect(backup / "lucifer.sqlite3")
            try:
                source.backup(destination)
            finally:
                source.close()
                destination.close()
            (backup / "lucifer.sqlite3").chmod(0o600)
            changes = self.file_changes()
            for path in changes:
                saved = backup / "files" / path.relative_to(self.root)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, saved)
            (backup / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            for move in plan["moves"]:
                self._move_with_alias(Path(move["source"]), Path(move["target"]))
            database = self.target_data / "lucifer.sqlite3"
            applied = database_changes(database, self.replacements, self.project / "scripts/vllm", apply=True)
            for path, updated in changes.items():
                temporary = path.with_name(path.name + ".lucifer-migration-tmp")
                if temporary.exists():
                    raise RuntimeError("Migration temporary file already exists: " + str(temporary))
                try:
                    temporary.write_text(updated)
                    shutil.copymode(path, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            removed_links = []
            for link in plan["remove_dangling_service_links"]:
                path = Path(link["path"])
                if path.is_symlink() and not path.exists() and str(path.readlink()) == link["target"]:
                    path.unlink()
                    removed_links.append(link)
            result = {**plan, "applied": True, "backup": str(backup), "database": applied}
            result["removed_service_links"] = removed_links
            (backup / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply only after the backend has stopped; otherwise report a read-only plan")
    parser.add_argument("--canonical-root", default="/home/Lucifer")
    parser.add_argument("--legacy-home", default="/home/d3vast4tor")
    parser.add_argument("--source-data")
    args = parser.parse_args(argv)
    migration = PathMigration(canonical_root=args.canonical_root, legacy_home=args.legacy_home, source_data=args.source_data)
    try:
        result = migration.apply() if args.apply else migration.plan()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
