import importlib.util
import json
from pathlib import Path

import pytest


def runtime_helper():
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_strata.py"
    spec = importlib.util.spec_from_file_location("prepare_strata", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runtime(tmp_path):
    root = tmp_path / "moved-project/var/strata"
    old = "/previous/location/var/strata"
    pack = root / "models/packs/model"
    pack.mkdir(parents=True)
    (pack / "tokenizer").mkdir()
    (pack / "ds4.txt").write_text(
        f"max_position_embeddings 1048576\nmodel_dir {old}/models/models/model\n")
    terminal = root / "data/terminal/0.1.59.1"
    terminal.mkdir(parents=True)
    (terminal.parent / "current").symlink_to(f"{old}/data/terminal/0.1.59.1")
    engine = root / "engine/bin/server"
    engine.parent.mkdir(parents=True)
    engine.touch()
    (root / "bin").mkdir()
    (root / "bin/strata").symlink_to(f"{old}/data/terminal/current/bin/strata")
    profile = {"exe": f"{old}/engine/bin/server", "cwd": f"{old}/data/terminal/0.1.59.1",
               "args": ["--pack", f"{old}/models/packs/model"],
               "tokenizer": f"{old}/models/packs/model/tokenizer",
               "lib_dirs": [f"{old}/engine/lib"], "model_name": "Lucifer"}
    (root / "lucifer-profile.json").write_text(json.dumps(profile))
    (root / "installation.json").write_text(json.dumps({
        "profile": f"{old}/lucifer-profile.json", "source_model": "Example/Checkpoint"}))
    return root


def test_offline_strata_relocation_updates_paths_and_preserves_model_metadata(tmp_path):
    root = make_runtime(tmp_path)
    settings = root / "config/strata/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"data_dir": "/previous/location/var/strata/models",
                                    "installs": ["/previous/location/var/strata/data/strata/install"]}))
    helper = runtime_helper()
    helper.relocate_runtime(root)
    profile_path = root / "lucifer-profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["exe"] == str(root / "engine/bin/server")
    assert profile["model_name"] == "Lucifer"
    assert profile["lib_dirs"] == [str(root / "engine/lib")]
    assert (root / "data/terminal/current").readlink() == Path("0.1.59.1")
    assert (root / "bin/strata").readlink() == Path("../data/terminal/current/bin/strata")
    assert (root / "models/packs/model/ds4.txt").read_text() == (
        f"max_position_embeddings 1048576\nmodel_dir {root}/models/models/model\n")
    installation = json.loads((root / "installation.json").read_text())
    assert installation["source_model"] == "Example/Checkpoint"
    assert installation["runtime_command"] == [str(root.parents[1] / "scripts/serve-strata")]
    assert json.loads(settings.read_text()) == {"data_dir": str(root / "models"),
                                               "installs": [str(root / "data/strata/install")]}
    previous = profile_path.stat().st_mtime_ns
    link_mtime = (root / "data/terminal/current").lstat().st_mtime_ns
    helper.relocate_runtime(root)
    assert profile_path.stat().st_mtime_ns == previous
    assert (root / "data/terminal/current").lstat().st_mtime_ns == link_mtime


def test_strata_repair_recovers_missing_links_and_stale_auxiliary_manifest(tmp_path):
    root = make_runtime(tmp_path)
    helper = runtime_helper()
    helper.relocate_runtime(root)
    # Simulate auxiliary files missed by a previous, partial relocation.
    manifest = root / "axel-download-manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "destination": "/previous/location/var/strata/models/models/shard.gguf",
        "url": "https://example.org/shard.gguf", "sha256": "unchanged"}]}))
    launcher = root / "data/strata/install/run-model.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('#!/bin/sh\nexec "/previous/location/var/strata/bin/strata"\n')
    launcher.chmod(0o755)
    (root / "data/terminal/current").unlink()
    (root / "bin/strata").unlink()
    helper.relocate_runtime(root)
    assert (root / "data/terminal/current").is_dir()
    entry = json.loads(manifest.read_text())["files"][0]
    assert entry["destination"] == str(root / "models/models/shard.gguf")
    assert entry["url"] == "https://example.org/shard.gguf"
    assert entry["sha256"] == "unchanged"
    assert launcher.read_text() == f'#!/bin/sh\nexec "{root}/bin/strata"\n'
    assert launcher.stat().st_mode & 0o777 == 0o755


def test_incomplete_strata_installation_fails_before_rewriting_profile(tmp_path):
    root = make_runtime(tmp_path)
    profile = root / "lucifer-profile.json"
    before = profile.read_bytes()
    (root / "engine/bin/server").unlink()
    with pytest.raises(FileNotFoundError, match="Strata exe is unavailable"):
        runtime_helper().relocate_runtime(root)
    assert profile.read_bytes() == before
