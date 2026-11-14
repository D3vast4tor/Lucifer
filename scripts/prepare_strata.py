"""Relocate the installed offline runtime's generated paths without copying weights."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def relocate_runtime(root: Path) -> None:
    root = root.resolve()
    profile_path = root / "lucifer-profile.json"
    profile = json.loads(profile_path.read_text())
    args = profile["args"]
    old_pack = Path(args[args.index("--pack") + 1])
    if old_pack.parent.name != "packs" or old_pack.parent.parent.name != "models":
        raise ValueError("Expected an installed Strata pack under models/packs")
    old_root = str(old_pack.parents[2])

    def rebase(value):
        if isinstance(value, str):
            if value == old_root or value.startswith(old_root + "/"):
                return str(root) + value[len(old_root):]
            # Auxiliary manifests may predate a previously repaired profile.
            # Only absolute paths inside the same installation layout qualify.
            marker = "/var/strata/"
            if value.startswith("/") and marker in value:
                return str(root) + "/" + value.split(marker, 1)[1]
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        return value

    def write_changed(path, content):
        if path.read_text() != content:
            mode = path.stat().st_mode
            temporary = path.with_name(path.name + ".relocate.tmp")
            temporary.write_text(content)
            temporary.chmod(mode)
            temporary.replace(path)

    relocated = rebase(profile)
    pack = Path(relocated["args"][args.index("--pack") + 1])
    for field in ("exe", "cwd", "tokenizer"):
        if not Path(relocated[field]).exists():
            raise FileNotFoundError(f"Strata {field} is unavailable: {relocated[field]}")
    target = Path(relocated["cwd"])
    terminal_directory = root / "data/terminal"
    if target.parent != terminal_directory:
        raise ValueError("Strata cwd must be a version directory under data/terminal")
    metadata = pack / "ds4.txt"
    lines = metadata.read_text().splitlines(keepends=True)
    lines = ["model_dir " + rebase(line[len("model_dir "):].rstrip("\r\n")) + "\n"
             if line.startswith("model_dir ") else line for line in lines]
    write_changed(metadata, "".join(lines))
    write_changed(profile_path, json.dumps(relocated, indent=2) + "\n")
    installation_path = root / "installation.json"
    if installation_path.exists():
        installation = rebase(json.loads(installation_path.read_text()))
        installation["runtime_command"] = [str(root.parents[1] / "scripts/serve-strata")]
        write_changed(installation_path, json.dumps(installation, indent=2) + "\n")
    manifests = [root / "axel-download-manifest.json",
                 root / "config/strata/settings.json",
                 *(root / "data/strata/install").glob("strata-*.json")]
    for path in manifests:
        if path.exists():
            write_changed(path, json.dumps(rebase(json.loads(path.read_text())), indent=2) + "\n")
    for path in (root / "data/strata/install").glob("run-*.sh"):
        # Generated launchers use quoted absolute paths, one installed root per
        # argument. Keep executable permissions when replacing their contents.
        content = path.read_text()
        for word in content.split('"')[1::2]:
            rebased = rebase(word)
            if rebased != word:
                content = content.replace('"' + word + '"', '"' + rebased + '"')
        write_changed(path, content)

    def repair_link(path, destination):
        if path.is_symlink() and path.readlink() == Path(destination):
            return
        if path.exists() and not path.is_symlink():
            raise ValueError(f"Refusing to replace a non-symlink runtime path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".relocate.tmp")
        if os.path.lexists(temporary):
            temporary.unlink()
        temporary.symlink_to(destination)
        temporary.replace(path)

    repair_link(root / "data/terminal/current", target.name)
    repair_link(root / "bin/strata", "../data/terminal/current/bin/strata")


if __name__ == "__main__":
    relocate_runtime(Path(sys.argv[1]))
