"""Bounded file access through pinned directory descriptors.

Every component is opened with O_NOFOLLOW. Relative paths, not model-produced
absolute paths or shell commands, are the only accepted filesystem authority.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from uuid import uuid4


EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", ".train-venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".cache", ".ssh", ".aws",
    ".gnupg", "var", "dist", "build",
})
SECRET_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", "credentials.json", "credentials.*.json", "secrets.json")
MAX_SCAN_ENTRIES = 10000
MAX_FILE_BYTES = 262144


class ToolError(ValueError):
    """An invalid argument, precondition or bounded execution failure."""


class ToolPolicyError(PermissionError):
    """The configured scope does not authorize the effect."""


def excluded(name: str) -> bool:
    return name in EXCLUDED_DIRS or any(fnmatch.fnmatchcase(name.lower(), pattern) for pattern in SECRET_PATTERNS)


def path_parts(value: str, *, allow_root: bool = False, allow_excluded: bool = False) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value or "\\" in value:
        raise ToolPolicyError("Provide a bounded workspace-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ToolPolicyError("Absolute paths and parent traversal are outside the workspace scope")
    parts = tuple(p for p in path.parts if p != ".")
    if not parts and not allow_root:
        raise ToolPolicyError("This tool requires a file or directory below the workspace")
    if not allow_excluded and any(excluded(part) for part in parts):
        raise ToolPolicyError("This path is excluded by the workspace's secret/cache policy")
    return parts


def confined_path(value: str, workspace: Path, *, allow_root: bool = False) -> str:
    """Accept lexical aliases inside the selected project without following links.

    Models commonly copy the absolute project path from their startup prompt.
    Strip that prefix only when it is lexically contained in the chosen root;
    directory-descriptor operations still reject symlink components afterwards.
    """
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value or "\\" in value:
        raise ToolPolicyError("Provide a bounded project path")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ToolPolicyError("Parent traversal is outside the workspace scope")
    if path.is_absolute():
        try:
            path = path.relative_to(PurePosixPath(str(workspace)))
        except ValueError:
            raise ToolPolicyError("Path is outside the selected workspace") from None
    parts = path_parts(str(path), allow_root=allow_root)
    return "/".join(parts) if parts else "."


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Workspace:
    def __init__(self, root: Path, *, unrestricted: bool = False, allow_root: bool = False):
        self.unrestricted = unrestricted
        self.root = Path(root).expanduser().resolve(strict=True)
        if (self.root == Path("/") and not allow_root) or not self.root.is_dir():
            raise ToolPolicyError("Select a project directory rather than the filesystem root")
        self.fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in self.root.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.fd)
                os.close(self.fd)
                self.fd = child
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise ToolPolicyError("Workspace path changed or contains an inaccessible symlink") from None

    def parts(self, value: str, *, allow_root=False):
        return path_parts(value, allow_root=allow_root, allow_excluded=self.unrestricted)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def directory(self, parts: tuple[str, ...]) -> int:
        fd = os.dup(self.fd)
        try:
            for part in parts:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd
        except OSError as exc:
            os.close(fd)
            raise ToolPolicyError("Directory is missing, inaccessible or contains a symlink") from exc

    def check_components(self, parts: tuple[str, ...]):
        """Reject existing links without requiring a replay target to exist."""
        fd = os.dup(self.fd)
        try:
            for index, part in enumerate(parts):
                try:
                    info = os.stat(part, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if stat.S_ISLNK(info.st_mode):
                    raise ToolPolicyError("Symlink components are outside the workspace tool policy")
                if index < len(parts) - 1:
                    if not stat.S_ISDIR(info.st_mode):
                        raise ToolPolicyError("A path parent is not a directory")
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                    os.close(fd)
                    fd = child
        finally:
            os.close(fd)

    def read(self, path: str, limit: int = MAX_FILE_BYTES) -> tuple[bytes, os.stat_result]:
        parts = self.parts(path)
        parent = self.directory(parts[:-1])
        try:
            try:
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
            except OSError as exc:
                raise ToolPolicyError("File is missing, inaccessible or is a symlink") from exc
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ToolPolicyError("Only ordinary, singly-linked workspace files can be read")
                if info.st_size > limit:
                    raise ToolError(f"File exceeds the {limit:,}-byte read limit; request a smaller file")
                chunks = []
                remaining = limit + 1
                while remaining:
                    chunk = os.read(fd, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                if len(raw) > limit:
                    raise ToolError("File grew beyond the read limit")
                return raw, info
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def metadata(self, path: str, *, allow_directory: bool = False) -> os.stat_result:
        """Fresh file identity without reading bytes or following symlinks."""
        parts = self.parts(path, allow_root=allow_directory)
        if not parts:
            return os.fstat(self.fd)
        parent = self.directory(parts[:-1])
        try:
            try:
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
            except OSError as exc:
                raise ToolPolicyError("File is missing, inaccessible or is a symlink") from exc
            try:
                info = os.fstat(fd)
                if not (allow_directory and stat.S_ISDIR(info.st_mode)) and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                    raise ToolPolicyError("Only ordinary, singly-linked files have readable source versions")
                return info
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def current(self, parent: int, name: str, limit: int = MAX_FILE_BYTES) -> tuple[bytes | None, os.stat_result | None]:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        except FileNotFoundError:
            return None, None
        except OSError as exc:
            raise ToolPolicyError("Target is inaccessible or a symlink") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ToolPolicyError("Mutation targets must be ordinary, singly-linked files")
            if info.st_size > limit:
                raise ToolError(f"Mutation target exceeds the {limit:,}-byte limit")
            chunks = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > limit:
                raise ToolError("Mutation target grew beyond its limit")
            return raw, info
        finally:
            os.close(fd)

    @staticmethod
    def check_expected(raw: bytes | None, expected: str | None):
        actual = digest(raw) if raw is not None else None
        if actual != expected:
            raise ToolError(f"File precondition changed: expected {expected or 'absent'}, found {actual or 'absent'}. Read and review again.")

    def write(self, path: str, text: str, expected: str | None, *, check_cancel=lambda: None) -> dict:
        raw = text.encode("utf-8")
        if len(raw) > MAX_FILE_BYTES:
            raise ToolError(f"New content exceeds the {MAX_FILE_BYTES:,}-byte write limit")
        parts = self.parts(path)
        parent = self.directory(parts[:-1])
        temporary = ".lucifer-write-" + uuid4().hex
        try:
            current, info = self.current(parent, parts[-1])
            if current == raw:
                return {"path": "/".join(parts), "sha256": digest(raw), "bytes": len(raw), "changed": False, "already_applied": True}
            self.check_expected(current, expected)
            check_cancel()
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fchmod(handle.fileno(), stat.S_IMODE(info.st_mode) if info else 0o644)
                    os.fsync(handle.fileno())
                # The descriptor pins the parent; recheck the original inode and
                # bytes immediately before the atomic replacement.
                again, again_info = self.current(parent, parts[-1])
                self.check_expected(again, expected)
                if info and again_info and (info.st_ino, info.st_dev) != (again_info.st_ino, again_info.st_dev):
                    raise ToolError("Target inode changed during preparation")
                check_cancel()
                os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            return {"path": "/".join(parts), "sha256": digest(raw), "bytes": len(raw), "changed": True, "already_applied": False}
        finally:
            os.close(parent)

    def delete(self, path: str, expected: str, *, check_cancel=lambda: None) -> dict:
        parts = self.parts(path)
        parent = self.directory(parts[:-1])
        try:
            current, info = self.current(parent, parts[-1])
            if current is None:
                return {"path": "/".join(parts), "deleted": False, "already_applied": True}
            self.check_expected(current, expected)
            check_cancel()
            again, again_info = self.current(parent, parts[-1])
            self.check_expected(again, expected)
            if (info.st_ino, info.st_dev) != (again_info.st_ino, again_info.st_dev):
                raise ToolError("Target inode changed during preparation")
            os.unlink(parts[-1], dir_fd=parent)
            os.fsync(parent)
            return {"path": "/".join(parts), "deleted": True, "already_applied": False}
        finally:
            os.close(parent)

    def mkdir(self, path: str) -> dict:
        parts = self.parts(path)
        parent = self.directory(parts[:-1])
        try:
            try:
                os.mkdir(parts[-1], 0o755, dir_fd=parent)
                os.fsync(parent)
                changed = True
            except FileExistsError:
                fd = self.directory(parts)
                os.close(fd)
                changed = False
            return {"path": "/".join(parts), "created": changed, "already_applied": not changed}
        finally:
            os.close(parent)

    def walk(self, path: str = ".", *, max_depth: int = 8, check_cancel=lambda: None):
        parts = self.parts(path, allow_root=True)
        queue = [(parts, 0)]
        scanned = 0
        while queue:
            current, depth = queue.pop(0)
            fd = self.directory(current)
            try:
                with os.scandir(fd) as entries:
                    # Bounded even before sorting: an adversarial directory may
                    # contain millions of entries.
                    listed = []
                    for entry in entries:
                        check_cancel()
                        scanned += 1
                        if scanned > MAX_SCAN_ENTRIES:
                            raise ToolError(f"Workspace scan exceeded {MAX_SCAN_ENTRIES:,} entries; choose a narrower path")
                        listed.append(entry)
                for entry in sorted(listed, key=lambda item: item.name):
                    if excluded(entry.name):
                        continue
                    check_cancel()
                    relative = current + (entry.name,)
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        yield "/".join(relative), "directory", info
                        if depth < max_depth:
                            queue.append((relative, depth + 1))
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        yield "/".join(relative), "file", info
            finally:
                os.close(fd)


def text_file(raw: bytes) -> str:
    if b"\x00" in raw:
        raise ToolError("Binary file; use a supported image tool or choose a text file")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("File is not UTF-8 text") from exc


def read_file(workspace: Workspace, args: dict, *, source_callback=None) -> dict:
    raw, _ = workspace.read(args["path"])
    lines = text_file(raw).splitlines()
    start = args.get("start_line", 1)
    count = args.get("max_lines", 200)
    selected = lines[start - 1:start - 1 + count]
    result = {"path": args["path"], "sha256": digest(raw), "start_line": start,
            "end_line": start - 1 + len(selected), "total_lines": len(lines),
            "content": "\n".join(f"{start + index}: {line}" for index, line in enumerate(selected)),
            "truncated": start - 1 + len(selected) < len(lines),
            "line_truncated": False,
            "next_start_line": start + len(selected) if start - 1 + len(selected) < len(lines) else None}
    if source_callback is not None and selected:
        text = "\n".join(lines)
        if len(text) <= 256000 and len(raw) <= 2 * 1024 * 1024:
            start_char = sum(len(line) + 1 for line in lines[:start - 1])
            result["knowledge_source"] = source_callback(raw, text, start_char,
                start_char + len("\n".join(selected)), result["content"])
    return result


def image_metadata(raw: bytes) -> tuple[str, int, int]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24 and raw[12:16] == b"IHDR":
        width, height = struct.unpack(">II", raw[16:24])
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8"):
        pos = 2
        width = height = 0
        while pos + 4 <= len(raw):
            if raw[pos] != 0xFF:
                pos += 1
                continue
            marker = raw[pos + 1]
            pos += 2
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            length = int.from_bytes(raw[pos:pos + 2], "big")
            if length < 2 or pos + length > len(raw):
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and length >= 7:
                height, width = struct.unpack(">HH", raw[pos + 3:pos + 7])
                break
            pos += length
        mime = "image/jpeg"
    elif len(raw) >= 30 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        if raw[12:16] == b"VP8X":
            width = 1 + int.from_bytes(raw[24:27], "little")
            height = 1 + int.from_bytes(raw[27:30], "little")
        elif raw[12:16] == b"VP8 " and raw[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", raw[26:30])
            width, height = width & 0x3FFF, height & 0x3FFF
        elif raw[12:16] == b"VP8L" and raw[20] == 0x2F:
            bits = int.from_bytes(raw[21:25], "little")
            width, height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        else:
            raise ToolError("Unsupported or malformed WebP header")
        mime = "image/webp"
    else:
        raise ToolError("Only local PNG, JPEG and WebP images with valid dimension headers are supported")
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or width * height > 16777216:
        raise ToolError("Image dimensions exceed the 16-megapixel input limit")
    return mime, width, height


def image_payload(path: str, context) -> tuple[dict, list[dict]]:
    if not context.allow_images:
        raise ToolPolicyError("Image access was not enabled for this run")
    context.check_cancel()
    with scoped_workspace(context, path) as (workspace, relative, _):
        raw, _ = workspace.read(relative, context.max_image_bytes)
    mime, width, height = image_metadata(raw)
    metadata = {"path": path, "mime": mime, "width": width, "height": height,
                "sha256": digest(raw), "bytes": len(raw)}
    encoded = base64.b64encode(raw).decode("ascii")
    return metadata, [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]



@contextmanager
def scoped_workspace(context, requested: str, *, directory: bool | None = False):
    """Normal runs stay confined; computer scope explicitly selects host paths."""
    if context.access == "computer":
        if not isinstance(requested, str) or not requested or "\x00" in requested or len(requested) > 4096:
            raise ToolPolicyError("Provide a bounded host path")
        target = Path(requested).expanduser()
        if not target.is_absolute():
            target = context.workspace / target
        target = target.resolve(strict=False)
        if directory is None:
            directory = target.is_dir()
        authority = target if directory else target.parent
        relative = "." if directory else target.name
        with Workspace(authority, unrestricted=True, allow_root=True) as workspace:
            yield workspace, relative, target
    else:
        relative = confined_path(requested, context.workspace, allow_root=directory is not False)
        with Workspace(context.workspace) as workspace:
            yield workspace, relative, context.workspace / relative


def operation_fingerprint(name: str, arguments: dict, workspace: Path, *, resolve_paths: bool = True) -> str:
    canonical = dict(arguments)
    local_files = {"list_files", "read_file", "search_text", "read_image", "write_file", "edit_file", "delete_file", "make_directory"}
    keys = ("path",) if name in local_files else ("cwd",) if name == "run_command" else ()
    for key in keys:
        value = canonical.get(key)
        if isinstance(value, str):
            target = Path(value).expanduser()
            target = target if target.is_absolute() else Path(workspace) / target
            canonical[key] = str(target.resolve(strict=False)) if resolve_paths else str(PurePosixPath(target))
    return digest(json.dumps({"tool": name, "arguments": canonical, "workspace": str(Path(workspace).resolve())},
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
