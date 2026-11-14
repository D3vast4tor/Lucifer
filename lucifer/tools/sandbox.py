"""Bubblewrap-backed commands, with no fallback to unsandboxed execution."""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess

from .workspace import MAX_SCAN_ENTRIES, ToolError, ToolPolicyError, Workspace, confined_path, excluded, path_parts


async def process_exit(process):
    # asyncio's Process.wait can wait for inherited pipe EOF after the leader
    # exited. Observe its child-watcher returncode independently of pipe drains.
    while process.returncode is None:
        await asyncio.sleep(0.02)
    return process.returncode


def base_command(executable: str) -> list[str]:
    command = [executable, "--unshare-all", "--unshare-user", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--disable-userns", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
               "--setenv", "LANG", "C.UTF-8", "--setenv", "LC_ALL", "C.UTF-8"]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    return command + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/workspace"]


class CommandSandbox:
    """Prepare fixed isolation support once; individual commands add run scope."""

    def __init__(self):
        self.executable = shutil.which("bwrap")
        self.limiter = shutil.which("prlimit")
        self.available = False
        self.error = "bubblewrap and prlimit are required for isolated commands"
        if self.executable and self.limiter:
            try:
                result = subprocess.run(base_command(self.executable) + ["--chdir", "/workspace", "/bin/true"],
                                        capture_output=True, text=True, timeout=5)
                self.available = result.returncode == 0
                self.error = None if self.available else (result.stderr.strip()[:500] or "Bubblewrap namespace setup was rejected")
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.error = str(exc)

    @staticmethod
    def _inventory(workspace: Workspace, context) -> tuple[list[tuple[str, bool]], dict]:
        """Pin directories while locating hidden paths and file change metadata."""
        queue = [()]
        masks = []
        entries_seen = 0
        files = {}
        while queue:
            parts = queue.pop(0)
            fd = workspace.directory(parts)
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        context.check_cancel()
                        entries_seen += 1
                        if entries_seen > MAX_SCAN_ENTRIES:
                            raise ToolError("Command workspace is too broad; choose a smaller directory")
                        relative = parts + (entry.name,)
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode):
                            raise ToolPolicyError("Commands require a workspace without symlinked entries; choose a narrower scope")
                        path = "/".join(relative)
                        is_dir = stat.S_ISDIR(info.st_mode)
                        if excluded(entry.name):
                            masks.append((path, is_dir))
                        elif is_dir:
                            queue.append(relative)
                        elif stat.S_ISREG(info.st_mode):
                            if info.st_nlink != 1:
                                raise ToolPolicyError("Commands cannot expose multiply-linked files")
                            files[path] = (info.st_size, info.st_mtime_ns, info.st_ino)
            finally:
                os.close(fd)
        return masks, files

    async def execute(self, args: dict, context) -> tuple[dict, list[dict], bool]:
        if not self.available:
            raise ToolPolicyError("Isolated command execution is unavailable: " + str(self.error))
        if context.access != "execute" or not context.approved:
            raise ToolPolicyError("A concrete command requires execute scope and approval")
        context.check_cancel()
        timeout = min(float(args.get("timeout_seconds", 30)), context.command_timeout)
        cwd = path_parts(confined_path(args.get("cwd", "."), context.workspace, allow_root=True), allow_root=True)
        with Workspace(context.workspace) as workspace:
            cwd_fd = workspace.directory(cwd)
            os.close(cwd_fd)
            masks, before = self._inventory(workspace, context)
            command = base_command(self.executable)
            # A bind-fd pins the authorized project directory even if a parent is
            # renamed; the sandbox receives no host home, environment or GPU.
            command += ["--bind-fd", str(workspace.fd), "/workspace"]
            for path, directory in masks:
                command += ["--tmpfs", "/workspace/" + path] if directory else ["--ro-bind", "/dev/null", "/workspace/" + path]
            command += ["--chdir", "/workspace" + ("/" + "/".join(cwd) if cwd else "")]
            command += [self.limiter, "--cpu=" + str(math.ceil(timeout) + 1), "--fsize=16777216", "--as=2147483648",
                        "--nofile=256", "--nproc=256", "--", *args["argv"]]
            process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, pass_fds=(workspace.fd,), start_new_session=True)
            output = {"stdout": bytearray(), "stderr": bytearray()}
            output_limited = asyncio.Event()
            truncated = False

            async def drain(name, reader):
                nonlocal truncated
                while data := await reader.read(4096):
                    room = context.max_output_bytes - sum(len(item) for item in output.values())
                    output[name].extend(data[:max(0, room)])
                    if len(data) > room:
                        truncated = True
                        output_limited.set()
                        return

            readers = [asyncio.create_task(drain(name, reader)) for name, reader in (("stdout", process.stdout), ("stderr", process.stderr))]
            waiter = asyncio.create_task(process_exit(process))
            cancellation = asyncio.create_task(context.cancel.wait()) if context.cancel else None
            limited = asyncio.create_task(output_limited.wait())
            reason = None
            try:
                pending = [waiter, limited] + ([cancellation] if cancellation else [])
                completed, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if cancellation and cancellation in completed:
                    raise asyncio.CancelledError("Command cancelled")
                if not completed:
                    reason = "timeout"
                elif limited in completed:
                    reason = "output_limit"
                if reason:
                    await self._stop(process)
                else:
                    await waiter
                try:
                    await asyncio.wait_for(asyncio.gather(*readers), timeout=2)
                except TimeoutError:
                    reason = "background_processes"
                    await self._stop(process)
            except BaseException:
                await self._stop(process)
                raise
            finally:
                for task in [waiter, limited, cancellation, *readers]:
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(*[task for task in [waiter, limited, cancellation, *readers] if task is not None], return_exceptions=True)
            context.check_cancel()
            _, after = self._inventory(workspace, context)
            changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
            artifacts = [{"path": path, "kind": "file", "change": "deleted" if path not in after else "created" if path not in before else "modified"}
                         for path in changed[:100]]
            result = {"argv": args["argv"], "cwd": args.get("cwd", "."), "exit_code": process.returncode,
                      "stdout": output["stdout"].decode("utf-8", errors="replace"),
                      "stderr": output["stderr"].decode("utf-8", errors="replace"),
                      "isolation": "bubblewrap", "network": "disabled", "stop_reason": reason,
                      "changed_files": len(changed), "artifacts_truncated": len(changed) > 100,
                      "requested_timeout_seconds": float(args.get("timeout_seconds", 30)),
                      "effective_timeout_seconds": timeout}
            return result, artifacts, truncated

    @staticmethod
    async def _stop(process):
        # The leader may have exited while descendants still hold the pipes.
        # Reap the owned process group even in that case.
        already_exited = process.returncode is not None
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if already_exited:
            await asyncio.sleep(.05)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            pass
        finally:
            # A responsive leader or closed pipes do not prove its descendants
            # accepted TERM. Finish cleanup of the owned group in all cases.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()


async def execute_host(args: dict, context) -> tuple[dict, list[dict], bool]:
    """Explicit computer scope: ordinary host execution, never called implicitly."""
    if context.access != "computer" or not context.approved:
        raise ToolPolicyError("Host commands require computer scope and concrete approval")
    context.check_cancel()
    cwd = Path(args.get("cwd", ".")).expanduser()
    if not cwd.is_absolute():
        cwd = context.workspace / cwd
    cwd = cwd.resolve(strict=True)
    if not cwd.is_dir():
        raise ToolError("Host command cwd must be a directory")
    timeout = min(float(args.get("timeout_seconds", 30)), context.command_timeout)
    process = await asyncio.create_subprocess_exec(*args["argv"], cwd=cwd, env=dict(os.environ),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)
    output = {"stdout": bytearray(), "stderr": bytearray()}
    output_limited = asyncio.Event()
    truncated = False

    async def drain(name, reader):
        nonlocal truncated
        while data := await reader.read(4096):
            room = context.max_output_bytes - sum(len(item) for item in output.values())
            output[name].extend(data[:max(0, room)])
            if len(data) > room:
                truncated = True
                output_limited.set()
                return

    readers = [asyncio.create_task(drain(name, reader)) for name, reader in (("stdout", process.stdout), ("stderr", process.stderr))]
    waiter = asyncio.create_task(process_exit(process))
    cancellation = asyncio.create_task(context.cancel.wait()) if context.cancel else None
    limited = asyncio.create_task(output_limited.wait())
    reason = None
    try:
        watched = [waiter, limited] + ([cancellation] if cancellation else [])
        completed, _ = await asyncio.wait(watched, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if cancellation and cancellation in completed:
            raise asyncio.CancelledError("Host command cancelled")
        if not completed:
            reason = "timeout"
        elif limited in completed:
            reason = "output_limit"
        if reason:
            await CommandSandbox._stop(process)
        else:
            await waiter
        try:
            await asyncio.wait_for(asyncio.gather(*readers), timeout=2)
        except TimeoutError:
            reason = "background_processes"
            await CommandSandbox._stop(process)
    except BaseException:
        await CommandSandbox._stop(process)
        raise
    finally:
        tasks = [task for task in [waiter, limited, cancellation, *readers] if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    context.check_cancel()
    return {"argv": args["argv"], "cwd": str(cwd), "exit_code": process.returncode,
            "stdout": output["stdout"].decode("utf-8", errors="replace"),
            "stderr": output["stderr"].decode("utf-8", errors="replace"), "isolation": "none",
            "host_access": True, "network": "host", "stop_reason": reason,
            "requested_timeout_seconds": float(args.get("timeout_seconds", 30)),
            "effective_timeout_seconds": timeout}, [], truncated
