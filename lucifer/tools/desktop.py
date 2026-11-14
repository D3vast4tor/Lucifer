"""Explicit desktop screen capture for image-enabled Direct PC runs.

The capture is a host effect: it is available only in computer scope, only when
the run has image input enabled, and only after concrete approval. The captured
PNG is read into a transient ToolResult image payload and the temporary file is
removed.
"""
from __future__ import annotations

import tempfile
import functools
import subprocess
from pathlib import Path

from .sandbox import execute_host
from .workspace import ToolError, ToolPolicyError, image_payload

SCREEN_CAPTURE_TOOLS = ("gnome-screenshot", "grim", "scrot", "spectacle", "flameshot", "maim", "import")
PORTAL = "xdg-desktop-portal"


@functools.lru_cache(maxsize=1)
def portal_python() -> str | None:
    """Discover once at initialization, without requesting a screenshot."""
    executable = "/usr/bin/python3"
    probe = ("import dbus,gi; b=dbus.SessionBus(); "
             "p=b.get_object('org.freedesktop.portal.Desktop','/org/freedesktop/portal/desktop'); "
             "assert 'org.freedesktop.portal.Screenshot' in "
             "dbus.Interface(p,'org.freedesktop.DBus.Introspectable').Introspect(timeout=2)")
    try:
        result = subprocess.run([executable, "-c", probe], capture_output=True, timeout=3)
        return executable if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _argv(name: str, utility: str, path: str, include_cursor: bool) -> list[str]:
    if name == "gnome-screenshot":
        return [utility, "-p", "-f", path] if include_cursor else [utility, "-f", path]
    if name == "grim":
        return [utility, "-c", path] if include_cursor else [utility, path]
    if name == "scrot":
        return [utility, path]
    if name == "spectacle":
        return [utility, "-b", "-o", path]
    if name == "flameshot":
        return [utility, "screen", "-p", path]
    if name == "maim":
        return [utility, path]
    if name == "import":
        return [utility, "-window", "root", path]
    raise ToolError("Unsupported screen capture utility: " + name)


async def capture_screen(args: dict, context, tools: dict[str, str | None]) -> tuple[dict, list[dict]]:
    if context.access != "computer":
        raise ToolPolicyError("Screen capture requires Direct PC / computer scope")
    if not context.allow_images:
        raise ToolPolicyError("Screen capture requires an image-enabled run")
    available = [(name, tools.get(name)) for name in SCREEN_CAPTURE_TOOLS if tools.get(name)]
    if tools.get(PORTAL):
        # Prefer desktop-mediated consent on Wayland; do not bypass denial by
        # silently trying another backend.
        available = [(PORTAL, tools[PORTAL])]
    if not available:
        raise ToolPolicyError("No desktop screenshot utility is available")
    context.check_cancel()
    timeout = min(float(args.get("timeout_seconds", 30)), context.command_timeout)
    include_cursor = bool(args.get("include_cursor", False))
    errors = []
    with tempfile.TemporaryDirectory(prefix="lucifer-screen-") as directory:
        path = Path(directory) / "screen.png"
        for name, utility in available:
            context.check_cancel()
            if name == PORTAL:
                if include_cursor:
                    raise ToolPolicyError("The desktop portal controls cursor capture; set include_cursor=false")
                argv = [utility, str(Path(__file__).with_name("portal_capture.py")), str(path), str(max(0.1, timeout - 1))]
            else:
                argv = _argv(name, utility, str(path), include_cursor)
            try:
                content, _, _ = await execute_host(
                    {"argv": argv, "cwd": str(context.workspace), "timeout_seconds": timeout}, context)
            except (ToolError, ToolPolicyError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                continue
            if content.get("exit_code") == 0 and path.is_file():
                metadata, images = image_payload(str(path), context)
                metadata.update({"path": "screen", "capture": name, "include_cursor": include_cursor,
                                 "host_access": True, "utility": utility})
                return metadata, images
            errors.append(f"{name}: exit {content.get('exit_code')} {str(content.get('stderr', ''))[:200]}")
    raise ToolError("Screen capture failed: " + "; ".join(errors[:5]))
