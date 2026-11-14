"""System-Python helper for the desktop portal's consent-mediated screenshot.

Only invoked after the calling run approves capture_screen. No screenshot URI
or image is printed. The destination belongs to the caller's temporary folder.
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import sys
from urllib.parse import unquote, urlsplit


def main():
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
    from gi.repository import GLib

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    loop = GLib.MainLoop()
    token = "lucifer_" + secrets.token_hex(12)
    sender = bus.get_unique_name()[1:].replace(".", "_")
    request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
    result = {}

    def response(code, values):
        result.update(code=int(code), uri=str(values.get("uri", "")))
        loop.quit()

    def timeout():
        result["code"] = 3
        try:
            request = bus.get_object("org.freedesktop.portal.Desktop", request_path)
            dbus.Interface(request, "org.freedesktop.portal.Request").Close(timeout=1)
        except dbus.DBusException:
            pass
        loop.quit()
        return False

    # Subscribe before requesting: even a fast backend response cannot be lost.
    match = bus.add_signal_receiver(response, signal_name="Response",
        dbus_interface="org.freedesktop.portal.Request", path=request_path,
        bus_name="org.freedesktop.portal.Desktop")
    timer = GLib.timeout_add(max(1, int(float(sys.argv[2]) * 1000)), timeout)
    termination = GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, timeout)
    try:
        portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
        handle = dbus.Interface(portal, "org.freedesktop.portal.Screenshot").Screenshot(
            "", {"handle_token": token, "interactive": dbus.Boolean(True)}, timeout=5)
        if str(handle) != request_path:
            raise RuntimeError("Portal returned an unexpected request handle")
        if not result:
            loop.run()
    finally:
        if result.get("code") != 3:
            GLib.source_remove(timer)
            GLib.source_remove(termination)
        match.remove()
    if result.get("code") != 0:
        # Consent denial must not cause retries via another screenshot backend.
        return 4
    uri = urlsplit(result.get("uri", ""))
    if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
        raise RuntimeError("Portal did not return a local image")
    source = Path(unquote(uri.path))
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as reader:
        info = os.fstat(reader.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or not 0 < info.st_size <= 20 * 1024 * 1024:
            raise RuntimeError("Portal image exceeds the permitted file bounds")
        with open(sys.argv[1], "xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
    # The portal owns its source URI; never delete an arbitrary user image.
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("Desktop portal capture failed", file=sys.stderr)
        sys.exit(2)
