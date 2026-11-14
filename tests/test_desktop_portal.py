"""Portal event ordering, consent and owned request/file boundaries."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lucifer.tools import portal_capture


def portal_env(tmp_path, monkeypatch, *, code=0, uri=None, timed_out=False):
    source = tmp_path / "desktop-source.png"
    source.write_bytes(b"portal image bytes")
    destination = tmp_path / "owned-copy.png"
    state = {"closed": [], "removed": [], "subscriptions": []}
    callbacks = {}

    class Loop:
        def run(self):
            callbacks["timeout"]()

        def quit(self):
            state["quit"] = True

    class Bus:
        def get_unique_name(self):
            return ":1.42"

        def add_signal_receiver(self, callback, **kwargs):
            callbacks["response"] = callback
            state["subscriptions"].append(kwargs["path"])
            return SimpleNamespace(remove=lambda: state.update(match_removed=True))

        def get_object(self, name, path):
            state["object_path"] = path
            return SimpleNamespace(path=path)

    def interface(obj, name):
        if name == "org.freedesktop.portal.Request":
            return SimpleNamespace(Close=lambda **kwargs: state["closed"].append(obj.path))
        def screenshot(parent, options, **kwargs):
            request = "/org/freedesktop/portal/desktop/request/1_42/" + options["handle_token"]
            assert state["subscriptions"] == [request]  # Subscribe before effect.
            if not timed_out:
                callbacks["response"](code, {"uri": uri or source.as_uri()})
            return request
        return SimpleNamespace(Screenshot=screenshot)

    dbus = ModuleType("dbus")
    dbus.SessionBus = Bus
    dbus.Interface = interface
    dbus.Boolean = bool
    dbus.DBusException = RuntimeError
    mainloop = ModuleType("dbus.mainloop.glib")
    mainloop.DBusGMainLoop = lambda **kwargs: state.update(mainloop=kwargs)
    glib = SimpleNamespace(MainLoop=Loop, PRIORITY_DEFAULT=0,
        timeout_add=lambda ms, cb: callbacks.update(timeout=cb) or 1,
        unix_signal_add=lambda priority, sig, cb: callbacks.update(termination=cb) or 2,
        source_remove=lambda ident: state["removed"].append(ident))
    gi = ModuleType("gi.repository")
    gi.GLib = glib
    for name, module in (("dbus", dbus), ("dbus.mainloop", ModuleType("dbus.mainloop")),
                         ("dbus.mainloop.glib", mainloop), ("gi", ModuleType("gi")), ("gi.repository", gi)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["portal_capture.py", str(destination), "1"])
    return source, destination, state


def test_portal_catches_immediate_response_and_preserves_desktop_owned_source(tmp_path, monkeypatch):
    source, destination, state = portal_env(tmp_path, monkeypatch)
    assert portal_capture.main() == 0
    assert destination.read_bytes() == source.read_bytes()
    assert state["mainloop"] == {"set_as_default": True} and state["match_removed"]
    assert not state["closed"] and state["removed"] == [1, 2]


def test_portal_denial_returns_without_copy_or_retry(tmp_path, monkeypatch):
    source, destination, state = portal_env(tmp_path, monkeypatch, code=1)
    assert portal_capture.main() == 4
    assert not destination.exists() and source.exists()
    assert state["match_removed"]


def test_portal_timeout_closes_only_its_request(tmp_path, monkeypatch):
    _, destination, state = portal_env(tmp_path, monkeypatch, timed_out=True)
    assert portal_capture.main() == 4
    assert state["closed"] == state["subscriptions"]
    assert not destination.exists()


@pytest.mark.parametrize("uri", ["https://example.com/screen.png", "file://remote-host/screen.png"])
def test_portal_rejects_remote_image_uri(tmp_path, monkeypatch, uri):
    _, destination, _ = portal_env(tmp_path, monkeypatch, uri=uri)
    with pytest.raises(RuntimeError, match="local image"):
        portal_capture.main()
    assert not destination.exists()


def test_portal_refuses_symlinked_source(tmp_path, monkeypatch):
    link = tmp_path / "link.png"
    source, destination, _ = portal_env(tmp_path, monkeypatch, uri=link.as_uri())
    link.symlink_to(source)
    with pytest.raises(OSError):
        portal_capture.main()
    assert source.exists() and not destination.exists()
