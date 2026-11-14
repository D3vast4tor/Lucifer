"""An open streaming response must not indefinitely delay owned-server cleanup."""
import errno
import json
import socket
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from lucifer.config import Settings


def test_tui_session_flag_only_opens_existing_session_and_configures_private_safe_diagnostics(tmp_path, monkeypatch):
    from lucifer import cli, tui
    opened = []
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml")
    monkeypatch.setattr(cli, "settings_from", lambda args: settings)
    monkeypatch.setattr(sys, "argv", ["lucifer", "tui", "--session", "agent-recover"])
    monkeypatch.setattr(tui, "LuciferApp", lambda **kwargs: SimpleNamespace(run=lambda: opened.append(kwargs)))
    cli.main()
    assert len(opened) == 1
    assert opened[0]["initial_agent_run_id"] == "agent-recover"
    assert opened[0]["crash_report_path"] == settings.data_dir / "logs" / "tui-crash.json"
    assert opened[0]["api_url"] == "http://127.0.0.1:8000"
    assert opened[0]["initial_view"] is None


def test_tui_can_open_the_main_sleep_view_with_a_saved_session(tmp_path, monkeypatch):
    from lucifer import cli, tui
    opened = []
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml")
    monkeypatch.setattr(cli, "settings_from", lambda args: settings)
    monkeypatch.setattr(sys, "argv", ["lucifer", "tui", "--session", "agent-recover", "--view", "sleep"])
    monkeypatch.setattr(tui, "LuciferApp", lambda **kwargs: SimpleNamespace(run=lambda: opened.append(kwargs)))
    cli.main()
    assert len(opened) == 1 and opened[0]["initial_view"] == "sleep"
    assert opened[0]["initial_agent_run_id"] == "agent-recover"
    assert cli.parser().parse_args(["tui"]).view is None
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["tui", "--view", "monitor"])


def test_fatal_tui_propagates_failure_and_keeps_native_terminal_open(tmp_path, monkeypatch):
    from lucifer import cli, tui
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml")
    monkeypatch.setattr(tui, "LuciferApp", lambda **kwargs: SimpleNamespace(run=lambda: None, return_code=1))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))
    with pytest.raises(RuntimeError, match="terminal stopped unexpectedly"):
        cli.run_tui("http://127.0.0.1:8000", settings, session="agent-recover")
    assert prompts == ["Press Enter to close this terminal. "]


def test_escaped_frontend_error_has_only_sanitized_diagnostics(tmp_path, monkeypatch):
    from lucifer import cli, tui
    settings = Settings(data_dir=tmp_path / "state", models_file=tmp_path / "models.yaml")
    def failed_frontend():
        private_value = "PRIVATE_PROVIDER_THINKING_NOT_FOR_LOGS"
        raise ValueError(private_value)
    monkeypatch.setattr(tui, "LuciferApp", lambda **kwargs: SimpleNamespace(run=failed_frontend))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(RuntimeError, match="terminal stopped unexpectedly") as error:
        cli.run_tui("http://127.0.0.1:8000", settings)
    report = settings.data_dir / "logs" / "tui-crash.json"
    saved = report.read_text()
    assert "PRIVATE_PROVIDER_THINKING_NOT_FOR_LOGS" not in saved + str(error.value)
    assert json.loads(saved)["failures"][0]["type"] == "builtins.ValueError"
    assert report.stat().st_mode & 0o777 == 0o600


def test_serve_drains_streams_then_runs_lifespan_cleanup(tmp_path):
    marker = tmp_path / "cleaned-up"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = '''
import asyncio
from pathlib import Path
from types import SimpleNamespace
from lucifer import api, cli

class StreamingApp:
    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    Path(MARKER).write_text("lifespan cleanup completed")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"connected", "more_body": True})
            await asyncio.Event().wait()

cli.settings_from = lambda args: SimpleNamespace(api_token=None)
api.create_app = lambda *args, **kwargs: StreamingApp()
cli.main()
'''.replace("MARKER", repr(str(marker)))
    with (tmp_path / "server.log").open("w+") as log:
        proc = subprocess.Popen([sys.executable, "-c", script, "serve", "--port", str(port)],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with client.stream("GET", f"http://127.0.0.1:{port}/stream") as response:
                            assert next(response.iter_bytes()) == b"connected"
                            started = time.monotonic()
                            proc.terminate()
                            proc.wait(timeout=9)
                            assert time.monotonic() - started < 8
                            break
                    except httpx.ConnectError:
                        if proc.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise AssertionError(log.read()) from None
                        time.sleep(.02)
            # Recent Uvicorn versions re-raise the original signal only after
            # their graceful shutdown and application lifespan have completed.
            assert proc.returncode in {0, -signal.SIGTERM}
            assert marker.read_text() == "lifespan cleanup completed"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.parametrize("frontend_failure", [False, True])
def test_owned_launcher_distinguishes_live_http_listener_from_its_stopped_time_wait_port(tmp_path, monkeypatch, frontend_failure):
    from lucifer import cli, tui
    # The server actively closes the HTTP connection; once the client closes its
    # half, the server port remains in TIME_WAIT after the listener has exited.
    code = '''
import socket
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(15)
    print(listener.getsockname()[1], flush=True)
    connection, _ = listener.accept()
    with connection:
        connection.recv(4096)
        connection.sendall(b"HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\nConnection: close\\r\\n\\r\\nOK")
        connection.shutdown(socket.SHUT_WR)
        while connection.recv(4096):
            pass
'''
    with (tmp_path / 'http.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  stderr=log, text=True)
        try:
            port = int(server.stdout.readline().strip())
            settings = Settings(data_dir=tmp_path / 'state', models_file=tmp_path / 'models.yaml')
            settings.default_model = None
            args = SimpleNamespace(host='127.0.0.1', port=port, command='run', demo=False, load=None)
            attached, launched, opened = [], [], []
            monkeypatch.setattr(cli, 'attach_existing', lambda *args: attached.append(args))
            monkeypatch.setattr(cli, 'startup_timeout', lambda *args: .1)
            class OwnedBackend:
                running = True
                def poll(self):
                    return None if self.running else 0
                def terminate(self):
                    self.running = False
                def wait(self, timeout=None):
                    return 0
            owned = OwnedBackend()
            def launch(argv, **kwargs):
                launched.append((argv, kwargs))
                return owned
            monkeypatch.setattr(cli.subprocess, 'Popen', launch)
            class ReadyClient:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def get(self, *args, **kwargs):
                    return SimpleNamespace(is_success=True)
            monkeypatch.setattr(cli.httpx, 'Client', lambda **kwargs: ReadyClient())
            monkeypatch.setattr(tui, 'LuciferApp', lambda **kwargs: SimpleNamespace(run=lambda: opened.append(kwargs), return_code=int(frontend_failure)))
            monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
            cli.run_owned(args, settings)
            assert len(attached) == 1 and launched == []  # A live server still owns the port.
            with socket.create_connection(('127.0.0.1', port), timeout=3) as client:
                client.sendall(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
                body = b''
                while part := client.recv(4096):
                    body += part
                assert body.endswith(b'OK')
            server.wait(timeout=3)
            assert server.returncode == 0
            # Prove this is the relevant kernel state: an ordinary non-reusing
            # bind fails, while the launcher's reusable probe must succeed.
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError as error:
                    assert error.errno == errno.EADDRINUSE
                else:
                    raise AssertionError('Controlled HTTP close did not establish TIME_WAIT')
            if frontend_failure:
                with pytest.raises(RuntimeError, match="terminal stopped unexpectedly"):
                    cli.run_owned(args, settings)
            else:
                cli.run_owned(args, settings)
            assert len(attached) == 1 and len(launched) == len(opened) == 1
            assert '--port' in launched[0][0] and str(port) in launched[0][0]
            assert owned.running is frontend_failure  # Fatal frontends retain their backend; normal exit cleans up.
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()
