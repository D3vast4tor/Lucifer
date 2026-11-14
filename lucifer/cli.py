"""Console entry points. `lucifer run` owns and cleans up its backend."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from rich.console import Console
from .config import Settings
from . import __version__


def parser():
    p = argparse.ArgumentParser(prog="lucifer", description="Local AI agent, adapter cortex and offline model learning")
    p.add_argument("--version", action="version", version=f"Lucifer {__version__}")
    commands = p.add_subparsers(dest="command")
    for name in ("run", "demo", "serve", "tui", "doctor", "init"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path)
        sub.add_argument("--models", type=Path)
        sub.add_argument("--evals", type=Path)
        if name in {"run", "demo", "serve"}:
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8000)
            sub.add_argument("--load", help="Model id to load on startup")
        if name in {"run", "demo", "tui"}:
            sub.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                             help="Terminal colors; auto respects NO_COLOR")
        if name == "run":
            sub.add_argument("--demo", action="store_true")
        if name == "tui":
            sub.add_argument("--api-url", default="http://127.0.0.1:8000")
            sub.add_argument("--session", help="Open an existing agent session without resuming it")
            sub.add_argument("--view", choices=("agent", "chat", "brain", "sleep", "meditation"),
                             help="Open this application view after loading the saved session")
        if name == "init":
            sub.add_argument("--directory", type=Path, default=Path("configs"))
    return p


def settings_from(args):
    opts = {k: v for k in ("data_dir", "models_file", "eval_file") if (v := getattr(args, {"models_file": "models", "eval_file": "evals"}.get(k, k), None)) is not None}
    settings = Settings(**opts)
    # Installed wheels carry defaults; explicit paths must exist.
    for attr, arg, resource in (("models_file", "models", "models.yaml"), ("eval_file", "evals", "evals.yaml")):
        path = getattr(settings, attr)
        if not path.exists():
            if getattr(args, arg, None) or os.environ.get({"models": "LUCIFER_MODELS", "evals": "LUCIFER_EVALS"}[arg]):
                raise ValueError(f"Configuration file does not exist: {path}")
            destination = settings.data_dir / resource
            if not destination.exists():
                destination.write_bytes(files("lucifer").joinpath("resources", resource).read_bytes())
                destination.chmod(0o600)
            setattr(settings, attr, destination)
    return settings


def doctor(settings):
    import importlib.metadata
    from .learning import training_python
    info = {"python": sys.version.split()[0], "data_dir": str(settings.data_dir),
            "models_file": str(settings.models_file), "eval_file": str(settings.eval_file),
            "versions": {k: importlib.metadata.version(k) for k in ("fastapi", "textual", "httpx")},
            "training_python": training_python(), "default_model": settings.default_model}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        info["gpu"] = gpu.stdout.strip() or "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        info["gpu"] = "unavailable"
    from .inference import ModelSpec
    import yaml
    import shutil
    specs = [ModelSpec.model_validate(m).resolve_local_paths(settings.models_file.parent)
             for m in (yaml.safe_load(settings.models_file.read_text()) or {}).get("models", [])]
    info["models"] = [{"id": m.id, "display_name": m.display_name, "backend": m.backend, "binary_available": bool(shutil.which(m.runtime_command[0] if m.backend == "strata" else m.executable or ("vllm" if m.backend == "vllm" else "llama-server"))) if m.managed else None} for m in specs]
    Console().print_json(json.dumps(info))


def startup_timeout(settings, model_id):
    import yaml
    payload = yaml.safe_load(settings.models_file.read_text()) or {}
    profile = next((m for m in payload.get("models", []) if m.get("id") == model_id), {})
    return float(profile.get("startup_timeout") or settings.startup_timeout)


def run_tui(url, settings, *, session=None, view=None):
    """Keep a fatal frontend visible without exposing private exception data."""
    from .tui import LuciferApp, sanitized_exception
    report = settings.data_dir / "logs" / "tui-crash.json"
    failed, app = False, None
    try:
        app = LuciferApp(api_url=url, api_token=settings.api_token,
                         initial_agent_run_id=session, initial_view=view, crash_report_path=report)
        app.run()
    except Exception as error:
        failed = True
        temporary = None
        try:
            report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                             prefix=".tui-crash-", suffix=".json", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(sanitized_exception(error), handle, indent=2)
            temporary.replace(report)
        except OSError:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    if failed or getattr(app, "return_code", 0):
        detail = (f"Lucifer terminal stopped unexpectedly. Diagnostic: {report}. "
                  "The separate backend retains its session. Reopen History or use tui --session RUN_ID.")
        if sys.stdin.isatty():
            Console(stderr=True).print(detail, markup=False)
            try:
                input("Press Enter to close this terminal. ")
            except (EOFError, KeyboardInterrupt):
                pass
        raise RuntimeError(detail)


def attach_existing(url, settings, load=None):
    """Reuse only this project's authenticated, compatible backend."""
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    try:
        with httpx.Client(timeout=settings.request_timeout, headers=headers, trust_env=False) as client:
            response = client.get(url + "/system/status")
            response.raise_for_status()
            status = response.json()
            if (status.get("version") != __version__
                    or status.get("data_dir") != str(settings.data_dir)
                    or status.get("models_file") != str(settings.models_file)):
                raise RuntimeError("This port belongs to a different backend. Choose another --port.")
            if load and (status.get("model") or {}).get("id") != load:
                body, load_headers = {"model_id": load}, {}
                registry = client.get(url + "/models")
                registry.raise_for_status()
                selected = next((model for model in registry.json() if model.get("id") == load), None)
                if selected and selected.get("registry_source") == "file":
                    import yaml
                    from .inference import ModelSpec
                    configuration = yaml.safe_load(settings.models_file.read_text()) or {}
                    profile = next((model for model in configuration.get("models", [])
                                    if model.get("id") == load), None)
                    if profile is not None:
                        spec = ModelSpec.model_validate(profile).resolve_local_paths(settings.models_file.parent)
                        cached = selected.get("spec") or {key: value for key, value in selected.items()
                                                         if key in ModelSpec.model_fields}
                        if ModelSpec.model_validate(cached) != spec:
                            # An already-running backend may predate relative
                            # path support. Send the resolved snapshot through
                            # the existing load API instead of trusting its
                            # stale registry or relative-path implementation.
                            body = {"spec": spec.model_dump()}
                            load_headers["X-Lucifer-Profile-Source"] = "file"
                response = client.post(url + "/models/load", json=body, headers=load_headers,
                                       timeout=startup_timeout(settings, load) + 20)
                if response.is_error:
                    raise RuntimeError(f"Could not select the model: {response.text}")
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Port is in use and no compatible Lucifer backend is available: {exc}") from exc
    run_tui(url, settings)


def run_owned(args, settings):
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The combined terminal launcher requires a loopback host")
    load = "demo" if args.command == "demo" or getattr(args, "demo", False) else args.load or settings.default_model
    url = f"http://127.0.0.1:{args.port}"
    try:
        with socket.socket() as sock:
            # A stopped HTTP server can leave TIME_WAIT sockets. Probe listener
            # ownership, rather than mistaking those sockets for a live backend.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", args.port))
    except OSError:
        return attach_existing(url, settings, load)
    cmd = [sys.executable, "-m", "lucifer", "serve", "--data-dir", str(settings.data_dir), "--models", str(settings.models_file), "--evals", str(settings.eval_file), "--port", str(args.port)]
    if load:
        cmd += ["--load", load]
    logs = settings.data_dir / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / "backend.log"
    if load:
        Console(stderr=True).print(f"Loading Lucifer. Startup details: {path}", markup=False)
    with path.open("ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        frontend_failed = False
        try:
            headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
            deadline = time.monotonic() + startup_timeout(settings, load) + 20
            with httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Backend exited. Inspect {path}")
                    try:
                        response = client.get(url + "/health", headers=headers)
                        if response.is_success:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Backend startup timed out. Inspect {path}")
                    time.sleep(0.15)
            try:
                run_tui(url, settings)
            except Exception:
                # A frontend failure must not discard the backend's retained
                # conversation/thinking. A later run attaches to this owner.
                frontend_failed = True
                raise
        finally:
            if proc.poll() is None and not frontend_failed:
                # SIGTERM lets Uvicorn run lifespan cleanup, including model children.
                proc.terminate()
                try:
                    proc.wait(timeout=100)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main():
    p = parser()
    args = p.parse_args()
    if args.command is None:
        args = p.parse_args(["run"])
    try:
        settings = settings_from(args)
        color = getattr(args, "color", "auto")
        if color == "always":
            os.environ.pop("NO_COLOR", None)
            os.environ["FORCE_COLOR"] = "1"
            os.environ["COLORTERM"] = "truecolor"
        elif color == "never":
            os.environ["NO_COLOR"] = "1"
        if args.command == "init":
            args.directory.mkdir(parents=True, exist_ok=True)
            for name in ("models.yaml", "evals.yaml", "replay.jsonl"):
                path = args.directory / name
                if not path.exists():
                    path.write_bytes(files("lucifer").joinpath("resources", name).read_bytes())
            Console().print(f"Configuration ready: {args.directory.resolve()}")
        elif args.command == "doctor":
            doctor(settings)
        elif args.command == "tui":
            run_tui(args.api_url, settings, session=args.session, view=args.view)
        elif args.command == "serve":
            import uvicorn
            from .api import create_app
            if args.host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_token:
                raise ValueError("Set LUCIFER_API_TOKEN before binding beyond loopback")
            # Bound only HTTP connection draining during shutdown. Long-lived
            # agent SSE streams must not prevent lifespan cleanup from stopping
            # the owned agent/model processes; missions have no implicit timer.
            uvicorn.run(create_app(settings, autoload=args.load), host=args.host, port=args.port,
                        workers=1, timeout_graceful_shutdown=5)
        else:
            run_owned(args, settings)
    except (ValueError, RuntimeError, OSError) as exc:
        Console(stderr=True).print(str(exc), markup=False)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
