"""Render reproducible terminal previews against Lucifer's real demo API.

Run from the source checkout: .venv/bin/python scripts/preview_tui.py
No model downloads or training occur. GPU values come from hardware discovery.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import httpx

# Preview in full color even when a CI shell explicitly requests plain output.
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["COLORTERM"] = "truecolor"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lucifer.api import create_app  # noqa: E402
from lucifer.client import APIClient  # noqa: E402
from lucifer.config import Settings  # noqa: E402
from lucifer.tui import LuciferApp  # noqa: E402


def save_preview(terminal: LuciferApp, target: Path):
    # Rich defaults to web-hosted Fira Code. Use an installed monospace family so
    # these exported assets stay readable when opened without network access.
    svg = terminal.export_screenshot(simplify=True)
    svg = re.sub(r"    @font-face \{.*?\n    \}", "", svg, flags=re.S)
    svg = svg.replace("font-family: Fira Code, monospace;", "font-family: 'DejaVu Sans Mono', monospace;")
    target.write_text(svg, encoding="utf-8")


async def main():
    target = Path(__file__).resolve().parents[1] / "docs" / "assets"
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lucifer-preview-") as temporary:
        root = Path(temporary)
        settings = Settings(data_dir=root / "state", models_file=root / "models.yaml", eval_file=root / "evals.yaml")
        backend = create_app(settings, autoload="demo")
        async with backend.router.lifespan_context(backend):
            terminal = LuciferApp(client=APIClient(transport=httpx.ASGITransport(app=backend)))
            async with terminal.run_test(size=(140, 45)) as pilot:
                async def submit(text):
                    terminal.submit(text)
                    await pilot.pause()
                    await terminal.workers.wait_for_complete()
                    await pilot.pause()
                await submit("What is the capital of France?")
                await submit("/chat save")
                await submit("/feedback good")
                await submit("/privacy training on")
                await submit("/sleep start --dry-run")
                await submit(f"/chat open {terminal.conversation['id']}")
                await submit("Return JSON containing ok: true.")
                await terminal.refresh_panels()
                save_preview(terminal, target / "terminal-demo.svg")
                await pilot.resize_terminal(88, 30)
                await pilot.pause()
                await submit("/chat new")
                await submit("What is 2 + 2?")
                save_preview(terminal, target / "terminal-demo-compact.svg")
    print(f"Demo previews: {target / 'terminal-demo.svg'} and {target / 'terminal-demo-compact.svg'}")


if __name__ == "__main__":
    asyncio.run(main())
