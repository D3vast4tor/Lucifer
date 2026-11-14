from dataclasses import dataclass, field
from pathlib import Path
import os

import yaml


def installation_root() -> Path:
    return Path(os.environ.get("LUCIFER_ROOT", "/home/Lucifer")).expanduser()


def default_data_dir() -> Path:
    root = installation_root()
    return root / "Soul/var/state/runtime" if (root / "Soul/lucifer").is_dir() else Path("~/.local/share/lucifer").expanduser()


def default_config(name: str) -> Path:
    root = installation_root() / "Soul/configs"
    return root / name if root.is_dir() else Path("configs") / name


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LUCIFER_DATA_DIR", default_data_dir())).expanduser())
    models_file: Path = field(default_factory=lambda: Path(os.environ.get("LUCIFER_MODELS", default_config("models.yaml"))))
    eval_file: Path = field(default_factory=lambda: Path(os.environ.get("LUCIFER_EVALS", default_config("evals.yaml"))))
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout: float = 180.0
    request_timeout: float = 600.0
    api_token: str | None = field(default_factory=lambda: os.environ.get("LUCIFER_API_TOKEN"))
    default_model: str | None = field(default_factory=lambda: os.environ.get("LUCIFER_DEFAULT_MODEL"))
    agent_workspace: Path = field(default_factory=lambda: Path(os.environ.get("LUCIFER_WORKSPACE", installation_root() if installation_root().is_dir() else os.getcwd())))
    agent_mcp_file: Path | None = field(default_factory=lambda: Path(os.environ["LUCIFER_MCP_CONFIG"]) if os.environ.get("LUCIFER_MCP_CONFIG") else None)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.models_file = Path(self.models_file).expanduser().resolve()
        self.eval_file = Path(self.eval_file).expanduser().resolve()
        self.agent_workspace = Path(self.agent_workspace).expanduser().resolve()
        if self.agent_mcp_file is not None:
            self.agent_mcp_file = Path(self.agent_mcp_file).expanduser().resolve()
        if self.default_model is None and self.models_file.exists():
            registry = yaml.safe_load(self.models_file.read_text()) or {}
            self.default_model = registry.get("default_model")
        if self.default_model is not None and (not isinstance(self.default_model, str) or not self.default_model.strip()):
            raise ValueError("default_model must be a nonempty model id")
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_dir.chmod(0o700)
