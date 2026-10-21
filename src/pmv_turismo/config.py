import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

# Secrets never live in YAML: they are injected from the environment (or a
# local, git-ignored .env file) into these config paths.
SECRET_ENV_VARS = {
    "ORS_API_KEY": ("parameters", "planning", "routing", "api_key"),
    "GOOGLE_PLACES_API_KEY": ("ingestion", "google_places", "api_key"),
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader; existing environment variables take precedence."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _inject_secrets(data: Dict[str, Any]) -> None:
    for env_var, keys in SECRET_ENV_VARS.items():
        value = os.environ.get(env_var)
        if not value:
            continue
        node = data
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value


@dataclass
class Config:
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        load_dotenv()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _inject_secrets(data)
        return cls(data=data)
