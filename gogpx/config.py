from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env if present (non-destructive)."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def get_graphhopper_key() -> str | None:
    return os.environ.get("GRAPHOPPER_API_KEY") or os.environ.get("GRAPHHOPPER_API_KEY")
