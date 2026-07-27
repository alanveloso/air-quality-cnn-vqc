"""Load YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return repository root (parent of config/)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load PoC YAML config; default is config/poc.yaml at repo root."""
    cfg_path = Path(path) if path else project_root() / "config" / "poc.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
