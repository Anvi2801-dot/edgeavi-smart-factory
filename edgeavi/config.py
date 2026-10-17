"""Config loading. Every path in config.yaml is resolved relative to the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve every path-like value against the repo root
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(resolve(value))
    for key in ("fp32_onnx", "int8_onnx", "checkpoint"):
        cfg["model"][key] = str(resolve(cfg["model"][key]))
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
