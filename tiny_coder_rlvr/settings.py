"""Load project knobs from ``config/config.yaml``.

Call ``load_config(path)`` to reload (e.g. tests / alternate configs).
Attributes are available as ``settings.pass_reward``, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Legacy ALL_CAPS wandb keys still accepted when reading older configs.
_LEGACY_KEY_MAP = {
    "WANDB_ENABLED": "wandb_enabled",
    "WANDB_PROJECT": "wandb_project",
    "WANDB_ENTITY": "wandb_entity",
    "WANDB_RUN_NAME": "wandb_run_name",
}

_data: dict[str, Any] = {}


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for old, new in _LEGACY_KEY_MAP.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config into the module-level settings namespace."""
    global _data
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    _data = _normalize(raw)
    return _data


def as_dict() -> dict[str, Any]:
    if not _data:
        load_config()
    return dict(_data)


def get(key: str, default: Any = None) -> Any:
    if not _data:
        load_config()
    return _data.get(key, default)


def __getattr__(name: str) -> Any:
    if not _data:
        load_config()
    if name in _data:
        return _data[name]
    raise AttributeError(f"settings has no attribute {name!r}")


load_config()
