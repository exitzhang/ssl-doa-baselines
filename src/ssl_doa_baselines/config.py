"""TOML configuration loading and shallow merging."""

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.8--3.10
    import tomli as _toml


def load_toml(path: str) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        return _toml.load(f)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_configs(paths: Iterable[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        if p:
            cfg = deep_merge(cfg, load_toml(p))
    return cfg


def model_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either a flat ``[model]`` table or a flat mapping."""
    if "model" in config and isinstance(config["model"], dict):
        return config["model"]
    return config
