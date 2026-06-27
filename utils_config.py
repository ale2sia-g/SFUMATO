"""Configuration loading helpers for the SFUMATO pipeline.

JSON is the recommended format because it needs no third-party dependency.
YAML is supported only when PyYAML is available in the execution environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
            return _attach_config_paths(config, path)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML configs require PyYAML. Use the provided JSON config "
                "or install PyYAML in the environment."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return _attach_config_paths(config, path)

    raise ValueError(f"Unsupported config format: {path.suffix}")


def _attach_config_paths(config: dict[str, Any], path: Path) -> dict[str, Any]:
    config_dir = path.resolve().parent
    project_root = config_dir.parent if config_dir.name == "configs" else config_dir
    config["__config_path"] = str(path.resolve())
    config["__config_dir"] = str(config_dir)
    config["__project_root"] = str(project_root)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = Path(config.get("__project_root", "."))
    return project_root / path


def require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {section}")
    return value


def get_path(
    config: dict[str, Any],
    key: str,
    default: str | Path | None = None,
) -> Path:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"Missing required path config value: {key}")
    return resolve_path(config, value)


def default_results_dir(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    name = cache.name
    if name.startswith("preprocess_cache"):
        return cache.with_name(name.replace("preprocess_cache", "results", 1))
    return cache.parent / f"results_{name}"