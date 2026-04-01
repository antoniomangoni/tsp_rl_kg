from __future__ import annotations

import copy
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ConfigMapping = dict[str, Any]


def load_config_file(path: str | Path) -> ConfigMapping:
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    elif suffix == ".toml":
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    else:
        raise ValueError("Unsupported config file format. Use a .json or .toml file.")

    if not isinstance(data, Mapping):
        raise ValueError("Config file root must be a mapping.")

    return copy.deepcopy(dict(data))


def merge_nested_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> ConfigMapping:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def find_mapping_section(
    data: Mapping[str, Any],
    *candidate_paths: tuple[str, ...],
) -> ConfigMapping | None:
    for path in candidate_paths:
        value = _get_nested_value(data, path)
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    return None


def find_list_section(
    data: Mapping[str, Any],
    *candidate_paths: tuple[str, ...],
) -> list[Any] | None:
    for path in candidate_paths:
        value = _get_nested_value(data, path)
        if isinstance(value, list):
            return copy.deepcopy(value)
    return None


def _get_nested_value(data: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current
