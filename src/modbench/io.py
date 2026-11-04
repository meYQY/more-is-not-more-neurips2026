"""Data I/O: load configs, load tasks, save results."""

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(base_path: Path, condition_path: Path | None = None) -> dict:
    """Load base config, optionally merge with a condition config."""
    with open(base_path) as f:
        config = yaml.safe_load(f)
    if condition_path:
        with open(condition_path) as f:
            condition = yaml.safe_load(f)
        config["condition"] = condition
    return config


def load_tasks(path: Path) -> list[dict[str, str]]:
    """Load tasks from CSV. Returns list of dicts with task_id, query, etc."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_result(path: Path, record: dict[str, Any]) -> None:
    """Append one result record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
