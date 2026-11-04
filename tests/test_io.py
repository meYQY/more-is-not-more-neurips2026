"""Tests for io.py — config loading and merging."""

import json
from pathlib import Path

import yaml

from modbench.io import load_config, load_tasks, save_result


def test_load_config_base_only(tmp_path: Path):
    base = {"model": {"name": "test"}, "generation": {"temperature": 0.7}}
    base_path = tmp_path / "base.yaml"
    with open(base_path, "w") as f:
        yaml.dump(base, f)

    config = load_config(base_path)
    assert config["model"]["name"] == "test"
    assert "condition" not in config


def test_load_config_with_condition(tmp_path: Path):
    base = {"model": {"name": "test"}, "generation": {"temperature": 0.7}}
    condition = {"name": "SINGLE", "framework": "independent"}

    base_path = tmp_path / "base.yaml"
    cond_path = tmp_path / "condition.yaml"
    with open(base_path, "w") as f:
        yaml.dump(base, f)
    with open(cond_path, "w") as f:
        yaml.dump(condition, f)

    config = load_config(base_path, cond_path)
    assert config["model"]["name"] == "test"
    assert config["condition"]["name"] == "SINGLE"


def test_load_tasks(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    csv_path.write_text("task_id,query\nMOD-001,What is life?\nMOD-002,Why?\n")

    tasks = load_tasks(csv_path)
    assert len(tasks) == 2
    assert tasks[0]["task_id"] == "MOD-001"
    assert tasks[1]["query"] == "Why?"


def test_save_result_creates_dir(tmp_path: Path):
    path = tmp_path / "sub" / "dir" / "test.jsonl"
    save_result(path, {"key": "value"})
    assert path.exists()
    with open(path) as f:
        record = json.loads(f.readline())
    assert record["key"] == "value"


def test_save_result_appends(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    save_result(path, {"a": 1})
    save_result(path, {"b": 2})
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 2
