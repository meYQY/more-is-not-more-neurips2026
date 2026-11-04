"""Tests for runners/common.py — prompt building, counting, verification."""

import json
from pathlib import Path

import pytest
import yaml

from modbench.runners.common import (
    build_system_prompts,
    count_existing,
    result_path,
    verify_and_summarize,
)


# --- build_system_prompts ---


def test_build_system_prompts_no_persona():
    """Without personas, returns n_responses copies of base prompt."""
    entries = build_system_prompts(
        condition={"persona_depth": "none"},
        base_system_prompt="You are helpful.",
        n_responses=5,
        base_config_path=Path("dummy"),
    )
    assert len(entries) == 5
    assert all(e["name"] is None for e in entries)
    assert all(e["prompt"] == "You are helpful." for e in entries)


def test_build_system_prompts_no_persona_independent_dicts():
    """Each entry should be an independent dict (not shared reference)."""
    entries = build_system_prompts(
        condition={"persona_depth": "none"},
        base_system_prompt="test",
        n_responses=3,
        base_config_path=Path("dummy"),
    )
    entries[0]["name"] = "modified"
    assert entries[1]["name"] is None


def test_build_system_prompts_inline_personas():
    """Inline personas should be appended to base prompt."""
    entries = build_system_prompts(
        condition={"personas": ["You are bold.", "You are shy."]},
        base_system_prompt="Base.",
        n_responses=20,
        base_config_path=Path("dummy"),
    )
    assert len(entries) == 2
    assert entries[0]["prompt"] == "Base.\n\nYou are bold."
    assert entries[1]["prompt"] == "Base.\n\nYou are shy."


def test_build_system_prompts_from_file(tmp_path: Path):
    """Load personas from YAML file with depth level."""
    personas_yaml = {
        "personas": [
            {"id": 1, "name": "Alice", "role": "You are a teacher.", "basic": "Alice, 30, teacher."},
            {"id": 2, "name": "Bob", "role": "You are an engineer.", "basic": "Bob, 45, engineer."},
        ]
    }
    personas_path = tmp_path / "personas.yaml"
    with open(personas_path, "w") as f:
        yaml.dump(personas_yaml, f)

    entries = build_system_prompts(
        condition={"personas_file": "personas.yaml", "persona_depth": "role"},
        base_system_prompt="Base.",
        n_responses=20,
        base_config_path=tmp_path / "base.yaml",
    )
    assert len(entries) == 2
    assert entries[0]["name"] == "Alice"
    assert "You are a teacher." in entries[0]["prompt"]


def test_build_system_prompts_subset(tmp_path: Path):
    """persona_subset filters to specific IDs."""
    personas_yaml = {
        "personas": [
            {"id": 1, "name": "Alice", "role": "teacher"},
            {"id": 2, "name": "Bob", "role": "engineer"},
            {"id": 3, "name": "Carol", "role": "nurse"},
        ]
    }
    personas_path = tmp_path / "personas.yaml"
    with open(personas_path, "w") as f:
        yaml.dump(personas_yaml, f)

    entries = build_system_prompts(
        condition={"personas_file": "personas.yaml", "persona_depth": "role", "persona_subset": [1, 3]},
        base_system_prompt="Base.",
        n_responses=20,
        base_config_path=tmp_path / "base.yaml",
    )
    assert len(entries) == 2
    assert entries[0]["name"] == "Alice"
    assert entries[1]["name"] == "Carol"


def test_build_system_prompts_file_with_none_depth(tmp_path: Path):
    """personas_file + persona_depth='none' should raise ValueError."""
    personas_path = tmp_path / "personas.yaml"
    with open(personas_path, "w") as f:
        yaml.dump({"personas": [{"id": 1, "name": "A", "role": "x"}]}, f)

    with pytest.raises(ValueError, match="persona_depth"):
        build_system_prompts(
            condition={"personas_file": "personas.yaml", "persona_depth": "none"},
            base_system_prompt="Base.",
            n_responses=1,
            base_config_path=tmp_path / "base.yaml",
        )


# --- count_existing ---


def test_count_existing_no_file(tmp_path: Path):
    assert count_existing(tmp_path / "nonexistent.jsonl") == 0


def test_count_existing_valid_lines(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    records = [
        {"response": {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}},
        {"response": {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    assert count_existing(path) == 2


def test_count_existing_skips_corrupt_lines(tmp_path: Path):
    """Corrupt JSON lines should be ignored, not counted."""
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"valid": True}) + "\n")
        f.write('{"truncated\n')
        f.write(json.dumps({"also_valid": True}) + "\n")
    assert count_existing(path) == 2


def test_count_existing_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"a": 1}) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps({"b": 2}) + "\n")
    assert count_existing(path) == 2


# --- result_path ---


def test_result_path():
    p = result_path(Path("results/raw"), "single", "MOD-001")
    assert p == Path("results/raw/single/MOD-001.jsonl")


# --- verify_and_summarize ---


def _make_condition_dir(tmp_path: Path, n_tasks: int, n_per_task: int) -> Path:
    """Helper: create a condition dir with valid JSONL files."""
    d = tmp_path / "test_condition"
    d.mkdir()
    for i in range(1, n_tasks + 1):
        path = d / f"MOD-{i:03d}.jsonl"
        with open(path, "w") as f:
            for _ in range(n_per_task):
                record = {"response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}}
                f.write(json.dumps(record) + "\n")
    return d


def test_verify_and_summarize_success(tmp_path: Path):
    d = _make_condition_dir(tmp_path, n_tasks=3, n_per_task=5)
    verify_and_summarize(d, 3, 5)
    summary = json.loads((d / "_summary.json").read_text())
    assert summary["n_calls"] == 15
    assert summary["prompt_tokens"] == 1500
    assert summary["completion_tokens"] == 750


def test_verify_and_summarize_more_files_than_expected(tmp_path: Path):
    """Extra files from prior runs should not cause failure."""
    d = _make_condition_dir(tmp_path, n_tasks=5, n_per_task=3)
    verify_and_summarize(d, 2, 3)  # only asked for 2, but 5 exist — should pass


def test_verify_and_summarize_fewer_files_than_expected(tmp_path: Path):
    d = _make_condition_dir(tmp_path, n_tasks=2, n_per_task=5)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        verify_and_summarize(d, 3, 5)


def test_verify_and_summarize_wrong_record_count(tmp_path: Path):
    d = _make_condition_dir(tmp_path, n_tasks=2, n_per_task=3)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        verify_and_summarize(d, 2, 5)


def test_verify_and_summarize_no_summary_on_failure(tmp_path: Path):
    """_summary.json should NOT be written if verification fails."""
    d = _make_condition_dir(tmp_path, n_tasks=2, n_per_task=3)
    with pytest.raises(RuntimeError):
        verify_and_summarize(d, 2, 5)
    assert not (d / "_summary.json").exists()
