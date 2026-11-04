"""Shared utilities for all runner types."""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def build_system_prompts(
    condition: dict,
    base_system_prompt: str,
    n_responses: int,
    base_config_path: Path,
) -> list[dict[str, str | None]]:
    """Build system prompt list with persona names.

    Returns list of {"name": str|None, "prompt": str}.
    Priority: personas_file > inline personas > repeat base prompt.
    """
    if "personas_file" in condition:
        level = condition.get("persona_depth", "none")
        if level == "none":
            raise ValueError("personas_file requires a valid persona_depth (role/basic/mid/pro), not 'none'")
        personas_path = base_config_path.parent / condition["personas_file"]
        with open(personas_path, encoding="utf-8") as f:
            personas_data = yaml.safe_load(f)
        personas = personas_data["personas"]
        if "persona_subset" in condition:
            subset_ids = set(condition["persona_subset"])
            personas = [p for p in personas if p["id"] in subset_ids]
        return [
            {"name": p["name"], "prompt": f"{base_system_prompt}\n\n{p[level]}"}
            for p in personas
        ]
    if "personas" in condition:
        return [
            {"name": None, "prompt": f"{base_system_prompt}\n\n{p}"}
            for p in condition["personas"]
        ]
    return [{"name": None, "prompt": base_system_prompt} for _ in range(n_responses)]


def result_path(output_dir: Path, condition_name: str, task_id: str) -> Path:
    return output_dir / condition_name / f"{task_id}.jsonl"


def count_existing(path: Path) -> int:
    """Count valid JSON lines in a file. Corrupt lines are ignored."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count


def verify_and_summarize(
    condition_dir: Path, n_tasks: int, n_per_task: int,
) -> None:
    """Verify record counts and write _summary.json. Raises on mismatch."""
    n_calls = 0
    prompt = 0
    completion = 0
    errors = []

    files = sorted(condition_dir.glob("*.jsonl"))
    if len(files) < n_tasks:
        errors.append(f"Expected at least {n_tasks} task files, found {len(files)}")

    for jsonl_file in files:
        file_records = 0
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    usage = json.loads(line)["response"]["usage"]
                except (json.JSONDecodeError, KeyError) as e:
                    errors.append(f"{jsonl_file.name}:{line_num}: corrupt record: {e}")
                    continue
                file_records += 1
                n_calls += 1
                prompt += usage["prompt_tokens"]
                completion += usage["completion_tokens"]
        if file_records < n_per_task:
            errors.append(f"{jsonl_file.name}: {file_records} records (expected at least {n_per_task})")

    if errors:
        for e in errors:
            print(f"  ✗ {e}", flush=True)
        raise RuntimeError(f"Data integrity check failed for {condition_dir.name}: {errors}")

    summary = {
        "n_calls": n_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = condition_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Done. {len(files)} tasks × {n_per_task} records = {n_calls} total. "
        f"Tokens: {prompt + completion:,}. ✓",
        flush=True,
    )
