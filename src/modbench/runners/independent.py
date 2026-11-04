"""Runner for Independent framework conditions.

Handles: SINGLE, ROLE, PERSONA-*, ENHANCED, DIMENSION, TEMP-HIGH, RANDOM-SYS.
All share the same execution loop — build system prompts, call LLM once per
prompt, save results. Differences are purely in prompt construction (via yaml).
"""

import asyncio
from pathlib import Path

from ..llm import call_llm
from ..io import save_result
from .common import build_system_prompts, result_path, count_existing, verify_and_summarize


async def run(
    tasks: list[dict],
    config: dict,
    output_dir: Path,
    base_config_path: Path,
) -> None:
    """Run all tasks for an Independent condition."""
    condition = config["condition"]
    base_system_prompt = condition.get("system_prompt", config["system_prompt"])
    condition_name = condition["name"].lower()
    n_responses = config["experiment"]["n_responses"]
    condition_dir = output_dir / condition_name

    prompt_entries = build_system_prompts(
        condition, base_system_prompt, n_responses, base_config_path
    )
    n_per_task = len(prompt_entries)

    max_concurrent = config["experiment"].get("max_concurrent", 10)
    sem = asyncio.Semaphore(max_concurrent)

    async def limited_call(prompt: str, query: str) -> dict:
        async with sem:
            return await call_llm(prompt, query, config)

    totals = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    for i, task in enumerate(tasks, 1):
        path = result_path(output_dir, condition_name, task["task_id"])
        existing = count_existing(path)
        if existing >= n_per_task:
            print(f"[{i}/{len(tasks)}] {task['task_id']}: {existing} existing ≥ {n_per_task} expected, skipping", flush=True)
            continue
        if existing > 0:
            print(f"[{i}/{len(tasks)}] {task['task_id']}: resuming from {existing}/{n_per_task}", flush=True)

        entries_to_run = prompt_entries[existing:]
        calls = [
            limited_call(entry["prompt"], task["query"])
            for entry in entries_to_run
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)

        n_saved = 0
        for entry, result in zip(entries_to_run, results):
            if isinstance(result, BaseException):
                print(f"  ✗ {task['task_id']}: API error: {result}", flush=True)
                continue
            result["task_id"] = task["task_id"]
            result["persona_name"] = entry["name"]
            result["raw_question"] = task["query"]
            save_result(path, result)
            n_saved += 1
            usage = result["response"]["usage"]
            totals["n_calls"] += 1
            totals["prompt_tokens"] += usage["prompt_tokens"]
            totals["completion_tokens"] += usage["completion_tokens"]

        total_tok = totals["prompt_tokens"] + totals["completion_tokens"]
        final_count = count_existing(path)
        print(
            f"[{i}/{len(tasks)}] {task['task_id']}: {final_count}/{n_per_task} records "
            f"({n_saved} new, {total_tok:,} tokens so far)",
            flush=True,
        )

    verify_and_summarize(condition_dir, len(tasks), n_per_task)
