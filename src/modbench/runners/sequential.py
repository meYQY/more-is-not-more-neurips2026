"""Runner for Sequential framework conditions.

Handles: MT-NONE, MT+BASIC, MT+MID, MT+PRO.
Each persona gets one multi-turn conversation. Turns are sequential within
a conversation (model sees full history), conversations run in parallel.
"""

import asyncio
from pathlib import Path

from ..llm import call_llm_messages
from ..io import save_result
from .common import build_system_prompts, result_path, count_existing, verify_and_summarize


async def _run_conversation(
    system_prompt: str,
    question: str,
    follow_up: str,
    n_turns: int,
    config: dict,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Run one multi-turn conversation. Returns list of per-turn records.

    Semaphore is held for the entire conversation (all turns). This means
    max_concurrent limits parallel conversations, not parallel API calls.
    Within each conversation, turns are necessarily sequential.
    """
    async with sem:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        records = []
        for turn in range(n_turns):
            result = await call_llm_messages(list(messages), config)
            result["turn_number"] = turn + 1
            records.append(result)
            messages.append({"role": "assistant", "content": result["response"]["content"]})
            if turn < n_turns - 1:
                messages.append({"role": "user", "content": follow_up})
        return records


async def run(
    tasks: list[dict],
    config: dict,
    output_dir: Path,
    base_config_path: Path,
) -> None:
    """Run all tasks for a Sequential condition."""
    condition = config["condition"]
    base_system_prompt = condition.get("system_prompt", config["system_prompt"])
    condition_name = condition["name"].lower()
    n_responses = condition.get("n_responses", config["experiment"]["n_responses"])
    condition_dir = output_dir / condition_name
    max_concurrent = config["experiment"].get("max_concurrent", 10)
    sem = asyncio.Semaphore(max_concurrent)

    seq_config = config.get("sequential", {})
    n_turns = condition.get("n_turns", seq_config.get("n_turns", 3))
    follow_up = condition.get("follow_up", seq_config.get("follow_up", "What else do you think about this topic?"))

    prompt_entries = build_system_prompts(
        condition, base_system_prompt, n_responses, base_config_path
    )
    n_per_task = len(prompt_entries) * n_turns

    totals = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    for i, task in enumerate(tasks, 1):
        path = result_path(output_dir, condition_name, task["task_id"])
        existing = count_existing(path)
        if existing >= n_per_task:
            print(f"[{i}/{len(tasks)}] {task['task_id']}: {existing} existing ≥ {n_per_task} expected, skipping", flush=True)
            continue

        # Skip fully completed conversations, truncate any partial records
        completed_convs = existing // n_turns
        partial = existing % n_turns
        if partial > 0:
            print(f"  Warning: {partial} partial records from incomplete conversation, truncating", flush=True)
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[: completed_convs * n_turns])
        entries_to_run = prompt_entries[completed_convs:]

        conversations = [
            _run_conversation(entry["prompt"], task["query"], follow_up, n_turns, config, sem)
            for entry in entries_to_run
        ]
        all_results = await asyncio.gather(*conversations, return_exceptions=True)

        n_saved_convs = 0
        for entry, result in zip(entries_to_run, all_results):
            if isinstance(result, BaseException):
                print(f"  ✗ {task['task_id']}: conversation error: {result}", flush=True)
                continue
            n_saved_convs += 1
            for record in result:
                record["task_id"] = task["task_id"]
                record["persona_name"] = entry["name"]
                record["raw_question"] = task["query"]
                save_result(path, record)
                usage = record["response"]["usage"]
                totals["n_calls"] += 1
                totals["prompt_tokens"] += usage["prompt_tokens"]
                totals["completion_tokens"] += usage["completion_tokens"]

        final_count = count_existing(path)
        total_tok = totals["prompt_tokens"] + totals["completion_tokens"]
        print(
            f"[{i}/{len(tasks)}] {task['task_id']}: {final_count}/{n_per_task} records "
            f"({n_saved_convs} convs × {n_turns} turns, {total_tok:,} tokens so far)",
            flush=True,
        )

    verify_and_summarize(condition_dir, len(tasks), n_per_task)
