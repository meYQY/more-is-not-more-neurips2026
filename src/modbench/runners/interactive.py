"""Runner for Interactive framework conditions.

Handles: MINIMAL-BARE, MINIMAL+BASIC, MINIMAL+MID, MINIMAL+PRO.
Multiple agents discuss a question across rounds. Each agent sees all
previous responses, enabling cross-persona interaction.

Round 1: all agents answer independently (parallelized).
Round 2+: sequential within round — each agent sees previous speakers.
"""

import asyncio
from pathlib import Path

import yaml

from ..llm import call_llm
from ..io import save_result
from .common import result_path, count_existing, verify_and_summarize


def _build_agents(
    condition: dict,
    base_system_prompt: str,
    base_config_path: Path,
) -> list[dict[str, str]]:
    """Build agent list with names and system prompts.

    With personas_file: load subset, use persona names.
    Without: create n_agents anonymous participants.
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
            {"name": p["name"], "system_prompt": f"{base_system_prompt}\n\n{p[level]}"}
            for p in personas
        ]
    n_agents = condition.get("n_agents", 5)
    return [
        {"name": f"Participant {i + 1}", "system_prompt": base_system_prompt}
        for i in range(n_agents)
    ]


def _format_user_prompt(
    question: str,
    agents: list[dict],
    responses: dict[tuple[int, int], str],
    round_num: int,
    agent_idx: int,
) -> str:
    """Build user prompt with discussion history up to current point."""
    if round_num == 1:
        return (
            "A group is discussing the following question. "
            "Please share your perspective.\n\n"
            f"Question: {question}"
        )

    parts = [f"A group is discussing the following question.\n\nQuestion: {question}"]

    # All completed previous rounds
    for r in range(1, round_num):
        parts.append(f"\nRound {r}:")
        for i, agent in enumerate(agents):
            parts.append(f"{agent['name']}: {responses[(r, i)]}")

    # Current round: agents who already spoke this round
    spoke_this_round = [
        f"{agents[i]['name']}: {responses[(round_num, i)]}"
        for i in range(agent_idx)
        if (round_num, i) in responses
    ]
    if spoke_this_round:
        parts.append(f"\nRound {round_num} (so far):")
        parts.extend(spoke_this_round)

    suffix = "Please respond to the discussion." if round_num == 2 else "Please continue the discussion."
    parts.append(f"\n{suffix}")
    return "\n".join(parts)


async def _run_discussion(
    agents: list[dict],
    question: str,
    n_rounds: int,
    config: dict,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Run one multi-agent discussion. Returns list of per-call records."""
    responses: dict[tuple[int, int], str] = {}
    records: list[dict] = []

    for round_num in range(1, n_rounds + 1):
        if round_num == 1:
            # Round 1: all agents independent — parallelize
            async def _r1_call(idx: int) -> dict:
                async with sem:
                    prompt = _format_user_prompt(question, agents, responses, 1, idx)
                    return await call_llm(agents[idx]["system_prompt"], prompt, config)

            r1_results = await asyncio.gather(*[_r1_call(i) for i in range(len(agents))])
            for idx, result in enumerate(r1_results):
                responses[(1, idx)] = result["response"]["content"]
                result["agent_name"] = agents[idx]["name"]
                result["round_number"] = 1
                records.append(result)
        else:
            # Round 2+: sequential — each agent sees earlier speakers in this round
            for idx, agent in enumerate(agents):
                prompt = _format_user_prompt(question, agents, responses, round_num, idx)
                async with sem:
                    result = await call_llm(agent["system_prompt"], prompt, config)
                responses[(round_num, idx)] = result["response"]["content"]
                result["agent_name"] = agent["name"]
                result["round_number"] = round_num
                records.append(result)

    return records


async def run(
    tasks: list[dict],
    config: dict,
    output_dir: Path,
    base_config_path: Path,
) -> None:
    """Run all tasks for an Interactive condition."""
    condition = config["condition"]
    base_system_prompt = condition.get("system_prompt", config["system_prompt"])
    condition_name = condition["name"].lower()
    condition_dir = output_dir / condition_name
    max_concurrent = config["experiment"].get("max_concurrent", 10)
    sem = asyncio.Semaphore(max_concurrent)
    # Task-level concurrency: Interactive rounds 2+ are sequential within a task,
    # so running multiple tasks in parallel is the main lever for speedup.
    task_concurrency = config["experiment"].get("task_concurrency", 5)
    task_sem = asyncio.Semaphore(task_concurrency)

    n_rounds = condition.get("n_rounds", 3)
    agents = _build_agents(condition, base_system_prompt, base_config_path)
    n_per_task = len(agents) * n_rounds

    totals = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def _process_task(i: int, task: dict) -> None:
        async with task_sem:
            path = result_path(output_dir, condition_name, task["task_id"])
            existing = count_existing(path)
            if existing >= n_per_task:
                print(f"[{i}/{len(tasks)}] {task['task_id']}: {existing} existing ≥ {n_per_task} expected, skipping", flush=True)
                return
            if existing > 0:
                # Partial discussion is unusable (rounds depend on each other).
                # Truncate and re-run from scratch.
                print(f"  Warning: {existing} partial records for {task['task_id']}, truncating and re-running", flush=True)
                with open(path, "w", encoding="utf-8") as f:
                    pass  # empty the file

            try:
                records = await _run_discussion(agents, task["query"], n_rounds, config, sem)
            except Exception as e:
                print(f"  ✗ {task['task_id']}: discussion error: {e}", flush=True)
                return

            for record in records:
                record["task_id"] = task["task_id"]
                record["persona_name"] = record.pop("agent_name")
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
                f"({len(agents)} agents × {n_rounds} rounds, {total_tok:,} tokens so far)",
                flush=True,
            )

    await asyncio.gather(*[_process_task(i, t) for i, t in enumerate(tasks, 1)])

    verify_and_summarize(condition_dir, len(tasks), n_per_task)
