#!/usr/bin/env python3
"""Run the 30-task Multi-Agent prompt ablation.

Round 1 is generated once per task and shared exactly across neutral,
agreement, and disagreement. Only the round-2/3 interaction instruction
changes. All raw requests, responses, usage, and metadata are retained.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from modbench.llm import call_llm


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiment_config.yaml"
CONDITIONS = ("neutral", "agreement", "disagreement")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def source_root(config: dict[str, Any]) -> Path:
    """Repository root that holds tasks.csv, personas.yaml, and .env.

    A relative ``source.repository`` is resolved against this folder. The
    MODBENCH_ROOT environment variable overrides the configured value.
    """
    configured = os.environ.get("MODBENCH_ROOT") or config["source"]["repository"]
    path = Path(configured)
    return path if path.is_absolute() else (ROOT / path).resolve()


def stable_score(seed: int, task_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).digest()


def proportional_allocations(
    groups: dict[str, list[dict[str, str]]], n_total: int
) -> dict[str, int]:
    population = sum(len(rows) for rows in groups.values())
    quotas = {
        category: n_total * len(rows) / population
        for category, rows in groups.items()
    }
    allocation = {category: math.floor(quota) for category, quota in quotas.items()}
    remainder = n_total - sum(allocation.values())
    ranking = sorted(
        groups,
        key=lambda category: (
            -(quotas[category] - allocation[category]),
            category,
        ),
    )
    for category in ranking[:remainder]:
        allocation[category] += 1
    return allocation


def select_tasks(config: dict[str, Any]) -> list[dict[str, str]]:
    tasks_path = source_root(config) / config["source"]["tasks"]
    n_tasks = int(config["experiment"]["n_tasks"])
    selection_path = ROOT / "data" / f"selected_tasks_{n_tasks}.csv"
    manifest_path = ROOT / "data" / f"selection_manifest_{n_tasks}.json"
    seed = int(config["experiment"]["seed"])

    with tasks_path.open(newline="", encoding="utf-8-sig") as handle:
        all_tasks = list(csv.DictReader(handle))

    groups: dict[str, list[dict[str, str]]] = {}
    for task in all_tasks:
        groups.setdefault(task["subcategory"], []).append(task)
    allocation = proportional_allocations(groups, n_tasks)

    selected: list[dict[str, str]] = []
    for category in sorted(groups):
        ranked = sorted(
            groups[category],
            key=lambda task: stable_score(seed, task["task_id"]),
        )
        selected.extend(ranked[: allocation[category]])
    selected.sort(key=lambda task: task["task_id"])

    expected_ids = [task["task_id"] for task in selected]
    legacy_selection_path = ROOT / "data" / "selected_tasks.csv"
    if (
        n_tasks == 30
        and not selection_path.exists()
        and legacy_selection_path.exists()
    ):
        selection_path = legacy_selection_path

    if selection_path.exists():
        with selection_path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
        existing_ids = [task["task_id"] for task in existing]
        if existing_ids != expected_ids:
            raise RuntimeError(f"{selection_path.name} does not match the frozen selection.")
        return existing

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_tasks[0].keys())
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    write_json_atomic(
        manifest_path,
        {
            "seed": seed,
            "n_tasks": n_tasks,
            "source": str(tasks_path),
            "allocation_by_subcategory": allocation,
            "task_ids": expected_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return selected


def build_agents(config: dict[str, Any]) -> list[dict[str, str]]:
    personas_path = source_root(config) / config["source"]["personas"]
    with personas_path.open(encoding="utf-8") as handle:
        personas = yaml.safe_load(handle)["personas"]
    requested = set(config["experiment"]["persona_ids"])
    depth = config["experiment"]["persona_depth"]
    base_system = config["system_prompt"]
    selected = [persona for persona in personas if persona["id"] in requested]
    if len(selected) != len(requested):
        raise RuntimeError("One or more configured personas are missing.")
    return [
        {
            "id": persona["id"],
            "name": persona["name"],
            "system_prompt": f"{base_system}\n\n{persona[depth]}",
        }
        for persona in selected
    ]


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "generation": config["generation"],
        "extra_body": config["extra_body"],
    }


def discussion_prompt(
    *,
    question: str,
    agents: list[dict[str, str]],
    responses: dict[tuple[int, int], str],
    round_number: int,
    agent_index: int,
    condition: str,
    config: dict[str, Any],
) -> str:
    if round_number == 1:
        return config["prompts"]["round1"].format(question=question)

    parts = [f"A group is discussing the following question.\n\nQuestion: {question}"]
    for previous_round in range(1, round_number):
        parts.append(f"\nRound {previous_round}:")
        for index, agent in enumerate(agents):
            parts.append(
                f"{agent['name']}: {responses[(previous_round, index)]}"
            )

    current = [
        f"{agents[index]['name']}: {responses[(round_number, index)]}"
        for index in range(agent_index)
        if (round_number, index) in responses
    ]
    if current:
        parts.append(f"\nRound {round_number} (so far):")
        parts.extend(current)

    key = "round2" if round_number == 2 else "round3"
    parts.append(f"\n{config['prompts'][condition][key]}")
    return "\n".join(parts)


def enrich_record(
    record: dict[str, Any],
    *,
    task: dict[str, str],
    agent: dict[str, str],
    round_number: int,
    condition: str,
    shared_round1: bool,
) -> dict[str, Any]:
    output = deepcopy(record)
    output.update(
        {
            "task_id": task["task_id"],
            "persona_id": agent["id"],
            "persona_name": agent["name"],
            "raw_question": task["query"],
            "subcategory": task["subcategory"],
            "round_number": round_number,
            "prompt_condition": condition,
            "shared_round1": shared_round1,
        }
    )
    return output


async def generate_round1(
    task: dict[str, str],
    agents: list[dict[str, str]],
    config: dict[str, Any],
    call_sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    path = ROOT / "raw" / "_shared_round1" / f"{task['task_id']}.jsonl"
    existing = read_jsonl(path)
    if len(existing) == len(agents):
        return existing

    async def one(index: int) -> dict[str, Any]:
        prompt = discussion_prompt(
            question=task["query"],
            agents=agents,
            responses={},
            round_number=1,
            agent_index=index,
            condition="neutral",
            config=config,
        )
        async with call_sem:
            result = await call_llm(
                agents[index]["system_prompt"], prompt, model_config(config)
            )
        return enrich_record(
            result,
            task=task,
            agent=agents[index],
            round_number=1,
            condition="shared",
            shared_round1=True,
        )

    records = await asyncio.gather(*(one(index) for index in range(len(agents))))
    write_jsonl_atomic(path, records)
    return records


async def generate_branch(
    task: dict[str, str],
    condition: str,
    agents: list[dict[str, str]],
    round1: list[dict[str, Any]],
    config: dict[str, Any],
    call_sem: asyncio.Semaphore,
) -> None:
    output_path = ROOT / "raw" / condition / f"{task['task_id']}.jsonl"
    if len(read_jsonl(output_path)) == len(agents) * int(
        config["experiment"]["n_rounds"]
    ):
        return

    responses: dict[tuple[int, int], str] = {}
    records: list[dict[str, Any]] = []
    for index, shared in enumerate(round1):
        responses[(1, index)] = shared["response"]["content"]
        copied = deepcopy(shared)
        copied["prompt_condition"] = condition
        records.append(copied)

    for round_number in range(2, int(config["experiment"]["n_rounds"]) + 1):
        for index, agent in enumerate(agents):
            prompt = discussion_prompt(
                question=task["query"],
                agents=agents,
                responses=responses,
                round_number=round_number,
                agent_index=index,
                condition=condition,
                config=config,
            )
            async with call_sem:
                result = await call_llm(
                    agent["system_prompt"], prompt, model_config(config)
                )
            responses[(round_number, index)] = result["response"]["content"]
            records.append(
                enrich_record(
                    result,
                    task=task,
                    agent=agent,
                    round_number=round_number,
                    condition=condition,
                    shared_round1=False,
                )
            )

    write_jsonl_atomic(output_path, records)
    usage = {
        "prompt": sum(r["response"]["usage"]["prompt_tokens"] for r in records),
        "completion": sum(
            r["response"]["usage"]["completion_tokens"] for r in records
        ),
    }
    print(
        f"  ✓ {condition:12s} {task['task_id']} "
        f"({usage['prompt'] + usage['completion']:,} tokens)",
        flush=True,
    )


def summarize(config: dict[str, Any], tasks: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": config["model"]["name"],
        "conditions": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for condition in CONDITIONS:
        records: list[dict[str, Any]] = []
        complete = 0
        for task in tasks:
            current = read_jsonl(
                ROOT / "raw" / condition / f"{task['task_id']}.jsonl"
            )
            if len(current) == 15:
                complete += 1
                records.extend(current)
        result["conditions"][condition] = {
            "complete_tasks": complete,
            "records": len(records),
            "prompt_tokens": sum(
                record["response"]["usage"]["prompt_tokens"] for record in records
            ),
            "completion_tokens": sum(
                record["response"]["usage"]["completion_tokens"]
                for record in records
            ),
        }
    return result


async def run(dry_run: bool, limit: int | None) -> None:
    config = load_config()
    load_dotenv(source_root(config) / ".env")
    api_env = config["model"]["api_key_env"]
    if not os.environ.get(api_env):
        raise RuntimeError(f"{api_env} is not configured.")

    tasks = select_tasks(config)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        tasks = tasks[:limit]
    agents = build_agents(config)
    manifest = {
        "config": config,
        "conditions": list(CONDITIONS),
        "task_ids": [task["task_id"] for task in tasks],
        "personas": [{"id": a["id"], "name": a["name"]} for a in agents],
        "expected_unique_generation_calls": (
            len(tasks) * len(agents)
            + len(tasks) * len(agents) * 2 * len(CONDITIONS)
        ),
        "dry_run": dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(ROOT / "generation_manifest.json", manifest)
    print(
        f"Design: {len(tasks)} tasks × {len(agents)} agents × 3 rounds; "
        f"conditions={','.join(CONDITIONS)}",
        flush=True,
    )
    print(
        f"Model: {config['model']['name']} "
        f"(thinking={config['extra_body']['thinking']['type']})",
        flush=True,
    )
    print(
        f"Expected unique calls: {manifest['expected_unique_generation_calls']}",
        flush=True,
    )
    if dry_run:
        print("Dry run complete; no API calls made.", flush=True)
        return

    call_sem = asyncio.Semaphore(
        int(config["experiment"]["max_concurrent_calls"])
    )
    round1_sem = asyncio.Semaphore(
        int(config["experiment"]["round1_task_concurrency"])
    )
    round1_by_task: dict[str, list[dict[str, Any]]] = {}

    async def prepare(task: dict[str, str]) -> None:
        async with round1_sem:
            round1_by_task[task["task_id"]] = await generate_round1(
                task, agents, config, call_sem
            )
            print(f"  ✓ shared-r1    {task['task_id']}", flush=True)

    print("Phase 1/2: shared round 1", flush=True)
    await asyncio.gather(*(prepare(task) for task in tasks))

    branch_sem = asyncio.Semaphore(
        int(config["experiment"]["branch_concurrency"])
    )

    async def branch(task: dict[str, str], condition: str) -> None:
        async with branch_sem:
            await generate_branch(
                task,
                condition,
                agents,
                round1_by_task[task["task_id"]],
                config,
                call_sem,
            )

    print("Phase 2/2: neutral/agreement/disagreement branches", flush=True)
    await asyncio.gather(
        *(branch(task, condition) for task in tasks for condition in CONDITIONS)
    )

    summary = summarize(config, tasks)
    write_json_atomic(ROOT / "generation_summary.json", summary)
    write_json_atomic(ROOT / f"generation_summary_{len(tasks)}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate design without API calls."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N frozen tasks; completed outputs are reused later.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.dry_run, args.limit))


if __name__ == "__main__":
    main()
