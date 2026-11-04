#!/usr/bin/env python3
"""Run the frozen 100-task diversity-coverage prompt condition.

The already-saved shared round 1 is reused exactly. Only rounds 2 and 3 are
generated, and every other experimental setting matches the completed prompt
ablation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

import run_generation as base


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "coverage_config.yaml"
CONDITION = "coverage"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def coverage_summary(
    config: dict[str, Any], tasks: list[dict[str, str]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    complete = 0
    for task in tasks:
        current = base.read_jsonl(
            ROOT / "raw" / CONDITION / f"{task['task_id']}.jsonl"
        )
        if len(current) == 15:
            complete += 1
            records.extend(current)
    intervention = [
        record for record in records if int(record["round_number"]) >= 2
    ]
    return {
        "model": config["model"]["name"],
        "condition": CONDITION,
        "complete_tasks": complete,
        "records": len(records),
        "unique_intervention_calls": len(intervention),
        "prompt_tokens_rounds_2_3": sum(
            record["response"]["usage"]["prompt_tokens"]
            for record in intervention
        ),
        "completion_tokens_rounds_2_3": sum(
            record["response"]["usage"]["completion_tokens"]
            for record in intervention
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def validate_shared_round1(
    tasks: list[dict[str, str]], n_agents: int
) -> dict[str, list[dict[str, Any]]]:
    shared: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        task_id = task["task_id"]
        records = base.read_jsonl(
            ROOT / "raw" / "_shared_round1" / f"{task_id}.jsonl"
        )
        if len(records) != n_agents:
            raise RuntimeError(
                f"{task_id}: expected {n_agents} saved shared-round-1 records; "
                f"found {len(records)}"
            )
        shared[task_id] = records
    return shared


async def run(dry_run: bool) -> None:
    config = load_config()
    load_dotenv(base.source_root(config) / ".env")
    api_env = config["model"]["api_key_env"]
    if not os.environ.get(api_env):
        raise RuntimeError(f"{api_env} is not configured.")

    tasks = base.select_tasks(config)
    agents = base.build_agents(config)
    shared = validate_shared_round1(tasks, len(agents))
    prompt = config["prompts"][CONDITION]
    manifest = {
        "config": config,
        "condition": CONDITION,
        "task_ids": [task["task_id"] for task in tasks],
        "personas": [
            {"id": agent["id"], "name": agent["name"]} for agent in agents
        ],
        "reused_shared_round1": True,
        "expected_new_calls": len(tasks) * len(agents) * 2,
        "frozen_round2_prompt": prompt["round2"],
        "frozen_round3_prompt": prompt["round3"],
        "dry_run": dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    base.write_json_atomic(ROOT / "coverage_generation_manifest.json", manifest)

    print(
        f"Coverage design: {len(tasks)} tasks × {len(agents)} agents × "
        "2 new discussion rounds",
        flush=True,
    )
    print(
        f"Model: {config['model']['name']} "
        f"(thinking={config['extra_body']['thinking']['type']})",
        flush=True,
    )
    print(
        f"Expected new calls: {manifest['expected_new_calls']}; "
        "saved shared round 1 will be reused",
        flush=True,
    )
    if dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        print("Dry run complete; no API calls made.", flush=True)
        return

    call_sem = asyncio.Semaphore(
        int(config["experiment"]["max_concurrent_calls"])
    )
    branch_sem = asyncio.Semaphore(
        int(config["experiment"]["branch_concurrency"])
    )

    async def generate(task: dict[str, str]) -> None:
        async with branch_sem:
            await base.generate_branch(
                task,
                CONDITION,
                agents,
                shared[task["task_id"]],
                config,
                call_sem,
            )

    await asyncio.gather(*(generate(task) for task in tasks))
    summary = coverage_summary(config, tasks)
    base.write_json_atomic(
        ROOT / "coverage_generation_summary_100.json", summary
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen design without API calls.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    main()
