#!/usr/bin/env python3
"""Run extraction, embedding, metrics, paired inference, and report generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from scipy.stats import wilcoxon

from modbench.embedding import embed_texts
from modbench.extractor import extract_condition
from modbench.metrics import cluster_count, compute_metrics, vendi_score


ROOT = Path(__file__).resolve().parent
# Repository root (holds .env); override with the MODBENCH_ROOT environment variable.
SOURCE_REPO = Path(os.environ.get("MODBENCH_ROOT", ROOT.parents[1])).resolve()
CONDITIONS = ("neutral", "agreement", "disagreement")
SEED = 1529
N_BOOTSTRAP = 20_000
N_RAREFACTION = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_selected_ids() -> list[str]:
    import csv

    config = yaml.safe_load(
        (ROOT / "experiment_config.yaml").read_text(encoding="utf-8")
    )
    n_tasks = int(config["experiment"]["n_tasks"])
    path = ROOT / "data" / f"selected_tasks_{n_tasks}.csv"
    if n_tasks == 30 and not path.exists():
        path = ROOT / "data" / "selected_tasks.csv"
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return [row["task_id"] for row in csv.DictReader(handle)]


async def run_extraction(force: bool) -> None:
    config_path = ROOT / "extraction_config.yaml"
    for condition in CONDITIONS:
        print(f"Extracting {condition}...", flush=True)
        await extract_condition(
            config_path=config_path,
            raw_dir=ROOT / "raw" / condition,
            extracted_dir=ROOT / "extracted" / condition,
            max_concurrent=30,
            force=force,
        )


def synchronize_shared_round1_extractions() -> bool:
    """Make extracted round 1 identical, matching the shared raw responses.

    Temperature-zero API extraction is not guaranteed to be bitwise
    deterministic. The neutral extraction of each shared round-1 record is
    therefore treated as the single extraction and copied into the other two
    conditions. This prevents extractor jitter from entering an all-round
    condition contrast.
    """
    changed = False
    for task_id in load_selected_ids():
        neutral_path = ROOT / "extracted" / "neutral" / f"{task_id}.jsonl"
        neutral = read_jsonl(neutral_path)
        if len(neutral) != 15 or any(
            int(record["round_number"]) != 1 for record in neutral[:5]
        ):
            raise RuntimeError(f"{task_id}: malformed neutral extraction")
        shared_opinions = [record.get("opinions", []) for record in neutral[:5]]

        for condition in ("agreement", "disagreement"):
            path = ROOT / "extracted" / condition / f"{task_id}.jsonl"
            records = read_jsonl(path)
            if len(records) != 15 or any(
                int(record["round_number"]) != 1 for record in records[:5]
            ):
                raise RuntimeError(f"{condition}/{task_id}: malformed extraction")
            for index in range(5):
                if records[index].get("opinions", []) != shared_opinions[index]:
                    records[index]["opinions"] = list(shared_opinions[index])
                    records[index]["shared_round1_extraction_source"] = "neutral"
                    changed = True
            if changed:
                path.write_text(
                    "".join(
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
    return changed


def opinions_and_rounds(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    rounds: list[int] = []
    for record in read_jsonl(path):
        round_number = int(record["round_number"])
        for opinion in record.get("opinions", []):
            if opinion.strip():
                texts.append(opinion)
                rounds.append(round_number)
    return texts, rounds


def run_embedding(force: bool) -> None:
    task_ids = load_selected_ids()
    for condition in CONDITIONS:
        print(f"Embedding {condition}...", flush=True)
        for index, task_id in enumerate(task_ids, start=1):
            full_path = (
                ROOT
                / "embeddings"
                / "openai-3-small"
                / condition
                / f"{task_id}.npy"
            )
            post_path = (
                ROOT
                / "embeddings_post"
                / "openai-3-small"
                / condition
                / f"{task_id}.npy"
            )
            if full_path.exists() and post_path.exists() and not force:
                continue
            texts, rounds = opinions_and_rounds(
                ROOT / "extracted" / condition / f"{task_id}.jsonl"
            )
            if len(texts) < 2:
                raise RuntimeError(f"{condition}/{task_id}: fewer than 2 opinions")
            embeddings = embed_texts(texts, embedder="openai-3-small")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            post_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(full_path, embeddings.astype(np.float32))
            post_indices = np.asarray(
                [i for i, round_number in enumerate(rounds) if round_number >= 2]
            )
            np.save(post_path, embeddings[post_indices].astype(np.float32))
            if index % 5 == 0:
                print(
                    f"  {condition}: {index}/{len(task_ids)} tasks",
                    flush=True,
                )


def corrected_rarefaction(
    embeddings: np.ndarray,
    n_minimum: int,
    seed: int,
) -> dict[str, float]:
    if len(embeddings) == n_minimum:
        return {
            "cc": float(cluster_count(embeddings)),
            "vs": float(vendi_score(embeddings)),
        }
    rng = np.random.default_rng(seed)
    cc_values: list[float] = []
    vs_values: list[float] = []
    for _ in range(N_RAREFACTION):
        indices = rng.choice(len(embeddings), size=n_minimum, replace=False)
        subset = embeddings[indices]
        cc_values.append(float(cluster_count(subset)))
        vs_values.append(float(vendi_score(subset)))
    return {"cc": float(np.mean(cc_values)), "vs": float(np.mean(vs_values))}


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def run_metrics() -> dict[str, Any]:
    task_ids = load_selected_ids()
    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conditions": list(CONDITIONS),
        "views": {},
        "rarefaction": {
            "repeats": N_RAREFACTION,
            "rule": (
                "Use the complete set when N equals per-task N_min; otherwise "
                "subsample without replacement."
            ),
        },
    }

    for view, directory_name in (
        ("all_rounds", "embeddings"),
        ("rounds_2_3", "embeddings_post"),
    ):
        arrays: dict[str, dict[str, np.ndarray]] = {
            condition: {} for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            for task_id in task_ids:
                arrays[condition][task_id] = np.load(
                    ROOT
                    / directory_name
                    / "openai-3-small"
                    / condition
                    / f"{task_id}.npy"
                )

        per_task: dict[str, dict[str, dict[str, float]]] = {
            condition: {} for condition in CONDITIONS
        }
        for task_id in task_ids:
            n_minimum = min(
                len(arrays[condition][task_id]) for condition in CONDITIONS
            )
            numeric_id = int(task_id.split("-")[1])
            for condition in CONDITIONS:
                embedding = arrays[condition][task_id]
                raw = compute_metrics(embedding)
                rarefied = corrected_rarefaction(
                    embedding,
                    n_minimum,
                    seed=SEED + numeric_id,
                )
                per_task[condition][task_id] = {
                    "n": int(raw["n"]),
                    "mpd": float(raw["mpd"]),
                    "cc_raw": float(raw["cc"]),
                    "vs_raw": float(raw["vs"]),
                    "n_min": n_minimum,
                    "cc_rarefied": rarefied["cc"],
                    "vs_rarefied": rarefied["vs"],
                }

        aggregate: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            aggregate[condition] = {}
            for metric in (
                "n",
                "mpd",
                "cc_raw",
                "vs_raw",
                "cc_rarefied",
                "vs_rarefied",
            ):
                values = np.asarray(
                    [per_task[condition][task_id][metric] for task_id in task_ids]
                )
                aggregate[condition][metric] = float(values.mean())
        output["views"][view] = {
            "aggregate": aggregate,
            "per_task": per_task,
        }

    write_json(ROOT / "metrics" / "metrics.json", output)
    write_json(ROOT / "metrics" / f"metrics_{len(task_ids)}.json", output)
    return output


def bootstrap_ci(values: np.ndarray, label: str) -> list[float]:
    digest = __import__("hashlib").sha256(f"{SEED}:{label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_analysis(metrics: dict[str, Any]) -> dict[str, Any]:
    task_ids = load_selected_ids()
    contrasts = (
        ("agreement_minus_neutral", "agreement", "neutral"),
        ("disagreement_minus_neutral", "disagreement", "neutral"),
        ("disagreement_minus_agreement", "disagreement", "agreement"),
    )
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(task_ids),
        "n_bootstrap": N_BOOTSTRAP,
        "contrasts": {},
    }
    for view in ("all_rounds", "rounds_2_3"):
        result["contrasts"][view] = {}
        per_task = metrics["views"][view]["per_task"]
        for label, left, right in contrasts:
            values = np.asarray(
                [
                    per_task[left][task_id]["mpd"]
                    - per_task[right][task_id]["mpd"]
                    for task_id in task_ids
                ],
                dtype=float,
            )
            p_value = (
                1.0 if np.allclose(values, 0) else float(wilcoxon(values).pvalue)
            )
            result["contrasts"][view][label] = {
                "left": left,
                "right": right,
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "bootstrap_95_ci": bootstrap_ci(values, f"{view}:{label}"),
                "wilcoxon_p": p_value,
                "left_wins": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "right_wins": int(np.sum(values < 0)),
                "per_task_difference": dict(zip(task_ids, values.tolist())),
            }
    write_json(ROOT / "analysis" / "paired_results.json", result)
    write_json(
        ROOT / "analysis" / f"paired_results_{len(task_ids)}.json",
        result,
    )
    return result


def make_report(metrics: dict[str, Any], paired: dict[str, Any]) -> None:
    generation = json.loads(
        (ROOT / "generation_summary.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (ROOT / "experiment_config.yaml").read_text(encoding="utf-8")
    )
    lines = [
        "# Multi-Agent Prompt Ablation Results",
        "",
        "## Design",
        "",
        f"- Model: `{config['model']['name']}` "
        f"(`thinking={config['extra_body']['thinking']['type']}`)",
        f"- {paired['n_tasks']} MODBench tasks",
        "- Original five Pro personas, three rounds",
        "- Identical shared round 1 across all three conditions",
        "- Only the round-2/3 interaction instruction differs",
        "- Primary metric: MPD with text-embedding-3-small",
        "",
        "## Generation completeness",
        "",
        "| Condition | Complete tasks | Records | Prompt tokens | Completion tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = generation["conditions"][condition]
        lines.append(
            f"| {condition} | {row['complete_tasks']} | {row['records']} | "
            f"{row['prompt_tokens']:,} | {row['completion_tokens']:,} |"
        )

    for view, title in (
        ("all_rounds", "All-round MPD (paper-comparable set definition)"),
        ("rounds_2_3", "Rounds 2–3 MPD (post-intervention diagnostic)"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Condition | Mean opinions | Mean MPD | Corrected rarefied CC | Corrected rarefied VS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        aggregate = metrics["views"][view]["aggregate"]
        for condition in CONDITIONS:
            row = aggregate[condition]
            lines.append(
                f"| {condition} | {row['n']:.1f} | {row['mpd']:.4f} | "
                f"{row['cc_rarefied']:.2f} | {row['vs_rarefied']:.2f} |"
            )
        lines.extend(
            [
                "",
                "| Contrast (left − right) | Mean ΔMPD | 95% task-bootstrap CI | Wins / Ties / Losses | Wilcoxon p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, row in paired["contrasts"][view].items():
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {row['left']} − {row['right']} | "
                f"{row['mean_difference']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{row['left_wins']} / {row['ties']} / {row['right_wins']} | "
                f"{row['wilcoxon_p']:.4g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            f"This single-model, {paired['n_tasks']}-task ablation tests sensitivity to interaction "
            "instructions. It does not validate MPD against human-perceived "
            "opinion diversity and does not establish a model-general effect.",
            "",
        ]
    )
    report_path = ROOT / "analysis" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "analysis" / f"report_{paired['n_tasks']}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(SOURCE_REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not configured for embeddings.")

    await run_extraction(force=args.force)
    shared_extraction_changed = synchronize_shared_round1_extractions()
    if shared_extraction_changed:
        print(
            "Synchronized shared round-1 extractions across conditions.",
            flush=True,
        )
    run_embedding(force=args.force or shared_extraction_changed)
    metrics = run_metrics()
    paired = paired_analysis(metrics)
    make_report(metrics, paired)
    print(f"Complete: {ROOT / 'analysis' / 'report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run extraction and embedding even when outputs exist.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
