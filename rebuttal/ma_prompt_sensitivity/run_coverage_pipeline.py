#!/usr/bin/env python3
"""Complete extraction, embedding, alpha/beta metrics, inference, and audit.

This pipeline compares the new coverage condition only against the already
completed neutral control. Existing neutral/agreement/disagreement artifacts are
read but never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.stats import wilcoxon

import analyze_alpha_beta as beta_core
import run_pipeline as metric_core
from modbench.embedding import embed_texts
from modbench.extractor import extract_condition
from modbench.metrics import compute_metrics


ROOT = Path(__file__).resolve().parent
# Repository root (holds .env); override with the MODBENCH_ROOT environment variable.
SOURCE_REPO = Path(os.environ.get("MODBENCH_ROOT", ROOT.parents[1])).resolve()
CONDITIONS = ("neutral", "coverage")
VIEWS = {
    "all_rounds": ROOT / "embeddings" / "openai-3-small",
    "rounds_2_3": ROOT / "embeddings_post" / "openai-3-small",
}
N_BOOTSTRAP = 20_000
BASE_SEED = 1529


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_task_ids() -> list[str]:
    path = ROOT / "data" / "selected_tasks_100.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        task_ids = [row["task_id"] for row in csv.DictReader(handle)]
    if len(task_ids) != 100:
        raise RuntimeError(f"Expected 100 frozen tasks; found {len(task_ids)}")
    return task_ids


async def run_extraction(force: bool) -> None:
    print("Extracting coverage...", flush=True)
    await extract_condition(
        config_path=ROOT / "extraction_config.yaml",
        raw_dir=ROOT / "raw" / "coverage",
        extracted_dir=ROOT / "extracted" / "coverage",
        max_concurrent=30,
        force=force,
    )


def synchronize_shared_round1_extractions(task_ids: list[str]) -> bool:
    """Copy the single neutral extraction of identical round-1 responses."""
    changed_any = False
    for task_id in task_ids:
        neutral = read_jsonl(
            ROOT / "extracted" / "neutral" / f"{task_id}.jsonl"
        )
        coverage_path = (
            ROOT / "extracted" / "coverage" / f"{task_id}.jsonl"
        )
        coverage = read_jsonl(coverage_path)
        if len(neutral) != 15 or len(coverage) != 15:
            raise RuntimeError(f"{task_id}: malformed extraction file")
        if any(int(row["round_number"]) != 1 for row in neutral[:5]):
            raise RuntimeError(f"{task_id}: malformed neutral round 1")
        if any(int(row["round_number"]) != 1 for row in coverage[:5]):
            raise RuntimeError(f"{task_id}: malformed coverage round 1")

        changed = False
        for index in range(5):
            shared = list(neutral[index].get("opinions", []))
            if coverage[index].get("opinions", []) != shared:
                coverage[index]["opinions"] = shared
                coverage[index][
                    "shared_round1_extraction_source"
                ] = "neutral"
                changed = True
        if changed:
            write_jsonl(coverage_path, coverage)
            changed_any = True
    return changed_any


def opinions_and_rounds(path: Path) -> tuple[list[str], list[int]]:
    opinions: list[str] = []
    rounds: list[int] = []
    for record in read_jsonl(path):
        round_number = int(record["round_number"])
        for opinion in record.get("opinions", []):
            if opinion.strip():
                opinions.append(opinion)
                rounds.append(round_number)
    return opinions, rounds


def run_embedding(task_ids: list[str], force: bool) -> None:
    print("Embedding coverage with text-embedding-3-small...", flush=True)
    for index, task_id in enumerate(task_ids, start=1):
        full_path = (
            ROOT
            / "embeddings"
            / "openai-3-small"
            / "coverage"
            / f"{task_id}.npy"
        )
        post_path = (
            ROOT
            / "embeddings_post"
            / "openai-3-small"
            / "coverage"
            / f"{task_id}.npy"
        )
        if full_path.exists() and post_path.exists() and not force:
            continue
        opinions, rounds = opinions_and_rounds(
            ROOT / "extracted" / "coverage" / f"{task_id}.jsonl"
        )
        if len(opinions) < 2:
            raise RuntimeError(f"coverage/{task_id}: fewer than 2 opinions")
        embeddings = embed_texts(opinions, embedder="openai-3-small")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        post_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, embeddings.astype(np.float32))
        post_indices = np.asarray(
            [
                position
                for position, round_number in enumerate(rounds)
                if round_number >= 2
            ]
        )
        np.save(post_path, embeddings[post_indices].astype(np.float32))
        if index % 5 == 0:
            print(f"  coverage: {index}/{len(task_ids)} tasks", flush=True)


def run_alpha_metrics(task_ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conditions": list(CONDITIONS),
        "n_tasks": len(task_ids),
        "embedding_model": "text-embedding-3-small",
        "rarefaction": {
            "repeats": metric_core.N_RAREFACTION,
            "rule": "Per task, rarefy both conditions to their shared minimum N.",
        },
        "views": {},
    }
    for view, root in VIEWS.items():
        arrays = {
            condition: {
                task_id: np.load(root / condition / f"{task_id}.npy")
                for task_id in task_ids
            }
            for condition in CONDITIONS
        }
        per_task: dict[str, dict[str, dict[str, float]]] = {
            condition: {} for condition in CONDITIONS
        }
        for task_id in task_ids:
            n_min = min(
                len(arrays[condition][task_id]) for condition in CONDITIONS
            )
            numeric_id = int(task_id.split("-")[1])
            for condition in CONDITIONS:
                embedding = arrays[condition][task_id]
                raw = compute_metrics(embedding)
                rarefied = metric_core.corrected_rarefaction(
                    embedding,
                    n_min,
                    seed=BASE_SEED + numeric_id,
                )
                per_task[condition][task_id] = {
                    "n": int(raw["n"]),
                    "mpd": float(raw["mpd"]),
                    "cc_raw": float(raw["cc"]),
                    "vs_raw": float(raw["vs"]),
                    "n_min": int(n_min),
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
                    [
                        per_task[condition][task_id][metric]
                        for task_id in task_ids
                    ],
                    dtype=float,
                )
                aggregate[condition][metric] = float(values.mean())
        result["views"][view] = {
            "aggregate": aggregate,
            "per_task": per_task,
        }

    write_json(ROOT / "metrics" / "coverage_metrics_100.json", result)
    return result


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ci(values: np.ndarray, label: str) -> list[float]:
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(
        0, len(values), size=(N_BOOTSTRAP, len(values))
    )
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_alpha_analysis(
    metrics: dict[str, Any], task_ids: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(task_ids),
        "n_bootstrap": N_BOOTSTRAP,
        "contrast": "coverage_minus_neutral",
        "views": {},
    }
    for view in VIEWS:
        per_task = metrics["views"][view]["per_task"]
        result["views"][view] = {}
        for metric in ("mpd", "cc_rarefied", "vs_rarefied", "n"):
            differences = np.asarray(
                [
                    per_task["coverage"][task_id][metric]
                    - per_task["neutral"][task_id][metric]
                    for task_id in task_ids
                ],
                dtype=float,
            )
            p_value = (
                1.0
                if np.allclose(differences, 0)
                else float(wilcoxon(differences).pvalue)
            )
            result["views"][view][metric] = {
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "bootstrap_95_ci": bootstrap_ci(
                    differences, f"{view}:{metric}"
                ),
                "wilcoxon_p": p_value,
                "coverage_wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "neutral_wins": int(np.sum(differences < 0)),
                "per_task_difference": dict(
                    zip(task_ids, differences.tolist())
                ),
            }
    write_json(
        ROOT / "analysis" / "coverage_paired_results_100.json", result
    )
    return result


def run_beta_analysis(
    metrics: dict[str, Any], task_ids: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(task_ids),
        "n_repeats": beta_core.N_REPEATS,
        "embedding_model": "text-embedding-3-small",
        "alpha": {
            view: metrics["views"][view]["aggregate"] for view in VIEWS
        },
        "beta": {},
    }
    for view, root in VIEWS.items():
        print(f"Beta analysis: {view}, neutral vs coverage", flush=True)
        per_task: dict[str, dict[str, float]] = {}
        for index, task_id in enumerate(task_ids, start=1):
            neutral = np.load(root / "neutral" / f"{task_id}.npy")
            coverage = np.load(root / "coverage" / f"{task_id}.npy")
            per_task[task_id] = beta_core.task_beta(
                neutral,
                coverage,
                seed=beta_core.stable_seed(
                    view, "neutral_vs_coverage", task_id
                ),
            )
            if index % 10 == 0:
                print(f"  {index}/{len(task_ids)}", flush=True)
        result["beta"][view] = {
            "condition_a": "neutral",
            "condition_b": "coverage",
            "aggregate": beta_core.summarize(
                per_task, f"{view}:neutral_vs_coverage"
            ),
            "per_task": per_task,
        }
    write_json(ROOT / "analysis" / "coverage_alpha_beta_100.json", result)
    return result


def audit(
    task_ids: list[str],
    metrics: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    coverage_config = __import__("yaml").safe_load(
        (ROOT / "coverage_config.yaml").read_text(encoding="utf-8")
    )
    expected_prompts = coverage_config["prompts"]["coverage"]
    errors: list[str] = []
    raw_records = 0
    post_records = 0
    prompt_tokens = 0
    completion_tokens = 0
    opinion_counts = {"all_rounds": 0, "rounds_2_3": 0}

    for task_id in task_ids:
        raw = read_jsonl(ROOT / "raw" / "coverage" / f"{task_id}.jsonl")
        neutral = read_jsonl(
            ROOT / "raw" / "neutral" / f"{task_id}.jsonl"
        )
        if len(raw) != 15:
            errors.append(f"{task_id}: coverage raw rows={len(raw)}")
            continue
        if len(neutral) != 15:
            errors.append(f"{task_id}: neutral raw rows={len(neutral)}")
            continue
        raw_records += len(raw)
        rounds = [int(record["round_number"]) for record in raw]
        if [rounds.count(number) for number in (1, 2, 3)] != [5, 5, 5]:
            errors.append(f"{task_id}: malformed round counts")
        for index in range(5):
            if raw[index]["request"] != neutral[index]["request"]:
                errors.append(f"{task_id}: shared R1 request mismatch at {index}")
            if raw[index]["response"] != neutral[index]["response"]:
                errors.append(f"{task_id}: shared R1 response mismatch at {index}")
        for record in raw:
            if record.get("prompt_condition") != "coverage":
                errors.append(f"{task_id}: wrong prompt condition")
            if record["request"].get("model") != "deepseek-v4-flash":
                errors.append(f"{task_id}: wrong generation model")
            if record["response"].get("finish_reason") != "stop":
                errors.append(
                    f"{task_id}: finish={record['response'].get('finish_reason')}"
                )
            if not record["response"].get("content"):
                errors.append(f"{task_id}: empty response")
            if int(record["round_number"]) >= 2:
                post_records += 1
                prompt_tokens += record["response"]["usage"]["prompt_tokens"]
                completion_tokens += record["response"]["usage"][
                    "completion_tokens"
                ]
                key = (
                    "round2"
                    if int(record["round_number"]) == 2
                    else "round3"
                )
                user_prompt = record["request"]["messages"][-1]["content"]
                if not user_prompt.endswith(expected_prompts[key]):
                    errors.append(
                        f"{task_id}: {key} instruction mismatch"
                    )

        extracted = read_jsonl(
            ROOT / "extracted" / "coverage" / f"{task_id}.jsonl"
        )
        if len(extracted) != 15:
            errors.append(f"{task_id}: extracted rows={len(extracted)}")
            continue
        if any("error" in record for record in extracted):
            errors.append(f"{task_id}: extraction error present")
        for record in extracted:
            count = sum(
                1
                for opinion in record.get("opinions", [])
                if opinion.strip()
            )
            opinion_counts["all_rounds"] += count
            if int(record["round_number"]) >= 2:
                opinion_counts["rounds_2_3"] += count

        for view, root in VIEWS.items():
            array = np.load(root / "coverage" / f"{task_id}.npy")
            if array.ndim != 2 or array.shape[1] != 1536:
                errors.append(
                    f"{task_id}: {view} embedding shape={array.shape}"
                )
            expected_rows = sum(
                1
                for record in extracted
                if view == "all_rounds"
                or int(record["round_number"]) >= 2
                for opinion in record.get("opinions", [])
                if opinion.strip()
            )
            if len(array) != expected_rows:
                errors.append(
                    f"{task_id}: {view} rows={len(array)}, "
                    f"opinions={expected_rows}"
                )

    mpd_post = paired["views"]["rounds_2_3"]["mpd"]
    secondary = metrics["views"]["rounds_2_3"]["aggregate"]
    primary_pass = mpd_post["bootstrap_95_ci"][0] > 0
    secondary_pass = (
        secondary["coverage"]["cc_rarefied"]
        > secondary["neutral"]["cc_rarefied"]
        or secondary["coverage"]["vs_rarefied"]
        > secondary["neutral"]["vs_rarefied"]
    )
    audit_result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "n_tasks": len(task_ids),
        "raw_records": raw_records,
        "unique_new_generation_calls": post_records,
        "generation_model": "deepseek-v4-flash",
        "finish_reason_stop_required": True,
        "prompt_tokens_rounds_2_3": prompt_tokens,
        "completion_tokens_rounds_2_3": completion_tokens,
        "estimated_generation_cost_usd": (
            prompt_tokens * 0.14 / 1_000_000
            + completion_tokens * 0.28 / 1_000_000
        ),
        "coverage_opinion_counts": opinion_counts,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "preregistered_success": {
            "primary": (
                "Rounds-2-3 coverage-minus-neutral MPD task-bootstrap "
                "95% CI is entirely positive."
            ),
            "secondary": (
                "Rounds-2-3 rarefied CC or VS is directionally higher."
            ),
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "overall_pass": primary_pass and secondary_pass,
        },
    }
    write_json(ROOT / "analysis" / "coverage_audit_100.json", audit_result)
    return audit_result


def make_report(
    metrics: dict[str, Any],
    paired: dict[str, Any],
    beta: dict[str, Any],
    audit_result: dict[str, Any],
) -> None:
    lines = [
        "# Diversity-Coverage Prompt: 100-Task Results",
        "",
        "## Frozen comparison",
        "",
        "- Control: submitted neutral Multi-Agent interaction instruction.",
        "- Intervention: a repeated round-2/3 instruction to broaden genuine "
        "viewpoint coverage.",
        "- Shared round 1, model, questions, five Pro personas, agent order, "
        "rounds, generation budget, temperature, extractor, and embedder are fixed.",
        "- Primary analysis: task-paired rounds-2/3 MPD.",
        "- Secondary analysis: correctly rarefied Cluster Count and Vendi Score.",
        "",
    ]
    for view, title in (
        ("all_rounds", "All rounds"),
        ("rounds_2_3", "Rounds 2–3 (intervention effect)"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Condition | Mean opinions | MPD | Rarefied CC | Rarefied VS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            row = metrics["views"][view]["aggregate"][condition]
            lines.append(
                f"| {condition} | {row['n']:.1f} | {row['mpd']:.4f} | "
                f"{row['cc_rarefied']:.2f} | {row['vs_rarefied']:.2f} |"
            )
        lines.extend(
            [
                "",
                "| Metric, coverage − neutral | Mean Δ | 95% task-bootstrap CI | "
                "Wins / Ties / Losses | Wilcoxon p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ("mpd", "cc_rarefied", "vs_rarefied", "n"):
            row = paired["views"][view][metric]
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {row['mean_difference']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{row['coverage_wins']} / {row['ties']} / "
                f"{row['neutral_wins']} | {row['wilcoxon_p']:.4g} |"
            )
        beta_row = beta["beta"][view]["aggregate"]
        lines.extend(
            [
                "",
                "Neutral–coverage beta separation:",
                "",
                f"- β-VS: {beta_row['beta_vs']['mean']:.3f}",
                f"- β-VS above shuffled labels: "
                f"{beta_row['beta_vs_above_shuffle']['mean']:+.3f}",
                f"- Neutral unique-cluster ratio: "
                f"{beta_row['unique_a_ratio']['mean']:.1%}",
                f"- Coverage unique-cluster ratio: "
                f"{beta_row['unique_b_ratio']['mean']:.1%}",
                f"- Mean UCR above shuffled labels: "
                f"{beta_row['unique_ratio_above_shuffle']['mean']:+.1%}",
                "",
            ]
        )

    success = audit_result["preregistered_success"]
    lines.extend(
        [
            "## Frozen success criterion",
            "",
            f"- Primary MPD criterion: **{'PASS' if success['primary_pass'] else 'FAIL'}**",
            f"- Secondary direction criterion: "
            f"**{'PASS' if success['secondary_pass'] else 'FAIL'}**",
            f"- Overall: **{'PASS' if success['overall_pass'] else 'FAIL'}**",
            "",
            "## Audit and interpretation guardrail",
            "",
            f"- Artifact audit: **{audit_result['status']}** "
            f"({len(audit_result['errors'])} errors)",
            f"- New generation calls: "
            f"{audit_result['unique_new_generation_calls']:,}",
            f"- Estimated new generation cost: "
            f"${audit_result['estimated_generation_cost_usd']:.3f}",
            "- Opinion count is descriptive and cannot by itself establish diversity.",
            "- Beta diversity measures separation between conditions, not which "
            "condition is more internally diverse.",
            "- This single-model prompt optimization does not validate MPD against "
            "human-perceived diversity and does not establish a model-general effect.",
            "",
        ]
    )
    path = ROOT / "analysis" / "coverage_report_100.md"
    path.write_text("\n".join(lines), encoding="utf-8")


async def main_async(force: bool) -> None:
    load_dotenv(SOURCE_REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    task_ids = load_task_ids()
    await run_extraction(force=force)
    changed = synchronize_shared_round1_extractions(task_ids)
    if changed:
        print("Synchronized coverage round-1 extraction with neutral.", flush=True)
    run_embedding(task_ids, force=force or changed)
    metrics = run_alpha_metrics(task_ids)
    paired = paired_alpha_analysis(metrics, task_ids)
    beta = run_beta_analysis(metrics, task_ids)
    audit_result = audit(task_ids, metrics, paired)
    make_report(metrics, paired, beta, audit_result)
    print(
        f"Complete: {ROOT / 'analysis' / 'coverage_report_100.md'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run coverage extraction and embeddings.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.force))


if __name__ == "__main__":
    main()
