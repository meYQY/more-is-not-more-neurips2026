#!/usr/bin/env python3
"""Unified alpha-diversity table for the four matched prompt conditions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_pipeline as core
from modbench.metrics import compute_metrics


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("neutral", "agreement", "disagreement", "coverage")
VIEWS = {
    "all_rounds": ROOT / "embeddings" / "openai-3-small",
    "rounds_2_3": ROOT / "embeddings_post" / "openai-3-small",
}
SEED = 1529


def load_task_ids() -> list[str]:
    with (ROOT / "data" / "selected_tasks_100.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return [row["task_id"] for row in csv.DictReader(handle)]


def main() -> None:
    task_ids = load_task_ids()
    result = {
        "n_tasks": len(task_ids),
        "conditions": list(CONDITIONS),
        "rarefaction_repeats": core.N_RAREFACTION,
        "rarefaction_rule": (
            "For each task and view, rarefy every condition to the minimum "
            "opinion count across all four conditions, without replacement."
        ),
        "views": {},
    }
    for view, embedding_root in VIEWS.items():
        per_task = {condition: {} for condition in CONDITIONS}
        n_minimums = []
        for task_id in task_ids:
            arrays = {
                condition: np.load(
                    embedding_root / condition / f"{task_id}.npy"
                )
                for condition in CONDITIONS
            }
            n_minimum = min(len(array) for array in arrays.values())
            n_minimums.append(n_minimum)
            numeric_id = int(task_id.split("-")[1])
            for condition, embeddings in arrays.items():
                raw = compute_metrics(embeddings)
                rarefied = core.corrected_rarefaction(
                    embeddings,
                    n_minimum,
                    seed=SEED + numeric_id,
                )
                per_task[condition][task_id] = {
                    "n": int(raw["n"]),
                    "mpd": float(raw["mpd"]),
                    "cc_rarefied": rarefied["cc"],
                    "vs_rarefied": rarefied["vs"],
                }
        aggregate = {}
        for condition in CONDITIONS:
            aggregate[condition] = {
                metric: float(
                    np.mean(
                        [
                            per_task[condition][task_id][metric]
                            for task_id in task_ids
                        ]
                    )
                )
                for metric in ("n", "mpd", "cc_rarefied", "vs_rarefied")
            }
        result["views"][view] = {
            "mean_rarefied_n": float(np.mean(n_minimums)),
            "aggregate": aggregate,
            "per_task": per_task,
        }

    output = ROOT / "analysis" / "four_condition_alpha_100.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Unified Four-Condition Alpha Diversity",
        "",
        "All conditions use the same 100 tasks, model, five Pro personas, shared",
        "round 1, extractor, and embedding model. CC and VS are rarefied to the",
        "same per-task minimum N across all four conditions.",
        "",
    ]
    for view, title in (
        ("all_rounds", "All rounds"),
        ("rounds_2_3", "Rounds 2–3"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                f"Mean per-task rarefied N: "
                f"{result['views'][view]['mean_rarefied_n']:.1f}",
                "",
                "| Condition | Mean opinions | MPD | Rarefied CC | Rarefied VS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            row = result["views"][view]["aggregate"][condition]
            lines.append(
                f"| {condition} | {row['n']:.1f} | {row['mpd']:.4f} | "
                f"{row['cc_rarefied']:.2f} | {row['vs_rarefied']:.2f} |"
            )
        lines.append("")
    report = ROOT / "analysis" / "four_condition_alpha_report_100.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
