#!/usr/bin/env python3
"""Offline check of the Multi-Agent prompt-sensitivity numbers (Appendix L).

The full pipeline (run_pipeline.py, run_coverage_pipeline.py,
analyze_four_conditions.py) needs the raw responses, extracted opinions, and
embeddings, which are not part of the release. This script instead starts
from the released per-task metric files and recomputes:

1. the paper table "Multi-Agent prompt sensitivity" (Rounds 2-3 means of
   opinions, MPD, and jointly rarefied CC/VS for the four instructions) from
   analysis/four_condition_alpha_100.json;
2. the paired MPD contrasts and task-bootstrap CIs quoted in the text
   (Agreement/Disagreement vs Neutral) from metrics/metrics_100.json, using
   the same bootstrap routine and seeds as run_pipeline.py;
3. the Coverage-minus-Neutral contrasts from metrics/coverage_metrics_100.json,
   using the same routine and seeds as run_coverage_pipeline.py.

Each recomputed value is compared with the stored result file. No API calls
are made and nothing is written. Requires `pip install -e .` from the
repository root (for the `modbench` package imported by the pipeline scripts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_coverage_pipeline as coverage_core  # noqa: E402
import run_pipeline as core  # noqa: E402

TOL = 1e-9
failures: list[str] = []


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def check(label: str, computed: float, stored: float) -> None:
    if not np.isclose(computed, stored, rtol=0, atol=TOL):
        failures.append(f"{label}: computed {computed!r} != stored {stored!r}")


def table_four_conditions() -> None:
    data = load("analysis/four_condition_alpha_100.json")
    view = data["views"]["rounds_2_3"]
    print("Table: Multi-Agent prompt sensitivity (Rounds 2-3)")
    print(f"{'Instruction':<14}{'Opinions':>10}{'MPD':>9}{'CC(rar.)':>10}{'VS(rar.)':>10}")
    for condition in data["conditions"]:
        per_task = view["per_task"][condition]
        row = {
            metric: float(np.mean([task[metric] for task in per_task.values()]))
            for metric in ("n", "mpd", "cc_rarefied", "vs_rarefied")
        }
        for metric, value in row.items():
            check(f"four_condition/{condition}/{metric}", value, view["aggregate"][condition][metric])
        print(
            f"{condition:<14}{row['n']:>10.1f}{row['mpd']:>9.4f}"
            f"{row['cc_rarefied']:>10.2f}{row['vs_rarefied']:>10.2f}"
        )


def paired_contrasts() -> None:
    metrics = load("metrics/metrics_100.json")
    stored = load("analysis/paired_results_100.json")
    contrasts = (
        ("agreement_minus_neutral", "agreement", "neutral"),
        ("disagreement_minus_neutral", "disagreement", "neutral"),
        ("disagreement_minus_agreement", "disagreement", "agreement"),
    )
    print("\nPaired MPD contrasts (left - right), 95% task-bootstrap CI")
    for view in ("rounds_2_3", "all_rounds"):
        per_task = metrics["views"][view]["per_task"]
        task_ids = sorted(per_task["neutral"])
        for label, left, right in contrasts:
            values = np.asarray(
                [per_task[left][t]["mpd"] - per_task[right][t]["mpd"] for t in task_ids]
            )
            ci = core.bootstrap_ci(values, f"{view}:{label}")
            p_value = float(wilcoxon(values).pvalue)
            ref = stored["contrasts"][view][label]
            check(f"{view}/{label}/mean", float(values.mean()), ref["mean_difference"])
            check(f"{view}/{label}/ci_low", ci[0], ref["bootstrap_95_ci"][0])
            check(f"{view}/{label}/ci_high", ci[1], ref["bootstrap_95_ci"][1])
            check(f"{view}/{label}/wilcoxon_p", p_value, ref["wilcoxon_p"])
            print(
                f"  {view:<11} {left:>12} - {right:<9} {values.mean():+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p_value:.3g}"
            )


def coverage_contrasts() -> None:
    metrics = load("metrics/coverage_metrics_100.json")
    stored = load("analysis/coverage_paired_results_100.json")
    print("\nCoverage - Neutral, 95% task-bootstrap CI")
    for view in ("rounds_2_3", "all_rounds"):
        per_task = metrics["views"][view]["per_task"]
        task_ids = sorted(per_task["neutral"])
        for metric in ("mpd", "cc_rarefied", "vs_rarefied", "n"):
            values = np.asarray(
                [per_task["coverage"][t][metric] - per_task["neutral"][t][metric] for t in task_ids],
                dtype=float,
            )
            ci = coverage_core.bootstrap_ci(values, f"{view}:{metric}")
            ref = stored["views"][view][metric]
            check(f"coverage/{view}/{metric}/mean", float(values.mean()), ref["mean_difference"])
            check(f"coverage/{view}/{metric}/ci_low", ci[0], ref["bootstrap_95_ci"][0])
            check(f"coverage/{view}/{metric}/ci_high", ci[1], ref["bootstrap_95_ci"][1])
            print(f"  {view:<11} {metric:<12} {values.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")


def main() -> int:
    table_four_conditions()
    paired_contrasts()
    coverage_contrasts()
    if failures:
        print("\nMISMATCHES:")
        for line in failures:
            print("  " + line)
        return 1
    print("\nAll recomputed values match the stored result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
