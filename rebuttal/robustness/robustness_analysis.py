#!/usr/bin/env python3
"""Rebuttal-oriented robustness analyses over released per-task MPD results.

This script does not modify the paper repository. It checks:
1. Bootstrap uncertainty for the paper's core persona-depth contrasts.
2. Stability of conclusions under random task subsets.
3. Leave-one-subcategory-out robustness.
4. Formal paired difference-in-differences tests for persona × architecture.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


# Repository root with data/dataset/tasks.csv and results/metrics*/ (override: MODBENCH_ROOT).
REPO = Path(os.environ.get("MODBENCH_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent
SEED = 20260724
N_BOOT = 20_000
N_SUBSET = 10_000

MODEL_DIRS = {
    "DeepSeek": "metrics",
    "GPT-mini": "metrics_gpt54mini",
    "Gemini": "metrics_gemini",
    "Kimi": "metrics_kimi",
    "Llama": "metrics_llama",
    "Grok": "metrics_grok",
    "Qwen": "metrics_qwen",
}


def load_task_categories() -> dict[str, str]:
    with (REPO / "data/dataset/tasks.csv").open(newline="", encoding="utf-8") as f:
        return {row["task_id"]: row["subcategory"] for row in csv.DictReader(f)}


def load_metric(model_dir: str, condition: str, metric: str = "mpd") -> dict[str, float]:
    path = REPO / "results" / model_dir / "openai-3-small" / f"{condition}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for task_id, values in data.get("per_task", {}).items():
        value = float(values.get(metric, float("nan")))
        if math.isfinite(value):
            out[task_id] = value
    return out


def aligned_difference(
    model_dir: str,
    condition_a: str,
    condition_b: str,
) -> tuple[list[str], np.ndarray]:
    """Return B-A, aligned by task."""
    a = load_metric(model_dir, condition_a)
    b = load_metric(model_dir, condition_b)
    common = sorted(set(a) & set(b))
    return common, np.asarray([b[t] - a[t] for t in common], dtype=float)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def random_subset_positive_rate(
    values: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> float:
    if len(values) < size:
        return float("nan")
    if len(values) == size:
        return float(values.mean() > 0)
    positives = 0
    for _ in range(N_SUBSET):
        idx = rng.choice(len(values), size=size, replace=False)
        positives += int(values[idx].mean() > 0)
    return positives / N_SUBSET


def leave_one_category_out(
    task_ids: list[str],
    values: np.ndarray,
    categories: dict[str, str],
) -> dict:
    unique_categories = sorted({categories[t] for t in task_ids})
    estimates = {}
    for category in unique_categories:
        keep = np.asarray([categories[t] != category for t in task_ids])
        estimates[category] = float(values[keep].mean())
    vals = list(estimates.values())
    return {
        "all_positive": all(v > 0 for v in vals),
        "min": min(vals),
        "max": max(vals),
        "by_held_out_category": estimates,
    }


def paired_did(
    model_dir: str,
    architecture: str,
) -> tuple[list[str], np.ndarray]:
    """(Arch Pro-None) - (Independent Pro-None), paired by task."""
    ind_none = load_metric(model_dir, "independent-none")
    ind_pro = load_metric(model_dir, "independent-pro")
    arch_none = load_metric(model_dir, f"{architecture}-none")
    arch_pro = load_metric(model_dir, f"{architecture}-pro")
    common = sorted(set(ind_none) & set(ind_pro) & set(arch_none) & set(arch_pro))
    values = np.asarray(
        [
            (arch_pro[t] - arch_none[t]) - (ind_pro[t] - ind_none[t])
            for t in common
        ],
        dtype=float,
    )
    return common, values


def safe_wilcoxon(values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = len(p)
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return [float(x) for x in out]


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    categories = load_task_categories()
    rng = np.random.default_rng(SEED)
    report: dict = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_random_subsets": N_SUBSET,
        "models": {},
    }

    for model_name, model_dir in MODEL_DIRS.items():
        model_result: dict = {}

        task_ids, initial = aligned_difference(
            model_dir, "independent-none", "independent-role"
        )
        task_ids_extra, extra = aligned_difference(
            model_dir, "independent-role", "independent-pro"
        )
        common = sorted(set(task_ids) & set(task_ids_extra))
        initial_map = dict(zip(task_ids, initial))
        extra_map = dict(zip(task_ids_extra, extra))
        diminishing = np.asarray(
            [initial_map[t] - extra_map[t] for t in common], dtype=float
        )
        initial_common = np.asarray([initial_map[t] for t in common], dtype=float)
        extra_common = np.asarray([extra_map[t] for t in common], dtype=float)

        initial_ci = bootstrap_mean_ci(initial_common, rng)
        extra_ci = bootstrap_mean_ci(extra_common, rng)
        diminishing_ci = bootstrap_mean_ci(diminishing, rng)
        model_result["persona"] = {
            "n_tasks": len(common),
            "none_to_role_mean": float(initial_common.mean()),
            "none_to_role_ci95": initial_ci,
            "role_to_pro_mean": float(extra_common.mean()),
            "role_to_pro_ci95": extra_ci,
            "initial_minus_extra_mean": float(diminishing.mean()),
            "initial_minus_extra_ci95": diminishing_ci,
            "initial_gain_subset_positive_rate": {
                str(n): random_subset_positive_rate(initial_common, n, rng)
                for n in (25, 50, 75, 100)
                if len(initial_common) >= n
            },
            "initial_gt_extra_subset_rate": {
                str(n): random_subset_positive_rate(diminishing, n, rng)
                for n in (25, 50, 75, 100)
                if len(diminishing) >= n
            },
            "none_to_role_leave_one_category_out": leave_one_category_out(
                common, initial_common, categories
            ),
            "initial_gt_extra_leave_one_category_out": leave_one_category_out(
                common, diminishing, categories
            ),
        }

        model_result["low_cost"] = {}
        for condition in (
            "independent-enhanced",
            "independent-temp-high",
            "independent-dimension",
            "independent-random-sys",
        ):
            ids, delta = aligned_difference(model_dir, "independent-none", condition)
            if len(delta):
                model_result["low_cost"][condition] = {
                    "n_tasks": len(ids),
                    "mean_delta": float(delta.mean()),
                    "ci95": bootstrap_mean_ci(delta, rng),
                }

        model_result["matched_architecture"] = {}
        for condition in ("sequential-pro-5", "interactive-pro"):
            ids, delta = aligned_difference(model_dir, "independent-pro-5", condition)
            if len(delta):
                model_result["matched_architecture"][condition] = {
                    "n_tasks": len(ids),
                    "mean_delta": float(delta.mean()),
                    "ci95": bootstrap_mean_ci(delta, rng),
                    "subset_positive_rate": {
                        str(n): random_subset_positive_rate(delta, n, rng)
                        for n in (25, 50, 75, 100)
                        if len(delta) >= n
                    },
                }

        model_result["did"] = {}
        for architecture in ("sequential", "interactive"):
            ids, did = paired_did(model_dir, architecture)
            if len(did):
                model_result["did"][architecture] = {
                    "n_tasks": len(ids),
                    "mean": float(did.mean()),
                    "ci95": bootstrap_mean_ci(did, rng),
                    "wilcoxon_p_raw": safe_wilcoxon(did),
                }

        report["models"][model_name] = model_result

    did_refs = []
    did_p = []
    for model_name, model_result in report["models"].items():
        for architecture, result in model_result.get("did", {}).items():
            did_refs.append((model_name, architecture))
            did_p.append(result["wilcoxon_p_raw"])
    for (model_name, architecture), p_adj in zip(did_refs, bh_adjust(did_p)):
        report["models"][model_name]["did"][architecture]["wilcoxon_p_bh"] = p_adj

    (OUT / "robustness_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    lines = [
        "# Rebuttal robustness analysis",
        "",
        "All confidence intervals are nonparametric task-bootstrap percentile intervals.",
        "",
        "## Persona-depth stability",
        "",
        "| Model | N | None→Role ΔMPD [95% CI] | Role→Pro ΔMPD [95% CI] | Initial−extra [95% CI] | P(initial>0), n=50 | P(initial>extra), n=50 | LOO-category directions |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, model_result in report["models"].items():
        p = model_result["persona"]
        initial_ci = p["none_to_role_ci95"]
        extra_ci = p["role_to_pro_ci95"]
        dim_ci = p["initial_minus_extra_ci95"]
        subset_initial = p["initial_gain_subset_positive_rate"].get("50", float("nan"))
        subset_dim = p["initial_gt_extra_subset_rate"].get("50", float("nan"))
        loo_1 = p["none_to_role_leave_one_category_out"]["all_positive"]
        loo_2 = p["initial_gt_extra_leave_one_category_out"]["all_positive"]
        lines.append(
            f"| {model_name} | {p['n_tasks']} | "
            f"{fmt(p['none_to_role_mean'])} [{fmt(initial_ci[0])}, {fmt(initial_ci[1])}] | "
            f"{fmt(p['role_to_pro_mean'])} [{fmt(extra_ci[0])}, {fmt(extra_ci[1])}] | "
            f"{fmt(p['initial_minus_extra_mean'])} [{fmt(dim_ci[0])}, {fmt(dim_ci[1])}] | "
            f"{subset_initial:.1%} | {subset_dim:.1%} | initial>0: {loo_1}; initial>extra: {loo_2} |"
        )

    lines.extend(
        [
            "",
            "## Matched architecture contrasts at Pro",
            "",
            "Each contrast uses the same five Pro personas. Deltas are relative to "
            "Single-Call × Pro (five-persona subset).",
            "",
            "| Model | Architecture | N | ΔMPD [95% CI] | P(Δ>0), n=50 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    architecture_labels = {
        "sequential-pro-5": "Multi-Turn",
        "interactive-pro": "Multi-Agent",
    }
    for model_name, model_result in report["models"].items():
        for condition, result in model_result.get("matched_architecture", {}).items():
            ci = result["ci95"]
            subset_rate = result["subset_positive_rate"].get("50", float("nan"))
            lines.append(
                f"| {model_name} | {architecture_labels[condition]} | "
                f"{result['n_tasks']} | {fmt(result['mean_delta'])} "
                f"[{fmt(ci[0])}, {fmt(ci[1])}] | {subset_rate:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Formal persona × architecture difference-in-differences",
            "",
            "DID = (architecture Pro−None) − (Single-Call Pro−None). "
            "A value near zero is consistent with no detectable MPD interaction "
            "relative to Single-Call, but does not establish equivalence.",
            "",
            "| Model | Architecture | N | DID [95% CI] | Wilcoxon p (BH) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_name, model_result in report["models"].items():
        for architecture, result in model_result.get("did", {}).items():
            ci = result["ci95"]
            lines.append(
                f"| {model_name} | {architecture} | {result['n_tasks']} | "
                f"{fmt(result['mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] | "
                f"{result['wilcoxon_p_bh']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- These analyses quantify stability over the submitted 100-task sampling frame; they do not create evidence for other languages, geographies, or task families.",
            "- Failure to reject a zero interaction is not proof of exact orthogonality. The defensible wording is “no detectable MPD interaction in this design.”",
            "- The analyses use released per-task MPD values and do not address human construct validity.",
        ]
    )
    (OUT / "robustness_results.md").write_text("\n".join(lines) + "\n")
    print(OUT / "robustness_results.md")


if __name__ == "__main__":
    main()
