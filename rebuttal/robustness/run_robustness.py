#!/usr/bin/env python3
"""No-API robustness analyses for the More Is Not More rebuttal.

This script only reads the released per-task metric files. It writes all
derived tables beside itself and never modifies the source repository.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# Repository root with data/dataset/tasks.csv and results/metrics*/ (override: MODBENCH_ROOT).
REPO = Path(os.environ.get("MODBENCH_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent
SEED = 1529
N_BOOT = 10_000
N_SUBSAMPLE = 5_000

MODEL_DIRS = {
    "deepseek": "metrics",
    "gpt54mini": "metrics_gpt54mini",
    "gemini": "metrics_gemini",
    "kimi": "metrics_kimi",
    "llama": "metrics_llama",
    "grok": "metrics_grok",
    "qwen": "metrics_qwen",
}

FULL_GRID_MODELS = ["deepseek", "gemini", "kimi", "llama", "grok", "qwen"]


def load_tasks() -> tuple[dict[str, str], dict[str, str]]:
    categories: dict[str, str] = {}
    questions: dict[str, str] = {}
    with (REPO / "data/dataset/tasks.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            categories[row["task_id"]] = row["subcategory"]
            questions[row["task_id"]] = row["query"]
    return categories, questions


def load_all_mpd() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for model, dirname in MODEL_DIRS.items():
        root = REPO / "results" / dirname / "openai-3-small"
        data[model] = {}
        for p in root.glob("*.json"):
            if p.name.startswith("_"):
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            condition = obj.get("condition", p.stem)
            data[model][condition] = {
                task: float(metrics["mpd"])
                for task, metrics in obj.get("per_task", {}).items()
                if metrics.get("mpd") is not None
                and np.isfinite(float(metrics["mpd"]))
            }
    return data


def linear_contrast(
    data: dict[str, dict[str, dict[str, float]]],
    model: str,
    terms: list[tuple[str, float]],
) -> dict[str, float]:
    """Return a per-task linear contrast, aligned within a model."""
    if not all(condition in data[model] for condition, _ in terms):
        return {}
    common = set.intersection(*(set(data[model][c]) for c, _ in terms))
    return {
        task: float(sum(weight * data[model][condition][task] for condition, weight in terms))
        for task in sorted(common)
    }


def averaged_contrast(
    data: dict[str, dict[str, dict[str, float]]],
    model: str,
    component_terms: list[list[tuple[str, float]]],
) -> dict[str, float]:
    """Average several per-task linear contrasts (e.g., over architectures)."""
    components = [linear_contrast(data, model, terms) for terms in component_terms]
    if not components or any(not c for c in components):
        return {}
    common = set.intersection(*(set(c) for c in components))
    return {task: float(np.mean([c[task] for c in components])) for task in sorted(common)}


def make_matrix(
    per_model: dict[str, dict[str, float]],
    models: list[str] | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    selected = models if models is not None else sorted(m for m, values in per_model.items() if values)
    selected = [m for m in selected if per_model.get(m)]
    common = sorted(set.intersection(*(set(per_model[m]) for m in selected)))
    matrix = np.asarray([[per_model[m][task] for task in common] for m in selected], dtype=float)
    return selected, common, matrix


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_matrix(matrix: np.ndarray, seed: int) -> dict[str, float]:
    """Task-, model-, and crossed-bootstrap CIs for an M x T contrast matrix."""
    rng = np.random.default_rng(seed)
    n_models, n_tasks = matrix.shape
    task_boot = np.empty(N_BOOT)
    model_boot = np.empty(N_BOOT)
    crossed_boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        task_idx = rng.integers(0, n_tasks, n_tasks)
        model_idx = rng.integers(0, n_models, n_models)
        task_boot[i] = matrix[:, task_idx].mean()
        model_boot[i] = matrix[model_idx, :].mean()
        crossed_boot[i] = matrix[np.ix_(model_idx, task_idx)].mean()
    task_lo, task_hi = ci(task_boot)
    model_lo, model_hi = ci(model_boot)
    cross_lo, cross_hi = ci(crossed_boot)
    return {
        "estimate": float(matrix.mean()),
        "task_boot_ci_low": task_lo,
        "task_boot_ci_high": task_hi,
        "model_boot_ci_low": model_lo,
        "model_boot_ci_high": model_hi,
        "crossed_boot_ci_low": cross_lo,
        "crossed_boot_ci_high": cross_hi,
        "task_boot_p_positive": float(np.mean(task_boot > 0)),
        "crossed_boot_p_positive": float(np.mean(crossed_boot > 0)),
    }


def per_model_stats(
    contrast_name: str,
    per_model: dict[str, dict[str, float]],
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for j, model in enumerate(sorted(per_model)):
        values_dict = per_model[model]
        if not values_dict:
            continue
        values = np.asarray(list(values_dict.values()), dtype=float)
        rng = np.random.default_rng(seed + j)
        boots = np.asarray([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(N_BOOT)
        ])
        rb = float((np.sum(values > 0) - np.sum(values < 0)) / len(values))
        rb_boot = np.asarray([
            (
                np.sum(sample > 0) - np.sum(sample < 0)
            ) / len(sample)
            for sample in (
                rng.choice(values, size=len(values), replace=True)
                for _ in range(N_BOOT)
            )
        ])
        try:
            p = float(wilcoxon(values).pvalue) if not np.all(values == 0) else 1.0
        except ValueError:
            p = 1.0
        mean_lo, mean_hi = ci(boots)
        rb_lo, rb_hi = ci(rb_boot)
        rows.append({
            "contrast": contrast_name,
            "model": model,
            "n_tasks": len(values),
            "mean_effect": float(values.mean()),
            "mean_ci_low": mean_lo,
            "mean_ci_high": mean_hi,
            "task_positive_rate": float(np.mean(values > 0)),
            "rank_biserial_sign": rb,
            "rank_biserial_ci_low": rb_lo,
            "rank_biserial_ci_high": rb_hi,
            "wilcoxon_p_raw": p,
        })
    if rows:
        adjusted = multipletests([r["wilcoxon_p_raw"] for r in rows], method="fdr_bh")[1]
        for row, p_adj in zip(rows, adjusted):
            row["wilcoxon_p_fdr_within_contrast"] = float(p_adj)
    return rows


def subsample_stability(
    contrast_name: str,
    tasks: list[str],
    matrix: np.ndarray,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    full = float(matrix.mean())
    rows: list[dict] = []
    for n in [10, 25, 50, 75]:
        if n > len(tasks):
            continue
        estimates = np.empty(N_SUBSAMPLE)
        for i in range(N_SUBSAMPLE):
            idx = rng.choice(len(tasks), size=n, replace=False)
            estimates[i] = matrix[:, idx].mean()
        lo, hi = ci(estimates)
        direction = np.sign(full)
        same_direction = float(np.mean(np.sign(estimates) == direction))
        if abs(full) > 1e-12:
            within_20pct = float(np.mean(np.abs(estimates - full) <= 0.2 * abs(full)))
        else:
            within_20pct = float("nan")
        rows.append({
            "contrast": contrast_name,
            "n_subsample": n,
            "n_repeats": N_SUBSAMPLE,
            "full_estimate": full,
            "subsample_mean": float(estimates.mean()),
            "subsample_ci_low": lo,
            "subsample_ci_high": hi,
            "same_direction_rate": same_direction,
            "within_20pct_rate": within_20pct,
        })
    return rows


def category_analyses(
    contrast_name: str,
    tasks: list[str],
    matrix: np.ndarray,
    categories: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    full = float(matrix.mean())
    category_names = sorted({categories[t] for t in tasks})
    loo_rows: list[dict] = []
    category_rows: list[dict] = []
    for category in category_names:
        in_cat = np.asarray([categories[t] == category for t in tasks])
        cat_effect = float(matrix[:, in_cat].mean())
        loo_effect = float(matrix[:, ~in_cat].mean())
        category_rows.append({
            "contrast": contrast_name,
            "category": category,
            "n_tasks": int(in_cat.sum()),
            "category_effect": cat_effect,
            "same_direction_as_full": bool(np.sign(cat_effect) == np.sign(full)),
        })
        loo_rows.append({
            "contrast": contrast_name,
            "left_out_category": category,
            "n_left_out": int(in_cat.sum()),
            "n_retained": int((~in_cat).sum()),
            "full_effect": full,
            "leave_one_category_out_effect": loo_effect,
            "absolute_change": loo_effect - full,
            "relative_change": (loo_effect - full) / abs(full) if abs(full) > 1e-12 else float("nan"),
            "same_direction": bool(np.sign(loo_effect) == np.sign(full)),
        })
    return loo_rows, category_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    categories, _ = load_tasks()
    data = load_all_mpd()

    archs = ["independent", "sequential", "interactive"]
    none_to_role = [[(f"{a}-role", 1), (f"{a}-none", -1)] for a in archs]
    role_to_pro = [[(f"{a}-pro", 1), (f"{a}-role", -1)] for a in archs]
    initial_minus_extra = [
        [(f"{a}-role", 2), (f"{a}-none", -1), (f"{a}-pro", -1)]
        for a in archs
    ]
    basic_to_mid = [[(f"{a}-mid", 1), (f"{a}-basic", -1)] for a in archs]

    definitions: dict[str, tuple[str, object, list[str] | None]] = {
        "SC_None_to_Role": (
            "linear",
            [("independent-role", 1), ("independent-none", -1)],
            None,
        ),
        "AllArch_None_to_Role": ("average", none_to_role, FULL_GRID_MODELS),
        "AllArch_Role_to_Pro": ("average", role_to_pro, FULL_GRID_MODELS),
        "AllArch_InitialMinusExtraDepth": (
            "average",
            initial_minus_extra,
            FULL_GRID_MODELS,
        ),
        "AllArch_Basic_to_Mid": ("average", basic_to_mid, FULL_GRID_MODELS),
        "SC_None_to_Pro": (
            "linear",
            [("independent-pro", 1), ("independent-none", -1)],
            None,
        ),
        "Matched5_SC_to_MT_Pro": (
            "linear",
            [("sequential-pro-5", 1), ("independent-pro-5", -1)],
            None,
        ),
        "Matched5_SC_to_MA_Pro": (
            "linear",
            [("interactive-pro", 1), ("independent-pro-5", -1)],
            FULL_GRID_MODELS,
        ),
        "Matched5_MT_to_MA_Pro": (
            "linear",
            [("interactive-pro", 1), ("sequential-pro-5", -1)],
            FULL_GRID_MODELS,
        ),
        "SC_5_to_20_Personas_Pro": (
            "linear",
            [("independent-pro", 1), ("independent-pro-5", -1)],
            None,
        ),
        "SC_None_to_PersonalityLabel": (
            "linear",
            [("independent-random-sys", 1), ("independent-none", -1)],
            None,
        ),
        "SC_PersonalityLabel_to_Role": (
            "linear",
            [("independent-role", 1), ("independent-random-sys", -1)],
            None,
        ),
    }

    aggregate_summary: dict[str, dict] = {}
    model_rows: list[dict] = []
    subsample_rows: list[dict] = []
    loo_rows: list[dict] = []
    category_rows: list[dict] = []
    contrast_cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}

    for i, (name, (kind, terms, selected_models)) in enumerate(definitions.items()):
        if kind == "linear":
            per_model = {
                model: linear_contrast(data, model, terms)  # type: ignore[arg-type]
                for model in MODEL_DIRS
            }
        else:
            per_model = {
                model: averaged_contrast(data, model, terms)  # type: ignore[arg-type]
                for model in MODEL_DIRS
            }
        models, tasks, matrix = make_matrix(per_model, selected_models)
        contrast_cache[name] = (models, tasks, matrix)
        summary = bootstrap_matrix(matrix, SEED + 100 * i)
        summary.update({
            "models": models,
            "n_models": len(models),
            "n_common_tasks": len(tasks),
        })
        aggregate_summary[name] = summary
        model_rows.extend(per_model_stats(name, {m: per_model[m] for m in models}, SEED + 1000 * i))
        subsample_rows.extend(subsample_stability(name, tasks, matrix, SEED + 2000 * i))
        loo, cat = category_analyses(name, tasks, matrix, categories)
        loo_rows.extend(loo)
        category_rows.extend(cat)

    # Formal difference-in-differences for the None->Pro persona effect.
    did_specs = {
        "DiD_MT_minus_SC_NoneToPro": [
            ("sequential-pro", 1),
            ("sequential-none", -1),
            ("independent-pro", -1),
            ("independent-none", 1),
        ],
        "DiD_MA_minus_SC_NoneToPro": [
            ("interactive-pro", 1),
            ("interactive-none", -1),
            ("independent-pro", -1),
            ("independent-none", 1),
        ],
        "DiD_MA_minus_MT_NoneToPro": [
            ("interactive-pro", 1),
            ("interactive-none", -1),
            ("sequential-pro", -1),
            ("sequential-none", 1),
        ],
    }
    did_model_rows: list[dict] = []
    did_aggregate: dict[str, dict] = {}
    for i, (name, terms) in enumerate(did_specs.items()):
        per_model = {m: linear_contrast(data, m, terms) for m in FULL_GRID_MODELS}
        models, tasks, matrix = make_matrix(per_model, FULL_GRID_MODELS)
        summary = bootstrap_matrix(matrix, SEED + 50_000 + 100 * i)
        summary.update({
            "models": models,
            "n_models": len(models),
            "n_common_tasks": len(tasks),
        })
        did_aggregate[name] = summary
        did_model_rows.extend(per_model_stats(name, per_model, SEED + 60_000 + 1000 * i))
        subsample_rows.extend(subsample_stability(name, tasks, matrix, SEED + 70_000 + 2000 * i))
        loo, cat = category_analyses(name, tasks, matrix, categories)
        loo_rows.extend(loo)
        category_rows.extend(cat)

    # FDR across all 18 model-specific DiD tests, matching the requested formal audit.
    did_p = [r["wilcoxon_p_raw"] for r in did_model_rows]
    did_adj = multipletests(did_p, method="fdr_bh")[1]
    for row, p_adj in zip(did_model_rows, did_adj):
        row["wilcoxon_p_fdr_all_did"] = float(p_adj)

    # Compact diagnostics that are easy to cite.
    diagnostics: dict[str, dict] = {}
    for name, rows in defaultdict(list, {
        key: [r for r in loo_rows if r["contrast"] == key]
        for key in set(r["contrast"] for r in loo_rows)
    }).items():
        diagnostics[name] = {
            "loo_min": float(min(r["leave_one_category_out_effect"] for r in rows)),
            "loo_max": float(max(r["leave_one_category_out_effect"] for r in rows)),
            "loo_all_same_direction": bool(all(r["same_direction"] for r in rows)),
            "max_abs_loo_change": float(max(abs(r["absolute_change"]) for r in rows)),
        }
    for name in diagnostics:
        cat = [r for r in category_rows if r["contrast"] == name]
        diagnostics[name]["categories_same_direction"] = int(sum(r["same_direction_as_full"] for r in cat))
        diagnostics[name]["n_categories"] = len(cat)

    output = {
        "method": {
            "metric": "per-task MPD from released OpenAI text-embedding-3-small results",
            "task_bootstrap": f"{N_BOOT} resamples of task IDs, preserving cross-model dependence",
            "model_bootstrap": f"{N_BOOT} resamples of model IDs",
            "crossed_bootstrap": f"{N_BOOT} independent resamples of model and task IDs",
            "subsample": f"{N_SUBSAMPLE} random subsets without replacement at n=10,25,50,75",
            "leave_one_category_out": "drop each of the 10 released task subcategories in turn",
            "estimand": "equal-weight mean over included models and common tasks",
            "seed": SEED,
        },
        "aggregate_contrasts": aggregate_summary,
        "difference_in_differences": did_aggregate,
        "diagnostics": diagnostics,
    }
    (OUT / "robustness_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(OUT / "key_effects_by_model.csv", model_rows)
    write_csv(OUT / "subsample_stability.csv", subsample_rows)
    write_csv(OUT / "leave_one_category_out.csv", loo_rows)
    write_csv(OUT / "category_effects.csv", category_rows)
    write_csv(OUT / "did_results_by_model.csv", did_model_rows)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
