"""Step 3b: Compute pairwise statistical tests across conditions.

For each model, runs paired Wilcoxon + Cliff's delta on key comparisons,
then applies BH FDR correction. Reads per-task MPD from metrics JSON files.

Usage:
    python scripts/compute_stats.py
    python scripts/compute_stats.py --metrics-dir results/metrics_gemini
    python scripts/compute_stats.py --models deepseek,gemini,kimi
"""

import argparse
import json
from pathlib import Path

import numpy as np

from modbench.stats import paired_wilcoxon, paired_cliffs_delta, fdr_correction

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Key pairwise comparisons — same for all models
COMPARISONS = [
    # Persona depth (Independent)
    ("A1", "independent-none", "independent-role", "Independent: None→Role"),
    ("A2", "independent-role", "independent-basic", "Independent: Role→Basic"),
    ("A3", "independent-basic", "independent-mid", "Independent: Basic→Mid"),
    ("A4", "independent-mid", "independent-pro", "Independent: Mid→Pro"),
    ("A5", "independent-none", "independent-pro", "Independent: None→Pro"),
    # Ablations vs None
    ("B1", "independent-none", "independent-enhanced", "Ablation: None vs Enhanced"),
    ("B2", "independent-none", "independent-temp-high", "Ablation: None vs TempHigh"),
    ("B3", "independent-none", "independent-dimension", "Ablation: None vs Dimension"),
    ("B4", "independent-none", "independent-random-sys", "Ablation: None vs RandomSys"),
    # Framework effect (at None)
    ("C1", "independent-none", "sequential-none", "Framework: Indep→Seq (None)"),
    ("C2", "independent-none", "interactive-none", "Framework: Indep→Int (None)"),
    ("C3", "sequential-none", "interactive-none", "Framework: Seq vs Int (None)"),
    # Framework effect (at Pro)
    ("C4", "independent-pro", "sequential-pro", "Framework: Indep→Seq (Pro)"),
    ("C5", "independent-pro", "interactive-pro", "Framework: Indep→Int (Pro)"),
    ("C6", "sequential-pro", "interactive-pro", "Framework: Seq vs Int (Pro)"),
    # Framework effect (at Role)
    ("C7", "independent-role", "sequential-role", "Framework: Indep→Seq (Role)"),
    ("C8", "independent-role", "interactive-role", "Framework: Indep→Int (Role)"),
    ("C9", "sequential-role", "interactive-role", "Framework: Seq vs Int (Role)"),
    # Persona depth (Sequential)
    ("D0", "sequential-none", "sequential-role", "Sequential: None→Role"),
    ("D1", "sequential-none", "sequential-basic", "Sequential: None→Basic"),
    ("D2", "sequential-basic", "sequential-mid", "Sequential: Basic→Mid"),
    ("D3", "sequential-mid", "sequential-pro", "Sequential: Mid→Pro"),
    ("D4", "sequential-none", "sequential-pro", "Sequential: None→Pro"),
    # Persona depth (Interactive)
    ("E0", "interactive-none", "interactive-role", "Interactive: None→Role"),
    ("E1", "interactive-none", "interactive-basic", "Interactive: None→Basic"),
    ("E2", "interactive-basic", "interactive-mid", "Interactive: Basic→Mid"),
    ("E3", "interactive-mid", "interactive-pro", "Interactive: Mid→Pro"),
    ("E4", "interactive-none", "interactive-pro", "Interactive: None→Pro"),
    # Persona count
    ("F1", "independent-pro", "independent-pro-5", "Persona count: 20 vs 5"),
]


def classify_effect(d: float) -> str:
    """Romano et al. 2006 thresholds."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def load_per_task_metric(metrics_root: Path, condition: str, metric: str = "mpd") -> dict[str, float]:
    """Load per-task values for a given metric from a condition's metrics JSON."""
    f = metrics_root / f"{condition}.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        data = json.load(fh)
    return {tid: m[metric] for tid, m in data.get("per_task", {}).items()
            if not np.isnan(m.get(metric, float("nan")))}


def run_comparisons(metrics_root: Path, metric: str = "mpd") -> list[dict]:
    """Run all pairwise comparisons for one model on a given metric."""
    results = []
    raw_pvals = []

    for comp_id, cond_a, cond_b, label in COMPARISONS:
        vals_a_dict = load_per_task_metric(metrics_root, cond_a, metric)
        vals_b_dict = load_per_task_metric(metrics_root, cond_b, metric)

        if not vals_a_dict or not vals_b_dict:
            continue

        # Align by task
        common = sorted(set(vals_a_dict) & set(vals_b_dict))
        if len(common) < 10:
            continue

        vals_a = [vals_a_dict[t] for t in common]
        vals_b = [vals_b_dict[t] for t in common]

        p = paired_wilcoxon(vals_a, vals_b)
        d = paired_cliffs_delta(vals_a, vals_b)
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        b_wins = sum(1 for a, b in zip(vals_a, vals_b) if b > a)

        results.append({
            "id": comp_id,
            "a": cond_a,
            "b": cond_b,
            "label": label,
            "metric": metric,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "diff": mean_b - mean_a,
            "p_raw": p,
            "rank_biserial": round(d, 3),
            "effect": classify_effect(d),
            "b_wins": b_wins,
            "n_tasks": len(common),
        })
        raw_pvals.append(p)

    # FDR correction
    if raw_pvals:
        adjusted = fdr_correction(raw_pvals)
        for r, padj in zip(results, adjusted):
            r["p_fdr"] = padj
            r["sig"] = padj < 0.05

    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=None, help="Comma-separated model suffixes (default: all)")
    p.add_argument("--metrics-dir", default=None, help="Override: run for single metrics dir")
    return p.parse_args()


def main():
    args = parse_args()

    model_dirs = {
        "deepseek": PROJECT_ROOT / "results" / "metrics",
        "gpt54mini": PROJECT_ROOT / "results" / "metrics_gpt54mini",
        "gemini": PROJECT_ROOT / "results" / "metrics_gemini",
        "kimi": PROJECT_ROOT / "results" / "metrics_kimi",
        "llama": PROJECT_ROOT / "results" / "metrics_llama",
        "grok": PROJECT_ROOT / "results" / "metrics_grok",
        "qwen": PROJECT_ROOT / "results" / "metrics_qwen",
    }

    if args.metrics_dir:
        model_dirs = {"custom": Path(args.metrics_dir)}

    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        model_dirs = {k: v for k, v in model_dirs.items() if k in selected}

    metrics_to_run = ["mpd", "cc", "vs"]

    for model_name, base_dir in model_dirs.items():
        metrics_root = base_dir / "openai-3-small" if args.metrics_dir is None else base_dir / "openai-3-small"
        if not metrics_root.exists():
            metrics_root = base_dir  # fallback for --metrics-dir with full path
            if not metrics_root.exists():
                print(f"SKIP {model_name}: not found")
                continue

        all_results = {}
        for metric in metrics_to_run:
            results = run_comparisons(metrics_root, metric=metric)
            if results:
                all_results[metric] = results

        if not all_results:
            print(f"SKIP {model_name}: no valid comparisons")
            continue

        # Save all metrics together
        out_path = metrics_root / "_all_comparisons.json"
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

        # Print
        print(f"\n{'=' * 80}")
        print(f"  {model_name}")
        print(f"{'=' * 80}")
        for metric, results in all_results.items():
            print(f"\n  --- {metric.upper()} ---")
            print(f"  Saved {len(results)} comparisons")
            print_table(results, metric)


def print_table(results: list[dict], metric: str = "mpd") -> None:
    label = f"Δ {metric.upper()}"
    print(f"  {'ID':4s} {'Label':40s} {label:>8s} {'r_rb':>7s} {'Effect':>10s} {'p(FDR)':>10s} {'B wins':>7s}")
    print(f"  {'-' * 90}")
    for r in results:
        p_str = f"{r['p_fdr']:.2e}" if r.get('p_fdr') is not None else "—"
        sig = "*" if r.get('sig') else ""
        print(f"  {r['id']:4s} {r['label']:40s} {r['diff']:>+8.4f} {r['rank_biserial']:>+7.3f} {r['effect']:>10s} {p_str:>10s}{sig} {r['b_wins']:>5d}/{r['n_tasks']}")


if __name__ == "__main__":
    main()
