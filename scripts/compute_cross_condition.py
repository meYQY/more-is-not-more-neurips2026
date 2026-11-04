"""Compute cross-condition metrics: β-VS and Unique Cluster Ratio.

For each task, compares two conditions' opinion embeddings:
  - β-VS: how much NEW diversity emerges from combining A and B?
  - Unique Cluster Ratio: what fraction of A's clusters are absent in B?

Both metrics are computed at matched N (rarefy to min) for fairness.

Usage:
    python scripts/compute_cross_condition.py --embedder openai-3-small
    python scripts/compute_cross_condition.py --embedder openai-3-small --embeddings-dir results/embeddings_gemini --metrics-dir results/metrics_gemini
"""

import argparse
import json
from pathlib import Path

import numpy as np

from modbench.metrics import vendi_score, unique_cluster_ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def beta_vs_matched(emb_a: np.ndarray, emb_b: np.ndarray, n_sub: int, seed: int = 0, n_repeats: int = 30) -> float:
    """β-VS at matched N. Subsample both to n_sub, repeat n_repeats times."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_repeats):
        idx_a = rng.choice(len(emb_a), size=n_sub, replace=False)
        idx_b = rng.choice(len(emb_b), size=n_sub, replace=False)
        a, b = emb_a[idx_a], emb_b[idx_b]
        vs_a = vendi_score(a)
        vs_b = vendi_score(b)
        vs_union = vendi_score(np.vstack([a, b]))
        vals.append(float(vs_union - 0.5 * vs_a - 0.5 * vs_b))
    return float(np.mean(vals))


def unique_cluster_ratio_matched(emb_a: np.ndarray, emb_b: np.ndarray, n_sub: int, seed: int = 0) -> tuple[float, float]:
    """Unique cluster ratio at matched N."""
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(emb_a), size=n_sub, replace=False)
    idx_b = rng.choice(len(emb_b), size=n_sub, replace=False)
    return unique_cluster_ratio(emb_a[idx_a], emb_b[idx_b])


# Key comparisons for cross-condition analysis
COMPARISONS = [
    # Persona depth effect (does persona bring NEW content?)
    ("independent-none", "independent-role", "None vs Role"),
    ("independent-none", "independent-basic", "None vs Basic"),
    ("independent-none", "independent-pro", "None vs Pro"),
    ("independent-basic", "independent-pro", "Basic vs Pro"),
    # Framework effect (does framework bring NEW content?)
    ("independent-pro-5", "sequential-pro-5", "Indep vs Seq (5-persona Pro)"),
    ("independent-pro-5", "interactive-pro", "Indep vs Int (5-persona Pro)"),
    ("sequential-pro-5", "interactive-pro", "Seq vs Int (5-persona Pro)"),
    # Cross-framework at None
    ("independent-none", "sequential-none", "Indep vs Seq (None)"),
    ("independent-none", "interactive-none", "Indep vs Int (None)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", default="openai-3-small")
    p.add_argument("--embeddings-dir", default=None, help="Override embeddings root")
    p.add_argument("--metrics-dir", default=None, help="Override metrics root")
    args = p.parse_args()

    emb_base = Path(args.embeddings_dir) if args.embeddings_dir else PROJECT_ROOT / "results" / "embeddings"
    metrics_base = Path(args.metrics_dir) if args.metrics_dir else PROJECT_ROOT / "results" / "metrics"

    emb_root = emb_base / args.embedder
    if (emb_root / "original").exists():
        emb_root = emb_root / "original"

    print(f"Cross-condition metrics ({args.embedder})")
    print(f"{'Comparison':30s} {'β-VS':>8s} {'A uniq%':>8s} {'B uniq%':>8s}")
    print("-" * 60)

    results = []
    for cond_a, cond_b, label in COMPARISONS:
        dir_a = emb_root / cond_a
        dir_b = emb_root / cond_b
        if not dir_a.exists() or not dir_b.exists():
            print(f"{'skip: ' + label:30s} (missing dir)")
            continue

        beta_vs_list = []
        ucr_a_list = []
        ucr_b_list = []

        tasks_a = {f.stem for f in dir_a.glob("MOD-*.npy")}
        tasks_b = {f.stem for f in dir_b.glob("MOD-*.npy")}
        common = sorted(tasks_a & tasks_b)

        for tid in common:
            ea = np.load(dir_a / f"{tid}.npy")
            eb = np.load(dir_b / f"{tid}.npy")
            if len(ea) < 2 or len(eb) < 2:
                continue

            min_n = min(len(ea), len(eb))
            bvs = beta_vs_matched(ea, eb, min_n, seed=42 + int(tid.split('-')[1]))
            beta_vs_list.append(bvs)

            ucr_a, ucr_b = unique_cluster_ratio_matched(ea, eb, min_n, seed=42 + int(tid.split('-')[1]))
            ucr_a_list.append(ucr_a)
            ucr_b_list.append(ucr_b)

        if not beta_vs_list:
            continue

        mean_bvs = float(np.mean(beta_vs_list))
        mean_ucr_a = float(np.mean(ucr_a_list))
        mean_ucr_b = float(np.mean(ucr_b_list))

        print(f"{label:30s} {mean_bvs:>8.2f} {mean_ucr_a:>7.1%} {mean_ucr_b:>7.1%}")

        results.append({
            "a": cond_a, "b": cond_b, "label": label,
            "n_tasks": len(beta_vs_list),
            "beta_vs_mean": mean_bvs,
            "unique_cluster_ratio_a": mean_ucr_a,
            "unique_cluster_ratio_b": mean_ucr_b,
        })

    out_path = metrics_base / args.embedder / "_cross_condition.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
