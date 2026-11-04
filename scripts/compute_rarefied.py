"""Compute rarefied CC/VS + raw MPD per tier. Three metrics side by side.

Usage:
    python scripts/compute_rarefied.py
    python scripts/compute_rarefied.py --embeddings-dir results/embeddings_gemini --metrics-dir results/metrics_gemini
"""

import argparse
import json
import numpy as np
from pathlib import Path
from modbench.metrics import mean_pairwise_distance, rarefied_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIERS = {
    'Independent Main': ['independent-none', 'independent-role', 'independent-basic', 'independent-mid', 'independent-pro'],
    'Independent Ablation': ['independent-none', 'independent-enhanced', 'independent-temp-high', 'independent-dimension', 'independent-random-sys'],
    'Sequential': ['sequential-none', 'sequential-role', 'sequential-basic', 'sequential-mid', 'sequential-pro'],
    'Interactive': ['interactive-none', 'interactive-role', 'interactive-basic', 'interactive-mid', 'interactive-pro'],
    'Cross-Framework (5-persona)': ['independent-pro-5', 'sequential-pro-5', 'interactive-pro'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", default=None, help="Override embeddings root (default: results/embeddings)")
    p.add_argument("--metrics-dir", default=None, help="Override metrics root (default: results/metrics)")
    args = p.parse_args()

    emb_base = Path(args.embeddings_dir) if args.embeddings_dir else PROJECT_ROOT / 'results' / 'embeddings'
    metrics_base = Path(args.metrics_dir) if args.metrics_dir else PROJECT_ROOT / 'results' / 'metrics'

    embeddings = emb_base / 'openai-3-small'
    if (embeddings / 'original').exists():
        embeddings = embeddings / 'original'

    results = {}

    for tier_name, conditions in TIERS.items():
        print(f"\n{'='*80}", flush=True)
        print(f"  {tier_name}", flush=True)
        print(f"{'='*80}", flush=True)

        all_emb = {}
        skip_tier = False
        for cond in conditions:
            cond_dir = embeddings / cond
            if not cond_dir.exists():
                print(f"  SKIP tier: {cond} not found", flush=True)
                skip_tier = True
                break
            all_emb[cond] = {}
            for f in sorted(cond_dir.glob('MOD-*.npy')):
                all_emb[cond][f.stem] = np.load(f)
        if skip_tier:
            continue

        tasks = sorted(set.intersection(*[set(d.keys()) for d in all_emb.values()]))
        print(f"  {len(tasks)} tasks", flush=True)

        per_cond = {c: {'mpd': [], 'cc_rar': [], 'vs_rar': []} for c in conditions}

        for i, tid in enumerate(tasks):
            ns = [len(all_emb[c][tid]) for c in conditions]
            min_n = min(ns)
            for c in conditions:
                emb = all_emb[c][tid]
                per_cond[c]['mpd'].append(mean_pairwise_distance(emb))
                rar = rarefied_metrics(emb, min_n, n_repeats=30, seed=42 + int(tid.split('-')[1]))
                per_cond[c]['cc_rar'].append(rar['cc_rarefied']['mean'])
                per_cond[c]['vs_rar'].append(rar['vs_rarefied']['mean'])
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(tasks)} tasks done", flush=True)

        print(f"\n  {'Condition':28s} {'MPD':>8s} {'CC(rar)':>10s} {'VS(rar)':>10s}", flush=True)
        print(f"  {'-'*60}", flush=True)
        tier_results = {}
        for c in conditions:
            mpd_m = float(np.mean(per_cond[c]['mpd']))
            cc_m = float(np.mean(per_cond[c]['cc_rar']))
            vs_m = float(np.mean(per_cond[c]['vs_rar']))
            print(f"  {c:28s} {mpd_m:>8.4f} {cc_m:>10.2f} {vs_m:>10.2f}", flush=True)
            tier_results[c] = {'mpd': mpd_m, 'cc_rarefied': cc_m, 'vs_rarefied': vs_m}
        results[tier_name] = tier_results

        del all_emb

    out_path = metrics_base / 'openai-3-small' / '_rarefied_by_tier.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
