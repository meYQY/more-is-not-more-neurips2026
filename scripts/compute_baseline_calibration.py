"""Compute β-diversity baseline calibration for the paper.

Two calibration checks:
1. Split-half: randomly split independent-none into two halves,
   compute β-VS and UCR. This is the "noise floor" for β metrics.
2. None vs TempHigh: two conditions with nearly identical α-diversity,
   should produce β values at or below the split-half baseline.

Usage:
    python scripts/compute_baseline_calibration.py
    python scripts/compute_baseline_calibration.py --models deepseek,gemini
"""

import argparse
import json
from pathlib import Path

import numpy as np

from modbench.metrics import vendi_score, unique_cluster_ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRS = {
    "deepseek": "embeddings",
    "gemini": "embeddings_gemini",
    "grok": "embeddings_grok",
    "kimi": "embeddings_kimi",
    "llama": "embeddings_llama",
    "qwen": "embeddings_qwen",
}


def load_task_embeddings(emb_dir: Path, condition: str) -> dict[str, np.ndarray]:
    cond_dir = emb_dir / "openai-3-small" / "original" / condition
    if not cond_dir.exists():
        return {}
    result = {}
    for f in sorted(cond_dir.glob("MOD-*.npy")):
        result[f.stem] = np.load(f)
    return result


def split_half_beta(embeddings: np.ndarray, n_repeats: int = 30, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    if n < 4:
        return {"beta_vs": float("nan"), "ucr_a": float("nan"), "ucr_b": float("nan")}

    half = n // 2
    bvs_vals, ucr_a_vals, ucr_b_vals = [], [], []

    for i in range(n_repeats):
        perm = rng.permutation(n)
        a, b = embeddings[perm[:half]], embeddings[perm[half:half * 2]]
        vs_a = vendi_score(a)
        vs_b = vendi_score(b)
        vs_union = vendi_score(np.vstack([a, b]))
        bvs = vs_union - 0.5 * vs_a - 0.5 * vs_b
        bvs_vals.append(bvs)

        ua, ub = unique_cluster_ratio(a, b)
        ucr_a_vals.append(ua)
        ucr_b_vals.append(ub)

    return {
        "beta_vs": float(np.mean(bvs_vals)),
        "ucr_mean": float(np.mean(ucr_a_vals + ucr_b_vals)),
        "ucr_a": float(np.mean(ucr_a_vals)),
        "ucr_b": float(np.mean(ucr_b_vals)),
    }


def cross_condition_beta(emb_a: np.ndarray, emb_b: np.ndarray, n_repeats: int = 30, seed: int = 42) -> dict:
    n_sub = min(len(emb_a), len(emb_b))
    if n_sub < 4:
        return {"beta_vs": float("nan"), "ucr_a": float("nan"), "ucr_b": float("nan")}

    rng = np.random.default_rng(seed)
    bvs_vals, ucr_a_vals, ucr_b_vals = [], [], []

    for i in range(n_repeats):
        idx_a = rng.choice(len(emb_a), size=n_sub, replace=False)
        idx_b = rng.choice(len(emb_b), size=n_sub, replace=False)
        a, b = emb_a[idx_a], emb_b[idx_b]

        vs_a = vendi_score(a)
        vs_b = vendi_score(b)
        vs_union = vendi_score(np.vstack([a, b]))
        bvs = vs_union - 0.5 * vs_a - 0.5 * vs_b
        bvs_vals.append(bvs)

        ua, ub = unique_cluster_ratio(a, b)
        ucr_a_vals.append(ua)
        ucr_b_vals.append(ub)

    return {
        "beta_vs": float(np.mean(bvs_vals)),
        "ucr_mean": float(np.mean(ucr_a_vals + ucr_b_vals)),
        "ucr_a": float(np.mean(ucr_a_vals)),
        "ucr_b": float(np.mean(ucr_b_vals)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=None, help="Comma-separated model names")
    p.add_argument("--n-repeats", type=int, default=30)
    args = p.parse_args()

    models = MODEL_DIRS
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        models = {k: v for k, v in MODEL_DIRS.items() if k in selected}

    all_results = {}

    for model_name, emb_dir_name in models.items():
        emb_base = PROJECT_ROOT / "results" / emb_dir_name
        if not emb_base.exists():
            print(f"SKIP {model_name}: {emb_base} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {model_name}")
        print(f"{'=' * 70}")

        # 1. Split-half baseline on independent-none
        none_embs = load_task_embeddings(emb_base, "independent-none")
        if not none_embs:
            print(f"  SKIP: no independent-none embeddings")
            continue

        print(f"\n  Split-half baseline (independent-none, {len(none_embs)} tasks):")
        sh_bvs, sh_ucr = [], []
        for tid, emb in sorted(none_embs.items()):
            r = split_half_beta(emb, n_repeats=args.n_repeats, seed=42 + int(tid.split('-')[1]))
            sh_bvs.append(r["beta_vs"])
            sh_ucr.append(r["ucr_mean"])

        sh_bvs_mean = float(np.mean(sh_bvs))
        sh_ucr_mean = float(np.mean(sh_ucr))
        print(f"    β-VS = {sh_bvs_mean:.2f},  UCR = {sh_ucr_mean:.1%}")

        # 2. None vs TempHigh (sanity check: similar α → similar β)
        th_embs = load_task_embeddings(emb_base, "independent-temp-high")
        nvt_result = None
        if th_embs:
            print(f"\n  None vs TempHigh ({len(th_embs)} tasks):")
            nvt_bvs, nvt_ucr = [], []
            common = sorted(set(none_embs) & set(th_embs))
            for tid in common:
                r = cross_condition_beta(
                    none_embs[tid], th_embs[tid],
                    n_repeats=args.n_repeats,
                    seed=42 + int(tid.split('-')[1]),
                )
                nvt_bvs.append(r["beta_vs"])
                nvt_ucr.append(r["ucr_mean"])
            nvt_bvs_mean = float(np.mean(nvt_bvs))
            nvt_ucr_mean = float(np.mean(nvt_ucr))
            print(f"    β-VS = {nvt_bvs_mean:.2f},  UCR = {nvt_ucr_mean:.1%}")
            nvt_result = {"beta_vs": nvt_bvs_mean, "ucr": nvt_ucr_mean, "n_tasks": len(common)}
        else:
            print(f"  SKIP None vs TempHigh: no temp-high embeddings")

        all_results[model_name] = {
            "split_half": {"beta_vs": sh_bvs_mean, "ucr": sh_ucr_mean, "n_tasks": len(none_embs)},
            "none_vs_temphigh": nvt_result,
        }

    # Summary table
    print(f"\n\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  {'Model':12s} {'SH β-VS':>8s} {'SH UCR':>8s} {'NvT β-VS':>9s} {'NvT UCR':>8s}")
    print(f"  {'-' * 50}")
    for model_name, r in all_results.items():
        sh = r["split_half"]
        nvt = r.get("none_vs_temphigh") or {}
        nvt_bvs = f"{nvt['beta_vs']:.2f}" if nvt.get("beta_vs") else "—"
        nvt_ucr = f"{nvt['ucr']:.1%}" if nvt.get("ucr") else "—"
        print(f"  {model_name:12s} {sh['beta_vs']:>8.2f} {sh['ucr']:>7.1%} {nvt_bvs:>9s} {nvt_ucr:>8s}")

    # Save
    out_path = PROJECT_ROOT / "results" / "baseline_calibration.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
