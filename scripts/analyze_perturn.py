"""Per-turn/per-round diversity analysis + length confound check.

Analyzes:
1. How much diversity does each turn/round contribute? (cumulative MPD)
2. Opinion count per condition (length confound check)

Usage:
    python scripts/analyze_perturn.py
    python scripts/analyze_perturn.py --models deepseek,gemini
"""

import argparse
import json
from pathlib import Path

import numpy as np

from modbench.embedding import embed_texts, EMBEDDER_DIMS
from modbench.metrics import mean_pairwise_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "deepseek": ("results/extracted", "results/embeddings/openai-3-small"),
    "gemini": ("results/extracted_gemini", "results/embeddings_gemini/openai-3-small"),
    "kimi": ("results/extracted_kimi", "results/embeddings_kimi/openai-3-small"),
    "llama": ("results/extracted_llama", "results/embeddings_llama/openai-3-small"),
    "grok": ("results/extracted_grok", "results/embeddings_grok/openai-3-small"),
    "qwen": ("results/extracted_qwen", "results/embeddings_qwen/openai-3-small"),
}

# Handle DeepSeek's original/ subdir
for k, (ext, emb) in MODELS.items():
    emb_path = PROJECT_ROOT / emb
    if (emb_path / "original").exists():
        MODELS[k] = (ext, emb + "/original")


def load_opinions_by_step(extracted_dir: Path, condition: str, step_key: str) -> dict[str, dict[int, list[str]]]:
    """Load opinions grouped by task and turn/round number.

    Returns: {task_id: {step_number: [opinion_texts]}}
    """
    cond_dir = extracted_dir / condition
    if not cond_dir.exists():
        return {}

    result = {}
    for f in sorted(cond_dir.glob("MOD-*.jsonl")):
        tid = f.stem
        by_step: dict[int, list[str]] = {}
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                step = rec.get(step_key)
                if step is None:
                    continue
                ops = [o for o in rec.get("opinions", []) if o.strip()]
                if step not in by_step:
                    by_step[step] = []
                by_step[step].extend(ops)
        if by_step:
            result[tid] = by_step
    return result


def load_embeddings_by_step(emb_dir: Path, extracted_dir: Path, condition: str, step_key: str) -> dict[str, dict[int, np.ndarray]]:
    """Load embeddings split by turn/round, using opinion indices from extracted files.

    Since embeddings are stored as one .npy per task (all opinions concatenated),
    we need to reconstruct which embedding corresponds to which turn/round.
    """
    cond_emb = emb_dir / condition
    cond_ext = extracted_dir / condition
    if not cond_emb.exists() or not cond_ext.exists():
        return {}

    result = {}
    for npy_file in sorted(cond_emb.glob("MOD-*.npy")):
        tid = npy_file.stem
        ext_file = cond_ext / f"{tid}.jsonl"
        if not ext_file.exists():
            continue

        all_emb = np.load(npy_file)

        # Reconstruct opinion-to-step mapping
        step_indices: dict[int, list[int]] = {}
        idx = 0
        with open(ext_file) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                step = rec.get(step_key)
                if step is None:
                    continue
                n_ops = len([o for o in rec.get("opinions", []) if o.strip()])
                if step not in step_indices:
                    step_indices[step] = []
                step_indices[step].extend(range(idx, idx + n_ops))
                idx += n_ops

        if idx != len(all_emb):
            # Mismatch — skip this task
            continue

        by_step = {}
        for step, indices in step_indices.items():
            if indices:
                by_step[step] = all_emb[indices]
        if by_step:
            result[tid] = by_step

    return result


def cumulative_mpd(task_data: dict[int, np.ndarray], max_step: int) -> list[float]:
    """Compute cumulative MPD: step 1 only, steps 1-2, steps 1-2-3."""
    mpds = []
    accumulated = []
    for step in range(1, max_step + 1):
        if step in task_data:
            accumulated.append(task_data[step])
        if accumulated:
            combined = np.vstack(accumulated)
            mpds.append(mean_pairwise_distance(combined))
        else:
            mpds.append(float("nan"))
    return mpds


def analyze_perturn(model_name: str, ext_root: Path, emb_root: Path, results: dict) -> None:
    """Run per-turn analysis for Sequential and Interactive conditions."""

    for framework, step_key, conditions in [
        ("Sequential", "turn_number", ["sequential-none", "sequential-pro"]),
        ("Interactive", "round_number", ["interactive-none", "interactive-pro"]),
    ]:
        for cond in conditions:
            data = load_embeddings_by_step(emb_root, ext_root, cond, step_key)
            if not data:
                continue

            max_step = 3
            per_task_cumulative = []
            for tid in sorted(data):
                mpds = cumulative_mpd(data[tid], max_step)
                per_task_cumulative.append(mpds)

            if not per_task_cumulative:
                continue

            arr = np.array(per_task_cumulative)  # (n_tasks, 3)
            means = np.nanmean(arr, axis=0)

            key = f"{model_name}/{cond}"
            results["perturn"][key] = {
                "steps": list(range(1, max_step + 1)),
                "cumulative_mpd": [float(m) for m in means],
                "n_tasks": len(per_task_cumulative),
            }

            # Per-step marginal contribution
            marginal = [means[0]] + [means[i] - means[i-1] for i in range(1, len(means))]
            results["perturn"][key]["marginal_mpd"] = [float(m) for m in marginal]


def analyze_opinion_counts(model_name: str, ext_root: Path, results: dict) -> None:
    """Count mean opinions per task for each condition (length confound check)."""

    conditions = [
        "independent-none", "independent-role", "independent-basic",
        "independent-mid", "independent-pro",
        "independent-enhanced", "independent-temp-high",
        "independent-dimension", "independent-random-sys",
        "sequential-none", "sequential-pro",
        "interactive-none", "interactive-pro",
    ]

    for cond in conditions:
        cond_dir = ext_root / cond
        if not cond_dir.exists():
            continue

        counts = []
        for f in sorted(cond_dir.glob("MOD-*.jsonl")):
            task_ops = 0
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    task_ops += len([o for o in rec.get("opinions", []) if o.strip()])
            counts.append(task_ops)

        if counts:
            key = f"{model_name}/{cond}"
            results["opinion_counts"][key] = {
                "mean": float(np.mean(counts)),
                "std": float(np.std(counts, ddof=1)),
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
                "n_tasks": len(counts),
            }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=None, help="Comma-separated (default: all)")
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS.keys())

    results = {"perturn": {}, "opinion_counts": {}}

    for model_name in selected:
        if model_name not in MODELS:
            print(f"SKIP {model_name}")
            continue

        ext_rel, emb_rel = MODELS[model_name]
        ext_root = PROJECT_ROOT / ext_rel
        emb_root = PROJECT_ROOT / emb_rel

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        analyze_perturn(model_name, ext_root, emb_root, results)
        analyze_opinion_counts(model_name, ext_root, results)

    # Print per-turn results
    print(f"\n\n{'='*80}")
    print("  Per-turn/round cumulative MPD")
    print(f"{'='*80}")
    for key, data in sorted(results["perturn"].items()):
        cum = data["cumulative_mpd"]
        mar = data["marginal_mpd"]
        print(f"\n  {key} (n={data['n_tasks']})")
        for i, (c, m) in enumerate(zip(cum, mar)):
            pct = f" ({m/cum[-1]*100:+.0f}%)" if cum[-1] > 0 and i > 0 else ""
            print(f"    Step {i+1}: cumulative MPD={c:.4f}  marginal={m:+.4f}{pct}")

    # Print opinion counts
    print(f"\n\n{'='*80}")
    print("  Mean opinions per task (length confound check)")
    print(f"{'='*80}")

    # Group by model
    by_model: dict[str, dict[str, float]] = {}
    for key, data in results["opinion_counts"].items():
        model, cond = key.split("/", 1)
        if model not in by_model:
            by_model[model] = {}
        by_model[model][cond] = data["mean"]

    conds_order = [
        "independent-none", "independent-role", "independent-basic",
        "independent-mid", "independent-pro",
        "sequential-none", "sequential-pro",
        "interactive-none", "interactive-pro",
    ]

    print(f"\n{'Condition':24s}", end="")
    for m in sorted(by_model): print(f" {m:>9s}", end="")
    print()
    print("-" * (24 + 10 * len(by_model)))
    for c in conds_order:
        print(f"{c:24s}", end="")
        for m in sorted(by_model):
            v = by_model[m].get(c)
            print(f" {v:>9.0f}" if v else f" {'—':>9s}", end="")
        print()

    # Save
    out_path = PROJECT_ROOT / "results" / "_perturn_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
