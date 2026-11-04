"""Step 3: Compute diversity metrics from embeddings.

Reads .npy embedding files, computes MPD/CC/VS per task, aggregates per condition.
No text, no API, no scrub. Pure numpy → metric.

Usage:
    python scripts/compute_metrics.py --embedder openai-3-small
    python scripts/compute_metrics.py --embedder openai-3-small --conditions single,persona-pro
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modbench.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_provenance(extracted_root: Path, condition: str) -> dict:
    """Read upstream provenance from extracted manifest."""
    manifest = extracted_root / condition / "_manifest.json"
    if not manifest.exists():
        return {}
    with open(manifest) as fh:
        m = json.load(fh)
    return {
        "generation_model": m.get("generation_model"),
        "extraction_model": m.get("extraction_model"),
        "extraction_prompt_hash": m.get("prompt_hash"),
        "extraction_timestamp": m.get("timestamp"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", required=True, help="Embedder name (must match embeddings dir)")
    p.add_argument("--conditions", default=None, help="Comma-separated (default: all)")
    p.add_argument("--extracted-dir", default=None, help="Override extracted dir (default: results/extracted)")
    p.add_argument("--embeddings-dir", default=None, help="Override embeddings dir (default: results/embeddings)")
    p.add_argument("--metrics-dir", default=None, help="Override metrics dir (default: results/metrics)")
    return p.parse_args()


def aggregate(per_task: dict[str, dict]) -> dict:
    """Mean / std / min / max across tasks for each metric."""
    valid = {tid: m for tid, m in per_task.items() if not np.isnan(m.get("mpd", np.nan))}
    if not valid:
        return {}
    keys = ["mpd", "cc", "vs"]
    return {
        k: {
            "mean": float(np.mean([t[k] for t in valid.values()])),
            "std": float(np.std([t[k] for t in valid.values()], ddof=1)),
            "min": float(np.min([t[k] for t in valid.values()])),
            "max": float(np.max([t[k] for t in valid.values()])),
            "n": len(valid),
        }
        for k in keys
    }


def main() -> None:
    args = parse_args()

    extracted_root = Path(args.extracted_dir) if args.extracted_dir else PROJECT_ROOT / "results" / "extracted"
    embeddings_root = Path(args.embeddings_dir) if args.embeddings_dir else PROJECT_ROOT / "results" / "embeddings"
    metrics_root = Path(args.metrics_dir) if args.metrics_dir else PROJECT_ROOT / "results" / "metrics"

    # Embeddings live at embeddings/{embedder}/{condition}/ for new runs,
    # or embeddings/{embedder}/original/{condition}/ for the existing run.
    emb_root = embeddings_root / args.embedder
    if (emb_root / "original").exists():
        emb_root = emb_root / "original"  # use existing structure

    if not emb_root.exists():
        print(f"No embeddings at {emb_root}")
        raise SystemExit(1)

    conditions = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions
        else sorted(d.name for d in emb_root.iterdir() if d.is_dir())
    )

    out_root = metrics_root / args.embedder
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Embedder: {args.embedder}")
    print(f"Conditions: {len(conditions)}")
    print(f"{'condition':20s} {'tasks':>6s} {'MPD':>8s} {'CC':>8s} {'VS':>8s}")
    print("-" * 56)

    for cond in conditions:
        cond_dir = emb_root / cond
        if not cond_dir.exists():
            print(f"  {cond}: no embedding dir, skipping")
            continue

        per_task: dict[str, dict] = {}
        for npy_file in sorted(cond_dir.glob("MOD-*.npy")):
            emb = np.load(npy_file)
            per_task[npy_file.stem] = compute_metrics(emb)

        agg = aggregate(per_task)
        prov = load_provenance(extracted_root, cond)
        result = {
            "condition": cond,
            "embedder": args.embedder,
            **prov,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_tasks": len(per_task),
            "aggregate": agg,
            "per_task": per_task,
        }

        out_path = out_root / f"{cond}.json"
        out_path.write_text(json.dumps(result, indent=2))

        mpd = agg.get("mpd", {}).get("mean", float("nan"))
        cc = agg.get("cc", {}).get("mean", float("nan"))
        vs = agg.get("vs", {}).get("mean", float("nan"))
        print(f"{cond:20s} {len(per_task):>6d} {mpd:>8.4f} {cc:>8.1f} {vs:>8.2f}")

    print(f"\nSaved to {out_root}")


if __name__ == "__main__":
    main()
