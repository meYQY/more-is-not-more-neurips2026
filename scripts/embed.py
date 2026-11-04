"""Step 2.5: Embed extracted opinions.

Reads extracted opinions, calls embedding API, saves .npy files.
Run once after extraction. Subsequent steps only read the .npy files.

Usage:
    python scripts/embed.py --embedder openai-3-small
    python scripts/embed.py --embedder openai-3-small --conditions single,persona-pro
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from modbench.embedding import embed_texts, EMBEDDER_DIMS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", required=True, choices=list(EMBEDDER_DIMS))
    p.add_argument("--conditions", default=None, help="Comma-separated (default: all)")
    p.add_argument("--force", action="store_true", help="Re-embed even if .npy exists")
    p.add_argument("--extracted-dir", default=None, help="Override extracted dir (default: results/extracted)")
    p.add_argument("--embeddings-dir", default=None, help="Override embeddings dir (default: results/embeddings)")
    return p.parse_args()


def load_opinions(extracted_root: Path, condition: str, task_id: str) -> list[str]:
    """Load opinion texts from one extracted JSONL file."""
    f = extracted_root / condition / f"{task_id}.jsonl"
    opinions = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            opinions.extend(rec.get("opinions", []))
    return [o for o in opinions if o.strip()]


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    extracted_root = Path(args.extracted_dir) if args.extracted_dir else PROJECT_ROOT / "results" / "extracted"
    embeddings_root = Path(args.embeddings_dir) if args.embeddings_dir else PROJECT_ROOT / "results" / "embeddings"

    embedder = args.embedder
    conditions = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions
        else sorted(d.name for d in extracted_root.iterdir() if d.is_dir())
    )

    print(f"Embedder: {embedder} (dim={EMBEDDER_DIMS[embedder]})")
    print(f"Extracted: {extracted_root}")
    print(f"Embeddings: {embeddings_root}")
    print(f"Conditions: {len(conditions)}")
    t0 = time.monotonic()
    total_new = 0

    for cond in conditions:
        out_dir = embeddings_root / embedder / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = sorted(f.stem for f in (extracted_root / cond).glob("MOD-*.jsonl"))
        n_skip = n_new = n_err = 0

        for tid in tasks:
            npy_path = out_dir / f"{tid}.npy"
            if npy_path.exists() and not args.force:
                n_skip += 1
                continue

            opinions = load_opinions(extracted_root, cond, tid)
            if not opinions:
                continue

            for attempt in range(3):
                try:
                    emb = embed_texts(opinions, embedder=embedder)
                    np.save(npy_path, emb.astype(np.float32))
                    n_new += 1
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"    ERROR {cond}/{tid}: {e}")
                        n_err += 1

        total_new += n_new
        print(f"  {cond}: {n_new} new, {n_skip} skipped, {n_err} errors")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s. {total_new} files created.")
    print(f"Embeddings: {embeddings_root / embedder}")


if __name__ == "__main__":
    main()
