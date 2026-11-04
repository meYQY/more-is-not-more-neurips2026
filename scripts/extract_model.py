"""Extract opinions for a non-default model (reads raw_{model}, writes extracted_{model}).

Usage:
    python scripts/extract_model.py <model_suffix> [--conditions c1,c2,...] [--force]

Examples:
    python scripts/extract_model.py grok
    python scripts/extract_model.py llama --force
    python scripts/extract_model.py kimi --conditions interactive-basic,interactive-pro
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from modbench.extractor import extract_condition

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_suffix = sys.argv[1]
    force = "--force" in sys.argv

    # Parse --conditions
    conditions = None
    for i, arg in enumerate(sys.argv):
        if arg == "--conditions" and i + 1 < len(sys.argv):
            conditions = [c.strip() for c in sys.argv[i + 1].split(",")]

    load_dotenv(PROJECT_ROOT / ".env")

    raw_root = PROJECT_ROOT / "results" / f"raw_{model_suffix}"
    extracted_root = PROJECT_ROOT / "results" / f"extracted_{model_suffix}"
    # Parse --config
    config_name = "extraction.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]
    config_path = PROJECT_ROOT / "configs" / config_name

    if not raw_root.exists():
        print(f"No raw dir: {raw_root}")
        sys.exit(1)

    if conditions is None:
        conditions = sorted(d.name for d in raw_root.iterdir() if d.is_dir())

    print(f"Model: {model_suffix}")
    print(f"Raw: {raw_root}")
    print(f"Extracted: {extracted_root}")
    print(f"Conditions: {len(conditions)}")
    print(f"Force: {force}")
    print()

    for cond in conditions:
        raw_dir = raw_root / cond
        extracted_dir = extracted_root / cond
        if not raw_dir.exists():
            print(f"  SKIP {cond}: no raw dir")
            continue
        print(f"  {cond}...", flush=True)
        try:
            asyncio.run(extract_condition(
                config_path=config_path,
                raw_dir=raw_dir,
                extracted_dir=extracted_dir,
                force=force,
            ))
        except Exception as e:
            print(f"  ERROR {cond}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
