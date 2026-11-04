"""Extract opinions from raw results for a given condition.

Usage:
    python scripts/extract.py <condition_name>
    python scripts/extract.py <condition_name> --force     # clear and re-extract
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

    condition_name = sys.argv[1]
    force = "--force" in sys.argv

    # Parse --config
    config_name = "extraction.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]

    load_dotenv(PROJECT_ROOT / ".env")
    asyncio.run(extract_condition(
        config_path=PROJECT_ROOT / "configs" / config_name,
        raw_dir=PROJECT_ROOT / "results/raw" / condition_name,
        extracted_dir=PROJECT_ROOT / "results/extracted" / condition_name,
        force=force,
    ))


if __name__ == "__main__":
    main()
