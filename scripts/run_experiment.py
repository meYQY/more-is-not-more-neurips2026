"""Run experiment for a given condition.

Usage: python scripts/run_experiment.py <condition_yaml> [--n-tasks N]
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from modbench.runner import run_condition

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TeeStream:
    """Write to both a stream and a file simultaneously."""

    def __init__(self, stream, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = stream
        self._file = open(file_path, "a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    # Forward everything else (e.g. fileno, isatty) to the real stream.
    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition_yaml", type=Path, help="Path to condition YAML file")
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N tasks (default: all)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base config YAML (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for raw results (default: results/raw)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    condition_name = args.condition_yaml.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"{condition_name}_{timestamp}.log"

    tee_out = TeeStream(sys.stdout, log_path)
    tee_err = TeeStream(sys.stderr, log_path)
    sys.stdout = tee_out  # type: ignore[assignment]
    sys.stderr = tee_err  # type: ignore[assignment]

    try:
        load_dotenv(PROJECT_ROOT / ".env")
        base_config = args.base_config or (PROJECT_ROOT / "configs/base.yaml")
        output_dir = args.output_dir or (PROJECT_ROOT / "results/raw")
        asyncio.run(run_condition(
            base_config_path=base_config,
            condition_path=args.condition_yaml,
            tasks_path=PROJECT_ROOT / "data/dataset/tasks.csv",
            output_dir=output_dir,
            n_tasks=args.n_tasks,
        ))
    finally:
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()


if __name__ == "__main__":
    main()
