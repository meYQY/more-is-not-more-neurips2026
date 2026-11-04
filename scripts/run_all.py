"""Run all (or a subset of) experiment conditions.

Usage: python scripts/run_all.py [--n-tasks N] [--conditions c1,c2,...] [--parallel N]
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS_DIR = PROJECT_ROOT / "configs" / "conditions"
LOG_DIR = PROJECT_ROOT / "logs"
RUN_EXPERIMENT = PROJECT_ROOT / "scripts" / "run_experiment.py"


def discover_conditions() -> dict[str, Path]:
    """Return {stem: path} for all condition YAMLs under CONDITIONS_DIR."""
    return {p.stem: p for p in sorted(CONDITIONS_DIR.rglob("*.yaml"))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N tasks per condition (default: all)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        metavar="c1,c2,...",
        help="Comma-separated list of condition names to run (default: all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of conditions to run in parallel (default: 1)",
    )
    return parser.parse_args()


def run_one(
    condition_name: str,
    condition_path: Path,
    n_tasks: int | None,
    timestamp: str,
) -> bool:
    """Run a single condition via subprocess. Returns True on success."""
    log_path = LOG_DIR / f"{condition_name}_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(RUN_EXPERIMENT), str(condition_path)]
    if n_tasks is not None:
        cmd += ["--n-tasks", str(n_tasks)]

    print(f"[run_all] Starting: {condition_name}")
    # run_experiment.py handles its own logging via TeeStream,
    # so we just inherit stdout/stderr here to avoid duplicate logs.
    result = subprocess.run(cmd)

    log_path = sorted(LOG_DIR.glob(f"{condition_name}_*.log"))[-1] if LOG_DIR.exists() else None
    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit {result.returncode})"
    print(f"[run_all] {condition_name}: {status}  -> {log_path}")
    return success


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_conditions = discover_conditions()

    if args.conditions:
        requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in requested if c not in all_conditions]
        if unknown:
            print(f"[run_all] Unknown conditions: {unknown}", file=sys.stderr)
            print(f"[run_all] Available: {sorted(all_conditions)}", file=sys.stderr)
            sys.exit(1)
        selected = {c: all_conditions[c] for c in requested}
    else:
        selected = all_conditions

    print(f"[run_all] Running {len(selected)} condition(s) with parallel={args.parallel}: {list(selected)}")
    if args.n_tasks is not None:
        print(f"[run_all] n_tasks={args.n_tasks} (pilot mode)")

    overall_log = LOG_DIR / f"run_all_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    with open(overall_log, "w", encoding="utf-8") as overall:
        overall.write(f"run_all started at {timestamp}\n")
        overall.write(f"conditions: {list(selected)}\n")
        overall.write(f"parallel: {args.parallel}\n")
        if args.n_tasks is not None:
            overall.write(f"n_tasks: {args.n_tasks}\n")
        overall.write("\n")
        overall.flush()

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_one, name, path, args.n_tasks, timestamp): name
                for name, path in selected.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                success = future.result()
                results[name] = success
                overall.write(f"{name}: {'OK' if success else 'FAILED'}\n")
                overall.flush()

    # Print summary
    succeeded = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if not ok]

    print("\n" + "=" * 50)
    print(f"[run_all] Summary: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print(f"[run_all] Failed: {failed}")
    print(f"[run_all] Overall log: {overall_log}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
