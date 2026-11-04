"""Experiment runner: dispatcher that routes to framework-specific runners."""

from pathlib import Path

from .io import load_config, load_tasks
from .runners import independent, sequential, interactive

RUNNERS = {
    "independent": independent,
    "sequential": sequential,
    "interactive": interactive,
}


async def run_condition(
    base_config_path: Path,
    condition_path: Path,
    tasks_path: Path,
    output_dir: Path,
    n_tasks: int | None = None,
) -> None:
    """Load config and dispatch to the appropriate framework runner."""
    config = load_config(base_config_path, condition_path)
    condition = config["condition"]
    tasks = load_tasks(tasks_path)
    tasks = tasks[:n_tasks]

    # Allow condition to override generation params (e.g. TEMP-HIGH)
    if "generation" in condition:
        config["generation"].update(condition["generation"])

    framework = condition["framework"]
    if framework not in RUNNERS:
        raise ValueError(f"Unknown framework: {framework}")

    await RUNNERS[framework].run(tasks, config, output_dir, base_config_path)
