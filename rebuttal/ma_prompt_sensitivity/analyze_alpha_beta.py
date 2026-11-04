#!/usr/bin/env python3
"""Compute within-condition alpha and cross-condition beta diversity.

Alpha metrics are read from the completed 100-task metric artifact. Beta metrics
use equal-size, without-replacement rarefaction for each task and condition pair.
A shuffled-label null is computed on the same balanced union so that observed
cross-condition separation can be distinguished from random split variability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from modbench.metrics import unique_cluster_ratio, vendi_score


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("neutral", "agreement", "disagreement")
PAIRS = (
    ("neutral", "agreement"),
    ("neutral", "disagreement"),
    ("agreement", "disagreement"),
)
VIEWS = {
    "all_rounds": ROOT / "embeddings" / "openai-3-small",
    "rounds_2_3": ROOT / "embeddings_post" / "openai-3-small",
}
N_REPEATS = 30
N_BOOTSTRAP = 20_000
BASE_SEED = 1529


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode()).digest()
    return BASE_SEED + int.from_bytes(digest[:4], "big")


def beta_vs_equal(a: np.ndarray, b: np.ndarray) -> float:
    union = np.vstack([a, b])
    return float(vendi_score(union) - 0.5 * vendi_score(a) - 0.5 * vendi_score(b))


def bootstrap_ci(values: np.ndarray, label: str) -> list[float]:
    rng = np.random.default_rng(stable_seed("bootstrap", label))
    draws = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    means = draws.mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def task_beta(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    n = min(len(emb_a), len(emb_b))
    if n < 4:
        raise ValueError("Beta diversity requires at least four opinions per set.")

    rng = np.random.default_rng(seed)
    observed_beta: list[float] = []
    observed_ucr_a: list[float] = []
    observed_ucr_b: list[float] = []
    null_beta: list[float] = []
    null_ucr: list[float] = []

    for _ in range(N_REPEATS):
        a = emb_a[rng.choice(len(emb_a), size=n, replace=False)]
        b = emb_b[rng.choice(len(emb_b), size=n, replace=False)]

        observed_beta.append(beta_vs_equal(a, b))
        ucr_a, ucr_b = unique_cluster_ratio(a, b)
        observed_ucr_a.append(ucr_a)
        observed_ucr_b.append(ucr_b)

        union = np.vstack([a, b])
        permutation = rng.permutation(2 * n)
        null_a = union[permutation[:n]]
        null_b = union[permutation[n:]]
        null_beta.append(beta_vs_equal(null_a, null_b))
        null_a_ucr, null_b_ucr = unique_cluster_ratio(null_a, null_b)
        null_ucr.append(0.5 * (null_a_ucr + null_b_ucr))

    beta = float(np.mean(observed_beta))
    ucr_a = float(np.mean(observed_ucr_a))
    ucr_b = float(np.mean(observed_ucr_b))
    ucr_mean = 0.5 * (ucr_a + ucr_b)
    beta_null = float(np.mean(null_beta))
    ucr_null = float(np.mean(null_ucr))
    return {
        "n_rarefied": n,
        "beta_vs": beta,
        "unique_a_ratio": ucr_a,
        "unique_b_ratio": ucr_b,
        "unique_ratio_mean": ucr_mean,
        "shuffled_beta_vs": beta_null,
        "shuffled_unique_ratio": ucr_null,
        "beta_vs_above_shuffle": beta - beta_null,
        "unique_ratio_above_shuffle": ucr_mean - ucr_null,
    }


def summarize(per_task: dict[str, dict[str, float]], label: str) -> dict[str, Any]:
    keys = (
        "n_rarefied",
        "beta_vs",
        "unique_a_ratio",
        "unique_b_ratio",
        "unique_ratio_mean",
        "shuffled_beta_vs",
        "shuffled_unique_ratio",
        "beta_vs_above_shuffle",
        "unique_ratio_above_shuffle",
    )
    result: dict[str, Any] = {"n_tasks": len(per_task)}
    for key in keys:
        values = np.asarray([row[key] for row in per_task.values()], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "bootstrap_95_ci": bootstrap_ci(values, f"{label}:{key}"),
        }
    return result


def main() -> None:
    alpha = json.loads((ROOT / "metrics" / "metrics_100.json").read_text())
    task_ids = sorted(alpha["views"]["all_rounds"]["per_task"]["neutral"])
    result: dict[str, Any] = {
        "n_tasks": len(task_ids),
        "n_repeats": N_REPEATS,
        "embedding_model": "text-embedding-3-small",
        "alpha": {
            view: alpha["views"][view]["aggregate"]
            for view in VIEWS
        },
        "beta": {},
    }

    for view, embedding_root in VIEWS.items():
        result["beta"][view] = {}
        for condition_a, condition_b in PAIRS:
            label = f"{condition_a}_vs_{condition_b}"
            print(f"{view}: {label}", flush=True)
            per_task: dict[str, dict[str, float]] = {}
            for index, task_id in enumerate(task_ids, start=1):
                emb_a = np.load(
                    embedding_root / condition_a / f"{task_id}.npy"
                )
                emb_b = np.load(
                    embedding_root / condition_b / f"{task_id}.npy"
                )
                per_task[task_id] = task_beta(
                    emb_a,
                    emb_b,
                    seed=stable_seed(view, label, task_id),
                )
                if index % 10 == 0:
                    print(f"  {index}/{len(task_ids)}", flush=True)
            result["beta"][view][label] = {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "aggregate": summarize(per_task, f"{view}:{label}"),
                "per_task": per_task,
            }

    out_dir = ROOT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "alpha_beta_100.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Alpha and Beta Diversity: 100-Task Prompt Ablation",
        "",
        "Alpha diversity measures within-condition spread. Beta diversity measures",
        "how much two conditions occupy different regions of opinion space.",
        "Cross-condition sets are rarefied to equal size without replacement over",
        f"{N_REPEATS} repeats. The shuffled-label baseline uses the same balanced union.",
        "",
    ]
    for view, title in (
        ("all_rounds", "All rounds"),
        ("rounds_2_3", "Rounds 2–3"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "### Alpha diversity",
                "",
                "| Condition | Mean opinions | MPD | Rarefied CC | Rarefied VS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            row = result["alpha"][view][condition]
            lines.append(
                f"| {condition} | {row['n']:.1f} | {row['mpd']:.4f} | "
                f"{row['cc_rarefied']:.2f} | {row['vs_rarefied']:.2f} |"
            )
        lines.extend(
            [
                "",
                "### Beta diversity",
                "",
                "| Pair | β-VS | β-VS above shuffle | Unique A | Unique B | Mean UCR above shuffle |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, pair in result["beta"][view].items():
            aggregate = pair["aggregate"]
            lines.append(
                f"| {pair['condition_a']} vs {pair['condition_b']} | "
                f"{aggregate['beta_vs']['mean']:.3f} | "
                f"{aggregate['beta_vs_above_shuffle']['mean']:+.3f} | "
                f"{aggregate['unique_a_ratio']['mean']:.1%} | "
                f"{aggregate['unique_b_ratio']['mean']:.1%} | "
                f"{aggregate['unique_ratio_above_shuffle']['mean']:+.1%} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "Beta diversity is symmetric separation/novelty, not a ranking of which",
            "condition is intrinsically more diverse. Directional UCR indicates which",
            "condition contains clusters not recovered by the other condition.",
            "",
        ]
    )
    report_path = out_dir / "alpha_beta_report_100.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
