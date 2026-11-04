"""Analysis: load responses or extracted statements → embed → compute metrics."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .embedding import embed_texts
from .metrics import compute_metrics, rarefied_metrics, mean_pairwise_distance


def _load_responses(jsonl_path: Path) -> list[str]:
    """Extract response contents from one raw JSONL file."""
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line)["response"]["content"] for line in f if line.strip()]


def _load_statements(json_path: Path) -> list[str]:
    """Extract opinion statement texts from one extracted JSON/JSONL file."""
    texts = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            texts.extend(record.get("opinions", []))
    return texts


def _aggregate(per_task: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-task metrics: mean / std / min / max for each metric."""
    keys = ["mpd", "cc", "vs"]
    return {
        key: {
            "mean": float(np.mean([t[key] for t in per_task.values()])),
            "std": float(np.std([t[key] for t in per_task.values()])),
            "min": float(np.min([t[key] for t in per_task.values()])),
            "max": float(np.max([t[key] for t in per_task.values()])),
        }
        for key in keys
    }


def analyze_condition(
    condition_dir: Path,
    extracted: bool = False,
) -> dict[str, Any]:
    """Compute diversity metrics for all tasks in a condition directory.

    Args:
        condition_dir: path to raw (.jsonl) or extracted (.jsonl) files
        extracted: if True, read opinion statements instead of raw responses
    """
    loader = _load_statements if extracted else _load_responses

    per_task: dict[str, dict] = {}
    for f in sorted(condition_dir.glob("*.jsonl")):
        if f.name.startswith("_"):
            continue
        texts = loader(f)
        if not texts:
            continue
        embeddings = embed_texts(texts)
        per_task[f.stem] = compute_metrics(embeddings)

    return {
        "condition": condition_dir.name,
        "n_tasks": len(per_task),
        "aggregate": _aggregate(per_task) if per_task else {},
        "per_task": per_task,
    }


def analyze_condition_rarefied(
    condition_dirs: dict[str, Path],
    extracted: bool = False,
    n_repeats: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compute rarefied CC/VS + raw MPD across multiple conditions.

    For each task, finds the min N across all conditions, then subsamples
    each condition to that N for CC/VS. MPD is computed directly (N-insensitive).

    Args:
        condition_dirs: {condition_name: path_to_dir}
        extracted: if True, read extracted statements
    """
    loader = _load_statements if extracted else _load_responses

    # First pass: load all texts and embed
    all_embeddings: dict[str, dict[str, np.ndarray]] = {}  # cond -> task_id -> emb
    task_ids: set[str] = set()

    for cond_name, cond_dir in condition_dirs.items():
        all_embeddings[cond_name] = {}
        for f in sorted(cond_dir.glob("*.jsonl")):
            if f.name.startswith("_"):
                continue
            texts = loader(f)
            if not texts:
                continue
            all_embeddings[cond_name][f.stem] = embed_texts(texts)
            task_ids.add(f.stem)

    # Second pass: for each task, rarefy to min N
    results: dict[str, dict[str, Any]] = {c: {"per_task": {}} for c in condition_dirs}

    for task_id in sorted(task_ids):
        # Find min N for this task across conditions that have it
        ns = [
            len(all_embeddings[c][task_id])
            for c in condition_dirs
            if task_id in all_embeddings.get(c, {})
        ]
        if not ns:
            continue
        min_n = min(ns)

        for cond_name in condition_dirs:
            emb = all_embeddings.get(cond_name, {}).get(task_id)
            if emb is None:
                continue
            metrics = {"mpd": mean_pairwise_distance(emb)}
            if len(emb) > min_n:
                metrics.update(rarefied_metrics(emb, min_n, n_repeats))
            else:
                from .metrics import cluster_count, vendi_score
                metrics["cc_rarefied"] = {"mean": float(cluster_count(emb)), "std": 0.0}
                metrics["vs_rarefied"] = {"mean": float(vendi_score(emb)), "std": 0.0}
            results[cond_name]["per_task"][task_id] = metrics

    # Aggregate
    for cond_name in condition_dirs:
        pt = results[cond_name]["per_task"]
        if pt:
            results[cond_name]["aggregate"] = {
                "mpd": {
                    "mean": float(np.mean([t["mpd"] for t in pt.values()])),
                    "std": float(np.std([t["mpd"] for t in pt.values()])),
                },
                "cc_rarefied": {
                    "mean": float(np.mean([t["cc_rarefied"]["mean"] for t in pt.values()])),
                    "std": float(np.std([t["cc_rarefied"]["mean"] for t in pt.values()])),
                },
                "vs_rarefied": {
                    "mean": float(np.mean([t["vs_rarefied"]["mean"] for t in pt.values()])),
                    "std": float(np.std([t["vs_rarefied"]["mean"] for t in pt.values()])),
                },
            }
        results[cond_name]["n_tasks"] = len(pt)

    return results
