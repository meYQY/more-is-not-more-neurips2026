#!/usr/bin/env python3
"""Post-hoc unit-of-analysis diagnostic for the prompt ablation.

The frozen primary analysis treats every extracted atomic opinion as an
equally weighted item. This script asks whether the Neutral/Agreement/
Disagreement ordering changes when each response, final-round response, or
agent is instead represented by one centroid.

No API calls are made. Only existing extraction and embedding artifacts are
read, and no primary result files are modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from modbench.metrics import mean_pairwise_distance


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("neutral", "agreement", "disagreement")
CONTRASTS = (
    ("agreement_minus_neutral", "agreement", "neutral"),
    ("disagreement_minus_neutral", "disagreement", "neutral"),
    ("disagreement_minus_agreement", "disagreement", "agreement"),
)
METRICS = (
    "atomic_opinion_mpd",
    "r2_r3_response_centroid_mpd",
    "final_round_response_centroid_mpd",
    "agent_centroid_mpd",
    "within_response_mpd",
)
FOCUS_TASKS = ("MOD-085", "MOD-091", "MOD-023", "MOD-064", "MOD-048")
N_BOOTSTRAP = 20_000
BASE_SEED = 1529


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_ids() -> list[str]:
    with (ROOT / "data" / "selected_tasks_100.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return [row["task_id"] for row in csv.DictReader(handle)]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Encountered a non-finite or zero-length centroid")
    return vector / norm


def response_centroid_mpd(groups: list[np.ndarray]) -> float:
    centroids = [normalize(group.mean(axis=0)) for group in groups if len(group)]
    if len(centroids) < 2:
        return float("nan")
    return float(mean_pairwise_distance(np.vstack(centroids)))


def agent_centroid_mpd(groups: list[np.ndarray]) -> float:
    """Pool each persona's R2 and R3 opinions before taking one centroid."""
    centroids: list[np.ndarray] = []
    for persona_index in range(5):
        present = [
            groups[persona_index],
            groups[5 + persona_index],
        ]
        present = [group for group in present if len(group)]
        if present:
            centroids.append(normalize(np.vstack(present).mean(axis=0)))
    if len(centroids) < 2:
        return float("nan")
    return float(mean_pairwise_distance(np.vstack(centroids)))


def within_response_mpd(groups: list[np.ndarray]) -> float:
    values = [
        float(mean_pairwise_distance(group))
        for group in groups
        if len(group) >= 2
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ci(values: np.ndarray, label: str) -> list[float]:
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_and_audit_task(
    condition: str,
    task_id: str,
    audit: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Return ten R2/R3 response groups after structural mapping checks."""
    raw_path = ROOT / "raw" / condition / f"{task_id}.jsonl"
    extracted_path = ROOT / "extracted" / condition / f"{task_id}.jsonl"
    full_path = (
        ROOT
        / "embeddings"
        / "openai-3-small"
        / condition
        / f"{task_id}.npy"
    )
    post_path = (
        ROOT
        / "embeddings_post"
        / "openai-3-small"
        / condition
        / f"{task_id}.npy"
    )
    raw = read_jsonl(raw_path)
    extracted = read_jsonl(extracted_path)
    full = np.load(full_path)
    post = np.load(post_path)

    if len(raw) != 15 or len(extracted) != 15:
        raise RuntimeError(
            f"{condition}/{task_id}: expected 15 raw and extracted records"
        )
    expected_rounds = [1] * 5 + [2] * 5 + [3] * 5
    full_rounds: list[int] = []
    full_counts: list[int] = []
    for index, (raw_record, extracted_record, expected_round) in enumerate(
        zip(raw, extracted, expected_rounds)
    ):
        raw_round = int(raw_record["round_number"])
        extracted_round = int(extracted_record["round_number"])
        if raw_round != expected_round or extracted_round != expected_round:
            raise RuntimeError(
                f"{condition}/{task_id}: malformed round order at row {index}"
            )
        if int(extracted_record["call_index"]) != index:
            raise RuntimeError(
                f"{condition}/{task_id}: call_index mismatch at row {index}"
            )
        for key in ("task_id", "persona_name", "raw_question"):
            if raw_record[key] != extracted_record[key]:
                raise RuntimeError(
                    f"{condition}/{task_id}: {key} mismatch at row {index}"
                )
        opinions = [
            opinion
            for opinion in extracted_record.get("opinions", [])
            if opinion.strip()
        ]
        full_counts.append(len(opinions))
        full_rounds.extend([expected_round] * len(opinions))

    if full.ndim != 2 or post.ndim != 2 or full.shape[1:] != (1536,):
        raise RuntimeError(
            f"{condition}/{task_id}: unexpected embedding dimensions "
            f"{full.shape} / {post.shape}"
        )
    if post.shape[1:] != (1536,):
        raise RuntimeError(
            f"{condition}/{task_id}: unexpected post dimensions {post.shape}"
        )
    if len(full) != sum(full_counts):
        raise RuntimeError(
            f"{condition}/{task_id}: full rows={len(full)}, "
            f"opinions={sum(full_counts)}"
        )
    post_indices = np.asarray(
        [index for index, round_number in enumerate(full_rounds) if round_number >= 2]
    )
    if len(post) != len(post_indices):
        raise RuntimeError(
            f"{condition}/{task_id}: post rows={len(post)}, "
            f"expected={len(post_indices)}"
        )
    if not np.array_equal(post, full[post_indices]):
        raise RuntimeError(
            f"{condition}/{task_id}: post embeddings are not the exact "
            "R2/R3 subset of full embeddings"
        )
    if not np.isfinite(full).all() or not np.isfinite(post).all():
        raise RuntimeError(f"{condition}/{task_id}: non-finite embedding value")

    groups: list[np.ndarray] = []
    offset = 0
    for count in full_counts[5:]:
        groups.append(post[offset : offset + count])
        offset += count
    if offset != len(post) or len(groups) != 10:
        raise RuntimeError(
            f"{condition}/{task_id}: response grouping did not exhaust post rows"
        )

    audit["task_condition_pairs_checked"] += 1
    audit["raw_records_checked"] += len(raw)
    audit["extracted_records_checked"] += len(extracted)
    audit["full_embedding_rows_checked"] += len(full)
    audit["post_embedding_rows_checked"] += len(post)
    audit["exact_post_subset_checks_passed"] += 1
    audit["zero_opinion_r2_r3_responses"] += sum(len(group) == 0 for group in groups)
    audit["one_opinion_r2_r3_responses"] += sum(len(group) == 1 for group in groups)
    return groups, {
        "all_round_opinions": len(full),
        "r2_r3_opinions": len(post),
        "r2_r3_responses_with_2plus_opinions": sum(
            len(group) >= 2 for group in groups
        ),
    }


def summarize_contrast(
    per_task: dict[str, dict[str, dict[str, float]]],
    metric: str,
    label: str,
    left: str,
    right: str,
    ids: list[str],
) -> dict[str, Any]:
    left_values = np.asarray(
        [per_task[left][task_id][metric] for task_id in ids], dtype=float
    )
    right_values = np.asarray(
        [per_task[right][task_id][metric] for task_id in ids], dtype=float
    )
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[finite]
    right_values = right_values[finite]
    valid_ids = [task_id for task_id, keep in zip(ids, finite) if keep]
    differences = left_values - right_values
    if not len(differences):
        raise RuntimeError(f"No finite paired tasks for {metric}/{label}")
    return {
        "left": left,
        "right": right,
        "n_paired_tasks": len(differences),
        "left_paired_mean": float(left_values.mean()),
        "right_paired_mean": float(right_values.mean()),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": bootstrap_ci(differences, f"{metric}:{label}"),
        "wilcoxon_p": (
            1.0
            if np.allclose(differences, 0)
            else float(wilcoxon(differences).pvalue)
        ),
        "left_wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "right_wins": int(np.sum(differences < 0)),
        "per_task_difference": dict(zip(valid_ids, differences.tolist())),
    }


def fmt(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main() -> None:
    ids = task_ids()
    if len(ids) != 100 or len(set(ids)) != 100:
        raise RuntimeError("Expected 100 unique selected task IDs")

    audit: dict[str, Any] = {
        "mapping_status": "pending",
        "task_condition_pairs_checked": 0,
        "raw_records_checked": 0,
        "extracted_records_checked": 0,
        "full_embedding_rows_checked": 0,
        "post_embedding_rows_checked": 0,
        "exact_post_subset_checks_passed": 0,
        "zero_opinion_r2_r3_responses": 0,
        "one_opinion_r2_r3_responses": 0,
        "checks": [
            "15 raw and 15 extracted rows per task-condition",
            "call_index 0..14 and rounds 1/2/3 in 5-row blocks",
            "task, persona, question aligned between raw and extracted rows",
            "full embedding rows equal non-empty extracted opinion count",
            "post embedding array exactly equals the R2/R3 subset of full array",
            "post rows exhaustively regrouped by extracted response opinion counts",
            "all embedding values finite and dimensionality 1536",
        ],
    }
    per_task: dict[str, dict[str, dict[str, float]]] = {
        condition: {} for condition in CONDITIONS
    }
    counts: dict[str, dict[str, dict[str, int]]] = {
        condition: {} for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for task_id in ids:
            groups, task_counts = load_and_audit_task(
                condition, task_id, audit
            )
            per_task[condition][task_id] = {
                "atomic_opinion_mpd": float(
                    mean_pairwise_distance(
                        np.vstack([group for group in groups if len(group)])
                    )
                ),
                "r2_r3_response_centroid_mpd": response_centroid_mpd(groups),
                "final_round_response_centroid_mpd": response_centroid_mpd(
                    groups[5:]
                ),
                "agent_centroid_mpd": agent_centroid_mpd(groups),
                "within_response_mpd": within_response_mpd(groups),
                "mean_opinions_per_response": float(
                    np.mean([len(group) for group in groups])
                ),
            }
            counts[condition][task_id] = task_counts
    audit["mapping_status"] = "passed"

    aggregates = {
        condition: {
            metric: float(
                np.nanmean(
                    [
                        per_task[condition][task_id][metric]
                        for task_id in ids
                    ]
                )
            )
            for metric in METRICS + ("mean_opinions_per_response",)
        }
        for condition in CONDITIONS
    }
    contrasts = {
        metric: {
            label: summarize_contrast(
                per_task, metric, label, left, right, ids
            )
            for label, left, right in CONTRASTS
        }
        for metric in METRICS + ("mean_opinions_per_response",)
    }

    focus = {
        task_id: {
            "condition_values": {
                condition: per_task[condition][task_id]
                for condition in CONDITIONS
            },
            "disagreement_minus_neutral": {
                metric: (
                    per_task["disagreement"][task_id][metric]
                    - per_task["neutral"][task_id][metric]
                )
                for metric in METRICS + ("mean_opinions_per_response",)
            },
            "agreement_minus_neutral": {
                metric: (
                    per_task["agreement"][task_id][metric]
                    - per_task["neutral"][task_id][metric]
                )
                for metric in METRICS + ("mean_opinions_per_response",)
            },
        }
        for task_id in FOCUS_TASKS
    }
    focus_alignment = {
        "reference": (
            "The preceding metric-blind model-agent read judged each focus "
            "task to contain substantive additions under Disagreement versus "
            "Neutral. This is an internal qualitative diagnostic, not human "
            "annotation."
        ),
        "n_focus_tasks": len(FOCUS_TASKS),
        "positive_disagreement_minus_neutral": {
            metric: int(
                sum(
                    focus[task_id]["disagreement_minus_neutral"][metric] > 0
                    for task_id in FOCUS_TASKS
                )
            )
            for metric in METRICS
        },
    }
    result = {
        "created_from_existing_artifacts": True,
        "api_calls": 0,
        "status": "post_hoc_unit_of_analysis_diagnostic_not_primary_evidence",
        "n_tasks": len(ids),
        "conditions": list(CONDITIONS),
        "n_task_bootstrap": N_BOOTSTRAP,
        "definitions": {
            "atomic_opinion_mpd": (
                "Primary-style MPD over every R2/R3 atomic opinion."
            ),
            "r2_r3_response_centroid_mpd": (
                "Each of the ten R2/R3 responses gets equal weight via the "
                "normalized centroid of its atomic-opinion embeddings."
            ),
            "final_round_response_centroid_mpd": (
                "MPD over the five normalized R3 response centroids."
            ),
            "agent_centroid_mpd": (
                "For each persona, pool all R2/R3 atomic opinions and take one "
                "normalized centroid; then compute MPD across five personas."
            ),
            "within_response_mpd": (
                "Mean of response-level atomic-opinion MPDs over R2/R3 "
                "responses containing at least two opinions."
            ),
            "mean_opinions_per_response": (
                "Mean number of extracted atomic opinions per R2/R3 response."
            ),
        },
        "mapping_audit": audit,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "focus_tasks": focus,
        "focus_alignment_summary": focus_alignment,
        "per_task": per_task,
        "per_task_counts": counts,
    }
    output = ROOT / "analysis" / "prompt_unit_of_analysis_100.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_labels = {
        "atomic_opinion_mpd": "Atomic-opinion MPD (R2–3)",
        "r2_r3_response_centroid_mpd": "Response-centroid MPD (R2–3)",
        "final_round_response_centroid_mpd": "Final-round response-centroid MPD",
        "agent_centroid_mpd": "Agent-centroid MPD",
        "within_response_mpd": "Within-response MPD",
    }
    lines = [
        "# Prompt Ablation: Unit-of-Analysis Diagnostic",
        "",
        "Post-hoc diagnostic only. It uses the existing extracted atomic opinions",
        "and `text-embedding-3-small` arrays; it made no API calls and does not",
        "replace the frozen primary opinion-unit MPD analysis.",
        "",
        "## Mapping audit",
        "",
        f"- Status: **{audit['mapping_status']}**.",
        f"- Checked {audit['task_condition_pairs_checked']} task-condition pairs, "
        f"{audit['extracted_records_checked']:,} extracted response rows, "
        f"{audit['full_embedding_rows_checked']:,} full embedding rows, and "
        f"{audit['post_embedding_rows_checked']:,} R2/R3 embedding rows.",
        f"- All {audit['exact_post_subset_checks_passed']} post arrays were exact "
        "row-preserving R2/R3 subsets of their full arrays.",
        f"- R2/R3 response groups with zero / one extracted opinion: "
        f"{audit['zero_opinion_r2_r3_responses']} / "
        f"{audit['one_opinion_r2_r3_responses']}.",
        "",
        "## Condition means",
        "",
        "| Metric | Neutral | Agreement | Disagreement |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric_labels[metric]} | "
            f"{aggregates['neutral'][metric]:.4f} | "
            f"{aggregates['agreement'][metric]:.4f} | "
            f"{aggregates['disagreement'][metric]:.4f} |"
        )
    lines.extend(
        [
            f"| Extracted opinions per R2/R3 response | "
            f"{aggregates['neutral']['mean_opinions_per_response']:.2f} | "
            f"{aggregates['agreement']['mean_opinions_per_response']:.2f} | "
            f"{aggregates['disagreement']['mean_opinions_per_response']:.2f} |",
            "",
            "## Paired contrasts",
            "",
        ]
    )
    for label, left, right in (
        ("Disagreement − Neutral", "disagreement", "neutral"),
        ("Agreement − Neutral", "agreement", "neutral"),
        ("Disagreement − Agreement", "disagreement", "agreement"),
    ):
        contrast_key = f"{left}_minus_{right}"
        lines.extend(
            [
                f"### {label}",
                "",
                "| Metric | Paired tasks | Left | Right | Mean Δ | "
                "95% task-bootstrap CI | Wins / Ties / Losses |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            row = contrasts[metric][contrast_key]
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {metric_labels[metric]} | {row['n_paired_tasks']} | "
                f"{row['left_paired_mean']:.4f} | "
                f"{row['right_paired_mean']:.4f} | "
                f"{row['mean_difference']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{row['left_wins']} / {row['ties']} / "
                f"{row['right_wins']} |"
            )
        count_row = contrasts["mean_opinions_per_response"][contrast_key]
        count_ci = count_row["bootstrap_95_ci"]
        lines.append(
            f"| Extracted opinions per response | "
            f"{count_row['n_paired_tasks']} | "
            f"{count_row['left_paired_mean']:.2f} | "
            f"{count_row['right_paired_mean']:.2f} | "
            f"{count_row['mean_difference']:+.2f} | "
            f"[{count_ci[0]:+.2f}, {count_ci[1]:+.2f}] | "
            f"{count_row['left_wins']} / {count_row['ties']} / "
            f"{count_row['right_wins']} |"
        )
        lines.append("")

    lines.extend(
        [
            "## Focus tasks: Disagreement − Neutral",
            "",
            "| Task | Atomic MPD | Response centroids | Final-round centroids | "
            "Agent centroids | Within-response MPD | Opinions/response |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task_id in FOCUS_TASKS:
        values = focus[task_id]["disagreement_minus_neutral"]
        lines.append(
            f"| {task_id} | {fmt(values['atomic_opinion_mpd'], True)} | "
            f"{fmt(values['r2_r3_response_centroid_mpd'], True)} | "
            f"{fmt(values['final_round_response_centroid_mpd'], True)} | "
            f"{fmt(values['agent_centroid_mpd'], True)} | "
            f"{fmt(values['within_response_mpd'], True)} | "
            f"{values['mean_opinions_per_response']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "The preceding metric-blind model-agent inspection judged all five",
            "focus tasks to contain substantive additions under Disagreement",
            "relative to Neutral. Atomic MPD was positive on 2/5 of these tasks;",
            "response-, final-round-, and agent-centroid MPD were each positive",
            "on 0/5. Thus centroiding does not reconcile the automated scores",
            "with that qualitative read; it makes the mismatch larger.",
            "",
            "## What this diagnostic indicates",
            "",
            "- Disagreement produced 4.88 more extracted opinions per R2/R3",
            "  response than Neutral, but their mean within-response dispersion",
            "  was nearly identical (Δ = -0.0006).",
            "- Giving every response equal weight did not recover a positive",
            "  Disagreement effect. Response-centroid MPD was much lower than",
            "  Neutral (Δ = -0.0934), as were final-round and agent centroids.",
            "- Disagreement exceeded Agreement on atomic MPD, but not reliably on",
            "  response-, final-round-, or agent-centroid MPD. Its reliable",
            "  advantage over Agreement appears inside responses: agents surface",
            "  counterarguments, while different agents often cover a similar",
            "  multi-sided argument set.",
            "- Agreement remained below Neutral across every between-unit view.",
            "  The visible Agreement convergence is therefore not an artifact of",
            "  weighting verbose responses by their number of extracted claims.",
            "",
            "## Interpretation boundary",
            "",
            "- Response and agent centroids remove or reduce the primary analysis's",
            "  implicit extra weight for responses that yield more atomic opinions.",
            "- The observed ordering does not support the simple hypothesis that",
            "  verbosity/supporting-claim weighting alone suppressed a genuinely",
            "  higher Disagreement score. It instead exposes a distinction between",
            "  within-response viewpoint coverage and between-agent stance",
            "  separation.",
            "- These results are post-hoc, embedding-based diagnostics. They are not",
            "  human verification and should not be presented as a new primary",
            "  endpoint without an explicit scope statement.",
            "- For rebuttal, this is useful as an internal guardrail. It supports",
            "  the Agreement-convergence claim, but it is not positive evidence",
            "  that MPD captures human-perceived diversity, and the centroid",
            "  endpoints are not worth foregrounding as a rebuttal result.",
            "",
        ]
    )
    report = ROOT / "analysis" / "prompt_unit_of_analysis_report_100.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
