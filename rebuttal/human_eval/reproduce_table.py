#!/usr/bin/env python3
"""Recompute the human-evaluation table (Appendix J) from the released data.

Inputs (both in this folder):
  data/study_a_judgments_deidentified.csv   one row per judgment (168)
  study_a/item_manifest.csv          one row per comparison item (84)

The statistics reuse the frozen analyzer in analysis/analyze_annotations.py
(same functions, iteration counts, and seeds as the original run), so the
output should match results/final_human_summary.json exactly. The script uses
only the Python standard library, makes no network calls, and writes nothing.

Run from the repository root:
  python rebuttal/human_eval/reproduce_table.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "analysis"))

from analyze_annotations import (  # noqa: E402
    aggregate_a_items,
    bootstrap_contrast_by_participant,
    bootstrap_contrast_by_triplet,
    nominal_agreement,
    study_a_length_diagnostics,
)

SEED = 20_260_727  # analyze_annotations.py default --seed
ITERATIONS = 10_000  # analyze_annotations.py default --bootstrap-iterations
CONTRASTS = ("role_basic", "role_pro")  # sorted order used by the analyzer


def _int_or_none(value: str) -> int | None:
    return int(value) if value != "" else None


def load_judgments() -> list[dict]:
    with (HERE / "study_a" / "item_manifest.csv").open(encoding="utf-8") as handle:
        items = {row["item_id"]: row for row in csv.DictReader(handle)}
    with (HERE / "data" / "study_a_judgments_deidentified.csv").open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    judgments = []
    for row in raw:
        item = items[row["item_id"]]
        judgments.append(
            {
                "sample": "all_paid",
                "judgment_id": row["judgment_id"],
                "phase": row["phase"],
                # The analyzer clusters participants on this key; here it holds
                # the de-identified annotator index.
                "prolific_pid": row["annotator"],
                "item_id": row["item_id"],
                "triplet_id": row["triplet_id"],
                "contrast": row["contrast"],
                "other_condition": row["other_condition"],
                "model": row["model"],
                "task_id": row["task_id"],
                "subcategory": item["subcategory"],
                "role_position": row["role_position"],
                "code": row["choice"],
                "contrast_score": int(row["contrast_score"]),
                "confidence": int(row["confidence"]),
                "role_words": _int_or_none(item["role_words"]),
                "basic_words": _int_or_none(item["basic_words"]),
                "pro_words": _int_or_none(item["pro_words"]),
                "other_words": _int_or_none(item["other_words"]),
                "signed_log_length_ratio": float(item["signed_log_length_ratio"]),
                "max_min_length_ratio": float(item["max_min_length_ratio"]),
                "length_matched_1_5x": item["length_matched_1_5x"] == "True",
            }
        )
    return judgments


def exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def position_mcnemar(rows: list[dict]) -> tuple[int, int, float]:
    """Paired exact McNemar on 'non-Role response chosen', other-on-A vs other-on-B.

    Every item is rated once with the non-Role response on A and once on B.
    """
    by_item: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        other_side = "B" if row["role_position"] == "A" else "A"
        by_item[row["item_id"]][other_side] = row["code"] not in {"role", "tie", "unable"}
    a_only = sum(1 for d in by_item.values() if d["A"] and not d["B"])
    b_only = sum(1 for d in by_item.values() if d["B"] and not d["A"])
    return a_only, b_only, exact_two_sided_binomial(a_only, a_only + b_only)


def main() -> int:
    judgments = load_judgments()
    summary = json.loads((HERE / "results" / "final_human_summary.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []

    def check(label: str, computed: float, stored: float) -> None:
        if not math.isclose(computed, stored, rel_tol=0, abs_tol=1e-12):
            mismatches.append(f"{label}: computed {computed!r} != stored {stored!r}")

    with (HERE / "data" / "study_a_judgments_deidentified.csv").open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    sessions = {r["session"]: r["phase"] for r in raw}
    print(
        f"Sample: {len(sessions)} sessions "
        f"({sum(p == 'pilot' for p in sessions.values())} pilot + "
        f"{sum(p == 'main' for p in sessions.values())} main), "
        f"{len({r['annotator'] for r in raw})} annotators, {len(raw)} judgments\n"
    )

    header = (
        f"{'Comparison':<11}{'N':>4}{'Role':>13}{'Other':>13}{'Same':>13}{'Unable':>7}"
        f"{'D [95% CI]':>28}{'Agreement (alpha)':>21}"
    )
    print(header)
    details = []
    for index, contrast in enumerate(CONTRASTS):
        rows = [j for j in judgments if j["contrast"] == contrast]
        other = rows[0]["other_condition"]
        counts = Counter(row["code"] for row in rows)
        n = len(rows)
        items = aggregate_a_items(rows, 2)
        triplet = bootstrap_contrast_by_triplet(items, ITERATIONS, SEED + index)
        participant = bootstrap_contrast_by_participant(rows, ITERATIONS, SEED + 100 + index)
        agreement = nominal_agreement(rows)
        length = study_a_length_diagnostics(rows, items, ITERATIONS, SEED + 200 + index)
        directional = counts["role"] + counts[other]
        sign_p = exact_two_sided_binomial(counts[other], directional)
        a_only, b_only, mcnemar_p = position_mcnemar(rows)

        stored = summary["primary"][contrast]
        for code in ("role", other, "tie", "unable"):
            check(f"{contrast}/count/{code}", counts[code], stored["counts"][code])
        for key in ("D", "ci_low", "ci_high"):
            check(f"{contrast}/triplet/{key}", triplet["estimate" if key == "D" else key], stored["triplet_cluster"][key])
        # The participant-cluster bootstrap draws clusters in sorted-label order.
        # Annotator labels here are re-indexed (not derived from Prolific IDs),
        # so its CI uses a different random draw than the stored run and is not
        # compared; only the point estimate is. It is not reported in the paper.
        check(f"{contrast}/pid/D", participant["estimate"], stored["pid_cluster"]["D"])
        check(f"{contrast}/pair_agreement", agreement["pairwise_percent_agreement"], stored["pair_agreement"])
        check(f"{contrast}/alpha", agreement["krippendorff_alpha_nominal"], stored["nominal_alpha"])
        check(f"{contrast}/mcnemar", mcnemar_p, summary["position"][f"{contrast}_paired_mcnemar_p"])

        def pct(k: int) -> str:
            return f"{k} ({100 * k / n:.1f}%)"

        print(
            f"{contrast:<11}{n:>4}{pct(counts['role']):>13}{pct(counts[other]):>13}"
            f"{pct(counts['tie']):>13}{counts['unable']:>7}"
            f"{triplet['estimate']:>+10.3f} [{triplet['ci_low']:+.3f}, {triplet['ci_high']:+.3f}]"
            f"{100 * agreement['pairwise_percent_agreement']:>13.1f}% ({agreement['krippendorff_alpha_nominal']:.3f})"
        )
        details.append(
            f"{contrast}: sign test on directional judgments p = {sign_p:.3f} "
            f"({counts[other]} {other} vs {counts['role']} Role); "
            f"position McNemar p = {mcnemar_p:.3f} ({a_only} vs {b_only} discordant items); "
            f"longer response chosen among directional = "
            f"{100 * length['longer_selected_among_directional']:.1f}%; "
            f"participant-cluster sensitivity CI [{participant['ci_low']:+.3f}, {participant['ci_high']:+.3f}] "
            f"(stored run: [{stored['pid_cluster']['ci_low']:+.3f}, {stored['pid_cluster']['ci_high']:+.3f}])"
        )

    print()
    for line in details:
        print(line)

    # Repeated-annotator sensitivity: drop the later (main-study) exposure of any
    # item an annotator saw twice.
    seen: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in judgments:
        seen[(row["prolific_pid"], row["item_id"])].append(row)
    drop = {
        row["judgment_id"]
        for rows in seen.values()
        if len(rows) > 1
        for row in rows
        if row["phase"] == "main"
    }
    deduped = [row for row in judgments if row["judgment_id"] not in drop]
    rb = [row["contrast_score"] for row in deduped if row["contrast"] == "role_basic"]
    raw_d = sum(rb) / len(rb)
    check("first_exposure/role_basic_raw_D", raw_d, summary["first_exposure_sensitivity"]["role_basic_raw_D"])
    print(
        f"First-exposure sensitivity: removed {len(drop)} repeated judgment(s); "
        f"Role-Basic raw D = {raw_d:.4f} (n = {len(rb)})"
    )

    if mismatches:
        print("\nMISMATCHES:")
        for line in mismatches:
            print("  " + line)
        return 1
    print("\nAll recomputed values match results/final_human_summary.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
