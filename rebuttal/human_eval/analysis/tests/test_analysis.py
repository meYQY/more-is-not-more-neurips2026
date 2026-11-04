from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze_annotations import (  # noqa: E402
    AnalysisError,
    DEFAULT_A_BUNDLES,
    DEFAULT_A_MANIFEST,
    DEFAULT_A_PUBLIC,
    DEFAULT_B_BUNDLES,
    DEFAULT_B_PROVENANCE,
    DEFAULT_B_VALIDATION,
    analyze,
    blocked_exact_binary_comparison,
    blocked_exact_fraction_comparison,
    load_export,
    load_metadata,
    nominal_agreement,
    parse_args,
    strict_majority,
)
from tests.synthetic_fixture import (  # noqa: E402
    A_ITEMS,
    build_fixture,
    write_fixture,
    write_records,
)


class EndToEndAnalysisTest(unittest.TestCase):
    def test_legacy_three_rater_study_a_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            bundles_path = root / "bundles-private.json"
            public_path = root / "bundles-public.json"

            item_ids = [f"SA-{index + 1:03d}" for index in range(21)]
            manifest_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "item_id": item_id,
                                "model_code": f"model-{index % 7}",
                                "task_id": f"task-{index:03d}",
                                "subcategory": "synthetic",
                                "variant_condition_key": {
                                    "X": "independent-role",
                                    "Y": "independent-pro",
                                },
                                "displayed_word_counts": {
                                    "role_response": 100 + index,
                                    "pro_response": 110 + index,
                                },
                            }
                            for index, item_id in enumerate(item_ids)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            private_bundles = []
            public_bundles = []
            for bundle_index in range(21):
                # Each bundle receives its own and the two preceding cyclic
                # items: 21 bundles × 3 items = 63 legacy assignments.
                indices = [
                    bundle_index,
                    (bundle_index - 1) % 21,
                    (bundle_index - 2) % 21,
                ]
                bundle_id = f"SA-B{bundle_index + 1:02d}"
                private_items = [
                    {
                        "item_id": item_ids[index],
                        "response_a_condition": "independent-role",
                        "response_b_condition": "independent-pro",
                    }
                    for index in indices
                ]
                public_items = [
                    {
                        "item_id": item_ids[index],
                        "response_a_variant_id": f"{item_ids[index]}-X",
                        "response_b_variant_id": f"{item_ids[index]}-Y",
                    }
                    for index in indices
                ]
                private_bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "items": private_items,
                        "rationale_item_ids_after_primary_lock": [
                            item_ids[indices[0]],
                        ],
                    }
                )
                public_bundles.append(
                    {"bundle_id": bundle_id, "items": public_items}
                )

            bundles_path.write_text(
                json.dumps({"bundles": private_bundles}), encoding="utf-8"
            )
            public_path.write_text(
                json.dumps({"bundles": public_bundles}), encoding="utf-8"
            )
            metadata = load_metadata(
                manifest_path,
                bundles_path,
                public_path,
                DEFAULT_B_PROVENANCE,
                DEFAULT_B_BUNDLES,
                DEFAULT_B_VALIDATION,
            )
            self.assertEqual(len(metadata.a_items), 21)
            self.assertEqual(len(metadata.a_bundle_items), 21)
            self.assertEqual(metadata.a_items_per_bundle, 3)
            self.assertEqual(metadata.a_ratings_per_item, 3)
            self.assertEqual(metadata.a_expected_judgments, 63)

    def test_current_study_a_freeze_is_dual_contrast_two_rater(self) -> None:
        metadata = load_metadata(
            DEFAULT_A_MANIFEST,
            DEFAULT_A_BUNDLES,
            DEFAULT_A_PUBLIC,
            DEFAULT_B_PROVENANCE,
            DEFAULT_B_BUNDLES,
            DEFAULT_B_VALIDATION,
        )
        self.assertEqual(len(metadata.a_items), 84)
        self.assertEqual(len(metadata.a_bundle_items), 28)
        self.assertEqual(metadata.a_items_per_bundle, 6)
        self.assertEqual(metadata.a_ratings_per_item, 2)
        self.assertEqual(metadata.a_expected_judgments, 168)
        self.assertEqual(
            {item["contrast"] for item in metadata.a_items.values()},
            {"role_basic", "role_pro"},
        )
        self.assertEqual(
            len({item["triplet_id"] for item in metadata.a_items.values()}),
            42,
        )
        self.assertEqual(
            strict_majority(
                {"role": 1, "pro": 2, "tie": 0, "unable": 0}, 3
            ),
            "pro",
        )
        self.assertEqual(
            strict_majority(
                {"role": 1, "pro": 1, "tie": 1, "unable": 0}, 3
            ),
            "no_consensus",
        )

    def test_integrity_guards_reject_partial_payment_and_repeat_pid(self) -> None:
        for mutation in ("partial_payment", "repeat_pid", "manual_without_reason"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sessions_data, responses_data = build_fixture()
                if mutation == "partial_payment":
                    sessions_data[1]["paid"] = ""
                elif mutation == "repeat_pid":
                    sessions_data[1]["prolific_pid"] = sessions_data[0]["prolific_pid"]
                else:
                    sessions_data[0]["qc_include"] = 0
                sessions, responses = write_records(root, sessions_data, responses_data)
                with self.assertRaises(AnalysisError):
                    analyze(
                        sessions_path=sessions,
                        responses_path=responses,
                        outdir=root / "results",
                        bootstrap_iterations=5,
                        permutation_iterations=5,
                    )

    def test_analysis_hard_gates_recorded_consent_and_completion(self) -> None:
        for mutation in ("missing_consent", "missing_completion_timestamp"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sessions_data, responses_data = build_fixture()
                if mutation == "missing_consent":
                    sessions_data[0]["consented_at"] = ""
                else:
                    sessions_data[0]["completed_at"] = ""
                sessions, responses = write_records(root, sessions_data, responses_data)
                summary = analyze(
                    sessions_path=sessions,
                    responses_path=responses,
                    outdir=root / "results",
                    bootstrap_iterations=5,
                    permutation_iterations=5,
                )
                self.assertEqual(
                    summary["qc"]["by_study_sample"]["A"]["all_paid"]["sessions"],
                    2,
                )
                self.assertEqual(
                    summary["qc"]["by_study_sample"]["A"]["qc_filtered"]["sessions"],
                    1,
                )

    def test_wranger_json_and_variable_unit_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "d1.json"
            export.write_text(json.dumps([{"results": [{"id": "one"}, {"id": "two"}]}]))
            self.assertEqual(
                [row["id"] for row in load_export(export, "sessions")],
                ["one", "two"],
            )

        judgments = [
            {"item_id": "one", "code": "a"},
            {"item_id": "one", "code": "a"},
            {"item_id": "one", "code": "b"},
            {"item_id": "two", "code": "b"},
            {"item_id": "two", "code": "b"},
            {"item_id": "two", "code": "b"},
        ]
        agreement = nominal_agreement(judgments)
        self.assertAlmostEqual(agreement["pairwise_percent_agreement"], 2 / 3)
        self.assertAlmostEqual(agreement["krippendorff_alpha_nominal"], 0.375)

        blocked_items = [
            {
                "item_type": "control" if index == 0 else "unique",
                "majority_binary": "same" if index == 0 else "no_same",
                "n_ratings": 3,
                "same_count": 3 if index == 0 else 0,
                "unable_count": 0,
                "model": "model",
                "direction": "direction",
            }
            for index in range(4)
        ]
        blocked = blocked_exact_binary_comparison(blocked_items)
        self.assertAlmostEqual(blocked["exact_blocked_permutation_p"], 0.25)
        blocked_fraction = blocked_exact_fraction_comparison(blocked_items)
        self.assertAlmostEqual(blocked_fraction["exact_blocked_fraction_p"], 0.25)

    def test_synthetic_export_runs_and_decodes_frozen_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions, responses = write_fixture(root)
            outdir = root / "results"
            summary = analyze(
                sessions_path=sessions,
                responses_path=responses,
                outdir=outdir,
                bootstrap_iterations=500,
                permutation_iterations=500,
                seed=20260727,
            )

            self.assertEqual(
                summary["qc"]["by_study_sample"]["A"]["all_paid"]["sessions"], 3
            )
            self.assertEqual(
                summary["qc"]["by_study_sample"]["A"]["qc_filtered"]["sessions"], 2
            )
            self.assertEqual(
                summary["study_a"]["all_paid"]["raw_counts"],
                {"role": 2, "basic": 2, "pro": 1, "tie": 1, "unable": 0},
            )
            self.assertAlmostEqual(
                summary["study_a"]["qc_filtered"]["contrasts"]["role_basic"][
                    "contrast_difference"
                ]["estimate"],
                1.0,
            )
            self.assertEqual(
                summary["configuration"]["frozen_design"]["study_a"],
                {
                    "unique_items": 84,
                    "matched_triplets": 42,
                    "items_by_contrast": {
                        "role_basic": 42,
                        "role_pro": 42,
                    },
                    "bundles": 28,
                    "items_per_bundle": 6,
                    "ratings_per_item": 2,
                    "expected_judgments": 168,
                },
            )
            self.assertEqual(
                summary["study_b"]["qc_filtered"]["raw_counts"],
                {"same": 5, "related": 2, "different": 2, "unable": 1},
            )
            exact = summary["study_b"]["qc_filtered"]["exact_comparison"]
            self.assertEqual(exact["unique_same_majority"], 1)
            self.assertEqual(exact["control_same_majority"], 1)
            self.assertEqual(exact["excluded_unresolved_or_insufficient_control"], 1)
            self.assertIn(
                "exact_comparison",
                summary["study_b"]["qc_filtered"]["conservative_unable"],
            )

            with (outdir / "study_a_item_summaries.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                item_rows = list(csv.DictReader(handle))
            filtered_sa001 = next(
                row
                for row in item_rows
                if row["sample"] == "qc_filtered" and row["item_id"] == A_ITEMS[0]
            )
            self.assertEqual(
                filtered_sa001["majority_label"], "not_used_two_rater_design"
            )
            self.assertEqual(filtered_sa001["pair_pattern"], "agreement_basic")

            saved = json.loads((outdir / "summary.json").read_text())
            self.assertEqual(saved["configuration"]["seed"], 20260727)
            for filename in (
                "report.md",
                "session_qc.csv",
                "rationale_review.csv",
                "study_a_confidence_timing.csv",
                "telemetry_events.csv",
                "telemetry_session_summary.csv",
                "study_b_group_summaries.csv",
                "study_b_similarity_gradient.csv",
                "study_b_unable_sensitivity.csv",
                "agreement.csv",
            ):
                self.assertTrue((outdir / filename).is_file(), filename)

            with (outdir / "rationale_review.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rationale_rows = list(csv.DictReader(handle))
            a1_rationale = next(
                row
                for row in rationale_rows
                if row["session_id"] == "A1" and row["item_id"] == A_ITEMS[0]
            )
            self.assertEqual(a1_rationale["decoded_primary_choice"], "basic")
            self.assertEqual(a1_rationale["contrast"], "role_basic")
            self.assertEqual(a1_rationale["other_condition"], "basic")
            self.assertTrue(a1_rationale["triplet_id"])
            self.assertTrue(a1_rationale["persona_name"])
            self.assertTrue(a1_rationale["model"])
            self.assertTrue(a1_rationale["question"])
            self.assertTrue(a1_rationale["role_response"])
            self.assertTrue(a1_rationale["basic_response"])
            self.assertTrue(a1_rationale["other_response"])
            self.assertEqual(a1_rationale["manual_rationale_valid"], "")
            self.assertEqual(a1_rationale["manual_review_reason"], "")
            self.assertEqual(a1_rationale["manual_reviewer"], "")

    def test_optional_confidence_and_telemetry_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_data, responses_data = build_fixture()
            for session_id, value in (("A1", "5"), ("A2", "4"), ("A3", "3")):
                for item_id in A_ITEMS:
                    responses_data.append(
                        {
                            "id": f"{session_id}:{item_id}:confidence",
                            "session_id": session_id,
                            "item_id": item_id,
                            "item_kind": "task",
                            "stage": "confidence",
                            "answer": value,
                            "presentation_order": [],
                            "duration_ms": 500,
                            "locked": True,
                            "updated_at": "2026-07-27T12:00:00Z",
                        }
                    )
                responses_data.append(
                    {
                        "id": f"{session_id}:A-IMC-1:confidence",
                        "session_id": session_id,
                        "item_id": "A-IMC-1",
                        "item_kind": "imc",
                        "stage": "confidence",
                        "answer": "1",
                        "presentation_order": [],
                        "duration_ms": 500,
                        "locked": True,
                        "updated_at": "2026-07-27T12:00:00Z",
                    }
                )
            sessions, responses = write_records(
                root, sessions_data, responses_data
            )
            events = root / "events.json"
            events.write_text(
                json.dumps(
                    {
                        "study_events": [
                            {
                                "id": "EV-1",
                                "session_id": "A1",
                                "event_type": "primary_locked",
                                "event_data": json.dumps(
                                    {"itemId": A_ITEMS[0], "durationMs": 1250}
                                ),
                                "elapsed_ms": 5000,
                                "created_at": "2026-07-27T11:05:00Z",
                            },
                            {
                                "id": "EV-2",
                                "session_id": "A2",
                                "event_type": "final_commit",
                                "event_data": "{}",
                                "elapsed_ms": 9000,
                                "created_at": "2026-07-27T11:06:00Z",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            outdir = root / "results"
            summary = analyze(
                sessions_path=sessions,
                responses_path=responses,
                events_path=events,
                outdir=outdir,
                bootstrap_iterations=5,
                permutation_iterations=5,
            )
            diagnostic = summary["study_a"]["all_paid"][
                "confidence_and_timing"
            ]
            self.assertEqual(diagnostic["confidence_available"], 6)
            self.assertEqual(diagnostic["confidence_missing"], 0)
            self.assertEqual(diagnostic["confidence"]["mean"], 4)
            self.assertIn(
                "do not reweight or exclude", diagnostic["analysis_role"]
            )
            self.assertTrue(summary["telemetry"]["present"])
            self.assertEqual(summary["telemetry"]["events"], 2)
            self.assertEqual(summary["telemetry"]["events_with_item_id"], 1)

            with (outdir / "study_a_judgments.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                judgments = list(csv.DictReader(handle))
            self.assertIn("confidence", judgments[0])
            self.assertIn("confidence_duration_ms", judgments[0])
            with (outdir / "telemetry_events.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                telemetry = list(csv.DictReader(handle))
            self.assertEqual(len(telemetry), 2)
            self.assertEqual(telemetry[0]["item_id"], A_ITEMS[0])

    def test_study_a_low_confidence_and_failed_imc_are_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_data, responses_data = build_fixture()
            for session_id, value in (("A1", "2"), ("A2", "4"), ("A3", "3")):
                for item_id in A_ITEMS:
                    responses_data.append(
                        {
                            "id": f"{session_id}:{item_id}:confidence",
                            "session_id": session_id,
                            "item_id": item_id,
                            "item_kind": "task",
                            "stage": "confidence",
                            "answer": value,
                            "presentation_order": [],
                            "duration_ms": 500,
                            "locked": True,
                            "updated_at": "2026-07-27T12:00:00Z",
                        }
                    )
                responses_data.append(
                    {
                        "id": f"{session_id}:A-IMC-1:confidence",
                        "session_id": session_id,
                        "item_id": "A-IMC-1",
                        "item_kind": "imc",
                        "stage": "confidence",
                        "answer": "5",
                        "presentation_order": [],
                        "duration_ms": 500,
                        "locked": True,
                        "updated_at": "2026-07-27T12:00:00Z",
                    }
                )
            a1_imc = next(
                row
                for row in responses_data
                if row["session_id"] == "A1"
                and row["item_id"] == "A-IMC-1"
                and row["stage"] == "primary"
            )
            a1_imc["answer"] = "first"

            sessions, responses = write_records(
                root, sessions_data, responses_data
            )
            outdir = root / "results"
            analyze(
                sessions_path=sessions,
                responses_path=responses,
                outdir=outdir,
                bootstrap_iterations=5,
                permutation_iterations=5,
            )

            with (outdir / "session_qc.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                qc_rows = list(csv.DictReader(handle))
            a1 = next(row for row in qc_rows if row["session_id"] == "A1")
            self.assertEqual(a1["in_qc_filtered"], "True")
            self.assertEqual(a1["mean_task_confidence"], "2")
            self.assertEqual(a1["low_mean_task_confidence"], "True")
            self.assertEqual(a1["attention_check_confidence"], "5")
            self.assertEqual(a1["attention_check_failed_diagnostic"], "True")
            self.assertEqual(
                a1["diagnostic_flags"],
                "attention_check_failed|mean_task_confidence_below_3",
            )
            self.assertEqual(a1["qc_flags"], "")

    def test_current_study_a_without_imc_is_valid_and_legacy_imc_stays_audit_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_data, responses_data = build_fixture()

            # Simulate a current-protocol session with no Study A attention
            # check while retaining the legacy check rows for the other pilot
            # sessions in the same export.
            a1_session = next(
                row for row in sessions_data if row["id"] == "A1"
            )
            a1_session["required_imc_ids"] = "[]"
            responses_data = [
                row
                for row in responses_data
                if not (
                    row["session_id"] == "A1"
                    and row["item_id"] == "A-IMC-1"
                )
            ]

            sessions, responses = write_records(
                root, sessions_data, responses_data
            )
            outdir = root / "results"
            analyze(
                sessions_path=sessions,
                responses_path=responses,
                outdir=outdir,
                bootstrap_iterations=5,
                permutation_iterations=5,
            )

            with (outdir / "session_qc.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                qc_rows = list(csv.DictReader(handle))
            a1 = next(row for row in qc_rows if row["session_id"] == "A1")
            self.assertEqual(a1["in_qc_filtered"], "True")
            self.assertNotIn("missing_required_imc", a1["qc_flags"])

            with (outdir / "study_a_judgments.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                judgments = list(csv.DictReader(handle))
            self.assertTrue(judgments)
            self.assertTrue(
                all(row["item_id"] != "A-IMC-1" for row in judgments)
            )

    def test_missing_or_invalid_required_study_a_confidence_excludes_qc(self) -> None:
        for mutation in ("missing_imc_confidence", "invalid_task_confidence"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sessions_data, responses_data = build_fixture()
                for session_id in ("A1", "A2", "A3"):
                    for item_id in A_ITEMS:
                        answer = (
                            "9"
                            if mutation == "invalid_task_confidence"
                            and session_id == "A1"
                            and item_id == A_ITEMS[0]
                            else "4"
                        )
                        responses_data.append(
                            {
                                "id": f"{session_id}:{item_id}:confidence",
                                "session_id": session_id,
                                "item_id": item_id,
                                "item_kind": "task",
                                "stage": "confidence",
                                "answer": answer,
                                "presentation_order": [],
                                "duration_ms": 500,
                                "locked": True,
                                "updated_at": "2026-07-27T12:00:00Z",
                            }
                        )
                    if not (
                        mutation == "missing_imc_confidence"
                        and session_id == "A1"
                    ):
                        responses_data.append(
                            {
                                "id": f"{session_id}:A-IMC-1:confidence",
                                "session_id": session_id,
                                "item_id": "A-IMC-1",
                                "item_kind": "imc",
                                "stage": "confidence",
                                "answer": "4",
                                "presentation_order": [],
                                "duration_ms": 500,
                                "locked": True,
                                "updated_at": "2026-07-27T12:00:00Z",
                            }
                        )
                sessions, responses = write_records(
                    root, sessions_data, responses_data
                )
                outdir = root / "results"
                analyze(
                    sessions_path=sessions,
                    responses_path=responses,
                    outdir=outdir,
                    bootstrap_iterations=5,
                    permutation_iterations=5,
                )
                with (outdir / "session_qc.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    qc_rows = list(csv.DictReader(handle))
                a1 = next(row for row in qc_rows if row["session_id"] == "A1")
                self.assertEqual(a1["in_all_paid"], "True")
                self.assertEqual(a1["in_qc_filtered"], "False")
                expected_flag = (
                    "missing_or_unlocked_required_confidence"
                    if mutation == "missing_imc_confidence"
                    else "invalid_required_confidence"
                )
                self.assertIn(expected_flag, a1["qc_flags"])

    def test_valid_pilot_is_included_by_default_and_superseded_id_can_be_excluded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_data, responses_data = build_fixture()
            sessions_data[0]["prolific_study_id"] = "SUPERSEDED-PILOT-A"
            sessions, responses = write_records(root, sessions_data, responses_data)
            baseline = analyze(
                sessions_path=sessions,
                responses_path=responses,
                outdir=root / "baseline-results",
                bootstrap_iterations=5,
                permutation_iterations=5,
            )
            self.assertEqual(
                baseline["qc"]["by_study_sample"]["A"]["all_paid"]["sessions"],
                3,
            )
            outdir = root / "results"
            summary = analyze(
                sessions_path=sessions,
                responses_path=responses,
                outdir=outdir,
                bootstrap_iterations=5,
                permutation_iterations=5,
                exclude_prolific_study_ids=[
                    "SUPERSEDED-PILOT-A",
                    "UNMATCHED-SUPERSEDED",
                ],
            )

            self.assertEqual(
                summary["configuration"]["excluded_prolific_study_ids"],
                ["SUPERSEDED-PILOT-A", "UNMATCHED-SUPERSEDED"],
            )
            self.assertEqual(summary["qc"]["excluded_pilot_sessions"], 1)
            self.assertEqual(
                summary["qc"]["excluded_paid_eligible_pilot_sessions"], 1
            )
            self.assertEqual(
                summary["qc"]["by_study_sample"]["A"]["all_paid"]["sessions"], 2
            )
            self.assertEqual(
                summary["qc"]["by_study_sample"]["A"]["qc_filtered"]["sessions"], 1
            )
            self.assertTrue(
                any(
                    "Explicit study-listing exclusion SUPERSEDED-PILOT-A: excluded 1 session"
                    in warning
                    for warning in summary["warnings"]
                )
            )
            self.assertTrue(
                any(
                    "Explicit study-listing exclusion UNMATCHED-SUPERSEDED: no exported session"
                    in warning
                    for warning in summary["warnings"]
                )
            )

            with (outdir / "session_qc.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                qc_rows = list(csv.DictReader(handle))
            pilot = next(row for row in qc_rows if row["session_id"] == "A1")
            self.assertEqual(pilot["pilot_excluded"], "True")
            self.assertEqual(pilot["in_all_paid"], "False")
            self.assertEqual(pilot["in_qc_filtered"], "False")
            self.assertEqual(
                pilot["analysis_exclusion_reason"],
                "explicit_prolific_study_id_exclusion:SUPERSEDED-PILOT-A",
            )

    def test_excluded_study_id_cli_flag_is_repeatable(self) -> None:
        parsed = parse_args(
            [
                "--sessions",
                "sessions.json",
                "--responses",
                "responses.json",
                "--outdir",
                "results",
                "--exclude-prolific-study-id",
                "PILOT-A",
                "--exclude-prolific-study-id",
                "PILOT-B",
            ]
        )
        self.assertEqual(
            parsed.exclude_prolific_study_id, ["PILOT-A", "PILOT-B"]
        )


if __name__ == "__main__":
    unittest.main()
