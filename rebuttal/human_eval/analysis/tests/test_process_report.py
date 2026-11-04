from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from print_participant_process_report import (  # noqa: E402
    DEFAULT_MANIFEST,
    _markdown,
    build_report,
)


class ParticipantProcessReportTest(unittest.TestCase):
    def test_decodes_visible_side_and_omits_identifiers(self) -> None:
        export = {
            "exported_at": "2026-07-29T00:00:00Z",
            "study_sessions": [
                {
                    "id": "private-internal-session",
                    "study_type": "A",
                    "bundle_id": "SA-TEST",
                    "prolific_pid": "private-prolific-pid",
                    "prolific_study_id": "private-study-id",
                    "prolific_session_id": "private-external-session",
                    "status": "complete",
                    "comprehension_attempts": 1,
                    "comprehension_passed": 1,
                    "completed_at": "2026-07-29T00:10:00Z",
                }
            ],
            "study_responses": [
                {
                    "id": "private-response-id",
                    "session_id": "private-internal-session",
                    "item_id": "SA-C001",
                    "item_kind": "task",
                    "stage": "primary",
                    "answer": "SA-C001-X",
                    "presentation_order": '["SA-C001-Y","SA-C001-X"]',
                    "duration_ms": 65000,
                    "locked": 1,
                },
                {
                    "session_id": "private-internal-session",
                    "item_id": "SA-C001",
                    "item_kind": "task",
                    "stage": "confidence",
                    "answer": "4",
                    "presentation_order": "[]",
                    "duration_ms": 0,
                    "locked": 1,
                },
            ],
            "study_events": [
                {
                    "id": "private-event-id-1",
                    "session_id": "private-internal-session",
                    "event_type": "page_view",
                    "event_data": json.dumps(
                        {
                            "itemId": "SA-C001",
                            "itemKind": "task",
                            "stage": "primary",
                            "status": "tasks",
                            "progressIndex": 0,
                        }
                    ),
                    "elapsed_ms": 1000,
                },
                {
                    "id": "private-event-id-2",
                    "session_id": "private-internal-session",
                    "event_type": "runtime_snapshot",
                    "event_data": json.dumps(
                        {
                            "itemId": "SA-C001",
                            "itemKind": "task",
                            "stage": "primary",
                        }
                    ),
                    "elapsed_ms": 30000,
                },
                {
                    "id": "private-event-id-3",
                    "session_id": "private-internal-session",
                    "event_type": "scroll_milestone",
                    "event_data": json.dumps(
                        {
                            "itemId": "SA-C001",
                            "itemKind": "task",
                            "stage": "primary",
                            "milestone": 75,
                        }
                    ),
                    "elapsed_ms": 40000,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tempdir:
            export_path = Path(tempdir) / "export.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")
            report = build_report(export_path, DEFAULT_MANIFEST)

        participant = report["participants"][0]
        item = participant["items"][0]
        self.assertEqual(item["visible_mapping"], "A=role / B=pro")
        self.assertEqual(item["selected_visible_option"], "Response B")
        self.assertEqual(item["decoded_choice"], "pro")
        self.assertEqual(item["heartbeat_count"], 1)
        self.assertEqual(item["scroll_milestones"], [75])
        self.assertIsNone(participant["attention_check_passed"])
        self.assertFalse(
            report["telemetry_semantics"]["mouse_coordinates_collected"]
        )

        rendered = _markdown(report)
        self.assertIn("Attention check: not used", rendered)
        for secret in (
            "private-internal-session",
            "private-prolific-pid",
            "private-study-id",
            "private-external-session",
            "private-response-id",
            "private-event-id",
        ):
            self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
