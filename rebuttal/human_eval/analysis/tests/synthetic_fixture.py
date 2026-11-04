"""Small deterministic export fixture spanning both studies and every B stratum."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PILOT_ROOT = Path(__file__).resolve().parents[2]
_A_MANIFEST = json.loads(
    (
        PILOT_ROOT
        / "study_a/generated/private/manifest_full.json"
    ).read_text(encoding="utf-8")
)
_A_BY_CONTRAST = {
    contrast: next(
        item
        for item in _A_MANIFEST["items"]
        if item.get("contrast", "role_pro") == contrast
    )
    for contrast in ("role_basic", "role_pro")
}
A_ITEMS = tuple(
    _A_BY_CONTRAST[contrast]["item_id"]
    for contrast in ("role_basic", "role_pro")
)
B_ITEMS = ("SB001", "SB003", "SB004", "SB008", "SB010")


def _a_variant(contrast: str, condition: str) -> str:
    item = _A_BY_CONTRAST[contrast]
    opaque = next(
        key
        for key, value in item["variant_condition_key"].items()
        if value == condition
    )
    return f"{item['item_id']}-{opaque}"


def _session(session_id: str, study: str, failed_qc: bool) -> dict[str, object]:
    required = A_ITEMS if study == "A" else B_ITEMS
    return {
        "id": session_id,
        "study_type": study,
        "bundle_id": f"SYNTHETIC-{study}",
        "prolific_pid": f"PID-{session_id}",
        "prolific_study_id": f"PAID-{study}",
        "prolific_session_id": f"PROLIFIC-{session_id}",
        "status": "complete",
        "comprehension_passed": (
            0 if study == "A" and failed_qc else 1
        ),
        "consented_at": "2026-07-27T11:00:00Z",
        "rationale_item_ids": json.dumps([A_ITEMS[0]] if study == "A" else []),
        "required_item_ids": json.dumps(list(required)),
        "required_imc_ids": json.dumps(
            ["A-IMC-1"]
            if study == "A"
            else ["B-IMC-1", "B-IMC-2"]
        ),
        "completed_at": "2026-07-27T12:00:00Z",
        "paid": 1,
        "fixture_failed_qc": failed_qc,
    }


def _response(
    session_id: str,
    item_id: str,
    answer: str,
    *,
    item_kind: str = "task",
    stage: str = "primary",
    order: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": f"{session_id}:{item_id}:{stage}",
        "session_id": session_id,
        "item_id": item_id,
        "item_kind": item_kind,
        "stage": stage,
        "answer": answer,
        "presentation_order": order or [],
        "duration_ms": 1250,
        "locked": True,
        "updated_at": "2026-07-27T12:00:00Z",
    }


def build_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sessions = [
        _session("A1", "A", False),
        _session("A2", "A", False),
        _session("A3", "A", True),
        _session("B1", "B", False),
        _session("B2", "B", False),
        _session("B3", "B", True),
    ]
    responses: list[dict[str, object]] = []

    role_basic, role_pro = A_ITEMS
    a_answers = {
        "A1": {
            role_basic: _a_variant("role_basic", "basic"),
            role_pro: "tie",
        },
        "A2": {
            role_basic: _a_variant("role_basic", "basic"),
            role_pro: _a_variant("role_pro", "role"),
        },
        "A3": {
            role_basic: _a_variant("role_basic", "role"),
            role_pro: _a_variant("role_pro", "pro"),
        },
    }
    for session_id, answers in a_answers.items():
        for item_id, answer in answers.items():
            responses.append(
                _response(
                    session_id,
                    item_id,
                    answer,
                    order=[f"{item_id}-X", f"{item_id}-Y"],
                )
            )
        responses.append(
            _response(
                session_id,
                A_ITEMS[0],
                "The selected responses differ in their substantive reasons and recommendations.",
                stage="rationale",
            )
        )
        failed = session_id == "A3"
        responses.append(
            _response(
                session_id,
                "A-IMC-1",
                "first" if failed else "second",
                item_kind="imc",
            )
        )

    b_answers = {
        "B1": {
            "SB001": "none_related",
            "SB003": "SB003-C-A",
            "SB004": "SB004-C-B",
            "SB008": "none_unrelated",
            "SB010": "SB010-C-A",
        },
        "B2": {
            "SB001": "none_unrelated",
            "SB003": "SB003-C-B",
            "SB004": "SB004-C-C",
            "SB008": "none_related",
            "SB010": "cannot_context",
        },
        "B3": {
            "SB001": "SB001-C-C",
            "SB003": "none_related",
            "SB004": "cannot_context",
            "SB008": "none_unrelated",
            "SB010": "none_related",
        },
    }
    for session_id, answers in b_answers.items():
        for item_id, answer in answers.items():
            candidates = [f"{item_id}-C-A", f"{item_id}-C-B", f"{item_id}-C-C"]
            responses.append(_response(session_id, item_id, answer, order=candidates))
            responses.append(
                _response(
                    session_id,
                    item_id,
                    "The source and candidates either share or differ in their central claim.",
                    stage="rationale",
                )
            )
        failed = session_id == "B3"
        responses.extend(
            [
                _response(
                    session_id,
                    "B-IMC-1",
                    "candidate-a" if failed else "candidate-b",
                    item_kind="imc",
                ),
                _response(
                    session_id,
                    "B-IMC-2",
                    "candidate-a" if failed else "candidate-c",
                    item_kind="imc",
                ),
            ]
        )
    return sessions, responses


def write_records(
    directory: Path,
    sessions: list[dict[str, object]],
    responses: list[dict[str, object]],
) -> tuple[Path, Path]:
    sessions_path = directory / "synthetic_sessions.csv"
    responses_path = directory / "synthetic_responses.json"
    with sessions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(field for row in sessions for field in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sessions)
    responses_path.write_text(
        json.dumps({"study_responses": responses}, indent=2) + "\n",
        encoding="utf-8",
    )
    return sessions_path, responses_path


def write_fixture(directory: Path) -> tuple[Path, Path]:
    """Write sessions as CSV and responses as wrapped JSON to exercise both readers."""

    return write_records(directory, *build_fixture())
