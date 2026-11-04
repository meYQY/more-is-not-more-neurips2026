#!/usr/bin/env python3
"""Print an anonymized, human-readable Study A process report.

This is a descriptive audit companion to ``analyze_annotations.py``.  It
prints what a participant selected and the interaction telemetry that the site
actually collected.  It deliberately does not invent mouse trajectories or
interpret a visible/focused browser window as proof of human attention.

The report never emits Prolific IDs, external/internal session IDs, run IDs,
study IDs, assignment tokens, or response database IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MANIFEST = ROOT / "study_a/generated/private/manifest_full.json"
DEFAULT_EXPORT = (
    ROOT / "prolific-operations/private/production-export-pilot-live.json"
)

PRIMARY_STAGES = {"primary"}
INTERRUPTION_TYPES = {"window_blur", "page_visibility_hidden"}
FOCUS_RETURN_TYPES = {"window_focus", "page_visibility_visible"}
SENSITIVE_KEY_RE = re.compile(
    r"\b(?:prolific_(?:pid|study_id|session_id)|session_id|run_id|runId|"
    r"assignment_token|response_id)\b",
    re.IGNORECASE,
)


class ReportError(RuntimeError):
    """Raised when the export cannot support an unambiguous process report."""


def _json_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _ms(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _seconds(value: Any) -> float:
    return round(_ms(value) / 1000.0, 1)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def _is_demo(session: dict[str, Any]) -> bool:
    return any(
        str(session.get(key, "")).upper().startswith("DEMO-")
        for key in ("prolific_pid", "prolific_study_id", "prolific_session_id")
    )


def _latest_responses(
    responses: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in responses:
        key = (str(row.get("item_id", "")), str(row.get("stage", "")))
        previous = latest.get(key)
        rank = (
            int(bool(row.get("locked"))),
            str(row.get("updated_at", "")),
            str(row.get("created_at", "")),
        )
        previous_rank = (
            int(bool(previous.get("locked"))) if previous else -1,
            str(previous.get("updated_at", "")) if previous else "",
            str(previous.get("created_at", "")) if previous else "",
        )
        if previous is None or rank >= previous_rank:
            latest[key] = row
    return latest


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    return {str(item["item_id"]): item for item in items}


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    parsed = _json_value(event.get("event_data"), {})
    return parsed if isinstance(parsed, dict) else {}


def _events_for_stage(
    events: list[dict[str, Any]], item_id: str, stage: str
) -> list[dict[str, Any]]:
    result = []
    for event in events:
        data = _event_data(event)
        if str(data.get("itemId", "")) != item_id:
            continue
        if str(data.get("stage", "")) != stage:
            continue
        result.append(event)
    return result


def _actual_sequence(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover randomized task order from first primary page views."""

    first_by_position: dict[int, dict[str, Any]] = {}
    for event in sorted(events, key=lambda row: (_ms(row.get("elapsed_ms")), str(row.get("id", "")))):
        if event.get("event_type") != "page_view":
            continue
        data = _event_data(event)
        if data.get("stage") != "primary" or data.get("status") != "tasks":
            continue
        try:
            position = int(data.get("progressIndex"))
        except (TypeError, ValueError):
            continue
        first_by_position.setdefault(
            position,
            {
                "position": position,
                "item_id": str(data.get("itemId", "")),
                "item_kind": str(data.get("itemKind", "")),
                "primary_started_ms": _ms(event.get("elapsed_ms")),
            },
        )
    return [first_by_position[key] for key in sorted(first_by_position)]


def _variant_condition(item: dict[str, Any], variant_id: str) -> str:
    opaque = variant_id.rsplit("-", 1)[-1]
    return str(item.get("variant_condition_key", {}).get(opaque, opaque))


def _visible_side_and_condition(
    primary: dict[str, Any], item: dict[str, Any]
) -> tuple[str, str, str]:
    order = _json_value(primary.get("presentation_order"), [])
    if not isinstance(order, list):
        order = []
    answer = str(primary.get("answer", ""))
    side_map = {
        str(variant): ("Response A" if index == 0 else "Response B")
        for index, variant in enumerate(order[:2])
    }
    condition_map = {
        str(variant): _variant_condition(item, str(variant)) for variant in order[:2]
    }
    sides = " / ".join(
        f"{'A' if index == 0 else 'B'}={condition_map.get(str(variant), '?')}"
        for index, variant in enumerate(order[:2])
    )
    if answer == "tie":
        return "About the same", "tie", sides
    if answer in {"unable", "cannot", "cannot_context"}:
        return "Unable to judge", "unable", sides
    return side_map.get(answer, answer), condition_map.get(answer, "?"), sides


def _rationale_dwell_ms(
    events: list[dict[str, Any]], item_id: str, progress_index: int
) -> int | None:
    rationale_views = _events_for_stage(events, item_id, "rationale")
    if not rationale_views:
        return None
    start = min(_ms(event.get("elapsed_ms")) for event in rationale_views)
    end_candidates = []
    for event in events:
        if event.get("event_type") != "task_advanced":
            continue
        data = _event_data(event)
        if data.get("currentIndex") == progress_index + 1:
            elapsed = _ms(event.get("elapsed_ms"))
            if elapsed >= start:
                end_candidates.append(elapsed)
    if not end_candidates:
        return None
    return min(end_candidates) - start


def _primary_metrics(
    events: list[dict[str, Any]], item_id: str
) -> dict[str, Any]:
    stage_events = _events_for_stage(events, item_id, "primary")
    scroll = sorted(
        {
            int(_event_data(event).get("milestone"))
            for event in stage_events
            if event.get("event_type") == "scroll_milestone"
            and str(_event_data(event).get("milestone", "")).isdigit()
        }
    )
    heartbeats = sum(
        event.get("event_type") == "runtime_snapshot" for event in stage_events
    )
    interruptions = [
        event
        for event in stage_events
        if event.get("event_type") in INTERRUPTION_TYPES
    ]
    returns = [
        event
        for event in stage_events
        if event.get("event_type") in FOCUS_RETURN_TYPES
    ]
    return {
        "scroll_milestones": scroll,
        "heartbeat_count": heartbeats,
        "interruption_count": len(interruptions),
        "focus_return_count": len(returns),
        "window_proxy": (
            "no observed blur/hidden event"
            if not interruptions
            else f"{len(interruptions)} observed blur/hidden event(s)"
        ),
    }


def _run_duration(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    commits = [
        event for event in events if event.get("event_type") == "final_commit"
    ]
    if not commits:
        return None, None
    data = _event_data(max(commits, key=lambda row: _ms(row.get("elapsed_ms"))))
    client = data.get("clientRunElapsedMs", data.get("runElapsedMs"))
    server = data.get("serverWallElapsedMs")
    return (_ms(client) if client is not None else None, _ms(server) if server is not None else None)


def _consent_delay(events: list[dict[str, Any]]) -> int | None:
    values = [
        _ms(event.get("elapsed_ms"))
        for event in events
        if event.get("event_type") == "consent_given"
    ]
    return min(values) if values else None


def build_report(
    export_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    export_bytes = export_path.read_bytes()
    payload = json.loads(export_bytes)
    sessions = payload.get("study_sessions", [])
    responses = payload.get("study_responses", [])
    events = payload.get("study_events", [])
    manifest = _load_manifest(manifest_path)

    eligible = [
        session
        for session in sessions
        if session.get("study_type") == "A"
        and session.get("status") == "complete"
        and not _is_demo(session)
    ]
    eligible.sort(key=lambda row: (str(row.get("bundle_id", "")), str(row.get("completed_at", ""))))

    responses_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        responses_by_session[str(response.get("session_id", ""))].append(response)
    for event in events:
        events_by_session[str(event.get("session_id", ""))].append(event)

    participants: list[dict[str, Any]] = []
    for anonymous_index, session in enumerate(eligible, 1):
        internal_id = str(session.get("id", ""))
        session_responses = responses_by_session[internal_id]
        session_events = events_by_session[internal_id]
        session_events.sort(
            key=lambda row: (_ms(row.get("elapsed_ms")), str(row.get("id", "")))
        )
        response_map = _latest_responses(session_responses)
        sequence = _actual_sequence(session_events)
        if not sequence:
            required = _json_value(session.get("required_item_ids"), [])
            sequence = [
                {
                    "position": position,
                    "item_id": str(item_id),
                    "item_kind": "task",
                    "primary_started_ms": None,
                }
                for position, item_id in enumerate(required)
            ]

        item_rows = []
        for sequence_entry in sequence:
            item_id = sequence_entry["item_id"]
            item_kind = sequence_entry["item_kind"]
            primary = response_map.get((item_id, "primary"), {})
            confidence = response_map.get((item_id, "confidence"), {})
            rationale = response_map.get((item_id, "rationale"), {})
            metrics = _primary_metrics(session_events, item_id)
            if item_kind == "imc":
                answer = str(primary.get("answer", ""))
                choice_display = {
                    "first": "Response A contains more",
                    "second": "Response B contains more",
                    "tie": "About the same",
                    "unable": "Unable to judge",
                }.get(answer, answer)
                decoded = "correct" if answer == "second" else "incorrect"
                question = "Embedded attention check"
                contrast = "attention_check"
                model = ""
                persona = ""
                visible_mapping = "A/B labels only"
            else:
                item = manifest.get(item_id)
                if item is None:
                    raise ReportError(f"Manifest item missing: {item_id}")
                choice_display, decoded, visible_mapping = _visible_side_and_condition(
                    primary, item
                )
                question = str(item.get("query", ""))
                contrast = str(item.get("contrast", ""))
                model = str(item.get("model_code", ""))
                persona = str(item.get("persona_name", ""))

            rationale_text = str(rationale.get("answer", "")).strip()
            rationale_dwell = _rationale_dwell_ms(
                session_events, item_id, int(sequence_entry["position"])
            )
            rationale_stage = _events_for_stage(
                session_events, item_id, "rationale"
            )
            item_rows.append(
                {
                    "order": int(sequence_entry["position"]) + 1,
                    "item_id": item_id,
                    "item_kind": item_kind,
                    "question": question,
                    "contrast": contrast,
                    "model": model,
                    "persona": persona,
                    "visible_mapping": visible_mapping,
                    "selected_visible_option": choice_display,
                    "decoded_choice": decoded,
                    "confidence": (
                        int(confidence["answer"])
                        if str(confidence.get("answer", "")).isdigit()
                        else None
                    ),
                    "primary_duration_ms": _ms(primary.get("duration_ms")),
                    **metrics,
                    "rationale_required": bool(rationale_text),
                    "rationale_text": rationale_text,
                    "rationale_word_count": _word_count(rationale_text),
                    "rationale_dwell_ms": rationale_dwell,
                    "rationale_heartbeat_count": sum(
                        event.get("event_type") == "runtime_snapshot"
                        for event in rationale_stage
                    ),
                }
            )

        event_counts = Counter(str(event.get("event_type", "")) for event in session_events)
        client_ms, server_ms = _run_duration(session_events)
        task_confidences = [
            row["confidence"]
            for row in item_rows
            if row["item_kind"] == "task" and row["confidence"] is not None
        ]
        attention_check_rows = [
            row for row in item_rows if row["item_kind"] == "imc"
        ]
        duplicate_milestones = [
            event_type
            for event_type in ("primary_tasks_complete", "final_commit")
            if event_counts[event_type] > 1
        ]
        participants.append(
            {
                "anonymous_participant": f"P{anonymous_index:02d}",
                "bundle_id": str(session.get("bundle_id", "")),
                "status": str(session.get("status", "")),
                "comprehension_attempts": int(
                    session.get("comprehension_attempts") or 0
                ),
                "comprehension_passed": bool(
                    session.get("comprehension_passed")
                ),
                "formal_choice_count": sum(
                    row["item_kind"] == "task" for row in item_rows
                ),
                "mean_formal_confidence": (
                    round(sum(task_confidences) / len(task_confidences), 2)
                    if task_confidences
                    else None
                ),
                "attention_check_passed": (
                    all(
                        row["decoded_choice"] == "correct"
                        for row in attention_check_rows
                    )
                    if attention_check_rows
                    else None
                ),
                "consent_delay_ms": _consent_delay(session_events),
                "client_post_consent_ms": client_ms,
                "server_post_consent_ms": server_ms,
                "heartbeat_count": event_counts["runtime_snapshot"],
                "locked_response_rows": sum(
                    bool(response.get("locked"))
                    for response in session_responses
                ),
                "total_response_rows": len(session_responses),
                "page_view_count": event_counts["page_view"],
                "comparison_locked_count": event_counts["comparison_locked"],
                "task_advanced_count": event_counts["task_advanced"],
                "final_commit_count": event_counts["final_commit"],
                "blur_count_all_stages": event_counts["window_blur"],
                "hidden_count_all_stages": event_counts[
                    "page_visibility_hidden"
                ],
                "resumed_run": any(
                    bool(_event_data(event).get("resumed"))
                    for event in session_events
                    if event.get("event_type") == "session_opened"
                ),
                "duplicate_milestones": duplicate_milestones,
                "items": item_rows,
            }
        )

    return {
        "schema_version": "pilot-print-v1",
        "source_export_name": export_path.name,
        "source_export_sha256": hashlib.sha256(export_bytes).hexdigest(),
        "source_exported_at": payload.get("exported_at"),
        "source_row_counts": {
            "sessions": len(sessions),
            "responses": len(responses),
            "events": len(events),
        },
        "participant_count": len(participants),
        "telemetry_semantics": {
            "mouse_coordinates_collected": False,
            "mouse_trajectory_available": False,
            "heartbeat_interval_seconds": 30,
            "active_proxy_definition": (
                "runtime snapshots accrue only while the document is visible "
                "and the browser window is focused"
            ),
            "scroll_definition": (
                "document-level 25/50/75/100 percent milestones; not mouse "
                "movement and not per-card reading proof"
            ),
            "use_for_exclusion": False,
        },
        "participants": participants,
    }


def _format_seconds(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1000.0:.1f}s"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Anonymized participant process report",
        "",
        f"- Completed non-demo Study A participants: {report['participant_count']}",
        "- Literal mouse coordinates/trajectory collected: **No**",
        "- `runtime_snapshot` interpretation: the page was visible and the "
        "browser window focused at a 30-second sampling point.",
        "- Scroll milestones are document-level behavior traces, not proof "
        "that every sentence was read.",
        "- Telemetry is descriptive and is not an automatic exclusion rule.",
        f"- Source export SHA-256: `{report['source_export_sha256']}`",
        f"- Source export timestamp: `{report['source_exported_at']}`",
        "",
    ]
    for participant in report["participants"]:
        attention_check_result = participant["attention_check_passed"]
        attention_check_label = (
            "not used"
            if attention_check_result is None
            else (
                "passed"
                if attention_check_result
                else "failed (legacy diagnostic only)"
            )
        )
        lines.extend(
            [
                f"## {participant['anonymous_participant']} — "
                f"{participant['bundle_id']}",
                "",
                f"- Comprehension: "
                f"{'passed' if participant['comprehension_passed'] else 'failed'} "
                f"on attempt {participant['comprehension_attempts']}",
                f"- Attention check: {attention_check_label}",
                f"- Formal choices: {participant['formal_choice_count']}; "
                f"mean formal confidence: {participant['mean_formal_confidence']}",
                f"- Pre-consent delay: "
                f"{_format_seconds(participant['consent_delay_ms'])}; "
                f"post-consent client/server: "
                f"{_format_seconds(participant['client_post_consent_ms'])} / "
                f"{_format_seconds(participant['server_post_consent_ms'])}",
                f"- 30s heartbeats: {participant['heartbeat_count']}; "
                f"all-stage blur/hidden: "
                f"{participant['blur_count_all_stages']} / "
                f"{participant['hidden_count_all_stages']}; "
                f"resumed run: {participant['resumed_run']}",
                f"- Pipeline rows/events: "
                f"{participant['locked_response_rows']}/"
                f"{participant['total_response_rows']} responses locked; "
                f"page views {participant['page_view_count']}; "
                f"comparison locks {participant['comparison_locked_count']}; "
                f"task advances {participant['task_advanced_count']}; "
                f"final commits {participant['final_commit_count']}",
                f"- Duplicate milestone flags: "
                f"{', '.join(participant['duplicate_milestones']) or 'none'}",
                "",
                "| # | Item | Type | Contrast | Visible sides | Selected | "
                "Decoded | Conf. | Primary | Scroll | HB | Interruptions |",
                "|---:|---|---|---|---|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for item in participant["items"]:
            scroll = "/".join(str(value) for value in item["scroll_milestones"])
            lines.append(
                f"| {item['order']} | {item['item_id']} | {item['item_kind']} | "
                f"{item['contrast']} | {item['visible_mapping']} | "
                f"{item['selected_visible_option']} | {item['decoded_choice']} | "
                f"{item['confidence'] if item['confidence'] is not None else ''} | "
                f"{_format_seconds(item['primary_duration_ms'])} | "
                f"{scroll or 'none'} | {item['heartbeat_count']} | "
                f"{item['interruption_count']} |"
            )
        lines.append("")
        lines.append("### Item details and written rationale")
        lines.append("")
        for item in participant["items"]:
            lines.extend(
                [
                    f"- **{item['order']}. {item['item_id']}** — "
                    f"{item['question'] or 'Attention check'}",
                    f"  - model/persona: "
                    f"{item['model'] or 'n/a'} / {item['persona'] or 'n/a'}; "
                    f"window proxy: {item['window_proxy']}",
                ]
            )
            if item["rationale_text"]:
                lines.append(
                    f"  - rationale "
                    f"({_format_seconds(item['rationale_dwell_ms'])}, "
                    f"{item['rationale_word_count']} words, "
                    f"{item['rationale_heartbeat_count']} heartbeat(s)): "
                    f"“{item['rationale_text']}”"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_item_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "anonymous_participant",
        "bundle_id",
        "order",
        "item_id",
        "item_kind",
        "question",
        "contrast",
        "model",
        "persona",
        "visible_mapping",
        "selected_visible_option",
        "decoded_choice",
        "confidence",
        "primary_duration_ms",
        "scroll_milestones",
        "heartbeat_count",
        "interruption_count",
        "focus_return_count",
        "window_proxy",
        "rationale_text",
        "rationale_word_count",
        "rationale_dwell_ms",
        "rationale_heartbeat_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for participant in report["participants"]:
            for item in participant["items"]:
                row = {
                    "anonymous_participant": participant[
                        "anonymous_participant"
                    ],
                    "bundle_id": participant["bundle_id"],
                    **item,
                    "scroll_milestones": "|".join(
                        str(value) for value in item["scroll_milestones"]
                    ),
                }
                writer.writerow({key: row.get(key, "") for key in fields})


def _assert_anonymous_text(text: str) -> None:
    # Key names are also checked so future edits cannot accidentally make the
    # private identifiers part of a public-facing report schema.
    for line in text.splitlines():
        if SENSITIVE_KEY_RE.search(line):
            raise ReportError(
                "Potentially identifying field leaked into report output: "
                f"{line[:120]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.export, args.manifest)
    args.outdir.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(report)
    # The Markdown report is the portable print artifact.  JSON/CSV contain the
    # same anonymized content for scripted review.
    _assert_anonymous_text(markdown)
    (args.outdir / "participant_process_report.md").write_text(
        markdown, encoding="utf-8"
    )
    (args.outdir / "participant_process_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_item_csv(report, args.outdir / "participant_item_process.csv")
    os.chmod(args.outdir, 0o700)
    for name in (
        "participant_process_report.md",
        "participant_process_report.json",
        "participant_item_process.csv",
    ):
        os.chmod(args.outdir / name, 0o600)
    print(
        f"Wrote anonymized process report for "
        f"{report['participant_count']} participant(s) to {args.outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
