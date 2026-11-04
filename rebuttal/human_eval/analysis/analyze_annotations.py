#!/usr/bin/env python3
"""Reproducible analysis for the frozen MODBench human annotation studies.

The script intentionally uses only the Python standard library.  It consumes
separate study_sessions and study_responses exports (CSV or JSON), decodes the
opaque answers with the frozen private manifests, applies the predeclared QC
sensitivity, and writes machine-readable tables plus a compact Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
DEFAULT_A_MANIFEST = PILOT_ROOT / "study_a/generated/private/manifest_full.json"
DEFAULT_A_BUNDLES = PILOT_ROOT / "study_a/generated/private/bundle_assignment_key.json"
DEFAULT_A_PUBLIC = PILOT_ROOT / "study_a/generated/public/bundles_public.json"
DEFAULT_B_PROVENANCE = PILOT_ROOT / "study_b/outputs/provenance_manifest.json"
DEFAULT_B_BUNDLES = PILOT_ROOT / "study_b/outputs/bundles_manifest.json"
DEFAULT_B_VALIDATION = PILOT_ROOT / "study_b/outputs/validation.json"

A_CODES = ("role", "basic", "pro", "tie", "unable")
A_CONTRASTS = ("role_basic", "role_pro")
B_CODES = ("same", "related", "different", "unable")
B_STRATUM_ORDER = ("far", "medium", "near", "just_above", "clear")
SAMPLES = ("all_paid", "qc_filtered")
SUPPORTED_A_RATINGS_PER_ITEM = frozenset({2, 3, 4})
EXPECTED_B_RATINGS_PER_ITEM = 3


class AnalysisError(RuntimeError):
    """Raised when an export is ambiguous enough to threaten validity."""


@dataclass(frozen=True)
class FrozenMetadata:
    a_items: dict[str, dict[str, Any]]
    a_bundle_items: dict[str, list[str]]
    a_bundle_rationales: dict[str, list[str]]
    a_items_per_bundle: int
    a_ratings_per_item: int
    a_expected_judgments: int
    b_items: dict[str, dict[str, Any]]
    b_bundle_items: dict[str, list[str]]


def _first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _bool(value: Any, *, default: bool | None = False) -> bool | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "include", "included", "paid"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "exclude", "excluded", "unpaid"}:
        return False
    raise AnalysisError(f"Cannot parse boolean value {value!r}")


def _list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [_text(entry) for entry in value if _text(entry)]
    if isinstance(value, tuple):
        return [_text(entry) for entry in value if _text(entry)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Invalid JSON list {value!r}") from exc
            if not isinstance(parsed, list):
                raise AnalysisError(
                    f"Expected JSON list, received {type(parsed).__name__}"
                )
            return [_text(entry) for entry in parsed if _text(entry)]
        return [entry.strip() for entry in stripped.split("|") if entry.strip()]
    raise AnalysisError(f"Cannot parse list value {value!r}")


def _normalize_choice(value: Any) -> str:
    return (
        _text(value)
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _normalize_condition(value: Any) -> str:
    """Map frozen condition labels to Role/Basic/Pro without reading text."""

    normalized = _normalize_choice(value)
    for prefix in ("independent_", "single_call_", "singlecall_"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
    aliases = {
        "persona_role": "role",
        "persona_basic": "basic",
        "persona_pro": "pro",
    }
    return aliases.get(normalized, normalized)


def _contrast_from_conditions(conditions: Sequence[str]) -> str:
    labels = set(conditions)
    if labels == {"role", "basic"}:
        return "role_basic"
    if labels == {"role", "pro"}:
        return "role_pro"
    raise AnalysisError(
        "Study A comparison must contain exactly Role–Basic or Role–Pro; "
        f"received {sorted(labels)}"
    )


def _is_demo_session(session: Mapping[str, Any]) -> bool:
    return any(
        _text(session.get(field)).upper().startswith("DEMO-")
        for field in (
            "prolific_pid",
            "prolific_study_id",
            "prolific_session_id",
        )
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON in {path}: {exc}") from exc


def load_export(path: Path, kind: str) -> list[dict[str, Any]]:
    """Load a CSV, JSON array, D1 result object, or named export object."""

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = _load_json(path)
    if isinstance(payload, list):
        # `wrangler d1 execute --json` wraps each SQL statement as an object
        # with a `results` array.  Plain JSON-array exports are also accepted.
        if payload and all(
            isinstance(entry, dict) and isinstance(entry.get("results"), list)
            for entry in payload
        ):
            rows = [row for statement in payload for row in statement["results"]]
        else:
            rows = payload
    elif isinstance(payload, dict):
        candidates = {
            "sessions": ("sessions", "study_sessions", "results", "rows"),
            "responses": ("responses", "study_responses", "results", "rows"),
            "events": ("events", "study_events", "results", "rows"),
        }[kind]
        rows = None
        for key in candidates:
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            raise AnalysisError(
                f"{path} must be a JSON array or contain one of {candidates}"
            )
    else:
        raise AnalysisError(f"{path} does not contain a record collection")
    if not all(isinstance(row, dict) for row in rows):
        raise AnalysisError(f"{path} contains a non-object record")
    return [dict(row) for row in rows]


def load_optional_events(
    explicit_path: Path | None,
    fallback_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], Path | None]:
    """Load telemetry explicitly or discover it in a combined JSON export."""

    if explicit_path is not None:
        return load_export(explicit_path, "events"), explicit_path
    seen: set[Path] = set()
    for path in fallback_paths:
        resolved = path.resolve()
        if resolved in seen or path.suffix.lower() == ".csv":
            continue
        seen.add(resolved)
        payload = _load_json(path)
        if isinstance(payload, dict) and any(
            isinstance(payload.get(key), list)
            for key in ("events", "study_events")
        ):
            return load_export(path, "events"), path
    return [], None


def load_metadata(
    a_manifest_path: Path,
    a_bundles_path: Path,
    a_public_path: Path,
    b_provenance_path: Path,
    b_bundles_path: Path,
    b_validation_path: Path,
) -> FrozenMetadata:
    a_manifest = _load_json(a_manifest_path)
    a_triplet_conditions = {
        _text(triplet.get("triplet_id")): {
            _normalize_condition(label): payload
            for label, payload in triplet.get("conditions", {}).items()
        }
        for triplet in a_manifest.get("triplets", [])
        if _text(triplet.get("triplet_id"))
    }
    a_items: dict[str, dict[str, Any]] = {}
    for item in a_manifest.get("items", []):
        item_id = item["item_id"]
        if item_id in a_items:
            raise AnalysisError(f"Duplicate Study A manifest item {item_id}")
        conditions = {
            _normalize_condition(label): payload
            for label, payload in item.get("conditions", {}).items()
        }
        variants = {
            f"{item_id}-{opaque}": _normalize_condition(condition)
            for opaque, condition in item["variant_condition_key"].items()
        }
        contrast = _text(item.get("contrast")).lower().replace("-", "_")
        if contrast not in A_CONTRASTS:
            contrast = _contrast_from_conditions(list(variants.values()))
        other_condition = "basic" if contrast == "role_basic" else "pro"
        triplet_id = _text(
            item.get("triplet_id")
            or item.get("matched_triplet_id")
            or item.get("source_triplet_id")
            or item_id
        )
        complete_conditions = a_triplet_conditions.get(
            triplet_id, conditions
        )
        word_counts = item["displayed_word_counts"]
        role_words = int(word_counts["role_response"])
        other_words = int(word_counts[f"{other_condition}_response"])
        a_items[item_id] = {
            "item_id": item_id,
            "triplet_id": triplet_id,
            "contrast": contrast,
            "other_condition": other_condition,
            "model": item["model_code"],
            "task_id": item["task_id"],
            "subcategory": item.get("subcategory", ""),
            "question": item.get("query", ""),
            "persona_id": item.get("persona_id", ""),
            "persona_name": item.get("persona_name", ""),
            "role_response": complete_conditions.get("role", {}).get(
                "response_text", ""
            ),
            "basic_response": complete_conditions.get("basic", {}).get(
                "response_text", ""
            ),
            "pro_response": complete_conditions.get("pro", {}).get(
                "response_text", ""
            ),
            "other_response": conditions.get(other_condition, {}).get(
                "response_text", ""
            ),
            "variants": variants,
            "role_words": role_words,
            "basic_words": (
                int(word_counts["basic_response"])
                if "basic_response" in word_counts
                else None
            ),
            "pro_words": (
                int(word_counts["pro_response"])
                if "pro_response" in word_counts
                else None
            ),
            "other_words": other_words,
        }
        a_items[item_id]["signed_log_length_ratio"] = math.log(
            other_words / role_words
        )
        a_items[item_id]["max_min_length_ratio"] = max(
            role_words, other_words
        ) / min(role_words, other_words)
        a_items[item_id]["length_matched_1_5x"] = (
            a_items[item_id]["max_min_length_ratio"] <= 1.5
        )
    if not a_items:
        raise AnalysisError("Frozen Study A manifest contains no items")

    a_bundles_payload = _load_json(a_bundles_path)
    a_bundle_items: dict[str, list[str]] = {}
    a_bundle_rationales: dict[str, list[str]] = {}
    a_bundle_orientation: dict[tuple[str, str], tuple[str, str]] = {}
    for bundle in a_bundles_payload.get("bundles", []):
        bundle_id = bundle["bundle_id"]
        if bundle_id in a_bundle_items:
            raise AnalysisError(f"Duplicate Study A bundle {bundle_id}")
        a_bundle_items[bundle_id] = [entry["item_id"] for entry in bundle["items"]]
        a_bundle_rationales[bundle_id] = list(
            bundle.get("rationale_item_ids_after_primary_lock", [])
        )
        for entry in bundle["items"]:
            a_bundle_orientation[(bundle_id, entry["item_id"])] = (
                _normalize_condition(entry["response_a_condition"]),
                _normalize_condition(entry["response_b_condition"]),
            )
    a_replication = Counter(
        item_id for item_ids in a_bundle_items.values() for item_id in item_ids
    )
    a_bundle_sizes = {len(item_ids) for item_ids in a_bundle_items.values()}
    if not a_bundle_items or a_bundle_sizes == {0} or len(a_bundle_sizes) != 1:
        raise AnalysisError(
            "Frozen Study A bundles must be non-empty and have a consistent size"
        )
    if set(a_replication) != set(a_items):
        raise AnalysisError(
            "Frozen Study A bundle items do not match the private manifest"
        )
    a_replication_counts = set(a_replication.values())
    if (
        len(a_replication_counts) != 1
        or next(iter(a_replication_counts)) not in SUPPORTED_A_RATINGS_PER_ITEM
    ):
        raise AnalysisError(
            "Frozen Study A bundle replication must be uniformly two "
            "(current), three, or four (archived) ratings per item"
        )
    a_ratings_per_item = next(iter(a_replication_counts))
    a_items_per_bundle = next(iter(a_bundle_sizes))
    a_expected_judgments = sum(map(len, a_bundle_items.values()))
    if a_expected_judgments != len(a_items) * a_ratings_per_item:
        raise AnalysisError(
            "Frozen Study A assignment total is inconsistent with its "
            "uniform per-item replication"
        )
    manifest_contrasts = Counter(
        item["contrast"] for item in a_items.values()
    )
    if set(manifest_contrasts) == set(A_CONTRASTS):
        if a_ratings_per_item != 2:
            raise AnalysisError(
                "Current dual-contrast Study A freeze requires exactly two "
                "ratings per comparison item"
            )
        if manifest_contrasts != Counter({"role_basic": 42, "role_pro": 42}):
            raise AnalysisError(
                "Current dual-contrast Study A freeze must contain 42 "
                "Role–Basic and 42 Role–Pro comparison items"
            )
        by_triplet: dict[str, set[str]] = defaultdict(set)
        for item in a_items.values():
            by_triplet[item["triplet_id"]].add(item["contrast"])
        if len(by_triplet) != 42 or any(
            contrasts != set(A_CONTRASTS)
            for contrasts in by_triplet.values()
        ):
            raise AnalysisError(
                "Current Study A freeze must pair both contrasts within each "
                "of 42 matched triplets"
            )
        for bundle_id, item_ids in a_bundle_items.items():
            contrasts = Counter(a_items[item_id]["contrast"] for item_id in item_ids)
            triplets = [a_items[item_id]["triplet_id"] for item_id in item_ids]
            if contrasts != Counter({"role_basic": 3, "role_pro": 3}):
                raise AnalysisError(
                    f"{bundle_id}: expected 3 Role–Basic and 3 Role–Pro items"
                )
            if len(triplets) != len(set(triplets)):
                raise AnalysisError(
                    f"{bundle_id}: contains both contrasts from one triplet"
                )

    a_public = _load_json(a_public_path)
    public_bundles = {
        bundle["bundle_id"]: bundle for bundle in a_public.get("bundles", [])
    }
    if set(public_bundles) != set(a_bundle_items):
        raise AnalysisError("Study A private/public bundle IDs do not match")
    for bundle_id, bundle in public_bundles.items():
        public_item_ids = [entry["item_id"] for entry in bundle["items"]]
        if public_item_ids != a_bundle_items[bundle_id]:
            raise AnalysisError(
                f"Study A private/public item order differs in {bundle_id}"
            )
        for entry in bundle["items"]:
            item_id = entry["item_id"]
            item = a_items.get(item_id)
            if item is None:
                raise AnalysisError(f"Unknown public Study A item {item_id}")
            response_a = entry["response_a_variant_id"]
            response_b = entry["response_b_variant_id"]
            expected_a, expected_b = a_bundle_orientation[(bundle_id, item_id)]
            if (
                item["variants"].get(response_a) != expected_a
                or item["variants"].get(response_b) != expected_b
            ):
                raise AnalysisError(
                    f"Study A private/public orientation mismatch for {bundle_id}/{item_id}"
                )

    b_provenance = _load_json(b_provenance_path)
    b_items: dict[str, dict[str, Any]] = {}
    for item in b_provenance.get("selected_items", []):
        item_id = item["blind_item_id"]
        b_items[item_id] = {
            "item_id": item_id,
            "model": item["model_key"],
            "model_display": item.get("model_display", item["model_key"]),
            "task_id": item["task_id"],
            "direction": item["direction"],
            "item_type": "unique" if item["machine_label"] == "unique" else "control",
            "machine_label": item["machine_label"],
            "stratum": item["sampling_band"],
            "similarity": float(item["max_centroid_similarity"]),
        }
    if len(b_items) != 56:
        raise AnalysisError(
            f"Frozen Study B provenance has {len(b_items)} items, expected 56"
        )

    validation = _load_json(b_validation_path)
    if validation.get("status") != "PASS":
        raise AnalysisError("Frozen Study B validation is not PASS")

    b_bundles_payload = _load_json(b_bundles_path)
    b_bundle_items = {
        bundle["bundle_id"]: [task["item_id"] for task in bundle["tasks"]]
        for bundle in b_bundles_payload.get("bundles", [])
    }
    b_replication = Counter(
        item_id for item_ids in b_bundle_items.values() for item_id in item_ids
    )
    if (
        len(b_bundle_items) != 21
        or any(len(item_ids) != 8 for item_ids in b_bundle_items.values())
        or set(b_replication) != set(b_items)
        or set(b_replication.values()) != {EXPECTED_B_RATINGS_PER_ITEM}
    ):
        raise AnalysisError("Frozen Study B bundle replication is invalid")
    return FrozenMetadata(
        a_items=a_items,
        a_bundle_items=a_bundle_items,
        a_bundle_rationales=a_bundle_rationales,
        a_items_per_bundle=a_items_per_bundle,
        a_ratings_per_item=a_ratings_per_item,
        a_expected_judgments=a_expected_judgments,
        b_items=b_items,
        b_bundle_items=b_bundle_items,
    )


def canonical_sessions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    paid_field_present = False
    for index, row in enumerate(rows, start=1):
        session_id = _text(_first(row, "id", "internal_session_id", "session_id"))
        if not session_id:
            raise AnalysisError(f"Session row {index} has no database session ID")
        if session_id in seen:
            raise AnalysisError(f"Duplicate session ID {session_id}")
        seen.add(session_id)
        paid_names = ("paid", "is_paid", "prolific_paid")
        paid_raw = _first(row, *paid_names, default=None)
        if any(name in row for name in paid_names):
            paid_field_present = True
        study_type = _text(_first(row, "study_type", "studyType")).upper()
        if study_type not in {"A", "B"}:
            raise AnalysisError(
                f"Session {session_id} has invalid study type {study_type!r}"
            )
        prolific_pid = _text(_first(row, "prolific_pid", "prolificPid"))
        if not prolific_pid:
            raise AnalysisError(f"Session {session_id} has no Prolific participant ID")
        manual_qc = _first(row, "qc_include", "include_in_qc", default=None)
        sessions.append(
            {
                "session_id": session_id,
                "study_type": study_type,
                "bundle_id": _text(_first(row, "bundle_id", "bundleId")),
                "prolific_pid": prolific_pid,
                "prolific_study_id": _text(
                    _first(row, "prolific_study_id", "prolificStudyId")
                ),
                "prolific_session_id": _text(
                    _first(row, "prolific_session_id", "prolificSessionId")
                ),
                "status": _text(_first(row, "status")).lower(),
                "comprehension_passed": bool(
                    _bool(
                        _first(row, "comprehension_passed", "comprehensionPassed"),
                        default=False,
                    )
                ),
                "consented_at": _text(
                    _first(row, "consented_at", "consentedAt")
                ),
                "rationale_item_ids": _list(
                    _first(row, "rationale_item_ids", "rationaleItemIds")
                ),
                "required_item_ids": _list(
                    _first(row, "required_item_ids", "requiredItemIds")
                ),
                "required_imc_ids": _list(
                    _first(row, "required_imc_ids", "requiredImcIds")
                ),
                "completed_at": _text(_first(row, "completed_at", "completedAt")),
                "paid": _bool(paid_raw, default=None),
                "manual_qc_include": _bool(manual_qc, default=None),
                "manual_exclude": bool(
                    _bool(_first(row, "exclude_from_qc", "qc_exclude"), default=False)
                ),
                "manual_qc_reason": _text(
                    _first(row, "qc_exclusion_reason", "qc_reason", "exclusion_reason")
                ),
            }
        )
    return sessions, paid_field_present


def canonical_responses(
    rows: Sequence[Mapping[str, Any]], warnings: list[str]
) -> list[dict[str, Any]]:
    """Canonicalize and collapse repeated autosave snapshots deterministically."""

    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        response = {
            "response_id": _text(_first(row, "id", "response_id")),
            "session_id": _text(_first(row, "session_id", "sessionId")),
            "item_id": _text(_first(row, "item_id", "itemId")),
            "item_kind": _text(_first(row, "item_kind", "itemKind")).lower(),
            "stage": _text(_first(row, "stage")).lower(),
            "answer": _text(_first(row, "answer")),
            "presentation_order": _list(
                _first(row, "presentation_order", "presentationOrder")
            ),
            "duration_ms": int(
                float(_first(row, "duration_ms", "durationMs", default=0) or 0)
            ),
            "locked": bool(_bool(_first(row, "locked"), default=False)),
            "updated_at": _text(_first(row, "updated_at", "updatedAt")),
            "_row": index,
        }
        if (
            not response["session_id"]
            or not response["item_id"]
            or not response["stage"]
        ):
            raise AnalysisError(
                f"Response row {index} lacks session_id, item_id, or stage"
            )
        key = (response["session_id"], response["item_id"], response["stage"])
        previous = selected.get(key)
        if previous is None:
            selected[key] = response
            continue
        if (
            previous["stage"] == "primary"
            and previous["locked"]
            and response["locked"]
            and previous["answer"] != response["answer"]
        ):
            raise AnalysisError(f"Conflicting locked primary exports for {key}")
        previous_sort = (previous["updated_at"], previous["_row"])
        current_sort = (response["updated_at"], response["_row"])
        if current_sort >= previous_sort:
            selected[key] = response
        warnings.append(f"Collapsed duplicate exported response snapshot for {key}")
    return sorted(
        selected.values(),
        key=lambda row: (row["session_id"], row["item_id"], row["stage"]),
    )


def canonical_events(
    rows: Sequence[Mapping[str, Any]], warnings: list[str]
) -> list[dict[str, Any]]:
    """Canonicalize optional study telemetry without using it for exclusion."""

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        event_id = _text(_first(row, "id", "event_id", default=f"event-{index}"))
        session_id = _text(_first(row, "session_id", "sessionId"))
        event_type = _text(_first(row, "event_type", "eventType")).lower()
        if not session_id or not event_type:
            warnings.append(f"Telemetry row {index} lacks session_id or event_type")
            continue
        if event_id in seen_ids:
            warnings.append(f"Duplicate telemetry event ID {event_id} was ignored")
            continue
        seen_ids.add(event_id)
        raw_data = _first(row, "event_data", "eventData", default={})
        if isinstance(raw_data, Mapping):
            event_data = dict(raw_data)
        elif raw_data in (None, ""):
            event_data = {}
        else:
            try:
                parsed = json.loads(str(raw_data))
                event_data = dict(parsed) if isinstance(parsed, Mapping) else {
                    "value": parsed
                }
            except (json.JSONDecodeError, TypeError):
                event_data = {"unparsed": str(raw_data)}
                warnings.append(
                    f"Telemetry event {event_id} has non-JSON event_data"
                )
        elapsed_raw = _first(row, "elapsed_ms", "elapsedMs", default=0)
        try:
            elapsed_ms = max(0, int(float(elapsed_raw or 0)))
        except (TypeError, ValueError):
            elapsed_ms = 0
            warnings.append(f"Telemetry event {event_id} has invalid elapsed_ms")
        item_id = _text(
            _first(
                event_data,
                "item_id",
                "itemId",
                default=_first(row, "item_id", "itemId", default=""),
            )
        )
        duration_raw = _first(event_data, "duration_ms", "durationMs", default=None)
        try:
            event_duration_ms = (
                max(0, int(float(duration_raw)))
                if duration_raw not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            event_duration_ms = None
        events.append(
            {
                "event_id": event_id,
                "session_id": session_id,
                "event_type": event_type,
                "event_data": json.dumps(
                    event_data, ensure_ascii=False, sort_keys=True
                ),
                "item_id": item_id,
                "event_duration_ms": event_duration_ms,
                "elapsed_ms": elapsed_ms,
                "created_at": _text(_first(row, "created_at", "createdAt")),
                "_row": index,
            }
        )
    return sorted(
        events,
        key=lambda row: (
            row["session_id"],
            row["created_at"],
            row["elapsed_ms"],
            row["_row"],
        ),
    )


def _parse_confidence(answer: Any) -> int | None:
    normalized = _normalize_choice(answer)
    if normalized.startswith("confidence_"):
        normalized = normalized.removeprefix("confidence_")
    try:
        value = int(normalized)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 5 else None


def code_study_a(
    answer: str, presentation_order: Sequence[str], item: Mapping[str, Any]
) -> str | None:
    normalized = _normalize_choice(answer)
    if normalized in {"tie", "about_the_same", "same_breadth"}:
        return "tie"
    if normalized in {"cannot", "unable", "cannot_judge"}:
        return "unable"
    if normalized in {"role", "role_broader"}:
        return "role"
    if normalized in {"basic", "basic_broader"}:
        return "basic"
    if normalized in {"pro", "pro_broader"}:
        return "pro"
    if normalized in {"first", "response_a", "a"} and presentation_order:
        answer = presentation_order[0]
    elif normalized in {"second", "response_b", "b"} and len(presentation_order) >= 2:
        answer = presentation_order[1]
    if presentation_order and answer not in presentation_order:
        return None
    return item["variants"].get(answer)


def code_study_b(
    answer: str, presentation_order: Sequence[str], item_id: str
) -> str | None:
    normalized = _normalize_choice(answer)
    if answer in presentation_order:
        return "same"
    if not presentation_order and answer in {
        f"{item_id}-C-A",
        f"{item_id}-C-B",
        f"{item_id}-C-C",
    }:
        return "same"
    if normalized in {
        "candidate_a",
        "candidate_b",
        "candidate_c",
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "a",
        "b",
        "c",
        "multiple_candidates",
        "more_than_one_candidate",
    }:
        return "same"
    if normalized in {"none_related", "related", "related_but_distinct"}:
        return "related"
    if normalized in {"none_unrelated", "different", "substantively_unrelated"}:
        return "different"
    if normalized in {
        "cannot_context",
        "not_enough_context",
        "unable",
        "cannot_tell",
    }:
        return "unable"
    return None


def _correct_imc(study_type: str, item_id: str, answer: str) -> bool:
    expected = {
        ("A", "A-IMC-1"): "second",
        ("B", "B-IMC-1"): "candidate_b",
        ("B", "B-IMC-2"): "candidate_c",
    }
    return _normalize_choice(answer) == expected.get(
        (study_type, item_id), "__invalid__"
    )


def assess_qc(
    sessions: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    metadata: FrozenMetadata,
    paid_field_present: bool,
    excluded_prolific_study_ids: frozenset[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    response_map = {
        (row["session_id"], row["item_id"], row["stage"]): row for row in responses
    }
    study_a_confidence_present = any(
        row["item_kind"] == "task" and row["stage"] == "confidence"
        for row in responses
    )
    sample_ids = {sample: set() for sample in SAMPLES}
    qc_rows: list[dict[str, Any]] = []
    excluded_session_counts: Counter[str] = Counter()
    excluded_paid_eligible_counts: Counter[str] = Counter()

    for session in sessions:
        session_id = session["session_id"]
        study = session["study_type"]
        bundle = session["bundle_id"]
        prolific_study_id = session["prolific_study_id"]
        pilot_excluded = prolific_study_id in excluded_prolific_study_ids
        is_demo = _is_demo_session(session)
        consent_recorded = bool(session["consented_at"])
        completion_recorded = bool(
            session["status"] == "complete" and session["completed_at"]
        )
        analysis_eligible = (
            consent_recorded and completion_recorded and not is_demo
        )
        if paid_field_present:
            paid_analysis_eligible = session["paid"] is True and analysis_eligible
            paid_source = "explicit_paid_field"
        else:
            paid_analysis_eligible = analysis_eligible
            paid_source = "completed_fallback"
        in_all_paid = paid_analysis_eligible and not pilot_excluded
        if pilot_excluded:
            excluded_session_counts[prolific_study_id] += 1
            if paid_analysis_eligible:
                excluded_paid_eligible_counts[prolific_study_id] += 1

        bundle_items = (
            metadata.a_bundle_items.get(bundle, [])
            if study == "A"
            else metadata.b_bundle_items.get(bundle, [])
        )
        required_items = session["required_item_ids"] or list(bundle_items)
        if not required_items:
            required_items = sorted(
                row["item_id"]
                for row in responses
                if row["session_id"] == session_id
                and row["item_kind"] == "task"
                and row["stage"] == "primary"
            )
            warnings.append(
                f"{session_id}: required items unavailable; inferred from exported task responses"
            )
        frozen_imcs = (
            {"A-IMC-1"}
            if study == "A"
            else {"B-IMC-1", "B-IMC-2"}
        )
        # Study A's current protocol intentionally has no attention check.
        # Legacy pilot sessions explicitly retain A-IMC-1 in their saved
        # assignment, while new sessions explicitly store an empty list.
        required_imcs = (
            list(session["required_imc_ids"])
            if study == "A"
            else session["required_imc_ids"] or sorted(frozen_imcs)
        )
        assignment_mismatch = bool(
            bundle_items
            and session["required_item_ids"]
            and set(session["required_item_ids"]) != set(bundle_items)
        )
        if study == "A":
            required_rationales = session[
                "rationale_item_ids"
            ] or metadata.a_bundle_rationales.get(bundle, [])
            frozen_rationales = metadata.a_bundle_rationales.get(bundle)
            if (
                frozen_rationales is not None
                and session["rationale_item_ids"]
                and set(session["rationale_item_ids"]) != set(frozen_rationales)
            ):
                assignment_mismatch = True
        else:
            required_rationales = required_items
        if (
            session["required_imc_ids"]
            and set(session["required_imc_ids"]) != frozen_imcs
        ):
            assignment_mismatch = True

        flags: list[str] = []
        if assignment_mismatch:
            flags.append("frozen_assignment_mismatch")
        if not consent_recorded:
            flags.append("consent_not_recorded")
        if not completion_recorded:
            flags.append("not_complete")
        if not session["comprehension_passed"]:
            flags.append("comprehension_not_passed")

        missing_primary: list[str] = []
        unlocked_primary: list[str] = []
        invalid_primary: list[str] = []
        for item_id in required_items:
            response = response_map.get((session_id, item_id, "primary"))
            if response is None:
                missing_primary.append(item_id)
                continue
            if not response["locked"]:
                unlocked_primary.append(item_id)
            if study == "A":
                item = metadata.a_items.get(item_id)
                code = (
                    code_study_a(
                        response["answer"], response["presentation_order"], item
                    )
                    if item
                    else None
                )
            else:
                code = (
                    code_study_b(
                        response["answer"], response["presentation_order"], item_id
                    )
                    if item_id in metadata.b_items
                    else None
                )
            if code is None:
                invalid_primary.append(item_id)
        if missing_primary:
            flags.append("missing_required_primary")
        if unlocked_primary:
            flags.append("unlocked_required_primary")
        if invalid_primary:
            flags.append("invalid_primary_answer")

        missing_confidence: list[str] = []
        invalid_confidence: list[str] = []
        task_confidences: list[int] = []
        missing_imc_confidence: list[str] = []
        invalid_imc_confidence: list[str] = []
        imc_confidences: list[int] = []
        if study == "A" and study_a_confidence_present:
            for item_id in required_items:
                response = response_map.get((session_id, item_id, "confidence"))
                if response is None or not response["locked"]:
                    missing_confidence.append(item_id)
                    continue
                if response["item_kind"] != "task":
                    invalid_confidence.append(item_id)
                    continue
                parsed = _parse_confidence(response["answer"])
                if parsed is None:
                    invalid_confidence.append(item_id)
                else:
                    task_confidences.append(parsed)
            for item_id in required_imcs:
                response = response_map.get((session_id, item_id, "confidence"))
                if response is None or not response["locked"]:
                    missing_imc_confidence.append(item_id)
                    continue
                if response["item_kind"] != "imc":
                    invalid_imc_confidence.append(item_id)
                    continue
                parsed = _parse_confidence(response["answer"])
                if parsed is None:
                    invalid_imc_confidence.append(item_id)
                else:
                    imc_confidences.append(parsed)
            if missing_confidence or missing_imc_confidence:
                flags.append("missing_or_unlocked_required_confidence")
            if invalid_confidence or invalid_imc_confidence:
                flags.append("invalid_required_confidence")
        mean_task_confidence = (
            statistics.mean(task_confidences) if task_confidences else None
        )
        low_mean_task_confidence = bool(
            len(task_confidences) == len(required_items)
            and mean_task_confidence is not None
            and mean_task_confidence < 3.0
        )

        missing_imc: list[str] = []
        wrong_imc: list[str] = []
        for item_id in required_imcs:
            response = response_map.get((session_id, item_id, "primary"))
            if response is None or not response["locked"]:
                missing_imc.append(item_id)
            elif not _correct_imc(study, item_id, response["answer"]):
                wrong_imc.append(item_id)
        if missing_imc:
            flags.append("missing_required_imc")
        if (
            study == "B"
            and len(wrong_imc) == len(required_imcs)
            and required_imcs
        ):
            flags.append("both_attention_checks_failed")
        diagnostic_flags: list[str] = []
        if study == "A" and wrong_imc:
            diagnostic_flags.append("attention_check_failed")
        if study == "A" and low_mean_task_confidence:
            diagnostic_flags.append("mean_task_confidence_below_3")

        missing_rationale: list[str] = []
        for item_id in required_rationales:
            response = response_map.get((session_id, item_id, "rationale"))
            if response is None or not response["locked"]:
                missing_rationale.append(item_id)
        if missing_rationale:
            flags.append("missing_or_unlocked_required_rationale")

        manual_exclusion = (
            session["manual_qc_include"] is False or session["manual_exclude"]
        )
        if manual_exclusion and not session["manual_qc_reason"]:
            raise AnalysisError(
                f"{session_id}: manual QC exclusion has no qc_exclusion_reason"
            )
        if manual_exclusion:
            flags.append("manual_exclusion")
        if session["manual_qc_reason"] and "manual_exclusion" not in flags:
            warnings.append(
                f"{session_id}: qc reason supplied without an exclusion flag"
            )

        qc_include = in_all_paid and not flags
        if in_all_paid:
            sample_ids["all_paid"].add(session_id)
        if qc_include:
            sample_ids["qc_filtered"].add(session_id)
        qc_rows.append(
            {
                "session_id": session_id,
                "study_type": study,
                "bundle_id": bundle,
                "prolific_pid": session["prolific_pid"],
                "prolific_study_id": prolific_study_id,
                "paid_source": paid_source,
                "paid_analysis_eligible_before_pilot_exclusion": paid_analysis_eligible,
                "pilot_excluded": pilot_excluded,
                "analysis_exclusion_reason": (
                    f"explicit_prolific_study_id_exclusion:{prolific_study_id}"
                    if pilot_excluded
                    else ""
                ),
                "in_all_paid": in_all_paid,
                "in_qc_filtered": qc_include,
                "is_demo": is_demo,
                "status": session["status"],
                "consent_recorded": consent_recorded,
                "completion_recorded": completion_recorded,
                "required_task_count": len(required_items),
                "frozen_assignment_mismatch": assignment_mismatch,
                "missing_primary_count": len(missing_primary),
                "invalid_primary_count": len(invalid_primary),
                "missing_confidence_count": len(missing_confidence),
                "invalid_confidence_count": len(invalid_confidence),
                "task_confidence_count": len(task_confidences),
                "mean_task_confidence": mean_task_confidence,
                "low_mean_task_confidence": low_mean_task_confidence,
                "missing_imc_confidence_count": len(missing_imc_confidence),
                "invalid_imc_confidence_count": len(invalid_imc_confidence),
                "attention_check_confidence": (
                    imc_confidences[0]
                    if study == "A" and len(imc_confidences) == 1
                    else None
                ),
                "wrong_imc_count": len(wrong_imc),
                "attention_check_failed_diagnostic": (
                    study == "A" and bool(wrong_imc)
                ),
                "missing_imc_count": len(missing_imc),
                "missing_rationale_count": len(missing_rationale),
                "diagnostic_flags": "|".join(diagnostic_flags),
                "qc_flags": "|".join(flags),
                "manual_qc_reason": session["manual_qc_reason"],
            }
        )

    if not paid_field_present:
        warnings.append(
            "No paid field was exported; all-paid uses recorded consent plus status=complete and completed_at as a documented fallback"
        )
    for study_id in sorted(excluded_prolific_study_ids):
        matched = excluded_session_counts[study_id]
        paid_eligible = excluded_paid_eligible_counts[study_id]
        if matched:
            warnings.append(
                f"Explicit study-listing exclusion {study_id}: excluded {matched} session(s) "
                f"from both analysis samples ({paid_eligible} otherwise eligible "
                "for all_paid)"
            )
        else:
            warnings.append(
                f"Explicit study-listing exclusion {study_id}: no exported session matched "
                "this Prolific study ID"
            )
    return qc_rows, sample_ids


def rationale_review_rows(
    sessions: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    metadata: FrozenMetadata,
    qc_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a private manual-review sheet without making content judgments."""

    session_map = {row["session_id"]: row for row in sessions}
    qc_map = {row["session_id"]: row for row in qc_rows}
    response_map = {
        (row["session_id"], row["item_id"], row["stage"]): row
        for row in responses
    }
    result: list[dict[str, Any]] = []
    for response in responses:
        if response["stage"] != "rationale":
            continue
        session = session_map.get(response["session_id"])
        if session is None:
            continue
        if session["study_type"] == "A":
            required_ids = session[
                "rationale_item_ids"
            ] or metadata.a_bundle_rationales.get(session["bundle_id"], [])
        else:
            required_ids = session["required_item_ids"] or metadata.b_bundle_items.get(
                session["bundle_id"], []
            )
        qc = qc_map[response["session_id"]]
        item: Mapping[str, Any] = {}
        decoded_primary_choice = ""
        question = ""
        role_response = ""
        basic_response = ""
        pro_response = ""
        other_response = ""
        contrast = ""
        other_condition = ""
        triplet_id = ""
        persona_id: Any = ""
        persona_name = ""
        model = ""
        task_id = ""
        if session["study_type"] == "A":
            item = metadata.a_items.get(response["item_id"], {})
            primary = response_map.get(
                (response["session_id"], response["item_id"], "primary")
            )
            if primary is not None and item:
                decoded_primary_choice = (
                    code_study_a(
                        primary["answer"],
                        primary["presentation_order"],
                        item,
                    )
                    or ""
                )
            question = _text(item.get("question"))
            role_response = _text(item.get("role_response"))
            basic_response = _text(item.get("basic_response"))
            pro_response = _text(item.get("pro_response"))
            other_response = _text(item.get("other_response"))
            contrast = _text(item.get("contrast"))
            other_condition = _text(item.get("other_condition"))
            triplet_id = _text(item.get("triplet_id"))
            persona_id = item.get("persona_id", "")
            persona_name = _text(item.get("persona_name"))
            model = _text(item.get("model"))
            task_id = _text(item.get("task_id"))
        else:
            item = metadata.b_items.get(response["item_id"], {})
            model = _text(item.get("model"))
            task_id = _text(item.get("task_id"))
        result.append(
            {
                "session_id": response["session_id"],
                "prolific_pid": session["prolific_pid"],
                "study_type": session["study_type"],
                "bundle_id": session["bundle_id"],
                "item_id": response["item_id"],
                "triplet_id": triplet_id,
                "contrast": contrast,
                "other_condition": other_condition,
                "required_rationale": response["item_id"] in required_ids,
                "locked": response["locked"],
                "model": model,
                "task_id": task_id,
                "persona_id": persona_id,
                "persona_name": persona_name,
                "question": question,
                "role_response": role_response,
                "basic_response": basic_response,
                "pro_response": pro_response,
                "other_response": other_response,
                "decoded_primary_choice": decoded_primary_choice,
                "word_count": len(response["answer"].split()),
                "rationale": response["answer"],
                "in_all_paid": qc["in_all_paid"],
                "in_qc_filtered_current_run": qc["in_qc_filtered"],
                "current_qc_flags": qc["qc_flags"],
                "current_diagnostic_flags": qc["diagnostic_flags"],
                "manual_rationale_valid": "",
                "manual_review_reason": "",
                "manual_reviewer": "",
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["study_type"],
            row["session_id"],
            row["item_id"],
        ),
    )


def telemetry_output_rows(
    events: Sequence[dict[str, Any]],
    sessions: Sequence[dict[str, Any]],
    qc_rows: Sequence[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create descriptive raw-event and participant-session telemetry tables."""

    session_map = {row["session_id"]: row for row in sessions}
    qc_map = {row["session_id"]: row for row in qc_rows}
    event_rows: list[dict[str, Any]] = []
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        session = session_map.get(event["session_id"])
        if session is None:
            warnings.append(
                f"Telemetry event {event['event_id']} references an unknown session"
            )
            continue
        qc = qc_map[event["session_id"]]
        output = {
            key: value for key, value in event.items() if key != "_row"
        }
        output.update(
            {
                "study_type": session["study_type"],
                "bundle_id": session["bundle_id"],
                "prolific_pid": session["prolific_pid"],
                "in_all_paid": qc["in_all_paid"],
                "in_qc_filtered": qc["in_qc_filtered"],
            }
        )
        event_rows.append(output)
        by_session[event["session_id"]].append(event)

    session_rows: list[dict[str, Any]] = []
    for session_id, rows in sorted(by_session.items()):
        session = session_map[session_id]
        qc = qc_map[session_id]
        elapsed = [row["elapsed_ms"] for row in rows]
        event_types = Counter(row["event_type"] for row in rows)
        per_item_events = sum(bool(row["item_id"]) for row in rows)
        session_rows.append(
            {
                "session_id": session_id,
                "study_type": session["study_type"],
                "bundle_id": session["bundle_id"],
                "prolific_pid": session["prolific_pid"],
                "in_all_paid": qc["in_all_paid"],
                "in_qc_filtered": qc["in_qc_filtered"],
                "n_events": len(rows),
                "n_event_types": len(event_types),
                "n_events_with_item_id": per_item_events,
                "min_elapsed_ms": min(elapsed) if elapsed else None,
                "max_elapsed_ms": max(elapsed) if elapsed else None,
                "event_type_counts": json.dumps(
                    dict(sorted(event_types.items())), sort_keys=True
                ),
            }
        )
    return event_rows, session_rows


def telemetry_summary(
    event_rows: Sequence[dict[str, Any]],
    session_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    event_types = Counter(row["event_type"] for row in event_rows)
    return {
        "present": bool(event_rows),
        "events": len(event_rows),
        "sessions_with_events": len(session_rows),
        "events_with_item_id": sum(bool(row["item_id"]) for row in event_rows),
        "event_type_counts": dict(sorted(event_types.items())),
        "session_max_elapsed_seconds": numeric_summary(
            [
                row["max_elapsed_ms"] / 1000
                for row in session_rows
                if row.get("max_elapsed_ms") is not None
            ]
        ),
        "analysis_role": (
            "descriptive process telemetry only; event timing is not an "
            "automatic exclusion or rejection rule"
        ),
    }


def extract_judgments(
    sessions: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    metadata: FrozenMetadata,
    sample_ids: Mapping[str, set[str]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_map = {row["session_id"]: row for row in sessions}
    response_map = {
        (row["session_id"], row["item_id"], row["stage"]): row
        for row in responses
    }
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    for response in responses:
        if response["item_kind"] != "task" or response["stage"] != "primary":
            continue
        if not response["locked"]:
            warnings.append(
                f"Unlocked primary {response['session_id']}/{response['item_id']} was not analyzed"
            )
            continue
        session = session_map.get(response["session_id"])
        if session is None:
            warnings.append(
                f"Response for unknown session {response['session_id']} was ignored"
            )
            continue
        for sample in SAMPLES:
            if response["session_id"] not in sample_ids[sample]:
                continue
            base = {
                "sample": sample,
                "session_id": response["session_id"],
                "prolific_pid": session["prolific_pid"],
                "bundle_id": session["bundle_id"],
                "item_id": response["item_id"],
                "answer_raw": response["answer"],
                "presentation_order": json.dumps(
                    response["presentation_order"], ensure_ascii=False
                ),
                "duration_ms": response["duration_ms"],
                "locked": response["locked"],
            }
            if session["study_type"] == "A":
                item = metadata.a_items.get(response["item_id"])
                if item is None:
                    warnings.append(
                        f"Unknown Study A item {response['item_id']} was ignored"
                    )
                    continue
                code = code_study_a(
                    response["answer"], response["presentation_order"], item
                )
                if code is None:
                    warnings.append(
                        f"Unrecognized Study A answer for {response['session_id']}/{response['item_id']}"
                    )
                    continue
                confidence_response = response_map.get(
                    (
                        response["session_id"],
                        response["item_id"],
                        "confidence",
                    )
                )
                confidence = (
                    _parse_confidence(confidence_response["answer"])
                    if confidence_response is not None
                    and confidence_response["locked"]
                    else None
                )
                if (
                    confidence_response is not None
                    and confidence_response["locked"]
                    and confidence is None
                ):
                    warnings.append(
                        "Invalid Study A confidence for "
                        f"{response['session_id']}/{response['item_id']} was ignored"
                    )
                a_rows.append(
                    {
                        **base,
                        "triplet_id": item["triplet_id"],
                        "contrast": item["contrast"],
                        "other_condition": item["other_condition"],
                        "model": item["model"],
                        "task_id": item["task_id"],
                        "subcategory": item["subcategory"],
                        "role_words": item["role_words"],
                        "basic_words": item["basic_words"],
                        "pro_words": item["pro_words"],
                        "other_words": item["other_words"],
                        "signed_log_length_ratio": item[
                            "signed_log_length_ratio"
                        ],
                        "max_min_length_ratio": item["max_min_length_ratio"],
                        "length_matched_1_5x": item["length_matched_1_5x"],
                        "confidence": confidence,
                        "confidence_duration_ms": (
                            confidence_response["duration_ms"]
                            if confidence_response is not None
                            else None
                        ),
                        "code": code,
                        "contrast_score": 1
                        if code == item["other_condition"]
                        else -1
                        if code == "role"
                        else 0,
                    }
                )
            else:
                item = metadata.b_items.get(response["item_id"])
                if item is None:
                    warnings.append(
                        f"Unknown Study B item {response['item_id']} was ignored"
                    )
                    continue
                code = code_study_b(
                    response["answer"],
                    response["presentation_order"],
                    response["item_id"],
                )
                if code is None:
                    warnings.append(
                        f"Unrecognized Study B answer for {response['session_id']}/{response['item_id']}"
                    )
                    continue
                b_rows.append(
                    {
                        **base,
                        **item,
                        "code": code,
                        "same_binary": code == "same",
                        "no_same_binary": code in {"related", "different"},
                    }
                )
    return a_rows, b_rows


def strict_majority(counts: Mapping[str, int], n: int) -> str:
    if n < 2:
        return "insufficient_ratings"
    threshold = n // 2 + 1
    winners = [label for label, count in counts.items() if count >= threshold]
    return winners[0] if len(winners) == 1 else "no_consensus"


def aggregate_a_items(
    judgments: Sequence[dict[str, Any]], expected_ratings_per_item: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[(row["sample"], row["item_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (sample, item_id), rows in sorted(grouped.items()):
        counts = Counter(row["code"] for row in rows)
        n = len(rows)
        if expected_ratings_per_item == 2:
            pair_pattern = (
                f"agreement_{rows[0]['code']}"
                if n == 2 and rows[0]["code"] == rows[1]["code"]
                else "disagreement"
                if n == 2
                else "incomplete"
            )
            majority_label = "not_used_two_rater_design"
        else:
            pair_pattern = "not_applicable_archived_design"
            majority_label = strict_majority(counts, n)
        result.append(
            {
                "sample": sample,
                "item_id": item_id,
                "triplet_id": rows[0]["triplet_id"],
                "contrast": rows[0]["contrast"],
                "other_condition": rows[0]["other_condition"],
                "model": rows[0]["model"],
                "task_id": rows[0]["task_id"],
                "subcategory": rows[0]["subcategory"],
                "role_words": rows[0]["role_words"],
                "basic_words": rows[0]["basic_words"],
                "pro_words": rows[0]["pro_words"],
                "other_words": rows[0]["other_words"],
                "signed_log_length_ratio": rows[0]["signed_log_length_ratio"],
                "max_min_length_ratio": rows[0]["max_min_length_ratio"],
                "length_matched_1_5x": rows[0]["length_matched_1_5x"],
                "n_ratings": n,
                **{f"{code}_count": counts[code] for code in A_CODES},
                "pair_pattern": pair_pattern,
                "majority_label": majority_label,
                "complete_design_ratings": n == expected_ratings_per_item,
                "contrast_preference": sum(
                    row["contrast_score"] for row in rows
                )
                / n,
            }
        )
    return result


def aggregate_b_items(judgments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[(row["sample"], row["item_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (sample, item_id), rows in sorted(grouped.items()):
        counts = Counter(row["code"] for row in rows)
        n = len(rows)
        four_majority = strict_majority(counts, n)
        binary_counts = {
            "same": counts["same"],
            "no_same": counts["related"] + counts["different"],
            "unable": counts["unable"],
        }
        binary_majority = strict_majority(binary_counts, n)
        item_type = rows[0]["item_type"]
        conservative_counts = dict(binary_counts)
        if item_type == "unique":
            conservative_counts["same"] += conservative_counts["unable"]
        else:
            conservative_counts["no_same"] += conservative_counts["unable"]
        conservative_counts["unable"] = 0
        conservative_majority = strict_majority(conservative_counts, n)
        primary_supported = (
            binary_majority == "no_same"
            if item_type == "unique"
            else binary_majority == "same"
        )
        conservative_supported = (
            conservative_majority == "no_same"
            if item_type == "unique"
            else conservative_majority == "same"
        )
        result.append(
            {
                "sample": sample,
                "item_id": item_id,
                "model": rows[0]["model"],
                "task_id": rows[0]["task_id"],
                "item_type": item_type,
                "machine_label": rows[0]["machine_label"],
                "direction": rows[0]["direction"],
                "stratum": rows[0]["stratum"],
                "similarity": rows[0]["similarity"],
                "n_ratings": n,
                **{f"{code}_count": counts[code] for code in B_CODES},
                "majority_4way": four_majority,
                "majority_binary": binary_majority,
                "conservative_unable_majority": conservative_majority,
                "complete_design_ratings": n == EXPECTED_B_RATINGS_PER_ITEM,
                "complete_triplicate": n == EXPECTED_B_RATINGS_PER_ITEM,
                "same_fraction": counts["same"] / n,
                "no_same_fraction": (counts["related"] + counts["different"]) / n,
                "conservative_same_fraction": conservative_counts["same"] / n,
                "conservative_no_same_fraction": conservative_counts["no_same"] / n,
                "primary_machine_supported": primary_supported,
                "conservative_machine_supported": conservative_supported,
            }
        )
    return result


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    retained = [float(value) for value in values]
    return {
        "n": len(retained),
        "mean": statistics.mean(retained) if retained else None,
        "median": statistics.median(retained) if retained else None,
        "q25": quantile(retained, 0.25),
        "q75": quantile(retained, 0.75),
        "min": min(retained) if retained else None,
        "max": max(retained) if retained else None,
    }


def study_a_confidence_timing(
    judgments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    confidences = [
        float(row["confidence"])
        for row in judgments
        if row.get("confidence") is not None
    ]
    duration_seconds = [
        row["duration_ms"] / 1000
        for row in judgments
        if row.get("duration_ms", 0) > 0
    ]
    confidence_duration_seconds = [
        row["confidence_duration_ms"] / 1000
        for row in judgments
        if row.get("confidence_duration_ms") not in (None, 0)
    ]
    by_code: dict[str, Any] = {}
    for code in A_CODES:
        rows = [row for row in judgments if row["code"] == code]
        code_confidences = [
            float(row["confidence"])
            for row in rows
            if row.get("confidence") is not None
        ]
        code_durations = [
            row["duration_ms"] / 1000
            for row in rows
            if row.get("duration_ms", 0) > 0
        ]
        by_code[code] = {
            "judgments": len(rows),
            "confidence": numeric_summary(code_confidences),
            "primary_duration_seconds": numeric_summary(code_durations),
        }
    return {
        "confidence_available": len(confidences),
        "confidence_missing": len(judgments) - len(confidences),
        "confidence_distribution": {
            str(value): sum(
                row.get("confidence") == value for row in judgments
            )
            for value in range(1, 6)
        },
        "confidence": numeric_summary(confidences),
        "primary_duration_seconds": numeric_summary(duration_seconds),
        "confidence_duration_seconds": numeric_summary(
            confidence_duration_seconds
        ),
        "by_outcome": by_code,
        "analysis_role": (
            "descriptive secondary measure over formal Study A task items only "
            "(the attention-check confidence is excluded); confidence and "
            "timing do not reweight or exclude primary judgments"
        ),
    }


def confidence_timing_rows(
    judgments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sample in SAMPLES:
        sample_rows = [row for row in judgments if row["sample"] == sample]
        for contrast in ("overall", *A_CONTRASTS):
            contrast_rows = (
                sample_rows
                if contrast == "overall"
                else [row for row in sample_rows if row["contrast"] == contrast]
            )
            for code in ("overall", *A_CODES):
                rows = (
                    contrast_rows
                    if code == "overall"
                    else [row for row in contrast_rows if row["code"] == code]
                )
                confidence = numeric_summary(
                    [
                        float(row["confidence"])
                        for row in rows
                        if row.get("confidence") is not None
                    ]
                )
                durations = numeric_summary(
                    [
                        row["duration_ms"] / 1000
                        for row in rows
                        if row.get("duration_ms", 0) > 0
                    ]
                )
                result.append(
                    {
                        "sample": sample,
                        "contrast": contrast,
                        "outcome": code,
                        "n_judgments": len(rows),
                        "n_confidence": confidence["n"],
                        "mean_confidence": confidence["mean"],
                        "median_confidence": confidence["median"],
                        "q25_confidence": confidence["q25"],
                        "q75_confidence": confidence["q75"],
                        "n_timed": durations["n"],
                        "mean_primary_seconds": durations["mean"],
                        "median_primary_seconds": durations["median"],
                        "q25_primary_seconds": durations["q25"],
                        "q75_primary_seconds": durations["q75"],
                    }
                )
    return result


def bootstrap_contrast_by_triplet(
    items: Sequence[dict[str, Any]], iterations: int, seed: int
) -> dict[str, Any]:
    if not items:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
            "method": "model-stratified matched-triplet cluster percentile bootstrap",
        }
    # Each frozen model×question×persona triplet is one inferential unit.
    # Within a contrast there is exactly one item per triplet. Equal triplet
    # weighting prevents QC-retained rating counts from changing scientific
    # weight.
    estimate = statistics.mean(row["contrast_preference"] for row in items)
    if len(items) == 1 or iterations <= 0:
        return {
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
            "method": "model-stratified matched-triplet cluster percentile bootstrap",
        }
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        by_model[row["model"]].append(row)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for model in sorted(by_model):
            group = by_model[model]
            sampled.extend(rng.choice(group) for _ in range(len(group)))
        draws.append(statistics.mean(row["contrast_preference"] for row in sampled))
    return {
        "estimate": estimate,
        "ci_low": quantile(draws, 0.025),
        "ci_high": quantile(draws, 0.975),
        "iterations": iterations,
        "method": "model-stratified matched-triplet cluster percentile bootstrap; all judgments for a sampled comparison stay together",
    }


def bootstrap_contrast_by_participant(
    judgments: Sequence[dict[str, Any]], iterations: int, seed: int
) -> dict[str, Any]:
    if not judgments:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
            "method": "participant-cluster percentile bootstrap",
        }
    estimate = statistics.mean(row["contrast_score"] for row in judgments)
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_participant[row["prolific_pid"]].append(row)
    participants = sorted(by_participant)
    if len(participants) == 1 or iterations <= 0:
        return {
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
            "method": "participant-cluster percentile bootstrap",
        }
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[dict[str, Any]] = []
        for _ in participants:
            sampled_rows.extend(by_participant[rng.choice(participants)])
        draws.append(statistics.mean(row["contrast_score"] for row in sampled_rows))
    return {
        "estimate": estimate,
        "ci_low": quantile(draws, 0.025),
        "ci_high": quantile(draws, 0.975),
        "iterations": iterations,
        "method": "participant-cluster percentile bootstrap; all judgments from a participant stay together",
    }


def directional_interpretation(
    item_bootstrap: Mapping[str, Any],
    other_condition: str,
) -> dict[str, str]:
    estimate = item_bootstrap.get("estimate")
    low = item_bootstrap.get("ci_low")
    high = item_bootstrap.get("ci_high")
    if estimate is None:
        label = "insufficient_data"
        wording = "Insufficient retained judgments for a directional conclusion."
    elif low is not None and low > 0:
        label = f"consistent_{other_condition}_advantage"
        wording = (
            f"Human judgments show a consistent {other_condition.title()} "
            "advantage over Role in this sample."
        )
    elif high is not None and high < 0:
        label = "consistent_role_advantage"
        wording = (
            f"Human judgments show a consistent Role advantage over "
            f"{other_condition.title()} in this sample."
        )
    else:
        label = f"no_stable_{other_condition}_advantage_detected"
        wording = (
            f"No stable human-perceived {other_condition.title()} advantage "
            "over Role was detected in this "
            "sample; this is not an equivalence claim."
        )
    return {
        "label": label,
        "report_wording": wording,
        "guardrail": (
            "A confidence interval containing zero is not evidence that Role "
            f"and {other_condition.title()} are equivalent."
        ),
    }


def pearson_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = statistics.mean(first)
    second_mean = statistics.mean(second)
    numerator = sum(
        (x - first_mean) * (y - second_mean)
        for x, y in zip(first, second, strict=True)
    )
    first_ss = sum((x - first_mean) ** 2 for x in first)
    second_ss = sum((y - second_mean) ** 2 for y in second)
    denominator = math.sqrt(first_ss * second_ss)
    return numerator / denominator if denominator else None


def study_a_length_diagnostics(
    judgments: Sequence[dict[str, Any]],
    items: Sequence[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    choice_counts: Counter[str] = Counter()
    for row in judgments:
        if row["code"] in {"tie", "unable"}:
            choice_counts[row["code"]] += 1
            continue
        if row["other_words"] == row["role_words"]:
            choice_counts["directional_equal_length"] += 1
            continue
        longer_condition = (
            row["other_condition"]
            if row["other_words"] > row["role_words"]
            else "role"
        )
        choice_counts[
            "longer_selected" if row["code"] == longer_condition else "shorter_selected"
        ] += 1

    matched_items = [row for row in items if row["length_matched_1_5x"]]
    matched_item_ids = {row["item_id"] for row in matched_items}
    matched_judgments = [
        row for row in judgments if row["item_id"] in matched_item_ids
    ]
    item_length_correlation = pearson_correlation(
        [row["signed_log_length_ratio"] for row in items],
        [row["contrast_preference"] for row in items],
    )
    directional = choice_counts["longer_selected"] + choice_counts["shorter_selected"]
    return {
        "all_items": len(items),
        "length_matched_items_1_5x": len(matched_items),
        "all_judgments": len(judgments),
        "length_matched_judgments_1_5x": len(matched_judgments),
        "judgment_counts": dict(sorted(choice_counts.items())),
        "longer_selected_among_directional": (
            choice_counts["longer_selected"] / directional if directional else None
        ),
        "item_level_pearson_signed_log_length_vs_contrast_preference": (
            item_length_correlation
        ),
        "matched_length_contrast_triplet_bootstrap": bootstrap_contrast_by_triplet(
            matched_items, iterations, seed
        ),
        "matched_length_contrast_participant_bootstrap": (
            bootstrap_contrast_by_participant(
                matched_judgments, iterations, seed + 1
            )
        ),
    }


def nominal_agreement(
    judgments: Sequence[dict[str, Any]], code_key: str = "code"
) -> dict[str, Any]:
    by_item: dict[str, list[str]] = defaultdict(list)
    for row in judgments:
        by_item[row["item_id"]].append(str(row[code_key]))
    disagree_pairs = 0
    total_pairs = 0
    agree_pairs = 0
    coincidence_marginals: Counter[str] = Counter()
    weighted_disagreement = 0.0
    coincidence_total = 0.0
    for labels in by_item.values():
        for first, second in itertools.combinations(labels, 2):
            total_pairs += 1
            if first == second:
                agree_pairs += 1
            else:
                disagree_pairs += 1
        if len(labels) < 2:
            continue
        # Krippendorff's coincidence construction gives each value in a unit
        # total weight one, irrespective of how many other raters that unit has.
        for first_index, first in enumerate(labels):
            coincidence_marginals[first] += 1
            for second_index, second in enumerate(labels):
                if first_index == second_index:
                    continue
                weight = 1 / (len(labels) - 1)
                coincidence_total += weight
                if first != second:
                    weighted_disagreement += weight
    n_total = sum(coincidence_marginals.values())
    observed_disagreement = (
        weighted_disagreement / coincidence_total if coincidence_total else None
    )
    expected_disagreement = (
        (
            n_total * n_total
            - sum(count * count for count in coincidence_marginals.values())
        )
        / (n_total * (n_total - 1))
        if n_total > 1
        else None
    )
    alpha = (
        1 - observed_disagreement / expected_disagreement
        if observed_disagreement is not None and expected_disagreement
        else None
    )
    return {
        "items": len(by_item),
        "judgments": sum(len(labels) for labels in by_item.values()),
        "judgments_in_multi_rated_items": n_total,
        "pairwise_percent_agreement": agree_pairs / total_pairs
        if total_pairs
        else None,
        "krippendorff_alpha_nominal": alpha,
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
    }


def _distribution_rows(
    judgments: Sequence[dict[str, Any]],
    codes: Sequence[str],
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    definitions: list[tuple[str, tuple[str, ...]]] = [("overall", ())]
    definitions.extend((dimension, (dimension,)) for dimension in dimensions)
    result: list[dict[str, Any]] = []
    for sample in SAMPLES:
        sample_rows = [row for row in judgments if row["sample"] == sample]
        for grouping, keys in definitions:
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            if not keys:
                groups[()] = sample_rows
            else:
                for row in sample_rows:
                    groups[tuple(row[key] for key in keys)].append(row)
            for values, rows in sorted(
                groups.items(), key=lambda pair: tuple(map(str, pair[0]))
            ):
                counts = Counter(row["code"] for row in rows)
                n = len(rows)
                for code in codes:
                    output = {
                        "sample": sample,
                        "grouping": grouping,
                        "n_judgments": n,
                        "code": code,
                        "count": counts[code],
                        "proportion": counts[code] / n if n else None,
                    }
                    output.update({key: value for key, value in zip(keys, values)})
                    result.append(output)
    return result


def a_model_descriptives(
    judgments: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sample in SAMPLES:
        sample_j = [row for row in judgments if row["sample"] == sample]
        sample_i = [row for row in items if row["sample"] == sample]
        groups = sorted(
            {(row["contrast"], row["model"]) for row in sample_j}
        )
        for contrast, model in groups:
            rows = [
                row
                for row in sample_j
                if row["model"] == model and row["contrast"] == contrast
            ]
            item_rows = [
                row
                for row in sample_i
                if row["model"] == model and row["contrast"] == contrast
            ]
            counts = Counter(row["code"] for row in rows)
            pair_patterns = Counter(row["pair_pattern"] for row in item_rows)
            result.append(
                {
                    "sample": sample,
                    "contrast": contrast,
                    "model": model,
                    "n_items": len(item_rows),
                    "n_judgments": len(rows),
                    **{f"{code}_judgments": counts[code] for code in A_CODES},
                    "agreement_items": sum(
                        count
                        for pattern, count in pair_patterns.items()
                        if pattern.startswith("agreement_")
                    ),
                    "disagreement_items": pair_patterns["disagreement"],
                    "incomplete_items": pair_patterns["incomplete"],
                    "contrast_preference": (
                        statistics.mean(
                            row["contrast_preference"] for row in item_rows
                        )
                        if item_rows
                        else None
                    ),
                }
            )
    return result


def b_group_summaries(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize item majorities at every predeclared Study B grouping."""

    definitions: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ()),
        ("item_type", ("item_type",)),
        ("item_type_x_direction", ("item_type", "direction")),
        ("item_type_x_stratum", ("item_type", "stratum")),
        (
            "item_type_x_direction_x_stratum",
            ("item_type", "direction", "stratum"),
        ),
        ("item_type_x_model", ("item_type", "model")),
    ]
    result: list[dict[str, Any]] = []
    for sample in SAMPLES:
        sample_rows = [row for row in items if row["sample"] == sample]
        for grouping, keys in definitions:
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            if not keys:
                groups[()] = sample_rows
            else:
                for row in sample_rows:
                    groups[tuple(row[key] for key in keys)].append(row)
            for values, rows in sorted(
                groups.items(), key=lambda pair: tuple(map(str, pair[0]))
            ):
                four = Counter(row["majority_4way"] for row in rows)
                binary = Counter(row["majority_binary"] for row in rows)
                n_judgments = sum(row["n_ratings"] for row in rows)
                output: dict[str, Any] = {
                    "sample": sample,
                    "grouping": grouping,
                    "n_items": len(rows),
                    "complete_triplicate_items": sum(
                        bool(row["complete_triplicate"]) for row in rows
                    ),
                    "n_judgments": n_judgments,
                    **{
                        f"majority_{code}_items": four[code]
                        for code in (
                            *B_CODES,
                            "no_consensus",
                            "insufficient_ratings",
                        )
                    },
                    **{
                        f"majority_{code}_rate": four[code] / len(rows)
                        if rows
                        else None
                        for code in (
                            *B_CODES,
                            "no_consensus",
                            "insufficient_ratings",
                        )
                    },
                    "binary_same_items": binary["same"],
                    "binary_no_same_items": binary["no_same"],
                    "binary_unable_items": binary["unable"],
                    "binary_no_consensus_items": binary["no_consensus"],
                    "binary_same_rate": (binary["same"] / len(rows) if rows else None),
                    "binary_no_same_rate": (
                        binary["no_same"] / len(rows) if rows else None
                    ),
                    "same_judgment_rate": (
                        sum(row["same_count"] for row in rows) / n_judgments
                        if n_judgments
                        else None
                    ),
                    "related_judgment_rate": (
                        sum(row["related_count"] for row in rows) / n_judgments
                        if n_judgments
                        else None
                    ),
                    "different_judgment_rate": (
                        sum(row["different_count"] for row in rows) / n_judgments
                        if n_judgments
                        else None
                    ),
                    "unable_judgment_rate": (
                        sum(row["unable_count"] for row in rows) / n_judgments
                        if n_judgments
                        else None
                    ),
                    "primary_machine_supported_items": sum(
                        bool(row["primary_machine_supported"]) for row in rows
                    ),
                    "conservative_machine_supported_items": sum(
                        bool(row["conservative_machine_supported"]) for row in rows
                    ),
                    "primary_machine_supported_rate": (
                        sum(bool(row["primary_machine_supported"]) for row in rows)
                        / len(rows)
                        if rows
                        else None
                    ),
                    "conservative_machine_supported_rate": (
                        sum(bool(row["conservative_machine_supported"]) for row in rows)
                        / len(rows)
                        if rows
                        else None
                    ),
                    "mean_similarity": (
                        statistics.fmean(row["similarity"] for row in rows)
                        if rows
                        else None
                    ),
                }
                output.update({key: value for key, value in zip(keys, values)})
                result.append(output)
    return result


def _hypergeom_probability(
    first_successes: int, first_size: int, total_successes: int, total_size: int
) -> float:
    if (
        first_successes < 0
        or first_successes > first_size
        or total_successes - first_successes < 0
        or total_successes - first_successes > total_size - first_size
    ):
        return 0.0
    return (
        math.comb(total_successes, first_successes)
        * math.comb(total_size - total_successes, first_size - first_successes)
        / math.comb(total_size, first_size)
    )


def _resolved_binary_items(
    items: Sequence[dict[str, Any]], majority_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def is_eligible(row: dict[str, Any]) -> bool:
        return row["n_ratings"] >= 2 and row[majority_key] in {
            "same",
            "no_same",
        }

    eligible = [row for row in items if is_eligible(row)]
    excluded = [row for row in items if not is_eligible(row)]
    return eligible, excluded


def exact_binary_comparison(
    items: Sequence[dict[str, Any]],
    *,
    majority_key: str = "majority_binary",
    analysis_variant: str = "primary",
) -> dict[str, Any]:
    """Fisher exact and unblocked exact randomization on resolved majorities."""

    eligible, excluded = _resolved_binary_items(items, majority_key)
    unique = [row for row in eligible if row["item_type"] == "unique"]
    control = [row for row in eligible if row["item_type"] == "control"]
    excluded_unique = sum(row["item_type"] == "unique" for row in excluded)
    excluded_control = sum(row["item_type"] == "control" for row in excluded)
    if not unique or not control:
        return {
            "unique_items": len(unique),
            "control_items": len(control),
            "excluded_unresolved_or_insufficient_unique": excluded_unique,
            "excluded_unresolved_or_insufficient_control": excluded_control,
            "fisher_two_sided_p": None,
            "exact_permutation_p": None,
            "analysis_variant": analysis_variant,
        }
    unique_same = sum(row[majority_key] == "same" for row in unique)
    control_same = sum(row[majority_key] == "same" for row in control)
    unique_not = len(unique) - unique_same
    control_not = len(control) - control_same
    total = len(eligible)
    total_same = unique_same + control_same
    observed_probability = _hypergeom_probability(
        unique_same, len(unique), total_same, total
    )
    lower = max(0, len(unique) - (total - total_same))
    upper = min(len(unique), total_same)
    fisher_p = 0.0
    exact_difference_p = 0.0
    observed_difference = control_same / len(control) - unique_same / len(unique)
    for possible_unique_same in range(lower, upper + 1):
        probability = _hypergeom_probability(
            possible_unique_same, len(unique), total_same, total
        )
        if probability <= observed_probability + 1e-15:
            fisher_p += probability
        possible_control_same = total_same - possible_unique_same
        possible_difference = possible_control_same / len(
            control
        ) - possible_unique_same / len(unique)
        if abs(possible_difference) >= abs(observed_difference) - 1e-15:
            exact_difference_p += probability
    if unique_not * control_same:
        odds_ratio: float | None = (
            unique_same * control_not / (unique_not * control_same)
        )
    elif unique_same * control_not == 0:
        odds_ratio = 0.0 if unique_not * control_same else None
    else:
        odds_ratio = None
    return {
        "unique_items": len(unique),
        "control_items": len(control),
        "excluded_unresolved_or_insufficient_unique": excluded_unique,
        "excluded_unresolved_or_insufficient_control": excluded_control,
        "unique_same_majority": unique_same,
        "unique_not_same_majority": unique_not,
        "control_same_majority": control_same,
        "control_not_same_majority": control_not,
        "unique_same_majority_rate": unique_same / len(unique),
        "control_same_majority_rate": control_same / len(control),
        "control_minus_unique_same_majority_rate": observed_difference,
        "odds_ratio_same_unique_vs_control": odds_ratio,
        "fisher_two_sided_p": min(1.0, fisher_p),
        "exact_permutation_p": min(1.0, exact_difference_p),
        "analysis_variant": analysis_variant,
        "endpoint": "strict resolved binary item majority; insufficient, Unable-majority, and no-consensus items are excluded",
    }


def blocked_exact_binary_comparison(
    items: Sequence[dict[str, Any]],
    *,
    majority_key: str = "majority_binary",
    analysis_variant: str = "primary",
) -> dict[str, Any]:
    """Exact randomization preserving control counts within model×direction."""

    eligible, excluded = _resolved_binary_items(items, majority_key)
    unique = [row for row in eligible if row["item_type"] == "unique"]
    control = [row for row in eligible if row["item_type"] == "control"]
    if not unique or not control:
        return {
            "analysis_variant": analysis_variant,
            "blocks": 0,
            "exact_blocked_permutation_p": None,
            "excluded_unresolved_or_insufficient": len(excluded),
        }
    blocks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        blocks[(row["model"], row["direction"])].append(row)

    # Distribution of the total number of Same outcomes assigned to controls.
    distribution: dict[int, float] = {0: 1.0}
    informative_blocks = 0
    for rows in blocks.values():
        n = len(rows)
        n_control = sum(row["item_type"] == "control" for row in rows)
        total_same = sum(row[majority_key] == "same" for row in rows)
        if 0 < n_control < n:
            informative_blocks += 1
        block_distribution = {
            possible: _hypergeom_probability(possible, n_control, total_same, n)
            for possible in range(
                max(0, n_control - (n - total_same)),
                min(n_control, total_same) + 1,
            )
        }
        updated: dict[int, float] = defaultdict(float)
        for accumulated, accumulated_probability in distribution.items():
            for possible, probability in block_distribution.items():
                updated[accumulated + possible] += accumulated_probability * probability
        distribution = dict(updated)

    n_control = len(control)
    n_unique = len(unique)
    total_same = sum(row[majority_key] == "same" for row in eligible)
    observed_control_same = sum(row[majority_key] == "same" for row in control)

    def difference(control_same: int) -> float:
        return control_same / n_control - (total_same - control_same) / n_unique

    observed_difference = difference(observed_control_same)
    p_value = sum(
        probability
        for control_same, probability in distribution.items()
        if abs(difference(control_same)) >= abs(observed_difference) - 1e-15
    )
    return {
        "analysis_variant": analysis_variant,
        "blocks": len(blocks),
        "informative_blocks": informative_blocks,
        "eligible_items": len(eligible),
        "excluded_unresolved_or_insufficient": len(excluded),
        "control_minus_unique_same_majority_rate": observed_difference,
        "exact_blocked_permutation_p": min(1.0, p_value),
        "endpoint": "resolved Same majority; exact randomization preserves the observed control count in every model×direction block",
    }


def blocked_exact_fraction_comparison(
    items: Sequence[dict[str, Any]], *, conservative_unable: bool = False
) -> dict[str, Any]:
    """Exact blocked randomization on item Same fractions.

    For the conservative variant, Unable is recoded after each hypothetical
    group assignment, so the randomization distribution honors the
    direction-dependent sensitivity rule rather than treating recoded outcomes
    as fixed.
    """

    eligible = [row for row in items if row["n_ratings"] >= 2]
    unique = [row for row in eligible if row["item_type"] == "unique"]
    control = [row for row in eligible if row["item_type"] == "control"]
    if not unique or not control:
        return {
            "analysis_variant": (
                "conservative_unable" if conservative_unable else "primary"
            ),
            "exact_blocked_fraction_p": None,
            "excluded_insufficient_items": len(items) - len(eligible),
        }

    def outcome(row: dict[str, Any], assigned_type: str) -> Fraction:
        same = row["same_count"]
        if conservative_unable and assigned_type == "unique":
            same += row["unable_count"]
        return Fraction(same, row["n_ratings"])

    blocks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        blocks[(row["model"], row["direction"])].append(row)

    # Joint distribution of summed control and unique Same fractions.
    distribution: dict[tuple[Fraction, Fraction], float] = {
        (Fraction(0), Fraction(0)): 1.0
    }
    for rows in blocks.values():
        n_control = sum(row["item_type"] == "control" for row in rows)
        assignments = list(itertools.combinations(range(len(rows)), n_control))
        block_distribution: Counter[tuple[Fraction, Fraction]] = Counter()
        for assignment in assignments:
            assigned_control = set(assignment)
            control_sum = Fraction(0)
            unique_sum = Fraction(0)
            for index, row in enumerate(rows):
                if index in assigned_control:
                    control_sum += outcome(row, "control")
                else:
                    unique_sum += outcome(row, "unique")
            block_distribution[(control_sum, unique_sum)] += 1
        updated: dict[tuple[Fraction, Fraction], float] = defaultdict(float)
        for prior, prior_probability in distribution.items():
            for current, count in block_distribution.items():
                key = (prior[0] + current[0], prior[1] + current[1])
                updated[key] += prior_probability * count / len(assignments)
        distribution = dict(updated)

    n_control = len(control)
    n_unique = len(unique)
    observed_control = sum((outcome(row, "control") for row in control), Fraction(0))
    observed_unique = sum((outcome(row, "unique") for row in unique), Fraction(0))
    observed_difference = float(
        observed_control / n_control - observed_unique / n_unique
    )
    p_value = 0.0
    for (control_sum, unique_sum), probability in distribution.items():
        difference = float(control_sum / n_control - unique_sum / n_unique)
        if abs(difference) >= abs(observed_difference) - 1e-15:
            p_value += probability
    return {
        "analysis_variant": (
            "conservative_unable" if conservative_unable else "primary"
        ),
        "blocks": len(blocks),
        "eligible_items": len(eligible),
        "excluded_insufficient_items": len(items) - len(eligible),
        "estimate": observed_difference,
        "exact_blocked_fraction_p": min(1.0, p_value),
        "endpoint": (
            "control minus unique adversarial Same fraction; exact assignments within model×direction and Unable recoded after assignment"
            if conservative_unable
            else "control minus unique Same fraction; exact assignments within model×direction"
        ),
    }


def item_label_permutation(
    items: Sequence[dict[str, Any]],
    iterations: int,
    seed: int,
    *,
    conservative_unable: bool = False,
) -> dict[str, Any]:
    eligible = [row for row in items if row["n_ratings"] >= 2]
    unique = [row for row in eligible if row["item_type"] == "unique"]
    control = [row for row in eligible if row["item_type"] == "control"]

    def outcome(row: dict[str, Any], assigned_type: str) -> float:
        if not conservative_unable:
            return row["same_fraction"]
        if assigned_type == "unique":
            return (row["same_count"] + row["unable_count"]) / row["n_ratings"]
        return row["same_count"] / row["n_ratings"]

    if not unique or not control:
        return {
            "estimate": None,
            "p_value": None,
            "iterations": iterations,
            "endpoint": "item Same-vote fraction",
            "excluded_insufficient_items": len(items) - len(eligible),
            "analysis_variant": (
                "conservative_unable" if conservative_unable else "primary"
            ),
        }
    observed = statistics.fmean(
        outcome(row, "control") for row in control
    ) - statistics.fmean(outcome(row, "unique") for row in unique)
    blocks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        blocks[(row["model"], row["direction"])].append(row)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        perm_control: list[float] = []
        perm_unique: list[float] = []
        for rows in blocks.values():
            n_control_in_block = sum(row["item_type"] == "control" for row in rows)
            control_indices = set(rng.sample(range(len(rows)), n_control_in_block))
            for index, row in enumerate(rows):
                assigned_type = "control" if index in control_indices else "unique"
                target = perm_control if assigned_type == "control" else perm_unique
                target.append(outcome(row, assigned_type))
        difference = statistics.fmean(perm_control) - statistics.fmean(perm_unique)
        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "estimate": observed,
        "p_value": ((extreme + 1) / (iterations + 1) if iterations > 0 else None),
        "iterations": iterations,
        "seed": seed,
        "blocks": len(blocks),
        "excluded_insufficient_items": len(items) - len(eligible),
        "analysis_variant": (
            "conservative_unable" if conservative_unable else "primary"
        ),
        "endpoint": (
            "control minus unique mean item adversarially recoded Same-vote fraction; "
            "two-sided labels permuted within model×direction"
            if conservative_unable
            else "control minus unique mean item Same-vote fraction; two-sided labels permuted within model×direction"
        ),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(first, second)
    )
    first_ss = sum((value - mean_first) ** 2 for value in first)
    second_ss = sum((value - mean_second) ** 2 for value in second)
    if first_ss == 0 or second_ss == 0:
        return None
    return numerator / math.sqrt(first_ss * second_ss)


def similarity_gradient(
    items: Sequence[dict[str, Any]], iterations: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum in B_STRATUM_ORDER:
        group = [row for row in items if row["stratum"] == stratum]
        n_judgments = sum(row["n_ratings"] for row in group)
        rows.append(
            {
                "stratum": stratum,
                "n_items": len(group),
                "n_judgments": n_judgments,
                "mean_similarity": (
                    statistics.fmean(row["similarity"] for row in group)
                    if group
                    else None
                ),
                "same_judgment_rate": (
                    sum(row["same_count"] for row in group) / n_judgments
                    if n_judgments
                    else None
                ),
                "mean_item_same_fraction": (
                    statistics.fmean(row["same_fraction"] for row in group)
                    if group
                    else None
                ),
                "same_majority_items": sum(
                    row["majority_binary"] == "same" for row in group
                ),
                "same_majority_rate": (
                    sum(row["majority_binary"] == "same" for row in group) / len(group)
                    if group
                    else None
                ),
            }
        )
    eligible = [row for row in items if row["n_ratings"] >= 2]
    similarities = [row["similarity"] for row in eligible]
    outcomes = [row["same_fraction"] for row in eligible]
    rho = _pearson(_average_ranks(similarities), _average_ranks(outcomes))
    permutation_p = None
    blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(eligible):
        blocks[(row["model"], row["direction"])].append(index)
    if rho is not None and iterations > 0:
        rng = random.Random(seed)
        extreme = 0
        similarity_ranks = _average_ranks(similarities)
        for _ in range(iterations):
            permuted = list(outcomes)
            for indices in blocks.values():
                block_values = [permuted[index] for index in indices]
                rng.shuffle(block_values)
                for index, value in zip(indices, block_values):
                    permuted[index] = value
            candidate = _pearson(similarity_ranks, _average_ranks(permuted))
            if candidate is not None and abs(candidate) >= abs(rho) - 1e-15:
                extreme += 1
        permutation_p = (extreme + 1) / (iterations + 1)
    trend = {
        "spearman_rho_similarity_vs_item_same_fraction": rho,
        "permutation_p_two_sided": permutation_p,
        "iterations": iterations,
        "seed": seed,
        "unit": "item",
        "blocks": len(blocks),
        "excluded_insufficient_items": len(items) - len(eligible),
        "permutation_scheme": "Same-fraction outcomes permuted within model×direction",
    }
    return rows, trend


def sensitivity_rows(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    definitions: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ()),
        ("item_type", ("item_type",)),
        ("item_type_x_direction", ("item_type", "direction")),
        ("item_type_x_stratum", ("item_type", "stratum")),
        (
            "item_type_x_direction_x_stratum",
            ("item_type", "direction", "stratum"),
        ),
    ]
    for sample in SAMPLES:
        sample_rows = [row for row in items if row["sample"] == sample]
        for grouping, keys in definitions:
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            if not keys:
                groups[()] = sample_rows
            else:
                for row in sample_rows:
                    groups[tuple(row[key] for key in keys)].append(row)
            for values, rows in sorted(
                groups.items(), key=lambda pair: tuple(map(str, pair[0]))
            ):
                n = len(rows)
                output: dict[str, Any] = {
                    "sample": sample,
                    "grouping": grouping,
                    "n_items": n,
                    "unable_judgments": sum(row["unable_count"] for row in rows),
                    "primary_supported_items": sum(
                        bool(row["primary_machine_supported"]) for row in rows
                    ),
                    "primary_supported_rate": (
                        sum(bool(row["primary_machine_supported"]) for row in rows) / n
                        if n
                        else None
                    ),
                    "conservative_supported_items": sum(
                        bool(row["conservative_machine_supported"]) for row in rows
                    ),
                    "conservative_supported_rate": (
                        sum(bool(row["conservative_machine_supported"]) for row in rows)
                        / n
                        if n
                        else None
                    ),
                    "conservative_rule": (
                        "Unable→Same for unique; Unable→no-Same for controls"
                    ),
                }
                output.update({key: value for key, value in zip(keys, values)})
                result.append(output)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    summary: Mapping[str, Any],
    a_distribution: Sequence[dict[str, Any]],
    b_gradient: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Frozen human-annotation analysis",
        "",
        "This report is descriptive evidence for the two targeted audits. It does not establish universal metric validity or an exhaustive human recomputation of UCR.",
        "",
        "## QC samples",
        "",
        "| Study | Sample | Sessions | Coded judgments |",
        "|---|---|---:|---:|",
    ]
    for study in ("A", "B"):
        for sample in SAMPLES:
            row = summary["qc"]["by_study_sample"][study][sample]
            lines.append(
                f"| {study} | {sample} | {row['sessions']} | {row['coded_judgments']} |"
            )

    lines.extend(
        [
            "",
            "## Study A: matched persona-depth contrasts",
            "",
            "| Sample | Contrast | Items | Role | Other | Tie | Unable | D = P(Other) − P(Role) [95% triplet-cluster CI] | Nominal α |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in SAMPLES:
        for contrast, analysis in summary["study_a"][sample]["contrasts"].items():
            other = analysis["other_condition"]
            counts = analysis["raw_counts"]
            bootstrap = analysis["contrast_difference"]
            ci = (
                f"{_format_number(bootstrap['estimate'])} "
                f"[{_format_number(bootstrap['ci_low'])}, "
                f"{_format_number(bootstrap['ci_high'])}]"
            )
            lines.append(
                f"| {sample} | {contrast} | {analysis['items']} | "
                f"{counts['role']} | {counts[other]} | {counts['tie']} | "
                f"{counts['unable']} | {ci} | "
                f"{_format_number(analysis['agreement']['krippendorff_alpha_nominal'])} |"
            )

    lines.extend(
        [
            "",
            "### Study A directional interpretations",
            "",
            "| Sample | Contrast | Prespecified descriptive wording |",
            "|---|---|---|",
        ]
    )
    for sample in SAMPLES:
        for contrast, analysis in summary["study_a"][sample]["contrasts"].items():
            wording = analysis["directional_interpretation"]["report_wording"]
            lines.append(f"| {sample} | {contrast} | {wording} |")

    lines.extend(
        [
            "",
            "### Study A length and participant-cluster sensitivities",
            "",
            "| Sample | Contrast | ≤1.5× items | Longer selected among directional | Matched D [95% triplet CI] | All-item D [95% participant CI] | Length–preference r |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in SAMPLES:
        for contrast, analysis in summary["study_a"][sample]["contrasts"].items():
            diagnostics = analysis["length_diagnostics"]
            matched = diagnostics[
                "matched_length_contrast_triplet_bootstrap"
            ]
            participant = analysis["participant_cluster_sensitivity"]
            matched_ci = (
                f"{_format_number(matched['estimate'])} "
                f"[{_format_number(matched['ci_low'])}, "
                f"{_format_number(matched['ci_high'])}]"
            )
            participant_ci = (
                f"{_format_number(participant['estimate'])} "
                f"[{_format_number(participant['ci_low'])}, "
                f"{_format_number(participant['ci_high'])}]"
            )
            lines.append(
                f"| {sample} | {contrast} | "
                f"{diagnostics['length_matched_items_1_5x']} | "
                f"{_format_number(diagnostics['longer_selected_among_directional'])} | "
                f"{matched_ci} | {participant_ci} | "
                f"{_format_number(diagnostics['item_level_pearson_signed_log_length_vs_contrast_preference'])} |"
            )

    lines.extend(
        [
            "",
            "Each contrast is analyzed separately. Every comparison has exactly two human ratings; no item-level majority is manufactured. Both ratings are retained, so opposite directional judgments contribute a net item score of zero.",
            "D is P(Basic) − P(Role) for Role–Basic and P(Pro) − P(Role) for Role–Pro, using +1 for the non-Role condition, −1 for Role, and 0 for Tie/Unable. The primary CI resamples matched triplets within model and retains all ratings for each sampled comparison.",
            "The participant-cluster sensitivity instead keeps all judgments from each participant together. The ≤1.5× analysis retains natural full responses but removes comparison items whose longer response exceeds 1.5 times the shorter response. These are diagnostics, not post-hoc exclusion rules.",
            "A confidence interval containing zero does not establish equivalence; the supported wording is limited to whether participants showed a consistent directional preference.",
            "",
            "### Study A confidence and response time",
            "",
            "| Sample / contrast | Confidence available | Missing | Mean confidence | Median confidence | Median primary seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in SAMPLES:
        for contrast, analysis in summary["study_a"][sample]["contrasts"].items():
            diagnostic = analysis["confidence_and_timing"]
            lines.append(
                f"| {sample} / {contrast} | "
                f"{diagnostic['confidence_available']} | "
                f"{diagnostic['confidence_missing']} | "
                f"{_format_number(diagnostic['confidence']['mean'])} | "
                f"{_format_number(diagnostic['confidence']['median'])} | "
                f"{_format_number(diagnostic['primary_duration_seconds']['median'])} |"
            )
    lines.extend(
        [
            "",
            "Confidence and response time are descriptive secondary measures only. They do not reweight judgments or trigger automatic exclusion.",
            "",
            "## Study B: architecture audit",
            "",
            "| Sample | Unique same-majority | Control same-majority | Control − unique | Fisher p | Block-exact p | Blocked item-label p | 4-way nominal α |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in SAMPLES:
        analysis = summary["study_b"][sample]
        exact = analysis["exact_comparison"]
        blocked = analysis["blocked_exact_fraction_comparison"]
        perm = analysis["item_label_permutation"]
        lines.append(
            f"| {sample} | {_format_number(exact.get('unique_same_majority_rate'))} | "
            f"{_format_number(exact.get('control_same_majority_rate'))} | "
            f"{_format_number(exact.get('control_minus_unique_same_majority_rate'))} | "
            f"{_format_number(exact.get('fisher_two_sided_p'))} | "
            f"{_format_number(blocked.get('exact_blocked_fraction_p'))} | "
            f"{_format_number(perm.get('p_value'))} | "
            f"{_format_number(analysis['agreement_4way']['krippendorff_alpha_nominal'])} |"
        )

    lines.extend(
        [
            "",
            "Unresolved binary majorities and items with fewer than two retained ratings are reported but excluded from the exact majority comparison.",
            "",
            "### Conservative Unable inference",
            "",
            "| Sample | Unique same-majority | Control same-majority | Control − unique | Fisher p | Block-exact p | Blocked item-label p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in SAMPLES:
        sensitivity = summary["study_b"][sample]["conservative_unable"]
        exact = sensitivity["exact_comparison"]
        blocked = sensitivity["blocked_exact_fraction_comparison"]
        perm = sensitivity["item_label_permutation"]
        lines.append(
            f"| {sample} | {_format_number(exact.get('unique_same_majority_rate'))} | "
            f"{_format_number(exact.get('control_same_majority_rate'))} | "
            f"{_format_number(exact.get('control_minus_unique_same_majority_rate'))} | "
            f"{_format_number(exact.get('fisher_two_sided_p'))} | "
            f"{_format_number(blocked.get('exact_blocked_fraction_p'))} | "
            f"{_format_number(perm.get('p_value'))} |"
        )

    lines.extend(
        [
            "",
            "### Similarity gradient (QC-filtered)",
            "",
            "| Stratum | Items | Mean similarity | Same judgment rate | Same-majority rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in b_gradient:
        if row["sample"] != "qc_filtered":
            continue
        lines.append(
            f"| {row['stratum']} | {row['n_items']} | {_format_number(row['mean_similarity'])} | "
            f"{_format_number(row['same_judgment_rate'])} | {_format_number(row['same_majority_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The conservative sensitivity recodes every Unable judgment against the machine label: Unable→Same for machine-unique items and Unable→no-Same for overlap controls. See `study_b_unable_sensitivity.csv`.",
            "",
            "Study B is a targeted top-3 retrieval audit; top-3 candidates do not exhaust the opposite architecture's viewpoint pool.",
            "",
        ]
    )
    telemetry = summary.get("telemetry", {})
    if telemetry.get("present"):
        elapsed = telemetry["session_max_elapsed_seconds"]
        lines.extend(
            [
                "## Descriptive telemetry",
                "",
                f"- Events: {telemetry['events']}",
                f"- Sessions with events: {telemetry['sessions_with_events']}",
                f"- Median maximum recorded elapsed time: {_format_number(elapsed['median'])} seconds",
                "- Telemetry is retained for descriptive process auditing only; it is not an automatic rejection or scientific-exclusion rule.",
                "",
            ]
        )
    if summary.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
        lines.append("")
    return "\n".join(lines)


def analyze(
    *,
    sessions_path: Path,
    responses_path: Path,
    events_path: Path | None = None,
    outdir: Path,
    a_manifest_path: Path = DEFAULT_A_MANIFEST,
    a_bundles_path: Path = DEFAULT_A_BUNDLES,
    a_public_path: Path = DEFAULT_A_PUBLIC,
    b_provenance_path: Path = DEFAULT_B_PROVENANCE,
    b_bundles_path: Path = DEFAULT_B_BUNDLES,
    b_validation_path: Path = DEFAULT_B_VALIDATION,
    bootstrap_iterations: int = 10_000,
    permutation_iterations: int = 10_000,
    seed: int = 20_260_727,
    exclude_prolific_study_ids: Sequence[str] = (),
    allow_repeat_study_a_participant: bool = False,
) -> dict[str, Any]:
    warnings: list[str] = []
    normalized_excluded_study_ids = [
        _text(study_id) for study_id in exclude_prolific_study_ids
    ]
    if any(not study_id for study_id in normalized_excluded_study_ids):
        raise AnalysisError("Excluded Prolific study IDs must be non-empty")
    excluded_study_ids = frozenset(normalized_excluded_study_ids)
    metadata = load_metadata(
        a_manifest_path,
        a_bundles_path,
        a_public_path,
        b_provenance_path,
        b_bundles_path,
        b_validation_path,
    )
    session_rows, paid_field_present = canonical_sessions(
        load_export(sessions_path, "sessions")
    )
    if paid_field_present:
        missing_payment = [
            row["session_id"]
            for row in session_rows
            if row["paid"] is None and not _is_demo_session(row)
        ]
        if missing_payment:
            preview = ", ".join(missing_payment[:5])
            raise AnalysisError(
                f"Partial payment join: {len(missing_payment)} non-demo session(s) "
                f"have blank paid status ({preview})"
            )
    response_rows = canonical_responses(
        load_export(responses_path, "responses"), warnings
    )
    raw_event_rows, event_source_path = load_optional_events(
        events_path, (responses_path, sessions_path)
    )
    canonical_event_rows = canonical_events(raw_event_rows, warnings)
    qc_rows, sample_ids = assess_qc(
        session_rows,
        response_rows,
        metadata,
        paid_field_present,
        excluded_study_ids,
        warnings,
    )
    telemetry_events, telemetry_sessions = telemetry_output_rows(
        canonical_event_rows, session_rows, qc_rows, warnings
    )
    paid_pid_counts = Counter(
        (row["study_type"], row["prolific_pid"])
        for row in qc_rows
        if row["in_all_paid"]
    )
    repeated_paid_participants = [
        key for key, count in paid_pid_counts.items() if count > 1
    ]
    disallowed_repeated_participants = [
        key
        for key in repeated_paid_participants
        if not (allow_repeat_study_a_participant and key[0] == "A")
    ]
    if disallowed_repeated_participants:
        raise AnalysisError(
            f"{len(disallowed_repeated_participants)} paid participant/study pair(s) "
            "have multiple sessions; exactly one paid session per participant "
            "per study is required"
        )
    if repeated_paid_participants:
        warnings.append(
            f"Documented override retained {len(repeated_paid_participants)} "
            "repeated Study A participant(s); participant-cluster bootstrap "
            "groups all of each participant's judgments together"
        )
    paid_participants = {
        study: {
            row["prolific_pid"]
            for row in qc_rows
            if row["study_type"] == study and row["in_all_paid"]
        }
        for study in ("A", "B")
    }
    cross_study_participants = paid_participants["A"] & paid_participants["B"]
    if cross_study_participants:
        raise AnalysisError(
            f"{len(cross_study_participants)} paid participant(s) appear in both "
            "Study A and B; the frozen design requires disjoint samples"
        )
    rationale_rows = rationale_review_rows(
        session_rows, response_rows, metadata, qc_rows
    )
    a_judgments, b_judgments = extract_judgments(
        session_rows, response_rows, metadata, sample_ids, warnings
    )
    for study, rows in (("A", a_judgments), ("B", b_judgments)):
        participant_item_counts = Counter(
            (row["prolific_pid"], row["item_id"])
            for row in rows
            if row["sample"] == "all_paid" and row["prolific_pid"]
        )
        duplicates = [
            key for key, count in participant_item_counts.items() if count > 1
        ]
        if duplicates:
            if study == "A" and allow_repeat_study_a_participant:
                warnings.append(
                    f"Documented override retained {len(duplicates)} Study A "
                    "participant/item repeat(s); interpret agreement with the "
                    "first-exposure-deduplicated sensitivity"
                )
            else:
                raise AnalysisError(
                    f"Study {study} has {len(duplicates)} participant/item duplicate(s); "
                    "resolve duplicate sessions before analysis"
                )
    a_items = aggregate_a_items(
        a_judgments, metadata.a_ratings_per_item
    )
    b_items = aggregate_b_items(b_judgments)
    a_distribution = _distribution_rows(
        a_judgments,
        A_CODES,
        ("contrast", "model", "subcategory"),
    )
    b_distribution = _distribution_rows(
        b_judgments, B_CODES, ("item_type", "direction", "stratum", "model")
    )
    a_models = a_model_descriptives(a_judgments, a_items)
    a_confidence_timing = confidence_timing_rows(a_judgments)
    b_groups = b_group_summaries(b_items)
    b_sensitivity = sensitivity_rows(b_items)

    agreement_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "configuration": {
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "excluded_prolific_study_ids": sorted(excluded_study_ids),
            "allow_repeat_study_a_participant": (
                allow_repeat_study_a_participant
            ),
            "frozen_design": {
                "study_a": {
                    "unique_items": len(metadata.a_items),
                    "matched_triplets": len(
                        {
                            item["triplet_id"]
                            for item in metadata.a_items.values()
                        }
                    ),
                    "items_by_contrast": dict(
                        sorted(
                            Counter(
                                item["contrast"]
                                for item in metadata.a_items.values()
                            ).items()
                        )
                    ),
                    "bundles": len(metadata.a_bundle_items),
                    "items_per_bundle": metadata.a_items_per_bundle,
                    "ratings_per_item": metadata.a_ratings_per_item,
                    "expected_judgments": metadata.a_expected_judgments,
                },
                "study_b": {
                    "unique_items": len(metadata.b_items),
                    "bundles": len(metadata.b_bundle_items),
                    "expected_judgments": sum(
                        map(len, metadata.b_bundle_items.values())
                    ),
                },
            },
            "input_sha256": {
                "sessions": _sha256(sessions_path),
                "responses": _sha256(responses_path),
                "events": (
                    _sha256(event_source_path)
                    if event_source_path is not None
                    else None
                ),
                "study_a_manifest": _sha256(a_manifest_path),
                "study_a_bundles": _sha256(a_bundles_path),
                "study_a_public": _sha256(a_public_path),
                "study_b_provenance": _sha256(b_provenance_path),
                "study_b_bundles": _sha256(b_bundles_path),
                "study_b_validation": _sha256(b_validation_path),
            },
        },
        "qc": {
            "paid_field_present": paid_field_present,
            "excluded_pilot_sessions": sum(
                bool(row["pilot_excluded"]) for row in qc_rows
            ),
            "excluded_paid_eligible_pilot_sessions": sum(
                bool(row["pilot_excluded"])
                and bool(row["paid_analysis_eligible_before_pilot_exclusion"])
                for row in qc_rows
            ),
            "by_study_sample": {},
        },
        "study_a": {},
        "study_b": {},
        "telemetry": telemetry_summary(
            telemetry_events, telemetry_sessions
        ),
        "warnings": sorted(set(warnings)),
    }
    for study, judgments in (("A", a_judgments), ("B", b_judgments)):
        summary["qc"]["by_study_sample"][study] = {}
        for sample in SAMPLES:
            summary["qc"]["by_study_sample"][study][sample] = {
                "sessions": sum(
                    row["study_type"] == study
                    and bool(
                        row["in_all_paid" if sample == "all_paid" else "in_qc_filtered"]
                    )
                    for row in qc_rows
                ),
                "coded_judgments": sum(row["sample"] == sample for row in judgments),
            }

    gradient_output: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(SAMPLES):
        sample_a_j = [row for row in a_judgments if row["sample"] == sample]
        sample_a_i = [row for row in a_items if row["sample"] == sample]
        raw_a = Counter(row["code"] for row in sample_a_j)
        complete_a_items = sum(
            row["complete_design_ratings"] for row in sample_a_i
        )
        summary["study_a"][sample] = {
            "items": len(sample_a_i),
            "complete_design_items": complete_a_items,
            "judgments": len(sample_a_j),
            "raw_counts": {code: raw_a[code] for code in A_CODES},
            "raw_proportions": {
                code: raw_a[code] / len(sample_a_j) if sample_a_j else None
                for code in A_CODES
            },
            "confidence_and_timing": study_a_confidence_timing(sample_a_j),
            "contrasts": {},
        }
        available_contrasts = sorted(
            {row["contrast"] for row in sample_a_i}
            | {
                item["contrast"]
                for item in metadata.a_items.values()
            }
        )
        for contrast_index, contrast in enumerate(available_contrasts):
            contrast_j = [
                row for row in sample_a_j if row["contrast"] == contrast
            ]
            contrast_i = [
                row for row in sample_a_i if row["contrast"] == contrast
            ]
            other_condition = (
                "basic" if contrast == "role_basic" else "pro"
            )
            a_agreement = nominal_agreement(contrast_j)
            agreement_rows.append(
                {
                    "study": "A",
                    "sample": sample,
                    "contrast": contrast,
                    "coding": (
                        f"Role/{other_condition.title()}/Tie/Unable nominal"
                    ),
                    **a_agreement,
                }
            )
            contrast_raw = Counter(row["code"] for row in contrast_j)
            pair_patterns = Counter(
                row["pair_pattern"] for row in contrast_i
            )
            pattern_labels = (
                "agreement_role",
                f"agreement_{other_condition}",
                "agreement_tie",
                "agreement_unable",
                "disagreement",
                "incomplete",
            )
            triplet_bootstrap = bootstrap_contrast_by_triplet(
                contrast_i,
                bootstrap_iterations,
                seed + sample_index * 10 + contrast_index,
            )
            summary["study_a"][sample]["contrasts"][contrast] = {
                "other_condition": other_condition,
                "items": len(contrast_i),
                "complete_design_items": sum(
                    row["complete_design_ratings"] for row in contrast_i
                ),
                "judgments": len(contrast_j),
                "raw_counts": {
                    code: contrast_raw[code]
                    for code in ("role", other_condition, "tie", "unable")
                },
                "raw_proportions": {
                    code: (
                        contrast_raw[code] / len(contrast_j)
                        if contrast_j
                        else None
                    )
                    for code in ("role", other_condition, "tie", "unable")
                },
                "pair_pattern_counts": {
                    label: pair_patterns[label]
                    for label in pattern_labels
                },
                "pair_pattern_proportions": {
                    label: (
                        pair_patterns[label] / len(contrast_i)
                        if contrast_i
                        else None
                    )
                    for label in pattern_labels
                },
                "contrast_difference": triplet_bootstrap,
                "participant_cluster_sensitivity": (
                    bootstrap_contrast_by_participant(
                        contrast_j,
                        bootstrap_iterations,
                        seed + 100 + sample_index * 10 + contrast_index,
                    )
                ),
                "directional_interpretation": directional_interpretation(
                    triplet_bootstrap, other_condition
                ),
                "confidence_and_timing": study_a_confidence_timing(
                    contrast_j
                ),
                "length_diagnostics": study_a_length_diagnostics(
                    contrast_j,
                    contrast_i,
                    bootstrap_iterations,
                    seed + 200 + sample_index * 10 + contrast_index,
                ),
                "agreement": a_agreement,
            }

        sample_b_j = [row for row in b_judgments if row["sample"] == sample]
        sample_b_i = [row for row in b_items if row["sample"] == sample]
        b_agreement = nominal_agreement(sample_b_j)
        binary_judgments = [
            {
                **row,
                "binary_code": (
                    "same"
                    if row["code"] == "same"
                    else "no_same"
                    if row["code"] in {"related", "different"}
                    else "unable"
                ),
            }
            for row in sample_b_j
        ]
        b_binary_agreement = nominal_agreement(binary_judgments, code_key="binary_code")
        agreement_rows.extend(
            [
                {
                    "study": "B",
                    "sample": sample,
                    "coding": "Same/Related/Different/Unable nominal",
                    **b_agreement,
                },
                {
                    "study": "B",
                    "sample": sample,
                    "coding": "Same/no-Same/Unable nominal",
                    **b_binary_agreement,
                },
            ]
        )
        exact = exact_binary_comparison(sample_b_i)
        blocked_exact = blocked_exact_binary_comparison(sample_b_i)
        blocked_fraction_exact = blocked_exact_fraction_comparison(sample_b_i)
        item_permutation = item_label_permutation(
            sample_b_i, permutation_iterations, seed + 100 + sample_index
        )
        conservative_exact = exact_binary_comparison(
            sample_b_i,
            majority_key="conservative_unable_majority",
            analysis_variant="conservative_unable",
        )
        conservative_blocked_fraction_exact = blocked_exact_fraction_comparison(
            sample_b_i, conservative_unable=True
        )
        conservative_permutation = item_label_permutation(
            sample_b_i,
            permutation_iterations,
            seed + 150 + sample_index,
            conservative_unable=True,
        )
        gradient, trend = similarity_gradient(
            sample_b_i, permutation_iterations, seed + 200 + sample_index
        )
        for row in gradient:
            gradient_output.append({"sample": sample, **row})
        comparison_rows.extend(
            [
                {"sample": sample, "test": "fisher_and_exact_binary", **exact},
                {
                    "sample": sample,
                    "test": "blocked_exact_binary",
                    **blocked_exact,
                },
                {
                    "sample": sample,
                    "test": "blocked_exact_same_fraction",
                    **blocked_fraction_exact,
                },
                {
                    "sample": sample,
                    "test": "item_label_permutation",
                    **item_permutation,
                },
                {
                    "sample": sample,
                    "test": "conservative_fisher_and_exact_binary",
                    **conservative_exact,
                },
                {
                    "sample": sample,
                    "test": "conservative_blocked_exact_same_fraction",
                    **conservative_blocked_fraction_exact,
                },
                {
                    "sample": sample,
                    "test": "conservative_item_label_permutation",
                    **conservative_permutation,
                },
                {"sample": sample, "test": "similarity_trend", **trend},
            ]
        )
        raw_b = Counter(row["code"] for row in sample_b_j)
        summary["study_b"][sample] = {
            "items": len(sample_b_i),
            "complete_triplicate_items": sum(
                row["complete_triplicate"] for row in sample_b_i
            ),
            "judgments": len(sample_b_j),
            "raw_counts": {code: raw_b[code] for code in B_CODES},
            "raw_proportions": {
                code: raw_b[code] / len(sample_b_j) if sample_b_j else None
                for code in B_CODES
            },
            "item_4way_majorities": dict(
                Counter(row["majority_4way"] for row in sample_b_i)
            ),
            "item_binary_majorities": dict(
                Counter(row["majority_binary"] for row in sample_b_i)
            ),
            "exact_comparison": exact,
            "blocked_exact_majority_comparison": blocked_exact,
            "blocked_exact_fraction_comparison": blocked_fraction_exact,
            "item_label_permutation": item_permutation,
            "similarity_trend": trend,
            "agreement_4way": b_agreement,
            "agreement_binary": b_binary_agreement,
            "conservative_unable": {
                "primary_supported_items": sum(
                    row["primary_machine_supported"] for row in sample_b_i
                ),
                "conservative_supported_items": sum(
                    row["conservative_machine_supported"] for row in sample_b_i
                ),
                "total_items": len(sample_b_i),
                "rule": "Unable→Same for unique; Unable→no-Same for controls",
                "exact_comparison": conservative_exact,
                "blocked_exact_fraction_comparison": (
                    conservative_blocked_fraction_exact
                ),
                "item_label_permutation": conservative_permutation,
            },
        }

    # Coverage warnings are informative during the pilot and become hard review
    # checks before reporting the final frozen-design results.
    expected_a_items = len(metadata.a_items)
    expected_b_items = len(metadata.b_items)
    for sample in SAMPLES:
        a_count = len({row["item_id"] for row in a_items if row["sample"] == sample})
        b_count = len({row["item_id"] for row in b_items if row["sample"] == sample})
        if a_count != expected_a_items:
            warnings.append(
                f"{sample}: observed {a_count}/{expected_a_items} Study A items "
                "(expected during partial/pilot exports)"
            )
        if b_count != expected_b_items:
            warnings.append(
                f"{sample}: observed {b_count}/{expected_b_items} Study B items "
                "(expected during partial/pilot exports)"
            )
        if sample == "all_paid":
            incomplete_a = sum(
                not row["complete_design_ratings"]
                for row in a_items
                if row["sample"] == sample
            )
            incomplete_b = sum(
                not row["complete_design_ratings"]
                for row in b_items
                if row["sample"] == sample
            )
            if incomplete_a:
                warnings.append(
                    f"all_paid: {incomplete_a} Study A items do not have "
                    f"exactly {metadata.a_ratings_per_item} coded ratings"
                )
            if incomplete_b:
                warnings.append(
                    f"all_paid: {incomplete_b} Study B items do not have "
                    f"exactly {EXPECTED_B_RATINGS_PER_ITEM} coded ratings"
                )

    summary["warnings"] = sorted(set(warnings))

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "session_qc.csv", qc_rows)
    write_csv(outdir / "rationale_review.csv", rationale_rows)
    write_csv(outdir / "study_a_judgments.csv", a_judgments)
    write_csv(outdir / "study_a_item_summaries.csv", a_items)
    write_csv(outdir / "study_a_raw_distribution.csv", a_distribution)
    write_csv(outdir / "study_a_model_descriptives.csv", a_models)
    write_csv(
        outdir / "study_a_confidence_timing.csv", a_confidence_timing
    )
    write_csv(outdir / "study_b_judgments.csv", b_judgments)
    write_csv(outdir / "study_b_item_majorities.csv", b_items)
    write_csv(outdir / "study_b_raw_distribution.csv", b_distribution)
    write_csv(outdir / "study_b_group_summaries.csv", b_groups)
    write_csv(outdir / "study_b_similarity_gradient.csv", gradient_output)
    write_csv(outdir / "study_b_exact_and_permutation.csv", comparison_rows)
    write_csv(outdir / "study_b_unable_sensitivity.csv", b_sensitivity)
    write_csv(outdir / "agreement.csv", agreement_rows)
    write_csv(outdir / "telemetry_events.csv", telemetry_events)
    write_csv(outdir / "telemetry_session_summary.csv", telemetry_sessions)
    (outdir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (outdir / "report.md").write_text(
        build_report(summary, a_distribution, gradient_output), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frozen Study A/B D1 session and response exports."
    )
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help=(
            "Optional study_events CSV/JSON export. If omitted, events are "
            "auto-discovered when --sessions or --responses points to the "
            "combined site export."
        ),
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--study-a-manifest", type=Path, default=DEFAULT_A_MANIFEST)
    parser.add_argument("--study-a-bundles", type=Path, default=DEFAULT_A_BUNDLES)
    parser.add_argument("--study-a-public", type=Path, default=DEFAULT_A_PUBLIC)
    parser.add_argument("--study-b-provenance", type=Path, default=DEFAULT_B_PROVENANCE)
    parser.add_argument("--study-b-bundles", type=Path, default=DEFAULT_B_BUNDLES)
    parser.add_argument("--study-b-validation", type=Path, default=DEFAULT_B_VALIDATION)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--exclude-prolific-study-id",
        action="append",
        default=[],
        metavar="STUDY_ID",
        help=(
            "Exclude every session from this Prolific study ID from all_paid "
            "and qc_filtered while retaining it in session_qc.csv. Repeat for "
            "multiple superseded pilot listings. Valid scientific-pilot "
            "sessions are included by default."
        ),
    )
    parser.add_argument(
        "--allow-repeat-study-a-participant",
        action="store_true",
        help=(
            "Retain a documented repeated Study A participant instead of "
            "failing the design-integrity gate. Participant-cluster bootstrap "
            "groups both sessions by Prolific PID; any repeated participant/item "
            "rating is retained with an explicit warning and should be paired "
            "with a first-exposure-deduplicated sensitivity."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_iterations < 0 or args.permutation_iterations < 0:
        raise AnalysisError("Iteration counts must be non-negative")
    summary = analyze(
        sessions_path=args.sessions,
        responses_path=args.responses,
        events_path=args.events,
        outdir=args.outdir,
        a_manifest_path=args.study_a_manifest,
        a_bundles_path=args.study_a_bundles,
        a_public_path=args.study_a_public,
        b_provenance_path=args.study_b_provenance,
        b_bundles_path=args.study_b_bundles,
        b_validation_path=args.study_b_validation,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
        exclude_prolific_study_ids=args.exclude_prolific_study_id,
        allow_repeat_study_a_participant=(
            args.allow_repeat_study_a_participant
        ),
    )
    warning_count = len(summary["warnings"])
    print(f"Wrote analysis to {args.outdir} ({warning_count} warning(s)).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
