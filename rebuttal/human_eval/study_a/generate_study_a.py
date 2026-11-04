#!/usr/bin/env python3
"""Generate the formal dual-contrast Study A annotation sample.

The selection unit is a matched (model, question, persona) triplet containing
the complete Single-Call outputs for Role, Basic, and Pro.  Selection is
outcome-blind: after a symmetric readability screen, the optimizer sees only
metadata and a fixed-seed hash.  Each selected triplet yields two blinded
comparisons, Role--Basic and Role--Pro.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


DEFAULT_SEED = 20260727
# Root of a full local experiment tree with raw*/ Single-Call outputs for all seven models
# (not part of the public release); pass --modbench-root to point at it.
DEFAULT_MODBENCH_ROOT = Path(__file__).resolve().parents[3]
MODEL_SPECS = (
    ("deepseek", "raw"),
    ("gemini", "raw_gemini"),
    ("gpt54mini", "raw_gpt54mini"),
    ("grok", "raw_grok"),
    ("kimi", "raw_kimi"),
    ("llama", "raw_llama"),
    ("qwen", "raw_qwen"),
)
CONDITION_DIRS = {
    "role": "independent-role",
    "basic": "independent-basic",
    "pro": "independent-pro",
}
CONTRASTS = {
    "role_basic": ("role", "basic", "RB"),
    "role_pro": ("role", "pro", "RP"),
}
N_TRIPLETS = 42
TRIPLETS_PER_MODEL = 6
N_COMPARISONS = 84
N_BUNDLES = 28
ITEMS_PER_BUNDLE = 6
RATERS_PER_COMPARISON = 2
RATIONALES_PER_BUNDLE = 1
MAX_RESPONSE_WORDS = 325
MAX_TRIPLET_WORD_RANGE = 30
WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)
LETTER_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)
NON_LATIN_SCRIPT_RE = re.compile(
    "["
    "\u0370-\u052f"  # Greek and Cyrillic
    "\u0590-\u08ff"  # Hebrew and Arabic
    "\u0900-\u0dff"  # Indic scripts
    "\u3040-\u30ff"  # Japanese kana
    "\u3400-\u9fff"  # CJK
    "\uac00-\ud7af"  # Hangul
    "]"
)
ENGLISH_FUNCTION_WORDS = frozenset(
    """
    the a an and or but if then than that this these those is are was were be
    been being to of in on for from with without as at by it its you your yours
    we our they their he she his her not yes do does did can could should would
    may might will have has had all any some more most less very only just into
    about over under after before when where why how what which who while
    because so each every another also
    """.split()
)
NON_ENGLISH_LATIN_FUNCTION_WORDS = frozenset(
    """
    que de la el los las una para con como pero por del al se su sus más muy
    todo toda todos todas esto esta estos estas porque cuando donde sin nada ni
    ya también todavía aquí así mi mis tú usted ustedes nosotros ellos ellas
    eso esa unos unas
    le les des du et les une dans pour avec mais pas sur au aux ce cette ces
    est sont était être qui quoi où pourquoi quand sans très plus moins aussi
    nous vous ils elles leur leurs
    der die das den dem des ein eine einer einen und oder aber nicht ist sind
    war waren sein mit für von zu im auf aus als bei durch über unter nach vor
    wenn wo warum wie was welche wer sehr mehr weniger auch wir ihr sie
    il lo gli le un uno una dei degli delle di da dal con per tra fra che non
    sono era essere questo questa questi queste perché quando dove senza molto
    più meno anche noi voi loro
    os as um uma uns umas do da dos das em no na nos nas por para com sem mas
    não que se seu sua seus suas este esta estes estas porque quando onde muito
    mais menos também nós vocês eles elas
    """.split()
)
PRIMARY_ENGLISH_RULE_VERSION = "primary-english-v1"
PRIMARY_ENGLISH_MAX_NON_LATIN_LETTER_FRACTION = 0.20
PRIMARY_ENGLISH_MIN_NON_ENGLISH_FUNCTION_HITS = 20
PRIMARY_ENGLISH_NON_ENGLISH_TO_ENGLISH_RATIO = 2.0
TECHNICAL_TASK_EXCLUSIONS = {
    "MOD-047": "The source benchmark question is visibly truncated at 'Female Cr'.",
}
PROVISIONAL_FREEZE_REFERENCES = {
    (
        "grok",
        "MOD-090",
        "Maria Rodriguez",
    ): "SA-T019 in the prelaunch provisional freeze",
}


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    line_number: int
    record: dict[str, Any]


@dataclass(frozen=True)
class Triplet:
    triplet_id: str
    model_code: str
    result_dir: str
    model_api: str
    task_id: str
    query: str
    subcategory: str
    persona_id: int
    persona_name: str
    persona_prompts: dict[str, str]
    sources: dict[str, SourceRecord]
    responses: dict[str, str]
    word_counts: dict[str, int]


@dataclass(frozen=True)
class Comparison:
    item_id: str
    triplet_index: int
    contrast: str
    left_condition: str
    other_condition: str
    suffix: str
    displayed_word_total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--modbench-root", type=Path, default=DEFAULT_MODBENCH_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    return parser.parse_args()


def namespace_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def rng_for(seed: int, namespace: str) -> random.Random:
    return random.Random(namespace_seed(seed, namespace))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def primary_english_audit(text: str) -> dict[str, Any]:
    """Apply a symmetric, content-direction-blind English readability screen.

    This is deliberately a conservative language check, not a quality,
    relevance, refusal, or semantic-richness filter.  It rejects only responses
    that are clearly dominated by a non-Latin script or by common non-English
    function words relative to English function words.
    """

    tokens = [token.casefold() for token in WORD_RE.findall(text)]
    letters = LETTER_RE.findall(text)
    non_latin_letters = NON_LATIN_SCRIPT_RE.findall(text)
    english_hits = sum(token in ENGLISH_FUNCTION_WORDS for token in tokens)
    non_english_hits = sum(
        token in NON_ENGLISH_LATIN_FUNCTION_WORDS for token in tokens
    )
    non_latin_fraction = (
        len(non_latin_letters) / len(letters) if letters else 0.0
    )
    failure_reasons: list[str] = []
    if (
        non_latin_fraction
        > PRIMARY_ENGLISH_MAX_NON_LATIN_LETTER_FRACTION
    ):
        failure_reasons.append("non_latin_script_dominates")
    if (
        non_english_hits >= PRIMARY_ENGLISH_MIN_NON_ENGLISH_FUNCTION_HITS
        and non_english_hits
        > PRIMARY_ENGLISH_NON_ENGLISH_TO_ENGLISH_RATIO * english_hits
    ):
        failure_reasons.append("non_english_latin_function_words_dominate")
    return {
        "rule_version": PRIMARY_ENGLISH_RULE_VERSION,
        "passed": not failure_reasons,
        "token_count": len(tokens),
        "english_function_hits": english_hits,
        "non_english_latin_function_hits": non_english_hits,
        "non_latin_letter_fraction": non_latin_fraction,
        "failure_reasons": failure_reasons,
    }


def candidate_primary_english_audit(
    key: tuple[str, str, str],
    inventory: dict[tuple[str, str, str, str], SourceRecord],
) -> dict[str, Any]:
    condition_audits = {
        condition: primary_english_audit(
            str(
                inventory[(*key, condition)]
                .record["response"]["content"]
            ).strip()
        )
        for condition in CONDITION_DIRS
    }
    return {
        "passed": all(
            audit["passed"] for audit in condition_audits.values()
        ),
        "failed_conditions": [
            condition
            for condition, audit in condition_audits.items()
            if not audit["passed"]
        ],
        "condition_audits": condition_audits,
    }


def get_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("request", {}).get("messages", [])
    return messages if isinstance(messages, list) else []


def is_single_call(record: dict[str, Any]) -> bool:
    messages = get_messages(record)
    return (
        len(messages) == 2
        and [message.get("role") for message in messages] == ["system", "user"]
        and all(str(message.get("content", "")).strip() for message in messages)
    )


def load_tasks(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"Expected 100 tasks, found {len(rows)}")
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate task_id in tasks.csv")
    return rows, by_id


def load_personas(path: Path) -> list[dict[str, Any]]:
    personas = yaml.safe_load(path.read_text(encoding="utf-8"))["personas"]
    required = {"id", "name", *CONDITION_DIRS}
    if len(personas) != 20:
        raise ValueError(f"Expected 20 personas, found {len(personas)}")
    for persona in personas:
        if missing := required - set(persona):
            raise ValueError(f"Persona missing {missing}: {persona}")
    return personas


def match_persona(
    system_prompt: str,
    personas: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any] | None:
    matches = [
        persona
        for persona in personas
        if system_prompt.endswith(str(persona[condition]).strip())
    ]
    return matches[0] if len(matches) == 1 else None


def load_inventory(
    root: Path,
    tasks_by_id: dict[str, dict[str, str]],
    personas: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, str], SourceRecord],
    dict[str, dict[str, int]],
    dict[str, str],
]:
    index: dict[tuple[str, str, str, str], SourceRecord] = {}
    ambiguous: set[tuple[str, str, str, str]] = set()
    stats_by_model: dict[str, dict[str, int]] = {}
    api_names: dict[str, str] = {}

    for model_code, result_dir in MODEL_SPECS:
        stats = Counter()
        observed_apis: Counter[str] = Counter()
        for condition, condition_dir in CONDITION_DIRS.items():
            directory = root / "results" / result_dir / condition_dir
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for task_id in sorted(tasks_by_id):
                path = directory / f"{task_id}.jsonl"
                if not path.is_file():
                    continue
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        stats["jsonl_rows"] += 1
                        record = json.loads(line)
                        if not is_single_call(record):
                            stats["non_single_call"] += 1
                            continue
                        messages = get_messages(record)
                        persona = match_persona(
                            str(messages[0]["content"]).strip(), personas, condition
                        )
                        if persona is None:
                            stats["unmatched_system_prompt"] += 1
                            continue
                        response = str(
                            record.get("response", {}).get("content", "")
                        ).strip()
                        if not response:
                            stats["empty_response"] += 1
                            continue
                        query = tasks_by_id[task_id]["query"]
                        if not (
                            record.get("task_id") == task_id
                            and record.get("persona_name") == persona["name"]
                            and record.get("raw_question") == query
                            and str(messages[1]["content"]) == query
                        ):
                            stats["metadata_mismatch"] += 1
                            continue
                        key = (model_code, task_id, str(persona["name"]), condition)
                        if key in ambiguous:
                            stats["duplicate_key"] += 1
                            continue
                        if key in index:
                            index.pop(key)
                            ambiguous.add(key)
                            stats["indexed_records"] -= 1
                            stats["duplicate_key"] += 1
                            continue
                        index[key] = SourceRecord(path, line_number, record)
                        stats["indexed_records"] += 1
                        api = str(record.get("request", {}).get("model", ""))
                        if api:
                            observed_apis[api] += 1
        if len(observed_apis) != 1:
            raise ValueError(
                f"Expected one API model for {model_code}: {dict(observed_apis)}"
            )
        api_names[model_code] = observed_apis.most_common(1)[0][0]
        stats_by_model[model_code] = dict(stats)
    return index, stats_by_model, api_names


def hamilton_targets(
    rows: list[dict[str, str]], n_draws: int
) -> tuple[dict[str, int], dict[str, Fraction]]:
    counts = Counter(row["subcategory"] for row in rows)
    quotas = {
        category: Fraction(n_draws * count, len(rows))
        for category, count in counts.items()
    }
    targets = {category: int(quota) for category, quota in quotas.items()}
    remaining = n_draws - sum(targets.values())
    order = sorted(
        quotas,
        key=lambda category: (
            -(quotas[category] - int(quotas[category])),
            category,
        ),
    )
    for category in order[:remaining]:
        targets[category] += 1
    if any(value < 1 for value in targets.values()):
        raise ValueError("Hamilton allocation did not cover all categories")
    return targets, quotas


def add_constraint(
    row_indices: list[int],
    col_indices: list[int],
    values: list[float],
    lower: list[float],
    upper: list[float],
    coefficients: Iterable[tuple[int, float]],
    lb: float,
    ub: float,
) -> None:
    row = len(lower)
    for column, value in coefficients:
        row_indices.append(row)
        col_indices.append(column)
        values.append(float(value))
    lower.append(float(lb))
    upper.append(float(ub))


def solve_mixed_integer(
    n_variables: int,
    constraints: list[tuple[list[tuple[int, float]], float, float]],
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for coefficients, lb, ub in constraints:
        add_constraint(rows, columns, values, lower, upper, coefficients, lb, ub)
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(lower), n_variables)
    ).tocsc()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)
        ),
        options={"disp": False, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None or not result.success:
        raise RuntimeError(
            f"MILP failed: status={result.status}, message={result.message}"
        )
    return result.x


def solve_binary(
    n_variables: int,
    constraints: list[tuple[list[tuple[int, float]], float, float]],
    objective: np.ndarray,
) -> np.ndarray:
    return solve_mixed_integer(
        n_variables,
        constraints,
        objective,
        np.ones(n_variables, dtype=int),
        np.zeros(n_variables),
        np.ones(n_variables),
    )


def select_triplets(
    rows: list[dict[str, str]],
    personas: list[dict[str, Any]],
    inventory: dict[tuple[str, str, str, str], SourceRecord],
    category_targets: dict[str, int],
    seed: int,
) -> tuple[
    list[tuple[str, dict[str, str], dict[str, Any]]],
    dict[str, Any],
]:
    task_by_id = {row["task_id"]: row for row in rows}
    persona_by_name = {str(persona["name"]): persona for persona in personas}
    model_codes = [model for model, _ in MODEL_SPECS]
    technical_counts = Counter()
    length_eligible_counts = Counter()
    length_eligible_candidates: list[tuple[str, str, str]] = []

    for model_code in model_codes:
        for task in sorted(rows, key=lambda row: row["task_id"]):
            if task["task_id"] in TECHNICAL_TASK_EXCLUSIONS:
                continue
            for persona in sorted(personas, key=lambda value: int(value["id"])):
                key = (model_code, task["task_id"], str(persona["name"]))
                if not all((*key, condition) in inventory for condition in CONDITION_DIRS):
                    continue
                technical_counts[model_code] += 1
                counts = [
                    word_count(
                        str(
                            inventory[(*key, condition)]
                            .record["response"]["content"]
                        ).strip()
                    )
                    for condition in CONDITION_DIRS
                ]
                # Stronger than checking only the two Role-centered gaps: the
                # maximum minus minimum across all three outputs is <= 30.
                if (
                    max(counts) > MAX_RESPONSE_WORDS
                    or max(counts) - min(counts) > MAX_TRIPLET_WORD_RANGE
                ):
                    continue
                length_eligible_counts[model_code] += 1
                length_eligible_candidates.append(key)

    def selection_constraints(
        candidates: list[tuple[str, str, str]],
    ) -> list[tuple[list[tuple[int, float]], float, float]]:
        constraints: list[
            tuple[list[tuple[int, float]], float, float]
        ] = []
        for model_code in model_codes:
            coefficients = [
                (index, 1.0)
                for index, (model, _, _) in enumerate(candidates)
                if model == model_code
            ]
            constraints.append(
                (coefficients, TRIPLETS_PER_MODEL, TRIPLETS_PER_MODEL)
            )
        for persona in personas:
            name = str(persona["name"])
            coefficients = [
                (index, 1.0)
                for index, (_, _, candidate_name) in enumerate(candidates)
                if candidate_name == name
            ]
            constraints.append((coefficients, 2, 3))
        for task in rows:
            coefficients = [
                (index, 1.0)
                for index, (_, task_id, _) in enumerate(candidates)
                if task_id == task["task_id"]
            ]
            if coefficients:
                constraints.append((coefficients, 0, 1))
        for category, target in sorted(category_targets.items()):
            coefficients = [
                (index, 1.0)
                for index, (_, task_id, _) in enumerate(candidates)
                if task_by_id[task_id]["subcategory"] == category
            ]
            constraints.append((coefficients, target, target))
        for model_code in model_codes:
            for persona in personas:
                name = str(persona["name"])
                coefficients = [
                    (index, 1.0)
                    for index, (model, _, candidate_name) in enumerate(candidates)
                    if model == model_code and candidate_name == name
                ]
                if coefficients:
                    constraints.append((coefficients, 0, 1))
        return constraints

    metadata_hash = {
        candidate: int.from_bytes(
            hashlib.sha256(
                (
                    f"{seed}:triplet:{candidate[0]}:{candidate[1]}:"
                    f"{candidate[2]}"
                ).encode()
            ).digest()[:8],
            "big",
        )
        / 2**64
        for candidate in length_eligible_candidates
    }
    base_objective = np.asarray(
        [
            metadata_hash[candidate]
            for candidate in length_eligible_candidates
        ]
    )
    base_solution = solve_binary(
        len(length_eligible_candidates),
        selection_constraints(length_eligible_candidates),
        base_objective,
    )
    base_selected = [
        candidate
        for index, candidate in enumerate(length_eligible_candidates)
        if base_solution[index] > 0.5
    ]

    primary_english_audits = {
        candidate: candidate_primary_english_audit(candidate, inventory)
        for candidate in length_eligible_candidates
    }
    candidates = [
        candidate
        for candidate in length_eligible_candidates
        if primary_english_audits[candidate]["passed"]
    ]
    english_eligible_counts = Counter(candidate[0] for candidate in candidates)
    base_selected_set = set(base_selected)
    # The integer replacement penalty dominates the sum of all hash tie
    # breakers. This preserves every readable member of the original
    # outcome-blind freeze whenever the exact allocation permits it, then uses
    # only the pre-existing metadata hash to select among equally minimal
    # replacements.
    replacement_objective = np.asarray(
        [
            (0.0 if candidate in base_selected_set else 1.0)
            + metadata_hash[candidate] / 100_000.0
            for candidate in candidates
        ]
    )
    solution = solve_binary(
        len(candidates),
        selection_constraints(candidates),
        replacement_objective,
    )
    selected = [
        candidate
        for index, candidate in enumerate(candidates)
        if solution[index] > 0.5
    ]
    if len(selected) != N_TRIPLETS:
        raise AssertionError(f"Selected {len(selected)} triplets")
    removed = sorted(base_selected_set - set(selected))
    added = sorted(set(selected) - base_selected_set)
    assignments = [
        (model, task_by_id[task_id], persona_by_name[persona])
        for model, task_id, persona in selected
    ]
    diagnostics = {
        "method": (
            "fixed-seed metadata-hash MILP after symmetric three-condition "
            "Single-Call, length, and primary-English readability screens; "
            "minimal-change replacement preserves the readable base freeze"
        ),
        "technically_valid_triplets_total": sum(technical_counts.values()),
        "technically_valid_triplets_by_model": dict(technical_counts),
        "length_eligible_triplets_total": len(length_eligible_candidates),
        "length_eligible_triplets_by_model": dict(length_eligible_counts),
        "screen_eligible_triplets_total": len(candidates),
        "screen_eligible_triplets_by_model": dict(english_eligible_counts),
        "excluded_by_length_screen_total": (
            sum(technical_counts.values()) - len(length_eligible_candidates)
        ),
        "excluded_by_primary_english_screen_total": (
            len(length_eligible_candidates) - len(candidates)
        ),
        "primary_english_exclusions": [
            {
                "model_code": model,
                "task_id": task_id,
                "persona_name": persona_name,
                "provisional_freeze_reference": (
                    PROVISIONAL_FREEZE_REFERENCES.get(
                        (model, task_id, persona_name), ""
                    )
                ),
                **primary_english_audits[(model, task_id, persona_name)],
            }
            for model, task_id, persona_name in length_eligible_candidates
            if not primary_english_audits[
                (model, task_id, persona_name)
            ]["passed"]
        ],
        "base_freeze_non_english_exclusions": [
            {
                "model_code": model,
                "task_id": task_id,
                "persona_name": persona_name,
                "provisional_freeze_reference": (
                    PROVISIONAL_FREEZE_REFERENCES.get(
                        (model, task_id, persona_name), ""
                    )
                ),
                **primary_english_audits[(model, task_id, persona_name)],
            }
            for model, task_id, persona_name in removed
            if not primary_english_audits[
                (model, task_id, persona_name)
            ]["passed"]
        ],
        "minimal_change_removed": [
            {
                "model_code": model,
                "task_id": task_id,
                "persona_name": persona_name,
            }
            for model, task_id, persona_name in removed
        ],
        "minimal_change_added": [
            {
                "model_code": model,
                "task_id": task_id,
                "persona_name": persona_name,
            }
            for model, task_id, persona_name in added
        ],
        "retained_base_freeze_triplets": (
            len(base_selected_set & set(selected))
        ),
        "max_words_per_complete_response": MAX_RESPONSE_WORDS,
        "max_word_range_across_role_basic_pro": MAX_TRIPLET_WORD_RANGE,
        "primary_english_rule": {
            "version": PRIMARY_ENGLISH_RULE_VERSION,
            "triplet_pass_rule": (
                "Role, Basic, and Pro must each pass the same rule"
            ),
            "max_non_latin_letter_fraction": (
                PRIMARY_ENGLISH_MAX_NON_LATIN_LETTER_FRACTION
            ),
            "min_non_english_function_hits": (
                PRIMARY_ENGLISH_MIN_NON_ENGLISH_FUNCTION_HITS
            ),
            "non_english_to_english_function_hit_ratio": (
                PRIMARY_ENGLISH_NON_ENGLISH_TO_ENGLISH_RATIO
            ),
            "does_not_screen_relevance_refusal_style_or_opinion_breadth": True,
        },
        "selection_uses_response_semantics": False,
        "selection_uses_response_content_beyond_readability_screen": False,
        "selection_uses_mpd_or_expected_direction": False,
    }
    return assignments, diagnostics


def materialize_triplets(
    assignments: list[tuple[str, dict[str, str], dict[str, Any]]],
    inventory: dict[tuple[str, str, str, str], SourceRecord],
    api_names: dict[str, str],
    seed: int,
) -> list[Triplet]:
    result_dir_by_model = dict(MODEL_SPECS)
    shuffled = list(assignments)
    rng_for(seed, "opaque_triplet_ids").shuffle(shuffled)
    triplets = []
    for number, (model, task, persona) in enumerate(shuffled, 1):
        key = (model, task["task_id"], str(persona["name"]))
        sources = {
            condition: inventory[(*key, condition)] for condition in CONDITION_DIRS
        }
        responses = {
            condition: str(source.record["response"]["content"]).strip()
            for condition, source in sources.items()
        }
        counts = {
            condition: word_count(text) for condition, text in responses.items()
        }
        triplets.append(
            Triplet(
                triplet_id=f"SA-T{number:03d}",
                model_code=model,
                result_dir=result_dir_by_model[model],
                model_api=api_names[model],
                task_id=task["task_id"],
                query=task["query"],
                subcategory=task["subcategory"],
                persona_id=int(persona["id"]),
                persona_name=str(persona["name"]),
                persona_prompts={
                    condition: str(persona[condition]) for condition in CONDITION_DIRS
                },
                sources=sources,
                responses=responses,
                word_counts=counts,
            )
        )
    return triplets


def make_comparisons(triplets: list[Triplet], seed: int) -> list[Comparison]:
    specs = []
    for triplet_index, triplet in enumerate(triplets):
        for contrast, (role, other, suffix) in CONTRASTS.items():
            specs.append(
                (triplet_index, contrast, role, other, suffix)
            )
    rng_for(seed, "opaque_comparison_ids").shuffle(specs)
    comparisons = []
    for number, (triplet_index, contrast, role, other, suffix) in enumerate(
        specs, 1
    ):
        triplet = triplets[triplet_index]
        comparisons.append(
            Comparison(
                item_id=f"SA-C{number:03d}",
                triplet_index=triplet_index,
                contrast=contrast,
                left_condition=role,
                other_condition=other,
                suffix=suffix,
                displayed_word_total=(
                    word_count(triplet.query)
                    + triplet.word_counts[role]
                    + triplet.word_counts[other]
                ),
            )
        )
    return comparisons


def build_bundle_model_sets(seed: int) -> list[tuple[str, ...]]:
    models = [model for model, _ in MODEL_SPECS]
    sets = [
        tuple(model for model in models if model != omitted)
        for _round in range(4)
        for omitted in models
    ]
    rng_for(seed, "bundle_model_schedule").shuffle(sets)
    return sets


def assign_comparisons(
    triplets: list[Triplet],
    comparisons: list[Comparison],
    model_sets: list[tuple[str, ...]],
    seed: int,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    variables = [
        (item_index, bundle_index)
        for item_index, comparison in enumerate(comparisons)
        for bundle_index, models in enumerate(model_sets)
        if triplets[comparison.triplet_index].model_code in models
    ]
    index = {pair: position for position, pair in enumerate(variables)}
    constraints: list[tuple[list[tuple[int, float]], float, float]] = []
    for item_index in range(len(comparisons)):
        constraints.append(
            (
                [
                    (index[(item_index, bundle)], 1.0)
                    for bundle in range(N_BUNDLES)
                    if (item_index, bundle) in index
                ],
                RATERS_PER_COMPARISON,
                RATERS_PER_COMPARISON,
            )
        )
    for bundle, models in enumerate(model_sets):
        for model in models:
            constraints.append(
                (
                    [
                        (index[(item_index, bundle)], 1.0)
                        for item_index, comparison in enumerate(comparisons)
                        if triplets[comparison.triplet_index].model_code == model
                        and (item_index, bundle) in index
                    ],
                    1,
                    1,
                )
            )
        for contrast in CONTRASTS:
            constraints.append(
                (
                    [
                        (index[(item_index, bundle)], 1.0)
                        for item_index, comparison in enumerate(comparisons)
                        if comparison.contrast == contrast
                        and (item_index, bundle) in index
                    ],
                    ITEMS_PER_BUNDLE // 2,
                    ITEMS_PER_BUNDLE // 2,
                )
            )
        for persona_name in sorted({triplet.persona_name for triplet in triplets}):
            coefficients = [
                (index[(item_index, bundle)], 1.0)
                for item_index, comparison in enumerate(comparisons)
                if triplets[comparison.triplet_index].persona_name == persona_name
                and (item_index, bundle) in index
            ]
            if coefficients:
                constraints.append((coefficients, 0, 1))

    objective = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    (
                        f"{seed}:bundle:{comparisons[item].item_id}:"
                        f"{bundle}"
                    ).encode()
                ).digest()[:8],
                "big",
            )
            / 2**64
            for item, bundle in variables
        ]
    )
    solution = solve_binary(len(variables), constraints, objective)
    assignments: dict[int, list[int]] = defaultdict(list)
    for position, (item, bundle) in enumerate(variables):
        if solution[position] > 0.5:
            assignments[bundle].append(item)
    for bundle in assignments:
        assignments[bundle].sort(key=lambda item: comparisons[item].item_id)

    def workload_totals(
        current: dict[int, list[int]],
    ) -> list[int]:
        return [
            sum(
                comparisons[item].displayed_word_total
                for item in current[bundle]
            )
            for bundle in range(N_BUNDLES)
        ]

    def workload_objective(totals: list[int]) -> tuple[int, int]:
        center = sum(totals) / len(totals)
        return (
            max(totals) - min(totals),
            round(sum((total - center) ** 2 for total in totals)),
        )

    initial_workload_objective = workload_objective(
        workload_totals(assignments)
    )
    swap_count = 0
    while True:
        current_totals = workload_totals(assignments)
        current_objective = workload_objective(current_totals)
        best: tuple[
            tuple[int, int],
            tuple[str, str, int, int],
            int,
            int,
            int,
            int,
        ] | None = None
        for first_bundle in range(N_BUNDLES):
            for second_bundle in range(first_bundle + 1, N_BUNDLES):
                for first_item in assignments[first_bundle]:
                    first_comparison = comparisons[first_item]
                    first_triplet = triplets[first_comparison.triplet_index]
                    for second_item in assignments[second_bundle]:
                        if first_item == second_item:
                            continue
                        second_comparison = comparisons[second_item]
                        second_triplet = triplets[
                            second_comparison.triplet_index
                        ]
                        if (
                            first_comparison.contrast
                            != second_comparison.contrast
                            or first_triplet.model_code
                            != second_triplet.model_code
                        ):
                            continue
                        first_remaining = set(
                            assignments[first_bundle]
                        ) - {first_item}
                        second_remaining = set(
                            assignments[second_bundle]
                        ) - {second_item}
                        if (
                            second_item in first_remaining
                            or first_item in second_remaining
                        ):
                            continue
                        if any(
                            triplets[comparisons[item].triplet_index].persona_name
                            == second_triplet.persona_name
                            for item in first_remaining
                        ) or any(
                            triplets[comparisons[item].triplet_index].persona_name
                            == first_triplet.persona_name
                            for item in second_remaining
                        ):
                            continue
                        if any(
                            comparisons[item].triplet_index
                            == second_comparison.triplet_index
                            for item in first_remaining
                        ) or any(
                            comparisons[item].triplet_index
                            == first_comparison.triplet_index
                            for item in second_remaining
                        ):
                            continue
                        candidate_totals = list(current_totals)
                        candidate_totals[first_bundle] += (
                            second_comparison.displayed_word_total
                            - first_comparison.displayed_word_total
                        )
                        candidate_totals[second_bundle] += (
                            first_comparison.displayed_word_total
                            - second_comparison.displayed_word_total
                        )
                        candidate_objective = workload_objective(
                            candidate_totals
                        )
                        tie_key = (
                            comparisons[first_item].item_id,
                            comparisons[second_item].item_id,
                            first_bundle,
                            second_bundle,
                        )
                        candidate = (
                            candidate_objective,
                            tie_key,
                            first_bundle,
                            second_bundle,
                            first_item,
                            second_item,
                        )
                        if (
                            candidate_objective < current_objective
                            and (best is None or candidate[:2] < best[:2])
                        ):
                            best = candidate
        if best is None:
            break
        _, _, first_bundle, second_bundle, first_item, second_item = best
        assignments[first_bundle].remove(first_item)
        assignments[first_bundle].append(second_item)
        assignments[second_bundle].remove(second_item)
        assignments[second_bundle].append(first_item)
        swap_count += 1
    for bundle in assignments:
        assignments[bundle].sort(key=lambda item: comparisons[item].item_id)
    word_totals = [
        sum(comparisons[item].displayed_word_total for item in assignments[bundle])
        for bundle in range(N_BUNDLES)
    ]
    return dict(assignments), {
        "method": (
            "fixed-seed metadata-hash MILP followed by deterministic valid "
            "same-model/same-contrast local swaps that lexicographically "
            "reduce displayed-word range and squared imbalance"
        ),
        "local_balance_swaps": swap_count,
        "initial_range_displayed_words": initial_workload_objective[0],
        "bundle_displayed_words": word_totals,
        "min_displayed_words": min(word_totals),
        "median_displayed_words": statistics.median(word_totals),
        "max_displayed_words": max(word_totals),
        "range_displayed_words": max(word_totals) - min(word_totals),
    }


class Dinic:
    def __init__(self, n_nodes: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = [target, capacity, len(self.graph[target])]
        reverse = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for target, capacity, _ in self.graph[node]:
                    if capacity and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * len(self.graph)

            def send(node: int, flow: int) -> int:
                if node == sink:
                    return flow
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse = edge
                    if capacity and level[target] == level[node] + 1:
                        pushed = send(target, min(flow, capacity))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[target][reverse][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while pushed := send(source, 10**9):
                total += pushed


def assign_rationales(
    comparisons: list[Comparison],
    assignments: dict[int, list[int]],
    triplets: list[Triplet],
    seed: int,
) -> set[tuple[int, int]]:
    occurrences = [
        (item, bundle)
        for bundle, items in assignments.items()
        for item in items
    ]
    index = {occurrence: position for position, occurrence in enumerate(occurrences)}
    constraints = []
    for bundle, items in assignments.items():
        constraints.append(
            ([(index[(item, bundle)], 1.0) for item in items], 1, 1)
        )
    for item in range(len(comparisons)):
        constraints.append(
            (
                [
                    (index[(item, bundle)], 1.0)
                    for bundle, items in assignments.items()
                    if item in items
                ],
                0,
                1,
            )
        )
    for contrast in CONTRASTS:
        constraints.append(
            (
                [
                    (position, 1.0)
                    for position, (item, _) in enumerate(occurrences)
                    if comparisons[item].contrast == contrast
                ],
                N_BUNDLES // 2,
                N_BUNDLES // 2,
            )
        )
    for model, _ in MODEL_SPECS:
        constraints.append(
            (
                [
                    (position, 1.0)
                    for position, (item, _) in enumerate(occurrences)
                    if triplets[comparisons[item].triplet_index].model_code
                    == model
                ],
                4,
                4,
            )
        )
    objective = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{seed}:rationale:{item}:{bundle}".encode()
                ).digest()[:8],
                "big",
            )
            / 2**64
            for item, bundle in occurrences
        ]
    )
    solution = solve_binary(len(occurrences), constraints, objective)
    return {
        occurrence
        for occurrence, position in index.items()
        if solution[position] > 0.5
    }


def assign_variants_and_sides(
    comparisons: list[Comparison],
    assignments: dict[int, list[int]],
    seed: int,
) -> tuple[dict[int, dict[str, str]], dict[tuple[int, int], str]]:
    shuffled = list(range(len(comparisons)))
    rng_for(seed, "opaque_variant_mapping").shuffle(shuffled)
    role_is_x = set(shuffled[: len(shuffled) // 2])
    variant_map = {
        item: (
            {
                "X": comparison.left_condition,
                "Y": comparison.other_condition,
            }
            if item in role_is_x
            else {
                "X": comparison.other_condition,
                "Y": comparison.left_condition,
            }
        )
        for item, comparison in enumerate(comparisons)
    }

    occurrences = [
        (item, bundle)
        for bundle, items in assignments.items()
        for item in items
    ]
    index = {occurrence: position for position, occurrence in enumerate(occurrences)}
    constraints = []
    for item in range(len(comparisons)):
        constraints.append(
            (
                [
                    (index[(item, bundle)], 1.0)
                    for bundle, items in assignments.items()
                    if item in items
                ],
                1,
                1,
            )
        )
    for bundle, items in assignments.items():
        constraints.append(
            (
                [(index[(item, bundle)], 1.0) for item in items],
                ITEMS_PER_BUNDLE // 2,
                ITEMS_PER_BUNDLE // 2,
            )
        )
        for contrast in CONTRASTS:
            # Each contrast contributes three items, so the closest possible
            # within-bundle balance is 1/2 rather than 0/3.
            constraints.append(
                (
                    [
                        (index[(item, bundle)], 1.0)
                        for item in items
                        if comparisons[item].contrast == contrast
                    ],
                    1,
                    2,
                )
            )
    objective = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{seed}:orientation:{item}:{bundle}".encode()
                ).digest()[:8],
                "big",
            )
            / 2**64
            for item, bundle in occurrences
        ]
    )
    solution = solve_binary(len(occurrences), constraints, objective)
    a_condition = {
        occurrence: (
            comparisons[occurrence[0]].left_condition
            if solution[position] > 0.5
            else comparisons[occurrence[0]].other_condition
        )
        for occurrence, position in index.items()
    }
    return variant_map, a_condition


def source_payload(source: SourceRecord) -> dict[str, Any]:
    canonical = (
        json.dumps(
            source.record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode()
    return {
        "source_file": str(source.path),
        "source_line": source.line_number,
        "source_file_sha256": sha256_file(source.path),
        "canonical_record_sha256": sha256_bytes(canonical),
        "single_call_audit": {
            "passed": is_single_call(source.record),
            "message_roles": [
                message.get("role") for message in get_messages(source.record)
            ],
            "message_count": len(get_messages(source.record)),
        },
        "primary_english_audit": primary_english_audit(
            str(source.record["response"]["content"]).strip()
        ),
        "raw_record": source.record,
    }


def validate_invariants(
    triplets: list[Triplet],
    comparisons: list[Comparison],
    assignments: dict[int, list[int]],
    model_sets: list[tuple[str, ...]],
    rationales: set[tuple[int, int]],
    a_condition: dict[tuple[int, int], str],
    category_targets: dict[str, int],
) -> dict[str, Any]:
    errors = []
    if len(triplets) != N_TRIPLETS or len(comparisons) != N_COMPARISONS:
        errors.append("Wrong triplet/comparison count")
    if len({triplet.task_id for triplet in triplets}) != N_TRIPLETS:
        errors.append("Questions are not globally distinct")
    if len({triplet.persona_name for triplet in triplets}) != 20:
        errors.append("Not all 20 personas are covered")
    if sorted(Counter(t.persona_name for t in triplets).values()) != (
        [2] * 18 + [3] * 2
    ):
        errors.append("Persona allocation is not balanced 18x2 + 2x3")
    if Counter(triplet.model_code for triplet in triplets) != Counter(
        {model: TRIPLETS_PER_MODEL for model, _ in MODEL_SPECS}
    ):
        errors.append("Model allocation is not 7x6")
    if Counter(triplet.subcategory for triplet in triplets) != Counter(
        category_targets
    ):
        errors.append("Category allocation differs from target")
    for triplet in triplets:
        if max(triplet.word_counts.values()) > MAX_RESPONSE_WORDS:
            errors.append(f"{triplet.triplet_id}: response cap failed")
        if (
            max(triplet.word_counts.values()) - min(triplet.word_counts.values())
            > MAX_TRIPLET_WORD_RANGE
        ):
            errors.append(f"{triplet.triplet_id}: triplet gap failed")
        for condition, source in triplet.sources.items():
            if not is_single_call(source.record):
                errors.append(f"{triplet.triplet_id}: not Single-Call")
            audit = primary_english_audit(triplet.responses[condition])
            if not audit["passed"]:
                errors.append(
                    f"{triplet.triplet_id}/{condition}: primary-English "
                    "readability screen failed"
                )

    occurrences = Counter()
    role_a_global = 0
    role_a_by_contrast = Counter()
    for bundle in range(N_BUNDLES):
        items = assignments.get(bundle, [])
        if len(items) != ITEMS_PER_BUNDLE:
            errors.append(f"Bundle {bundle + 1}: wrong size")
            continue
        bundle_comparisons = [comparisons[item] for item in items]
        if Counter(item.contrast for item in bundle_comparisons) != Counter(
            {"role_basic": 3, "role_pro": 3}
        ):
            errors.append(f"Bundle {bundle + 1}: contrast split failed")
        triplet_ids = [
            triplets[item.triplet_index].triplet_id for item in bundle_comparisons
        ]
        if len(set(triplet_ids)) != ITEMS_PER_BUNDLE:
            errors.append(f"Bundle {bundle + 1}: duplicate triplet")
        models = {
            triplets[item.triplet_index].model_code for item in bundle_comparisons
        }
        if models != set(model_sets[bundle]):
            errors.append(f"Bundle {bundle + 1}: model schedule failed")
        if len(
            {
                triplets[item.triplet_index].persona_name
                for item in bundle_comparisons
            }
        ) != ITEMS_PER_BUNDLE:
            errors.append(f"Bundle {bundle + 1}: repeated persona")
        role_a = 0
        bundle_role_a_by_contrast = Counter()
        for item in items:
            occurrences[item] += 1
            if a_condition[(item, bundle)] == "role":
                role_a += 1
                role_a_global += 1
                role_a_by_contrast[comparisons[item].contrast] += 1
                bundle_role_a_by_contrast[comparisons[item].contrast] += 1
        if role_a != 3:
            errors.append(f"Bundle {bundle + 1}: A/B balance failed")
        if any(
            bundle_role_a_by_contrast[contrast] not in {1, 2}
            for contrast in CONTRASTS
        ):
            errors.append(
                f"Bundle {bundle + 1}: within-contrast A/B balance failed"
            )
        if sum((item, bundle) in rationales for item in items) != 1:
            errors.append(f"Bundle {bundle + 1}: rationale allocation failed")
    if any(
        occurrences[item] != RATERS_PER_COMPARISON
        for item in range(N_COMPARISONS)
    ):
        errors.append("Every comparison must occur exactly twice")
    if len(rationales) != N_BUNDLES:
        errors.append("Wrong rationale count")
    if len({item for item, _ in rationales}) != N_BUNDLES:
        errors.append("Rationale comparison IDs are not distinct")
    rationale_by_contrast = Counter(
        comparisons[item].contrast for item, _ in rationales
    )
    rationale_by_model = Counter(
        triplets[comparisons[item].triplet_index].model_code
        for item, _ in rationales
    )
    if rationale_by_contrast != Counter({"role_basic": 14, "role_pro": 14}):
        errors.append("Rationales are not balanced 14/14 by contrast")
    if rationale_by_model != Counter({model: 4 for model, _ in MODEL_SPECS}):
        errors.append("Rationales are not balanced at four per model")
    if role_a_global != 84 or role_a_by_contrast != Counter(
        {"role_basic": 42, "role_pro": 42}
    ):
        errors.append(
            f"Unexpected orientation balance: {role_a_global}, "
            f"{role_a_by_contrast}"
        )
    for item in range(N_COMPARISONS):
        observed = {
            a_condition[(item, bundle)]
            for bundle, items in assignments.items()
            if item in items
        }
        comparison = comparisons[item]
        if observed != {comparison.left_condition, comparison.other_condition}:
            errors.append(f"{comparison.item_id}: lacks both orientations")
    if errors:
        raise AssertionError("\n".join(errors))
    return {
        "all_invariants_passed": True,
        "n_triplets": N_TRIPLETS,
        "n_comparisons": N_COMPARISONS,
        "n_bundles": N_BUNDLES,
        "n_judgments": N_COMPARISONS * RATERS_PER_COMPARISON,
        "n_rationales": len(rationales),
        "role_as_response_a": role_a_global,
        "nonrole_as_response_a": 168 - role_a_global,
        "role_as_a_by_contrast": dict(role_a_by_contrast),
    }


def build_outputs(
    *,
    out_dir: Path,
    root: Path,
    seed: int,
    task_path: Path,
    persona_path: Path,
    triplets: list[Triplet],
    comparisons: list[Comparison],
    assignments: dict[int, list[int]],
    model_sets: list[tuple[str, ...]],
    rationales: set[tuple[int, int]],
    variant_map: dict[int, dict[str, str]],
    a_condition: dict[tuple[int, int], str],
    inventory_stats: dict[str, dict[str, int]],
    category_targets: dict[str, int],
    category_quotas: dict[str, Fraction],
    selection_diagnostics: dict[str, Any],
    bundle_diagnostics: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    private_dir = out_dir / "private"
    public_dir = out_dir / "public"
    bundle_dir = public_dir / "bundles"
    private_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for stale in bundle_dir.glob("SA-B*.json"):
        stale.unlink()

    common = {
        "schema_version": "study-a-v2",
        "study": "Matched Role-Basic and Role-Pro human comparisons",
        "seed": seed,
        "selection_population": {
            "models": [model for model, _ in MODEL_SPECS],
            "questions": 100,
            "personas": 20,
            "conditions": list(CONDITION_DIRS.values()),
            "technical_task_exclusions": TECHNICAL_TASK_EXCLUSIONS,
            "single_call_required": True,
            "readability_feasibility_screen": {
                "max_words_per_complete_response": MAX_RESPONSE_WORDS,
                "max_word_range_across_role_basic_pro": MAX_TRIPLET_WORD_RANGE,
                "primary_english_rule": (
                    selection_diagnostics["primary_english_rule"]
                ),
                "equivalent_pairwise_requirement": (
                    "Every pair among Role, Basic, and Pro differs by <=30 words"
                ),
                "applied_before_seeded_stratified_draw": True,
            },
            "category_targets": category_targets,
        },
    }
    private_triplets = []
    for triplet in triplets:
        private_triplets.append(
            {
                "triplet_id": triplet.triplet_id,
                "model_code": triplet.model_code,
                "result_directory": triplet.result_dir,
                "model_api": triplet.model_api,
                "task_id": triplet.task_id,
                "query": triplet.query,
                "subcategory": triplet.subcategory,
                "persona_id": triplet.persona_id,
                "persona_name": triplet.persona_name,
                "persona_prompts": triplet.persona_prompts,
                "displayed_word_counts": {
                    "query": word_count(triplet.query),
                    **{
                        f"{condition}_response": count
                        for condition, count in triplet.word_counts.items()
                    },
                    "triplet_response_range": (
                        max(triplet.word_counts.values())
                        - min(triplet.word_counts.values())
                    ),
                },
                "conditions": {
                    condition: {
                        "response_text": triplet.responses[condition],
                        "provenance": source_payload(triplet.sources[condition]),
                    }
                    for condition in CONDITION_DIRS
                },
            }
        )

    private_items = []
    public_items = []
    for item_index, comparison in enumerate(comparisons):
        triplet = triplets[comparison.triplet_index]
        conditions = [comparison.left_condition, comparison.other_condition]
        condition_for_variant = variant_map[item_index]
        private_items.append(
            {
                "item_id": comparison.item_id,
                "triplet_id": triplet.triplet_id,
                "contrast": comparison.contrast,
                "model_code": triplet.model_code,
                "model_api": triplet.model_api,
                "task_id": triplet.task_id,
                "query": triplet.query,
                "subcategory": triplet.subcategory,
                "persona_id": triplet.persona_id,
                "persona_name": triplet.persona_name,
                "variant_condition_key": condition_for_variant,
                "displayed_word_counts": {
                    "query": word_count(triplet.query),
                    **{
                        f"{condition}_response": triplet.word_counts[condition]
                        for condition in conditions
                    },
                    "total": comparison.displayed_word_total,
                },
                "conditions": {
                    condition: {
                        "response_text": triplet.responses[condition],
                        "provenance": source_payload(triplet.sources[condition]),
                    }
                    for condition in conditions
                },
            }
        )
        public_items.append(
            {
                "item_id": comparison.item_id,
                "question": triplet.query,
                "variants": [
                    {
                        "variant_id": f"{comparison.item_id}-{variant}",
                        "text": triplet.responses[condition_for_variant[variant]],
                    }
                    for variant in ("X", "Y")
                ],
            }
        )

    manifest = {
        **common,
        "privacy": "PRIVATE: contains condition, model, persona, and source keys.",
        "source_snapshots": {
            "tasks_csv": {"path": str(task_path), "sha256": sha256_file(task_path)},
            "personas_yaml": {
                "path": str(persona_path),
                "sha256": sha256_file(persona_path),
            },
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "inventory_technical_stats": inventory_stats,
        "selection_diagnostics": selection_diagnostics,
        "triplets": private_triplets,
        "items": private_items,
    }
    write_json(private_dir / "manifest_full.json", manifest)
    write_json(
        public_dir / "items_blinded.json",
        {
            "schema_version": "study-a-v2",
            "study": "Matched response comparison",
            "privacy": "PARTICIPANT-SAFE: condition and provenance labels removed.",
            "items": public_items,
        },
    )

    public_by_id = {item["item_id"]: item for item in public_items}
    public_bundles = []
    private_bundles = []
    for bundle in range(N_BUNDLES):
        bundle_id = f"SA-B{bundle + 1:02d}"
        item_indices = list(assignments[bundle])
        rng_for(seed, f"bundle_item_order:{bundle_id}").shuffle(item_indices)
        public_bundle_items = []
        private_bundle_items = []
        for display_position, item_index in enumerate(item_indices, 1):
            comparison = comparisons[item_index]
            triplet = triplets[comparison.triplet_index]
            condition_a = a_condition[(item_index, bundle)]
            variant_a = next(
                variant
                for variant, condition in variant_map[item_index].items()
                if condition == condition_a
            )
            variant_b = "Y" if variant_a == "X" else "X"
            variants = {
                variant["variant_id"][-1]: variant["text"]
                for variant in public_by_id[comparison.item_id]["variants"]
            }
            public_bundle_items.append(
                {
                    "display_position": display_position,
                    "item_id": comparison.item_id,
                    "question": triplet.query,
                    "response_a_variant_id": f"{comparison.item_id}-{variant_a}",
                    "response_a": variants[variant_a],
                    "response_b_variant_id": f"{comparison.item_id}-{variant_b}",
                    "response_b": variants[variant_b],
                }
            )
            private_bundle_items.append(
                {
                    "display_position": display_position,
                    "item_id": comparison.item_id,
                    "triplet_id": triplet.triplet_id,
                    "contrast": comparison.contrast,
                    "model_code": triplet.model_code,
                    "task_id": triplet.task_id,
                    "persona_id": triplet.persona_id,
                    "persona_name": triplet.persona_name,
                    "response_a_condition": condition_a,
                    "response_b_condition": (
                        comparison.other_condition
                        if condition_a == "role"
                        else "role"
                    ),
                    "rationale_required_after_lock": (
                        item_index,
                        bundle,
                    )
                    in rationales,
                    "displayed_word_total": comparison.displayed_word_total,
                }
            )
        public_bundle = {
            "schema_version": "study-a-bundle-v2",
            "study": "Matched response comparison",
            "bundle_id": bundle_id,
            "items": public_bundle_items,
        }
        private_bundle = {
            "bundle_id": bundle_id,
            "included_model_codes": list(model_sets[bundle]),
            "contrast_counts": dict(
                Counter(item["contrast"] for item in private_bundle_items)
            ),
            "displayed_word_total": sum(
                item["displayed_word_total"] for item in private_bundle_items
            ),
            "rationale_item_ids_after_primary_lock": [
                item["item_id"]
                for item in private_bundle_items
                if item["rationale_required_after_lock"]
            ],
            "items": private_bundle_items,
        }
        public_bundles.append(public_bundle)
        private_bundles.append(private_bundle)
        write_json(bundle_dir / f"{bundle_id}.json", public_bundle)
    write_json(
        public_dir / "bundles_public.json",
        {
            "schema_version": "study-a-bundles-v2",
            "study": "Matched response comparison",
            "bundles": public_bundles,
        },
    )
    write_json(
        private_dir / "bundle_assignment_key.json",
        {
            **common,
            "privacy": "PRIVATE SERVER-SIDE A/B and rationale key.",
            "bundles": private_bundles,
        },
    )

    triplet_gaps = [
        max(triplet.word_counts.values()) - min(triplet.word_counts.values())
        for triplet in triplets
    ]
    category_counts = Counter(triplet.subcategory for triplet in triplets)
    audit = f"""# Study A dual-contrast formal sample audit

Generated deterministically by `generate_study_a.py`.

## Frozen design

- Seed: `{seed}`
- Selection units: {N_TRIPLETS} matched Role--Basic--Pro Single-Call triplets
- Comparisons: {N_COMPARISONS} blinded pairs ({N_TRIPLETS} Role--Basic and {N_TRIPLETS} Role--Pro)
- Models: 7 × {TRIPLETS_PER_MODEL} triplets
- Questions: {N_TRIPLETS} distinct
- Personas: all 20 (18 twice, 2 three times)
- Subcategories: all 10, exact Hamilton allocation
- Readability: every complete response <= {MAX_RESPONSE_WORDS} words
- Strong three-way length match: max minus min across Role, Basic, Pro
  <= {MAX_TRIPLET_WORD_RANGE} words (therefore both Role--Basic and Role--Pro,
  and also Basic--Pro, are <= {MAX_TRIPLET_WORD_RANGE})
- English readability: Role, Basic, and Pro must each pass the same conservative
  `{PRIMARY_ENGLISH_RULE_VERSION}` screen. The rule only detects clearly
  non-primary-English output; it does not screen relevance, refusal, style,
  response quality, or opinion breadth.
- Bundles: {N_BUNDLES} × {ITEMS_PER_BUNDLE}; exactly 3 Role--Basic and 3 Role--Pro
- Replication: each comparison appears in {RATERS_PER_COMPARISON} distinct bundles
  ({N_COMPARISONS * RATERS_PER_COMPARISON} scientific judgments)
- No bundle contains both contrasts from the same triplet
- Every bundle contains six distinct models and six distinct personas
- A/B: exactly 84 Role-as-A and 84 comparator-as-A globally; 3/3 per bundle;
  every comparison is shown once in each orientation
- Rationale audit: one occurrence per bundle, on 28 distinct comparisons,
  balanced 14/14 by contrast and exactly four per model

## Selection firewall

The selection optimizer uses only the fixed seed, model, question ID/category,
persona identity, technical record availability, and a seed-derived metadata
hash. It does not use response meaning, embeddings, MPD, expected direction, or
human judgments. Response text enters selection only through the symmetric
pre-specified word-count and primary-English screens above. The original
metadata-hash freeze is first reproduced, then any failed triplet is replaced
with the smallest possible number of changes; the same metadata hash breaks
ties among equally minimal replacements.

All selected records pass the strict Single-Call audit: exactly two request
messages with roles `[system, user]`, with no assistant history.

## Eligibility

- Technically valid triplets: {selection_diagnostics["technically_valid_triplets_total"]}
- Length-eligible triplets: {selection_diagnostics["length_eligible_triplets_total"]}
- Screen-eligible triplets: {selection_diagnostics["screen_eligible_triplets_total"]}
- Excluded by symmetric length screen: {selection_diagnostics["excluded_by_length_screen_total"]}
- Excluded by symmetric primary-English screen: {selection_diagnostics["excluded_by_primary_english_screen_total"]}
- Original base-freeze triplets retained: {selection_diagnostics["retained_base_freeze_triplets"]} / {N_TRIPLETS}
- Prelaunch base-freeze language exclusions:
  `{json.dumps(selection_diagnostics["base_freeze_non_english_exclusions"], ensure_ascii=False, sort_keys=True)}`
- Minimal-change removals:
  `{json.dumps(selection_diagnostics["minimal_change_removed"], ensure_ascii=False, sort_keys=True)}`
- Minimal-change replacements added:
  `{json.dumps(selection_diagnostics["minimal_change_added"], ensure_ascii=False, sort_keys=True)}`

Only the triplet identified as provisional `SA-T019` failed the screen in the
base freeze. Any additional triplets in `minimal_change_removed` passed every
readability rule and were changed only because the exact model, category,
question-uniqueness, persona-frequency, and model--persona constraints make a
one-for-one replacement infeasible. The recorded three-cycle is the
minimum-change feasible repair; those additional removals are not content
exclusions.
- Selected triplet word-range min / median / max:
  {min(triplet_gaps)} / {statistics.median(triplet_gaps):.1f} / {max(triplet_gaps)}

## Category allocation

| Category | Quota | Target | Observed |
|---|---:|---:|---:|
{chr(10).join(f"| {category} | {float(category_quotas[category]):.2f} | {category_targets[category]} | {category_counts[category]} |" for category in sorted(category_targets))}

## Bundle workload

- Displayed-word total min / median / max:
  {bundle_diagnostics["min_displayed_words"]} /
  {bundle_diagnostics["median_displayed_words"]:.1f} /
  {bundle_diagnostics["max_displayed_words"]}
- Range: {bundle_diagnostics["range_displayed_words"]}

## Automated validation

```json
{json.dumps(validation, indent=2, sort_keys=True)}
```

## Reproducibility

- Python: `{platform.python_version()}`
- NumPy: `{np.__version__}`
- SciPy: `{scipy.__version__}`
- PyYAML: `{yaml.__version__}`
- Source root: `{root}`

Private condition/provenance material is isolated under `generated/private/`.
Participant-safe files expose only opaque item/variant IDs and response text.
"""
    (out_dir / "AUDIT.md").write_text(audit, encoding="utf-8")
    artifact_paths = sorted(
        (
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        ),
        key=lambda path: str(path.relative_to(out_dir)),
    )
    (out_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(out_dir)}"
            for path in artifact_paths
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.modbench_root.resolve()
    out_dir = args.out_dir.resolve()
    task_path = root / "data" / "dataset" / "tasks.csv"
    persona_path = root / "configs" / "personas.yaml"
    rows, tasks_by_id = load_tasks(task_path)
    personas = load_personas(persona_path)
    inventory, inventory_stats, api_names = load_inventory(
        root, tasks_by_id, personas
    )
    category_targets, category_quotas = hamilton_targets(rows, N_TRIPLETS)
    selected, selection_diagnostics = select_triplets(
        rows, personas, inventory, category_targets, args.seed
    )
    triplets = materialize_triplets(selected, inventory, api_names, args.seed)
    comparisons = make_comparisons(triplets, args.seed)
    model_sets = build_bundle_model_sets(args.seed)
    assignments, bundle_diagnostics = assign_comparisons(
        triplets, comparisons, model_sets, args.seed
    )
    rationales = assign_rationales(
        comparisons, assignments, triplets, args.seed
    )
    variant_map, a_condition = assign_variants_and_sides(
        comparisons, assignments, args.seed
    )
    validation = validate_invariants(
        triplets,
        comparisons,
        assignments,
        model_sets,
        rationales,
        a_condition,
        category_targets,
    )
    build_outputs(
        out_dir=out_dir,
        root=root,
        seed=args.seed,
        task_path=task_path,
        persona_path=persona_path,
        triplets=triplets,
        comparisons=comparisons,
        assignments=assignments,
        model_sets=model_sets,
        rationales=rationales,
        variant_map=variant_map,
        a_condition=a_condition,
        inventory_stats=inventory_stats,
        category_targets=category_targets,
        category_quotas=category_quotas,
        selection_diagnostics=selection_diagnostics,
        bundle_diagnostics=bundle_diagnostics,
        validation=validation,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "seed": args.seed,
                "selection": selection_diagnostics,
                "bundle_workload": bundle_diagnostics,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
