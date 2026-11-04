#!/usr/bin/env python3
"""Independent cross-file validator for generated dual-contrast Study A."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "generated"
PRIVATE_MANIFEST = ROOT / "private" / "manifest_full.json"
PRIVATE_KEY = ROOT / "private" / "bundle_assignment_key.json"
PUBLIC_ITEMS = ROOT / "public" / "items_blinded.json"
PUBLIC_BUNDLES = ROOT / "public" / "bundles_public.json"
CHECKSUMS = ROOT / "checksums.sha256"
N_TRIPLETS = 42
N_COMPARISONS = 84
N_BUNDLES = 28
ITEMS_PER_BUNDLE = 6
RATERS_PER_COMPARISON = 2
MAX_RESPONSE_WORDS = 325
MAX_TRIPLET_WORD_RANGE = 30
PRIMARY_ENGLISH_RULE_VERSION = "primary-english-v1"
WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)
EXPECTED_CATEGORY_TARGETS = {
    "Abstract Conceptual Questions": 2,
    "Analytical and Interpretive Questions": 9,
    "Controversial Questions": 5,
    "Ideation and Brainstorming": 2,
    "Opinion-Based Questions": 4,
    "Personal Advice": 3,
    "Philosophical Questions": 3,
    "Recommendations": 1,
    "Speculative and Hypothetical Scenarios": 8,
    "Value-Laden Questions": 5,
}
PUBLIC_FORBIDDEN_KEYS = {
    "model",
    "model_code",
    "model_api",
    "persona",
    "persona_id",
    "persona_name",
    "role",
    "basic",
    "pro",
    "contrast",
    "triplet_id",
    "condition",
    "score",
    "similarity",
    "source_file",
    "source_path",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def main() -> None:
    errors: list[str] = []
    manifest = load(PRIVATE_MANIFEST)
    key = load(PRIVATE_KEY)
    public_items = load(PUBLIC_ITEMS)
    public_bundles = load(PUBLIC_BUNDLES)

    triplets = manifest["triplets"]
    private_items = manifest["items"]
    if len(triplets) != N_TRIPLETS:
        errors.append(f"Expected {N_TRIPLETS} private triplets")
    if len(private_items) != N_COMPARISONS:
        errors.append(f"Expected {N_COMPARISONS} private comparisons")
    triplet_by_id = {triplet["triplet_id"]: triplet for triplet in triplets}
    private_by_id = {item["item_id"]: item for item in private_items}
    public_by_id = {item["item_id"]: item for item in public_items["items"]}
    if len(triplet_by_id) != N_TRIPLETS:
        errors.append("Triplet IDs are not unique")
    if (
        len(private_by_id) != N_COMPARISONS
        or len(public_by_id) != N_COMPARISONS
        or set(private_by_id) != set(public_by_id)
    ):
        errors.append("Private/public comparison ID sets differ")
    if any(not re.fullmatch(r"SA-C\d{3}", item_id) for item_id in public_by_id):
        errors.append("Public comparison IDs are not opaque SA-C identifiers")

    if len({triplet["task_id"] for triplet in triplets}) != N_TRIPLETS:
        errors.append("Selected questions are not globally distinct")
    if Counter(triplet["model_code"] for triplet in triplets) != Counter(
        {
            "deepseek": 6,
            "gemini": 6,
            "gpt54mini": 6,
            "grok": 6,
            "kimi": 6,
            "llama": 6,
            "qwen": 6,
        }
    ):
        errors.append("Model allocation is not exactly 7×6")
    persona_counts = sorted(
        Counter(triplet["persona_name"] for triplet in triplets).values()
    )
    if persona_counts != [2] * 18 + [3] * 2:
        errors.append("Persona allocation is not balanced 18×2 plus 2×3")
    if Counter(triplet["subcategory"] for triplet in triplets) != Counter(
        EXPECTED_CATEGORY_TARGETS
    ):
        errors.append("Category allocation differs from exact target")

    screen = manifest["selection_population"]["readability_feasibility_screen"]
    if screen.get("max_words_per_complete_response") != MAX_RESPONSE_WORDS:
        errors.append("Wrong response-word cap in manifest")
    if (
        screen.get("max_word_range_across_role_basic_pro")
        != MAX_TRIPLET_WORD_RANGE
    ):
        errors.append("Wrong three-way word-range cap in manifest")
    if not manifest["selection_population"].get("single_call_required"):
        errors.append("Manifest does not require Single-Call")
    english_rule = screen.get("primary_english_rule", {})
    if (
        english_rule.get("version") != PRIMARY_ENGLISH_RULE_VERSION
        or not english_rule.get(
            "does_not_screen_relevance_refusal_style_or_opinion_breadth"
        )
    ):
        errors.append("Primary-English readability rule is missing or changed")
    selection = manifest.get("selection_diagnostics", {})
    base_language_exclusions = selection.get(
        "base_freeze_non_english_exclusions", []
    )
    if (
        len(base_language_exclusions) != 1
        or base_language_exclusions[0].get("provisional_freeze_reference")
        != "SA-T019 in the prelaunch provisional freeze"
        or base_language_exclusions[0].get("failed_conditions") != ["basic"]
    ):
        errors.append(
            "Expected only provisional SA-T019/basic to fail in the base freeze"
        )
    if (
        selection.get("retained_base_freeze_triplets") != 39
        or len(selection.get("minimal_change_removed", [])) != 3
        or len(selection.get("minimal_change_added", [])) != 3
    ):
        errors.append("Minimal three-cycle replacement audit is invalid")

    for triplet in triplets:
        condition_records = triplet["conditions"]
        if set(condition_records) != {"role", "basic", "pro"}:
            errors.append(f"{triplet['triplet_id']}: wrong condition set")
            continue
        counts = {}
        for condition, record in condition_records.items():
            text = record["response_text"]
            counts[condition] = word_count(text)
            recorded = triplet["displayed_word_counts"][
                f"{condition}_response"
            ]
            if recorded != counts[condition]:
                errors.append(
                    f"{triplet['triplet_id']}/{condition}: word-count mismatch"
                )
            audit = record["provenance"]["single_call_audit"]
            if (
                not audit.get("passed")
                or audit.get("message_count") != 2
                or audit.get("message_roles") != ["system", "user"]
            ):
                errors.append(
                    f"{triplet['triplet_id']}/{condition}: Single-Call failed"
                )
            english_audit = record["provenance"].get(
                "primary_english_audit", {}
            )
            if (
                not english_audit.get("passed")
                or english_audit.get("rule_version")
                != PRIMARY_ENGLISH_RULE_VERSION
            ):
                errors.append(
                    f"{triplet['triplet_id']}/{condition}: "
                    "primary-English readability failed"
                )
        if max(counts.values()) > MAX_RESPONSE_WORDS:
            errors.append(f"{triplet['triplet_id']}: response cap failed")
        if max(counts.values()) - min(counts.values()) > MAX_TRIPLET_WORD_RANGE:
            errors.append(f"{triplet['triplet_id']}: word-range cap failed")

    contrasts_per_triplet = Counter()
    public_keys = {
        key_name.lower()
        for key_name in walk_keys([public_items, public_bundles])
    }
    if leaked := sorted(public_keys & PUBLIC_FORBIDDEN_KEYS):
        errors.append(f"Participant-safe JSON leaks keys: {leaked}")
    for item_id, private_item in private_by_id.items():
        public_item = public_by_id[item_id]
        triplet = triplet_by_id[private_item["triplet_id"]]
        contrast = private_item["contrast"]
        contrasts_per_triplet[(triplet["triplet_id"], contrast)] += 1
        expected_conditions = (
            {"role", "basic"}
            if contrast == "role_basic"
            else {"role", "pro"}
            if contrast == "role_pro"
            else set()
        )
        if set(private_item["conditions"]) != expected_conditions:
            errors.append(f"{item_id}: invalid contrast/condition mapping")
        if public_item["question"] != triplet["query"]:
            errors.append(f"{item_id}: public/private question mismatch")
        if sorted(variant["text"] for variant in public_item["variants"]) != sorted(
            record["response_text"]
            for record in private_item["conditions"].values()
        ):
            errors.append(f"{item_id}: public/private response mismatch")
        if set(private_item["variant_condition_key"].values()) != expected_conditions:
            errors.append(f"{item_id}: variant key condition mismatch")
    expected_contrasts = Counter(
        {
            (triplet_id, contrast): 1
            for triplet_id in triplet_by_id
            for contrast in ("role_basic", "role_pro")
        }
    )
    if contrasts_per_triplet != expected_contrasts:
        errors.append("Each triplet does not have exactly one RB and one RP item")

    bundles = public_bundles["bundles"]
    private_bundles = key["bundles"]
    if len(bundles) != N_BUNDLES or len(private_bundles) != N_BUNDLES:
        errors.append(f"Expected {N_BUNDLES} public/private bundles")
    expected_files = {
        f"SA-B{number:02d}.json" for number in range(1, N_BUNDLES + 1)
    }
    observed_files = {
        path.name for path in (ROOT / "public" / "bundles").glob("SA-B*.json")
    }
    if observed_files != expected_files:
        errors.append("Standalone bundle file set is stale or incomplete")
    private_bundle_by_id = {
        bundle["bundle_id"]: bundle for bundle in private_bundles
    }
    occurrences = Counter()
    a_conditions_by_item: dict[str, set[str]] = {
        item_id: set() for item_id in private_by_id
    }
    rationale_items = []
    role_a = 0
    role_a_by_contrast = Counter()
    for public_bundle in bundles:
        bundle_id = public_bundle["bundle_id"]
        private_bundle = private_bundle_by_id.get(bundle_id)
        if private_bundle is None:
            errors.append(f"{bundle_id}: missing private key")
            continue
        if (
            len(public_bundle["items"]) != ITEMS_PER_BUNDLE
            or len(private_bundle["items"]) != ITEMS_PER_BUNDLE
        ):
            errors.append(f"{bundle_id}: wrong bundle size")
            continue
        public_positions = {
            item["display_position"]: item for item in public_bundle["items"]
        }
        private_positions = {
            item["display_position"]: item for item in private_bundle["items"]
        }
        if set(public_positions) != set(range(1, ITEMS_PER_BUNDLE + 1)):
            errors.append(f"{bundle_id}: invalid public positions")
        if set(private_positions) != set(range(1, ITEMS_PER_BUNDLE + 1)):
            errors.append(f"{bundle_id}: invalid private positions")
        triplet_ids = set()
        models = set()
        personas = set()
        contrast_counts = Counter()
        bundle_role_a = 0
        bundle_role_a_by_contrast = Counter()
        for position in range(1, ITEMS_PER_BUNDLE + 1):
            public_item = public_positions[position]
            private_item = private_positions[position]
            item_id = public_item["item_id"]
            if item_id != private_item["item_id"]:
                errors.append(f"{bundle_id}/{position}: item mismatch")
                continue
            source = private_by_id[item_id]
            occurrences[item_id] += 1
            triplet_ids.add(private_item["triplet_id"])
            models.add(private_item["model_code"])
            personas.add(private_item["persona_name"])
            contrast_counts[private_item["contrast"]] += 1
            condition_a = private_item["response_a_condition"]
            a_conditions_by_item[item_id].add(condition_a)
            if condition_a == "role":
                role_a += 1
                bundle_role_a += 1
                role_a_by_contrast[private_item["contrast"]] += 1
                bundle_role_a_by_contrast[private_item["contrast"]] += 1
            if condition_a not in source["conditions"]:
                errors.append(f"{bundle_id}/{item_id}: invalid A condition")
            elif public_item["response_a"] != source["conditions"][condition_a][
                "response_text"
            ]:
                errors.append(f"{bundle_id}/{item_id}: Response A mismatch")
            if private_item["rationale_required_after_lock"]:
                rationale_items.append(item_id)
        if len(triplet_ids) != ITEMS_PER_BUNDLE:
            errors.append(f"{bundle_id}: both contrasts from one triplet")
        if len(models) != ITEMS_PER_BUNDLE:
            errors.append(f"{bundle_id}: models are not distinct")
        if len(personas) != ITEMS_PER_BUNDLE:
            errors.append(f"{bundle_id}: personas are not distinct")
        if contrast_counts != Counter({"role_basic": 3, "role_pro": 3}):
            errors.append(f"{bundle_id}: contrast split is not 3/3")
        if bundle_role_a != 3:
            errors.append(f"{bundle_id}: A/B condition balance is not 3/3")
        if any(
            bundle_role_a_by_contrast[contrast] not in {1, 2}
            for contrast in ("role_basic", "role_pro")
        ):
            errors.append(
                f"{bundle_id}: within-contrast A/B balance is not 1/2"
            )
        rationale_ids = private_bundle["rationale_item_ids_after_primary_lock"]
        if len(rationale_ids) != 1:
            errors.append(f"{bundle_id}: rationale count is not one")
        separate = ROOT / "public" / "bundles" / f"{bundle_id}.json"
        if not separate.is_file() or load(separate) != public_bundle:
            errors.append(f"{bundle_id}: standalone bundle mismatch")
    if any(
        occurrences[item_id] != RATERS_PER_COMPARISON
        for item_id in private_by_id
    ):
        errors.append("Every comparison must occur exactly twice")
    for item_id, conditions in a_conditions_by_item.items():
        expected = set(private_by_id[item_id]["conditions"])
        if conditions != expected:
            errors.append(
                f"{item_id}: two ratings do not use opposite A/B orientations"
            )
    if len(rationale_items) != N_BUNDLES or len(set(rationale_items)) != N_BUNDLES:
        errors.append("Rationale items must be 28 distinct comparisons")
    rationale_by_contrast = Counter(
        private_by_id[item_id]["contrast"] for item_id in rationale_items
    )
    rationale_by_model = Counter(
        private_by_id[item_id]["model_code"] for item_id in rationale_items
    )
    if rationale_by_contrast != Counter({"role_basic": 14, "role_pro": 14}):
        errors.append("Rationale allocation is not 14/14 by contrast")
    if rationale_by_model != Counter(
        {
            "deepseek": 4,
            "gemini": 4,
            "gpt54mini": 4,
            "grok": 4,
            "kimi": 4,
            "llama": 4,
            "qwen": 4,
        }
    ):
        errors.append("Rationale allocation is not four per model")
    if role_a != 84 or role_a_by_contrast != Counter(
        {"role_basic": 42, "role_pro": 42}
    ):
        errors.append(
            f"Orientation balance failed: Role-A={role_a}, "
            f"by contrast={role_a_by_contrast}"
        )

    checksum_pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        match = checksum_pattern.fullmatch(line)
        if not match:
            errors.append(f"Malformed checksum line: {line}")
            continue
        expected, relative = match.groups()
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"Checksum mismatch: {relative}")

    if errors:
        raise AssertionError("\n".join(errors))
    print(
        json.dumps(
            {
                "status": "ok",
                "all_cross_file_checks_passed": True,
                "triplets": N_TRIPLETS,
                "comparisons": N_COMPARISONS,
                "bundles": N_BUNDLES,
                "judgments": N_COMPARISONS * RATERS_PER_COMPARISON,
                "rationales": N_BUNDLES,
                "role_as_a": role_a,
                "nonrole_as_a": 168 - role_a,
                "role_as_a_by_contrast": dict(role_a_by_contrast),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
