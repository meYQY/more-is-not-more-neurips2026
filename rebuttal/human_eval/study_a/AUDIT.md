# Study A dual-contrast formal sample audit

Generated deterministically by `generate_study_a.py`.

## Frozen design

- Seed: `20260727`
- Selection units: 42 matched Role--Basic--Pro Single-Call triplets
- Comparisons: 84 blinded pairs (42 Role--Basic and 42 Role--Pro)
- Models: 7 × 6 triplets
- Questions: 42 distinct
- Personas: all 20 (18 twice, 2 three times)
- Subcategories: all 10, exact Hamilton allocation
- Readability: every complete response <= 325 words
- Strong three-way length match: max minus min across Role, Basic, Pro
  <= 30 words (therefore both Role--Basic and Role--Pro,
  and also Basic--Pro, are <= 30)
- English readability: Role, Basic, and Pro must each pass the same conservative
  `primary-english-v1` screen. The rule only detects clearly
  non-primary-English output; it does not screen relevance, refusal, style,
  response quality, or opinion breadth.
- Bundles: 28 × 6; exactly 3 Role--Basic and 3 Role--Pro
- Replication: each comparison appears in 2 distinct bundles
  (168 scientific judgments)
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

- Technically valid triplets: 13837
- Length-eligible triplets: 1086
- Screen-eligible triplets: 1083
- Excluded by symmetric length screen: 12751
- Excluded by symmetric primary-English screen: 3
- Original base-freeze triplets retained: 39 / 42
- Prelaunch base-freeze language exclusions:
  `[{"condition_audits": {"basic": {"english_function_hits": 4, "failure_reasons": ["non_english_latin_function_words_dominate"], "non_english_latin_function_hits": 60, "non_latin_letter_fraction": 0.0, "passed": false, "rule_version": "primary-english-v1", "token_count": 226}, "pro": {"english_function_hits": 61, "failure_reasons": [], "non_english_latin_function_hits": 4, "non_latin_letter_fraction": 0.0, "passed": true, "rule_version": "primary-english-v1", "token_count": 251}, "role": {"english_function_hits": 51, "failure_reasons": [], "non_english_latin_function_hits": 3, "non_latin_letter_fraction": 0.0, "passed": true, "rule_version": "primary-english-v1", "token_count": 222}}, "failed_conditions": ["basic"], "model_code": "grok", "passed": false, "persona_name": "Maria Rodriguez", "provisional_freeze_reference": "SA-T019 in the prelaunch provisional freeze", "task_id": "MOD-090"}]`
- Minimal-change removals:
  `[{"model_code": "deepseek", "persona_name": "Tyrone Jackson", "task_id": "MOD-060"}, {"model_code": "grok", "persona_name": "Maria Rodriguez", "task_id": "MOD-090"}, {"model_code": "kimi", "persona_name": "Rosa Hernandez", "task_id": "MOD-095"}]`
- Minimal-change replacements added:
  `[{"model_code": "deepseek", "persona_name": "Rosa Hernandez", "task_id": "MOD-058"}, {"model_code": "grok", "persona_name": "Tyrone Jackson", "task_id": "MOD-090"}, {"model_code": "kimi", "persona_name": "Maria Rodriguez", "task_id": "MOD-094"}]`

Only the triplet identified as provisional `SA-T019` failed the screen in the
base freeze. Any additional triplets in `minimal_change_removed` passed every
readability rule and were changed only because the exact model, category,
question-uniqueness, persona-frequency, and model--persona constraints make a
one-for-one replacement infeasible. The recorded three-cycle is the
minimum-change feasible repair; those additional removals are not content
exclusions.
- Selected triplet word-range min / median / max:
  4 / 21.0 / 30

## Category allocation

| Category | Quota | Target | Observed |
|---|---:|---:|---:|
| Abstract Conceptual Questions | 2.10 | 2 | 2 |
| Analytical and Interpretive Questions | 8.40 | 9 | 9 |
| Controversial Questions | 5.04 | 5 | 5 |
| Ideation and Brainstorming | 1.68 | 2 | 2 |
| Opinion-Based Questions | 4.20 | 4 | 4 |
| Personal Advice | 3.36 | 3 | 3 |
| Philosophical Questions | 3.36 | 3 | 3 |
| Recommendations | 1.26 | 1 | 1 |
| Speculative and Hypothetical Scenarios | 7.56 | 8 | 8 |
| Value-Laden Questions | 5.04 | 5 | 5 |

## Bundle workload

- Displayed-word total min / median / max:
  2745 /
  2776.5 /
  2802
- Range: 57

## Automated validation

```json
{
  "all_invariants_passed": true,
  "n_bundles": 28,
  "n_comparisons": 84,
  "n_judgments": 168,
  "n_rationales": 28,
  "n_triplets": 42,
  "nonrole_as_response_a": 84,
  "role_as_a_by_contrast": {
    "role_basic": 42,
    "role_pro": 42
  },
  "role_as_response_a": 84
}
```

## Reproducibility

- Python: `3.12.8`
- NumPy: `1.26.4`
- SciPy: `1.16.0`
- PyYAML: `6.0.3`
- Source root: `<local full experiment tree>`

Private condition/provenance material is isolated under `generated/private/`.
Participant-safe files expose only opaque item/variant IDs and response text.
