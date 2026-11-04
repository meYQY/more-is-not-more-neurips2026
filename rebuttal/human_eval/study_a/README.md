# Study A dual-contrast formal sample generator

> **Release note.** This folder is the item-selection code for the human
> evaluation (paper Appendix J). The public release contains `README.md`,
> `AUDIT.md` (the generator's audit of the frozen sample; originally under
> `generated/`), `generate_study_a.py`, `validate_generated.py`, and
> `item_manifest.csv` (the 84 comparison items with model, question, persona,
> word counts, and bundle placements, derived from the analyzed judgments).
> The `generated/` outputs (private crosswalks containing the raw source
> records, and participant-facing bundles containing full response texts) are
> not released. Rerunning the generator requires the full local tree of
> Single-Call raw outputs for all seven models (`--modbench-root`), which is
> also not released. The study web site mentioned below is not released.

This directory contains the deterministic generator for the human study that
compares Role with both Basic and Pro.

Run:

```bash
python3 generate_study_a.py
python3 validate_generated.py
```

The frozen data design is:

- 42 outcome-blind matched `(model, question, persona)` triplets;
- complete Single-Call outputs for Role, Basic, and Pro in every triplet;
- 7 models × 6 triplets, 42 distinct questions, and all 20 personas
  (18 appear twice and 2 appear three times);
- all 10 question subcategories, using the exact Hamilton allocation;
- every complete response contains at most 325 displayed words;
- `max(words) - min(words) <= 30` across Role, Basic, and Pro, which implies
  that every pair among the three differs by at most 30 words;
- Role, Basic, and Pro each pass the same conservative primary-English
  readability rule; if any one response is clearly not primarily English, the
  whole triplet is ineligible for English-language annotators;
- 84 blinded comparisons: 42 Role--Basic and 42 Role--Pro;
- 28 participant bundles × 6 scientific comparisons;
- exactly 3 Role--Basic and 3 Role--Pro comparisons per bundle;
- each comparison appears in 2 different bundles, once in each A/B
  orientation, for 168 judgments;
- no bundle contains both contrasts from one triplet;
- one post-lock rationale prompt per bundle on 28 distinct comparisons,
  balanced 14/14 by contrast and exactly four per model;
- exact global A/B balance (84 Role-as-A, 84 comparator-as-A), with 3/3 in
  every bundle and 1/1 within every comparison.

Selection never uses response meaning, relevance, refusal behavior, style,
opinion breadth, embeddings, MPD, human judgments, or an expected outcome.
After the symmetric length/primary-English screen, only metadata and a
fixed-seed hash enter the selection optimizer.

The prelaunch audit found one failure inside the provisional frozen sample:
provisional `SA-T019` had a Basic response that was predominantly Spanish,
although its Role and Pro responses passed. The generator first reproduces the
original metadata-hash freeze, then minimizes the number of replacements while
preserving every model, category, question-uniqueness, persona-frequency, and
model–persona constraint. A one-item substitution was infeasible under those
constraints, so the recorded repair is the minimum three-cycle: 39 of 42
original triplets remain. The other two removed triplets passed every
readability rule and moved only to preserve the frozen stratification; they
were not content exclusions. Exact source keys and language-audit counts are
recorded privately in `generated/AUDIT.md` and `manifest_full.json`.

After the frozen assignment MILP, deterministic valid same-model/same-contrast
local swaps reduce bundle workload imbalance without changing any scientific
allocation or using outcome information.

Generated files:

- `generated/private/manifest_full.json`: private triplets, comparison keys,
  raw source records, and exact provenance;
- `generated/private/bundle_assignment_key.json`: private A/B, contrast,
  triplet, and rationale keys;
- `generated/public/items_blinded.json`: participant-safe comparison bank;
- `generated/public/bundles_public.json`: all participant-safe bundles;
- `generated/public/bundles/`: one JSON file per bundle;
- `generated/AUDIT.md`: design and validation audit;
- `generated/checksums.sha256`: hashes for every generated artifact.

Never deploy files under `generated/private/` to a browser client.

To refresh the site-compatible public payload after validation:

```bash
cd ../prolific-study-site
npm run data:build
```
