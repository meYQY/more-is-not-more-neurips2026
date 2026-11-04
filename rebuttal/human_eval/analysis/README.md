# Frozen Study A/B analysis package

> **Release note.** This is the frozen analyzer used for the human evaluation
> (paper Appendix J), released for inspection. Its inputs are the raw study
> database exports (which contain Prolific identifiers and free-text
> rationales) and the private frozen crosswalks under
> `../study_a/generated/private/`; none of these are released, so neither the
> analyzer nor the tests in `tests/` (which load the private Study A manifest
> and the inactive Study B artifacts) run from the public files alone. To
> recompute the reported table from the released de-identified judgments, run
> `python rebuttal/human_eval/reproduce_table.py`, which reuses this module's
> statistics functions. Study B mentioned below was archived/inactive; the
> analyzed sample contains no Study B judgments, and it is not reported in the
> paper.

This directory is the post-collection analysis for active Study A and the
archived/inactive Study B audit. It reads database exports, joins them to the private frozen
crosswalks, and writes both the all-paid and predeclared QC-filtered results.
It does not modify the annotation site, public bundles, private manifests, or
sample-selection artifacts.

The implementation uses only the Python 3.12 standard library. Every stochastic
procedure has an explicit seed, and `summary.json` records SHA-256 hashes of
both exports and every frozen metadata input.

The current Study A freeze contains 42 matched Role/Basic/Pro triplets. Each
triplet yields one Role–Basic and one Role–Pro comparison, for 84 comparison
items arranged into 28 six-item bundles (3 of each contrast per bundle). Every
comparison item occurs in two distinct bundles. The three-person scientific
pilot covers `SA-B03`, `SA-B05`, and `SA-B21`; the 25-person continuation
covers the other bundles, so the combined export contains 168 judgments. The
analyzer derives all of these values from
the private frozen manifest and bundle key. It remains backward compatible
with the archived Role–Pro-only freeze.
`summary.json.configuration.frozen_design` records the exact derived contract
used in each run. The legacy Study B frozen inputs remain supported unchanged.

## Run the synthetic end-to-end test

From this directory:

```bash
python3 -m unittest discover -s tests -v
```

The fixture contains three paid sessions per study, one participant per study
with a predeclared exclusion, opaque Study A X/Y responses, compatibility
coverage for the completed pilot's legacy `A-IMC-1`, and Study B examples from
all five frozen similarity strata. It writes a session CSV and response JSON
into a temporary directory and verifies the generated analysis tables.

## Export and prepare real data

Export `study_sessions`, `study_responses`, and optionally `study_events` as
CSV or JSON. A combined site export containing all three arrays can be supplied
to both required arguments; telemetry is auto-discovered. The reader accepts:

- a CSV with the database column names;
- a JSON array of rows;
- `{ "sessions": [...] }`, `{ "study_sessions": [...] }`,
  `{ "responses": [...] }`, `{ "study_responses": [...] }`,
  `{ "events": [...] }`, or `{ "study_events": [...] }`;
- Cloudflare D1 / Wrangler JSON shaped as
  `[{ "results": [...], "success": true, ... }]`.

The site database deliberately does not contain Prolific payment status. Before
the final analysis, join the Prolific payment ledger to the session export and
add a boolean `paid` column. This is necessary for “all-paid” to mean exactly
what it says. Every analysis sample additionally requires a server-recorded
consent timestamp and a final commit with both `status=complete` and
`completed_at`. If `paid` is absent, the script uses those consented, completed
sessions as a clearly warned pilot fallback; do not use that fallback in the
paper.
If any row has an explicit payment field, every non-demo row must have a
nonblank true/false payment value; a partial ledger join is a fatal error.

Manual QC review may add these participant-level columns to the session export:

- `qc_include`: `0` only for a predeclared manual exclusion, such as confirmed
  gibberish throughout the required rationales;
- `qc_exclusion_reason`: the auditable reason;
- alternatively, `exclude_from_qc=1`.

Do not set these based on whether a participant supports the paper. The script
does not try to detect gibberish automatically. A first run creates the private
`rationale_review.csv`; review it, record any predeclared manual decision in the
session export, and rerun into a fresh output directory.

Demo records whose Prolific identifiers begin with `DEMO-` are always kept out
of both analysis samples.

Valid frozen-protocol labels from the scientific pilot are part of the final
sample. Do **not** pass the pilot study ID to
`--exclude-prolific-study-id` merely because the session came from the pilot
listing. In particular, the completed `SA-B03`, `SA-B05`, and `SA-B21`
sessions contribute their 18 formal task judgments and must not be rerun.
Their legacy `A-IMC-1` rows are retained only for audit and are never extracted
as preferences. If a different listing is explicitly declared superseded, use
the flag only for that listing; those sessions remain auditable in
`session_qc.csv` but enter neither analysis sample:

```bash
python3 analyze_annotations.py \
  --sessions /absolute/path/modbench-study-export.json \
  --responses /absolute/path/modbench-study-export.json \
  --outdir /absolute/path/analysis-results \
  --exclude-prolific-study-id SUPERSEDED_PILOT_STUDY_ID
```

The flag is repeatable. The configuration and warning report record every
requested ID, the number of matched sessions, and the number that would
otherwise have entered `all_paid`. This is a predeclared superseded-listing
exclusion, not a post-hoc QC decision.

The final run also fails if a Prolific participant ID is blank, if one paid PID
has multiple sessions in the same study, or if a paid PID appears in both
studies. These are frozen design-integrity checks, not QC exclusions.

## Run the frozen analysis

```bash
python3 analyze_annotations.py \
  --sessions /absolute/path/study_sessions.csv \
  --responses /absolute/path/study_responses.json \
  --events /absolute/path/study_events.json \
  --outdir /absolute/path/analysis-results
```

Defaults point to the frozen Study A and Study B artifacts one directory above
this package. They can be overridden only when reproducing an archived version:

```bash
python3 analyze_annotations.py \
  --sessions /absolute/path/study_sessions.csv \
  --responses /absolute/path/study_responses.csv \
  --outdir /absolute/path/analysis-results \
  --study-a-manifest ../study_a/generated/private/manifest_full.json \
  --study-a-bundles ../study_a/generated/private/bundle_assignment_key.json \
  --study-a-public ../study_a/generated/public/bundles_public.json \
  --study-b-provenance ../study_b/outputs/provenance_manifest.json \
  --study-b-bundles ../study_b/outputs/bundles_manifest.json \
  --study-b-validation ../study_b/outputs/validation.json \
  --bootstrap-iterations 10000 \
  --permutation-iterations 10000 \
  --seed 20260727
```

The command returns exit status 2 for ambiguous or validity-threatening input,
such as conflicting locked primary responses.

## QC samples

`all_paid` contains every non-demo session with `paid=1`, recorded consent, and
a completed final server commit, except explicitly identified superseded
listings. The QC-filtered sensitivity is a subset and excludes a session for
any of:

- consent not recorded;
- incomplete final server commit;
- comprehension not passed;
- any required primary answer missing, unlocked, or invalid;
- when the export uses the current Study A confidence protocol, any required
  scientific-task confidence missing, unlocked, or outside `1`–`5`;
- any explicitly required legacy check missing or unlocked;
- both Study B attention checks answered incorrectly;
- any required rationale missing or unlocked;
- an explicit, predeclared manual exclusion.

Current Study A main sessions have no required instruction/attention check
(`required_imc_ids=[]`). The completed pilot sessions may carry the legacy
`A-IMC-1`; its answer and confidence remain audit-only, never enter
`study_a_judgments.csv` or task-confidence means, and never change `all_paid`
or `qc_filtered`. The legacy Study B protocol retains its two checks, and a
Study B participant is excluded only for failing both. Likewise,
`mean_task_confidence < 3.0` is recorded as a participant-level diagnostic, not
an exclusion. Answers that disagree with the paper never cause exclusion.
Exclusion reasons and diagnostic flags are separate in `session_qc.csv`.

QC removal can leave fewer than two judgments on a Study A item. The current
design does not infer an item majority from two raters. Both judgments are
retained; opposing directional judgments contribute `−1` and `+1`, hence an
item-level contrast preference of zero. Agreement/disagreement remains
descriptive. The output reports `n_ratings` and `complete_design_ratings`.

## Study A estimands

Opaque comparison answers are mapped to Role/Basic/Pro through
`manifest_full.json`; the displayed side and contrast are never inferred from
text or length.

Role–Basic and Role–Pro are reported separately. For each contrast the package
reports:

- the full Role / other-condition / Tie / Unable judgment distribution;
- two-rater agreement/disagreement patterns without manufacturing a majority;
- signed contrast `D = P(other condition broader) − P(Role broader)`, where
  each judgment is `+1` for Basic/Pro, `−1` for Role, and `0` for Tie/Unable;
- a 95% percentile bootstrap CI that resamples matched triplets with
  replacement within model while retaining all judgments for each sampled
  comparison; the two contrast analyses use the same triplet-level scientific
  unit;
- a participant-cluster bootstrap sensitivity that retains all six judgments
  from a sampled participant together;
- descriptive 1–5 confidence and per-response timing summaries over the six
  formal task items when confidence rows are present; any legacy pilot
  `A-IMC-1` confidence is never included in these means, and confidence never
  reweights primary judgments;
- model-level descriptives only;
- contrast-specific pairwise agreement and nominal Krippendorff’s alpha.

`D` is a directional effect, not an equivalence test. A CI containing zero
must not be described as proof that Role and Basic/Pro are equal. The generated
wording is limited to whether a stable human-perceived non-Role advantage was
detected.

## Study B estimands

The frozen site answers are coded as:

- any displayed candidate ID or `multiple_candidates` → `Same`;
- `none_related` → `Related-but-distinct`;
- `none_unrelated` → `Different`;
- `cannot_context` → `Unable`.

The package reports:

- full Same / Related / Different / Unable distributions;
- four-way and Same-vs-no-Same item majorities;
- unique/control results overall, by direction, by stratum, by their
  interaction, and descriptively by model;
- a two-sided Fisher exact test on item Same-majority status;
- an unblocked exact fixed-margin majority test, plus block-exact majority and
  Same-fraction randomization tests that preserve the control count within every
  `model × direction` block;
- a seeded item-label permutation test on the item Same-vote fraction,
  restricted within the same frozen blocks;
- far → medium → near → just-above → clear descriptive rates and a
  item-level Spearman trend whose outcomes are permuted within blocks;
- four-way and collapsed nominal agreement;
- a conservative sensitivity that recodes every Unable against the machine
  label: Unable becomes Same for machine-unique items and no-Same for controls,
  rerunning Fisher and both exact/Monte Carlo blocked Same-fraction tests; the
  recode is recalculated after every hypothetical group assignment.

The majority-based exact tables include only items with at least two retained
ratings and a resolved Same/no-Same strict majority. Unable-majority,
no-consensus, and insufficient items remain in the descriptive files but are
not silently coded as not-Same. The continuous Same-fraction permutation keeps
items with at least two retained ratings.

Study B remains a targeted top-three retrieval audit. These outputs cannot
support claims of exhaustive human UCR recomputation or universal embedding
validity.

## Outputs

- `summary.json`: compact results, methods, seeds, hashes, and warnings.
- `report.md`: reviewer-facing descriptive summary with scope caveats.

## Standard participant process printout

For a private, anonymized inspection of what each participant selected and how
the browser session progressed, run:

```bash
python3 print_participant_process_report.py \
  --export /absolute/path/modbench-study-export.json \
  --outdir /absolute/path/process-report
```

This writes:

- `participant_process_report.md`: the standard human-readable printout;
- `participant_process_report.json`: the same anonymized structure;
- `participant_item_process.csv`: one row per displayed comparison/check.

The standard includes actual randomized item order, visible A/B condition
mapping, selected option, decoded condition, confidence, primary-response time,
document scroll milestones, 30-second runtime heartbeats, focus/visibility
interruptions, and required rationale text/dwell time.

The site does **not** collect mouse coordinates or cursor trajectories.
`runtime_snapshot` is only a browser-window proxy: it accrues while the document
is visible and the window is focused. Scroll and focus telemetry are
descriptive, not automatic grounds for exclusion. The printout deliberately
omits Prolific IDs, internal/external session IDs, study IDs, run IDs,
assignment tokens, and database row IDs.
- `session_qc.csv`: all participant-level inclusion decisions and reasons.
- `rationale_review.csv`: private rationale-review sheet with the Study A
  contrast/triplet mapping, question, full relevant Role/Basic/Pro texts,
  decoded choice, persona/model metadata, rationale text, and blank reviewer
  fields; it does not assign a semantic quality label.
- `study_a_judgments.csv`: decoded judgment-level Study A data.
- `study_a_item_summaries.csv`: two-rater response counts, descriptive
  agreement/disagreement pattern, and signed contrast preference; no majority
  is inferred.
- `study_a_raw_distribution.csv`: overall/model/subcategory distributions.
- `study_a_model_descriptives.csv`: per-model descriptive results.
- `study_a_confidence_timing.csv`: formal-task confidence and primary-response
  time summaries overall and by contrast/outcome; any legacy pilot `A-IMC-1`
  confidence is excluded.
- `study_b_judgments.csv`: coded Study B judgments with frozen metadata.
- `study_b_item_majorities.csv`: four-way/binary majorities and sensitivities.
- `study_b_raw_distribution.csv`: overall and stratified raw distributions.
- `study_b_group_summaries.csv`: unique/control/direction/stratum/model tables.
- `study_b_similarity_gradient.csv`: five-stratum gradient.
- `study_b_exact_and_permutation.csv`: exact, permutation, and trend tests.
- `study_b_unable_sensitivity.csv`: primary versus adversarial Unable results.
- `agreement.csv`: pairwise agreement and nominal alpha under each coding.
- `telemetry_events.csv`: optional normalized server telemetry with session/QC
  joins.
- `telemetry_session_summary.csv`: optional event counts and elapsed-time
  coverage by participant session.

Telemetry and per-task time are descriptive process diagnostics. They are not
automatic Prolific rejection or scientific-exclusion rules.

See `DATA_DICTIONARY.md` for field-level definitions.
