# Analysis data dictionary

Both snake_case database names and the site’s camelCase API names are accepted
where noted. JSON array fields may be actual arrays or JSON-encoded strings.
Boolean fields accept `0/1`, `true/false`, and `yes/no`.

## Input: sessions

| Canonical field | Accepted source fields | Required | Meaning |
|---|---|---:|---|
| `session_id` | `id`, `internal_session_id`, `session_id` | yes | Internal D1 primary key. This is not Prolific’s session identifier. |
| `study_type` | `study_type`, `studyType` | yes | `A` or `B`. |
| `bundle_id` | `bundle_id`, `bundleId` | yes for final | Frozen assigned bundle. |
| `prolific_pid` | `prolific_pid`, `prolificPid` | yes | Anonymous participant ID. |
| `prolific_study_id` | `prolific_study_id`, `prolificStudyId` | yes | Prolific study ID. |
| `prolific_session_id` | `prolific_session_id`, `prolificSessionId` | yes | Prolific session ID. |
| `status` | `status` | yes | Site phase; final commits are `complete`. |
| `comprehension_passed` | `comprehension_passed`, `comprehensionPassed` | yes | Whether the comprehension gate passed. |
| `consented_at` | `consented_at`, `consentedAt` | yes | Server-recorded consent timestamp; blank sessions are never analyzed. |
| `rationale_item_ids` | `rationale_item_ids`, `rationaleItemIds` | yes | Study A’s one server-revealed rationale item ID in the current bundle design. |
| `required_item_ids` | `required_item_ids`, `requiredItemIds` | yes | Frozen formal item IDs required for that session. |
| `required_imc_ids` | `required_imc_ids`, `requiredImcIds` | yes | Empty for current Study A main sessions; the completed pilot may explicitly store legacy `A-IMC-1`. Study B uses legacy `B-IMC-1` and `B-IMC-2`. |
| `completed_at` | `completed_at`, `completedAt` | yes | Final server commit timestamp; blank before completion. |
| `paid` | `paid`, `is_paid`, `prolific_paid` | required for final all-paid analysis | Payment status joined from Prolific. Not stored by the site. |
| `qc_include` | `qc_include`, `include_in_qc` | optional | Set false only after predeclared manual QC review. |
| `exclude_from_qc` | `exclude_from_qc`, `qc_exclude` | optional | Equivalent explicit exclusion flag. |
| `qc_exclusion_reason` | `qc_exclusion_reason`, `qc_reason`, `exclusion_reason` | required with manual exclusion | Auditable manual reason, never result direction. |

If required item arrays are absent, the script first derives them from the
frozen bundle manifests. Inferring them from observed responses is a warned
pilot-only fallback.

A partially populated payment column is rejected. Final analysis also requires
one nonblank PID and at most one paid session per PID per study, with no paid
PID shared across A and B.

## Input: responses

| Canonical field | Accepted source fields | Required | Meaning |
|---|---|---:|---|
| `response_id` | `id`, `response_id` | no | D1 response primary key. |
| `session_id` | `session_id`, `sessionId` | yes | Internal session foreign key. |
| `item_id` | `item_id`, `itemId` | yes | Formal task or attention-check ID. |
| `item_kind` | `item_kind`, `itemKind` | yes | `task` or `imc`. |
| `stage` | `stage` | yes | `primary`, `confidence`, or `rationale`. |
| `answer` | `answer` | yes | Opaque primary answer, Study A scientific-task confidence `1`–`5`, rationale text, or an audit-only legacy check answer/confidence. |
| `presentation_order` | `presentation_order`, `presentationOrder` | yes for primary | JSON list of opaque displayed response/candidate IDs. |
| `duration_ms` | `duration_ms`, `durationMs` | yes | Client-measured time on that task. Descriptive only. |
| `locked` | `locked` | yes | Primary/finalized rationale immutability flag. |
| `updated_at` | `updated_at`, `updatedAt` | recommended | Used to select the latest repeated autosave snapshot. |

Conflicting locked primary exports are fatal. Repeated rationale autosaves are
collapsed to the latest `updated_at` value and generate a warning.

Confidence is joined to the matching Study A primary by
`(session_id, item_id)`. Archived exports with no confidence protocol at all
remain valid. Once current Study A confidence rows are present, every required
formal task must have a valid locked `1`–`5` response; missing or invalid
required confidence excludes that session from `qc_filtered` but not from
`all_paid`. Current main sessions require no IMC. If a completed pilot session
explicitly lists legacy `A-IMC-1`, that row remains audit-only and its
confidence is never included in a formal-task mean. Confidence never changes
the primary code or judgment weight.

## Optional input: telemetry events

`--events` accepts CSV/JSON fields `id`, `session_id`, `event_type`,
`event_data`, `elapsed_ms`, and `created_at` (camelCase aliases are accepted).
When sessions/responses point to the combined site JSON export,
`study_events` is auto-discovered. JSON `event_data` is retained canonically;
`item_id` and `duration_ms` are extracted when present. Telemetry is descriptive
only and never creates an automatic exclusion.

## `session_qc.csv`

| Field | Meaning |
|---|---|
| `paid_analysis_eligible_before_pilot_exclusion` | Legacy-named field: whether the row satisfies the paid/consent/completion/non-demo gate before an explicit superseded-listing exclusion. |
| `pilot_excluded` | Legacy-named field: whether `--exclude-prolific-study-id` explicitly removes this superseded listing from both scientific samples. Valid scientific-pilot sessions are included by default. |
| `analysis_exclusion_reason` | Auditable explicit study-ID exclusion reason; blank otherwise. |
| `in_all_paid` | Explicitly paid, consented, server-completed non-demo session that is not from an explicitly excluded superseded listing, or the same consent/completion gate used as a warned fallback when payment is unavailable. |
| `in_qc_filtered` | `in_all_paid` and no automatic/manual exclusion flag. |
| `paid_source` | `explicit_paid_field` or `completed_fallback`. |
| `consent_recorded` | Whether the server stored a nonblank consent timestamp. |
| `completion_recorded` | Whether both `status=complete` and `completed_at` are present. |
| `required_task_count` | Expected formal items from session/bundle. |
| `frozen_assignment_mismatch` | Session-required items/rationales/IMCs conflict with the frozen bundle key. |
| `missing_primary_count` | Required formal primary answers absent. |
| `invalid_primary_count` | Answers not decodable under the frozen task. |
| `missing_confidence_count` | Study A formal task items without a locked confidence row under the current confidence protocol. |
| `invalid_confidence_count` | Present Study A formal-task confidence outside `1`–`5`. |
| `task_confidence_count` | Number of required formal Study A task confidences successfully parsed; six in a complete current session. |
| `mean_task_confidence` | Mean over valid required formal task confidences only; never includes legacy pilot `A-IMC-1` confidence. |
| `low_mean_task_confidence` | `true` only when all required formal-task confidences are valid and their mean is below `3.0`; diagnostic only. |
| `missing_imc_confidence_count` | Missing locked confidence for an explicitly required legacy check; zero by design for current Study A main sessions. |
| `invalid_imc_confidence_count` | Invalid confidence on an explicitly required legacy check; zero by design for current Study A main sessions. |
| `attention_check_confidence` | Parsed legacy pilot `A-IMC-1` confidence when present; kept separate from formal-task means. |
| `wrong_imc_count` | Present but incorrect explicit legacy checks. |
| `attention_check_failed_diagnostic` | Whether legacy pilot `A-IMC-1` was answered incorrectly; audit diagnostic only. |
| `missing_imc_count` | Missing or unlocked explicitly required legacy checks; zero by design for current Study A main sessions. |
| `missing_rationale_count` | Missing or unlocked required rationales. |
| `diagnostic_flags` | Pipe-separated non-exclusion diagnostics: legacy pilot Study A check failure when applicable and/or mean formal-task confidence below `3.0`. |
| `qc_flags` | Pipe-separated exclusion reasons. Empty means eligible. |

## `rationale_review.csv`

This private sheet contains every latest exported rationale, its word count,
lock status, whether it was required, and the session’s current QC status. For
Study A it additionally includes `triplet_id`, `contrast`,
`other_condition`, the question, complete relevant Role/Basic/Pro responses,
decoded choice, task/model/persona metadata, and the current
exclusion/diagnostic flags. The following fields are intentionally blank for
manual completion:

- `manual_rationale_valid`;
- `manual_review_reason`;
- `manual_reviewer`.

It is provided for human review of the predeclared “clear gibberish throughout”
criterion. The script deliberately assigns no semantic quality label. If a
manual exclusion is justified, record it in the session export and rerun.

## Study A judgment and item files

`study_a_judgments.csv` contains one row per coded response per sample. Because
QC-filtered is a subset, an included judgment appears once under `all_paid` and
again under `qc_filtered`.

| Field | Meaning |
|---|---|
| `sample` | `all_paid` or `qc_filtered`. |
| `triplet_id` | Matched Role/Basic/Pro scientific cluster. |
| `contrast` | `role_basic` or `role_pro`. |
| `other_condition` | `basic` or `pro`, determined by `contrast`. |
| `code` | `role`, the relevant other condition, `tie`, or `unable`. |
| `contrast_score` | `−1` for Role, `+1` for the relevant other condition, and `0` for Tie/Unable. |
| `model`, `task_id`, `subcategory` | Frozen private item metadata. |
| `answer_raw` | Original opaque site value; retained for audit. |
| `confidence` | Optional locked Study A confidence from `1` to `5`. |
| `confidence_duration_ms` | Optional time recorded for the confidence stage. |
| `duration_ms` | Client-measured primary comparison time; descriptive only. |

`study_a_item_summaries.csv` adds:

| Field | Meaning |
|---|---|
| `*_count` | Number of available judgments in each four-way category. |
| `pair_pattern` | For the current two-rater design: `agreement_<label>`, `disagreement`, or `incomplete`; this is descriptive and not a majority. |
| `majority_label` | Archived-design compatibility field; current two-rater items use `not_used_two_rater_design`. |
| `complete_design_ratings` | The item has both ratings required by the current frozen manifest. |
| `contrast_preference` | Mean signed contrast score within that comparison item. The primary bootstrap resamples matched triplets within model. |

`study_a_raw_distribution.csv` is long-form. `grouping` is `overall`,
`contrast`, `model`, or `subcategory`; `count/n_judgments` defines
`proportion`.

`study_a_model_descriptives.csv` contains contrast-specific judgment counts,
two-rater agreement/disagreement counts, and signed contrast preference for each model. These are
descriptive, not fourteen separate powered hypothesis tests.

`study_a_confidence_timing.csv` reports formal-task confidence availability,
mean/median/IQR confidence, and primary-response time overall and separately
by contrast/outcome. It is built from `study_a_judgments.csv`, so
legacy pilot `A-IMC-1` confidence is never included. Confidence and timing do
not reweight the primary analysis.

`telemetry_events.csv` contains normalized optional events joined to study,
bundle, participant, and analysis-sample status.
`telemetry_session_summary.csv` contains event-type counts and minimum/maximum
elapsed time by session.

`print_participant_process_report.py` produces a separate private audit report
with actual randomized item order, choices, confidence, duration, scroll
milestones, runtime snapshots, focus/visibility changes, and rationale dwell
time. Its portable output is anonymized and excludes participant/session/study
IDs and tokens.

The production site does not collect mouse coordinates, cursor trajectories,
or continuous physical-input activity. A `runtime_snapshot` only shows that the
document was visible and its browser window was focused at a 30-second sampling
point. `scroll_milestone` records document-level 25/50/75/100 percent
thresholds; it does not prove that either response card or every sentence was
read. These fields remain descriptive and do not create an automatic
exclusion.

## Study B judgment and item files

`study_b_judgments.csv` contains:

| Field | Meaning |
|---|---|
| `code` | `same`, `related`, `different`, or `unable`. |
| `same_binary` | True only for `same`. |
| `no_same_binary` | True for `related` or `different`. |
| `item_type` | `unique` or overlap `control`. |
| `direction` | `sequential_to_interactive` or `interactive_to_sequential`. |
| `stratum` | `far`, `medium`, `near`, `just_above`, or `clear`. |
| `similarity` | Frozen maximum centroid similarity used for sampling. |
| `model`, `task_id` | Frozen private provenance. |

`study_b_item_majorities.csv` adds:

| Field | Meaning |
|---|---|
| `*_count` | Four-way judgment counts. |
| `majority_4way` | Strict four-way majority. Related and Different remain separate. |
| `majority_binary` | Strict majority after merging Related+Different as `no_same`. |
| `same_fraction` | Same judgments divided by all coded judgments, including Unable. |
| `no_same_fraction` | Related+Different divided by all coded judgments. |
| `primary_machine_supported` | Unique has a no-Same majority, or control has a Same majority. |
| `conservative_unable_majority` | Binary majority after adversarial Unable recoding. |
| `conservative_same_fraction` | Same fraction after Unable→Same on unique items and Unable→no-Same on controls. |
| `conservative_machine_supported` | Machine support under that recoding. |

## Study B summaries and tests

`study_b_group_summaries.csv` uses a `grouping` column to distinguish overall,
item-type, direction, stratum, interaction, and model tables. Blank dimension
columns mean “not grouped on this dimension.” It reports both item-majority
counts/rates and raw judgment-category rates.

`study_b_similarity_gradient.csv` has one row per sample and ordered stratum.
Rates retain Unable in the denominator.

`study_b_exact_and_permutation.csv` contains eight rows per sample:

| `test` | Endpoint |
|---|---|
| `fisher_and_exact_binary` | Two-sided Fisher test and exact fixed-margin absolute-difference randomization test on item Same-majority status. |
| `blocked_exact_binary` | Exact Same-majority randomization preserving control counts inside model×direction blocks. |
| `blocked_exact_same_fraction` | Exact blocked randomization of the continuous item Same-vote fraction. |
| `item_label_permutation` | Seeded Monte Carlo two-sided permutation test of control minus unique mean item Same-vote fraction, restricted within those blocks. |
| `conservative_*` | Fisher and exact/Monte Carlo blocked Same-fraction comparisons after adversarial Unable recoding; recoding is recomputed under every hypothetical assignment. |
| `similarity_trend` | Item-level Spearman correlation of frozen similarity with Same-vote fraction; outcomes are permuted within model×direction. |

Majority-based tests exclude items with fewer than two retained ratings or an
unresolved binary majority and report the excluded counts. They never equate
`insufficient_ratings` with an observed no-Same judgment.

`study_b_unable_sensitivity.csv` reports machine-support counts before and after
replacing Unable with Same on unique items and no-Same on controls.

## Agreement

`agreement.csv` reports pairwise exact agreement and Krippendorff’s nominal
alpha. Study A is separated by Role–Basic and Role–Pro contrast. Study B reports both the
four-way coding and a Same/no-Same/Unable collapse. Alpha includes Unable as an
explicit category; it is not silently dropped.

## Reproducibility metadata

`summary.json.configuration` records the random seed, iteration counts, and
SHA-256 hashes of:

- both exported input files;
- the Study A private item manifest and bundle key, plus the cross-validated
  participant-safe public bundle manifest;
- the Study B provenance, bundle manifest, and PASS validation record.

It also records `frozen_design`. For Study A, `matched_triplets`,
`items_by_contrast`, `unique_items`, `bundles`, `items_per_bundle`, and
`expected_judgments` are derived from the frozen private manifest and bundle
key; `ratings_per_item` is the uniformly observed frozen replication. The
current freeze therefore resolves to 42 triplets, 84 comparison items, 28
bundles, six items per bundle, two ratings per item, and 168 expected
judgments. The archived Role–Pro-only manifest remains supported.

`summary.json.warnings` must be reviewed before results are quoted. Partial
pilot exports intentionally warn when fewer than the manifest-derived Study A
item total or the legacy 56 Study B items are observed.
