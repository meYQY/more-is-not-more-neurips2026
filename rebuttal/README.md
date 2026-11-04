# Rebuttal-period analyses (paper Appendices J, K, L)

This folder contains the code, small result files, and documentation for the
three analyses added to the camera-ready paper after the review period. The
rest of the repository is unchanged.

| Folder | Paper section | Paper table | Offline check from released files |
|---|---|---|---|
| `human_eval/` | Appendix J, *Human Evaluation of Perceived Viewpoint Diversity* | "Blinded human evaluation of viewpoint breadth" (`tab:human-eval`) | `reproduce_table.py` |
| `robustness/` | Appendix K, *Robustness Within the 100-Question Benchmark* | "Stability of the Single-Call persona-depth results within the 100-question benchmark" (`tab:robustness`) | `robustness_analysis.py`, `run_robustness.py` |
| `ma_prompt_sensitivity/` | Appendix L, *Multi-Agent Prompt Sensitivity* | "Multi-Agent prompt sensitivity" (`tab:ma-prompt`) | `verify_from_metrics.py` |

## Setup

```bash
pip install -e .        # from the repository root; Python >= 3.10 (tested with 3.12)
```

All commands below are run from the repository root. Nothing in the offline
checks calls an API. Only the generation, extraction, and embedding steps of
the Multi-Agent experiment need API keys, read from `.env` in the repository
root (see the top-level README):

- `DEEPSEEK_API_KEY`: generation and opinion extraction (`deepseek-v4-flash`);
- `OPENROUTER_API_KEY`: `text-embedding-3-small` embeddings.

The scripts locate the repository root relative to their own location. Set
`MODBENCH_ROOT` to override it. Local absolute paths in the original scripts
were replaced by these relative defaults; the scripts are otherwise as they
were run, apart from the release notes added to the two copied READMEs in
`human_eval/` and the three helper files written for this release
(`human_eval/reproduce_table.py`, `ma_prompt_sensitivity/verify_from_metrics.py`,
`ma_prompt_sensitivity/prompts.md`).

## 1. Human evaluation (`human_eval/`, Appendix J)

**Design.** 42 matched Single-Call (model, question, persona) triplets, each
with the Role, Basic, and Pro responses from the same model and persona to the
same question, selected with a fixed seed from metadata hashes only (6 per
model, 42 distinct questions, all 10 subcategories, all 20 personas). Before
selection, every response in a triplet had to be at most 325 words and within
30 words of the other two, and all three had to pass a primary-English screen.
Only 1,086 of 13,837 technically valid triplets passed the length filter
(1,083 after the English screen), so the 42 items over-represent shorter
responses. Each triplet gives one Role-Basic and one Role-Pro comparison (84
items). Items were arranged into 28 bundles of six (3 Role-Basic + 3
Role-Pro); every item appears in two bundles, once with the Role response as
A and once as B, giving 168 judgments.

**Analyzed sample.** 28 sessions (3 pilot sessions run under the same protocol
+ 25 main-study sessions), one per bundle, from 27 unique annotators. One
annotator completed one pilot and one main session (12 judgments) and saw one
item (`SA-C045`) twice; both ratings were "About the same".

| File | Content |
|---|---|
| `data/study_a_judgments_deidentified.csv` | One row per judgment (168), de-identified (columns below). |
| `study_a/item_manifest.csv` | One row per comparison item (84): triplet, contrast, model, question (`task_id`), subcategory, persona (`persona_id` from `configs/personas.yaml`, fictional persona name), displayed word counts, length ratios, and the two bundle placements with the Role side. |
| `study_a/AUDIT.md` | Audit written by the item-selection generator: eligibility counts, the minimal-change repair after one non-English response, category allocation, bundle balance. |
| `study_a/generate_study_a.py`, `study_a/validate_generated.py`, `study_a/README.md` | Outcome-blind item selection and bundle assignment, and its validator (reference code; see below). |
| `instructions_and_consent.md` | Plain-text consent page, instructions, understanding check, practice items, task wording, confidence scale, and explanation prompts as shown to annotators (researcher email redacted). |
| `analysis/` | The frozen analyzer (`analyze_annotations.py`), its README, data dictionary, process-report printer, and tests (reference code; see below). |
| `results/final_human_analysis.md` | Full write-up of the final 28-session analysis. |
| `results/final_human_summary.json`, `.csv` | Counts, D with triplet-cluster and participant-cluster CIs, agreement, first-exposure sensitivity, position diagnostics. |
| `results/agreement.csv` | Pairwise agreement and nominal Krippendorff's alpha per contrast (Study B rows are empty). |
| `results/position_audit.csv` | A/B position diagnostic with paired McNemar p-values. |
| `reproduce_table.py` | Recomputes the paper table from the two CSVs above. |

Columns of `study_a_judgments_deidentified.csv`:

| Column | Meaning |
|---|---|
| `judgment_id` | `J001`-`J168`, ordered by session and within-session display order. |
| `session` | `session_01`-`session_28`; numbered in bundle order (`session_01` rated `SA-B01`, etc.). |
| `annotator` | `annotator_01`-`annotator_27`; numbered by first session. |
| `phase` | `pilot` (bundles `SA-B03`, `SA-B05`, `SA-B21`) or `main`. |
| `bundle_id`, `item_id`, `triplet_id` | Design identifiers (see `study_a/item_manifest.csv`). |
| `contrast`, `other_condition` | `role_basic`/`basic` or `role_pro`/`pro`. |
| `model`, `task_id`, `persona_id` | Generating model, question ID in `data/dataset/tasks.csv`, persona ID in `configs/personas.yaml`. |
| `role_position` | Side on which the Role response was shown (`A` or `B`). |
| `selected_side` | `A`, `B`, `about_same`, or `unable`. |
| `choice` | Decoded choice: `role`, `basic`, `pro`, `tie`, or `unable`. |
| `contrast_score` | +1 if the non-Role response was judged broader, -1 if Role, 0 for tie/unable. |
| `confidence` | 1-5 self-rated confidence. |

The CSV was derived from the private analysis export by replacing Prolific
participant IDs and session IDs with the indices above and keeping only the
listed columns. It contains no Prolific participant, session, or study IDs, no
timestamps, no per-item timing or browser telemetry, and no free-text
explanations. A/B placement was cross-checked against the opaque response
answers in the export.

**Run (offline, standard library only):**

```bash
python rebuttal/human_eval/reproduce_table.py
```

This prints the paper table (Role-Basic: 32/24/27/1, D = -0.095 [-0.286,
0.095], 52.4% agreement, alpha = 0.298; Role-Pro: 37/23/24/0, D = -0.167
[-0.357, 0.024], 42.9%, alpha = 0.131), the two-sided sign test on directional
Role-Basic judgments (p = 0.350), the paired position McNemar test (p = 0.388),
the share of directional judgments choosing the longer response (58.9%), and
the first-exposure sensitivity (Role-Basic D = -0.0964). It exits non-zero if
any value differs from `results/final_human_summary.json`. The
participant-cluster sensitivity CI is printed but not compared: the bootstrap
draws clusters in sorted label order, and the re-indexed annotator labels give
a different (equally valid) random draw from the stored run. It is not
reported in the paper.

**Not runnable from the release.** `analysis/analyze_annotations.py` and its
tests read the raw study database exports (Prolific identifiers, rationales,
telemetry) and the private item crosswalks, none of which are released.
`study_a/generate_study_a.py` needs the full local tree of Single-Call raw
outputs for all seven models (pass `--modbench-root`), which is not released.

## 2. Robustness within the 100-question benchmark (`robustness/`, Appendix K)

Inputs are files already in the repository: `data/dataset/tasks.csv` and the
per-condition metric JSONs in `results/metrics*/openai-3-small/` (per-question
MPD for the seven models). No API calls.

```bash
python rebuttal/robustness/robustness_analysis.py   # about 10 s
python rebuttal/robustness/run_robustness.py        # about 25 s
```

Both scripts write next to themselves, overwriting the shipped outputs.
Rerunning them reproduced every shipped file byte for byte, except that the
key order inside the `diagnostics` block of `robustness_summary.json` can
change (Python set iteration); the values are identical.

| Script | Outputs | Used in the paper for |
|---|---|---|
| `robustness_analysis.py` (seed 20260724; 20,000 task-bootstrap resamples; 10,000 random subsets) | `robustness_results.json`, `robustness_results.md` | The table: per-model Single-Call None->Role and Role->Pro gains, "initial - extra", and the share of random 50-question subsets with a positive None->Role gain and with None->Role > Role->Pro. |
| `run_robustness.py` (seed 1529; 10,000 bootstrap resamples; 5,000 random subsets per size) | `robustness_summary.json`, `subsample_stability.csv`, `leave_one_category_out.csv`, `key_effects_by_model.csv` | The pooled comparisons in the text: None->Role (`AllArch_None_to_Role`, +0.0665), initial gain exceeding later depth increases (`AllArch_InitialMinusExtraDepth`, +0.0566), Multi-Turn and Multi-Agent over Single-Call under matched five-persona Pro (`Matched5_SC_to_MT_Pro`, +0.0532; `Matched5_SC_to_MA_Pro`, +0.0416); random 25-question subsets and leave-one-subcategory-out. |

Both scripts also compute contrasts that the paper does not report (low-cost
conditions, other depth steps, persona x architecture difference-in-differences
`DiD_*`). `run_robustness.py` additionally writes `category_effects.csv` and
`did_results_by_model.csv`, which are not shipped.

## 3. Multi-Agent prompt sensitivity (`ma_prompt_sensitivity/`, Appendix L)

**Design.** DeepSeek-V4-Flash (`deepseek-v4-flash`, thinking disabled,
temperature 0.7), all 100 questions, Pro personas 1-5, five agents, three
rounds. Round 1 is generated once per question and copied into every
condition; only the Round 2-3 instruction differs (Neutral, Agreement,
Disagreement, Coverage; exact text in `prompts.md`, frozen in
`experiment_config.yaml` and `coverage_config.yaml`). Extraction uses the
repository's standard prompt with `deepseek-v4-flash` at temperature 0;
embeddings use `text-embedding-3-small`. Round-1 extractions are copied from
Neutral into the other conditions so that extractor jitter cannot enter the
contrast.

**Offline check (no API, no embeddings needed):**

```bash
python rebuttal/ma_prompt_sensitivity/verify_from_metrics.py
```

This recomputes the paper table from the per-question values in
`analysis/four_condition_alpha_100.json`, and the paired MPD contrasts and
task-bootstrap CIs from `metrics/metrics_100.json` and
`metrics/coverage_metrics_100.json` with the same routines and seeds as the
pipeline, and checks them against the stored result files.

**Full pipeline (needs API keys and regenerates about 627 MB of artifacts):**

| Step | Command (from `rebuttal/ma_prompt_sensitivity/`) | Needs | Writes |
|---|---|---|---|
| 1 | `python run_generation.py` (`--dry-run` to validate the design without calls) | `DEEPSEEK_API_KEY` | `raw/_shared_round1/`, `raw/{neutral,agreement,disagreement}/`, `generation_summary*.json` (3,500 calls) |
| 2 | `python run_pipeline.py` | `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | `extracted/`, `embeddings/`, `embeddings_post/`, `metrics/metrics*.json`, `analysis/paired_results*.json`, `analysis/report*.md` |
| 3 | `python run_coverage_generation.py` | `DEEPSEEK_API_KEY`; step 1's shared Round 1 | `raw/coverage/` (1,000 calls) |
| 4 | `python run_coverage_pipeline.py` | `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | coverage extraction/embeddings, `metrics/coverage_metrics_100.json`, `analysis/coverage_*` |
| 5 | `python analyze_four_conditions.py` | embeddings from steps 2 and 4 (offline) | `analysis/four_condition_alpha_100.json`, `analysis/four_condition_alpha_report_100.md` |
| 6 | `python analyze_prompt_unit_of_analysis.py` | raw, extracted, embeddings (offline) | `analysis/prompt_unit_of_analysis_*` |
| 7 | `python analyze_alpha_beta.py` | embeddings (offline) | `analysis/alpha_beta_100.json` (beta diversity; not reported in the paper, not shipped) |

Generation is stochastic, so a rerun will not reproduce the stored responses
or metrics exactly. The audits record about USD 2.19 of generation for the
three-condition run and USD 0.68 for Coverage.

**Shipped results and where the paper uses them.**

| File | Content / paper use |
|---|---|
| `analysis/four_condition_alpha_report_100.md`, `analysis/four_condition_alpha_100.json` | The paper table ("Rounds 2-3": opinions, MPD, CC and VS rarefied jointly across the four conditions), plus all-round values. |
| `analysis/report_100.md`, `analysis/paired_results_100.json` | Paired MPD contrasts quoted in the text: Agreement - Neutral -0.0332 [-0.0401, -0.0263]; Disagreement - Agreement +0.0262 [+0.0195, +0.0332]; Disagreement - Neutral -0.0070 [-0.0129, -0.0010] (Rounds 2-3) and -0.0016 [-0.0061, +0.0029] (all rounds). |
| `analysis/coverage_report_100.md`, `analysis/coverage_paired_results_100.json` | Coverage - Neutral: MPD -0.0434 [-0.0504, -0.0364], rarefied CC and VS, beta separation, and the frozen success criterion. |
| `analysis/audit_100.md`, `analysis/coverage_audit_100.json` | Artifact audits (record counts, finish reasons, shared Round 1, token counts). |
| `analysis/prompt_unit_of_analysis_report_100.md` | Post-hoc diagnostic (response-, round-, and agent-centroid MPD); not reported in the paper. |
| `metrics/metrics_100.json`, `metrics/coverage_metrics_100.json` | Per-question metrics for the three-condition and Neutral-vs-Coverage runs. |
| `data/selected_tasks_100.csv` | The 100 questions (identical to `data/dataset/tasks.csv`, sorted by ID). |
| `generation_summary*.json`, `coverage_generation_summary_100.json` | Completed tasks, record counts, and token usage per condition. |

The CC/VS values in `report_100.md` and `coverage_report_100.md` are rarefied
within three or two conditions, respectively, and therefore differ slightly
from the table, which rarefies jointly across all four conditions; MPD and
opinion counts are not rarefied and agree.

## Not included

- Multi-Agent raw responses, extracted opinions, and embeddings (about 627 MB);
  regenerate with the pipeline above.
- Multi-Agent 30-question checkpoint, beta-diversity and coverage-granularity
  outputs, generation manifests, logs, and internal working notes.
- Robustness analyses not used in the paper (external construct-validity
  checks, corrected-rarefaction recomputation, difference-in-differences
  tables).
- Human evaluation: raw study exports, Prolific identifiers, timestamps,
  telemetry, per-item timing, free-text explanations, the private item
  crosswalks, the participant-facing bundles with full response texts, the
  study web application, and recruitment materials.
