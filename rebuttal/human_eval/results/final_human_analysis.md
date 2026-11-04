# Final human-annotation analysis: Main 25 + Pilot 3

## Bottom line

The final selected scientific sample contains **28 completed sessions from 27
unique annotators**, covering all 28 frozen bundles and all 168 judgments.
Role was selected more often than Basic and Pro, but both prespecified
triplet-cluster confidence intervals include zero. The supported conclusion is
therefore:

> No stable human-perceived Basic or Pro advantage over Role was detected for
> viewpoint breadth within a matched complete response. This is not evidence
> that the conditions are equivalent, and it is not a universal validation of
> MPD.

## Sample and quality status

- Main: 25 sessions; frozen scientific pilot: 3 sessions.
- Unique annotators: 27. One annotator completed one pilot and one main bundle.
- Coverage: 28/28 bundles; 42 Role–Basic and 42 Role–Pro items; every item has
  exactly two ratings.
- Judgments: 84 Role–Basic + 84 Role–Pro = 168.
- 28/28 passed comprehension, supplied six formal choices, six task-confidence
  ratings, one non-empty rationale, and exactly one final commit.
- Main sessions have 13/13 locked rows. Pilot sessions have 15/15 because the
  frozen pilot protocol contains a legacy audit-only check and its confidence;
  these two rows do not enter the scientific analysis.
- The selected sessions have 28 unique session identifiers and 28 unique
  bundles. Participant-cluster inference groups the two sessions from the
  repeated annotator under the same PID.
- The source has no joined `paid` field. Thus the frozen analyzer's
  `all_paid` label uses its documented completed-session fallback; this report
  makes no payment decision.

## Primary counts

| Contrast | N | Role | Other | Tie | Unable |
|---|---:|---:|---:|---:|---:|
| Role–Basic | 84 | 32 (38.10%) | 24 Basic (28.57%) | 27 (32.14%) | 1 (1.19%) |
| Role–Pro | 84 | 37 (44.05%) | 23 Pro (27.38%) | 24 (28.57%) | 0 |
| Combined descriptive | 168 | 69 (41.07%) | 47 non-Role (27.98%) | 51 (30.36%) | 1 (0.60%) |

The combined row is descriptive only: “other” denotes Basic in one contrast
and Pro in the other.

Among directional judgments, Role received 32/56 (57.14%) in Role–Basic and
37/60 (61.67%) in Role–Pro.

## Clustered inference

The frozen estimand is

\[
D=P(\text{other broader})-P(\text{Role broader}),
\]

with Tie and Unable coded as zero. The primary CI resamples the 42 matched
triplets within model and retains all ratings attached to a sampled comparison.

| Contrast | Primary D [95% triplet-cluster CI] | PID-cluster sensitivity |
|---|---:|---:|
| Role–Basic | −0.095 [−0.286, 0.095] | −0.095 [−0.264, 0.074] |
| Role–Pro | −0.167 [−0.357, 0.024] | −0.167 [−0.357, 0.011] |

Both primary intervals include zero. The data do not show a stable Basic or
Pro advantage over Role, but they also do not establish equivalence. In
particular, the earlier 22-person Role–Pro result must not be described as a
final “consistent Role advantage”: after completing the design, its primary
CI includes zero.

## Item-level agreement

All 84 comparison items have two ratings; no item majority is manufactured.

| Contrast | Exact pair agreement | Nominal Krippendorff α | Agreement patterns | Disagreement |
|---|---:|---:|---|---:|
| Role–Basic | 52.38% | 0.298 | 10 Role / 6 Basic / 6 Tie | 20/42 |
| Role–Pro | 42.86% | 0.131 | 8 Role / 7 Pro / 3 Tie | 24/42 |

Agreement is modest, especially for Role–Pro. This supports treating item and
annotator clustering explicitly and cautions against presenting the task as if
there were one unambiguous human label per response pair.

## A/B position audit

The non-Role response appeared on A and B exactly 84 times each; within each
contrast the balance was 42/42.

| Contrast | Other chosen when on A | Other chosen when on B | Complete-item paired McNemar p |
|---|---:|---:|---:|
| Overall | 25/56 directional (44.64%) | 22/60 (36.67%) | 0.664 |
| Role–Basic | 14/28 (50.00%) | 10/28 (35.71%) | 0.388 |
| Role–Pro | 11/28 (39.29%) | 12/32 (37.50%) | 1.000 |

The Role–Basic side difference is a review signal, but the matched-item
diagnostic does not provide clear evidence of a position effect. Position does
not alter the primary conclusion.

## Repeated-annotator audit

One annotator completed a pilot bundle and a main bundle, producing 12
judgments. The participant-cluster bootstrap groups all 12 under one PID.
There was one repeated participant–item exposure, `SA-C045`; both judgments
were Tie.

- All labels: 168 judgments.
- First-exposure deduplication: 167 judgments, removing the later Tie.
- Role–Basic raw D changes only from −0.0952 to −0.0964.
- Role–Pro counts and D are unchanged.
- The frozen matched-triplet estimates remain −0.095 and −0.167.

Dropping either entire six-judgment session also preserves negative point
estimates for both contrasts. One session-drop/PID-cluster Role–Pro interval
excludes zero, but this does not persist under the primary bootstrap or the
alternative session treatment and is not the headline result.

## Length, confidence, and time

All 42 items in each contrast satisfy the frozen ≤1.5× length diagnostic.

| Diagnostic | Role–Basic | Role–Pro |
|---|---:|---:|
| Longer response selected among directional | 58.9% | 62.7% |
| Item-level length–preference correlation | 0.199 | 0.174 |
| Mean confidence | 3.940 | 3.893 |
| Median task time | 89.8 s | 101.7 s |

By selected outcome, mean confidence was 4.04 for Role, 3.75 for Basic, 3.74
for Pro, and 3.88 for Tie. Mean task times were 102.8, 111.6, 126.5, and 124.3
seconds, respectively. These are descriptive associations; length,
confidence, and time neither reweight nor exclude judgments.

## Model heterogeneity

The descriptive model-level D values are not uniform:

- Role–Basic ranges from −0.417 for Kimi to +0.500 for Gemini.
- Role–Pro ranges from −0.333 for Kimi/Llama/Qwen to +0.083 for
  Gemini/GPT-5.4-mini.

Each model contributes only 12 judgments per contrast, so these figures are
heterogeneity diagnostics, not model-specific inference. They reinforce the
clustered overall interpretation rather than a universal per-model claim.

## Rationale content

All 28 rationales are non-empty and content-grounded; they range from 13 to 80
words. Ten accompany Tie judgments, eight Role, five Basic, and five Pro.
Common patterns are:

- identifying a concrete additional argument, recommendation, causal factor,
  or context in one response;
- explaining that both answers cover substantively overlapping points;
- separating broad viewpoints from examples, persona details, or narrative
  elaboration.

One rationale states that the annotator reconsidered the already locked answer.
It remains a transparent review signal but is concrete and does not satisfy the
frozen blank/gibberish/unrelated exclusion rule.

## Change from the earlier 22-main snapshot

| Contrast | Earlier 22 main | Final 28 sessions | Direction |
|---|---|---|---|
| Role–Basic | 27 Role / 19 Basic / 19 Tie / 1 Unable | 32 / 24 / 27 / 1 | Role remains most frequent; raw gap attenuates |
| Role–Pro | 34 Role / 16 Pro / 16 Tie | 37 / 23 / 24 | Role remains most frequent; raw gap attenuates materially |

The final analysis supersedes every preliminary 22-person human count,
alignment statistic, and CI.

## Relation to the existing measurement decomposition

The human task evaluates **holistic breadth within one matched complete
response**. MPD pools atomic opinions over a **set of responses**. The existing
read-only decomposition remains classified as mixed:

- Pro−Role pooled atomic MPD: +0.0133;
- equal-response-pair atomic distance: +0.0111;
- response-centroid diversity: −0.0279;
- within-response atomic dispersion: +0.0337.

Thus the final human result does not reproduce the small Pro-over-Role atomic
MPD ordering at the holistic-response level. This is not by itself evidence
that MPD “failed”: the measures use different aggregation units, and the
decomposition shows that cross-atomic distance mixes within-response
dispersion with between-response separation. The safe interpretation is that
MPD remains an operational atomic semantic-dispersion proxy, not a validated
measure of human-perceived response breadth.

## Claim boundary

The final study can support:

- no stable human-perceived Basic or Pro breadth advantage over Role was
  detected in matched complete responses;
- targeted human grounding consistent with diminishing returns from increasing
  persona depth;
- the need to distinguish holistic response breadth from atomic set-level
  dispersion.

It cannot support:

- equivalence among Role, Basic, and Pro;
- a statistically established Role superiority claim;
- universal validation or invalidation of MPD;
- population-distribution fidelity, viewpoint-proportion calibration, or
  overall response quality.

