# Prompt Ablation: Unit-of-Analysis Diagnostic

Post-hoc diagnostic only. It uses the existing extracted atomic opinions
and `text-embedding-3-small` arrays; it made no API calls and does not
replace the frozen primary opinion-unit MPD analysis.

## Mapping audit

- Status: **passed**.
- Checked 300 task-condition pairs, 4,500 extracted response rows, 40,433 full embedding rows, and 28,670 R2/R3 embedding rows.
- All 300 post arrays were exact row-preserving R2/R3 subsets of their full arrays.
- R2/R3 response groups with zero / one extracted opinion: 14 / 0.

## Condition means

| Metric | Neutral | Agreement | Disagreement |
|---|---:|---:|---:|
| Atomic-opinion MPD (R2–3) | 0.7230 | 0.6898 | 0.7160 |
| Response-centroid MPD (R2–3) | 0.3202 | 0.2180 | 0.2268 |
| Final-round response-centroid MPD | 0.3326 | 0.2162 | 0.2217 |
| Agent-centroid MPD | 0.1966 | 0.1264 | 0.1296 |
| Within-response MPD | 0.6918 | 0.6758 | 0.6912 |
| Extracted opinions per R2/R3 response | 7.17 | 9.46 | 12.04 |

## Paired contrasts

### Disagreement − Neutral

| Metric | Paired tasks | Left | Right | Mean Δ | 95% task-bootstrap CI | Wins / Ties / Losses |
|---|---:|---:|---:|---:|---:|---:|
| Atomic-opinion MPD (R2–3) | 100 | 0.7160 | 0.7230 | -0.0070 | [-0.0129, -0.0011] | 39 / 0 / 61 |
| Response-centroid MPD (R2–3) | 100 | 0.2268 | 0.3202 | -0.0934 | [-0.1034, -0.0836] | 2 / 0 / 98 |
| Final-round response-centroid MPD | 99 | 0.2221 | 0.3326 | -0.1106 | [-0.1235, -0.0975] | 5 / 0 / 94 |
| Agent-centroid MPD | 99 | 0.1292 | 0.1966 | -0.0674 | [-0.0750, -0.0599] | 2 / 0 / 97 |
| Within-response MPD | 100 | 0.6912 | 0.6918 | -0.0006 | [-0.0074, +0.0062] | 47 / 0 / 53 |
| Extracted opinions per response | 100 | 12.04 | 7.17 | +4.88 | [+4.37, +5.39] | 98 / 0 / 2 |

### Agreement − Neutral

| Metric | Paired tasks | Left | Right | Mean Δ | 95% task-bootstrap CI | Wins / Ties / Losses |
|---|---:|---:|---:|---:|---:|---:|
| Atomic-opinion MPD (R2–3) | 100 | 0.6898 | 0.7230 | -0.0332 | [-0.0402, -0.0261] | 11 / 0 / 89 |
| Response-centroid MPD (R2–3) | 100 | 0.2180 | 0.3202 | -0.1022 | [-0.1122, -0.0924] | 1 / 0 / 99 |
| Final-round response-centroid MPD | 99 | 0.2165 | 0.3326 | -0.1161 | [-0.1309, -0.1014] | 5 / 0 / 94 |
| Agent-centroid MPD | 99 | 0.1264 | 0.1966 | -0.0703 | [-0.0780, -0.0626] | 2 / 0 / 97 |
| Within-response MPD | 100 | 0.6758 | 0.6918 | -0.0160 | [-0.0241, -0.0079] | 32 / 0 / 68 |
| Extracted opinions per response | 100 | 9.46 | 7.17 | +2.29 | [+1.83, +2.76] | 87 / 0 / 13 |

### Disagreement − Agreement

| Metric | Paired tasks | Left | Right | Mean Δ | 95% task-bootstrap CI | Wins / Ties / Losses |
|---|---:|---:|---:|---:|---:|---:|
| Atomic-opinion MPD (R2–3) | 100 | 0.7160 | 0.6898 | +0.0262 | [+0.0195, +0.0330] | 76 / 0 / 24 |
| Response-centroid MPD (R2–3) | 100 | 0.2268 | 0.2180 | +0.0088 | [-0.0008, +0.0184] | 62 / 0 / 38 |
| Final-round response-centroid MPD | 100 | 0.2217 | 0.2162 | +0.0055 | [-0.0087, +0.0193] | 54 / 0 / 46 |
| Agent-centroid MPD | 100 | 0.1296 | 0.1264 | +0.0032 | [-0.0036, +0.0101] | 60 / 0 / 40 |
| Within-response MPD | 100 | 0.6912 | 0.6758 | +0.0154 | [+0.0076, +0.0235] | 59 / 0 / 41 |
| Extracted opinions per response | 100 | 12.04 | 9.46 | +2.58 | [+2.00, +3.16] | 84 / 1 / 15 |

## Focus tasks: Disagreement − Neutral

| Task | Atomic MPD | Response centroids | Final-round centroids | Agent centroids | Within-response MPD | Opinions/response |
|---|---:|---:|---:|---:|---:|---:|
| MOD-085 | +0.0181 | -0.1145 | -0.1758 | -0.0851 | +0.0074 | +9.00 |
| MOD-091 | -0.0349 | -0.1191 | -0.1131 | -0.0978 | -0.0348 | +4.00 |
| MOD-023 | -0.0456 | -0.1316 | -0.0616 | -0.0925 | -0.0490 | +3.40 |
| MOD-064 | -0.0359 | -0.0719 | -0.1012 | -0.0527 | -0.0140 | +0.90 |
| MOD-048 | +0.0114 | -0.1737 | -0.2178 | -0.1038 | +0.0074 | +9.00 |

The preceding metric-blind model-agent inspection judged all five
focus tasks to contain substantive additions under Disagreement
relative to Neutral. Atomic MPD was positive on 2/5 of these tasks;
response-, final-round-, and agent-centroid MPD were each positive
on 0/5. Thus centroiding does not reconcile the automated scores
with that qualitative read; it makes the mismatch larger.

## What this diagnostic indicates

- Disagreement produced 4.88 more extracted opinions per R2/R3
  response than Neutral, but their mean within-response dispersion
  was nearly identical (Δ = -0.0006).
- Giving every response equal weight did not recover a positive
  Disagreement effect. Response-centroid MPD was much lower than
  Neutral (Δ = -0.0934), as were final-round and agent centroids.
- Disagreement exceeded Agreement on atomic MPD, but not reliably on
  response-, final-round-, or agent-centroid MPD. Its reliable
  advantage over Agreement appears inside responses: agents surface
  counterarguments, while different agents often cover a similar
  multi-sided argument set.
- Agreement remained below Neutral across every between-unit view.
  The visible Agreement convergence is therefore not an artifact of
  weighting verbose responses by their number of extracted claims.

## Interpretation boundary

- Response and agent centroids remove or reduce the primary analysis's
  implicit extra weight for responses that yield more atomic opinions.
- The observed ordering does not support the simple hypothesis that
  verbosity/supporting-claim weighting alone suppressed a genuinely
  higher Disagreement score. It instead exposes a distinction between
  within-response viewpoint coverage and between-agent stance
  separation.
- These results are post-hoc, embedding-based diagnostics. They are not
  human verification and should not be presented as a new primary
  endpoint without an explicit scope statement.
- For rebuttal, this is useful as an internal guardrail. It supports
  the Agreement-convergence claim, but it is not positive evidence
  that MPD captures human-perceived diversity, and the centroid
  endpoints are not worth foregrounding as a rebuttal result.
