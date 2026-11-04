# Multi-Agent Prompt Ablation Results

## Design

- Model: `deepseek-v4-flash` (`thinking=disabled`)
- 100 MODBench tasks
- Original five Pro personas, three rounds
- Identical shared round 1 across all three conditions
- Only the round-2/3 interaction instruction differs
- Primary metric: MPD with text-embedding-3-small

## Generation completeness

| Condition | Complete tasks | Records | Prompt tokens | Completion tokens |
|---|---:|---:|---:|---:|
| neutral | 100 | 1500 | 3,738,887 | 585,588 |
| agreement | 100 | 1500 | 4,127,846 | 668,087 |
| disagreement | 100 | 1500 | 4,526,110 | 771,872 |

## All-round MPD (paper-comparable set definition)

| Condition | Mean opinions | Mean MPD | Corrected rarefied CC | Corrected rarefied VS |
|---|---:|---:|---:|---:|
| neutral | 110.9 | 0.7165 | 87.77 | 33.34 |
| agreement | 133.8 | 0.6958 | 74.15 | 28.71 |
| disagreement | 159.6 | 0.7149 | 88.12 | 33.24 |

| Contrast (left − right) | Mean ΔMPD | 95% task-bootstrap CI | Wins / Ties / Losses | Wilcoxon p |
|---|---:|---:|---:|---:|
| agreement − neutral | -0.0207 | [-0.0258, -0.0157] | 18 / 0 / 82 | 1.382e-11 |
| disagreement − neutral | -0.0016 | [-0.0061, +0.0029] | 47 / 0 / 53 | 0.2697 |
| disagreement − agreement | +0.0190 | [+0.0137, +0.0245] | 75 / 0 / 25 | 1.498e-09 |

## Rounds 2–3 MPD (post-intervention diagnostic)

| Condition | Mean opinions | Mean MPD | Corrected rarefied CC | Corrected rarefied VS |
|---|---:|---:|---:|---:|
| neutral | 71.7 | 0.7230 | 59.83 | 27.40 |
| agreement | 94.6 | 0.6898 | 47.57 | 22.20 |
| disagreement | 120.4 | 0.7160 | 59.26 | 26.65 |

| Contrast (left − right) | Mean ΔMPD | 95% task-bootstrap CI | Wins / Ties / Losses | Wilcoxon p |
|---|---:|---:|---:|---:|
| agreement − neutral | -0.0332 | [-0.0401, -0.0263] | 11 / 0 / 89 | 1.48e-13 |
| disagreement − neutral | -0.0070 | [-0.0129, -0.0010] | 39 / 0 / 61 | 0.007473 |
| disagreement − agreement | +0.0262 | [+0.0195, +0.0332] | 76 / 0 / 24 | 1.369e-10 |

## Interpretation guardrail

This single-model, 100-task ablation tests sensitivity to interaction instructions. It does not validate MPD against human-perceived opinion diversity and does not establish a model-general effect.
