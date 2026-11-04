# Diversity-Coverage Prompt: 100-Task Results

## Frozen comparison

- Control: submitted neutral Multi-Agent interaction instruction.
- Intervention: a repeated round-2/3 instruction to broaden genuine viewpoint coverage.
- Shared round 1, model, questions, five Pro personas, agent order, rounds, generation budget, temperature, extractor, and embedder are fixed.
- Primary analysis: task-paired rounds-2/3 MPD.
- Secondary analysis: correctly rarefied Cluster Count and Vendi Score.

## All rounds

| Condition | Mean opinions | MPD | Rarefied CC | Rarefied VS |
|---|---:|---:|---:|---:|
| neutral | 110.9 | 0.7165 | 86.00 | 32.99 |
| coverage | 118.8 | 0.6886 | 87.62 | 30.43 |

| Metric, coverage − neutral | Mean Δ | 95% task-bootstrap CI | Wins / Ties / Losses | Wilcoxon p |
|---|---:|---:|---:|---:|
| mpd | -0.0279 | [-0.0328, -0.0231] | 9 / 0 / 91 | 1.024e-14 |
| cc_rarefied | +1.6210 | [+0.2843, +2.9753] | 58 / 0 / 42 | 0.03388 |
| vs_rarefied | -2.5546 | [-3.1635, -1.9437] | 23 / 0 / 77 | 8.892e-11 |
| n | +7.9500 | [+4.1898, +11.7300] | 65 / 1 / 34 | 0.0001914 |

Neutral–coverage beta separation:

- β-VS: 8.874
- β-VS above shuffled labels: -0.688
- Neutral unique-cluster ratio: 61.4%
- Coverage unique-cluster ratio: 60.1%
- Mean UCR above shuffled labels: -6.0%

## Rounds 2–3 (intervention effect)

| Condition | Mean opinions | MPD | Rarefied CC | Rarefied VS |
|---|---:|---:|---:|---:|
| neutral | 71.7 | 0.7230 | 57.90 | 26.86 |
| coverage | 79.6 | 0.6797 | 57.71 | 23.54 |

| Metric, coverage − neutral | Mean Δ | 95% task-bootstrap CI | Wins / Ties / Losses | Wilcoxon p |
|---|---:|---:|---:|---:|
| mpd | -0.0434 | [-0.0504, -0.0364] | 8 / 0 / 92 | 3.257e-15 |
| cc_rarefied | -0.1877 | [-1.3207, +0.9757] | 45 / 0 / 55 | 0.6024 |
| vs_rarefied | -3.3208 | [-3.9637, -2.6794] | 13 / 0 / 87 | 1.965e-13 |
| n | +7.9500 | [+4.1698, +11.7400] | 65 / 1 / 34 | 0.0001914 |

Neutral–coverage beta separation:

- β-VS: 12.667
- β-VS above shuffled labels: +1.426
- Neutral unique-cluster ratio: 91.3%
- Coverage unique-cluster ratio: 89.4%
- Mean UCR above shuffled labels: +8.0%

## Frozen success criterion

- Primary MPD criterion: **FAIL**
- Secondary direction criterion: **FAIL**
- Overall: **FAIL**

## Audit and interpretation guardrail

- Artifact audit: **PASS** (0 errors)
- New generation calls: 1,000
- Estimated new generation cost: $0.684
- Opinion count is descriptive and cannot by itself establish diversity.
- Beta diversity measures separation between conditions, not which condition is more internally diverse.
- This single-model prompt optimization does not validate MPD against human-perceived diversity and does not establish a model-general effect.
