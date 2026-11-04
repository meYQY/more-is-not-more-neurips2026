# Rebuttal robustness analysis

All confidence intervals are nonparametric task-bootstrap percentile intervals.

## Persona-depth stability

| Model | N | None→Role ΔMPD [95% CI] | Role→Pro ΔMPD [95% CI] | Initial−extra [95% CI] | P(initial>0), n=50 | P(initial>extra), n=50 | LOO-category directions |
|---|---:|---:|---:|---:|---:|---:|---|
| DeepSeek | 100 | 0.0681 [0.0567, 0.0802] | 0.0361 [0.0271, 0.0445] | 0.0320 [0.0154, 0.0494] | 100.0% | 100.0% | initial>0: True; initial>extra: True |
| GPT-mini | 100 | 0.0328 [0.0250, 0.0411] | -0.0135 [-0.0202, -0.0075] | 0.0463 [0.0342, 0.0599] | 100.0% | 100.0% | initial>0: True; initial>extra: True |
| Gemini | 100 | 0.0844 [0.0767, 0.0924] | 0.0090 [0.0036, 0.0145] | 0.0754 [0.0653, 0.0855] | 100.0% | 100.0% | initial>0: True; initial>extra: True |
| Kimi | 99 | 0.0843 [0.0760, 0.0928] | -0.0027 [-0.0083, 0.0029] | 0.0869 [0.0761, 0.0981] | 100.0% | 100.0% | initial>0: True; initial>extra: True |
| Llama | 100 | 0.0411 [0.0321, 0.0509] | 0.0393 [0.0306, 0.0477] | 0.0018 [-0.0107, 0.0152] | 100.0% | 60.7% | initial>0: True; initial>extra: False |
| Grok | 100 | 0.0278 [0.0211, 0.0353] | 0.0114 [0.0054, 0.0167] | 0.0165 [0.0059, 0.0280] | 100.0% | 100.0% | initial>0: True; initial>extra: True |
| Qwen | 97 | 0.0747 [0.0650, 0.0843] | 0.0140 [0.0067, 0.0212] | 0.0607 [0.0467, 0.0748] | 100.0% | 100.0% | initial>0: True; initial>extra: True |

## Matched architecture contrasts at Pro

Each contrast uses the same five Pro personas. Deltas are relative to Single-Call × Pro (five-persona subset).

| Model | Architecture | N | ΔMPD [95% CI] | P(Δ>0), n=50 |
|---|---|---:|---:|---:|
| DeepSeek | Multi-Turn | 100 | 0.0443 [0.0363, 0.0526] | 100.0% |
| DeepSeek | Multi-Agent | 100 | 0.0276 [0.0170, 0.0390] | 100.0% |
| GPT-mini | Multi-Turn | 100 | 0.0781 [0.0672, 0.0894] | 100.0% |
| GPT-mini | Multi-Agent | 100 | 0.0464 [0.0345, 0.0588] | 100.0% |
| Gemini | Multi-Turn | 100 | 0.0239 [0.0196, 0.0282] | 100.0% |
| Gemini | Multi-Agent | 100 | 0.0238 [0.0185, 0.0292] | 100.0% |
| Kimi | Multi-Turn | 99 | 0.0459 [0.0368, 0.0554] | 100.0% |
| Kimi | Multi-Agent | 100 | 0.0534 [0.0432, 0.0641] | 100.0% |
| Llama | Multi-Turn | 98 | 0.0586 [0.0464, 0.0713] | 100.0% |
| Llama | Multi-Agent | 98 | 0.0269 [0.0143, 0.0401] | 100.0% |
| Grok | Multi-Turn | 100 | 0.0599 [0.0516, 0.0693] | 100.0% |
| Grok | Multi-Agent | 100 | 0.0450 [0.0368, 0.0537] | 100.0% |
| Qwen | Multi-Turn | 99 | 0.0547 [0.0455, 0.0644] | 100.0% |
| Qwen | Multi-Agent | 100 | 0.0696 [0.0589, 0.0804] | 100.0% |

## Formal persona × architecture difference-in-differences

DID = (architecture Pro−None) − (Single-Call Pro−None). A value near zero is consistent with no detectable MPD interaction relative to Single-Call, but does not establish equivalence.

| Model | Architecture | N | DID [95% CI] | Wilcoxon p (BH) |
|---|---|---:|---:|---:|
| DeepSeek | sequential | 100 | -0.0197 [-0.0291, -0.0105] | 8.58e-05 |
| DeepSeek | interactive | 100 | -0.0021 [-0.0156, 0.0117] | 0.649 |
| Gemini | sequential | 100 | -0.0197 [-0.0265, -0.0130] | 5.09e-06 |
| Gemini | interactive | 100 | -0.0016 [-0.0097, 0.0062] | 0.956 |
| Kimi | sequential | 98 | -0.0157 [-0.0235, -0.0082] | 0.00104 |
| Kimi | interactive | 98 | -0.0115 [-0.0210, -0.0022] | 0.0653 |
| Llama | sequential | 99 | 0.0073 [-0.0018, 0.0166] | 0.239 |
| Llama | interactive | 100 | -0.0112 [-0.0270, 0.0044] | 0.239 |
| Grok | sequential | 100 | 0.0047 [-0.0016, 0.0109] | 0.239 |
| Grok | interactive | 100 | 0.0003 [-0.0076, 0.0083] | 0.936 |
| Qwen | sequential | 99 | 0.0067 [-0.0028, 0.0163] | 0.26 |
| Qwen | interactive | 99 | -0.0067 [-0.0184, 0.0047] | 0.544 |

## Interpretation guardrails

- These analyses quantify stability over the submitted 100-task sampling frame; they do not create evidence for other languages, geographies, or task families.
- Failure to reject a zero interaction is not proof of exact orthogonality. The defensible wording is “no detectable MPD interaction in this design.”
- The analyses use released per-task MPD values and do not address human construct validity.
