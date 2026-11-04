# 100-Task Artifact Audit

## Generation

- Conditions: neutral, agreement, disagreement.
- Tasks per condition: 100.
- Records per condition: 1,500 (5 personas × 3 rounds × 100 tasks).
- Missing, extra, invalid, or partial task files: 0.
- Model: `deepseek-v4-flash` for all 4,500 records.
- Thinking mode: disabled for all 4,500 records.
- Finish reason: `stop` for all 4,500 records.
- Shared round 1: request and response match exactly across all three
  conditions for all 100 tasks.
- Unique billed generation tokens after counting the shared first round once:
  12,163,433 prompt tokens and 1,732,315 completion tokens.
- Estimated DeepSeek generation cost: USD 2.187929 at the recorded V4 Flash
  rates of USD 0.14/M input and USD 0.28/M output.

## Extraction and embeddings

| Condition | Response rows | Extracted opinions (all rounds) | Extracted opinions (rounds 2–3) | Full embedding files | Post-intervention embedding files |
|---|---:|---:|---:|---:|---:|
| neutral | 1,500 | 11,087 | 7,166 | 100 | 100 |
| agreement | 1,500 | 13,382 | 9,461 | 100 | 100 |
| disagreement | 1,500 | 15,964 | 12,043 | 100 | 100 |

- Extraction errors: 0.
- Embedding model: `text-embedding-3-small`.
- Embedding dimensionality: 1,536 for every artifact.
- Opinion-count / embedding-row mismatches: 0.
- The shared round-1 extraction is synchronized across conditions for all
  tasks, preventing extractor jitter from entering the condition contrast.

## Primary results

### All rounds

| Condition | MPD |
|---|---:|
| neutral | 0.7165 |
| agreement | 0.6958 |
| disagreement | 0.7149 |

- Agreement − neutral: -0.0207, task-bootstrap 95% CI [-0.0258, -0.0157].
- Disagreement − neutral: -0.0016, 95% CI [-0.0061, +0.0029].
- Disagreement − agreement: +0.0190, 95% CI [+0.0137, +0.0245].

### Rounds 2–3

| Condition | MPD |
|---|---:|
| neutral | 0.7230 |
| agreement | 0.6898 |
| disagreement | 0.7160 |

- Agreement − neutral: -0.0332, task-bootstrap 95% CI [-0.0401, -0.0263].
- Disagreement − neutral: -0.0070, 95% CI [-0.0129, -0.0010].
- Disagreement − agreement: +0.0262, 95% CI [+0.0195, +0.0332].

The 100-task result has the same direction as the retained 30-task checkpoint.

## Safe interpretation

Holding the model, questions, five personas, agent order, first-round
responses, decoding, and analysis pipeline fixed, agreement-seeking
instructions reliably reduced MPD. Disagreement-seeking produced higher MPD
than agreement-seeking, but did not exceed the submitted neutral instruction.
Thus, in this controlled setting, the neutral prompt was not equivalent to an
agreement-seeking prompt, and explicitly encouraging disagreement did not
automatically increase semantic viewpoint dispersion beyond neutral.

This single-model experiment directly addresses prompt sensitivity. It does
not validate MPD against human-perceived opinion diversity and should not be
used to answer the separate human-verification concern.

