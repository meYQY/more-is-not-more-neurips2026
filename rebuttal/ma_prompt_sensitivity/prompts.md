# Multi-Agent prompt-sensitivity prompts

Readable copy of the prompts frozen in `experiment_config.yaml`
(Neutral/Agreement/Disagreement) and `coverage_config.yaml` (Coverage). The
YAML files are what the scripts read.

## Fixed parts (all four conditions)

- Model: `deepseek-v4-flash`, thinking disabled, temperature 0.7,
  `max_tokens` 4096, `top_p` 1.0.
- Agents: personas 1-5 from `configs/personas.yaml` at Pro depth, in that order.
- System prompt for each agent: `You are a helpful assistant.` followed by a
  blank line and the persona's Pro description.
- Round 1 (generated once per question and copied into every condition):

  ```
  A group is discussing the following question. Please share your perspective.

  Question: {question}
  ```

- Rounds 2 and 3: the user message contains the question, all earlier rounds,
  the responses already given in the current round, and then the round
  instruction:

  ```
  A group is discussing the following question.

  Question: {question}

  Round 1:
  {agent name}: {response}
  ...

  Round 2 (so far):
  {agent name}: {response}
  ...

  {round instruction}
  ```

## Round 2-3 instructions (the only part that differs)

| Condition | Round 2 | Round 3 |
|---|---|---|
| Neutral (main study) | Please respond to the discussion. | Please continue the discussion. |
| Agreement | Please respond to the discussion. Actively seek areas of agreement and work toward a shared conclusion, while clearly stating any genuine differences that remain. | Please continue the discussion. Actively seek areas of agreement and work toward a shared conclusion, while clearly stating any genuine differences that remain. |
| Disagreement | Please respond to the discussion. Actively surface genuine differences, counterarguments, and distinct conclusions, while acknowledging any genuine agreement that remains. | Please continue the discussion. Actively surface genuine differences, counterarguments, and distinct conclusions, while acknowledging any genuine agreement that remains. |
| Coverage | Please respond to the discussion with the goal of broadening the range of genuine viewpoints represented. From your own perspective, contribute one substantively different position, reason, trade-off, or condition that has not already been expressed, and explain how it affects your bottom-line view. Do not summarize or merely paraphrase earlier responses, and do not disagree for its own sake. There is no need to reach consensus or change your view unless a substantive argument genuinely persuades you. | Same text as Round 2, but beginning "Please continue the discussion with the goal of ..." |

## Extraction and embedding

Opinion extraction uses the repository's standard extraction prompt
(`src/modbench/extractor.py`) with `deepseek-v4-flash`, thinking disabled,
temperature 0 (`extraction_config.yaml`). Opinions are embedded with OpenAI
`text-embedding-3-small` through the repository's embedding module.
