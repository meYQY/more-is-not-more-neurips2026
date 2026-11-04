# More Is Not More: What Matters for Opinion Diversity in LLM Simulations?

Evaluation protocol and codebase for the paper *"More Is Not More: What Matters for Opinion Diversity in LLM Simulations?"* by Qiyang Yao (NeurIPS 2026, Track on Evaluations and Datasets).

```bibtex
@inproceedings{yao2026morenotmore,
  title     = {More Is Not More: What Matters for Opinion Diversity in {LLM} Simulations?},
  author    = {Yao, Qiyang},
  booktitle = {Advances in Neural Information Processing Systems (Track on Evaluations and Datasets)},
  year      = {2026}
}
```

## Overview

This codebase implements a factorial evaluation protocol that separates two primary intervention dimensions — **input conditioning** (persona depth) and **interaction architecture** — into a controlled design, and measures opinion diversity using complementary metrics at two levels:

- **α-diversity** (within-condition): Mean Pairwise Distance, Cluster Count, Vendi Score
- **β-diversity** (between-condition): β-Vendi Score, Unique Cluster Ratio

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Key dependencies: `openai`, `sentence-transformers`, `vendi-score`, `scipy`, `scikit-learn`, `statsmodels`.

## Pipeline

The evaluation pipeline has four stages:

```
Step 1: Generate    raw LLM responses
Step 2: Extract     atomic opinion statements
Step 3: Embed       opinion embeddings
Step 4: Metrics     diversity computation + statistical tests
```

### Step 1: Generate

Run a single condition:
```bash
python scripts/run_experiment.py configs/conditions/independent/single.yaml
```

Run all conditions:
```bash
python scripts/run_all.py [--n-tasks N] [--conditions c1,c2,...] [--parallel N]
```

Use `--base-config` to specify a different model (default: `configs/base.yaml`).

### Step 2: Extract

Extract atomic opinions from raw responses:
```bash
python scripts/extract.py <condition_name>
python scripts/extract_model.py <model_suffix>   # for non-default models
```

### Step 3: Embed

Embed extracted opinions:
```bash
python scripts/embed.py --embedder openai-3-small
```

### Step 4: Metrics

Compute per-task diversity metrics:
```bash
python scripts/compute_metrics.py --embedder openai-3-small
```

Compute rarefied metrics for fair cross-condition comparison:
```bash
python scripts/compute_rarefied.py
```

Compute pairwise statistical tests (Wilcoxon + Cliff's δ + FDR):
```bash
python scripts/compute_stats.py
```

Compute β-diversity (between-condition complementarity):
```bash
python scripts/compute_cross_condition.py
python scripts/compute_baseline_calibration.py
```

## Experimental Design

The factorial grid crosses 5 persona depth levels with 3 interaction architectures:

|                | Single-Call | Multi-Turn | Multi-Agent |
|----------------|:-----------:|:----------:|:-----------:|
| **None**       | ✓           | ✓          | ✓           |
| **Role**       | ✓           | ✓          | ✓           |
| **Basic**      | ✓           | ✓          | ✓           |
| **Mid**        | ✓           | ✓          | ✓           |
| **Pro**        | ✓           | ✓          | ✓           |

Plus 4 low-cost ablation conditions (Enhanced Prompt, High Temperature, Demographic Cueing, Trait Assignment).

All condition configurations are in `configs/conditions/`. The 20 personas at 4 depth levels are defined in `configs/personas.yaml`.

## Validation

- **Cross-extractor validation**: `scripts/validate_cross_extractor.py`
- **Embedding sensitivity check**: `scripts/sensitivity_embedder.py`
- **Per-turn diversity analysis**: `scripts/analyze_perturn.py`

Validation data (human annotations, cross-extractor reports) is in `validation/`.

## Data and Artifacts

This repository is the submitted code artifact, not a standalone dataset submission. It includes the task metadata, condition and persona configs, validation artifacts, and aggregated metric outputs needed to inspect the protocol and reproduce the reported summaries. It does not include the full raw LLM response corpus.

- `data/dataset/tasks.csv` — 100 open-ended question IDs and metadata
- `configs/personas.yaml` — the 20 persona profiles used across depth levels
- `configs/conditions/` — all condition manifests
- `results/metrics*/` — pre-computed diversity metrics for 7 models
- `validation/` — extraction validation data

## Configuration

API keys are read from environment variables specified in each model config (e.g., `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`). Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

## Tests

```bash
pytest
```

## License

CC BY 4.0. See `LICENSE`.

Task metadata is derived from WildChat-1M; third-party packages and model/API providers retain their own licenses and terms. The full raw LLM response corpus is not included.

## Rebuttal-Period Analyses

The `rebuttal/` folder contains the code, de-identified human-evaluation data, and result files for the three analyses added to the camera-ready appendix: the blinded human evaluation of viewpoint breadth (Appendix J), robustness within the 100-question benchmark (Appendix K), and Multi-Agent prompt sensitivity (Appendix L). Each includes an offline script that recomputes the corresponding paper table from the included files without API calls. See [`rebuttal/README.md`](rebuttal/README.md) for inputs, commands, and which steps require API keys.
