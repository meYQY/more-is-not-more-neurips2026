"""Statistical tests for comparing conditions.

Each task yields one metric value per condition → paired comparisons
across 100 tasks using non-parametric tests.
"""

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests


def paired_wilcoxon(values_a: list[float], values_b: list[float]) -> float:
    """Paired Wilcoxon signed-rank test. Returns p-value.

    Appropriate because metric values are paired by task (same 100 tasks
    across conditions) and we don't assume normality. NaN pairs dropped.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return 1.0
    diff = a - b
    if np.all(diff == 0):
        return 1.0
    _, p = wilcoxon(diff)
    return float(p)


def paired_cliffs_delta(values_a: list[float], values_b: list[float]) -> float:
    """Matched-pairs rank-biserial correlation. Range [-1, 1].

    For paired observations (a_i, b_i), counts how often a_i > b_i vs a_i < b_i.
    This is the appropriate effect size for matched/paired designs.

    Note: This is NOT the unpaired Cliff's delta (which is O(n²) all-pairs).
    Effect size classification uses Romano et al. 2006 thresholds as approximate
    guidelines: |r| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return 0.0
    diff = a - b
    more = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    return float((more - less) / n)


def cliffs_delta_unpaired(values_a: list[float], values_b: list[float]) -> float:
    """Original (unpaired) Cliff's delta. Use only when groups truly are independent.

    Vectorized: O(n_a * n_b) but uses numpy.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    mask_a = ~np.isnan(a); mask_b = ~np.isnan(b)
    a, b = a[mask_a], b[mask_b]
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.sum() / (n_a * n_b))


def fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values.

    Uses statsmodels for correctness on ties and edge cases.
    """
    if len(p_values) == 0:
        return []
    p = np.asarray(p_values, dtype=float)
    mask = ~np.isnan(p)
    if not mask.any():
        return [float("nan")] * len(p_values)
    adjusted = np.full(len(p), float("nan"))
    _, adj_subset, _, _ = multipletests(p[mask], alpha=alpha, method="fdr_bh")
    adjusted[mask] = adj_subset
    return [float(x) for x in adjusted]


def friedman_omnibus(values_per_condition: list[list[float]]) -> float:
    """Friedman test across multiple conditions on matched samples (paired by task).

    values_per_condition[c][t] = metric value for condition c on task t.
    Returns omnibus p-value. Use before pairwise comparisons to control family-wise error.
    """
    arrays = [np.asarray(v, dtype=float) for v in values_per_condition]
    n_tasks = len(arrays[0])
    if not all(len(a) == n_tasks for a in arrays):
        raise ValueError("All conditions must have the same number of tasks")
    if len(arrays) < 3:
        raise ValueError("Friedman requires ≥3 groups")
    stacked = np.stack(arrays, axis=0)  # (n_conditions, n_tasks)
    mask = ~np.isnan(stacked).any(axis=0)
    stacked = stacked[:, mask]
    if stacked.shape[1] < 2:
        return 1.0
    _, p = friedmanchisquare(*stacked)
    return float(p)


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a list of values."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(a, size=len(a), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
