"""Diversity metrics for response embeddings.

Single-condition metrics: take embeddings from one condition's responses,
return a scalar measuring diversity.
"""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from vendi_score import vendi


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance. Higher = more diverse."""
    if len(embeddings) < 2:
        return float("nan")
    return float(pdist(embeddings, metric="cosine").mean())


def cluster_count(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.65,
) -> int:
    """Number of clusters via agglomerative clustering with average linkage.

    Two clusters merge if their average pairwise cosine similarity exceeds
    similarity_threshold. Default 0.65 from pilot sensitivity analysis.
    """
    if len(embeddings) < 2:
        return len(embeddings)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
    )
    return int(len(set(clusterer.fit_predict(embeddings))))


def vendi_score(embeddings: np.ndarray) -> float:
    """Vendi Score: effective number of distinct items.

    Uses the official vendi-score package (Friedman & Dieng, TMLR 2023).
    Assumes L2-normalized embeddings → cosine similarity kernel is PSD with K_ii=1.
    """
    if len(embeddings) < 1:
        return float("nan")
    if len(embeddings) == 1:
        return 1.0
    K = embeddings @ embeddings.T
    return float(vendi.score_K(K, normalize=True))


def compute_metrics(embeddings: np.ndarray) -> dict[str, float]:
    """Compute all single-condition diversity metrics."""
    return {
        "n": len(embeddings),
        "mpd": mean_pairwise_distance(embeddings),
        "cc": cluster_count(embeddings),
        "vs": vendi_score(embeddings),
    }


# --- Rarefaction (for fair CC/VS comparison across different N) ---

def rarefied_metrics(
    embeddings: np.ndarray,
    n_subsample: int,
    n_repeats: int = 100,
    similarity_threshold: float = 0.65,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute CC and VS at a fixed subsample size, averaged over repeats.

    Use when comparing conditions with different N. Subsample to min N
    across conditions, repeat n_repeats times for stable estimates.
    Reproducible via seed.

    Bootstrap (with replacement) is used so the same logic applies whether
    n_subsample < len(embeddings) (true rarefaction) or n_subsample == len(embeddings)
    (resampling — gives non-zero MC variance for symmetric paired tests).
    """
    if len(embeddings) < 2:
        return {
            "cc_rarefied": {"mean": float(len(embeddings)), "std": 0.0, "n_repeats": 0},
            "vs_rarefied": {"mean": float(len(embeddings)), "std": 0.0, "n_repeats": 0},
        }
    rng = np.random.default_rng(seed)
    cc_vals = []
    vs_vals = []
    replace = n_subsample >= len(embeddings)
    for _ in range(n_repeats):
        idx = rng.choice(len(embeddings), size=n_subsample, replace=replace)
        sub = embeddings[idx]
        cc_vals.append(cluster_count(sub, similarity_threshold))
        vs_vals.append(vendi_score(sub))
    return {
        "cc_rarefied": {
            "mean": float(np.mean(cc_vals)),
            "std": float(np.std(cc_vals, ddof=1)),
            "n_repeats": n_repeats,
        },
        "vs_rarefied": {
            "mean": float(np.mean(vs_vals)),
            "std": float(np.std(vs_vals, ddof=1)),
            "n_repeats": n_repeats,
        },
    }


# --- Cross-condition metrics (β-diversity) ---

def beta_vs(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """β-Vendi Score: diversity gained by combining A and B.

    β-VS = VS(A∪B) - wA·VS(A) - wB·VS(B)
    where wA = |A|/(|A|+|B|). Adapted from Friedman & Dieng 2023 §5
    additive decomposition (analog of Lande 1996 for ecology).
    Higher = more unique diversity between the two sets.
    """
    n_a, n_b = len(emb_a), len(emb_b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    w_a = n_a / (n_a + n_b)
    w_b = n_b / (n_a + n_b)
    emb_union = np.vstack([emb_a, emb_b])
    return float(vendi_score(emb_union) - w_a * vendi_score(emb_a) - w_b * vendi_score(emb_b))


def unique_cluster_ratio(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    similarity_threshold: float = 0.65,
) -> tuple[float, float]:
    """Fraction of A's clusters not found in B, and vice versa.

    A cluster in A is "found" in B if any cluster centroid in B has
    cosine similarity > threshold with that cluster's centroid.
    Returns (unique_a_ratio, unique_b_ratio).

    Centroid is the L2-normalized mean of cluster members (Fréchet mean
    approximation on the unit sphere; exact only for tight clusters).
    """
    def _centroids(emb: np.ndarray) -> np.ndarray:
        if len(emb) == 0:
            return np.zeros((0, emb.shape[1] if emb.ndim == 2 else 0))
        if len(emb) == 1:
            return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        clusterer = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=1 - similarity_threshold,
        )
        labels = clusterer.fit_predict(emb)
        centers = np.array([emb[labels == l].mean(axis=0) for l in sorted(set(labels))])
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        return centers / (norms + 1e-12)

    def _unique_ratio(centroids_src: np.ndarray, centroids_tgt: np.ndarray) -> float:
        if len(centroids_src) == 0:
            return 0.0
        if len(centroids_tgt) == 0:
            return 1.0  # nothing in target → all of source is unique
        sim = centroids_src @ centroids_tgt.T
        found = (sim.max(axis=1) >= similarity_threshold)
        return float(1 - found.mean())

    centroids_a = _centroids(emb_a)
    centroids_b = _centroids(emb_b)
    return _unique_ratio(centroids_a, centroids_b), _unique_ratio(centroids_b, centroids_a)
