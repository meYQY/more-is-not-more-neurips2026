"""Disk-backed embedding cache.

Embed once, reuse forever. Cache key = (embedder_id, scrub_variant, condition, task_id).
Stored as numpy .npy files for fast load.

Layout:
  results/embeddings/{embedder}/{variant}/{condition}/MOD-XXX.npy   # shape (N, dim)
  results/embeddings/{embedder}/{variant}/{condition}/MOD-XXX.json  # opinion-text fingerprint

Fingerprint check ensures we only reuse cache when input opinions are identical.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np


def _fingerprint(opinions: list[str]) -> str:
    """SHA256 prefix of the joined opinion texts."""
    joined = "\n".join(opinions).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def cache_path(
    cache_root: Path,
    embedder_id: str,
    variant: str,
    condition: str,
    task_id: str,
) -> tuple[Path, Path]:
    """Return (npy_path, meta_path) for a given cache key."""
    base = cache_root / embedder_id / variant / condition
    return base / f"{task_id}.npy", base / f"{task_id}.json"


def load_cached(
    cache_root: Path,
    embedder_id: str,
    variant: str,
    condition: str,
    task_id: str,
    opinions: list[str],
) -> np.ndarray | None:
    """Return cached embeddings if fingerprint matches, else None."""
    npy_path, meta_path = cache_path(cache_root, embedder_id, variant, condition, task_id)
    if not (npy_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != _fingerprint(opinions):
            return None
        if meta.get("n_opinions") != len(opinions):
            return None
        return np.load(npy_path)
    except Exception:
        return None


def save_cached(
    cache_root: Path,
    embedder_id: str,
    variant: str,
    condition: str,
    task_id: str,
    opinions: list[str],
    embeddings: np.ndarray,
) -> None:
    """Save embeddings + fingerprint metadata."""
    npy_path, meta_path = cache_path(cache_root, embedder_id, variant, condition, task_id)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings.astype(np.float32))
    meta = {
        "embedder": embedder_id,
        "variant": variant,
        "condition": condition,
        "task_id": task_id,
        "n_opinions": len(opinions),
        "fingerprint": _fingerprint(opinions),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
    }
    meta_path.write_text(json.dumps(meta))


def embed_with_cache(
    opinions: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    cache_root: Path,
    embedder_id: str,
    variant: str,
    condition: str,
    task_id: str,
) -> tuple[np.ndarray, bool]:
    """Embed if not cached, return (embeddings, was_cached_hit)."""
    cached = load_cached(cache_root, embedder_id, variant, condition, task_id, opinions)
    if cached is not None:
        return cached, True
    emb = embed_fn(opinions)
    save_cached(cache_root, embedder_id, variant, condition, task_id, opinions, emb)
    return emb, False
