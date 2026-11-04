"""Sentence embedding adapters.

Two backends:
- bge-small (local, 384 dim, 512 token context, free)
- openai-3-small (API via OpenRouter, 1536 dim, 8192 token context, ~$0.02/M tokens)

Both return L2-normalized embeddings for cosine similarity.
"""

import os
from typing import Callable

import numpy as np

# === BGE local backend ===

_BGE_MODEL = None


def _get_bge_model():
    global _BGE_MODEL
    if _BGE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _BGE_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _BGE_MODEL


def embed_bge(texts: list[str]) -> np.ndarray:
    """BGE-small-en-v1.5 (local). Shape (n, 384), L2-normalized."""
    model = _get_bge_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# === OpenAI via OpenRouter backend ===

_OPENAI_CLIENT = None


def _get_openai_client():
    """OpenRouter-routed OpenAI client. Uses OPENROUTER_API_KEY."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in env")
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _OPENAI_CLIENT


def embed_openai_3small(texts: list[str], batch_size: int = 2048) -> np.ndarray:
    """OpenAI text-embedding-3-small via OpenRouter. Shape (n, 1536), L2-normalized."""
    client = _get_openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Skip empty strings (API rejects them)
        batch = [t if t.strip() else " " for t in batch]
        resp = client.embeddings.create(model="openai/text-embedding-3-small", input=batch)
        out.extend([d.embedding for d in resp.data])
    arr = np.asarray(out, dtype=np.float32)
    # OpenAI embeddings are not always exactly L2-normalized; normalize to be safe
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-12)


# === Registry ===

EMBEDDERS: dict[str, Callable[[list[str]], np.ndarray]] = {
    "bge-small": embed_bge,
    "openai-3-small": embed_openai_3small,
}

EMBEDDER_DIMS: dict[str, int] = {
    "bge-small": 384,
    "openai-3-small": 1536,
}


def embed_texts(texts: list[str], embedder: str = "bge-small") -> np.ndarray:
    """Embed texts using the named backend. Default = bge-small for backward compat."""
    if embedder not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{embedder}'. Available: {list(EMBEDDERS)}")
    return EMBEDDERS[embedder](texts)


# === Backward-compat: keep old name ===
def get_model():
    """Deprecated: kept for backward compat with code that imported the BGE model directly."""
    return _get_bge_model()
