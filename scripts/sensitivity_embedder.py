"""Embedding model sensitivity check (standalone, does not touch pipeline code).

Re-embeds DeepSeek and Gemini extracted opinions with alternative embedders,
computes MPD, and checks rank correlation with the primary embedder (openai-3-small).

Usage:
    python scripts/sensitivity_embedder.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Config ---

ALT_EMBEDDERS = {
    "qwen3-8b": {"model": "qwen/qwen3-embedding-8b", "dim": 4096},
    "bge-m3": {"model": "baai/bge-m3", "dim": 1024},
}

# Models × conditions to test
TEST_SETS = {
    "deepseek": {
        "extracted_dir": PROJECT_ROOT / "results" / "extracted",
        "primary_emb_dir": PROJECT_ROOT / "results" / "embeddings" / "openai-3-small" / "original",
    },
    "gemini": {
        "extracted_dir": PROJECT_ROOT / "results" / "extracted_gemini",
        "primary_emb_dir": PROJECT_ROOT / "results" / "embeddings_gemini" / "openai-3-small",
    },
}

CONDITIONS = [
    "independent-none", "independent-role", "independent-basic", "independent-mid", "independent-pro",
    "independent-pro-5", "independent-enhanced", "independent-temp-high", "independent-dimension", "independent-random-sys",
    "sequential-none", "sequential-basic", "sequential-mid", "sequential-pro",
    "interactive-none", "interactive-basic", "interactive-mid", "interactive-pro",
]

OUT_DIR = PROJECT_ROOT / "results" / "_sensitivity"


# --- Embedding ---

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _client


def embed_batch(texts: list[str], model: str, batch_size: int = 512) -> np.ndarray:
    """Embed texts via OpenRouter. Returns L2-normalized array."""
    client = get_client()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-12)


import asyncio
import concurrent.futures

async def embed_task_async(executor, emb_model: str, extracted_dir: Path, cond: str, tid: str, cache_dir: Path):
    """Embed one task asynchronously."""
    cache_file = cache_dir / f"{tid}.npy"
    if cache_file.exists():
        return cache_file, True  # cached

    opinions = load_opinions(extracted_dir, cond, tid)
    if not opinions:
        return None, False

    loop = asyncio.get_event_loop()
    for attempt in range(3):
        try:
            emb = await loop.run_in_executor(executor, embed_batch, opinions, emb_model)
            np.save(cache_file, emb.astype(np.float32))
            return cache_file, False
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"      ERROR {cond}/{tid}: {e}")
                return None, False


async def embed_condition_async(emb_name: str, emb_model: str, extracted_dir: Path, cond: str, cache_base: Path, max_concurrent: int = 50):
    """Embed all tasks in a condition concurrently."""
    cache_dir = cache_base / cond
    cache_dir.mkdir(parents=True, exist_ok=True)

    ext_cond = extracted_dir / cond
    if not ext_cond.exists():
        return 0, 0

    tasks_list = sorted(f.stem for f in ext_cond.glob("MOD-*.jsonl"))
    sem = asyncio.Semaphore(max_concurrent)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

    async def bounded(tid):
        async with sem:
            return await embed_task_async(executor, emb_model, extracted_dir, cond, tid, cache_dir)

    results = await asyncio.gather(*[bounded(tid) for tid in tasks_list])
    executor.shutdown(wait=False)

    n_new = sum(1 for _, cached in results if _ is not None and not cached)
    n_cached = sum(1 for _, cached in results if cached)
    return n_new, n_cached


# --- Opinion loading ---

def load_opinions(extracted_dir: Path, condition: str, task_id: str) -> list[str]:
    f = extracted_dir / condition / f"{task_id}.jsonl"
    if not f.exists():
        return []
    opinions = []
    with open(f) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            opinions.extend(rec.get("opinions", []))
    return [o for o in opinions if o.strip()]


# --- MPD ---

def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return float("nan")
    from scipy.spatial.distance import pdist
    return float(pdist(embeddings, metric="cosine").mean())


# --- Main ---

def run_one(gen_model: str, emb_name: str):
    """Embed + compute correlation for one (gen_model, embedder) pair."""
    paths = TEST_SETS[gen_model]
    extracted_dir = paths["extracted_dir"]
    primary_emb_dir = paths["primary_emb_dir"]
    emb_cfg = ALT_EMBEDDERS[emb_name]
    cache_base = OUT_DIR / emb_name / gen_model

    print(f"\n  {gen_model} × {emb_name}", flush=True)

    # Phase 1: embed all conditions concurrently
    for cond in CONDITIONS:
        ext_cond = extracted_dir / cond
        if not ext_cond.exists():
            continue
        n_new, n_cached = asyncio.run(embed_condition_async(
            emb_name, emb_cfg["model"], extracted_dir, cond, cache_base, max_concurrent=50
        ))
        print(f"    {cond}: {n_new} new, {n_cached} cached", flush=True)

    # Phase 2: compute MPD + correlation
    primary_mpds = []
    alt_mpds = []
    task_ids = []

    for cond in CONDITIONS:
        cond_primary_dir = primary_emb_dir / cond
        cache_dir = cache_base / cond
        if not cond_primary_dir.exists() or not cache_dir.exists():
            continue

        for npy_file in sorted(cond_primary_dir.glob("MOD-*.npy")):
            tid = npy_file.stem
            alt_file = cache_dir / f"{tid}.npy"
            if not alt_file.exists():
                continue

            p_mpd = mean_pairwise_distance(np.load(npy_file))
            a_mpd = mean_pairwise_distance(np.load(alt_file))
            if np.isnan(p_mpd) or np.isnan(a_mpd):
                continue

            primary_mpds.append(p_mpd)
            alt_mpds.append(a_mpd)
            task_ids.append(f"{cond}/{tid}")

    if len(primary_mpds) < 10:
        print(f"    Not enough data")
        return None

    rho, p = spearmanr(primary_mpds, alt_mpds)

    # Per-condition means
    cond_primary = {}
    cond_alt = {}
    for tid_full, pm, am in zip(task_ids, primary_mpds, alt_mpds):
        cond = tid_full.rsplit("/", 1)[0]
        if cond not in cond_primary:
            cond_primary[cond] = []
            cond_alt[cond] = []
        cond_primary[cond].append(pm)
        cond_alt[cond].append(am)

    cond_names = sorted(cond_primary.keys())
    primary_means = [np.mean(cond_primary[c]) for c in cond_names]
    alt_means = [np.mean(cond_alt[c]) for c in cond_names]
    rank_rho, rank_p = spearmanr(primary_means, alt_means)

    print(f"\n    Task-level ρ = {rho:.4f} (p={p:.2e}, n={len(primary_mpds)})")
    print(f"    Condition-level ρ = {rank_rho:.4f} (p={rank_p:.2e}, n={len(cond_names)})")
    print(f"\n    {'Condition':24s} {'OpenAI-3s':>10s} {emb_name:>10s}")
    print(f"    {'-'*46}")
    for c, pm, am in zip(cond_names, primary_means, alt_means):
        print(f"    {c:24s} {pm:>10.4f} {am:>10.4f}")

    return {
        "task_level_spearman_rho": rho,
        "task_level_spearman_p": p,
        "task_level_n": len(primary_mpds),
        "condition_level_spearman_rho": rank_rho,
        "condition_level_spearman_p": rank_p,
        "per_condition": {
            c: {"primary_mpd": float(np.mean(cond_primary[c])),
                "alt_mpd": float(np.mean(cond_alt[c])),
                "n_tasks": len(cond_primary[c])}
            for c in cond_names
        },
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb", default=None, help="Embedder name (qwen3-8b or bge-m3)")
    p.add_argument("--gen-model", default=None, help="Generation model (deepseek or gemini)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.emb and args.gen_model:
        # Single pair mode (for parallel launch)
        result = run_one(args.gen_model, args.emb)
        if result:
            out_path = OUT_DIR / f"sensitivity_{args.gen_model}_{args.emb}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved: {out_path}")
        return

    # All pairs mode
    all_results = {}
    for gen_model in TEST_SETS:
        for emb_name in ALT_EMBEDDERS:
            result = run_one(gen_model, emb_name)
            if result:
                all_results[f"{gen_model}/{emb_name}"] = result

    out_path = OUT_DIR / "embedder_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
