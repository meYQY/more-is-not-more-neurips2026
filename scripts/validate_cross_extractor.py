"""Cross-extractor validation: re-extract with Kimi K2.5, compare condition-level MPD.

Re-extracts a subset of tasks with a different LLM to verify that
condition-level diversity rankings are extractor-independent.

Phases:
  1. Extract 20 tasks × all conditions with Kimi K2.5
  2. Extract 150 validation records with Kimi K2.5
  3. Compute condition-level MPD (bge-small), Spearman vs DeepSeek, record-level Jaccard

Usage:
    python scripts/validate_cross_extractor.py                 # full run
    python scripts/validate_cross_extractor.py --n-tasks 1     # preflight
    python scripts/validate_cross_extractor.py --skip-extract  # compare only
"""

import argparse
import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

from modbench.embedding import embed_texts
from modbench.extractor import _extract_one
from modbench.io import load_config
from modbench.metrics import mean_pairwise_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "results" / "raw"
EXTRACTED_ROOT = PROJECT_ROOT / "results" / "extracted"
OUT_ROOT = PROJECT_ROOT / "validation" / "cross_extractor"
KIMI_CONFIG = PROJECT_ROOT / "configs" / "extraction_kimi.yaml"

EMBEDDER = "bge-small"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_tasks(n_tasks: int, seed: int) -> list[str]:
    all_tasks = sorted(f.stem for f in (RAW_ROOT / "independent-none").glob("MOD-*.jsonl"))
    rng = random.Random(seed)
    return sorted(rng.sample(all_tasks, min(n_tasks, len(all_tasks))))


def get_conditions() -> list[str]:
    return sorted(d.name for d in RAW_ROOT.iterdir() if d.is_dir())


def count_valid_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                break
    return count


def load_opinions_from_jsonl(path: Path) -> list[str]:
    ops: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ops.extend(rec.get("opinions", []))
    return [o for o in ops if o.strip()]


# ---------------------------------------------------------------------------
# Phase 1: Extract with Kimi (condition × task)
# ---------------------------------------------------------------------------

async def extract_task_file(
    raw_file: Path,
    out_file: Path,
    config: dict,
    sem: asyncio.Semaphore,
) -> tuple[int, int]:
    with open(raw_file, encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    n_done = count_valid_lines(out_file)
    if n_done >= len(raw_records):
        return n_done, 0

    remaining = raw_records[n_done:]

    async def _do(rec: dict) -> list[str]:
        q = rec["raw_question"]
        r = rec["response"]["content"]
        return await _extract_one(q, r, config, sem=sem)

    results = await asyncio.gather(*[_do(r) for r in remaining], return_exceptions=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for rec, result in zip(remaining, results):
            out_rec: dict = {
                "task_id": rec["task_id"],
                "persona_name": rec.get("persona_name"),
            }
            if isinstance(result, BaseException):
                out_rec["opinions"] = []
                out_rec["error"] = str(result)
                n_err += 1
            else:
                out_rec["opinions"] = result
                n_ok += 1
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    return n_done + n_ok, n_err


async def extract_all_conditions(
    tasks: list[str], config: dict, max_concurrent: int,
) -> None:
    conditions = get_conditions()
    sem = asyncio.Semaphore(max_concurrent)
    ext_root = OUT_ROOT / "extracted"

    total_remaining = 0
    for cond in conditions:
        for tid in tasks:
            raw = RAW_ROOT / cond / f"{tid}.jsonl"
            if not raw.exists():
                continue
            n_raw = sum(1 for line in open(raw, encoding="utf-8") if line.strip())
            n_done = count_valid_lines(ext_root / cond / f"{tid}.jsonl")
            total_remaining += max(0, n_raw - n_done)

    print(f"Phase 1: Extract with Kimi")
    print(f"  Conditions: {len(conditions)}, Tasks: {len(tasks)}")
    print(f"  Remaining calls: {total_remaining}")

    if total_remaining == 0:
        print("  Already complete, skipping.")
        return

    t0 = time.monotonic()
    jobs = []
    keys = []
    for cond in conditions:
        for tid in tasks:
            raw = RAW_ROOT / cond / f"{tid}.jsonl"
            if not raw.exists():
                continue
            out = ext_root / cond / f"{tid}.jsonl"
            jobs.append(extract_task_file(raw, out, config, sem))
            keys.append(f"{cond}/{tid}")

    results = await asyncio.gather(*jobs, return_exceptions=True)

    total_ok = total_err = 0
    for key, result in zip(keys, results):
        if isinstance(result, BaseException):
            print(f"  ERROR {key}: {result}")
            total_err += 1
        else:
            total_ok += result[0]
            total_err += result[1]

    elapsed = time.monotonic() - t0
    print(f"  Done in {elapsed:.1f}s. {total_ok} ok, {total_err} errors.")


# ---------------------------------------------------------------------------
# Phase 2: Extract 150 validation records
# ---------------------------------------------------------------------------

async def extract_validation_150(config: dict, max_concurrent: int) -> None:
    records_path = PROJECT_ROOT / "validation" / "records_data.json"
    out_path = OUT_ROOT / "validation_150_kimi.jsonl"

    if not records_path.exists():
        print("  validation/records_data.json not found, skipping 150-record extraction.")
        return

    records = json.loads(records_path.read_text(encoding="utf-8"))
    n_done = count_valid_lines(out_path)

    if n_done >= len(records):
        print(f"  Validation 150: already complete ({n_done} records)")
        return

    remaining = records[n_done:]
    sem = asyncio.Semaphore(max_concurrent)

    print(f"  Extracting {len(remaining)} validation records with Kimi...")

    async def _do(rec: dict) -> list[str]:
        return await _extract_one(rec["question"], rec["response"], config, sem=sem)

    results = await asyncio.gather(*[_do(r) for r in remaining], return_exceptions=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for rec, result in zip(remaining, results):
            out_rec: dict = {"index": rec["index"]}
            if isinstance(result, BaseException):
                out_rec["opinions"] = []
                out_rec["error"] = str(result)
                n_err += 1
            else:
                out_rec["opinions"] = result
                n_ok += 1
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"  Validation 150: {n_ok} ok, {n_err} errors.")


# ---------------------------------------------------------------------------
# Phase 3: Compare
# ---------------------------------------------------------------------------

def compute_task_mpd(opinions: list[str]) -> float:
    if len(opinions) < 2:
        return float("nan")
    emb = embed_texts(opinions, embedder=EMBEDDER)
    return mean_pairwise_distance(emb)


def jaccard_semantic(
    list_a: list[str], list_b: list[str], threshold: float = 0.85,
) -> float:
    if not list_a and not list_b:
        return 1.0
    if not list_a or not list_b:
        return 0.0
    emb = embed_texts(list_a + list_b, embedder=EMBEDDER)
    na, nb = len(list_a), len(list_b)
    ea, eb = emb[:na], emb[na:]
    norms_a = np.linalg.norm(ea, axis=1, keepdims=True)
    norms_b = np.linalg.norm(eb, axis=1, keepdims=True)
    ea = ea / (norms_a + 1e-12)
    eb = eb / (norms_b + 1e-12)
    sim = ea @ eb.T
    matched_b: set[int] = set()
    matches = 0
    for i in range(na):
        order = np.argsort(-sim[i])
        for j in order:
            if sim[i, j] < threshold:
                break
            if int(j) not in matched_b:
                matched_b.add(int(j))
                matches += 1
                break
    union = na + nb - matches
    return matches / union if union else 1.0


def compare(tasks: list[str], seed: int) -> dict:
    conditions = get_conditions()
    ext_root = OUT_ROOT / "extracted"

    print(f"\n=== Phase 3: Compare (embedder={EMBEDDER}) ===")

    # --- 3a: Condition-level MPD ---
    ds_cond_mpd: dict[str, float] = {}
    km_cond_mpd: dict[str, float] = {}
    granularity_cond: dict[str, dict] = {}

    for cond in conditions:
        ds_mpds = []
        km_mpds = []
        ds_ns = []
        km_ns = []

        for tid in tasks:
            ds_file = EXTRACTED_ROOT / cond / f"{tid}.jsonl"
            km_file = ext_root / cond / f"{tid}.jsonl"
            if not ds_file.exists() or not km_file.exists():
                continue

            ds_ops = load_opinions_from_jsonl(ds_file)
            km_ops = load_opinions_from_jsonl(km_file)

            ds_ns.append(len(ds_ops))
            km_ns.append(len(km_ops))

            ds_m = compute_task_mpd(ds_ops)
            km_m = compute_task_mpd(km_ops)
            if not np.isnan(ds_m):
                ds_mpds.append(ds_m)
            if not np.isnan(km_m):
                km_mpds.append(km_m)

        if ds_mpds:
            ds_cond_mpd[cond] = float(np.mean(ds_mpds))
        if km_mpds:
            km_cond_mpd[cond] = float(np.mean(km_mpds))

        ds_n_mean = float(np.mean(ds_ns)) if ds_ns else 0
        km_n_mean = float(np.mean(km_ns)) if km_ns else 0
        granularity_cond[cond] = {"deepseek": ds_n_mean, "kimi": km_n_mean}

        print(
            f"  {cond:25s}  DS={ds_cond_mpd.get(cond, float('nan')):.4f}  "
            f"KM={km_cond_mpd.get(cond, float('nan')):.4f}  "
            f"DS_n={ds_n_mean:.0f}  KM_n={km_n_mean:.0f}"
        )

    # Spearman on shared conditions
    shared = sorted(set(ds_cond_mpd) & set(km_cond_mpd))
    ds_vals = [ds_cond_mpd[c] for c in shared]
    km_vals = [km_cond_mpd[c] for c in shared]

    if len(shared) >= 3:
        sp = stats.spearmanr(ds_vals, km_vals)
        pr = stats.pearsonr(ds_vals, km_vals)
        print(f"\n  Spearman rho = {sp.statistic:.4f} (p = {sp.pvalue:.2e}, n = {len(shared)})")
        print(f"  Pearson  r   = {pr.statistic:.4f} (p = {pr.pvalue:.2e})")
    else:
        sp = pr = None
        print(f"\n  Not enough conditions ({len(shared)}) for correlation.")

    # --- 3b: Record-level Jaccard on 150 validation records ---
    jaccard_results = _compute_validation_jaccard()

    # --- 3c: Granularity summary ---
    all_ds_n = [granularity_cond[c]["deepseek"] for c in conditions if c in granularity_cond]
    all_km_n = [granularity_cond[c]["kimi"] for c in conditions if c in granularity_cond]
    ds_mean = float(np.mean(all_ds_n)) if all_ds_n else 0
    km_mean = float(np.mean(all_km_n)) if all_km_n else 0
    ratio = km_mean / ds_mean if ds_mean > 0 else float("nan")
    print(f"\n  Granularity (mean opinions/task across conditions):")
    print(f"    DeepSeek: {ds_mean:.1f}")
    print(f"    Kimi:     {km_mean:.1f}")
    print(f"    Ratio:    {ratio:.2f}")

    # --- Build report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_tasks": len(tasks),
            "tasks": tasks,
            "seed": seed,
            "extraction_model": "kimi-k2.5",
            "baseline_model": "deepseek-chat",
            "embedder": EMBEDDER,
        },
        "condition_mpd": {
            "deepseek": ds_cond_mpd,
            "kimi": km_cond_mpd,
            "conditions_used": shared,
        },
        "spearman": {
            "rho": float(sp.statistic) if sp else None,
            "p_value": float(sp.pvalue) if sp else None,
            "n_conditions": len(shared),
        },
        "pearson": {
            "r": float(pr.statistic) if pr else None,
            "p_value": float(pr.pvalue) if pr else None,
        },
        "record_jaccard": jaccard_results,
        "granularity": {
            "per_condition": granularity_cond,
            "overall_deepseek_mean": ds_mean,
            "overall_kimi_mean": km_mean,
            "ratio": ratio,
        },
    }

    return report


def _compute_validation_jaccard() -> dict:
    records_path = PROJECT_ROOT / "validation" / "records_data.json"
    kimi_path = OUT_ROOT / "validation_150_kimi.jsonl"

    if not records_path.exists() or not kimi_path.exists():
        print("\n  Validation Jaccard: files not found, skipping.")
        return {}

    records = json.loads(records_path.read_text(encoding="utf-8"))
    kimi_recs: list[dict] = []
    with open(kimi_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kimi_recs.append(json.loads(line))

    n = min(len(records), len(kimi_recs))
    if n == 0:
        return {}

    print(f"\n  Computing record-level Jaccard on {n} validation records...")
    jaccards = []
    ds_counts = []
    km_counts = []
    by_arch: dict[str, list[float]] = {}

    for i in range(n):
        ds_ops = records[i]["opinions"]
        km_ops = kimi_recs[i].get("opinions", [])
        ds_counts.append(len(ds_ops))
        km_counts.append(len(km_ops))

        if ds_ops and km_ops:
            j = jaccard_semantic(ds_ops, km_ops)
            jaccards.append(j)
            arch = records[i]["architecture"]
            by_arch.setdefault(arch, []).append(j)

    if not jaccards:
        return {}

    print(f"  Jaccard: mean={np.mean(jaccards):.3f}, median={np.median(jaccards):.3f}, "
          f"min={np.min(jaccards):.3f}, std={np.std(jaccards):.3f}")
    for arch in sorted(by_arch):
        vals = by_arch[arch]
        print(f"    {arch}: mean={np.mean(vals):.3f} (n={len(vals)})")
    print(f"  Granularity: DS={np.mean(ds_counts):.1f}, Kimi={np.mean(km_counts):.1f}")

    return {
        "mean": float(np.mean(jaccards)),
        "median": float(np.median(jaccards)),
        "min": float(np.min(jaccards)),
        "max": float(np.max(jaccards)),
        "std": float(np.std(jaccards)),
        "n": len(jaccards),
        "by_architecture": {
            arch: {"mean": float(np.mean(v)), "n": len(v)}
            for arch, v in sorted(by_arch.items())
        },
        "granularity": {
            "deepseek_mean": float(np.mean(ds_counts)),
            "kimi_mean": float(np.mean(km_counts)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, run comparison only")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    tasks = sample_tasks(args.n_tasks, args.seed)
    print(f"Sampled {len(tasks)} tasks (seed={args.seed}): {', '.join(tasks)}")

    if not args.skip_extract:
        config = load_config(KIMI_CONFIG)
        await extract_all_conditions(tasks, config, args.max_concurrent)
        print()
        await extract_validation_150(config, args.max_concurrent)

    report = compare(tasks, args.seed)

    out_path = OUT_ROOT / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
