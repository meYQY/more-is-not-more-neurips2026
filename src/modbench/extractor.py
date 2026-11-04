"""Extract opinion statements from raw LLM responses."""

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .llm import call_llm
from .io import load_config


EXTRACTION_SYSTEM_PROMPT = """You extract opinion statements from a response to a question.

An opinion statement is a claim that expresses a judgment, stance, or viewpoint that someone could agree or disagree with.

Extract:
- Judgments, evaluations, or stances on the topic
- Causal claims ("X leads to Y")
- Prescriptive claims ("We should do X")

Do NOT extract:
- Factual statements with no evaluative content ("Africa has 54 countries")
- Procedural advice or instructions ("Start small", "Journal about it")
- Self-descriptions or persona introductions ("As a teacher, my focus is...")
- Transitional phrases ("This is a great question", "Thank you for sharing")
- Hedging without substance ("It's complex", "There are many perspectives")

Rules:
- Each statement should stand alone as a complete thought
- Keep the original wording as much as possible
- Do not merge or summarize multiple opinions into one
- If a sentence contains multiple distinct claims, split them
- Output ONLY a JSON array of strings, nothing else."""


def prompt_hash() -> str:
    """SHA256 prefix of the current extraction prompt (for provenance tracking)."""
    return hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode()).hexdigest()[:8]


def _parse_opinions(text: str) -> list[str]:
    """Parse a JSON array of strings, stripping markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    result = json.loads(text)
    if not isinstance(result, list):
        raise ValueError(f"Expected a list, got {type(result).__name__}")
    if not all(isinstance(s, str) for s in result):
        raise ValueError("Non-string element in opinions list")
    return result


def _get_question(record: dict) -> str:
    """Extract the user question from a raw record."""
    question = record.get("raw_question")
    if question:
        return question
    messages = record["request"]["messages"]
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    raise ValueError(f"No user message found in record {record.get('task_id')}")


async def _extract_one(
    question: str,
    response: str,
    config: dict,
    *,
    sem: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> list[str]:
    """Extract opinions from one response. Retries on JSON parse failure."""
    user_prompt = f"Question: {question}\n\nResponse: {response}"
    last_error = None

    async def _call() -> list[str]:
        nonlocal last_error
        for attempt in range(max_retries):
            prompt = user_prompt
            if attempt > 0:
                prompt += "\n\n[Reminder: output ONLY a valid JSON array of strings.]"
            result = await call_llm(EXTRACTION_SYSTEM_PROMPT, prompt, config)
            try:
                return _parse_opinions(result["response"]["content"])
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
        raise RuntimeError(f"Failed to parse opinions after {max_retries} retries: {last_error}")

    if sem:
        async with sem:
            return await _call()
    return await _call()


def _check_prompt_consistency(extracted_dir: Path) -> None:
    """Check that existing manifest's prompt_hash matches current prompt.

    Raises RuntimeError on mismatch (prevents mixing old/new prompt results).
    """
    manifest_path = extracted_dir / "_manifest.json"
    if not manifest_path.exists():
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    existing_hash = manifest.get("prompt_hash")
    current_hash = prompt_hash()
    if existing_hash and existing_hash != current_hash:
        raise RuntimeError(
            f"Prompt hash mismatch in {extracted_dir}: "
            f"existing={existing_hash}, current={current_hash}. "
            f"Use --force to clear and re-extract, or --output-dir to write elsewhere."
        )


def _count_valid_lines(path: Path) -> int:
    """Count lines that parse as valid JSON. Truncate file at first corrupt line."""
    if not path.exists():
        return 0
    valid_lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                break  # truncated/corrupt line — stop here
    # If we truncated, rewrite file with only valid lines
    total_lines_on_disk = sum(1 for l in open(path, encoding="utf-8") if l.strip())
    if len(valid_lines) < total_lines_on_disk:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
    return len(valid_lines)


async def extract_file(
    raw_file: Path,
    out_file: Path,
    config: dict,
    sem: asyncio.Semaphore,
    extraction_meta: dict,
) -> tuple[int, int]:
    """Extract opinions for one raw JSONL file. Resumable.

    Returns (n_ok, n_err) where n_ok counts records with opinions,
    n_err counts records that failed extraction (across all runs, not just this one).
    """
    existing = _count_valid_lines(out_file)

    with open(raw_file, encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    if existing >= len(raw_records):
        # Already complete — count errors in existing file for accurate reporting
        n_err = 0
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if "error" in json.loads(line):
                    n_err += 1
        return existing - n_err, n_err

    remaining = raw_records[existing:]
    tasks = []
    for record in remaining:
        question = _get_question(record)
        response_text = record["response"]["content"]
        tasks.append(_extract_one(question, response_text, config, sem=sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for i, (record, result) in enumerate(zip(remaining, results)):
            rec = {
                "task_id": record["task_id"],
                "call_index": existing + i,
                "persona_name": record.get("persona_name") or record.get("agent_name"),
                "raw_question": _get_question(record),
            }
            if "turn_number" in record:
                rec["turn_number"] = record["turn_number"]
            if "round_number" in record:
                rec["round_number"] = record["round_number"]

            if isinstance(result, BaseException):
                n_err += 1
                rec["opinions"] = []
                rec["error"] = str(result)
            else:
                n_ok += 1
                rec["opinions"] = result

            rec["extraction"] = extraction_meta
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if n_err:
        print(f"  ⚠ {raw_file.stem}: {n_ok} ok, {n_err} errors", flush=True)
    return n_ok + existing, n_err


async def extract_condition(
    config_path: Path,
    raw_dir: Path,
    extracted_dir: Path,
    max_concurrent: int = 50,
    force: bool = False,
) -> dict:
    """Extract opinions for all raw files in raw_dir → write to extracted_dir.

    Files are processed in parallel (bounded by max_concurrent semaphore).
    Returns a manifest dict.
    """
    # Lock: prevent two processes from extracting the same condition concurrently
    extracted_dir.mkdir(parents=True, exist_ok=True)
    lock_path = extracted_dir / "_lock"
    if lock_path.exists():
        pid = lock_path.read_text().strip()
        # Check if the PID is still alive
        try:
            os.kill(int(pid), 0)
            raise RuntimeError(
                f"Another extraction process (PID {pid}) is running for {raw_dir.name}. "
                f"If this is stale, delete {lock_path}"
            )
        except (ProcessLookupError, ValueError):
            pass  # stale lock, safe to proceed
    lock_path.write_text(str(os.getpid()))

    try:
        return await _extract_condition_inner(
            config_path, raw_dir, extracted_dir, max_concurrent, force
        )
    finally:
        lock_path.unlink(missing_ok=True)


async def _extract_condition_inner(
    config_path: Path,
    raw_dir: Path,
    extracted_dir: Path,
    max_concurrent: int,
    force: bool,
) -> dict:
    """Inner extraction logic (called with lock held)."""
    config = load_config(config_path)
    model_name = config["model"]["name"]
    current_hash = prompt_hash()

    extraction_meta = {"model": model_name, "prompt_hash": current_hash}

    if force and extracted_dir.exists():
        import shutil
        shutil.rmtree(extracted_dir)
        print(f"  {raw_dir.name}: cleared existing extraction (--force)", flush=True)

    _check_prompt_consistency(extracted_dir)

    # Read generation model from first raw record (for provenance)
    raw_files = sorted(f for f in raw_dir.glob("*.jsonl") if not f.name.startswith("_"))
    generation_model = None
    if raw_files:
        with open(raw_files[0], encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                generation_model = json.loads(first_line)["request"]["model"]

    sem = asyncio.Semaphore(max_concurrent)

    async def _process_file(raw_file: Path) -> tuple[str, int, int]:
        out_file = extracted_dir / raw_file.name
        n_ok, n_err = await extract_file(raw_file, out_file, config, sem, extraction_meta)
        return raw_file.stem, n_ok, n_err

    tasks = [_process_file(f) for f in raw_files]
    results = await asyncio.gather(*tasks)

    total_ok = sum(n_ok for _, n_ok, _ in results)
    total_err = sum(n_err for _, _, n_err in results)

    print(
        f"  {raw_dir.name}: {len(raw_files)} files, {total_ok} ok, {total_err} errors.",
        flush=True,
    )

    # Verify extraction completeness
    verify_errors = _verify_extraction(raw_dir, extracted_dir)
    if verify_errors:
        for e in verify_errors:
            print(f"  ✗ {e}", flush=True)
    else:
        print(f"  ✓ {raw_dir.name}: extraction verified.", flush=True)

    # Derive project-relative config path for portability
    # raw_dir is results/raw/{condition}, so .parent.parent.parent is project root
    try:
        project_root = raw_dir.resolve().parent.parent.parent
        config_path_str = str(config_path.resolve().relative_to(project_root))
    except ValueError:
        config_path_str = str(config_path)

    manifest = {
        "condition": raw_dir.name,
        "generation_model": generation_model,
        "extraction_model": model_name,
        "prompt_hash": current_hash,
        "prompt_text": EXTRACTION_SYSTEM_PROMPT,
        "config_path": config_path_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_files": len(raw_files),
        "total_records": total_ok + total_err,
        "n_errors": total_err,
        "verify_errors": verify_errors,
    }

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with open(extracted_dir / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def _verify_extraction(raw_dir: Path, extracted_dir: Path) -> list[str]:
    """Check that every raw file has a corresponding extracted file with matching valid record count."""
    errors = []
    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        if raw_file.name.startswith("_"):
            continue
        ext_file = extracted_dir / raw_file.name
        if not ext_file.exists():
            errors.append(f"{raw_file.stem}: no extracted file")
            continue
        with open(raw_file, encoding="utf-8") as f:
            n_raw = sum(1 for line in f if line.strip())
        n_ext = 0
        with open(ext_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if "opinions" not in rec:
                        errors.append(f"{raw_file.stem}: record missing 'opinions' field")
                    n_ext += 1
                except json.JSONDecodeError:
                    errors.append(f"{raw_file.stem}: corrupt JSON line in extracted file")
        if n_ext != n_raw:
            errors.append(f"{raw_file.stem}: raw={n_raw} extracted={n_ext}")
    return errors
