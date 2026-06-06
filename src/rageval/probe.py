"""LLM-knowledge probing.

For each QA:
  1. Ask the BASE LLM (no RAG, no chunks) to answer just from its weights.
  2. Use a judge LLM to compare the candidate answer to the ground truth.
  3. Set `qa.llm_knowledge = "known" | "unknown"`.

This is the KG-PoC track's "AXIS-3" automation. The result lets us focus the
dataset on questions the LLM *cannot* answer without retrieval — those are the
ones where KG-RAG can plausibly add value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .llm import LLMError, _parse_json, generate as llm_generate
from .prompts import load_prompt
from .schema import QAItem

DEFAULT_PROMPT = "prompts/probe.md"


_SECTION_RE = re.compile(
    r"^##\s*\[(?P<name>[A-Z_]+)\]\s*\n(?P<body>.*?)(?=^##\s*\[|^---\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _split_sections(body: str) -> dict[str, str]:
    return {m.group("name"): m.group("body").strip() for m in _SECTION_RE.finditer(body)}


LLMCaller = Callable[..., Any]


def _parse_json_safely(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise LLMError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def probe_qa(
    qa: QAItem,
    *,
    probe_model: str,
    judge_model: str,
    probe_section: str,
    judge_section: str,
    llm: LLMCaller,
) -> str:
    """Return 'known' or 'unknown' for this QA."""
    # Step 1: Ask probe model to answer from weights alone
    probe_prompt = probe_section.replace("{question}", qa.question)
    try:
        raw = llm(prompt=probe_prompt, model=probe_model, temperature=0.0, max_tokens=1024)
        candidate = _parse_json_safely(raw).get("answer", "").strip()
    except (LLMError, json.JSONDecodeError) as e:
        print(f"[probe] {qa.qa_id} probe failed: {e}")
        return "unknown"  # be conservative: assume unknown if probe broke

    if not candidate or candidate == "不明":
        return "unknown"

    # Step 2: Ask judge model whether candidate matches ground truth
    judge_prompt = (
        judge_section
        .replace("{ground_truth}", qa.answer)
        .replace("{candidate}", candidate)
    )
    try:
        raw = llm(prompt=judge_prompt, model=judge_model, temperature=0.0, max_tokens=512)
        data = _parse_json_safely(raw)
        verdict = data.get("match", "").strip().lower()
        if verdict in ("known", "unknown"):
            return verdict
    except (LLMError, json.JSONDecodeError) as e:
        print(f"[probe] {qa.qa_id} judge failed: {e}")
    return "unknown"


def probe_batch(
    in_path: Path,
    *,
    probe_model: str,
    judge_model: str,
    prompt_path: Path = Path(DEFAULT_PROMPT),
    llm: Optional[LLMCaller] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Probe every QA in a JSONL; write back with llm_knowledge field set.

    If out_path is None, overwrites in_path. Otherwise writes to out_path.
    """
    _, body = load_prompt(str(prompt_path))
    sections = _split_sections(body)
    probe_section = sections.get("PROBE", "")
    judge_section = sections.get("JUDGE_MATCH", "")
    if not probe_section or not judge_section:
        raise RuntimeError(f"prompts/probe.md missing PROBE or JUDGE_MATCH section")

    caller = llm or llm_generate

    qas: list[QAItem] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    n_known = 0
    n_unknown = 0
    for qa in qas:
        verdict = probe_qa(
            qa,
            probe_model=probe_model,
            judge_model=judge_model,
            probe_section=probe_section,
            judge_section=judge_section,
            llm=caller,
        )
        qa.llm_knowledge = verdict  # type: ignore[assignment]
        if verdict == "known":
            n_known += 1
        else:
            n_unknown += 1
        print(f"[probe] {qa.qa_id} → {verdict}")

    out = out_path or in_path
    with out.open("w", encoding="utf-8") as f:
        for qa in qas:
            f.write(qa.model_dump_json() + "\n")

    print(f"[probe] done: {n_known} known / {n_unknown} unknown → {out}")
    return out
