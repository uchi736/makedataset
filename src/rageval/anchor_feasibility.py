"""Anchor pair feasibility judge (pre-generation gate).

For each anchor pair, ask a judge LLM: "can we write a QA where both docs are
required to reproduce answer's key tokens?" — return {feasible, planned_tokens, reason}.

Use this before generate.py's actual QA generation so we don't waste LLM cycles
on hopeless pairs (list-decoy partners, unrelated topics, etc.).

Design goal (v1.0): conservative bias — false negatives are cheap (skip a pair),
false positives are expensive (waste generation + AnswerGrounding calls +
poison the dataset if downstream filters miss).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm import LLMError, _parse_json, generate as llm_generate
from .prompts import load_prompt

DEFAULT_PROMPT = "prompts/anchor_feasibility.md"

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


def judge_pair(
    anchor_a: dict,
    anchor_b: dict,
    *,
    judge_section: str,
    judge_model: str,
    llm: LLMCaller,
) -> dict:
    """Return {'feasible': bool, 'planned_answer_tokens': list[str], 'reason': str}.

    anchor_a/b are dicts with keys: doc_id, text, references (list of str).
    """
    prompt = (
        judge_section
        .replace("{doc_id_a}", str(anchor_a["doc_id"]))
        .replace("{refs_a}", ", ".join(anchor_a.get("references", [])))
        .replace("{text_a}", anchor_a["text"])
        .replace("{doc_id_b}", str(anchor_b["doc_id"]))
        .replace("{refs_b}", ", ".join(anchor_b.get("references", [])))
        .replace("{text_b}", anchor_b["text"])
    )
    try:
        raw = llm(prompt=prompt, model=judge_model, temperature=0.0, max_tokens=512)
        data = _parse_json_safely(raw)
        return {
            "feasible": bool(data.get("feasible", False)),
            "planned_answer_tokens": data.get("planned_answer_tokens", []) or [],
            "reason": data.get("reason", "").strip(),
        }
    except (LLMError, json.JSONDecodeError) as e:
        print(f"[feasibility] judge failed: {e}")
        # Conservative: treat judge errors as not feasible (skip pair)
        return {"feasible": False, "planned_answer_tokens": [], "reason": f"(judge error: {e})"}


def feasibility_batch(
    pairs_path: Path,
    *,
    judge_model: str,
    prompt_path: Path | None = None,
    llm: LLMCaller | None = None,
    out_path: Path | None = None,
) -> Path:
    """Read anchor pairs JSONL, run feasibility judge on each, write results."""
    prompt_path = prompt_path or Path(DEFAULT_PROMPT)
    _, body = load_prompt(str(prompt_path))
    sections = _split_sections(body)
    if "FEASIBILITY_CHECK" not in sections:
        raise RuntimeError(f"FEASIBILITY_CHECK section not found in {prompt_path}")
    judge_section = sections["FEASIBILITY_CHECK"]

    llm = llm or llm_generate
    out_path = out_path or pairs_path.with_name(pairs_path.stem + ".feasibility.jsonl")

    total = 0
    feasible_count = 0
    with pairs_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            if len(pair.get("anchors", [])) < 2:
                continue
            a, b = pair["anchors"][0], pair["anchors"][1]
            verdict = judge_pair(
                a, b,
                judge_section=judge_section,
                judge_model=judge_model,
                llm=llm,
            )
            pair["feasibility"] = verdict
            pair["feasibility_judge_model"] = judge_model
            total += 1
            if verdict["feasible"]:
                feasible_count += 1
            mark = "*" if verdict["feasible"] else " "
            tokens = ", ".join(verdict["planned_answer_tokens"][:3])
            print(f"[feasibility] [{total}] {mark} {pair.get('pair_id', '?')} feasible={verdict['feasible']} tokens=[{tokens}] :: {verdict['reason'][:80]}")
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[feasibility] done: total={total} feasible={feasible_count} → {out_path}")
    return out_path
