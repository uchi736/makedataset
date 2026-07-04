"""AnswerGrounding: leave-one-out judging.

For each *evidence unit* in a QA's rationale, ask a judge LLM:
  "If we hide this unit's rationale text(s), can the remaining rationale alone
   reproduce the answer's key facts?"

Evidence unit = rationale.chunk_id when present, else rationale.doc_id.
Multi-doc corpora (plant) keep doc-level units for old data; single-doc
corpora (モデル就業規則) set chunk_id so leave-one-out operates per chunk.

If yes for *some* unit → that unit is unnecessary (per_doc_sufficient[u] = True).
If yes for *all* units → the answer is single-unit-reproducible → not 本丸.

We persist `qa.answer_grounding` to mark whether this QA passed the
leave-one-out test (i.e. every unit returned sufficient=False).
NOTE: the persisted field keeps its historical name `per_doc_sufficient`
but is keyed by evidence unit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .llm import LLMError, _parse_json, generate as llm_generate
from .prompts import load_prompt
from .schema import AnswerGrounding, QAItem

DEFAULT_PROMPT = "prompts/answer_grounding.md"

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


def _format_rationale_block(rationale_list: list, indent: str = "  ") -> str:
    """rationale を判定単位ごとに人間可読の塊で出す。"""
    if not rationale_list:
        return "(none)"
    out = []
    for r in rationale_list:
        unit = r.evidence_unit()
        out.append(f"{indent}- [doc_id={unit}] {r.text}")
    return "\n".join(out)


def judge_one_doc(
    qa: QAItem,
    hidden_doc_id: str,
    *,
    judge_section: str,
    judge_model: str,
    llm: LLMCaller,
) -> tuple[bool, str]:
    """Return (sufficient, judge_raw) for hiding one evidence unit's rationale entries.

    hidden_doc_id is an evidence unit key (chunk_id when rationale entries carry
    one, else doc_id — see Rationale.evidence_unit).
    sufficient = True means the remaining rationale alone can reproduce the answer
    (= hidden unit was unnecessary).
    """
    hidden = [r for r in qa.rationale if r.evidence_unit() == hidden_doc_id]
    remaining = [r for r in qa.rationale if r.evidence_unit() != hidden_doc_id]

    if not hidden:
        # nothing to hide → trivially sufficient (the doc wasn't there)
        return True, "(no rationale for this doc_id)"
    if not remaining:
        # only this doc → hiding it leaves nothing → not sufficient
        return False, "(no remaining rationale after hiding this doc)"

    prompt = (
        judge_section
        .replace("{question}", qa.question)
        .replace("{answer}", qa.answer)
        .replace("{hidden_doc_id}", hidden_doc_id)
        .replace("{hidden_text}", _format_rationale_block(hidden))
        .replace("{remaining_text}", _format_rationale_block(remaining))
    )
    try:
        raw = llm(prompt=prompt, model=judge_model, temperature=0.0, max_tokens=512)
        data = _parse_json_safely(raw)
        sufficient = bool(data.get("sufficient", False))
        return sufficient, json.dumps(data, ensure_ascii=False)
    except (LLMError, json.JSONDecodeError) as e:
        print(f"[answer-grounding] {qa.qa_id} judge failed for {hidden_doc_id}: {e}")
        # Conservative: if judge breaks, treat as not sufficient (= keep doc as needed)
        return False, f"(judge error: {e})"


def grounding_for_qa(
    qa: QAItem,
    *,
    judge_section: str,
    judge_model: str,
    llm: LLMCaller,
) -> AnswerGrounding:
    """Run leave-one-out judging for every unique evidence unit in qa.rationale."""
    doc_ids = list(dict.fromkeys(r.evidence_unit() for r in qa.rationale))
    per_doc: dict[str, bool] = {}
    raw_parts: list[str] = []
    for d in doc_ids:
        sufficient, raw = judge_one_doc(
            qa, d,
            judge_section=judge_section,
            judge_model=judge_model,
            llm=llm,
        )
        per_doc[d] = sufficient
        raw_parts.append(f"{d}: {raw}")
    return AnswerGrounding(
        per_doc_sufficient=per_doc,
        judge_model=judge_model,
        judge_raw="\n".join(raw_parts),
        verified_at=datetime.now(),
    )


def is_all_docs_required(ag: AnswerGrounding) -> bool:
    """全 doc が「伏せたら answer が決まらない」(= 全 doc 必須) と判定された場合のみ True。

    per_doc_sufficient[d] = True は「d を伏せても残りで answer が決まる」(= d は不要)。
    本丸の必要条件は「どの doc を伏せても answer が決まらない」= 全 doc 値が False。
    """
    if not ag.per_doc_sufficient:
        return False
    return not any(ag.per_doc_sufficient.values())


def grounding_batch(
    in_path: Path,
    *,
    judge_model: str,
    prompt_path: Path | None = None,
    llm: LLMCaller | None = None,
    out_path: Path | None = None,
) -> Path:
    """Run AnswerGrounding for every QA in a JSONL file. Writes a new JSONL
    with qa.answer_grounding populated."""
    prompt_path = prompt_path or Path(DEFAULT_PROMPT)
    _, body = load_prompt(str(prompt_path))
    sections = _split_sections(body)
    if "GROUND_CHECK" not in sections:
        raise RuntimeError(f"GROUND_CHECK section not found in {prompt_path}")
    judge_section = sections["GROUND_CHECK"]

    llm = llm or llm_generate
    out_path = out_path or in_path.with_name(in_path.stem + ".grounded.jsonl")

    total = 0
    all_required = 0
    single_doc = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            qa = QAItem.model_validate_json(line)
            doc_ids = list(dict.fromkeys(r.evidence_unit() for r in qa.rationale))
            if len(doc_ids) < 2:
                # nothing to test; mark all units as required vacuously
                qa.answer_grounding = AnswerGrounding(
                    per_doc_sufficient={d: False for d in doc_ids} if doc_ids else {},
                    judge_model=judge_model,
                    judge_raw="(single evidence unit; leave-one-out skipped)",
                    verified_at=datetime.now(),
                )
                single_doc += 1
            else:
                qa.answer_grounding = grounding_for_qa(
                    qa,
                    judge_section=judge_section,
                    judge_model=judge_model,
                    llm=llm,
                )
                if is_all_docs_required(qa.answer_grounding):
                    all_required += 1
            total += 1
            mark = "*" if qa.answer_grounding and is_all_docs_required(qa.answer_grounding) else " "
            print(f"[answer-grounding] [{total}] {mark} {qa.qa_id} per_doc={qa.answer_grounding.per_doc_sufficient}")
            fout.write(qa.model_dump_json() + "\n")
    print(f"[answer-grounding] done: total={total} all_required={all_required} single_doc={single_doc} → {out_path}")
    return out_path
