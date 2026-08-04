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

from .evidence import normalize_doc_id
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
) -> tuple[Optional[bool], str]:
    """Return (sufficient, judge_raw) for hiding one evidence unit's rationale entries.

    sufficient=None は「判定できなかった」(障害) を意味し、True/False の
    どちらとも異なる第3の状態として扱う。

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
    # NOTE: judge が落ちた場合は None を返す (呼び出し側が errors に積む)。
    # False (= この根拠は必須) に倒すと、障害がそのまま本丸認定になってしまう。

    prompt = (
        judge_section
        .replace("{question}", qa.question)
        .replace("{answer}", qa.answer)
        .replace("{hidden_doc_id}", hidden_doc_id)
        .replace("{hidden_text}", _format_rationale_block(hidden))
        .replace("{remaining_text}", _format_rationale_block(remaining))
    )
    try:
        raw = llm(prompt=prompt, model=judge_model, temperature=0.0, max_tokens=4096)
        data = _parse_json_safely(raw)
        if "sufficient" not in data:
            print(f"[answer-grounding] {qa.qa_id} judge payload に sufficient が無い "
                  f"({hidden_doc_id}): {str(data)[:100]}")
            return None, f"(judge unusable: {str(data)[:100]})"
        sufficient = bool(data["sufficient"])
        return sufficient, json.dumps(data, ensure_ascii=False)
    except (LLMError, json.JSONDecodeError) as e:
        print(f"[answer-grounding] {qa.qa_id} judge failed for {hidden_doc_id}: {e}")
        # 判定不能。False (= この根拠は必須) に倒すと障害が本丸認定に化けるので
        # None を返し、呼び出し側で errors に積む。
        return None, f"(judge error: {e})"


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
    errors: list[str] = []
    raw_parts: list[str] = []
    for d in doc_ids:
        sufficient, raw = judge_one_doc(
            qa, d,
            judge_section=judge_section,
            judge_model=judge_model,
            llm=llm,
        )
        if sufficient is None:
            errors.append(d)
        else:
            per_doc[d] = sufficient
        raw_parts.append(f"{d}: {raw}")
    return AnswerGrounding(
        per_doc_sufficient=per_doc,
        errors=errors,
        judge_model=judge_model,
        judge_raw="\n".join(raw_parts),
        verified_at=datetime.now(),
    )


def is_all_docs_required(ag: AnswerGrounding) -> bool:
    """全 doc が「伏せたら answer が決まらない」(= 全 doc 必須) と判定された場合のみ True。

    per_doc_sufficient[d] = True は「d を伏せても残りで answer が決まる」(= d は不要)。
    本丸の必要条件は「どの doc を伏せても answer が決まらない」= 全 doc 値が False。

    判定できなかった単位が1つでもある (errors) 場合は False。障害を「必須」と
    読み替えて本丸を水増ししないため。
    """
    if not ag.per_doc_sufficient or ag.errors:
        return False
    return not any(ag.per_doc_sufficient.values())


# leave-one-out で「全根拠必須」が不成立のときに降格対象となるクエリ型。
# (single_fact は元々単一根拠の設計なので対象外)
_DEMOTABLE_QUERY_TYPES = {"multi_hop", "traceability", "aggregation", "negation_exhaustive"}


def demote_single_unit(qa: QAItem) -> bool:
    """根拠単位が1つしかない複数根拠型 (multi_hop 等) を single_fact に降格する。

    leave-one-out は根拠単位が2つ以上ないと検査できないため、単位1つの
    multi_hop 表示はここで機械的に実態へ合わせる (決定論・LLM不要)。
    """
    # KG 経路由来の QA は型が素材の構造で決まっており自己申告でない。
    # 特に negation_exhaustive は除外条項1チャンクが正規の形なので、
    # 「根拠単位が1つ=降格」の規則を適用しない (2026-08-04: v3 の negation
    # 5問が single_fact に誤変換された事故で追加)。
    if qa.kg_path is not None:
        return False
    units = list(dict.fromkeys(r.evidence_unit() for r in qa.rationale))
    if len(units) >= 2 or qa.kg_query_type not in _DEMOTABLE_QUERY_TYPES:
        return False
    old = qa.kg_query_type
    qa.kg_query_type = "single_fact"
    note = (f"[降格 {datetime.now():%Y-%m-%d}] 根拠単位が1つのため {old}→single_fact "
            f"(複数根拠を要する型の表示と実態の矛盾)")
    qa.difficulty_rationale = (qa.difficulty_rationale + " / " + note).strip(" /")
    return True


def required_unit_count(ag: AnswerGrounding) -> int:
    """伏せると answer が決まらなくなる根拠単位の数 (= 本当に必要な根拠の数)。"""
    return sum(1 for v in ag.per_doc_sufficient.values() if not v)


def demote_if_not_required(qa: QAItem) -> bool:
    """leave-one-out の結果、実際に必要な根拠が1つ以下なら single_fact に降格する。

    「multi_hop を名乗るが実質1チャンクで解ける」問い (plant_v15 で 13/25) を
    ラベルごと実態に合わせる。

    2026-08-01: 以前は「1単位でも不要と判定されたら降格」だったため、3根拠中
    1つが冗長なだけで残り2つが必須の問い (正真正銘のマルチホップ) まで
    single_fact に落ちていた。**必須単位が2つ以上残るなら型は維持**する。
    判定できなかった単位がある場合は降格しない (根拠不足のまま型を変えない)。
    降格の履歴は difficulty_rationale に残す。Returns True if demoted.
    """
    # KG 経路由来の QA には LOO 降格を適用しない (2026-08-01, 実測 17/18 誤降格)。
    # 連鎖型は回答の固有値が終端チャンクに集中する = 終端だけで回答が導出できる
    # ため、LOO は始点側を「不要」と判定する。しかし始点側の価値は検索上の必要性
    # (規定の所在の特定) であり、型の確定は rag-verify 後の
    # rag_verify.sync_kg_query_type_from_retrieval (終端が top-k に直接入るか) が行う。
    if qa.kg_path is not None:
        return False
    ag = qa.answer_grounding
    if ag is None or not ag.per_doc_sufficient or len(ag.per_doc_sufficient) < 2:
        return False
    if ag.errors:
        return False
    n_required = required_unit_count(ag)
    if n_required >= 2:
        return False  # 実態としてまだ複数根拠を要する
    if qa.kg_query_type not in _DEMOTABLE_QUERY_TYPES:
        return False
    old = qa.kg_query_type
    qa.kg_query_type = "single_fact"
    note = (f"[降格 {datetime.now():%Y-%m-%d}] leave-one-out で必要な根拠が "
            f"{n_required} 件だったため {old}→single_fact")
    qa.difficulty_rationale = (qa.difficulty_rationale + " / " + note).strip(" /")
    return True


def sync_retrieval_difficulty(qa: QAItem) -> bool:
    """rationale の実態から multi_doc / multi_chunk を計算し直す。

    降格しても kg_query_type しか変わらず、retrieval_difficulty が
    生成時のまま古い値で残る問題への対応。
    """
    # doc_id 欄に chunk_id が書かれることがあるので必ず正規化する。
    # 正規化を忘れると同一文書の2チャンクを別文書と数え multi_doc=True を誤付与する。
    docs = {normalize_doc_id(r.doc_id) for r in qa.rationale}
    units = {r.evidence_unit() for r in qa.rationale}
    rd = qa.retrieval_difficulty
    before = (rd.multi_doc, rd.multi_chunk)
    rd.multi_doc = len(docs) >= 2
    rd.multi_chunk = len(units) >= 2
    return before != (rd.multi_doc, rd.multi_chunk)


def grounding_batch(
    in_path: Path,
    *,
    judge_model: str,
    prompt_path: Path | None = None,
    llm: LLMCaller | None = None,
    out_path: Path | None = None,
    demote: bool = False,
) -> Path:
    """Run AnswerGrounding for every QA in a JSONL file.

    out_path が in_path と同じなら in-place 更新 (先に全読みしてから書く)。
    demote=True なら全根拠必須が不成立の複数根拠 QA を single_fact に降格する。
    """
    prompt_path = prompt_path or Path(DEFAULT_PROMPT)
    _, body = load_prompt(str(prompt_path))
    sections = _split_sections(body)
    if "GROUND_CHECK" not in sections:
        raise RuntimeError(f"GROUND_CHECK section not found in {prompt_path}")
    judge_section = sections["GROUND_CHECK"]

    llm = llm or llm_generate
    out_path = out_path or in_path.with_name(in_path.stem + ".grounded.jsonl")

    qas: list[QAItem] = []
    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    total = 0
    all_required = 0
    single_doc = 0
    demoted = 0
    n_failed = 0
    for qa in qas:
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
            if demote and demote_single_unit(qa):
                demoted += 1
        else:
            qa.answer_grounding = grounding_for_qa(
                qa,
                judge_section=judge_section,
                judge_model=judge_model,
                llm=llm,
            )
            if qa.answer_grounding.errors:
                n_failed += 1
            elif is_all_docs_required(qa.answer_grounding):
                all_required += 1
            elif demote and demote_if_not_required(qa):
                demoted += 1
        if demote:
            sync_retrieval_difficulty(qa)
        total += 1
        mark = "*" if qa.answer_grounding and is_all_docs_required(qa.answer_grounding) else " "
        err = f" errors={qa.answer_grounding.errors}" if qa.answer_grounding.errors else ""
        print(f"[answer-grounding] [{total}] {mark} {qa.qa_id} "
              f"per_doc={qa.answer_grounding.per_doc_sufficient}{err}")

    with out_path.open("w", encoding="utf-8") as fout:
        for qa in qas:
            fout.write(qa.model_dump_json() + "\n")
    print(f"[answer-grounding] done: total={total} all_required={all_required} "
          f"single_doc={single_doc} demoted={demoted} → {out_path}")
    if n_failed:
        print(f"[answer-grounding] WARNING: {n_failed} 件で judge が判定できず、"
              f"本丸 (全根拠必須) とは認めていません")
    return out_path
