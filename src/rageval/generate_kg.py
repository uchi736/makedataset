"""KG の検証済み経路から QA を生成する (経路起点・段階分割)。

従来 (generate.py) は「2チャンクを渡して関係を推測させる」方式だったが、
対象コーパスの KG が既にあるので**経路を先に確定してから問いを書かせる**。

生成は1呼び出し1判断に分割する (2026-08-01 設計。1プロンプトに判定と作問を
積むと Gemma 26B 級では守られないことが実測で判明したため):

  段0: 辺ごとの含意判定 (kg.verify_path_provenance)
  段1: 経路の合成判定 — 適用範囲まで含めて1本の関係として成り立つかを判定し、
       関係文を書かせる。QA は書かせない (compose_path_relation)
  段2: QA 作成 — 検証済みの関係文を渡して問いを書かせるだけ。
       妥当性の判断はさせない
  段3: 機械検品 — 共通規則 + 型別規則のテーブル (run_checks)。
       型ごとに正反対の規則がある (multi_hop は中間露出を禁じ、traceability は
       要求する / multi_hop は分散を棄却し、aggregation は要求する) ため、
       ハードコードではなくテーブルで持つ

失敗は層で数える: 段1 reject = 素材不良 / 段2失敗 = 生成不良 / 段3 = 検品落ち。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from . import kg
from .chunker import load_chunks
from .evidence import classify_evidence_structure, normalize
from .generate import _coerce_to_dict, _stable_qa_id, detect_fabricated_terms
from .llm import LLMError, generate as llm_generate
from .prompts import load_prompt, render
from .schema import (
    Chunk,
    GenerationInfo,
    KGPathEdgeProvenance,
    KGPathProvenance,
    QAItem,
)

# 段1+段2 が安定するまでの比較基準として prompts/generate_kg_path.md (一体型) を残す
COMPOSE_PROMPT = "prompts/kg_compose_chain.md"
WRITE_PROMPT = "prompts/kg_write_qa.md"
COMPOSE_STAR_PROMPT = "prompts/kg_compose_star.md"
WRITE_STAR_PROMPT = "prompts/kg_write_qa_star.md"
WRITE_TRACE_PROMPT = "prompts/kg_write_qa_trace.md"
COMPOSE_EXCLUSION_PROMPT = "prompts/kg_compose_exclusion.md"
WRITE_NEGATION_PROMPT = "prompts/kg_write_qa_negation.md"

LLMCaller = Callable[..., Any]


# ---------------- プロンプト素材の整形 ----------------

def _format_path_block(path: kg.KGPath) -> str:
    lines = ["```"]
    lines.append(path.as_text())
    lines.append("```")
    lines.append("")
    for i, e in enumerate(path.edges, 1):
        lines.append(f"- 辺{i}: 「{e.subject}」 --{e.predicate}--> 「{e.object}」"
                     f"（{e.extraction_count} 箇所から抽出）")
    lines.append("")
    lines.append(f"**始点**: {path.nodes[0]} / **終端**: {path.nodes[-1]}")
    return "\n".join(lines)


def _format_evidence_block(
    path: kg.KGPath, support: list[str], chunks_by_id: dict[str, Chunk],
) -> str:
    lines: list[str] = []
    for i, (e, cid) in enumerate(zip(path.edges, support), 1):
        c = chunks_by_id.get(cid)
        if c is None:
            continue
        section = " > ".join(c.section_path) if c.section_path else "(no section)"
        lines.append(f"## 辺{i} の原文 — doc_id: {c.doc_id} / chunk_id: {c.chunk_id} "
                     f"[section: {section}]")
        lines.append(f"_この原文が「{e.subject} --{e.predicate}--> {e.object}」の根拠とされている_")
        lines.append("```")
        lines.append(c.text)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# ---------------- 経路の構造から仮ラベルを決める ----------------

def path_query_type(path: kg.KGPath) -> str:
    """経路の構造で生成タイプの**仮ラベル**を決める (確定は rag-verify の検索必要性)。"""
    if len(path.edges) >= 2 and path.chunks_are_disjoint():
        return "multi_hop"
    # 注意: KGQueryType に cross_doc_reference は無い (2026-08-03 に decoy が
    # この値でスキーマ不適合48件を出した)。出典が互いに素でない経路は
    # 1チャンクで答えうるので single_fact に寄せる。
    return "single_fact"


_PROCEDURAL_RELATIONS = frozenset({
    "REQUIRES", "REQUIRED_FOR", "HAS_STEP", "FOLLOWS", "PRECEDES",
    "CAUSES", "CAUSED_BY", "PREVENTS", "VALIDATES", "CONTROLS",
    "MONITORS", "PRODUCES", "PRODUCED_BY", "PASSES_THROUGH",
})


def path_novelty(path: kg.KGPath) -> str:
    """手順・順序・依存の経路だけを procedural_relation とする。"""
    predicates = {edge.predicate for edge in path.edges}
    if predicates and predicates <= _PROCEDURAL_RELATIONS:
        return "procedural_relation"
    return "unknown_relation"


# ---------------- 段3: 機械検品 (共通 + 型別テーブル) ----------------

@dataclass
class CheckContext:
    qa: QAItem
    path: kg.KGPath
    support: list[str]
    chunks_by_id: dict[str, Chunk]
    # 語彙漏れ検品用 (無ければその検品はスキップ)。コーパス全体の BM25 索引と、
    # chunk_id → 索引位置の対応。
    bm25: Optional[object] = None
    chunk_row: Optional[dict[str, int]] = None
    # 段1の関係文 (丸写し検品用)
    relation: str = ""


CheckFn = Callable[[CheckContext], Optional[str]]


def _rationale_validation_error(
    qa: QAItem, support: list[str], chunks_by_id: dict[str, Chunk],
) -> Optional[str]:
    """KG 辺ごとの根拠チャンクを、逐語引用が過不足なく覆うか確認する。"""
    required = set(support)
    covered: set[str] = set()
    for rationale in qa.rationale:
        cid = rationale.chunk_id
        if not cid or cid not in required:
            return f"経路外または未指定の rationale.chunk_id: {cid!r}"
        chunk = chunks_by_id[cid]
        if rationale.doc_id != chunk.doc_id:
            return f"rationale の doc_id 不一致: {rationale.doc_id!r} != {chunk.doc_id!r}"
        quote = normalize(rationale.text)
        if not quote or quote not in normalize(chunk.text):
            return f"rationale.text が {cid} の逐語引用でない"
        covered.add(cid)
    missing = required - covered
    if missing:
        return f"経路の根拠チャンクが rationale に無い: {sorted(missing)}"
    return None


def _rationale_cross_overlap_error(
    qa: QAItem, support: list[str], chunks_by_id: dict[str, Chunk],
) -> Optional[str]:
    """引用の本文が**他の辺の根拠チャンクにも**載っていないか確かめる。

    チャンカーの重複 (overlap 100字) により、同じ文が隣接2チャンクに載る。
    その文を両方の根拠として引くと、チャンクIDこそ違うが実体は1文の
    自作自演になる。各引用は自分の根拠チャンク**だけ**に現れることを求める。
    """
    others = {cid: normalize(chunks_by_id[cid].text)
              for cid in support if cid in chunks_by_id}
    for rationale in qa.rationale:
        quote = normalize(rationale.text)
        if not quote:
            continue
        for cid, text in others.items():
            if cid != rationale.chunk_id and quote in text:
                return (f"引用が複数チャンクに存在 (チャンク重複由来): "
                        f"{rationale.chunk_id} の引用が {cid} にも一致")
    return None


def _vacuous_answer(path: kg.KGPath, answer: str) -> bool:
    """回答が終端ノード名の言い換えだけで、事実の中身が無いか。"""
    norm_answer = normalize(answer)
    terminal = path.edges[-1]
    for form in (terminal.object_forms or [terminal.object]):
        norm_form = normalize(form)
        if norm_form and norm_form in norm_answer:
            if len(norm_answer) <= len(norm_form) + 12:
                return True
    return False


def _question_exposes_intermediate(path: kg.KGPath, question: str) -> Optional[str]:
    """中間ノードを質問に書いて1ホップ化していないか確認する。"""
    if len(path.edges) < 2:
        return None
    edge = path.edges[0]
    forms = edge.object_forms or [edge.object]
    normalized_question = normalize(question)
    for form in forms:
        normalized_form = normalize(form)
        if len(normalized_form) >= 3 and normalized_form in normalized_question:
            return form
    return None


def _check_relation_echo(ctx: CheckContext) -> Optional[str]:
    """回答が段1の関係文の丸写しになっていないか。

    実測 (2026-08-03 aggregation): 回答に「〜が文脈的に並列されている」のような
    判定用のメタ文がそのまま流れ込んだ。関係文は素材であって回答ではない。
    """
    relation = getattr(ctx, "relation", None) or ""
    if not relation:
        return None
    if normalize(relation) and normalize(relation) in normalize(ctx.qa.answer):
        return "回答が段1の関係文の丸写し"
    return None


def _check_rationale(ctx: CheckContext) -> Optional[str]:
    return _rationale_validation_error(ctx.qa, ctx.support, ctx.chunks_by_id)


def _check_cross_overlap(ctx: CheckContext) -> Optional[str]:
    return _rationale_cross_overlap_error(ctx.qa, ctx.support, ctx.chunks_by_id)


def _check_vacuous(ctx: CheckContext) -> Optional[str]:
    if _vacuous_answer(ctx.path, ctx.qa.answer):
        return f"終端ノード名の復唱のみ: {ctx.qa.answer[:40]}"
    return None


def _check_fabricated(ctx: CheckContext) -> Optional[str]:
    anchors = [ctx.chunks_by_id[c] for c in ctx.support if c in ctx.chunks_by_id]
    bad = detect_fabricated_terms(ctx.qa.question + " " + ctx.qa.answer, anchors)
    return f"原文に無い用語: {bad}" if bad else None


def _check_intermediate_hidden(ctx: CheckContext) -> Optional[str]:
    exposed = _question_exposes_intermediate(ctx.path, ctx.qa.question)
    return f"中間ノードが質問に露出: {exposed}" if exposed else None


LEXICAL_LEAK_TOP_K = 5


def _check_lexical_leak(ctx: CheckContext) -> Optional[str]:
    """質問が終端チャンクの語彙を漏らしていないか (BM25 だけで直撃されないか)。

    ハイブリッド検索が解けない問いの条件は「質問と正解チャンクが語彙を共有せず、
    繋がりが中間チャンクにしか無い」こと。終端の見出し語を質問に書くと BM25 が
    一発で当てて評価にならない (実測: 容器の再検査問が該当)。事後スクリーニングと
    同じ物差しを生成時に前倒しする。
    """
    if ctx.bm25 is None or ctx.chunk_row is None:
        return None
    # 小さいコーパスでは「上位5位」に判別力が無い (2チャンクなら必ず上位)。
    if len(ctx.chunk_row) < 50:
        return None
    terminal = ctx.support[-1]
    row = ctx.chunk_row.get(terminal)
    if row is None:
        return None
    import numpy as np
    scores = ctx.bm25.scores(ctx.qa.question)
    rank = int((np.argsort(-scores, kind="stable") == row).argmax())
    if rank < LEXICAL_LEAK_TOP_K:
        return (f"質問が終端チャンクの語彙を漏らしている "
                f"(BM25 のみで終端が {rank + 1} 位)")
    return None


def _check_terminal_concentration(ctx: CheckContext) -> Optional[str]:
    """束ね (parallel) と、答えが始点側に載っている実質1ホップを落とす。"""
    s = classify_evidence_structure(ctx.qa)
    if s.label == "parallel":
        return f"固有値が複数根拠に分散 (束ね): {list(s.token_units)[:3]}"
    if s.label == "chained" and s.covering_unit != ctx.support[-1]:
        return (f"固有値の所在が終端でない: {s.covering_unit} "
                f"(終端は {ctx.support[-1]})")
    return None


def _check_element_coverage(ctx: CheckContext) -> Optional[str]:
    """aggregation 用: 検証済みの各要素が回答に含まれ、その要素を含む根拠があるか。

    分散が正解の型なので終端集中は求めない。代わりに
    「要素の取りこぼし」と「根拠の無い要素の水増し」をここで落とす。
    """
    norm_answer = normalize(ctx.qa.answer)
    rationale_texts = [normalize(r.text) for r in ctx.qa.rationale]
    missing = []
    unsupported = []
    for e in ctx.path.edges:
        forms = [normalize(f) for f in (e.object_forms or [e.object])]
        forms = [f for f in forms if len(f) >= 2]
        if not any(f in norm_answer for f in forms):
            missing.append(e.object)
            continue
        if not any(f in t for f in forms for t in rationale_texts):
            unsupported.append(e.object)
    if missing:
        return f"回答に無い要素: {missing[:3]}"
    if unsupported:
        return f"根拠の引用に無い要素: {unsupported[:3]}"
    return None


def _check_trace_identifiers(ctx: CheckContext) -> Optional[str]:
    """traceability: 回答に各辺の出典の識別子 (条番号等) が全て含まれるか。

    この型は「根拠の系譜」を答えさせるので、multi_hop と逆に識別子の明記が必須。
    各根拠チャンクの固有値 (第N条・様式番号・数値+単位) のいずれかが回答に
    現れることを求める。順序は要求しない — 「火技省令に適合させる必要があり、
    それは第82条が求めている」のような自然な逆順の説明文まで弾いてしまうため
    (2026-08-03 パイロットで改訂)。
    """
    from .evidence import normalize_numbers, salient_tokens

    answer_norm = normalize_numbers(ctx.qa.answer)
    for cid in ctx.support:
        chunk = ctx.chunks_by_id.get(cid)
        if chunk is None:
            continue
        toks = [t for t in salient_tokens(chunk.text) if t in answer_norm]
        if not toks:
            return f"根拠 {cid} の識別子が回答に無い"
    return None


# 適用除外の文言に加えて、明示的な禁止・不使用の文言も認める (2026-08-03 設計改訂。
# 禁止も原文明示の否定であり、KG欠損由来の捏造否定にはならない)
_EXCLUSION_WORDS = ("除く", "除き", "適用しない", "適用されない", "この限りでない",
                    "適用を受けない", "準用しない", "以外",
                    "使用してはならない", "使用しない", "用いてはならない",
                    "設けてはならない", "してはならない")


def _check_exclusion_wording(ctx: CheckContext) -> Optional[str]:
    """negation: 根拠の引用に明示的な除外文言が含まれるか。

    除外文言の無い否定は「KG に無い＝存在しない」型の捏造否定に流れるため、
    原文の明示除外だけを認める (設計書の禁止事項)。
    """
    for r in ctx.qa.rationale:
        if any(w in (r.text or "") for w in _EXCLUSION_WORDS):
            return None
    return "引用に明示的な除外文言 (除く/適用しない/この限りでない等) が無い"


# 共通規則: 全ての型に適用。逐語・重複・無内容・造語。
def _check_docid_leak(ctx: CheckContext) -> Optional[str]:
    """質問・回答に内部の doc_id (ファイル名) が露出していないか。

    実測 (2026-08-04 decoy): 質問文に `33_denji_kisoku` がそのまま出た。
    実務者はファイル名を知らないので、出た時点で問いとして成立しない。
    """
    text = (ctx.qa.question or "") + (ctx.qa.answer or "")
    ids = {c.doc_id for c in ctx.chunks_by_id.values()} if ctx.chunks_by_id else set()
    for r in ctx.qa.rationale:
        ids.add(r.doc_id)
    hits = [d for d in ids if d and d in text]
    if hits:
        return f"内部ファイル名の露出: {hits[:3]}"
    # 完全一致だけだと「v49n4_01（正式名）」のような前半だけの露出を逃す。
    # ファイル名の接頭パターン (技報: vNNnN_NN / 法令: NN_ローマ字) も検出する
    import re as _re
    m = _re.search(r"v\d+n\d+_\d+|(?<![0-9A-Za-z])\d{2}_[a-z]+", text)
    return f"内部ファイル名の露出 (接頭形): {m.group(0)}" if m else None


COMMON_CHECKS: list[tuple[str, CheckFn]] = [
    ("根拠不適合", _check_rationale),
    ("ファイル名露出", _check_docid_leak),
    ("関係文丸写し", _check_relation_echo),
    ("重複根拠", _check_cross_overlap),
    ("無内容回答", _check_vacuous),
    ("造語", _check_fabricated),
]

# 型別規則。設計書の型別定義表と1対1で対応させる。
#   multi_hop:    中間を隠す ∧ 固有値は終端集中 (分散 = 束ね = 棄却)
#   aggregation:  分散が正解なので終端集中チェックは適用しない。
#                 代わりに要素の網羅と根拠づけを要求
#   traceability: (実装時) 経路識別子の露出が必須なので中間露出チェックは適用しない
CHECKS_BY_TYPE: dict[str, list[tuple[str, CheckFn]]] = {
    "multi_hop": [
        ("中間露出", _check_intermediate_hidden),
        ("構造", _check_terminal_concentration),
        ("語彙漏れ", _check_lexical_leak),
    ],
    "single_fact": [],
    "aggregation": [
        ("要素網羅", _check_element_coverage),
    ],
    # traceability: 識別子の明記が必須 (中間露出・終端集中は適用しない)。
    # 語彙漏れ検品も適用しない — この型の合格条件は「系譜を提示できるか」であって
    # 検索困難性ではない (2026-08-03 パイロットで誤適用が 21/40 を過剰棄却)
    "traceability": [
        ("識別子系列", _check_trace_identifiers),
    ],
    # negation: 原文の明示除外だけを認める。1辺素材なので構造系の検品は無い
    "negation_exhaustive": [
        ("除外根拠", _check_exclusion_wording),
    ],
    # kg_missing: 合格条件は「KG に無い実在参照を出題できること」。
    # 中間露出 (参照先条番号=答えの漏れ) だけ見る。語彙漏れ・終端集中は
    # この系統の目的 (KG 欠損の測定) には筋違いなので適用しない
    "kg_missing": [
        ("中間露出", _check_intermediate_hidden),
    ],
}


def run_checks(ctx: CheckContext, query_type: str) -> Optional[str]:
    """共通 + 型別の検品を順に掛け、最初の違反を「規則名: 理由」で返す。"""
    for name, fn in COMMON_CHECKS + CHECKS_BY_TYPE.get(query_type, []):
        err = fn(ctx)
        if err:
            return f"{name}: {err}"
    return None


# ---------------- 段1: 経路の合成判定 ----------------

def compose_path_relation(
    path: kg.KGPath,
    support: list[str],
    chunks_by_id: dict[str, Chunk],
    *,
    model: str,
    llm: Optional[LLMCaller] = None,
    prompt_path: Path = Path(COMPOSE_PROMPT),
    allow_thinking: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """経路が適用範囲まで含めて1本の関係として成り立つかを判定させる。

    成立なら (関係文, None)、不成立なら (None, 理由)。QA はまだ書かせない。
    想定外の応答は例外を伝播する (障害を「不成立」として静かに混ぜない)。
    """
    _, body = load_prompt(str(prompt_path))
    caller = llm or llm_generate
    raw = caller(
        prompt=render(body, {
            "path_block": _format_path_block(path),
            "evidence_block": _format_evidence_block(path, support, chunks_by_id),
        }),
        model=model, temperature=0.0, max_tokens=4096,
        response_model=None, force_json=True,
        # 既定は思考オフ (2026-08-02 ユーザ判断: 速度優先)。思考オフの Qwen は
        # 委任連鎖を一部誤棄却する (金看板 3/5、思考オンで 4/5)。供給が僅少な系統
        # (kg_missing) と、環境変数 KG_COMPOSE_THINKING=1 の採掘バッチだけ思考オン
        # (2026-08-04: 「KG しか解けない」型の増産は量より質)。
        allow_thinking=allow_thinking or os.environ.get("KG_COMPOSE_THINKING") == "1",
    )
    data = _coerce_to_dict(raw)
    if type(data.get("valid")) is not bool:
        raise LLMError(f"合成判定の応答が不正: {data!r}")
    if not data["valid"]:
        return None, str(data.get("reason") or "(理由なし)").strip()
    relation = str(data.get("relation") or "").strip()
    if not relation:
        raise LLMError(f"合成判定が関係文を返さない: {data!r}")
    return relation, None


# ---------------- 段1: 星の合成判定 ----------------

def _format_star_block(star: "kg.KGStar") -> str:
    lines = ["```", star.as_text(), "```", ""]
    lines.append(f"**中心**: {star.center} / 関係: {star.relation}")
    for i, e in enumerate(star.edges, 1):
        lines.append(f"- 要素{i}: 「{e.object}」（{e.extraction_count} 箇所から抽出）")
    return "\n".join(lines)


def _format_star_evidence_block(
    star: "kg.KGStar", verified: list, chunks_by_id: dict[str, Chunk],
) -> str:
    """星の裏付け原文。チャンク単位でまとめ、どの要素の根拠かを付記する。"""
    by_chunk: dict[str, list[str]] = {}
    for item in verified:
        by_chunk.setdefault(item.source_chunk_id, []).append(item.edge.object)
    lines: list[str] = []
    for cid, elements in by_chunk.items():
        c = chunks_by_id.get(cid)
        if c is None:
            continue
        section = " > ".join(c.section_path) if c.section_path else "(no section)"
        lines.append(f"## 原文 — doc_id: {c.doc_id} / chunk_id: {c.chunk_id} "
                     f"[section: {section}]")
        lines.append(f"_この原文が要素「{'、'.join(elements)}」の根拠_")
        lines.append("```")
        lines.append(c.text)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def compose_star_relation(
    star: "kg.KGStar",
    verified: list,
    chunks_by_id: dict[str, Chunk],
    *,
    model: str,
    llm: Optional[LLMCaller] = None,
    prompt_path: Path = Path(COMPOSE_STAR_PROMPT),
) -> tuple[Optional[str], list[str], Optional[str]]:
    """星の要素が同じ文脈の1集合として成り立つか判定させる。

    (関係文, 除外要素, None) または (None, [], 理由)。
    連鎖と違い「一部の要素だけ別文脈」を許し、excluded に返させて残りで進める。
    """
    _, body = load_prompt(str(prompt_path))
    caller = llm or llm_generate
    raw = caller(
        prompt=render(body, {
            "star_block": _format_star_block(star),
            "evidence_block": _format_star_evidence_block(star, verified, chunks_by_id),
        }),
        model=model, temperature=0.0, max_tokens=4096,
        response_model=None, force_json=True,
    )
    data = _coerce_to_dict(raw)
    if type(data.get("valid")) is not bool:
        raise LLMError(f"星の合成判定の応答が不正: {data!r}")
    if not data["valid"]:
        return None, [], str(data.get("reason") or "(理由なし)").strip()
    relation = str(data.get("relation") or "").strip()
    if not relation:
        raise LLMError(f"星の合成判定が関係文を返さない: {data!r}")
    excluded = [str(x) for x in (data.get("excluded") or []) if str(x).strip()]
    return relation, excluded, None


# ---------------- 生成本体 ----------------

def _load_used_paths(out_dir: Path) -> set[str]:
    """出力先の既存バッチが使った経路 (path_text)。追い足しの重複防止。"""
    used: set[str] = set()
    for prev in out_dir.glob("*.jsonl"):
        for line in prev.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                kg_meta = json.loads(line).get("kg_path") or {}
            except json.JSONDecodeError:
                continue
            if kg_meta.get("path_text"):
                used.add(kg_meta["path_text"])
    return used


def _next_out_path(out_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kgpath_{stamp}.jsonl"
    seq = 1
    while out_path.exists():
        out_path = out_dir / f"kgpath_{stamp}_{seq}.jsonl"
        seq += 1
    return out_path


def _finalize_qa(
    data: dict,
    *,
    qt: str,
    novelty: str,
    support: list[str],
    material,                      # KGPath / KGStar (duck-typed: .edges / .as_text)
    chunks_by_id: dict[str, Chunk],
    relation: str,
    kg_snapshot: str,
    provenance_model: str,
    verified_edges: list,
    shape: str,
    model: str,
    prompt_version: str,
    bm25: Optional[object] = None,
    chunk_row: Optional[dict[str, int]] = None,
    checks_key: Optional[str] = None,
) -> tuple[Optional[QAItem], Optional[str]]:
    """段2の応答に既定値とメタを与え、検証 → 段3検品まで通す (全型共通の後段)。

    checks_key: 検品テーブルのキーを型ラベルと分けたいとき (kg_decoy 等) に指定。"""
    data.pop("path_check", None)
    data.pop("reject", None)
    data.pop("reason", None)
    data.setdefault("category", [])
    data.setdefault("aspect", [])
    data.setdefault("reasoning_complexity", {})
    data.setdefault("retrieval_difficulty", {})
    data.setdefault("source_structure", {})
    data.setdefault("explainability", {})
    data["reasoning_complexity"]["multi_step"] = True
    data["retrieval_difficulty"]["multi_chunk"] = len(set(support)) >= 2
    data["retrieval_difficulty"]["multi_doc"] = len({
        chunks_by_id[c].doc_id for c in support if c in chunks_by_id
    }) >= 2
    data["explainability"]["evidence_strictness"] = "multi-ref"
    data["retrieval_level"] = "Hard"
    data.setdefault("answer_level", "Medium")
    data.setdefault("difficulty_rationale", "")
    # Qwen は difficulty_rationale を数値で返すことがある (2026-08-03 実測17件)。
    # 文字列に寄せてスキーマ不適合での取りこぼしを防ぐ
    if not isinstance(data.get("difficulty_rationale"), str):
        data["difficulty_rationale"] = str(data["difficulty_rationale"])
    data["kg_query_type"] = qt
    data["kg_novelty"] = novelty
    data["qa_id"] = _stable_qa_id(
        data.get("question", ""),
        [chunks_by_id[c] for c in support if c in chunks_by_id],
        qt,
    )
    data["generation"] = GenerationInfo(
        model=model, prompt_version=prompt_version, generated_at=datetime.now(),
    ).model_dump(mode="json")
    data["kg_path"] = KGPathProvenance(
        snapshot_id=kg_snapshot,
        path_text=material.as_text(),
        edges=[KGPathEdgeProvenance(
            subject=item.edge.subject,
            predicate=item.edge.predicate,
            object=item.edge.object,
            source_chunk_id=item.source_chunk_id,
            extraction_count=item.edge.extraction_count,
            verification_reason=item.reason,
        ) for item in verified_edges],
        verifier_model=provenance_model,
        verified_at=datetime.now(),
        shape=shape,
        composition_reason=relation,
    ).model_dump(mode="json")
    try:
        qa = QAItem.model_validate(data)
    except ValidationError as e:
        lines = str(e).splitlines()
        return None, f"スキーマ不適合: {' | '.join(lines[1:3])[:160]}"

    # ---- 段3: 機械検品 (共通 + 型別テーブル) ----
    # 検品には検証済みの枝だけを持つ素材を渡す (星は除外枝を数えない)
    check_material = material
    if shape == "star":
        check_material = kg.KGStar(
            center=material.center, relation=material.relation,
            edges=[item.edge for item in verified_edges],
        )
    check_error = run_checks(
        CheckContext(qa=qa, path=check_material, support=support,
                     chunks_by_id=chunks_by_id, bm25=bm25, chunk_row=chunk_row,
                     relation=relation),
        checks_key or qt,
    )
    if check_error:
        return None, f"{qa.qa_id} {check_error}"

    note_bits = [f"KG経路: {material.as_text()}", f"KG版: {kg_snapshot}",
                 f"接続根拠: {relation}"]
    qa.difficulty_rationale = " / ".join(
        note_bits + ([qa.difficulty_rationale] if qa.difficulty_rationale else [])
    )
    return qa, None


def generate_from_paths(
    paths: list[kg.KGPath],
    chunks_dir: Path,
    out_dir: Path,
    *,
    model: str,
    provenance_model: str,
    kg_snapshot: str = "unknown",
    prompt_path: Path = Path(WRITE_PROMPT),
    llm: Optional[LLMCaller] = None,
    compose_llm: Optional[LLMCaller] = None,
    provenance_llm: Optional[LLMCaller] = None,
    max_items: Optional[int] = None,
    query_type_override: Optional[str] = None,
    terminal_leak_free: bool = False,
) -> Path:
    """KG 経路を段0→段1→段2→段3 の順で処理し、通過した QA を書き出す。

    段1 (合成判定) は provenance_model (生成と別モデル推奨) で行い、
    段2 (QA作成) だけを生成モデルで行う。
    query_type_override: 同じ2辺素材から別型を作るときに指定する
    (traceability は multi_hop と同素材で、問いの契約と検品だけが違う)。
    terminal_leak_free: multi_hop のみ True。段0の終端辺候補を語彙断絶チャンク
    (始点表記を含まないもの) に絞り、答えの根拠に質問語が漏れないようにする。
    """
    chunks = load_chunks(chunks_dir)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    texts = {c.chunk_id: c.text for c in chunks}

    # 語彙漏れ検品用 (質問だけの BM25 で終端が直撃されないか)。1回だけ構築。
    from .rag_verify import BM25Index
    bm25 = BM25Index(chunks)
    chunk_row = {c.chunk_id: i for i, c in enumerate(chunks)}

    meta, write_body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "kg_write_qa")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 既出経路のスキップ: 出力先に既にあるバッチが使った経路は再利用しない。
    # 経路は毎回同じ順で列挙されるため、これが無いと途中で落ちた後の
    # 追い足しが先頭から同じ問いを作り直してしまう (v1 で実際に重複が出た)。
    used_paths = _load_used_paths(out_dir)
    if used_paths:
        before = len(paths)
        paths = [p for p in paths if p.as_text() not in used_paths]
        print(f"[kg-generate] 既出経路 {before - len(paths)} 本をスキップ "
              f"(既存バッチ由来)")

    out_path = _next_out_path(out_dir)

    written = 0
    n_provenance_rejected = 0   # 段0: 辺の含意が取れない
    n_compose_rejected = 0      # 段1: 合成が成り立たない (素材不良)
    n_check_rejected = 0        # 段3: 検品落ち
    n_failed = 0                # 段2ほか: 生成不良 (通信・スキーマ)
    n_consecutive_failures = 0  # サーバ容量超過等で全滅している徴候の検出
    with out_path.open("w", encoding="utf-8") as f:
        for i_path, path in enumerate(paths, 1):
            if i_path % 5 == 0:
                print(f"[kg-generate] 進捗 {i_path}/{len(paths)} 経路 "
                      f"(書き出し {written})", flush=True)
            if max_items is not None and written >= max_items:
                break
            # ---- 段0: 辺ごとの含意 ----
            try:
                verified_edges = kg.verify_path_provenance(
                    path, texts, model=provenance_model, llm=provenance_llm,
                    terminal_leak_free=terminal_leak_free,
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                # 判定モデルの出力不良で経路1本を失うのは仕方ないが、
                # バッチ全体を道連れにしない (2026-08-05: 迷子の { で全停止した)
                print(f"[kg-generate] 段0判定失敗 (経路を跳ばす): {str(e)[:80]}")
                n_failed += 1
                n_consecutive_failures += 1
                if n_consecutive_failures >= 10:
                    # 10連続の失敗はサーバ側の障害/容量超過 (2026-08-05 実測:
                    # :8001 が 260/269 で 503)。空振りで舐め切らず打ち切る
                    print("[kg-generate] 10連続失敗 — サーバ障害とみなし打ち切り")
                    break
                continue
            if verified_edges is None:
                n_provenance_rejected += 1
                continue
            support = [edge.source_chunk_id for edge in verified_edges]
            n_consecutive_failures = 0

            # ---- 段1: 合成判定 (判断だけ) ----
            try:
                relation, reject_reason = compose_path_relation(
                    path, support, chunks_by_id,
                    model=provenance_model, llm=compose_llm,
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 合成判定失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            if relation is None:
                n_compose_rejected += 1
                print(f"[kg-generate] 合成不成立 ({path.as_text()}): "
                      f"{(reject_reason or '')[:100]}")
                continue

            # ---- 段2: QA 作成 (作文だけ) ----
            qa_caller = llm or llm_generate
            try:
                raw = qa_caller(
                    prompt=render(write_body, {
                        "relation_sentence": relation,
                        "evidence_block": _format_evidence_block(
                            path, support, chunks_by_id),
                    }),
                    model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
                data = _coerce_to_dict(raw)
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 生成失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            qa, err = _finalize_qa(
                data, qt=query_type_override or path_query_type(path),
                novelty=path_novelty(path),
                support=support, material=path, chunks_by_id=chunks_by_id,
                relation=relation, kg_snapshot=kg_snapshot,
                provenance_model=provenance_model, verified_edges=verified_edges,
                shape="chain", model=model, prompt_version=prompt_version,
                bm25=bm25, chunk_row=chunk_row,
            )
            if qa is None:
                if err and err.startswith("スキーマ不適合"):
                    print(f"[kg-generate] {err}")
                    n_failed += 1
                else:
                    print(f"[kg-generate] skip ({err})")
                    n_check_rejected += 1
                continue

            apply_term_substitution(qa, path)
            f.write(qa.model_dump_json() + "\n")
            written += 1
            docs = sorted({c.split("__")[0] for c in support})
            print(f"[kg-generate] {qa.qa_id} {qa.kg_query_type} docs={'+'.join(docs)}")

    print(f"[kg-generate] wrote {written} QAs → {out_path}")
    print(f"[kg-generate] 段0棄却 {n_provenance_rejected} / 段1棄却 {n_compose_rejected} / "
          f"段3検品落ち {n_check_rejected} / 生成失敗 {n_failed}")
    return out_path


def generate_from_stars(
    stars: list["kg.KGStar"],
    chunks_dir: Path,
    out_dir: Path,
    *,
    model: str,
    provenance_model: str,
    kg_snapshot: str = "unknown",
    prompt_path: Path = Path(WRITE_STAR_PROMPT),
    llm: Optional[LLMCaller] = None,
    compose_llm: Optional[LLMCaller] = None,
    provenance_llm: Optional[LLMCaller] = None,
    max_items: Optional[int] = None,
) -> Path:
    """星型 (aggregation) を段0→段1→段2→段3 の順で処理し、通過した QA を書き出す。

    連鎖との違い:
      - 段0 は「落ちた枝を捨てて星を生かす」(verify_star_provenance)。
        正解は KG の枝全部ではなく、原文で裏付けの取れた要素の集合
      - 段1 は文脈の同一性 (別講習の科目の混在など) を判定し、別文脈の要素を
        除外できる
      - 段3 は終端集中を求めず、要素の網羅と根拠づけを求める
    """
    chunks = load_chunks(chunks_dir)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    texts = {c.chunk_id: c.text for c in chunks}

    meta, write_body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "kg_write_qa_star")

    out_dir.mkdir(parents=True, exist_ok=True)
    used_paths = _load_used_paths(out_dir)
    if used_paths:
        before = len(stars)
        stars = [s for s in stars if s.as_text() not in used_paths]
        print(f"[kg-generate] 既出の星 {before - len(stars)} 個をスキップ")
    out_path = _next_out_path(out_dir)

    written = 0
    n_provenance_rejected = 0
    n_compose_rejected = 0
    n_check_rejected = 0
    n_failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for star in stars:
            if max_items is not None and written >= max_items:
                break
            # ---- 段0: 枝ごとの含意 (落ちた枝は捨てる) ----
            verified = kg.verify_star_provenance(
                star, texts, model=provenance_model, llm=provenance_llm,
            )
            if verified is None:
                n_provenance_rejected += 1
                continue

            # ---- 段1: 文脈の同一性判定 ----
            try:
                relation, excluded, reject_reason = compose_star_relation(
                    star, verified, chunks_by_id,
                    model=provenance_model, llm=compose_llm,
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 合成判定失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            if relation is None:
                n_compose_rejected += 1
                print(f"[kg-generate] 合成不成立 ({star.as_text()}): "
                      f"{(reject_reason or '')[:100]}")
                continue
            if excluded:
                verified = [v for v in verified if v.edge.object not in excluded]
                print(f"[kg-generate] 別文脈の要素を除外: {excluded}")
                if len(verified) < 3:
                    n_compose_rejected += 1
                    continue
            support = list(dict.fromkeys(v.source_chunk_id for v in verified))
            if len(support) < 2:
                n_compose_rejected += 1
                continue

            # ---- 段2: QA 作成 ----
            qa_caller = llm or llm_generate
            elements_block = "\n".join(
                f"- {v.edge.object}" for v in verified)
            try:
                raw = qa_caller(
                    prompt=render(write_body, {
                        "relation_sentence": relation,
                        "center": star.center,
                        "elements_block": elements_block,
                        "evidence_block": _format_star_evidence_block(
                            star, verified, chunks_by_id),
                    }),
                    model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
                data = _coerce_to_dict(raw)
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 生成失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            qa, err = _finalize_qa(
                data, qt="aggregation", novelty=path_novelty(star),
                support=support, material=star, chunks_by_id=chunks_by_id,
                relation=relation, kg_snapshot=kg_snapshot,
                provenance_model=provenance_model, verified_edges=verified,
                shape="star", model=model, prompt_version=prompt_version,
            )
            if qa is None:
                if err and err.startswith("スキーマ不適合"):
                    print(f"[kg-generate] {err}")
                    n_failed += 1
                else:
                    print(f"[kg-generate] skip ({err})")
                    n_check_rejected += 1
                continue

            apply_term_substitution(qa, star)
            f.write(qa.model_dump_json() + "\n")
            written += 1
            print(f"[kg-generate] {qa.qa_id} aggregation "
                  f"要素{len(verified)} チャンク{len(support)}")

    print(f"[kg-generate] wrote {written} QAs → {out_path}")
    print(f"[kg-generate] 段0棄却 {n_provenance_rejected} / 段1棄却 {n_compose_rejected} / "
          f"段3検品落ち {n_check_rejected} / 生成失敗 {n_failed}")
    return out_path


def generate_from_exclusions(
    exclusions: list[kg.KGPath],
    chunks_dir: Path,
    out_dir: Path,
    *,
    model: str,
    provenance_model: str,
    kg_snapshot: str = "unknown",
    prompt_path: Path = Path(WRITE_NEGATION_PROMPT),
    llm: Optional[LLMCaller] = None,
    compose_llm: Optional[LLMCaller] = None,
    provenance_llm: Optional[LLMCaller] = None,
    max_items: Optional[int] = None,
) -> Path:
    """EXCLUDES 辺 (1辺経路) から negation_exhaustive の QA を生成する。

    流れは連鎖と同じ 段0→段1→段2→段3。段1は除外専用の判定
    (明示的な除外文言と範囲の確認) を使う。素材は62本しか無いので
    max_items を欲張らない (設計書: 10問目標)。
    """
    chunks = load_chunks(chunks_dir)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    texts = {c.chunk_id: c.text for c in chunks}

    meta, write_body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "kg_write_qa_negation")

    out_dir.mkdir(parents=True, exist_ok=True)
    used_paths = _load_used_paths(out_dir)
    if used_paths:
        before = len(exclusions)
        exclusions = [p for p in exclusions if p.as_text() not in used_paths]
        print(f"[kg-generate] 既出の除外辺 {before - len(exclusions)} 本をスキップ")
    out_path = _next_out_path(out_dir)

    written = 0
    n_provenance_rejected = n_compose_rejected = n_check_rejected = n_failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in exclusions:
            if max_items is not None and written >= max_items:
                break
            verified_edges = kg.verify_path_provenance(
                path, texts, model=provenance_model, llm=provenance_llm,
            )
            if verified_edges is None:
                n_provenance_rejected += 1
                continue
            support = [e.source_chunk_id for e in verified_edges]
            try:
                relation, reject_reason = compose_path_relation(
                    path, support, chunks_by_id,
                    model=provenance_model, llm=compose_llm,
                    prompt_path=Path(COMPOSE_EXCLUSION_PROMPT),
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 除外判定失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            if relation is None:
                n_compose_rejected += 1
                print(f"[kg-generate] 除外不成立 ({path.as_text()}): "
                      f"{(reject_reason or '')[:100]}")
                continue

            qa_caller = llm or llm_generate
            try:
                raw = qa_caller(
                    prompt=render(write_body, {
                        "relation_sentence": relation,
                        "evidence_block": _format_evidence_block(
                            path, support, chunks_by_id),
                    }),
                    model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
                data = _coerce_to_dict(raw)
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 生成失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            qa, err = _finalize_qa(
                data, qt="negation_exhaustive", novelty=path_novelty(path),
                support=support, material=path, chunks_by_id=chunks_by_id,
                relation=relation, kg_snapshot=kg_snapshot,
                provenance_model=provenance_model, verified_edges=verified_edges,
                shape="exclusion", model=model, prompt_version=prompt_version,
            )
            if qa is None:
                if err and err.startswith("スキーマ不適合"):
                    print(f"[kg-generate] {err}")
                    n_failed += 1
                else:
                    print(f"[kg-generate] skip ({err})")
                    n_check_rejected += 1
                continue

            apply_term_substitution(qa, path)
            f.write(qa.model_dump_json() + "\n")
            written += 1
            print(f"[kg-generate] {qa.qa_id} negation_exhaustive | {path.as_text()[:50]}")

    print(f"[kg-generate] wrote {written} QAs → {out_path}")
    print(f"[kg-generate] 段0棄却 {n_provenance_rejected} / 段1棄却 {n_compose_rejected} / "
          f"段3検品落ち {n_check_rejected} / 生成失敗 {n_failed}")
    return out_path


# ---------------- unknown_term 変換 (型に直交する後処理) ----------------

def apply_term_substitution(qa: QAItem, material, *, rate_mod: int = 10,
                            rate_lt: int = 3) -> bool:
    """質問文中の始点ノードの正規表記を、KG の surface_forms にある別表記へ置換する。

    kg_novelty=unknown_term (表記揺れ・社内呼称) のセルを埋めるための機械変換。
    - 適用は決定論的な抽選 (qa_id の16進を rate_mod で割った剰余 < rate_lt ≒ 30%)。
      乱数を使わないので再実行しても同じ問いが同じ判定になる
    - 置換できる条件: 始点の正規表記が質問に逐語で現れ、かつ正規化しても同じに
      ならない別表記が surface_forms にあること
    - 置換の事実は kg_path.term_substitution に記録し、vocabulary_mismatch を立てる。
      検索をさらに難しくする方向の変換なので、検品後に適用してよい
    """
    try:
        draw = int(qa.qa_id.split("_")[-1], 16) % rate_mod
    except ValueError:
        return False
    if draw >= rate_lt:
        return False
    edge = material.edges[0]
    canonical = edge.subject
    if not canonical or canonical not in qa.question:
        return False
    variants = [f for f in (edge.subject_forms or [])
                if f and f != canonical and normalize(f) != normalize(canonical)
                and canonical not in f and f not in canonical]
    if not variants:
        return False
    used = sorted(variants)[0]     # 決定論的に選ぶ
    qa.question = qa.question.replace(canonical, used, 1)
    qa.kg_novelty = "unknown_term"
    qa.retrieval_difficulty.vocabulary_mismatch = True
    if qa.kg_path is not None:
        qa.kg_path.term_substitution = {"canonical": canonical, "used": used}
    qa.difficulty_rationale = (
        qa.difficulty_rationale
        + f" / [unknown_term] 質問の「{canonical}」を別表記「{used}」に置換"
    ).strip(" /")
    return True


# ---------------- KG-missing 系統 (文書検証済みの明示参照から生成) ----------------

def find_missing_reference_pairs(
    chunks: list[Chunk],
    policy,
    chunk_nodes: dict[str, set],
) -> list[tuple[Chunk, Chunk, str, int]]:
    """明示参照 (文書名の名指し + 条番号) で解決できる異文書対のうち、
    KG 上に対応する繋がり (共有ノード) が無いものを列挙する。

    これが KG のカバレッジ不足の実測値そのもの。ここから作る問いは
    「KG-RAG が落とすはずの問い」であり、KG-covered だけで測ると隠れる
    失点面を可視化する (2026-08-03 実測: 明示参照17対のうち9対が KG に無い)。
    """
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    out: list[tuple[Chunk, Chunk, str, int]] = []
    for c in chunks:
        for target_doc, cands in by_doc.items():
            if target_doc == c.doc_id:
                continue
            n = policy.named_reference(c, target_doc)
            if n is None:
                continue
            t = policy.resolve_named_target(c, cands)
            if t is None:
                continue
            if chunk_nodes.get(c.chunk_id, set()) & chunk_nodes.get(t.chunk_id, set()):
                continue    # KG に対応あり → kg_covered 側の領分
            out.append((c, t, target_doc, n))
    return out


def _alias_of(policy, doc_id: str) -> str:
    names = (getattr(policy, "doc_aliases", {}) or {}).get(doc_id) or []
    return names[0] if names else doc_id


def reference_pair_to_path(
    anchor: Chunk, target: Chunk, target_doc: str, article: int, policy,
) -> kg.KGPath:
    """明示参照の対を、連鎖と同じ2辺経路の形に写す (段1/段2/検品を共用するため)。

    辺の検証は KG 由来でなく**文書側で済んでいる** (名指し + 条番号の照合)。
    """
    a_name = _alias_of(policy, anchor.doc_id)
    t_name = _alias_of(policy, target_doc)
    ref_node = f"{t_name}第{article}条"
    e1 = kg.PathEdge(
        subject=f"{a_name}の規定", predicate="REFERENCES", object=ref_node,
        source_chunks=[anchor.chunk_id], extraction_count=1,
        subject_forms=[a_name], object_forms=[ref_node, f"第{article}条"],
    )
    e2 = kg.PathEdge(
        subject=ref_node, predicate="DEFINED_BY", object=f"{ref_node}の規定内容",
        source_chunks=[target.chunk_id], extraction_count=1,
        subject_forms=[ref_node], object_forms=[f"{ref_node}の規定内容"],
    )
    return kg.KGPath(edges=[e1, e2])


def generate_from_references(
    pairs: list[tuple[Chunk, Chunk, str, int]],
    chunks_dir: Path,
    out_dir: Path,
    *,
    model: str,
    provenance_model: str,
    kg_snapshot: str = "unknown",
    prompt_path: Path = Path(WRITE_PROMPT),
    policy=None,
    llm: Optional[LLMCaller] = None,
    compose_llm: Optional[LLMCaller] = None,
    max_items: Optional[int] = None,
) -> Path:
    """KG-missing: 文書検証済みの明示参照から multi_hop 型の QA を生成する。

    段0 (KG 辺の含意判定) は**行わない** — 参照は条番号解決器が文書側で照合済みで、
    LLM 抽出の KG より強い裏付けを既に持つ。段1 (合成判定) 以降は連鎖と共用。
    """
    chunks = load_chunks(chunks_dir)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    from .rag_verify import BM25Index
    bm25 = BM25Index(chunks)
    chunk_row = {c.chunk_id: i for i, c in enumerate(chunks)}

    meta, write_body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "kg_write_qa")

    out_dir.mkdir(parents=True, exist_ok=True)
    used_paths = _load_used_paths(out_dir)
    out_path = _next_out_path(out_dir)

    written = 0
    n_compose_rejected = n_check_rejected = n_failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for anchor, target, target_doc, article in pairs:
            if max_items is not None and written >= max_items:
                break
            path = reference_pair_to_path(anchor, target, target_doc, article, policy)
            if path.as_text() in used_paths:
                continue
            support = [anchor.chunk_id, target.chunk_id]
            verified_edges = [
                kg.EdgeVerification(path.edges[0], anchor.chunk_id,
                                    "文書内の明示参照 (条番号解決器で照合済み)"),
                kg.EdgeVerification(path.edges[1], target.chunk_id,
                                    "参照先の条文 (見出し位置で照合済み)"),
            ]
            try:
                relation, reject_reason = compose_path_relation(
                    path, support, chunks_by_id,
                    model=provenance_model, llm=compose_llm,
                    # 供給が9対しか無く、思考オフの過剰厳格で全滅した実績がある
                    # (2026-08-03)。僅少系統に限り思考を許可する
                    allow_thinking=True,
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 合成判定失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            if relation is None:
                # kg_missing の参照は条番号解決器が文書側で照合済みで、僅少かつ
                # 人手レビュー前提。段1は**参考意見に格下げ**する (2026-08-03:
                # 思考オンでも実在の対応表参照・特例規定を誤棄却したため)。
                n_compose_rejected += 1
                print(f"[kg-generate] 段1不成立だが続行 ({path.as_text()}): "
                      f"{(reject_reason or '')[:80]}")
                relation = (f"文書内の明示参照 (条番号解決器で照合済み)。"
                            f"段1の参考意見: 不成立 — {reject_reason or ''}"[:300])

            qa_caller = llm or llm_generate
            try:
                raw = qa_caller(
                    prompt=render(write_body, {
                        "relation_sentence": relation,
                        "evidence_block": _format_evidence_block(
                            path, support, chunks_by_id),
                    }),
                    model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
                data = _coerce_to_dict(raw)
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 生成失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            qa, err = _finalize_qa(
                data, qt="multi_hop", novelty="unknown_relation",
                support=support, material=path, chunks_by_id=chunks_by_id,
                relation=relation, kg_snapshot=kg_snapshot,
                provenance_model="(document-verified)",
                verified_edges=verified_edges,
                shape="chain", model=model, prompt_version=prompt_version,
                bm25=bm25, chunk_row=chunk_row, checks_key="kg_missing",
            )
            if qa is None:
                if err and err.startswith("スキーマ不適合"):
                    print(f"[kg-generate] {err}")
                    n_failed += 1
                else:
                    print(f"[kg-generate] skip ({err})")
                    n_check_rejected += 1
                continue
            qa.kg_path.source = "kg_missing"
            f.write(qa.model_dump_json() + "\n")
            written += 1
            print(f"[kg-generate] {qa.qa_id} kg_missing | {path.as_text()[:60]}")

    print(f"[kg-generate] wrote {written} QAs → {out_path}")
    print(f"[kg-generate] 段1棄却 {n_compose_rejected} / 段3検品落ち {n_check_rejected} / "
          f"生成失敗 {n_failed}")
    return out_path


# ---------------- KG-decoy 系統 (関係の無い類似対 → 棄権が正解) ----------------

DECOY_WRITE_PROMPT = "prompts/kg_write_qa_decoy.md"

_ABSTAIN_WORDS = ("確認できない", "適用されない", "適用されません", "根拠がない",
                  "根拠はない", "定められていない", "規定されていない", "存在しない")


def _check_abstention_answer(ctx: CheckContext) -> Optional[str]:
    """decoy: 回答が棄権 (適用されない/根拠が確認できない) を明示しているか。"""
    if any(w in (ctx.qa.answer or "") for w in _ABSTAIN_WORDS):
        return None
    return "回答が棄権 (適用されない/根拠が確認できない) を明示していない"


_META_WORDS = ("SIMILAR_BUT_UNRELATED", "NO_RELATION", "始点", "終端", "辺1", "辺2")


def _check_meta_leak(ctx: CheckContext) -> Optional[str]:
    """decoy: 内部の関係ラベル・経路用語が問いや回答に漏れていないか。"""
    text = (ctx.qa.question or "") + (ctx.qa.answer or "")
    hits = [w for w in _META_WORDS if w in text]
    return f"内部用語の混入: {hits}" if hits else None


CHECKS_BY_TYPE["kg_decoy"] = [("棄権明示", _check_abstention_answer),
                              ("内部用語", _check_meta_leak)]


def find_decoy_pairs(
    chunks: list[Chunk],
    embeddings,
    policy,
    chunk_nodes: dict[str, set],
    *,
    top_n: int = 40,
    sim_floor: float = 0.75,
) -> list[tuple[Chunk, Chunk, float]]:
    """埋め込み類似が高いのに、参照関係も KG 上の繋がりも無い異文書対を列挙する。

    現行方式が「危険なので捨てていた」素材の転用。表面的な類似に釣られて
    誤った根拠で答えないか (棄権できるか) を測るデコイになる。
    関係の最終確認は段1 (合成判定の不成立) が行うので、ここは候補出しに徹する。
    """
    import numpy as np

    sims = embeddings @ embeddings.T
    n = len(chunks)
    cand: list[tuple[float, int, int]] = []
    for i in range(n):
        row = sims[i]
        for j in range(i + 1, n):
            if row[j] < sim_floor:
                continue
            if chunks[i].doc_id == chunks[j].doc_id:
                # 同一文書は原則除外だが、位置が大きく離れた高類似対は
                # 「同一法体系内の適用区画違い」(30m の罠型) の候補として残す
                # (2026-08-04: 本番ベンチで KG-ON の急所と実証された型)
                if abs(chunks[i].position - chunks[j].position) < 30:
                    continue
                # 同じ章・節に属する対は除外 — 同じ別表の行同士のような
                # 「実は関係のある対」を掴む (実測: SUS材の表の行2つを
                # 無関係と誤認し、引用と矛盾する棄権回答を生成した)
                si = chunks[i].section_path[0] if chunks[i].section_path else ""
                sj = chunks[j].section_path[0] if chunks[j].section_path else ""
                if si == sj:
                    continue
            cand.append((float(row[j]), i, j))
    cand.sort(reverse=True)

    out: list[tuple[Chunk, Chunk, float]] = []
    seen_docs: dict[tuple[str, str], int] = {}
    for sim, i, j in cand:
        a, b = chunks[i], chunks[j]
        # 実関係のある対を除く: 明示参照 (どちら向きでも) or KG 上の共有ノード
        if policy.named_reference(a, b.doc_id) is not None: continue
        if policy.named_reference(b, a.doc_id) is not None: continue
        if chunk_nodes.get(a.chunk_id, set()) & chunk_nodes.get(b.chunk_id, set()):
            continue
        key = tuple(sorted((a.doc_id, b.doc_id)))
        if seen_docs.get(key, 0) >= 2:     # 同じ文書対からは2問まで
            continue
        seen_docs[key] = seen_docs.get(key, 0) + 1
        out.append((a, b, sim))
        if len(out) >= top_n:
            break
    return out


def generate_from_decoys(
    pairs: list[tuple[Chunk, Chunk, float]],
    chunks_dir: Path,
    out_dir: Path,
    *,
    model: str,
    provenance_model: str,
    kg_snapshot: str = "unknown",
    prompt_path: Path = Path(DECOY_WRITE_PROMPT),
    llm: Optional[LLMCaller] = None,
    compose_llm: Optional[LLMCaller] = None,
    max_items: Optional[int] = None,
) -> Path:
    """KG-decoy: 類似だが無関係な対から「棄権が正解」の QA を生成する。

    段1 (合成判定) を逆向きに使う: **不成立の理由が返った対だけ**を採用し、
    その理由を「関係が無い根拠」として QA に刻む。段1が成立と言った対は
    デコイにせず捨てる (= KG も解決器も見落とした実関係の候補。ログに出す)。
    """
    chunks = load_chunks(chunks_dir)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    meta, write_body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "kg_write_qa_decoy")

    out_dir.mkdir(parents=True, exist_ok=True)
    used_paths = _load_used_paths(out_dir)
    out_path = _next_out_path(out_dir)

    written = 0
    n_not_decoy = n_check_rejected = n_failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for a, b, sim in pairs:
            if max_items is not None and written >= max_items:
                break
            e1 = kg.PathEdge(
                subject=f"{a.doc_id}の規定", predicate="SIMILAR_BUT_UNRELATED",
                object=f"{b.doc_id}の規定", source_chunks=[a.chunk_id],
                extraction_count=0, subject_forms=[a.doc_id], object_forms=[b.doc_id])
            e2 = kg.PathEdge(
                subject=f"{b.doc_id}の規定", predicate="NO_RELATION",
                object=f"{b.doc_id}の内容", source_chunks=[b.chunk_id],
                extraction_count=0, subject_forms=[b.doc_id],
                object_forms=[f"{b.doc_id}の内容"])
            path = kg.KGPath(edges=[e1, e2])
            if path.as_text() in used_paths:
                continue
            support = [a.chunk_id, b.chunk_id]
            # 段1を逆向きに: 不成立の理由が「関係なし」の根拠
            try:
                relation, reject_reason = compose_path_relation(
                    path, support, chunks_by_id,
                    model=provenance_model, llm=compose_llm,
                )
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 判定失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            if relation is not None:
                n_not_decoy += 1
                print(f"[kg-generate] デコイ不採用 (関係が成立と判定 = KG 見落とし候補): "
                      f"{a.chunk_id} × {b.chunk_id}")
                continue
            no_relation_reason = (reject_reason or "関係を裏付ける原文が無い")
            for w in _META_WORDS:
                no_relation_reason = no_relation_reason.replace(w, "")

            qa_caller = llm or llm_generate
            try:
                raw = qa_caller(
                    prompt=render(write_body, {
                        "relation_sentence": no_relation_reason,
                        "evidence_block": _format_evidence_block(
                            path, support, chunks_by_id),
                    }),
                    model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
                data = _coerce_to_dict(raw)
            except Exception as e:  # LLM出力不良に加え 503/接続断も1件の失敗として続行 (2026-08-05)
                print(f"[kg-generate] 生成失敗: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            qa, err = _finalize_qa(
                data, qt="negation_exhaustive", novelty="unknown_relation",
                support=support, material=path, chunks_by_id=chunks_by_id,
                relation=f"(関係なし) {no_relation_reason}",
                kg_snapshot=kg_snapshot,
                provenance_model=provenance_model,
                verified_edges=[
                    kg.EdgeVerification(e1, a.chunk_id, f"埋め込み類似 {sim:.2f} だが無関係"),
                    kg.EdgeVerification(e2, b.chunk_id, "参照関係・KG 接続とも無し"),
                ],
                shape="chain", model=model, prompt_version=prompt_version,
                checks_key="kg_decoy",
            )
            if qa is None:
                if err and err.startswith("スキーマ不適合"):
                    print(f"[kg-generate] {err}")
                    n_failed += 1
                else:
                    print(f"[kg-generate] skip ({err})")
                    n_check_rejected += 1
                continue
            qa.kg_path.source = "kg_decoy"
            f.write(qa.model_dump_json() + "\n")
            written += 1
            print(f"[kg-generate] {qa.qa_id} kg_decoy | {a.doc_id} × {b.doc_id} (sim={sim:.2f})")

    print(f"[kg-generate] wrote {written} QAs → {out_path}")
    print(f"[kg-generate] デコイ不採用 (関係成立) {n_not_decoy} / "
          f"段3検品落ち {n_check_rejected} / 生成失敗 {n_failed}")
    return out_path
