"""KG-PoC track: 3軸タグ + プロービングによる KG-RAG 評価設計フレーム.

3軸:
- AXIS-1 クエリ型: 5値
- AXIS-2 未知性: 3値
- AXIS-3 LLM既知性: 2値 (プロービングで決定)

配分は CLI --mix で明示指定する (未指定なら全 15 セルに等分)。
"""

from __future__ import annotations

from typing import Literal, get_args


# ---------------- AXIS-1 クエリ型 ----------------

KGQueryType = Literal[
    "single_fact",          # 単一ファクト (baseline / 一問一答確認)
    "multi_hop",            # マルチホップ
    "aggregation",          # 一覧・集約 (ベクターRAG弱点)
    "traceability",         # トレーサビリティ (要求→手順→記録の追跡)
    "negation_exhaustive",  # 否定・網羅 (高難度、少数で利く)
]

ALL_KG_QUERY_TYPES: tuple[str, ...] = tuple(get_args(KGQueryType))

KG_QUERY_TYPE_LABELS: dict[str, str] = {
    "single_fact":         "単一ファクト",
    "multi_hop":           "マルチホップ",
    "aggregation":         "一覧・集約",
    "traceability":        "トレーサビリティ",
    "negation_exhaustive": "否定・網羅",
}

KG_QUERY_TYPE_DESCRIPTIONS: dict[str, str] = {
    "single_fact":         "単一文・一段で答えられるファクト確認 (baseline)",
    "multi_hop":           "複数エンティティ・関係を辿る必要がある質問",
    "aggregation":         "複数箇所にまたがる項目を一覧化・集約する質問",
    "traceability":        "要求→手順→記録のような典型的な追跡 (QMS固有)",
    "negation_exhaustive": "否定条件や網羅性を要求する高難度質問",
}


# ---------------- AXIS-2 未知性 ----------------

KGNovelty = Literal[
    "unknown_term",         # 未知語: 表記揺れ・社内呼称を正規ノードへ
    "unknown_relation",     # ★主役: LLMが事前学習で持たない関係
    "procedural_relation",  # 手順的関係: precedes/requires/triggers
]

ALL_KG_NOVELTY: tuple[str, ...] = tuple(get_args(KGNovelty))

KG_NOVELTY_LABELS: dict[str, str] = {
    "unknown_term":        "未知語",
    "unknown_relation":    "未知の関係",
    "procedural_relation": "手順的関係",
}

KG_NOVELTY_DESCRIPTIONS: dict[str, str] = {
    "unknown_term":        "事前学習に無い語や表記揺れ。KG のノードに紐づけて解決する",
    "unknown_relation":    "関係そのものが事前学習に無い (★KG導入の最も強い主張根拠)",
    "procedural_relation": "順序・条件・依存。typed-edge (precedes/requires/triggers) で表現",
}


# ---------------- AXIS-3 LLM既知性 (プロービングで決定) ----------------

LLMKnowledge = Literal["known", "unknown"]

ALL_LLM_KNOWLEDGE: tuple[str, ...] = tuple(get_args(LLMKnowledge))

LLM_KNOWLEDGE_LABELS: dict[str, str] = {
    "known":   "既知 (RAG無しでも答えられる)",
    "unknown": "未知 (RAG必須)",
}


# 「主戦場」の概念は廃止。生成時の配分は CLI --mix で明示指定 (もしくは等分)。


# ---------------- 区分 (派生ラベル) ----------------
# kg_query_type / kg_novelty / rag_verification / rationale doc 数 から
# 集計時に「本丸 / 監査 / 対照 / 名寄せ / その他」のラベルを派生で出す。
# QAItem には列を追加せず、集計スクリプト側で band(qa) を呼んで貼る。

QABand = Literal["本丸", "監査", "対照", "名寄せ", "その他"]
ALL_BANDS: tuple[str, ...] = tuple(get_args(QABand))

QA_BAND_LABELS: dict[str, str] = {
    "本丸":   "本丸 (両 doc leave-one-out 通過、KG-RAG の主戦場)",
    "監査":   "監査 (片 doc で書けるが traceability/multi_hop 体裁)",
    "対照":   "対照 (vector RAG が解けるべき劣化確認用)",
    "名寄せ": "名寄せ (同一・別名の判定)",
    "その他": "その他 (上記の判定が成立しなかった問い)",
}


def band(qa) -> QABand:
    """QAItem を読んで派生区分ラベルを返す。

    v2.0 で意味条件化:
      1. 本丸: answer_grounding 必須。全 doc を leave-one-out で「単独で sufficient」と
         判定されなかった (= 全 doc が answer に必要) こと。さらにクエリ型が
         traceability か multi_hop、rationale が 2 doc 以上。
         旧 v1 の rv.answer_match==no_match 条件は外し、partial で漏れていた問いも救う。
      2. 監査: 形式上は traceability/multi_hop で 2 doc rationale だが、leave-one-out
         で片 doc が不要と判定された (= 体裁だけ multi-doc) 問い。本丸との重複を排除。
      3. 名寄せ: 未知性が unknown_term (表記揺れ・別名検出)
      4. 対照: クエリ型が single_fact で rag_verify が match (vector RAG が解けることを実証)
      5. その他

    AnswerGrounding が未付与 (= rageval answer-grounding 未実施) の場合は、
    旧 v1 ロジックで暫定判定する (後方互換)。

    QAItem を直接参照すると循環 import になるので duck-typing で扱う。
    """
    rv = getattr(qa, "rag_verification", None)
    ag = getattr(qa, "answer_grounding", None)
    # 根拠単位数: chunk_id があればチャンク単位、無ければ文書単位で数える。
    # 一文書コーパス (モデル就業規則) では離れたチャンク 2 つで本丸条件を満たす。
    rationale_doc_count = (
        len({(r.chunk_id or r.doc_id) for r in qa.rationale}) if qa.rationale else 0
    )
    qt = qa.kg_query_type
    nov = qa.kg_novelty

    if ag is not None and ag.per_doc_sufficient:
        # v2.0 経路: AnswerGrounding で意味条件を判定
        all_docs_required = not any(ag.per_doc_sufficient.values())
        any_doc_unnecessary = any(ag.per_doc_sufficient.values())

        # 本丸: 全 doc 必須 × traceability/multi_hop × 2 doc rationale
        if (
            all_docs_required
            and qt in ("traceability", "multi_hop")
            and rationale_doc_count >= 2
        ):
            return "本丸"
        # 監査: 体裁上は 2 doc 跨ぎだが leave-one-out で片 doc が不要だった
        if (
            any_doc_unnecessary
            and qt in ("traceability", "multi_hop")
            and rationale_doc_count >= 2
        ):
            return "監査"
        # 名寄せ
        if nov == "unknown_term":
            return "名寄せ"
        # 対照
        if qt == "single_fact" and rv is not None and rv.answer_match == "match":
            return "対照"
        return "その他"

    # v1 後方互換: AnswerGrounding 未実施の旧データ
    # 本丸: vector RAG が落とした × 構造を要する型 × 異 doc 跨ぎ
    if (
        rv is not None
        and rv.answer_match == "no_match"
        and qt in ("traceability", "multi_hop")
        and rationale_doc_count >= 2
    ):
        return "本丸"
    # 監査: traceability (本丸条件を満たさないもの)
    if qt == "traceability":
        return "監査"
    # 名寄せ: 未知語の同一性判定
    if nov == "unknown_term":
        return "名寄せ"
    # 対照: 単一ファクトを vector RAG が解けた実証
    if qt == "single_fact" and rv is not None and rv.answer_match == "match":
        return "対照"
    return "その他"
