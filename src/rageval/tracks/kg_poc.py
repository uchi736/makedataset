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
