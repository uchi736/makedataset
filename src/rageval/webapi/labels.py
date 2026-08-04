"""表示ラベル・目標値を1つの JSON にまとめて返す。

25観点 / KG 3軸の定義は aspects.py / tracks/kg_poc.py が正本で、ここでは
それをそのまま引く。診断タグ等の表示専用ラベルは review_app / stats_app と
同じ文言をここに持つ (Streamlit 側の表示は変更しない方針のため独立させる)。
"""

from __future__ import annotations

from typing import Any

from rageval.aspects import (
    ALL_ASPECTS,
    ASPECT_DESCRIPTIONS,
    ASPECT_EXAMPLES,
    ASPECT_LABELS,
    ASPECT_TO_CATEGORY,
)
from rageval.tracks.kg_poc import (
    ALL_KG_NOVELTY,
    ALL_KG_QUERY_TYPES,
    KG_NOVELTY_DESCRIPTIONS,
    KG_NOVELTY_LABELS,
    KG_QUERY_TYPE_DESCRIPTIONS,
    KG_QUERY_TYPE_LABELS,
    LLM_KNOWLEDGE_LABELS,
)

CATEGORY_ORDER = ["Integration", "Reasoning", "Logic", "Figure", "Abstention"]
CATEGORY_LABELS = {
    "Integration": "統合",
    "Reasoning":   "推論",
    "Logic":       "論理",
    "Figure":      "図表",
    "Abstention":  "棄権",
}

DIFF_ORDER = ["Easy", "Medium", "Hard"]
DIFFICULTY_LABELS = {"Easy": "易", "Medium": "中", "Hard": "難"}

LEAKAGE_LABELS = {"pass": "合格", "fail": "不合格"}
DIFFICULTY_MATCH_LABELS = {
    "aligned":  "一致",
    "too_easy": "易しすぎる",
    "too_hard": "難しすぎる",
}
REVIEW_STATUS_LABELS = {
    "pending":  "未判定",
    "accepted": "採用",
    "edited":   "修正済",
    "rejected": "却下",
}
RAG_ANSWER_MATCH_LABELS = {
    "match":    "一致",
    "partial":  "部分一致",
    "no_match": "不一致",
}
OUTPUT_TYPE_LABELS = {"summary": "要約", "trans": "翻訳", "list": "列挙", "none": "なし"}
EVIDENCE_LABELS = {
    "no-evidence": "根拠なし",
    "hier-ref":    "単一根拠",
    "coord-ref":   "並列根拠",
    "multi-ref":   "複数根拠厳密",
}

# 診断タグ (真のフラグだけ表示する)
DIAG_LABELS: dict[str, dict[str, str]] = {
    "reasoning_complexity": {
        "multi_step":   "多段推論",
        "quantitative": "数値計算",
        "negation":     "否定推論",
        "cause_effect": "因果推論",
        "comparison":   "比較",
        "temporal":     "時系列",
    },
    "retrieval_difficulty": {
        "multi_doc":               "複数文書",
        "multi_chunk":             "複数チャンク",
        "low_locality":            "局所性低",
        "remote_reference":        "遠隔参照",
        "doc_volume_large":        "大量文書",
        "chunk_size_large":        "大チャンク",
        "abstraction_discrepancy": "抽象度の乖離",
        "vocabulary_mismatch":     "語彙ミスマッチ",
    },
    "source_structure": {
        "tables_charts":     "表・図",
        "complex_layout":    "複雑レイアウト",
        "specific_area_ref": "領域参照",
        "logical_nesting":   "階層構造",
        "large_enumeration": "大量列挙",
        "redundancy":        "冗長性",
    },
}

# 計画上の目標値 (stats_app.py と同値)
TARGETS: dict[str, Any] = {
    "search_diff":   {"Easy": 0.375, "Medium": 0.375, "Hard": 0.25},
    "answer_diff":   {"Easy": 0.175, "Medium": 0.625, "Hard": 0.2},
    "abstention":    [0.10, 0.15],
    "pilot_size":    [30, 50],
    "aspect_min":    3,
    "answerability_pass": 4.0,
    "grounding_pass":     4.0,
    "uniqueness_pass":    1.0 - 0.92,
}

# レビュー時の減点式チェックリスト (review_app._checklist_groups と同じ文言)
CHECKLIST_KG = [
    {"group": "根拠の検証", "items": [
        {"key": "f_inf", "label": "回答が引用範囲を超えて推測している", "desc": "引用に無い事実・数値・結論を回答が含む"},
        {"key": "f_mis", "label": "引用が原文と一致していない", "desc": "✗印の引用がある / 黄色の箇所と食い違う"},
        {"key": "f_irr", "label": "根拠チャンクが質問と無関係", "desc": "別の話題のチャンクを根拠にしている"},
        {"key": "f_join", "label": "引用同士の結合が正当でない", "desc": "別文書の同番号 (第N条・様式第N・別表N) を同一視していないか。第N条がどの文書の条文かを必ず確認"},
    ]},
    {"group": "KG適性の検証", "items": [
        {"key": "f_single", "label": "単一チャンク／キーワード一致で解けてしまう", "desc": "ベクター検索だけで届く問い (基準点用の single_fact なら指摘不要)"},
        {"key": "f_norel", "label": "エンティティ間の関係をたどる必要がない", "desc": "規定→例外→条件のような辿りが無い"},
        {"key": "f_noagg", "label": "集約・多段推論の要素がない", "desc": "クエリ型タグ (集約/マルチホップ等) に実態が伴わない"},
    ]},
    {"group": "品質の検証", "items": [
        {"key": "f_amb", "label": "質問が一意に解釈できない", "desc": "複数の読み方ができる / 対象が曖昧"},
        {"key": "f_ans", "label": "回答に過不足がある", "desc": "誤り・抜け・余計な付け足し"},
        {"key": "f_diff", "label": "難易度根拠が妥当でない", "desc": "難易度タグや説明が実態と合わない"},
    ]},
]

CHECKLIST_GENERAL = [
    {"group": "根拠の検証", "items": [
        {"key": "f_inf", "label": "回答が引用範囲を超えて推測している", "desc": "引用に無い事実・数値・結論を回答が含む"},
        {"key": "f_mis", "label": "引用が原文と一致していない", "desc": "✗印の引用がある / 黄色の箇所と食い違う"},
        {"key": "f_irr", "label": "根拠チャンクが質問と無関係", "desc": "別の話題のチャンクを根拠にしている"},
    ]},
    {"group": "分類の検証", "items": [
        {"key": "f_asp", "label": "観点・分類タグの意図に合っていない", "desc": "タグ通りの問いの構造になっていない"},
    ]},
    {"group": "品質の検証", "items": [
        {"key": "f_amb", "label": "質問が一意に解釈できない", "desc": "複数の読み方ができる / 対象が曖昧"},
        {"key": "f_ans", "label": "回答に過不足がある", "desc": "誤り・抜け・余計な付け足し"},
        {"key": "f_leak", "label": "質問に答えそのものが書かれている", "desc": "読むだけで解けてしまう (漏れ)"},
        {"key": "f_diff", "label": "難易度タグが妥当でない", "desc": "検索/回答難易度が実態と合わない"},
    ]},
]


def build_labels() -> dict[str, Any]:
    """フロントエンドが参照するラベル・目標値の一括 JSON。"""
    return {
        "all_aspects": list(ALL_ASPECTS),
        "aspect_labels": dict(ASPECT_LABELS),
        "aspect_descriptions": dict(ASPECT_DESCRIPTIONS),
        "aspect_examples": dict(ASPECT_EXAMPLES),
        "aspect_to_category": dict(ASPECT_TO_CATEGORY),
        "category_order": CATEGORY_ORDER,
        "category_labels": CATEGORY_LABELS,
        "diff_order": DIFF_ORDER,
        "difficulty_labels": DIFFICULTY_LABELS,
        "leakage_labels": LEAKAGE_LABELS,
        "difficulty_match_labels": DIFFICULTY_MATCH_LABELS,
        "review_status_labels": REVIEW_STATUS_LABELS,
        "rag_answer_match_labels": RAG_ANSWER_MATCH_LABELS,
        "output_type_labels": OUTPUT_TYPE_LABELS,
        "evidence_labels": EVIDENCE_LABELS,
        "diag_labels": DIAG_LABELS,
        "all_kg_query_types": list(ALL_KG_QUERY_TYPES),
        "kg_query_type_labels": dict(KG_QUERY_TYPE_LABELS),
        "kg_query_type_descriptions": dict(KG_QUERY_TYPE_DESCRIPTIONS),
        "all_kg_novelty": list(ALL_KG_NOVELTY),
        "kg_novelty_labels": dict(KG_NOVELTY_LABELS),
        "kg_novelty_descriptions": dict(KG_NOVELTY_DESCRIPTIONS),
        "llm_knowledge_labels": dict(LLM_KNOWLEDGE_LABELS),
        "targets": TARGETS,
        "checklist_general": CHECKLIST_GENERAL,
        "checklist_kg": CHECKLIST_KG,
    }
