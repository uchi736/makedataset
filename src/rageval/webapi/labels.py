"""表示ラベル・目標値の正本 (Streamlit 版と Web 版が共有する1枚の表)。

値の定義そのものは aspects.py (25観点・カテゴリ) / tracks/kg_poc.py (KG 3軸) /
schema.py (列挙・診断3軸の項目) が正本で、ここは **その正本から並びを導き、
日本語の表示名だけを持つ**。

2026-08-08 の監査 (rank 37/38/39) まで、同じ表が review_app・stats_app・ここの
2〜3箇所に手書きされていた。schema.py に項目を足しても両画面のラベル表に足すまで
無警告で捨てられ、カテゴリを足すと絞り込みの初期値から漏れて件数だけが静かに減り、
減点式チェックリストは Web 側だけ f_join (別文書の同番号の同一視) を持つという
食い違いが実際に起きていた。以後は両画面ともこの表だけを読む。並びを `get_args`
から導くのは、値が増えたときに画面が黙って落とさないため。表示名を足し忘れた
場合は英語の値がそのまま出る (落とさない) 上で、tests/test_webapi.py の
キー集合の凍結試験が落ちる。

合格線 (answerability 等) は実際に QA を落とす filter.py の既定を引く。画面が
「合格」と描くのに filter は落とす、という食い違いを構造的に無くすため (rank 36)。
"""

from __future__ import annotations

from typing import Any, get_args

from rageval.aspects import (
    ALL_ASPECTS,
    ASPECT_DESCRIPTIONS,
    ASPECT_EXAMPLES,
    ASPECT_LABELS,
    ASPECT_TO_CATEGORY,
    CategoryName,
)
from rageval.filter import (
    DEFAULT_ANSWERABILITY_MIN,
    DEFAULT_GROUNDING_MIN,
    DEFAULT_UNIQUENESS_MAX,
)
from rageval.schema import (
    DifficultyLevel,
    DifficultyMatch,
    EvidenceStrictness,
    LeakageVerdict,
    OutputType,
    RAGAnswerMatch,
    ReasoningComplexity,
    RetrievalDifficulty,
    ReviewStatus,
    SourceStructure,
)
from rageval.tracks.kg_poc import (
    ALL_KG_NOVELTY,
    ALL_KG_QUERY_TYPES,
    ALL_LLM_KNOWLEDGE,
    KG_NOVELTY_DESCRIPTIONS,
    KG_NOVELTY_LABELS,
    KG_QUERY_TYPE_DESCRIPTIONS,
    KG_QUERY_TYPE_LABELS,
    LLM_KNOWLEDGE_LABELS,
)

# ---------------- 並び (すべて Literal から導出する) ----------------

CATEGORY_ORDER: tuple[str, ...] = tuple(get_args(CategoryName))
DIFF_ORDER: tuple[str, ...] = tuple(get_args(DifficultyLevel))
ALL_LEAKAGE: tuple[str, ...] = tuple(get_args(LeakageVerdict))
ALL_DIFFICULTY_MATCH: tuple[str, ...] = tuple(get_args(DifficultyMatch))
ALL_REVIEW_STATUS: tuple[str, ...] = tuple(get_args(ReviewStatus))
ALL_RAG_ANSWER_MATCH: tuple[str, ...] = tuple(get_args(RAGAnswerMatch))
ALL_OUTPUT_TYPE: tuple[str, ...] = tuple(get_args(OutputType))
ALL_EVIDENCE_STRICTNESS: tuple[str, ...] = tuple(get_args(EvidenceStrictness))

# ---------------- 表示名 (日本語だけをここに持つ) ----------------

CATEGORY_LABELS = {
    "Integration": "統合",
    "Reasoning":   "推論",
    "Logic":       "論理",
    "Figure":      "図表",
    "Abstention":  "棄権",
}

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

# ---------------- 診断3軸 (項目の正本は schema.py のモデル) ----------------

# 軸名 → schema のモデル。項目の増減はモデル側だけで起き、ここは表示名を足すだけ。
DIAG_MODELS = {
    "reasoning_complexity": ReasoningComplexity,
    "retrieval_difficulty": RetrievalDifficulty,
    "source_structure":     SourceStructure,
}

# 真偽の診断タグだけを表示対象にする (output_type は列挙なので別扱い)。
_DIAG_TEXT: dict[str, dict[str, str]] = {
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


def diag_flag_fields(axis: str) -> tuple[str, ...]:
    """診断軸の真偽フラグ項目を schema のモデルから引く。"""
    model = DIAG_MODELS[axis]
    return tuple(
        name for name, field in model.model_fields.items()
        if field.annotation is bool
    )


# 表示名が無い項目は英語の項目名のまま出す (画面から消さない)。
DIAG_LABELS: dict[str, dict[str, str]] = {
    axis: {f: _DIAG_TEXT[axis].get(f, f) for f in diag_flag_fields(axis)}
    for axis in DIAG_MODELS
}

# ---------------- 計画目標値と合格線 ----------------

# 合格線は filter.py の既定 (実際に QA を落とす値) をそのまま引く。
ANSWERABILITY_PASS: float = DEFAULT_ANSWERABILITY_MIN
GROUNDING_PASS: float = DEFAULT_GROUNDING_MIN
# filter は「類似度がこれを超えたら重複」なので、独自性の下限はその補数になる。
UNIQUENESS_PASS: float = 1.0 - DEFAULT_UNIQUENESS_MAX

TARGET_SEARCH_DIFF = {"Easy": 0.375, "Medium": 0.375, "Hard": 0.25}
TARGET_ANSWER_DIFF = {"Easy": 0.175, "Medium": 0.625, "Hard": 0.2}
TARGET_ABSTENTION = (0.10, 0.15)
PILOT_SIZE_RANGE = (30, 50)
ASPECT_MIN_PER = 3  # 各観点 最低3問 (R1 plan)

TARGETS: dict[str, Any] = {
    "search_diff":   dict(TARGET_SEARCH_DIFF),
    "answer_diff":   dict(TARGET_ANSWER_DIFF),
    "abstention":    list(TARGET_ABSTENTION),
    "pilot_size":    list(PILOT_SIZE_RANGE),
    "aspect_min":    ASPECT_MIN_PER,
    "answerability_pass": ANSWERABILITY_PASS,
    "grounding_pass":     GROUNDING_PASS,
    "uniqueness_pass":    UNIQUENESS_PASS,
}

# ---------------- レビュー時の減点式チェックリスト ----------------
# review_app._checklist_groups もここを読む (2026-08-08 一本化)。

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


def checklist_for(*, is_kg: bool) -> list[dict[str, Any]]:
    """減点式チェックリストを track ごとに引く (Streamlit / Web 共通の入口)。"""
    return CHECKLIST_KG if is_kg else CHECKLIST_GENERAL


def build_labels() -> dict[str, Any]:
    """フロントエンドが参照するラベル・目標値の一括 JSON。"""
    return {
        "all_aspects": list(ALL_ASPECTS),
        "aspect_labels": dict(ASPECT_LABELS),
        "aspect_descriptions": dict(ASPECT_DESCRIPTIONS),
        "aspect_examples": dict(ASPECT_EXAMPLES),
        "aspect_to_category": dict(ASPECT_TO_CATEGORY),
        "category_order": list(CATEGORY_ORDER),
        "category_labels": CATEGORY_LABELS,
        "diff_order": list(DIFF_ORDER),
        "difficulty_labels": DIFFICULTY_LABELS,
        "leakage_labels": LEAKAGE_LABELS,
        "difficulty_match_labels": DIFFICULTY_MATCH_LABELS,
        "review_status_labels": REVIEW_STATUS_LABELS,
        "rag_answer_match_labels": RAG_ANSWER_MATCH_LABELS,
        "output_type_labels": OUTPUT_TYPE_LABELS,
        "evidence_labels": EVIDENCE_LABELS,
        "diag_labels": DIAG_LABELS,
        # 画面側が並びを手写しせずに済むよう、Literal から導いた列挙を配る
        # (webui の手書き配列が schema の追加を取りこぼしていた: 監査 rank 42)。
        "all_leakage": list(ALL_LEAKAGE),
        "all_difficulty_match": list(ALL_DIFFICULTY_MATCH),
        "all_review_status": list(ALL_REVIEW_STATUS),
        "all_rag_answer_match": list(ALL_RAG_ANSWER_MATCH),
        "all_output_type": list(ALL_OUTPUT_TYPE),
        "all_evidence_strictness": list(ALL_EVIDENCE_STRICTNESS),
        "all_kg_query_types": list(ALL_KG_QUERY_TYPES),
        "kg_query_type_labels": dict(KG_QUERY_TYPE_LABELS),
        "kg_query_type_descriptions": dict(KG_QUERY_TYPE_DESCRIPTIONS),
        "all_kg_novelty": list(ALL_KG_NOVELTY),
        "kg_novelty_labels": dict(KG_NOVELTY_LABELS),
        "kg_novelty_descriptions": dict(KG_NOVELTY_DESCRIPTIONS),
        "all_llm_knowledge": list(ALL_LLM_KNOWLEDGE),
        "llm_knowledge_labels": dict(LLM_KNOWLEDGE_LABELS),
        "targets": TARGETS,
        "checklist_general": CHECKLIST_GENERAL,
        "checklist_kg": CHECKLIST_KG,
    }
