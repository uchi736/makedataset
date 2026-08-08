"""Pydantic data models for RAG evaluation QA items (R1 plan, 3-layer model)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .aspects import ALL_ASPECTS, CategoryName
from .tracks.kg_poc import KGNovelty, KGQueryType, LLMKnowledge


DifficultyLevel = Literal["Easy", "Medium", "Hard"]
LeakageVerdict = Literal["pass", "fail"]
DifficultyMatch = Literal["aligned", "too_easy", "too_hard"]
ReviewStatus = Literal["pending", "accepted", "edited", "rejected"]
EvidenceStrictness = Literal["no-evidence", "hier-ref", "coord-ref", "multi-ref"]
OutputType = Literal["summary", "trans", "list", "none"]
RAGAnswerMatch = Literal["match", "partial", "no_match"]


# ===== 根拠 =====

class Rationale(BaseModel):
    doc_id: str
    # 根拠チャンクの所在。一文書コーパス (例: モデル就業規則) では本丸判定の
    # leave-one-out 単位が「文書」だと 1 単位に潰れるため、チャンク単位で判定
    # できるようにここに持つ。None なら従来どおり doc_id が判定単位になる。
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    text: str

    def evidence_unit(self) -> str:
        """leave-one-out の判定単位: チャンク ID があればチャンク、無ければ文書。"""
        return self.chunk_id or self.doc_id


# ===== 階層② 診断軸（富士通4軸, bool多重タグ） =====

class ReasoningComplexity(BaseModel):
    multi_step: bool = False
    quantitative: bool = False
    negation: bool = False
    cause_effect: bool = False
    comparison: bool = False
    temporal: bool = False
    output_type: OutputType = "none"


class RetrievalDifficulty(BaseModel):
    multi_doc: bool = False
    multi_chunk: bool = False
    low_locality: bool = False
    remote_reference: bool = False
    doc_volume_large: bool = False
    chunk_size_large: bool = False
    abstraction_discrepancy: bool = False
    vocabulary_mismatch: bool = False


class SourceStructure(BaseModel):
    tables_charts: bool = False
    complex_layout: bool = False
    specific_area_ref: bool = False
    logical_nesting: bool = False
    large_enumeration: bool = False
    redundancy: bool = False


class Explainability(BaseModel):
    evidence_strictness: EvidenceStrictness


# ===== 運用メタ =====

class GenerationInfo(BaseModel):
    model: str
    prompt_version: str
    generated_at: datetime


class FilterScores(BaseModel):
    answerability: Optional[float] = None
    leakage: Optional[LeakageVerdict] = None
    grounding: Optional[float] = None
    uniqueness: Optional[float] = None
    difficulty_match: Optional[DifficultyMatch] = None
    # Fraction of rationale entries whose .text appears verbatim (whitespace-
    # insensitive) in some anchor chunk. 1.0 = all grounded, 0.0 = all fabricated.
    rationale_grounded: Optional[float] = None


# ===== Vector RAG ground-truth verification =====
# `rageval rag-verify` で後付け。生成→判定の自己ループから外れた信号として
# 「vector RAG で trivial に解ける問い」を識別するために持つ。

class RAGVerification(BaseModel):
    top_k: int
    retrieved_chunk_ids: list[str]
    # 検索方式。dense = 密ベクトルのみ (旧)、hybrid = 密 + BM25 の RRF 融合。
    # 実運用の RAG はハイブリッドがデファクトなので、2026-08-02 以降の
    # スクリーニング (難易度実測・型確定) は hybrid を基準にする。
    # hybrid_rerank = 密+BM25 の候補プールを交差エンコーダで並べ直す2段構成
    # (本番 KG-OFF と同じ)。2026-08-04 以降のスクリーニング既定。
    retrieval_mode: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"
    # 文書一致: 根拠と同じ文書IDのチャンクが上位 k に1個でも入ったか。
    # ただし1文書が50を超えるチャンクに割れる技報では、根拠と無関係な
    # 章のチャンクが入っただけで真になるので、これだけでは過大評価になる。
    retrieval_hit_doc: bool = False
    # チャンク一致: **全ての根拠**について、その根拠本文を逐語で含むチャンクが
    # 上位 k に入ったか (2026-08-01 に「いずれか1件」から変更)。
    # KG-RAG の検索健全性を見る本命指標。
    retrieval_hit_chunk: bool = False
    # 逐語一致した根拠の割合 (部分命中を捨てずに持つ)。1.0 = retrieval_hit_chunk。
    retrieval_hit_ratio: float = 0.0
    # 旧フィールド (既存 JSONL との互換のため残置)。
    # 新規書き出しでは retrieval_hit_doc と同じ値を入れる。
    retrieval_hit: bool = False
    rag_answer: str
    answer_match: RAGAnswerMatch
    # judge が返した生の判定文字列。想定外の値や空文字が来たときの事後検証用。
    # 正常に match / partial / no_match が返ったときも、そのまま小文字化して保存する。
    # judge を呼ばなかった場合 (rag_answer が空 / 回答不能) や judge が例外で落ちた場合は None。
    judge_raw: Optional[str] = None
    # judge が正常に判定を返したか。False は「照合できなかった」であって
    # 「RAG が答えられなかった (no_match)」ではない。障害由来の no_match を
    # KG候補として数えないための旗 (既定 True = 正常)。
    judge_ok: bool = True
    rag_model: str
    judge_model: str
    verified_at: datetime


# ===== KG エッジ (sampling.ReferenceFollow の照合済み事実を保存) =====
# Gemma に書かせるのではなく、sampling 段階で確定する (anchor_chunk_id, ref_token, target_chunk_id) を記録する。
# これにより:
#  - 質問単位で「KG が辿るべきエッジ列」が機械可読化される (M4 構造トレース採点の素)
#  - predicate (= ref_token) を 6 種の関係カテゴリに後付けマッピングできる
#  - 「KG パス由来の正答」と「リランカ偶然合成」を切り分けられる

class RelationEdge(BaseModel):
    subject_chunk_id: str    # anchor 側のチャンク
    predicate: str           # 述語 = anchor.references に登場した参照語 (「別表第7」「第107条」「JIS B 8267」等)
    object_chunk_id: str     # target 側のチャンク (述語を本文 or section_path に含む)


# ===== KG 経路 provenance (既存 KG から生成した QA のみ) =====

class KGPathEdgeProvenance(BaseModel):
    """原文との含意確認を通過した KG エッジ。"""

    subject: str
    predicate: str
    object: str
    source_chunk_id: str
    extraction_count: int = 0
    verification_reason: str


class KGPathProvenance(BaseModel):
    """QA の素材にした KG 経路と、その検証条件を固定する。"""

    snapshot_id: str
    path_text: str
    edges: list[KGPathEdgeProvenance]
    verifier_model: str
    verified_at: datetime
    # 供給系統。kg_covered = KG の経路から生成 (KG が有利)、
    # kg_missing = 文書側で検証した明示参照のうち KG に対応が無いもの
    # (= KG のカバレッジ不足を測る)、kg_decoy = 関係の無い対 (棄権が正解)。
    source: Literal["kg_covered", "kg_missing", "kg_decoy"] = "kg_covered"
    # 素材の形。chain = 2辺連鎖 (multi_hop/traceability)、star = 星型 (aggregation)、
    # exclusion = EXCLUDES 辺 (negation)。型別の検品規則の切替キーにもなる。
    shape: Literal["chain", "star", "exclusion"] = "chain"
    # 段1 (合成判定) が書いた関係文。「この経路はこういう1本の関係として成り立つ」の
    # 言語化。traceability の模範解答の素材にも使う。旧データは空文字。
    composition_reason: str = ""
    # unknown_term 変換の記録。質問文中の正規表記を KG の surface_forms にある
    # 別表記へ機械置換した場合に {"canonical": 正規表記, "used": 使用表記} が入る。
    # 空 dict = 置換なし。
    term_substitution: dict[str, str] = Field(default_factory=dict)
    # 認定ゲート (rag_verify.certify_kg_items) の結果。型別表の合格線を実測で
    # 切った結果で、False = 「KG が要る」を名乗らせない (対照群として保持する。
    # 項目は捨てない)。既定 True は旧 JSONL をそのまま読めるようにするための値で、
    # 「認定済み」ではなく「ゲート未適用」を意味する — 未適用かどうかは
    # certification_reason が空文字かどうかで区別する。
    certified: bool = True
    # 認定/不認定の理由 (型別表のどの条件で切ったか)。空文字 = ゲート未適用。
    certification_reason: str = ""


# ===== AnswerGrounding =====
# 「rationale[i] を伏せたら answer が決まらなくなるか」を doc 単位で判定する。
# Azure gpt-4.1 judge が doc ごとに「この doc を取り除いたとき answer の固有値が
# 残りの rationale だけから再現できるか」を yes/no で返し、yes ならその doc は
# answer に対して不要 (= sufficient_without)。
# 本丸の条件は「全 doc が伏せられたとき answer が決まらない」= 全 doc で per_doc_sufficient=False。
# 1 doc でも True (= その doc 単独で answer が出る) なら片 doc 完結問いと判定する。
#
# v3 で判明: rationale が 2 doc 並んでも 17/23 件は片 doc 完結だったため、これを
# 機械的に検出するための後付け判定スキーマとして導入する。

class AnswerGrounding(BaseModel):
    # doc_id -> その doc を伏せても answer が決まるか (True = この doc は要らない)
    # judge が落ちた単位はここに入れない (errors に入る)。
    per_doc_sufficient: dict[str, bool]
    # judge が判定できなかった単位。1件でもあると本丸 (全根拠必須) とは認めない。
    # 障害を「その根拠は必須」= 本丸認定に倒さないための旗。
    errors: list[str] = Field(default_factory=list)
    judge_model: str
    # judge LLM が返した生 JSON (debug 用)
    judge_raw: Optional[str] = None
    verified_at: datetime


# ===== QA本体 =====

class QAItem(BaseModel):
    qa_id: str
    question: str
    answer: str
    rationale: list[Rationale]

    # 階層① 能力評価軸
    # general トラックでは必須、kg_poc トラックでは省略可 (代わりに kg_query_type/kg_novelty を使う)
    category: list[CategoryName] = Field(default_factory=list)
    aspect: list[str] = Field(default_factory=list)

    # 階層② 診断軸
    reasoning_complexity: ReasoningComplexity
    retrieval_difficulty: RetrievalDifficulty
    source_structure: SourceStructure
    explainability: Explainability

    # 階層③ 難易度
    retrieval_level: DifficultyLevel
    answer_level: DifficultyLevel
    difficulty_rationale: str

    # 運用メタ (business_scenario は Phase 3 用に optional で残す。
    # 新規生成では使わず、既存 JSONL の互換性確保のためだけに残置)
    business_scenario: Optional[str] = None

    # KG-PoC track 専用フィールド (general track では None)
    # 詳細は tracks/kg_poc.py 参照
    kg_query_type: Optional[KGQueryType] = None
    kg_novelty: Optional[KGNovelty] = None
    llm_knowledge: Optional[LLMKnowledge] = None   # `rageval probe` で後付け
    rag_verification: Optional[RAGVerification] = None  # `rageval rag-verify` で後付け
    relation_edges: list[RelationEdge] = Field(default_factory=list)  # sampling/enrich で注入
    kg_path: Optional[KGPathProvenance] = None  # 検証済み既存 KG 経路から生成した場合に付与
    answer_grounding: Optional[AnswerGrounding] = None  # leave-one-out 判定で後付け or 生成時必須

    generation: GenerationInfo
    filter_scores: FilterScores = Field(default_factory=FilterScores)
    review_status: ReviewStatus = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    @field_validator("aspect")
    @classmethod
    def _validate_aspects(cls, v: list[str]) -> list[str]:
        # Empty is allowed (KG-PoC track uses kg_query_type/kg_novelty instead).
        unknown = [a for a in v if a not in ALL_ASPECTS]
        if unknown:
            raise ValueError(f"unknown aspect(s): {unknown}. allowed={list(ALL_ASPECTS)}")
        return v


# ===== Chunk =====

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    text: str
    position: int = 0
    section_path: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
