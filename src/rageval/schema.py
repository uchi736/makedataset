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
    page: Optional[int] = None
    text: str


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
    # 文書一致: 根拠と同じ文書IDのチャンクが上位 k に1個でも入ったか。
    # ただし1文書が50を超えるチャンクに割れる技報では、根拠と無関係な
    # 章のチャンクが入っただけで真になるので、これだけでは過大評価になる。
    retrieval_hit_doc: bool = False
    # チャンク一致: 根拠本文 (rationale.text) を逐語で含むチャンクが
    # 上位 k に入ったか。filter 側の逐語照合と同じ空白正規化規則を使う。
    # KG-RAG の検索健全性を見る本命指標。
    retrieval_hit_chunk: bool = False
    # 旧フィールド (既存 JSONL との互換のため残置)。
    # 新規書き出しでは retrieval_hit_doc と同じ値を入れる。
    retrieval_hit: bool = False
    rag_answer: str
    answer_match: RAGAnswerMatch
    # judge が返した生の判定文字列。想定外の値や空文字が来たときの事後検証用。
    # 正常に match / partial / no_match が返ったときも、そのまま小文字化して保存する。
    # judge を呼ばなかった場合 (rag_answer が空 / 回答不能) や judge が例外で落ちた場合は None。
    judge_raw: Optional[str] = None
    rag_model: str
    judge_model: str
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
