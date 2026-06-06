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
