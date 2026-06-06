"""Tests for the R1 Pydantic schema."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from rageval.aspects import ALL_ASPECTS, ASPECT_TO_CATEGORY
from rageval.schema import (
    Chunk,
    Explainability,
    FilterScores,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)


def _make_qa(**overrides) -> QAItem:
    base = dict(
        qa_id="qa_0001",
        question="ネジの締付トルクは？",
        answer="12 N·m",
        rationale=[Rationale(doc_id="doc1", page=3, text="締付トルクは12N·mとする")],
        category=["Reasoning"],
        aspect=["quantitative_calc"],
        reasoning_complexity=ReasoningComplexity(quantitative=True),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Medium",
        difficulty_rationale="必要チャンク=1, 推論ステップ=1",
        business_scenario="設計変更影響評価",
        generation=GenerationInfo(model="gpt-oss-20B", prompt_version="v1.0", generated_at=datetime.now()),
    )
    base.update(overrides)
    return QAItem(**base)


def test_qaitem_round_trip():
    qa = _make_qa()
    dumped = qa.model_dump_json()
    loaded = QAItem.model_validate_json(dumped)
    assert loaded.qa_id == qa.qa_id
    assert loaded.review_status == "pending"
    assert loaded.filter_scores.answerability is None
    assert loaded.aspect == ["quantitative_calc"]


def test_invalid_difficulty_rejected():
    with pytest.raises(ValidationError):
        _make_qa(retrieval_level="IMPOSSIBLE")  # type: ignore[arg-type]


def test_unknown_aspect_rejected():
    with pytest.raises(ValidationError):
        _make_qa(aspect=["nonexistent_aspect"])


def test_empty_aspect_allowed_for_kg_track():
    """aspect is optional (KG-PoC track uses kg_query_type/kg_novelty instead)."""
    qa = _make_qa(aspect=[])
    assert qa.aspect == []


def test_all_aspects_round_trip():
    """Every aspect listed in ALL_ASPECTS is accepted and maps to a known category."""
    for aspect in ALL_ASPECTS:
        qa = _make_qa(aspect=[aspect], category=[ASPECT_TO_CATEGORY[aspect]])
        assert aspect in qa.aspect


def test_filter_scores_defaults():
    fs = FilterScores()
    assert fs.answerability is None
    assert fs.leakage is None


def test_chunk_model():
    c = Chunk(chunk_id="c001", doc_id="doc1", page=1, text="hello")
    assert c.text == "hello"
