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


# ---------- 認定ゲートの記録欄 (2026-08-08 設計書「認定ゲート」節) ----------

def _kg_path(**overrides):
    from rageval.schema import KGPathProvenance
    base = dict(
        snapshot_id="snap1",
        path_text="A -[R]-> B",
        edges=[],
        verifier_model="judge",
        verified_at=datetime.now(),
    )
    base.update(overrides)
    return KGPathProvenance(**base)


def test_kg_path_certification_defaults():
    """既定は「ゲート未適用」を表す (certified=True / 理由は空文字)。"""
    kp = _kg_path()
    assert kp.certified is True
    assert kp.certification_reason == ""


def test_old_kg_path_json_without_certification_still_loads():
    """回帰: 認定欄を持たない既存 JSONL がそのまま読めること (後方互換)。"""
    from rageval.schema import KGPathProvenance
    old_json = (
        '{"snapshot_id": "snap1", "path_text": "A -[R]-> B", "edges": [],'
        ' "verifier_model": "judge", "verified_at": "2026-08-01T00:00:00",'
        ' "shape": "star", "source": "kg_covered"}'
    )
    kp = KGPathProvenance.model_validate_json(old_json)
    assert kp.certified is True
    assert kp.certification_reason == ""


def test_kg_path_certification_round_trip():
    from rageval.schema import KGPathProvenance
    kp = _kg_path(certified=False, certification_reason="要素チャンクが全て top-10 に揃う")
    loaded = KGPathProvenance.model_validate_json(kp.model_dump_json())
    assert loaded.certified is False
    assert "top-10" in loaded.certification_reason
