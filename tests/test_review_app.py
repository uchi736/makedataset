"""Regression tests for lossless review mirror persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rageval.review_app import _merge_reviewed_items, _save_review_item
from rageval.schema import (
    Explainability,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)


def _qa(qa_id: str, *, kg: bool = False) -> QAItem:
    return QAItem(
        qa_id=qa_id,
        question=f"{qa_id} question",
        answer=f"{qa_id} answer",
        rationale=[Rationale(doc_id="doc", text="根拠として十分な長さを持つテキストです。")],
        category=[] if kg else ["Reasoning"],
        aspect=[] if kg else ["quantitative_calc"],
        kg_query_type="multi_hop" if kg else None,
        kg_novelty="unknown_relation" if kg else None,
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="test",
        generation=GenerationInfo(
            model="mock", prompt_version="test", generated_at=datetime.now(),
        ),
    )


def test_merge_reviewed_items_restores_rows_missing_from_old_partial_mirror():
    general = _qa("general")
    kg = _qa("kg", kg=True)
    reviewed_general = general.model_copy(deep=True)
    reviewed_general.review_status = "accepted"

    merged = _merge_reviewed_items([general, kg], [reviewed_general])

    assert [qa.qa_id for qa in merged] == ["general", "kg"]
    assert [qa.review_status for qa in merged] == ["accepted", "pending"]


def test_save_review_item_preserves_other_tracks_legacy_lines_and_prior_reviews(
    tmp_path: Path,
):
    source = tmp_path / "filtered" / "batch.jsonl"
    source.parent.mkdir()
    general = _qa("general")
    kg = _qa("kg", kg=True)
    legacy = {"qa_id": "legacy", "old_schema": True}
    source.write_text(
        general.model_dump_json()
        + "\n"
        + kg.model_dump_json()
        + "\n"
        + json.dumps(legacy, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    source_before = source.read_bytes()

    mirror = tmp_path / "reviewed" / source.name
    mirror.parent.mkdir()
    reviewed_general = general.model_copy(deep=True)
    reviewed_general.review_status = "accepted"
    # Simulate a mirror written by the old track-filtered implementation.
    mirror.write_text(reviewed_general.model_dump_json() + "\n", encoding="utf-8")

    reviewed_kg = kg.model_copy(deep=True)
    reviewed_kg.review_status = "rejected"
    _save_review_item(reviewed_kg, source, mirror)

    lines = mirror.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert QAItem.model_validate_json(lines[0]).review_status == "accepted"
    assert QAItem.model_validate_json(lines[1]).review_status == "rejected"
    assert json.loads(lines[2]) == legacy
    assert source.read_bytes() == source_before
    assert not list(mirror.parent.glob(f".{mirror.name}.*.tmp"))
