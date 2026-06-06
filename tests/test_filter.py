"""Tests for Stage 2 filter (R1 schema)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rageval.chunker import chunk_directory
from rageval.filter import filter_batch
from rageval.schema import (
    Explainability,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)


def _make_qa(qa_id: str, question: str, answer: str, doc_id: str = "sample") -> QAItem:
    return QAItem(
        qa_id=qa_id,
        question=question,
        answer=answer,
        rationale=[Rationale(doc_id=doc_id, page=None, text="定格電圧: DC 24V ±10%")],
        category=["Reasoning"],
        aspect=["quantitative_calc"],
        reasoning_complexity=ReasoningComplexity(quantitative=True),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="必要チャンク=1",
        business_scenario="工程設計レビュー",
        generation=GenerationInfo(model="mock", prompt_version="v3.0", generated_at=datetime.now()),
    )


def _mock_judge_llm(*, prompt: str, **kwargs) -> str:
    if "ANSWERABILITY" in prompt:
        if "不明" in prompt:
            return json.dumps({"answerability": 2, "reason": "not answerable"})
        return json.dumps({"answerability": 5, "reason": "ok"})
    if "LEAKAGE" in prompt:
        return json.dumps({"leakage": "pass", "reason": "ok"})
    if "GROUNDING" in prompt:
        return json.dumps({"grounding": 5, "reason": "ok"})
    if "DIFFICULTY_MATCH" in prompt:
        return json.dumps({"answer_level": "Easy", "reason": "ok"})
    if "RATIONALE_COMPLETENESS" in prompt:
        return json.dumps({"rationale_completeness": 5, "reason": "ok"})
    return json.dumps({})


def test_compute_rationale_grounded_all_match():
    from rageval.filter import compute_rationale_grounded
    qa = _make_qa("q", "Q?", "A")  # rationale.text = "定格電圧: DC 24V ±10%"
    anchor_chunks = [("sample", None, "前置き 定格電圧: DC 24V ±10% その他")]
    assert compute_rationale_grounded(qa, anchor_chunks) == 1.0


def test_compute_rationale_grounded_whitespace_insensitive():
    from rageval.filter import compute_rationale_grounded
    qa = _make_qa("q", "Q?", "A")
    # Anchor has different whitespace
    anchor_chunks = [("sample", None, "定格電圧:\nDC  24V  ±10%\n")]
    assert compute_rationale_grounded(qa, anchor_chunks) == 1.0


def test_compute_rationale_grounded_fabricated():
    """LLM hallucinated rationale not in chunk."""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    qa.rationale = [Rationale(doc_id="sample", page=None, text="この文は元チャンクに存在しない")]
    anchor_chunks = [("sample", None, "定格電圧: DC 24V ±10%")]
    assert compute_rationale_grounded(qa, anchor_chunks) == 0.0


def test_compute_rationale_grounded_partial():
    """One of two rationale entries is grounded."""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    qa.rationale = [
        Rationale(doc_id="sample", page=None, text="定格電圧: DC 24V ±10%"),  # in chunk
        Rationale(doc_id="sample", page=None, text="ここは捏造された引用"),    # not in chunk
    ]
    anchor_chunks = [("sample", None, "定格電圧: DC 24V ±10%")]
    assert compute_rationale_grounded(qa, anchor_chunks) == 0.5


def test_grounding_matches_second_chunk_on_same_page(tmp_path: Path):
    """同一 (doc_id, page) に複数チャンクがある場合、根拠が2つ目のチャンクに
    あっても逐語照合が通ること (旧実装はページを1チャンクに潰して誤って落とす)。"""
    from rageval.schema import Chunk

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    # 根拠は先頭チャンクにある。旧実装は同ページを最後のチャンク(c2)で上書きするため
    # 根拠を含む c1 が消えて誤って落ちる。新実装はページ全チャンクを連結するので通る。
    c1 = Chunk(chunk_id="d__c0", doc_id="d", page=1, text="先頭チャンク。定格電圧: DC 24V ±10%", position=0)
    c2 = Chunk(chunk_id="d__c1", doc_id="d", page=1, text="二つ目チャンク。無関係な文。", position=1)
    (chunks_dir / "d.jsonl").write_text(
        c1.model_dump_json() + "\n" + c2.model_dump_json() + "\n", encoding="utf-8"
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    # 根拠は2つ目のチャンクにある文。doc/page は当たるが旧実装だと別チャンクと照合され外れる。
    qa = _make_qa("qa1", "定格電圧は？", "DC 24V", doc_id="d")
    qa.rationale[0].page = 1
    raw_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    out_dir = tmp_path / "filtered"
    out_path = filter_batch(
        raw_path=raw_path, out_dir=out_dir, chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
    )
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 1
    kept = QAItem.model_validate_json(lines[0])
    assert kept.filter_scores.rationale_grounded == 1.0


def test_filter_batch_drops_low_answerability(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    qa_good = _make_qa("qa_good", "定格電圧は？", "DC 24V")
    qa_bad = _make_qa("qa_bad", "製品の色は不明ですか？", "不明")

    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_good.model_dump_json() + "\n")
        f.write(qa_bad.model_dump_json() + "\n")

    out_dir = tmp_path / "filtered"
    out_path = filter_batch(
        raw_path=raw_path, out_dir=out_dir, chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
    )

    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    kept = QAItem.model_validate_json(lines[0])
    assert kept.qa_id == "qa_good"
    assert kept.filter_scores.answerability == 5.0
    assert kept.filter_scores.leakage == "pass"
