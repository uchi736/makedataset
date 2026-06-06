"""Tests for Stage 1 generation pipeline using a mock LLM (R1 schema)."""

from __future__ import annotations

import json
from pathlib import Path

from rageval.chunker import chunk_directory
from rageval.generate import generate_batch
from rageval.schema import QAItem


MOCK_QA_JSON = {
    "qa_id": "auto_deadbeef",
    "question": "製品Xの動作電圧範囲は何ボルトですか？",
    "answer": "DC 21.6V〜26.4V (24V±10%) です。",
    "rationale": [
        {"doc_id": "sample", "page": None, "text": "定格電圧: DC 24V ±10%"}
    ],
    "category": ["Reasoning"],
    "aspect": ["quantitative_calc"],
    "reasoning_complexity": {
        "multi_step": True, "quantitative": True, "negation": False,
        "cause_effect": False, "comparison": False, "temporal": False,
        "output_type": "none",
    },
    "retrieval_difficulty": {
        "multi_doc": False, "multi_chunk": False, "low_locality": False,
        "remote_reference": False, "doc_volume_large": False, "chunk_size_large": False,
        "abstraction_discrepancy": False, "vocabulary_mismatch": False,
    },
    "source_structure": {
        "tables_charts": False, "complex_layout": False, "specific_area_ref": False,
        "logical_nesting": False, "large_enumeration": False, "redundancy": False,
    },
    "explainability": {"evidence_strictness": "hier-ref"},
    "retrieval_level": "Easy",
    "answer_level": "Medium",
    "difficulty_rationale": "必要チャンク=1, 推論ステップ=1",
    "business_scenario": "工程設計レビュー",
}


def _mock_llm(**kwargs):
    # Echo the anchor_chunks_block content into the answer so we can verify the
    # prompt is being built with the multi-chunk template.
    prompt = kwargs.get("prompt", "")
    n_anchors = prompt.count("## チャンク")
    qa = dict(MOCK_QA_JSON)
    qa["answer"] = f"{MOCK_QA_JSON['answer']} (anchors={n_anchors})"
    return json.dumps(qa, ensure_ascii=False)


def test_parse_kg_mix_valid():
    from rageval.generate import parse_kg_mix
    out = parse_kg_mix("multi_hop:unknown_relation=5, traceability:procedural_relation=3")
    assert out == [
        ("multi_hop", "unknown_relation", 5),
        ("traceability", "procedural_relation", 3),
    ]


def test_parse_kg_mix_invalid_query_type():
    import pytest
    from rageval.generate import parse_kg_mix
    with pytest.raises(ValueError, match="unknown kg_query_type"):
        parse_kg_mix("invalid_qt:unknown_relation=5")


def test_parse_kg_mix_invalid_format():
    import pytest
    from rageval.generate import parse_kg_mix
    with pytest.raises(ValueError, match="format"):
        parse_kg_mix("multi_hop unknown_relation 5")


def test_build_kg_cell_queue_with_mix():
    import random
    from rageval.generate import _build_kg_cell_queue
    mix = [("multi_hop", "unknown_relation", 3), ("traceability", "procedural_relation", 2)]
    queue = _build_kg_cell_queue(5, mix, random.Random(0))
    assert len(queue) == 5
    assert queue.count(("multi_hop", "unknown_relation")) == 3
    assert queue.count(("traceability", "procedural_relation")) == 2


def test_build_kg_cell_queue_equal_default():
    import random
    from rageval.generate import _build_kg_cell_queue
    queue = _build_kg_cell_queue(15, None, random.Random(0))
    assert len(queue) == 15
    # 15 cells × 1 each
    from collections import Counter
    counts = Counter(queue)
    assert all(v == 1 for v in counts.values())


def test_detect_fabricated_terms_flags_neologism():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="本規程は振替休日を定める。")]
    fab = detect_fabricated_terms(
        "ディスミッション・エクスクルージョン条件として振替休日を設定する",
        chunks,
    )
    assert "ディスミッション" in " ".join(fab)
    assert "エクスクルージョン" in " ".join(fab)


def test_detect_fabricated_terms_passes_when_in_chunk():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="リストストラップを装着すること。")]
    fab = detect_fabricated_terms(
        "リストストラップの装着順序は?",
        chunks,
    )
    assert fab == []   # word appears in chunk → not fabricated


def test_detect_fabricated_terms_ignores_short_katakana():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="ボルトのトルクは…")]
    # 3-char katakana words shouldn't trip the filter (too noisy)
    fab = detect_fabricated_terms("ナット を締める", chunks)
    assert fab == []


def test_render_prompt_injects_aspect_examples():
    """Aspect の ✓良い例 / ✗悪い例 がプロンプト本文に埋め込まれることを確認。"""
    from rageval.generate import _render_generate_prompt
    from rageval.aspects import ASPECT_BAD_PATTERNS, ASPECT_GOOD_EXAMPLES
    from rageval.prompts import load_prompt
    from rageval.schema import Chunk

    _, body = load_prompt("prompts/generate.md")
    anchors = [Chunk(chunk_id="c0", doc_id="d", text="本規程は所定労働時間を定める。")]
    spec = {
        "aspect": "complex_layout",
        "category": "Figure",
        "retrieval_level": "Easy",
        "answer_level": "Medium",
    }
    rendered = _render_generate_prompt(body, anchors, spec, [])

    # 良い例(少なくとも1つ)が含まれる
    for good in ASPECT_GOOD_EXAMPLES["complex_layout"]:
        assert good in rendered
    # 悪い例の代表(月数を数える型)が含まれる
    bad = ASPECT_BAD_PATTERNS["complex_layout"]
    assert any(b in rendered for b in bad)
    # 共通禁止ルールも残っている
    assert "列挙を数えるだけ" in rendered


def test_generate_batch_with_mock_llm(tmp_path: Path, monkeypatch):
    # Stub embeddings so we don't try to hit a real endpoint.
    monkeypatch.setattr("rageval.generate.compute_embeddings", lambda chunks: None)
    # Pin to a single-chunk aspect: multi-chunk aspects would skip on a 1-chunk
    # corpus (sampler returns None → no QA written), making len(lines)==3 flaky.
    monkeypatch.setattr(
        "rageval.generate._build_spec",
        lambda rng: {"aspect": "quantitative_calc", "category": "Reasoning"},
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text(
        "製品X仕様書\n定格電圧: DC 24V ±10%\n定格電流: 2A\n",
        encoding="utf-8",
    )
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    out_dir = tmp_path / "raw"
    seeds = tmp_path / "seeds.json"
    seeds.write_text("[]", encoding="utf-8")

    out_path = generate_batch(
        chunks_dir=chunks_dir, out_dir=out_dir, n=3, model="mock",
        seeds_path=seeds, llm=_mock_llm,
    )

    assert out_path.exists()
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    for line in lines:
        qa = QAItem.model_validate_json(line)
        assert qa.generation.model == "mock"
        assert qa.review_status == "pending"
        assert qa.aspect == ["quantitative_calc"]
        assert qa.business_scenario
        # Confirm prompt was assembled with anchor_chunks_block
        assert "anchors=" in qa.answer
