"""Tests for anchor-chunk selection strategies."""

from __future__ import annotations

import random

import numpy as np

from rageval.sampling import (
    AnchorSampler,
    MultiDocByEmbedding,
    ReferenceFollow,
    SameDocRemote,
    SingleChunk,
)
from rageval.schema import Chunk


def _ch(doc, pos, text="x", sect=None, refs=None) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}__c{pos:04d}",
        doc_id=doc,
        page=None,
        text=text,
        position=pos,
        section_path=sect or [],
        references=refs or [],
    )


def test_single_chunk_returns_one():
    chunks = [_ch("d1", 0), _ch("d1", 1)]
    out = SingleChunk().select(chunks, None, random.Random(0))
    assert len(out) == 1


def test_same_doc_remote_picks_far_positions():
    chunks = [
        _ch("d1", 0, sect=["第1章"]),
        _ch("d1", 1, sect=["第1章"]),
        _ch("d1", 4, sect=["第2章"]),
        _ch("d1", 5, sect=["第2章"]),
    ]
    rng = random.Random(0)
    strat = SameDocRemote(n=2, min_position_gap=3)
    for _ in range(20):
        out = strat.select(chunks, None, rng)
        if len(out) == 2:
            assert abs(out[0].position - out[1].position) >= 3
            assert out[0].doc_id == out[1].doc_id


def test_same_doc_remote_fallback_when_doc_too_small():
    chunks = [_ch("d1", 0), _ch("d2", 0)]
    out = SameDocRemote(n=2, min_position_gap=3).select(chunks, None, random.Random(0))
    assert len(out) == 1


def test_reference_follow_links_pair():
    a = _ch("spec", 0, text="本ボルトは JIS B 1083 に従う", refs=["JIS B 1083"])
    b = _ch("jis", 0, text="JIS B 1083 第3表: 締結トルク表")
    other = _ch("other", 0, text="無関係")
    rng = random.Random(42)
    out = ReferenceFollow(n=2).select([a, b, other], None, rng)
    assert len(out) == 2
    doc_ids = {c.doc_id for c in out}
    assert "spec" in doc_ids and "jis" in doc_ids


def test_reference_follow_fallback_when_no_refs():
    chunks = [_ch("d", 0, text="参照なし")]
    out = ReferenceFollow(n=2).select(chunks, None, random.Random(0))
    assert len(out) == 1


def test_multi_doc_by_embedding_force_distinct_doc():
    chunks = [
        _ch("d1", 0, text="A"),
        _ch("d1", 1, text="A'"),
        _ch("d2", 0, text="A''"),
    ]
    # Make d1[0] and d2[0] highly similar; d1[1] also similar but same doc as anchor
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.14],
        [0.95, 0.31],
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rng = random.Random(0)

    # Force anchor to be d1[0] by repeating until rng picks it
    for _ in range(50):
        out = MultiDocByEmbedding(n=2, force_distinct_doc=True).select(chunks, embeddings, rng)
        if out[0].doc_id == "d1" and out[0].position == 0 and len(out) == 2:
            assert out[1].doc_id == "d2"
            return
    # If we never landed on d1[0] as anchor, at least all runs respected distinct-doc when 2 picked
    rng = random.Random(123)
    for _ in range(30):
        out = MultiDocByEmbedding(n=2, force_distinct_doc=True).select(chunks, embeddings, rng)
        if len(out) == 2:
            assert out[0].doc_id != out[1].doc_id


def test_multi_doc_by_embedding_no_embeddings_fallback():
    chunks = [_ch("d1", 0), _ch("d2", 0)]
    out = MultiDocByEmbedding(n=2).select(chunks, None, random.Random(0))
    assert len(out) == 1


def test_multi_doc_by_embedding_rejects_weak_cross_doc_pair():
    """Regression: cross-doc pairs with sim < sim_floor should be skipped
    (single-chunk fallback) rather than producing nonsense composites like
    時間外労働 × 標準トルク."""
    chunks = [_ch("hr", 0), _ch("manufacturing", 0)]
    # Cosine similarity ~0.3 (weak — different domains)
    embeddings = np.array([[1.0, 0.0], [0.3, 0.95]])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rng = random.Random(0)
    out = MultiDocByEmbedding(
        n=2, force_distinct_doc=True, sim_floor=0.55,
    ).select(chunks, embeddings, rng)
    # Should NOT return 2 chunks across docs given sim < floor
    assert len(out) == 1


def test_anchor_sampler_dispatch_unknown_aspect_single_chunk():
    chunks = [_ch("d1", 0), _ch("d1", 1)]
    sampler = AnchorSampler(chunks, embeddings=None)
    out = sampler.select("quantitative_calc", random.Random(0))
    assert len(out) == 1


def test_anchor_sampler_dispatch_remote_reference():
    chunks = [_ch("d1", 0, sect=["A"]), _ch("d1", 5, sect=["B"])]
    sampler = AnchorSampler(chunks, embeddings=None)
    out = sampler.select("remote_reference", random.Random(0))
    assert len(out) == 2


def test_compatibility_filter_simple_table_picks_only_tables():
    """A `simple_table` aspect must only consider chunks with table markers."""
    from rageval.sampling import find_compatible_chunks
    chunks = [
        _ch("d1", 0, text="ただの段落です。"),
        _ch("d1", 1, text="| 区分 | 値 |\n|---|---|\n| A | 1 |"),
        _ch("d2", 0, text="別のテキスト本文。"),
    ]
    compat = find_compatible_chunks("simple_table", chunks)
    assert len(compat) == 1
    assert "|---|" in compat[0].text


def test_compatibility_filter_standards_reference_requires_jis_iso():
    from rageval.sampling import find_compatible_chunks
    chunks = [
        _ch("d", 0, refs=["JIS B 1083"]),
        _ch("d", 1, refs=["第3章"]),  # not a standard
        _ch("d", 2, refs=[]),
    ]
    compat = find_compatible_chunks("standards_reference", chunks)
    assert len(compat) == 1
    assert "JIS B 1083" in compat[0].references


def test_compatibility_filter_complex_layout_needs_multi_section():
    from rageval.sampling import find_compatible_chunks
    chunks = [
        _ch("d", 0, text="t", sect=["第1章"]),
        _ch("d", 1, text="t", sect=["第1章", "1.1 概要"]),
        _ch("d", 2, text="t"),
    ]
    compat = find_compatible_chunks("complex_layout", chunks)
    assert len(compat) == 1
    assert compat[0].position == 1


def test_anchor_sampler_records_misses_on_unsatisfiable_aspect():
    """When no chunk matches the aspect, sampler still returns something but
    flags the miss in compat_misses."""
    chunks = [_ch("d", 0, text="plain text", refs=[])]
    sampler = AnchorSampler(chunks, embeddings=None)
    sampler.select("standards_reference", random.Random(0))
    assert sampler.compat_misses.get("standards_reference") == 1


def test_anchor_sampler_no_misses_when_compatible():
    chunks = [_ch("d", 0, text="x", refs=["JIS B 1083"])]
    sampler = AnchorSampler(chunks, embeddings=None)
    sampler.select("standards_reference", random.Random(0))
    assert sampler.compat_misses == {}


# ---------------- Retrieval difficulty scoring (deterministic) ----------------

def test_score_retrieval_level_single_chunk_is_easy():
    from rageval.sampling import score_retrieval_level
    assert score_retrieval_level([_ch("d1", 0)]) == "Easy"


def test_score_retrieval_level_same_doc_close_is_medium():
    from rageval.sampling import score_retrieval_level
    # 同一文書・複数チャンクで位置差が遠隔閾値未満 → Medium
    assert score_retrieval_level([_ch("d1", 0), _ch("d1", 2)]) == "Medium"


def test_score_retrieval_level_same_doc_remote_is_hard():
    from rageval.sampling import score_retrieval_level
    # 同一文書だが遠隔参照級に離れている → Hard
    assert score_retrieval_level([_ch("d1", 0), _ch("d1", 20)]) == "Hard"


def test_score_retrieval_level_multi_doc_is_hard():
    from rageval.sampling import score_retrieval_level
    assert score_retrieval_level([_ch("d1", 0), _ch("d2", 0)]) == "Hard"


def test_compute_retrieval_difficulty_bools():
    from rageval.sampling import compute_retrieval_difficulty
    rd = compute_retrieval_difficulty([_ch("d1", 0), _ch("d2", 20)])
    assert rd.multi_doc is True
    assert rd.multi_chunk is True
    rd2 = compute_retrieval_difficulty([_ch("d1", 0)])
    assert rd2.multi_doc is False
    assert rd2.multi_chunk is False


def test_select_skips_when_multi_chunk_unsatisfiable():
    """単一文書1チャンクで multi_doc 観点 → 組成を作れず None + difficulty_misses。"""
    chunks = [_ch("d1", 0)]
    sampler = AnchorSampler(chunks, embeddings=None)
    out = sampler.select("multi_doc_reference", random.Random(0), max_resamples=3)
    assert out is None
    assert sampler.difficulty_misses.get("multi_doc_reference") == 1


def test_select_succeeds_for_remote_reference_within_doc():
    """単一文書でも遠く離れた2チャンクがあれば remote_reference は成立。"""
    chunks = [_ch("d1", 0, sect=["A"]), _ch("d1", 8, sect=["B"])]
    sampler = AnchorSampler(chunks, embeddings=None)
    out = sampler.select("remote_reference", random.Random(0))
    assert out is not None and len(out) == 2
    assert sampler.difficulty_misses == {}
