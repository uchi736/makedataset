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
    _is_list_chunk,
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


# ---------- チャンク単位化: 同一文書遠隔ペア (2026-07-01) ----------

def test_reference_follow_same_doc_remote_allowed_with_gap():
    """min_same_doc_gap 指定時、同一 doc でも位置が離れていれば target に採用。"""
    chunks = [
        _ch("d1", 0, text="第52条 退職とする", refs=["第9条"]),
        _ch("d1", 3, text="第9条 休職 (近すぎる)", sect=["(休職)"]),
        _ch("d1", 40, text="第9条 労働者が次のいずれかに該当するときは休職とする", sect=["(休職)"]),
    ]
    rng = random.Random(0)
    hits = 0
    for _ in range(30):
        out = ReferenceFollow(
            n=2, prefer_specific=True, force_distinct_doc=True, min_same_doc_gap=10,
        ).select(chunks, None, rng)
        if out[0].position == 0 and len(out) >= 2:
            hits += 1
            # gap 3 の近接チャンクは選ばれず、gap 40 の遠隔チャンクが選ばれる
            assert out[1].position == 40
    assert hits > 0


def test_reference_follow_same_doc_still_excluded_without_gap_param():
    """min_same_doc_gap 未指定 + force_distinct_doc=True は従来どおり同一 doc を除外。"""
    chunks = [
        _ch("d1", 0, text="第52条 退職とする", refs=["第9条"]),
        _ch("d1", 40, text="第9条 休職とする"),
    ]
    rng = random.Random(0)
    for _ in range(20):
        out = ReferenceFollow(
            n=2, prefer_specific=True, force_distinct_doc=True,
        ).select(chunks, None, rng)
        if out[0].position == 0:
            assert len(out) == 1  # 同一 doc しかないので anchor 単独


# ---------- v2.0 リストチャンク除外 ----------

def test_is_list_chunk_flags_form_list_only_chunk():
    """45_tokutei_setsubi pos=45 のような『様式第1〜第27 を丸ごと貼ったチャンク』を True。"""
    text = (
        "様式第１（第５条関係）様式第２（第５条関係）様式第３（第４６条関係）"
        "様式第４（第４７条関係）様式第５（第４９条関係）様式第６（第５０条関係）"
        "様式第７（第５３条関係）"
    )
    c = _ch("45_tokutei_setsubi", 45, text=text)
    assert _is_list_chunk(c) is True


def test_is_list_chunk_does_not_flag_normal_article():
    """『様式第6号を参照』程度の言及のある条文本文は False。"""
    c = _ch("09_kagi", 116, text=(
        "ボイラー附属設備の検査結果は様式第6号に記録する。検査の方法は別途定める。"
        "なお、附属設備の範囲は第60条第3項に規定するとおりである。"
    ))
    assert _is_list_chunk(c) is False


def test_multi_doc_by_embedding_excludes_list_chunks():
    """exclude_list_chunks=True のとき、リストチャンクは partner にしない。"""
    list_text = (
        "様式第１（第５条関係）様式第２（第４６条関係）様式第３（第４７条関係）"
        "様式第４（第４９条関係）様式第５（第５０条関係）様式第６（第５３条関係）"
    )
    chunks = [
        _ch("d1", 0, text="本文 第5条 ..."),
        _ch("d2", 0, text=list_text),  # ← リスト専用
        _ch("d3", 0, text="本文 第5条 別 doc の関連規定"),
    ]
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.4],
        [0.85, 0.5],
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rng = random.Random(0)
    for _ in range(50):
        out = MultiDocByEmbedding(
            n=2, force_distinct_doc=True, sim_floor=0.5,
            exclude_list_chunks=True,
        ).select(chunks, embeddings, rng)
        if out[0].doc_id == "d1" and len(out) >= 2:
            # partner に d2 (リスト) は選ばれない
            assert out[1].doc_id != "d2"


def test_multi_doc_by_embedding_require_article_overlap():
    """anchor の references と partner の references / text に共通の参照が無いペアは reject。"""
    chunks = [
        _ch("d1", 0, text="anchor 本文 (第5条 言及)", refs=["第5条"]),
        _ch("d2", 0, text="無関係 partner (第999条のみ)", refs=["第999条"]),
        _ch("d3", 0, text="関連 partner (第5条 本文に登場)", refs=["第5条"]),
    ]
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.4],
        [0.85, 0.5],
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rng = random.Random(0)
    for _ in range(50):
        out = MultiDocByEmbedding(
            n=2, force_distinct_doc=True, sim_floor=0.5,
            require_article_overlap=True,
        ).select(chunks, embeddings, rng)
        if out[0].doc_id == "d1" and len(out) >= 2:
            # partner は d3 (overlap あり) のみ。d2 (overlap 無し) は外れる
            assert out[1].doc_id == "d3"


def test_multi_doc_by_embedding_no_samedoc_fallback_when_distinct_required():
    """Regression (2026-06-17): force_distinct_doc=True で異 doc partner が見つからない
    とき、同 doc から 2 個目を水増ししない。これをやると anchor 2 個・doc 1 個の
    「2 chunk だが 1 doc」になり、上位 (generate.py) の early-skip が機能せず
    偽 traceability が量産される。"""
    chunks = [
        _ch("d1", 0, text="anchor"),
        _ch("d1", 1, text="d1 別チャンク (高類似だが同 doc)"),
        _ch("d2", 0, text="d2 (低類似で sim_floor 下回る)"),
    ]
    # anchor (d1[0]) と d1[1] は cos≈0.99、 d2[0] とは cos≈0.1 (低類似)
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.14],
        [0.1, 0.99],
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rng = random.Random(0)
    # 何度試行しても、d1[0] が anchor になったときは「同 doc 水増ししない」を担保
    saw_d1_anchor = False
    for _ in range(50):
        out = MultiDocByEmbedding(
            n=2, force_distinct_doc=True, sim_floor=0.55,
        ).select(chunks, embeddings, rng)
        if out[0].doc_id == "d1" and out[0].position == 0:
            saw_d1_anchor = True
            # 異 doc partner が無いので picked は 1 個のままで返るべき
            assert len(out) == 1, f"同 doc で水増しされた: {[(c.doc_id, c.position) for c in out]}"
    assert saw_d1_anchor, "d1 anchor を 50 回引けなかった (テスト分岐の問題)"


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


# ---------------- v3: 文書スコープ規則 (誤結合対策) ----------------

def test_reference_follow_rejects_bare_article_number_cross_doc(monkeypatch):
    """裸の「第N条」一致だけでは異文書ペアを作らない (別法令の同番号は別物)。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    a = _ch("law_a", 0, text="省令第63条に規定する適切な措置とは…", refs=["第63条"])
    b = _ch("law_b", 0, text="様式第11（第63条関係）による申請書…")
    out = ReferenceFollow(n=2, force_distinct_doc=True).select([a, b], None, random.Random(0))
    assert len(out) == 1  # 相手が見つからず anchor 単独


def test_reference_follow_allows_ledger_chunk_pair(monkeypatch):
    """エッジ台帳に載っているチャンク対は裸番号でも成立 (解決済みエッジ)。"""
    import rageval.sampling as sampling
    a = _ch("law_a", 0, text="省令第63条に規定する…", refs=["第63条"])
    b = _ch("law_b", 0, text="様式第11（第63条関係）…")
    policy = sampling.CrossDocPolicy(
        allowed_chunk_pairs={frozenset((a.chunk_id, b.chunk_id))}
    )
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", policy)
    out = ReferenceFollow(n=2, force_distinct_doc=True).select([a, b], None, random.Random(0))
    assert {c.doc_id for c in out} == {"law_a", "law_b"}


def test_reference_follow_allows_global_standard_cross_doc(monkeypatch):
    """JIS 等の規格番号は全域一意なので異文書ペアの根拠になる (従来動作の維持)。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    a = _ch("spec", 0, text="本ボルトは JIS B 1083 に従う", refs=["JIS B 1083"])
    b = _ch("jis", 0, text="JIS B 1083 第3表: 締結トルク表")
    out = ReferenceFollow(n=2, force_distinct_doc=True).select([a, b], None, random.Random(0))
    assert {c.doc_id for c in out} == {"spec", "jis"}


def test_reference_follow_same_doc_bare_number_still_works(monkeypatch):
    """同一文書内なら裸の条番号で辿ってよい (本丸の遠隔参照)。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    a = _ch("law_a", 0, text="第100条の規定を準用する", refs=["第100条"])
    b = _ch("law_a", 20, text="第100条 省令第63条に規定する…")
    out = ReferenceFollow(n=2, min_same_doc_gap=10).select([a, b], None, random.Random(0))
    assert len(out) == 2 and {c.doc_id for c in out} == {"law_a"}


def test_named_alias_allows_cross_doc(monkeypatch):
    """相手文書を名称で名指ししている場合 (明示参照) は異文書ペア成立。"""
    import rageval.sampling as sampling
    a = _ch("kagi", 0, text="ボイラー則第百一条第二号ロ又はハに掲げる者", refs=["第101条"])
    b = _ch("boiler_rule", 0, text="第百一条 一級ボイラー技士免許…")
    policy = sampling.CrossDocPolicy(doc_aliases={"boiler_rule": ["ボイラー則"]})
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", policy)
    assert policy.join_ok(a, b) is True
    out = ReferenceFollow(n=2, force_distinct_doc=True).select([a, b], None, random.Random(0))
    assert {c.doc_id for c in out} == {"kagi", "boiler_rule"}


def test_multidoc_embedding_rejects_bare_number_cross_doc(monkeypatch):
    """埋め込み類似 + 裸番号一致でも異文書ペアは成立しない。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    chunks = [
        _ch("law_a", 0, text="第63条の適切な措置", refs=["第63条"]),
        _ch("law_b", 0, text="様式第11（第63条関係）", refs=["第63条"]),
    ]
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    strat = MultiDocByEmbedding(n=2, force_distinct_doc=True, require_article_overlap=True)
    out = strat.select(chunks, emb, random.Random(0))
    assert len(out) == 1


def test_multidoc_embedding_same_doc_overlap_still_works(monkeypatch):
    """同一文書内の参照番号重なりは従来どおり接続根拠になる。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    chunks = [
        _ch("law_a", 0, text="第63条の適切な措置", refs=["第63条"]),
        _ch("law_a", 9, text="第63条に基づく手続", refs=["第63条"]),
    ]
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    strat = MultiDocByEmbedding(n=2, force_distinct_doc=False, require_article_overlap=True)
    out = strat.select(chunks, emb, random.Random(0))
    assert len(out) == 2


def test_sampler_diversity_caps_doc_pair(monkeypatch):
    """同一の異文書対からはバッチ内 MAX_PER_DOC_PAIR 問まで。使用済みチャンク組は再抽選。"""
    import rageval.sampling as sampling
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", sampling.CrossDocPolicy())
    chunks = [
        _ch("d1", 0, text="JIS B 0001 による", refs=["JIS B 0001"]),
        _ch("d1", 1, text="JIS B 0001 参照", refs=["JIS B 0001"]),
        _ch("d2", 0, text="JIS B 0001 第1表", refs=["JIS B 0001"]),
        _ch("d2", 1, text="JIS B 0001 第2表", refs=["JIS B 0001"]),
    ]
    emb = np.ones((4, 2)) / np.sqrt(2)
    sampler = AnchorSampler(chunks, emb)
    rng = random.Random(0)
    sets = []
    for _ in range(AnchorSampler.MAX_PER_DOC_PAIR):
        out = sampler.select("multi_doc_reference", rng)
        assert out is not None
        # v3.1: select は候補を返すだけ。QA が通過したときに commit で確定する。
        sampler.commit(out)
        sets.append(frozenset(c.chunk_id for c in out))
    assert len(set(sets)) == len(sets)  # 同じチャンク組は出ない
    # 上限到達後は同じ文書対を返さない (None = skip)
    out3 = sampler.select("multi_doc_reference", rng)
    assert out3 is None
    assert sampler.diversity_rejects > 0


def test_uncommitted_anchor_can_be_reused():
    """QA が棄却されたアンカー組は commit されないので枠を消費しない。"""
    chunks = [
        _ch("d1", 0, text="JIS B 0001 による", refs=["JIS B 0001"]),
        _ch("d2", 0, text="JIS B 0001 第1表", refs=["JIS B 0001"]),
    ]
    emb = np.ones((2, 2)) / np.sqrt(2)
    sampler = AnchorSampler(chunks, emb)
    first = sampler.select("multi_doc_reference", random.Random(0))
    assert first is not None
    # commit しない = 生成が棄却された → もう一度同じ組が引ける
    again = sampler.select("multi_doc_reference", random.Random(0))
    assert again is not None
    assert {c.chunk_id for c in again} == {c.chunk_id for c in first}


def test_cross_capable_index_and_preference(monkeypatch):
    """異文書対を作れるアンカーの索引化と優先抽選。"""
    import rageval.sampling as sampling
    a = _ch("law_a", 0, text="ボイラー則第百一条に掲げる者", refs=["第101条"])
    b = _ch("law_b", 0, text="第百一条 一級ボイラー技士免許…", refs=["第1条"])
    noise = [_ch("law_c", i, text=f"無関係{i}", refs=["第9条"]) for i in range(20)]
    policy = sampling.CrossDocPolicy(doc_aliases={"law_b": ["ボイラー則"]})
    policy.index_cross_capable([a, b] + noise)
    assert a.chunk_id in policy.cross_capable_chunk_ids  # 明示参照側
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", policy)
    # 優先抽選: 多数回引けば law_a アンカーの異文書ペアが安定して出る
    strat = ReferenceFollow(n=2, force_distinct_doc=True)
    rng = random.Random(0)
    hit = sum(
        1 for _ in range(30)
        if {c.doc_id for c in strat.select([a, b] + noise, None, rng)} == {"law_a", "law_b"}
    )
    assert hit >= 10


# ---------------- v3.1: 名指し接続のチャンク解決 ----------------

def test_article_number_parses_all_notations():
    from rageval.sampling import article_number
    assert article_number("ボイラー則第百一条第二号ロ") == 101
    assert article_number("技術基準解釈第１０７条第１項") == 107
    assert article_number("第38条") == 38
    assert article_number("第十五条") == 15
    assert article_number("条番号なし") is None


def test_resolve_named_target_picks_the_referenced_article():
    """相手文書の全チャンクではなく、名指しされた条文のチャンクを選ぶ。"""
    import rageval.sampling as sampling
    anchor = _ch("law_a", 0, text="ボイラー則第百一条第二号ロ又はハに掲げる者")
    right = _ch("boiler", 5, text="第百一条 一級ボイラー技士免許を受けるには…")
    noise = [_ch("boiler", i, text=f"第{i}条 無関係な規定") for i in (1, 2, 3)]
    pol = sampling.CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]})
    picked = pol.resolve_named_target(anchor, noise + [right])
    assert picked is not None and picked.chunk_id == right.chunk_id


def test_resolve_named_target_returns_none_when_article_absent():
    """名指しの条番号が相手文書に無ければ、対を成立させない。"""
    import rageval.sampling as sampling
    anchor = _ch("law_a", 0, text="ボイラー則第百一条に掲げる者")
    cands = [_ch("boiler", i, text=f"第{i}条 別の規定") for i in (1, 2)]
    pol = sampling.CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]})
    assert pol.resolve_named_target(anchor, cands) is None


def test_resolve_named_target_similarity_fallback():
    """条番号が無い名指しは類似度で1点に絞る (下限未満なら不成立)。"""
    import rageval.sampling as sampling
    anchor = _ch("law_a", 0, text="詳細はボイラー則による")
    c1 = _ch("boiler", 1, text="関係の薄い章")
    c2 = _ch("boiler", 2, text="関係の濃い章")
    pol = sampling.CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]})
    sim = lambda c: 0.9 if c.chunk_id == c2.chunk_id else 0.1  # noqa: E731
    # 既定では類似フォールバックを使わない (「類似=関係」の再導入を防ぐ)
    assert pol.resolve_named_target(anchor, [c1, c2], similarity=sim) is None
    picked = pol.resolve_named_target(
        anchor, [c1, c2], similarity=sim, allow_similarity_fallback=True)
    assert picked is not None and picked.chunk_id == c2.chunk_id
    # 明示的に有効化しても、全部下限未満なら None
    assert pol.resolve_named_target(
        anchor, [c1, c2], similarity=lambda c: 0.1,
        allow_similarity_fallback=True) is None


def test_reference_follow_named_join_resolves_to_one_chunk(monkeypatch):
    """ReferenceFollow が相手文書の無関係チャンクを掴まないこと。"""
    import rageval.sampling as sampling
    anchor = _ch("law_a", 0, text="ボイラー則第百一条に掲げる者", refs=["第101条"])
    right = _ch("boiler", 9, text="第百一条 一級ボイラー技士免許…")
    noise = [_ch("boiler", i, text=f"第{i}条 無関係") for i in range(1, 6)]
    pol = sampling.CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]})
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY", pol)
    strat = ReferenceFollow(n=2, force_distinct_doc=True)
    for seed in range(8):
        out = strat.select([anchor, right] + noise, None, random.Random(seed))
        if len(out) == 2:
            assert out[1].chunk_id == right.chunk_id


def test_normalize_rationale_ids_moves_chunk_id():
    """doc_id 欄に書かれた chunk_id を chunk_id 欄へ移し、doc_id を正規化する。"""
    from datetime import datetime
    from rageval.evidence import normalize_rationale_ids
    from rageval.schema import (
        Explainability, GenerationInfo, QAItem, Rationale,
        ReasoningComplexity, RetrievalDifficulty, SourceStructure,
    )
    qa = QAItem(
        qa_id="q", question="Q?", answer="A",
        rationale=[
            Rationale(doc_id="09_kagi__c0118", page=None, text="t1"),
            Rationale(doc_id="09_kagi", page=None, text="t2"),
        ],
        category=[], aspect=[], reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(), source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy", answer_level="Easy", difficulty_rationale="x",
        generation=GenerationInfo(model="m", prompt_version="v", generated_at=datetime.now()),
    )
    assert normalize_rationale_ids(qa) == 1
    assert qa.rationale[0].doc_id == "09_kagi"
    assert qa.rationale[0].chunk_id == "09_kagi__c0118"
    # 同一文書の2チャンクなので「異文書」ではない
    assert len({r.doc_id for r in qa.rationale}) == 1


def test_sync_retrieval_difficulty_normalizes_doc_id():
    """chunk_id を別文書と数えて multi_doc=True を誤付与しないこと。"""
    from datetime import datetime
    from rageval.answer_grounding import sync_retrieval_difficulty
    from rageval.schema import (
        Explainability, GenerationInfo, QAItem, Rationale,
        ReasoningComplexity, RetrievalDifficulty, SourceStructure,
    )
    qa = QAItem(
        qa_id="q", question="Q?", answer="A",
        rationale=[
            Rationale(doc_id="09_kagi__c0118", page=None, text="t1"),
            Rationale(doc_id="09_kagi__c0520", page=None, text="t2"),
        ],
        category=[], aspect=[], reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(multi_doc=True),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Hard", answer_level="Easy", difficulty_rationale="x",
        generation=GenerationInfo(model="m", prompt_version="v", generated_at=datetime.now()),
    )
    sync_retrieval_difficulty(qa)
    assert qa.retrieval_difficulty.multi_doc is False   # 同一文書
    assert qa.retrieval_difficulty.multi_chunk is True  # チャンクは2つ


# ---------------- 2026-08-08 監査 rank 15: 同一文書内の位置差の正本 ----------------

def test_remote_reference_gap_follows_reference_follow_strategy():
    """難易度判定の遠隔閾値が ReferenceFollow.min_same_doc_gap と同じ出所であること。

    回帰: 生成本体・戦略表・難易度判定の3箇所に別々の直値 10 があり、片方だけ
    動かすと「素材は通したのに remote_reference=False で出荷」が静かに起きていた。
    """
    import rageval.sampling as sampling
    assert sampling.remote_reference_gap() == sampling.same_doc_gap_for("reference_follow")
    assert sampling.same_doc_gap_for("standards_reference") is None  # 同一文書の遠隔対は認めない


def test_set_min_same_doc_gap_moves_material_and_difficulty_together():
    """位置差を差し替えると、素材選び・難易度判定の両方が同時に動くこと。"""
    import rageval.sampling as sampling
    from rageval.sampling import compute_retrieval_difficulty, score_retrieval_level
    original = sampling.remote_reference_gap()
    anchors = [_ch("d1", 0, text="第9条を準用する", refs=["第9条"]), _ch("d1", 5, text="第9条 …")]
    try:
        # 既定 (10) では位置差 5 は遠隔ではない: 素材としても難易度としても不成立
        assert score_retrieval_level(anchors) == "Medium"
        assert compute_retrieval_difficulty(anchors).remote_reference is False
        assert ReferenceFollow(min_same_doc_gap=sampling.same_doc_gap_for("reference_follow")) \
            ._same_doc_ok(anchors[0], anchors[1]) is False
        sampling.set_min_same_doc_gap(4)
        assert sampling.same_doc_gap_for("reference_follow") == 4
        assert score_retrieval_level(anchors) == "Hard"
        assert compute_retrieval_difficulty(anchors).remote_reference is True
        strat = sampling.ASPECT_STRATEGIES["reference_follow"]
        assert strat._same_doc_ok(anchors[0], anchors[1]) is True
        # 戦略の他の設定は失われない
        assert strat.prefer_specific is True and strat.requires_multi_doc is True
    finally:
        sampling.set_min_same_doc_gap(original)


# ---------------- 2026-08-08 監査 rank 26: 跨ぎ必須は戦略側の宣言から導く ----------------

def test_multi_doc_required_aspects_derived_from_strategies():
    """異文書跨ぎ必須の素材キーが戦略表の宣言から導かれること (手書き集合の廃止)。"""
    import rageval.sampling as sampling
    assert sampling.multi_doc_required_aspects() == {
        "multi_source_integration", "multi_doc_reference", "multi_hop", "reference_follow",
    }


def test_multi_doc_required_aspects_picks_up_new_strategy(monkeypatch):
    """新しい多文書戦略を足したら、手を入れずに早期棄却の対象へ入ること。"""
    import rageval.sampling as sampling
    table = dict(sampling.ASPECT_STRATEGIES)
    table["新戦略"] = MultiDocByEmbedding(n=2, force_distinct_doc=True, requires_multi_doc=True)
    monkeypatch.setattr(sampling, "ASPECT_STRATEGIES", table)
    assert "新戦略" in sampling.multi_doc_required_aspects()


# ---------------- 2026-08-08 監査 rank 25: single_fact の素材キー ----------------

def test_kg_single_fact_key_registered_and_unfiltered():
    """single_fact 用の素材キーが登録され、数量系チャンクへ絞られないこと。

    回帰: 既存観点 "quantitative_calc" の流用で互換述語 _has_numeric_calc が掛かり、
    アンカーが「数値2個以上+比較語」を含むチャンクだけに縮んでいた。
    """
    from rageval.sampling import ASPECT_COMPATIBILITY, ASPECT_STRATEGIES, find_compatible_chunks
    chunks = [
        _ch("d", 0, text="点検は毎日行う。"),                    # 数値なし
        _ch("d", 1, text="厚さは 3mm 以上 5mm 以下とする。"),     # 数量系
    ]
    assert "kg_single_fact" in ASPECT_COMPATIBILITY          # 既定 _always への素通りに頼らない
    assert "kg_single_fact" not in ASPECT_STRATEGIES         # 1チャンクで完結する型
    assert len(find_compatible_chunks("kg_single_fact", chunks)) == 2
    assert len(find_compatible_chunks("quantitative_calc", chunks)) == 1  # 旧キーは絞る
    sampler = AnchorSampler(chunks, embeddings=None)
    out = sampler.select("kg_single_fact", random.Random(0))
    assert out is not None and len(out) == 1


# ---------------- 2026-08-08 監査 rank 27: 規格団体の接頭辞は1定義 ----------------

def test_standards_ref_and_global_ref_share_one_prefix_set():
    """「規格共有で対を組んでよい」と「規格参照である」の判定が食い違わないこと。

    回帰: _GLOBAL_REF_RE が JPI/JEC/JEM/API/EN/NFPA を含むのに、観点側の列挙は
    JIS|ISO|IEC|ASME だけで、同じチャンクが両方の規則で別扱いになっていた。
    """
    import rageval.sampling as sampling
    from rageval.sampling import _has_standards_ref, _is_global_ref
    for prefix in sampling.STANDARD_PREFIXES:
        ref = f"{prefix} 1234"
        assert _is_global_ref(ref), prefix
        assert _has_standards_ref(_ch("d", 0, refs=[ref])), prefix
    assert _has_standards_ref(_ch("d", 0, refs=["JPI-7S-77"])) is True
    assert _has_standards_ref(_ch("d", 0, refs=["第3章"])) is False


def test_set_standard_prefixes_moves_both_rules():
    """接頭辞を差し替えると、異文書対の成立判定と観点の適合述語が同時に動くこと。

    社内文書番号の体系へ替えたときに、コードを直さないと供給が消える状態にしない。
    """
    import rageval.sampling as sampling
    from rageval.sampling import _has_standards_ref, _is_global_ref
    original = sampling.STANDARD_PREFIXES
    try:
        sampling.set_standard_prefixes(["SPEC"])
        assert _is_global_ref("SPEC-001") is True
        assert _has_standards_ref(_ch("d", 0, refs=["SPEC-001"])) is True
        assert _is_global_ref("JIS B 1083") is False        # 台帳に無い体系は認めない
        assert _has_standards_ref(_ch("d", 0, refs=["JIS B 1083"])) is False
    finally:
        sampling.set_standard_prefixes(original)
    assert _is_global_ref("JIS B 1083") is True


def test_apply_standard_prefixes_from_ledger(tmp_path):
    """台帳 (data/seeds) から接頭辞を読み込めること。無ければ既定のまま。"""
    import json as _json
    import rageval.sampling as sampling
    original = sampling.STANDARD_PREFIXES
    ledger = tmp_path / "standard_prefixes.json"
    ledger.write_text(_json.dumps(["JIS", "SPEC"]), encoding="utf-8")
    try:
        assert sampling.apply_standard_prefixes(tmp_path / "missing.json") == original
        assert sampling.apply_standard_prefixes(ledger) == ("JIS", "SPEC")
    finally:
        sampling.set_standard_prefixes(original)


# ---------------- 2026-08-08 監査 rank 30: 台帳のパスと不在時の警告 ----------------

def test_seed_ledger_paths_share_one_root():
    """台帳の既定パスが1つの置き場から作られていること (直書きの再発防止)。"""
    from rageval.generate import DEFAULT_ANCHOR_PAIRS, DEFAULT_DOC_ALIASES, DEFAULT_SEEDS
    from rageval.sampling import (
        DEFAULT_ANCHOR_PAIRS_PATH, DEFAULT_DOC_ALIASES_PATH, SEEDS_DIR,
    )
    assert DEFAULT_ANCHOR_PAIRS == DEFAULT_ANCHOR_PAIRS_PATH == SEEDS_DIR / "anchor_pairs_manual_v1.jsonl"
    assert DEFAULT_DOC_ALIASES == DEFAULT_DOC_ALIASES_PATH == SEEDS_DIR / "doc_aliases.json"
    from pathlib import Path as _Path
    assert _Path(DEFAULT_SEEDS).parent == SEEDS_DIR


def test_load_cross_doc_policy_warns_on_missing_ledger(tmp_path, capsys):
    """台帳が見つからないとき、黙って空規則にせず警告を出すこと。"""
    from rageval.sampling import load_cross_doc_policy
    policy = load_cross_doc_policy(tmp_path / "no_pairs.jsonl", tmp_path / "no_aliases.json")
    out = capsys.readouterr().out
    assert "解決済みエッジ台帳" in out and "文書別名台帳" in out
    assert policy.allowed_chunk_pairs == set() and policy.doc_aliases == {}


def test_article_match_rejects_branch_numbered_articles():
    """回帰: 「第56条」が「第56条の6の2」(別の条) に部分一致しないこと。
    kg_missing 供給の偽参照 (2026-08-03 発覚) の再発防止。"""
    from rageval.sampling import article_number, _ARTICLE_AFTER_ALIAS_RE
    assert _ARTICLE_AFTER_ALIAS_RE.search("第五十六条の六の二第一項") is None
    assert _ARTICLE_AFTER_ALIAS_RE.search("第五十六条に基づき") is not None
    assert _ARTICLE_AFTER_ALIAS_RE.search("第56条の2") is None
    assert _ARTICLE_AFTER_ALIAS_RE.search("第56条を準用") is not None
