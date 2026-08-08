"""既存 KG の経路選定と provenance 検証。"""

from __future__ import annotations

import pytest

from rageval.kg import (
    DEFAULT_MAX_PAGERANK,
    MAX_SUPPORT_CANDIDATES,
    KGPath,
    KGStar,
    PathEdge,
    SupportTruncation,
    candidate_support_chunks,
    count_two_hop_paths,
    hub_pagerank_threshold,
    iter_star_groups,
    iter_two_hop_paths,
    verify_edge,
    verify_edge_relation,
    verify_path_provenance,
    verify_star_provenance,
)
from rageval.llm import LLMError


def _edge(**overrides) -> PathEdge:
    values = dict(
        subject="設備A",
        predicate="REQUIRES",
        object="検査B",
        source_chunks=["c1"],
        extraction_count=2,
        subject_forms=["設備A"],
        object_forms=["検査B"],
    )
    values.update(overrides)
    return PathEdge(**values)


def test_candidate_support_requires_both_endpoint_mentions():
    edge = _edge(source_chunks=["only_subject", "both", "missing"])
    texts = {
        "only_subject": "設備Aについて定める。",
        "both": "設備Aは、使用前に検査Bを必要とする。",
    }
    assert candidate_support_chunks(edge, texts) == ["both"]
    assert verify_edge(edge, texts) == "both"


def test_verify_edge_relation_rejects_cooccurrence_without_relation():
    edge = _edge()

    def judge(**kwargs):
        assert "共起は false" in kwargs["prompt"]
        return {"entailed": False, "reason": "両語の列挙だけで必要関係は書かれていない"}

    assert verify_edge_relation(
        edge, "c1", "設備Aと検査Bを列挙する。", model="judge", llm=judge,
    ) is None


def test_verify_path_provenance_tries_another_source_and_records_reason():
    edge1 = _edge(source_chunks=["c1_bad", "c1_good"])
    edge2 = _edge(
        subject="検査B", predicate="PRODUCES", object="記録C",
        source_chunks=["c2"], subject_forms=["検査B"], object_forms=["記録C"],
    )
    texts = {
        "c1_bad": "設備Aと検査Bを対象とする。",
        "c1_good": "設備Aは検査Bを必要とする。",
        "c2": "検査Bを実施したら記録Cを作成する。",
    }

    def judge(**kwargs):
        prompt = kwargs["prompt"]
        if "c1_bad" in prompt:
            return '{"entailed": false, "reason": "共起のみ"}'
        return '{"entailed": true, "reason": "述語と向きが明記されている"}'

    result = verify_path_provenance(
        KGPath([edge1, edge2]), texts, model="judge", llm=judge,
    )
    assert result is not None
    assert [item.source_chunk_id for item in result] == ["c1_good", "c2"]
    assert all(item.reason == "述語と向きが明記されている" for item in result)


def test_verify_edge_relation_fails_closed_on_invalid_judge_output():
    with pytest.raises(LLMError, match="Invalid provenance response"):
        verify_edge_relation(
            _edge(), "c1", "設備Aは検査Bを必要とする。",
            model="judge", llm=lambda **_: {"entailed": "yes", "reason": "ok"},
        )


# ---------- 始点閉包 (対象スコープが閉じているか) ----------

def test_start_closure_passes_when_start_appears_in_terminal_chunk():
    """始点の表記が終端辺の裏付けチャンクに現れる → 対象が閉じている。"""
    from rageval.kg import passes_start_closure
    path = KGPath(edges=[
        _edge(subject="容器", object="容器再検査",
              subject_forms=["容器"], object_forms=["容器再検査"],
              source_chunks=["law"]),
        _edge(subject="容器再検査", object="LPG容器",
              subject_forms=["容器再検査"], object_forms=["LPG容器"],
              source_chunks=["rule"]),
    ])
    texts = {
        "law": "容器は、期間を経過したときは容器再検査を受けなければならない。",
        "rule": "LPG容器の容器再検査の期間は、経過年数二十年未満のものは六年とする。",
    }
    assert passes_start_closure(path, texts) is True


def test_start_closure_rejects_scope_jump():
    """回帰: 岩盤貯槽→耐圧試験→最高使用圧力 型。終端の原文は燃料電池設備に
    ついて語っており、始点 (岩盤貯槽) が現れない → 別対象への飛び移り。"""
    from rageval.kg import passes_start_closure
    path = KGPath(edges=[
        _edge(subject="岩盤貯槽", object="耐圧試験",
              subject_forms=["岩盤貯槽"], object_forms=["耐圧試験"],
              source_chunks=["ganban"]),
        _edge(subject="耐圧試験", object="最高使用圧力",
              subject_forms=["耐圧試験"], object_forms=["最高使用圧力"],
              source_chunks=["fuel_cell"]),
    ])
    texts = {
        "ganban": "岩盤貯槽は、耐圧部分の強度を確認するため耐圧試験を行う。",
        "fuel_cell": "燃料電池設備の耐圧試験は、最高使用圧力の1.3倍の水圧により行う。",
    }
    assert passes_start_closure(path, texts) is False


def test_start_closure_single_edge_paths_pass():
    from rageval.kg import passes_start_closure
    path = KGPath(edges=[_edge(source_chunks=["c1"])])
    assert passes_start_closure(path, {"c1": "無関係"}) is True


# ---------- 終端根拠の語彙断絶 (閉包チャンクと根拠チャンクの役割分離) ----------

def _leaky_path() -> KGPath:
    """終端辺に閉包チャンク (始点あり) と語彙断絶チャンク (始点なし) が両方ある経路。"""
    return KGPath(edges=[
        _edge(subject="温水ボイラー", object="水圧試験",
              subject_forms=["温水ボイラー"], object_forms=["水圧試験"],
              source_chunks=["start"]),
        _edge(subject="水圧試験", object="最高使用圧力",
              subject_forms=["水圧試験"], object_forms=["最高使用圧力"],
              source_chunks=["closure", "clean"]),
    ])


_LEAK_TEXTS = {
    "start": "温水ボイラーは、使用前に水圧試験を行わなければならない。",
    "closure": "温水ボイラーの水圧試験は、最高使用圧力の1.5倍で行う。",
    "clean": "水圧試験は、最高使用圧力の1.5倍の圧力で実施し、保持時間を記録する。",
}


def test_leak_free_terminal_chunks_excludes_closure_chunk():
    """閉包チャンク (始点表記あり) は根拠候補から外れ、語彙断絶チャンクだけが残る。"""
    from rageval.kg import leak_free_terminal_chunks
    assert leak_free_terminal_chunks(_leaky_path(), _LEAK_TEXTS) == ["clean"]


def test_leak_free_terminal_chunks_empty_when_all_candidates_leak():
    """全候補に始点表記が漏れている経路は語彙断絶の根拠を持てない (素材として棄却)。"""
    from rageval.kg import leak_free_terminal_chunks
    texts = dict(_LEAK_TEXTS)
    texts["clean"] = "温水ボイラーの水圧試験は最高使用圧力の1.5倍とする。"
    assert leak_free_terminal_chunks(_leaky_path(), texts) == []


def test_leak_free_terminal_single_edge_passes_through():
    """1辺経路 (negation 等) には適用しない — 共起候補をそのまま返す。"""
    from rageval.kg import leak_free_terminal_chunks
    path = KGPath(edges=[_edge(source_chunks=["c1"])])
    texts = {"c1": "設備Aは検査Bを必要とする。"}
    assert leak_free_terminal_chunks(path, texts) == ["c1"]


def test_verify_path_provenance_terminal_leak_free_picks_clean_chunk():
    """段0が閉包チャンクでなく語彙断絶チャンクを終端の根拠に採ること。
    素材列挙が leak-free 候補ありの経路だけを通しても、段0が閉包チャンクを
    根拠に採れば質問語が正解チャンクに漏れる (2026-08-07 の実測に基づく回帰)。"""
    def judge(**kwargs):
        return {"entailed": True, "reason": "関係が明記されている"}

    result = verify_path_provenance(
        _leaky_path(), _LEAK_TEXTS, model="judge", llm=judge,
        terminal_leak_free=True,
    )
    assert result is not None
    assert result[-1].source_chunk_id == "clean"

    # 既定 (フラグなし) は従来どおり閉包チャンクが先に採られる
    result_default = verify_path_provenance(
        _leaky_path(), _LEAK_TEXTS, model="judge", llm=judge,
    )
    assert result_default is not None
    assert result_default[-1].source_chunk_id == "closure"


# ---------- 星の同一文書判定 (doc_id の正本を使う。監査 rank 14) ----------

class _FakeSession:
    """Neo4j セッションの代役。渡した行をそのまま返し、引数を控える。"""

    def __init__(self, rows, single=None):
        self._rows = rows
        self._single = single
        self.calls: list[dict] = []

    def run(self, _query, **params):
        self.calls.append(params)
        session = self

        class _Result:
            def __iter__(self_inner):
                return iter(session._rows)

            def single(self_inner):
                return session._single

        return _Result()


def _star_rows(chunk_ids: list[str], center: str = "定期自主検査") -> list[dict]:
    """1中心・同型関係の枝を chunk_ids と1対1で作る (枝名は互いに異なる)。"""
    return [
        {
            "center": center, "rel": "REQUIRES", "branch": f"要件{i}",
            "a_sf": [center], "b_sf": [f"要件{i}"],
            "sc": [cid], "ec": 3,
        }
        for i, cid in enumerate(chunk_ids)
    ]


def test_star_rejects_mixed_documents_whose_doc_id_contains_the_separator():
    """回帰: doc_id は利用者のファイル名 (path.stem) なので "__" を含みうる。
    chunk_id を split("__")[0] で切ると別文書の枝が同一文書に見え、混線した星が
    そのまま素材になっていた (監査 rank 14)。doc_id_of で切れば別文書と判る。"""
    rows = _star_rows([
        "安全管理規程__第1編__c0001",
        "安全管理規程__第1編__c0002",
        "安全管理規程__第2編__c0003",   # 別文書 (旧実装では同じ「安全管理規程」)
    ])
    assert list(iter_star_groups(_FakeSession(rows))) == []


def test_star_accepts_when_all_branches_share_one_document():
    rows = _star_rows([
        "安全管理規程__第1編__c0001",
        "安全管理規程__第1編__c0002",
        "安全管理規程__第1編__c0003",
    ])
    stars = list(iter_star_groups(_FakeSession(rows)))
    assert [s.center for s in stars] == ["定期自主検査"]
    assert len(stars[0].edges) == 3


def test_star_doc_id_map_is_the_authority_over_chunk_id_parsing():
    """回帰: 区切りの違う chunk_id を持ち込むと全枝が別文書に見えて星が全滅した。
    正本は Chunk.doc_id なので、対応表を渡せば chunk_id の書式に依存しない。"""
    ids = ["plant-001#0001", "plant-001#0002", "plant-001#0003"]
    rows = _star_rows(ids)
    same_doc = {cid: "plant-001" for cid in ids}
    assert len(list(iter_star_groups(_FakeSession(rows),
                                     doc_id_by_chunk=same_doc))) == 1

    # 逆に正本が別文書と言うなら、chunk_id が同じ形でも混線として落とす
    mixed = dict(same_doc)
    mixed[ids[-1]] = "plant-002"
    assert list(iter_star_groups(_FakeSession(_star_rows(ids)),
                                 doc_id_by_chunk=mixed)) == []


def test_star_position_spread_uses_chunker_position_of():
    """位置幅の比較も chunker の position_of 経由 (自前の正規表現を持たない)。"""
    rows = _star_rows([
        "plant__c0001", "plant__c0002", "plant__c0100",
    ])
    assert list(iter_star_groups(_FakeSession(rows), max_position_spread=5)) == []
    assert len(list(iter_star_groups(_FakeSession(_star_rows([
        "plant__c0001", "plant__c0002", "plant__c0100",
    ])), max_position_spread=200))) == 1


# ---------- 裏付け候補の上限 (監査 rank 16) ----------

def _many_candidate_edge(n: int) -> PathEdge:
    return _edge(source_chunks=[f"c{i}" for i in range(n)])


def _many_candidate_texts(n: int) -> dict[str, str]:
    return {f"c{i}": "設備Aは検査Bを必要とする。" for i in range(n)}


def test_path_provenance_candidate_limit_is_one_named_constant():
    """回帰: 上限が直値 4 で2箇所に散っていた。既定は MAX_SUPPORT_CANDIDATES 1つ。"""
    assert MAX_SUPPORT_CANDIDATES == 4
    tried: list[str] = []

    def judge(**kwargs):
        for cid in _many_candidate_texts(8):
            if f"source_chunk_id: {cid}\n" in kwargs["prompt"]:
                tried.append(cid)
        return {"entailed": False, "reason": "共起のみ"}

    path = KGPath(edges=[_many_candidate_edge(8)])
    assert verify_path_provenance(
        path, _many_candidate_texts(8), model="judge", llm=judge) is None
    assert tried == ["c0", "c1", "c2", "c3"]


def test_path_provenance_limit_can_be_raised_and_truncation_is_counted():
    """5番目にしか含意が無い辺は既定の上限で黙って落ちる。落ちた事実 (打ち切り
    件数) が残らないと、供給が痩せた原因が上限か素材不良か切り分けられない。"""
    texts = _many_candidate_texts(8)

    def judge(**kwargs):
        ok = "source_chunk_id: c4\n" in kwargs["prompt"]
        return {"entailed": ok, "reason": "関係が明記されている" if ok else "共起のみ"}

    stats = SupportTruncation()
    assert verify_path_provenance(
        KGPath(edges=[_many_candidate_edge(8)]), texts,
        model="judge", llm=judge, truncation=stats) is None
    assert (stats.edges_seen, stats.edges_truncated, stats.candidates_dropped) == (1, 1, 4)
    assert "打ち切り" in stats.summary()

    result = verify_path_provenance(
        KGPath(edges=[_many_candidate_edge(8)]), texts,
        model="judge", llm=judge, max_support_candidates=6)
    assert result is not None
    assert result[0].source_chunk_id == "c4"


def test_star_provenance_shares_the_same_candidate_limit():
    """星側の上限も同じ定数から来る (連鎖と別値だと型ごとに合格条件がずれる)。
    5番目の候補にしか含意が無い枝は既定の上限で全滅し、星ごと消えてしまう。"""
    star = KGStar(
        center="設備A", relation="REQUIRES",
        edges=[_edge(object=f"検査{b}", object_forms=[f"検査{b}"],
                     source_chunks=[f"b{b}c{i}" for i in range(8)])
               for b in range(3)],
    )
    texts = {f"b{b}c{i}": f"設備Aは検査{b}を必要とする。"
             for b in range(3) for i in range(8)}

    def judge(**kwargs):
        ok = any(f"source_chunk_id: b{b}c4\n" in kwargs["prompt"] for b in range(3))
        return {"entailed": ok, "reason": "関係が明記されている" if ok else "共起のみ"}

    stats = SupportTruncation()
    assert verify_star_provenance(
        star, texts, model="judge", llm=judge, truncation=stats) is None
    assert (stats.edges_seen, stats.edges_truncated, stats.candidates_dropped) == (3, 3, 12)

    verified = verify_star_provenance(
        star, texts, model="judge", llm=judge, max_support_candidates=6)
    assert verified is not None
    assert [v.source_chunk_id for v in verified] == ["b0c4", "b1c4", "b2c4"]


# ---------- ハブ判定の閾値 (監査 rank 18) ----------

def test_max_pagerank_default_comes_from_one_constant():
    """回帰: 0.0005 が3箇所の直値だった。列挙と件数計測が同じ定数を使うこと。"""
    import inspect
    assert (inspect.signature(iter_two_hop_paths).parameters["max_pagerank"].default
            == DEFAULT_MAX_PAGERANK)
    sess = _FakeSession(rows=[], single={"n": 0})
    count_two_hop_paths(sess)
    assert sess.calls[0]["max_pr"] == DEFAULT_MAX_PAGERANK


def test_hub_pagerank_threshold_follows_the_actual_distribution():
    """絶対値はコーパス規模で意味が変わるので、分位で引き直せる足場を持つ。"""
    rows = [{"pr": v} for v in (0.001, 0.002, 0.003, 0.004, 0.005,
                                0.006, 0.007, 0.008, 0.009, 0.010)]
    # 上位2割を弾く → 通してよい上限は下から8番目 (0.008)
    assert hub_pagerank_threshold(_FakeSession(rows), top_fraction=0.2) == 0.008
    # 分布が無い KG では None (呼び出し側が「ハブ判定なし」に倒せる)
    assert hub_pagerank_threshold(_FakeSession([])) is None
