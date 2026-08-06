"""既存 KG の経路選定と provenance 検証。"""

from __future__ import annotations

import pytest

from rageval.kg import (
    KGPath,
    PathEdge,
    candidate_support_chunks,
    verify_edge,
    verify_edge_relation,
    verify_path_provenance,
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
