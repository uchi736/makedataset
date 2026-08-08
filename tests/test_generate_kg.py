"""検証済み KG 経路からの QA 生成。"""

from __future__ import annotations

import json
from pathlib import Path

from rageval.generate_kg import generate_from_paths, path_novelty
from rageval.kg import KGPath, PathEdge
from rageval.schema import Chunk, QAItem


def _fixture_path() -> KGPath:
    return KGPath(edges=[
        PathEdge(
            "ポンプA", "REQUIRES", "検査B", ["doc1__c0000"], 3,
            ["ポンプA"], ["検査B"],
        ),
        PathEdge(
            "検査B", "PRODUCES", "記録C", ["doc2__c0000"], 2,
            ["検査B"], ["記録C"],
        ),
    ])


def _write_chunks(directory: Path) -> None:
    directory.mkdir()
    chunks = [
        Chunk(
            chunk_id="doc1__c0000", doc_id="doc1",
            text="ポンプAは、運転開始前に検査Bを必要とする。",
        ),
        Chunk(
            chunk_id="doc2__c0000", doc_id="doc2",
            text="検査Bを実施した場合は、記録Cを作成する。",
        ),
    ]
    (directory / "fixture.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8",
    )


_COMPOSE_OK = {"valid": True,
               "relation": "ポンプAは運転開始前に検査Bを要し、検査Bの実施は記録Cの作成を求める"}


def _generation_result() -> dict:
    return {
        "question": "ポンプAの運転を開始する前の対応を終えた場合、何を作成しますか？",
        # 終端ノード名の復唱だけの回答は _vacuous_answer に弾かれるため、
        # 実質のある文で書く (実運用でもこの形を要求している)。
        "answer": "実施した検査の記録として記録Cを作成する必要があります。",
        "rationale": [
            {
                "doc_id": "doc1", "chunk_id": "doc1__c0000", "page": None,
                "text": "ポンプAは、運転開始前に検査Bを必要とする。",
            },
            {
                "doc_id": "doc2", "chunk_id": "doc2__c0000", "page": None,
                "text": "検査Bを実施した場合は、記録Cを作成する。",
            },
        ],
        "difficulty_rationale": "必要チャンク=2、推論ステップ=2",
    }


def test_generate_from_paths_persists_verified_kg_provenance(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    out_dir = tmp_path / "out"
    _write_chunks(chunks_dir)

    output = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge", kg_snapshot="snapshot-1",
        llm=lambda **_: _generation_result(),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {
            "entailed": True, "reason": "原文が有向関係を明記している",
        },
        max_items=1,
    )

    qa = QAItem.model_validate_json(output.read_text(encoding="utf-8").strip())
    assert qa.kg_query_type == "multi_hop"
    assert qa.kg_novelty == "procedural_relation"
    assert qa.reasoning_complexity.multi_step is True
    assert qa.retrieval_difficulty.multi_doc is True
    assert qa.explainability.evidence_strictness == "multi-ref"
    assert qa.kg_path is not None
    assert qa.kg_path.snapshot_id == "snapshot-1"
    assert qa.kg_path.verifier_model == "judge"
    assert [e.source_chunk_id for e in qa.kg_path.edges] == [
        "doc1__c0000", "doc2__c0000",
    ]


def test_generate_from_paths_rejects_rationale_outside_verified_support(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    out_dir = tmp_path / "out"
    _write_chunks(chunks_dir)
    bad = _generation_result()
    bad["rationale"][1]["chunk_id"] = "not-a-support"

    output = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge",
        llm=lambda **_: bad,
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert output.read_text(encoding="utf-8") == ""


def test_path_novelty_keeps_nonprocedural_dependency_unknown():
    path = KGPath([PathEdge("A", "APPLIES_TO", "B", ["c"], 2)])
    assert path_novelty(path) == "unknown_relation"


def test_generate_from_paths_skips_paths_used_in_existing_batches(tmp_path: Path):
    """追い足し実行が、既存バッチと同じ経路から重複問を作らないこと。"""
    chunks_dir = tmp_path / "chunks"
    out_dir = tmp_path / "out"
    _write_chunks(chunks_dir)

    first = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge", kg_snapshot="s",
        llm=lambda **_: _generation_result(),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert first.read_text(encoding="utf-8").strip()  # 1問書けた

    calls = []
    second = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge", kg_snapshot="s",
        llm=lambda **kw: calls.append(1) or _generation_result(),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert second.read_text(encoding="utf-8") == ""   # 同じ経路は使わない
    assert not calls                                   # 生成LLM も呼ばれない


def test_compose_reject_stops_before_qa_generation(tmp_path: Path):
    """段1が不成立と判定した経路は、段2 (QA作成) を呼ばずに捨てられること。"""
    chunks_dir = tmp_path / "chunks"
    out_dir = tmp_path / "out"
    _write_chunks(chunks_dir)

    qa_calls = []
    output = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge",
        llm=lambda **kw: qa_calls.append(1) or _generation_result(),
        compose_llm=lambda **_: {"valid": False, "reason": "適用範囲が別の設備"},
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert output.read_text(encoding="utf-8") == ""
    assert not qa_calls   # 段2は呼ばれない


def test_composition_reason_persisted(tmp_path: Path):
    """段1の関係文が kg_path.composition_reason に保存されること。"""
    chunks_dir = tmp_path / "chunks"
    out_dir = tmp_path / "out"
    _write_chunks(chunks_dir)
    output = generate_from_paths(
        [_fixture_path()], chunks_dir, out_dir,
        model="generator", provenance_model="judge", kg_snapshot="s",
        llm=lambda **_: _generation_result(),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    qa = QAItem.model_validate_json(output.read_text(encoding="utf-8").strip())
    assert qa.kg_path.composition_reason == _COMPOSE_OK["relation"]
    assert qa.kg_path.shape == "chain"
    assert "接続根拠" in qa.difficulty_rationale


def test_checks_table_multi_hop_keeps_frozen_rules():
    """凍結: multi_hop の型別検品に中間露出と構造 (終端集中) が入っていること。
    テーブル化で規則が落ちていないかの回帰。"""
    from rageval.generate_kg import CHECKS_BY_TYPE, COMMON_CHECKS
    common_names = [n for n, _ in COMMON_CHECKS]
    assert common_names == ["根拠不適合", "ファイル名露出", "関係文丸写し",
                            "重複根拠", "無内容回答", "造語"]
    mh_names = [n for n, _ in CHECKS_BY_TYPE["multi_hop"]]
    assert "中間露出" in mh_names and "構造" in mh_names


def _write_decoy_chunks(directory: Path) -> None:
    directory.mkdir()
    chunks = [
        Chunk(chunk_id="syn01_alpha__c0000", doc_id="syn01_alpha", position=0,
              text="溶接線の始点から十五メートルごとに検査する。始点及び終端は入念に確認する。"),
        Chunk(chunk_id="syn02_beta__c0000", doc_id="syn02_beta", position=0,
              text="貯蔵設備の周囲には防消火設備を設ける。"),
    ]
    (directory / "fixture.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")


def test_generate_from_decoys_keeps_reject_reason_intact(tmp_path: Path):
    """回帰 (監査 rank 8): 段1の不成立理由を伏せ字化して保存しないこと。

    旧実装は理由文から「始点」「終端」を無条件に空文字へ置換しており、
    data/kg_decoy_v1 の 4/4 件で composition_reason が
    「からへの関係が成立しない」に壊れたまま出荷されていた。
    """
    from rageval.generate_kg import generate_from_decoys

    chunks_dir = tmp_path / "chunks"
    _write_decoy_chunks(chunks_dir)
    a = Chunk(chunk_id="syn01_alpha__c0000", doc_id="syn01_alpha", position=0,
              text="溶接線の始点から十五メートルごとに検査する。始点及び終端は入念に確認する。")
    b = Chunk(chunk_id="syn02_beta__c0000", doc_id="syn02_beta", position=0,
              text="貯蔵設備の周囲には防消火設備を設ける。")
    reason = "始点と終端に関する溶接の規定と、貯蔵設備の規定に関係は無い"

    out = generate_from_decoys(
        [(a, b, 0.82)], chunks_dir, tmp_path / "out",
        model="gen", provenance_model="judge", kg_snapshot="s",
        compose_llm=lambda **_: {"valid": False, "reason": reason},
        llm=lambda **_: {
            "question": "溶接線の始点から終端までの間隔の定めは、貯蔵設備にも適用されますか。",
            "answer": "適用されない。両文書に関係を示す規定は確認できない。",
            "rationale": [
                {"doc_id": "syn01_alpha", "chunk_id": "syn01_alpha__c0000",
                 "page": None, "text": "溶接線の始点から十五メートルごとに検査する。"},
                {"doc_id": "syn02_beta", "chunk_id": "syn02_beta__c0000",
                 "page": None, "text": "貯蔵設備の周囲には防消火設備を設ける。"},
            ],
            "difficulty_rationale": "類似だが無関係",
        },
    )
    qa = QAItem.model_validate_json(out.read_text(encoding="utf-8").strip())
    assert qa.kg_path.composition_reason == f"(関係なし) {reason}"
    assert "始点" in qa.kg_path.composition_reason
    assert "終端" in qa.kg_path.composition_reason
    assert qa.kg_path.source == "kg_decoy"


def test_find_decoy_pairs_thresholds_are_named_and_logged(capsys):
    """回帰 (監査 rank 10): 3つの閾値が引数で動き、採用値と候補数が記録に残る。"""
    import numpy as np

    from rageval.generate_kg import (
        DECOY_MAX_PER_DOC_PAIR, DECOY_MIN_POSITION_GAP, DECOY_SIM_FLOOR,
        find_decoy_pairs,
    )

    class _Policy:
        def named_reference(self, chunk, doc_id):
            return None

    chunks = [
        Chunk(chunk_id="a__c0000", doc_id="a", position=0, text="甲"),
        Chunk(chunk_id="b__c0000", doc_id="b", position=0, text="乙"),
        Chunk(chunk_id="b__c0001", doc_id="b", position=1, text="丙"),
    ]
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.9, 0.4359]])

    pairs = find_decoy_pairs(chunks, emb, _Policy(), {}, top_n=10)
    # 同一文書 (b__c0000 × b__c0001) は位置差 1 < DECOY_MIN_POSITION_GAP で落ちる
    assert {(x.chunk_id, y.chunk_id) for x, y, _ in pairs} == {
        ("a__c0000", "b__c0000"), ("a__c0000", "b__c0001")}
    log = capsys.readouterr().out
    assert f"類似下限={DECOY_SIM_FLOOR}" in log
    assert f"同一文書内の位置差下限={DECOY_MIN_POSITION_GAP}" in log
    assert f"文書対あたり上限={DECOY_MAX_PER_DOC_PAIR}" in log

    # 文書対あたりの上限で削れること
    assert len(find_decoy_pairs(chunks, emb, _Policy(), {},
                                top_n=10, max_per_doc_pair=1)) == 1

    # 目盛りが変わって候補 0 対になったら警告を出す (黙って空ファイルを書かない)
    assert find_decoy_pairs(chunks, emb, _Policy(), {},
                            top_n=10, sim_floor=1.01) == []
    assert "警告: デコイ候補が0対" in capsys.readouterr().out


def test_apply_term_substitution_swaps_and_records():
    """unknown_term 変換: 別表記へ置換し、記録と分類が付くこと (決定論的)。"""
    from rageval.generate_kg import apply_term_substitution
    from rageval.kg import KGPath, PathEdge
    from rageval.schema import KGPathEdgeProvenance, KGPathProvenance, QAItem
    from datetime import datetime

    def make_qa(qa_id: str) -> QAItem:
        return QAItem(
            qa_id=qa_id, question="ボイラー則の検査周期を教えてください。", answer="A",
            rationale=[], reasoning_complexity={}, retrieval_difficulty={},
            source_structure={}, explainability={"evidence_strictness": "multi-ref"},
            retrieval_level="Hard", answer_level="Medium", difficulty_rationale="",
            kg_novelty="unknown_relation",
            kg_path=KGPathProvenance(
                snapshot_id="s", path_text="p", edges=[], verifier_model="j",
                verified_at=datetime.now()),
            generation={"model": "m", "prompt_version": "v",
                        "generated_at": datetime.now().isoformat()},
        )
    path = KGPath(edges=[PathEdge(
        "ボイラー則", "REQUIRES", "検査", ["c1"], 2,
        ["ボイラー則", "ボイラー及び圧力容器安全規則"], ["検査"])])

    qa = make_qa("auto_00000001")   # 0x1 % 10 = 1 < 3 → 当選
    assert apply_term_substitution(qa, path) is True
    assert "ボイラー及び圧力容器安全規則" in qa.question
    assert qa.kg_novelty == "unknown_term"
    assert qa.retrieval_difficulty.vocabulary_mismatch is True
    assert qa.kg_path.term_substitution["canonical"] == "ボイラー則"

    qa2 = make_qa("auto_00000009")  # 0x9 % 10 = 9 → 落選
    assert apply_term_substitution(qa2, path) is False
    assert qa2.kg_novelty == "unknown_relation"
