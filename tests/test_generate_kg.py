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
