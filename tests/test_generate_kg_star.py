"""aggregation (星型) の生成と型別検品のテスト。"""

from __future__ import annotations

from pathlib import Path

from rageval.generate_kg import generate_from_stars
from rageval.kg import KGStar, PathEdge
from rageval.schema import Chunk, QAItem


def _star() -> KGStar:
    mk = lambda obj, cid: PathEdge(  # noqa: E731
        "附属品検査", "HAS_STEP", obj, [cid], 2, ["附属品検査"], [obj])
    return KGStar(center="附属品検査", relation="HAS_STEP", edges=[
        mk("外観検査", "d__c0001"),
        mk("耐圧試験", "d__c0001"),
        mk("気密試験", "d__c0002"),
    ])


def _write_chunks(directory: Path) -> None:
    directory.mkdir()
    chunks = [
        Chunk(chunk_id="d__c0001", doc_id="d",
              text="附属品検査は、外観検査及び耐圧試験を行うこと。"),
        Chunk(chunk_id="d__c0002", doc_id="d",
              text="附属品検査においては、気密試験を行い漏れがないことを確認する。"),
    ]
    (directory / "fixture.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")


_COMPOSE_OK = {"valid": True, "excluded": [],
               "relation": "附属品検査は外観検査・耐圧試験・気密試験の項目で行うことが定められている"}


def _qa_result(answer: str = "外観検査、耐圧試験、気密試験を行う必要があります。") -> dict:
    return {
        "question": "附属品の検査を実施する際、どのような項目を行う必要がありますか？",
        "answer": answer,
        "rationale": [
            {"doc_id": "d", "chunk_id": "d__c0001", "page": None,
             "text": "附属品検査は、外観検査及び耐圧試験を行うこと。"},
            {"doc_id": "d", "chunk_id": "d__c0002", "page": None,
             "text": "気密試験を行い漏れがないことを確認する。"},
        ],
        "difficulty_rationale": "要素3・チャンク2",
    }


def test_generate_from_stars_happy_path(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir)
    out = generate_from_stars(
        [_star()], chunks_dir, tmp_path / "out",
        model="gen", provenance_model="judge", kg_snapshot="s",
        llm=lambda **_: _qa_result(),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    qa = QAItem.model_validate_json(out.read_text(encoding="utf-8").strip())
    assert qa.kg_query_type == "aggregation"
    assert qa.kg_path.shape == "star"
    assert qa.kg_path.composition_reason.startswith("附属品検査は")
    assert len(qa.kg_path.edges) == 3
    assert qa.retrieval_difficulty.multi_chunk is True


def test_generate_from_stars_rejects_missing_element(tmp_path: Path):
    """要素の取りこぼし (気密試験が回答に無い) は段3で落ちる。"""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir)
    out = generate_from_stars(
        [_star()], chunks_dir, tmp_path / "out",
        model="gen", provenance_model="judge",
        llm=lambda **_: _qa_result(answer="外観検査と耐圧試験を行います。"),
        compose_llm=lambda **_: _COMPOSE_OK,
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert out.read_text(encoding="utf-8") == ""


def test_generate_from_stars_compose_can_exclude_offcontext_branch(tmp_path: Path):
    """段1が別文脈と判定した要素は除外され、残り3未満なら星ごと棄却。"""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir)
    qa_calls = []
    out = generate_from_stars(
        [_star()], chunks_dir, tmp_path / "out",
        model="gen", provenance_model="judge",
        llm=lambda **kw: qa_calls.append(1) or _qa_result(),
        compose_llm=lambda **_: {"valid": True, "relation": "x",
                                 "excluded": ["気密試験"]},
        provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
    )
    assert out.read_text(encoding="utf-8") == ""
    assert not qa_calls   # 3要素未満に縮んだので段2は呼ばれない


def test_aggregation_checks_do_not_require_terminal_concentration():
    """凍結: aggregation の型別検品に構造 (終端集中) が入っていないこと。"""
    from rageval.generate_kg import CHECKS_BY_TYPE
    names = [n for n, _ in CHECKS_BY_TYPE["aggregation"]]
    assert "構造" not in names
    assert "中間露出" not in names
    assert "要素網羅" in names


def test_star_position_spread_filter():
    """回帰: 出典位置が離れすぎた星 (別表の脚注の寄せ集め) を列挙段階で落とす。
    _chunk_position と位置差の計算だけを純粋関数で確かめる。"""
    from rageval.kg import _chunk_position
    assert _chunk_position("09_kagi__c0302") == 302
    assert _chunk_position("09_kagi__c0433") == 433
    assert _chunk_position("no_suffix") is None
    # 実測ケース: 131 チャンク差 > 50 → 落ちる側
    assert 433 - 302 > 50
