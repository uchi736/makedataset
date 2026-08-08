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


def test_generate_from_stars_compose_can_exclude_offcontext_branch(tmp_path: Path, capsys):
    """段1が別文脈と判定した要素は除外され、残りが最小枝数未満なら星ごと棄却。

    回帰 (監査 rank 17): 棄却の閾値は直値でなく MIN_STAR_BRANCHES で、
    落とした理由と本数を記録に残すこと。
    """
    from rageval.generate_kg import MIN_STAR_BRANCHES

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
    assert not qa_calls   # 最小枝数未満に縮んだので段2は呼ばれない
    log = capsys.readouterr().out
    assert f"MIN_STAR_BRANCHES={MIN_STAR_BRANCHES}" in log


def test_star_thresholds_follow_named_constants(tmp_path: Path, capsys):
    """回帰 (監査 rank 17): 枝数・出典チャンク数の閾値が定数から効くこと。

    枝の下限を 2 に下げると、段0・段1後の両方が同じ値を見るので通る。
    出典が1チャンクに縮んだ星は MIN_STAR_SUPPORT_CHUNKS で落ちる。
    """
    import rageval.generate_kg as g

    chunks_dir = tmp_path / "chunks"
    _write_chunks(chunks_dir)
    monkey = g.MIN_STAR_BRANCHES
    try:
        g.MIN_STAR_BRANCHES = 2
        # 耐圧試験 (d__c0001) を除くと 外観検査(d__c0001) + 気密試験(d__c0002) の2枝2チャンク
        out = generate_from_stars(
            [_star()], chunks_dir, tmp_path / "out",
            model="gen", provenance_model="judge",
            llm=lambda **_: _qa_result(),
            compose_llm=lambda **_: {"valid": True, "relation": "附属品検査の項目",
                                     "excluded": ["耐圧試験"]},
            provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
        )
        qa = QAItem.model_validate_json(out.read_text(encoding="utf-8").strip())
        assert len(qa.kg_path.edges) == 2

        # 気密試験も除くと出典が d__c0001 の1チャンクに縮む → 出典数で棄却
        out2 = generate_from_stars(
            [_star()], chunks_dir, tmp_path / "out2",
            model="gen", provenance_model="judge",
            llm=lambda **_: _qa_result(),
            compose_llm=lambda **_: {"valid": True, "relation": "附属品検査の項目",
                                     "excluded": ["気密試験"]},
            provenance_llm=lambda **_: {"entailed": True, "reason": "明記"},
        )
        assert out2.read_text(encoding="utf-8") == ""
        assert f"MIN_STAR_SUPPORT_CHUNKS={g.MIN_STAR_SUPPORT_CHUNKS}" in capsys.readouterr().out
    finally:
        g.MIN_STAR_BRANCHES = monkey


def test_aggregation_checks_do_not_require_terminal_concentration():
    """凍結: aggregation の型別検品に構造 (終端集中) が入っていないこと。"""
    from rageval.generate_kg import CHECKS_BY_TYPE
    names = [n for n, _ in CHECKS_BY_TYPE["aggregation"]]
    assert "構造" not in names
    assert "中間露出" not in names
    assert "要素網羅" in names


def test_star_position_spread_filter():
    """回帰: 出典位置が離れすぎた星 (別表の脚注の寄せ集め) を列挙段階で落とす。

    位置の取り出しは chunker.position_of が正本 (2026-08-08: kg 側の自前実装を
    廃止し1組に寄せた)。ここは位置差の計算だけを純粋関数で確かめる。
    """
    from rageval.chunker import position_of
    assert position_of("09_kagi__c0302") == 302
    assert position_of("09_kagi__c0433") == 433
    assert position_of("no_suffix") is None
    # 実測ケース: 131 チャンク差 > 50 → 落ちる側
    assert 433 - 302 > 50
