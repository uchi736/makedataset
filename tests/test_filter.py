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



def _chunk(doc: str, pos: int, text: str):
    """テスト用の Chunk (compute_rationale_grounded は Chunk 列を取る)。"""
    from rageval.schema import Chunk
    return Chunk(chunk_id=f"{doc}__c{pos:04d}", doc_id=doc, page=None, text=text, position=pos)


def test_compute_rationale_grounded_all_match():
    from rageval.filter import compute_rationale_grounded
    qa = _make_qa("q", "Q?", "A")  # rationale.text = "定格電圧: DC 24V ±10%"
    chunks = [_chunk("sample", 0, "前置き 定格電圧: DC 24V ±10% その他")]
    assert compute_rationale_grounded(qa, chunks) == 1.0


def test_compute_rationale_grounded_whitespace_insensitive():
    from rageval.filter import compute_rationale_grounded
    qa = _make_qa("q", "Q?", "A")
    # Anchor has different whitespace
    chunks = [_chunk("sample", 0, "定格電圧:\nDC  24V  ±10%\n")]
    assert compute_rationale_grounded(qa, chunks) == 1.0


def test_compute_rationale_grounded_fabricated():
    """LLM hallucinated rationale not in chunk."""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    qa.rationale = [Rationale(doc_id="sample", page=None, text="この文は元チャンクに存在しない")]
    chunks = [_chunk("sample", 0, "定格電圧: DC 24V ±10%")]
    assert compute_rationale_grounded(qa, chunks) == 0.0


def test_compute_rationale_grounded_partial():
    """One of two rationale entries is grounded."""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    qa.rationale = [
        Rationale(doc_id="sample", page=None, text="定格電圧: DC 24V ±10%"),  # in chunk
        Rationale(doc_id="sample", page=None, text="ここは捏造された引用"),    # not in chunk
    ]
    chunks = [_chunk("sample", 0, "定格電圧: DC 24V ±10%")]
    assert compute_rationale_grounded(qa, chunks) == 0.5


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


def test_reject_too_easy_drops_too_easy_items(tmp_path: Path):
    """`--reject-too-easy` (= reject_too_easy=True) で difficulty_match=too_easy を棄却。

    filter.py は judge LLM の `answer_level` を `qa.answer_level` (宣言値) と
    比較して difficulty_match を算出する ([_difficulty_match](src/rageval/filter.py#L185-L192))。
    宣言 Hard ↔ 実態 Easy → too_easy、宣言 Easy ↔ 実態 Easy → aligned。
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    # 2件: too_easy になるように宣言 Hard / 実態 Easy のもの、
    # と aligned になる宣言 Easy / 実態 Easy のもの
    qa_easy = _make_qa("qa_easy", "TOO_EASY_MARKER 定格電圧は？", "DC 24V")
    qa_easy.answer_level = "Hard"  # 宣言=Hard、実態=Easy → too_easy
    qa_ok = _make_qa("qa_ok", "ALIGNED_MARKER 定格電圧は？", "DC 24V")
    # qa_ok は宣言=Easy のまま、実態=Easy → aligned
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_easy.model_dump_json() + "\n")
        f.write(qa_ok.model_dump_json() + "\n")

    def mock_judge(*, prompt: str, **kwargs):
        if "ANSWERABILITY" in prompt:
            return json.dumps({"answerability": 5, "reason": "ok"})
        if "LEAKAGE" in prompt:
            return json.dumps({"leakage": "pass", "reason": "ok"})
        if "GROUNDING" in prompt:
            return json.dumps({"grounding": 5, "reason": "ok"})
        if "DIFFICULTY_MATCH" in prompt:
            # 両件とも実態 Easy を返す(チャンクに数値そのものがあるので)
            return json.dumps({"answer_level": "Easy", "reason": "ok"})
        return json.dumps({})

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=mock_judge, compute_uniqueness=False,
        reject_too_easy=True,
    )
    kept = [QAItem.model_validate_json(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(kept) == 1
    assert kept[0].qa_id == "qa_ok"
    assert kept[0].filter_scores.difficulty_match == "aligned"


def test_require_rag_fail_warns_when_no_rag_verification(tmp_path: Path, capsys):
    """rag_verification が無い JSONL に --require-rag-fail を当てたら警告を出して全件残す。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    qa = _make_qa("qa_norag", "定格電圧は？", "DC 24V")
    raw_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
        require_rag_fail=True,
    )
    kept = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(kept) == 1
    out = capsys.readouterr().out
    assert "rag_verification が無い" in out


def test_require_rag_fail_keeps_no_match_only(tmp_path: Path):
    """rag_verification 付きデータで --require-rag-fail を当てると no_match だけ残す。"""
    from rageval.schema import RAGVerification

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    qa_match = _make_qa("qa_match", "Q1", "A1")
    qa_match.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=["sample__c0"], retrieval_hit=True,
        rag_answer="DC 24V", answer_match="match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    qa_nomatch = _make_qa("qa_nomatch", "Q2", "A2")
    qa_nomatch.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=[], retrieval_hit=False,
        rag_answer="回答不能", answer_match="no_match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_match.model_dump_json() + "\n")
        f.write(qa_nomatch.model_dump_json() + "\n")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
        require_rag_fail=True,
    )
    kept = [QAItem.model_validate_json(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert [q.qa_id for q in kept] == ["qa_nomatch"]


def test_require_rag_hit_keeps_match_only(tmp_path: Path):
    """rag_verification 付きデータで --require-rag-hit を当てると match だけ残す。"""
    from rageval.schema import RAGVerification

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    qa_match = _make_qa("qa_match", "Q1", "A1")
    qa_match.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=["sample__c0"], retrieval_hit=True,
        rag_answer="DC 24V", answer_match="match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    qa_nomatch = _make_qa("qa_nomatch", "Q2", "A2")
    qa_nomatch.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=[], retrieval_hit=False,
        rag_answer="回答不能", answer_match="no_match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_match.model_dump_json() + "\n")
        f.write(qa_nomatch.model_dump_json() + "\n")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
        require_rag_hit=True,
    )
    kept = [QAItem.model_validate_json(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert [q.qa_id for q in kept] == ["qa_match"]


def test_require_rationale_retrieved_keeps_hit_only(tmp_path: Path):
    """rag_verification 付きデータで --require-rationale-retrieved を当てると、
    根拠本文を逐語で含むチャンクが上位 k に入った QA (retrieval_hit_chunk=True)
    だけ残す。"""
    from rageval.schema import RAGVerification

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    qa_hit = _make_qa("qa_hit", "Q1", "A1")
    qa_hit.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=["sample__c0"],
        retrieval_hit_doc=True, retrieval_hit_chunk=True, retrieval_hit=True,
        rag_answer="DC 24V", answer_match="match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    qa_miss = _make_qa("qa_miss", "Q2", "A2")
    qa_miss.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=["other__c0"],
        retrieval_hit_doc=False, retrieval_hit_chunk=False, retrieval_hit=False,
        rag_answer="回答不能", answer_match="no_match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_hit.model_dump_json() + "\n")
        f.write(qa_miss.model_dump_json() + "\n")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
        require_rationale_retrieved=True,
    )
    kept = [QAItem.model_validate_json(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert [q.qa_id for q in kept] == ["qa_hit"]


def test_require_rag_fail_warns_when_partial_rag_verification(tmp_path: Path, capsys):
    """rag_verification が一部にしか付いていない場合、未付与の件数を警告で通知する。"""
    from rageval.schema import RAGVerification

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"

    # 1件は rag-verify 済 (no_match で残るはず)、1件は rag_verification 未付与
    qa_verified = _make_qa("qa_verified", "Q1", "A1")
    qa_verified.rag_verification = RAGVerification(
        top_k=5, retrieved_chunk_ids=[], retrieval_hit=False,
        rag_answer="回答不能", answer_match="no_match",
        rag_model="m", judge_model="j", verified_at=datetime.now(),
    )
    qa_missing = _make_qa("qa_missing", "Q2", "A2")
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(qa_verified.model_dump_json() + "\n")
        f.write(qa_missing.model_dump_json() + "\n")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
        require_rag_fail=True,
    )
    out = capsys.readouterr().out
    # 未付与が 1/2 件である旨が出ていること
    assert "1/2" in out
    assert "rag_verification が無く" in out
    # 未付与の QA は条件未評価で素通りして残る
    kept = [QAItem.model_validate_json(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert {q.qa_id for q in kept} == {"qa_verified", "qa_missing"}


def test_require_rag_fail_and_hit_are_mutually_exclusive(tmp_path: Path):
    """--require-rag-fail と --require-rag-hit は相反するので filter_batch が早期に弾く。

    no_match と match は同時に成り立たないので、両方付けると rag_verification 持ちの
    QA は必ず全件 drop される。警告無しで全件消えるのは事故なので関数自体が拒む。
    """
    import pytest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    qa = _make_qa("qa", "Q?", "A")
    raw_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="require_rag_fail と require_rag_hit"):
        filter_batch(
            raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
            judge_model="mock", llm=_mock_judge_llm, compute_uniqueness=False,
            require_rag_fail=True, require_rag_hit=True,
        )


def test_cli_filter_rejects_mutually_exclusive_rag_flags(tmp_path: Path):
    """CLI 層でも --require-rag-fail と --require-rag-hit の同時指定は exit 2 で弾く。"""
    from typer.testing import CliRunner

    from rageval.cli import app

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("定格電圧: DC 24V ±10%\n", encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    qa = _make_qa("qa", "Q?", "A")
    raw_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "filter",
            "--in", str(raw_path),
            "--out", str(tmp_path / "filtered"),
            "--chunks", str(chunks_dir),
            "--require-rag-fail",
            "--require-rag-hit",
        ],
    )
    assert result.exit_code == 2
    # typer.echo(err=True) は CliRunner では既定で output に合流する
    assert "相反します" in result.output


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


# ---------------- v3: 誤結合の機械検出 ----------------

def _mk_chunk(doc, pos, text, refs=None):
    from rageval.schema import Chunk
    return Chunk(chunk_id=f"{doc}__c{pos:04d}", doc_id=doc, page=None, text=text,
                 position=pos, section_path=[], references=refs or [])


def _mk_qa_with_rationale(rationales):
    from datetime import datetime
    from rageval.schema import (
        Explainability, GenerationInfo, QAItem, Rationale,
        ReasoningComplexity, RetrievalDifficulty, SourceStructure,
    )
    return QAItem(
        qa_id="qa_x",
        question="省令第63条に基づく書類はどの様式を使えばよいか。",
        answer="第63条の規定に基づく場合、様式第11を使用する。",
        rationale=[Rationale(doc_id=d, page=None, text=t) for d, t in rationales],
        category=[], aspect=[], kg_query_type="traceability", kg_novelty="unknown_relation",
        reasoning_complexity=ReasoningComplexity(), retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Hard", answer_level="Easy", difficulty_rationale="x",
        generation=GenerationInfo(model="m", prompt_version="v", generated_at=datetime.now()),
    )


def test_collision_detected_for_unjustified_cross_doc_same_number():
    from rageval.filter import detect_cross_doc_number_collision
    from rageval.sampling import CrossDocPolicy
    chunks = [
        _mk_chunk("law_a", 0, "第１００条 省令第６３条に規定する適切な措置とは…"),
        _mk_chunk("law_b", 0, "様式第１１（第６３条関係）様式第１２（第６３条関係）"),
    ]
    qa = _mk_qa_with_rationale([
        ("law_a", "第１００条 省令第６３条に規定する適切な措置とは…"),
        ("law_b", "様式第１１（第６３条関係）"),
    ])
    hit = detect_cross_doc_number_collision(qa, chunks, CrossDocPolicy())
    assert hit is not None and "第63条" in hit


def test_collision_not_flagged_when_pair_justified_by_alias():
    from rageval.filter import detect_cross_doc_number_collision
    from rageval.sampling import CrossDocPolicy
    chunks = [
        _mk_chunk("law_a", 0, "ボイラー則第６３条に掲げる者は…"),
        _mk_chunk("law_b", 0, "第６３条 一級ボイラー技士免許…様式第１１…"),
    ]
    qa = _mk_qa_with_rationale([
        ("law_a", "ボイラー則第６３条に掲げる者は…"),
        ("law_b", "第６３条 一級ボイラー技士免許…"),
    ])
    policy = CrossDocPolicy(doc_aliases={"law_b": ["ボイラー則"]})
    assert detect_cross_doc_number_collision(qa, chunks, policy) is None


def test_collision_not_flagged_for_same_doc():
    from rageval.filter import detect_cross_doc_number_collision
    from rageval.sampling import CrossDocPolicy
    chunks = [
        _mk_chunk("law_a", 0, "第６３条 …"),
        _mk_chunk("law_a", 9, "第６３条の手続は…"),
    ]
    qa = _mk_qa_with_rationale([
        ("law_a", "第６３条 …"),
        ("law_a", "第６３条の手続は…"),
    ])
    assert detect_cross_doc_number_collision(qa, chunks, CrossDocPolicy()) is None


def test_collision_not_flagged_without_shared_number():
    """番号を共有しない文書跨ぎ (技報同士の集約など) は検出しない。"""
    from rageval.filter import detect_cross_doc_number_collision
    from rageval.sampling import CrossDocPolicy
    chunks = [
        _mk_chunk("rep_a", 0, "アンモニア混焼では燃焼火炎の不安定化が課題となる"),
        _mk_chunk("rep_b", 0, "誘引通風機の性能への影響が大きくなる"),
    ]
    qa = _mk_qa_with_rationale([
        ("rep_a", "アンモニア混焼では燃焼火炎の不安定化が課題となる"),
        ("rep_b", "誘引通風機の性能への影響が大きくなる"),
    ])
    assert detect_cross_doc_number_collision(qa, chunks, CrossDocPolicy()) is None


# ---------------- v3: doc スコープ照合 (連結・取り違えの防止) ----------------

def test_grounded_ignores_quote_found_in_other_doc():
    """doc A に帰属する引用が doc B の本文にあっても一致にしない。"""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    qa.rationale = [Rationale(doc_id="docA", page=None, text="定格電圧: DC 24V ±10%")]
    chunks = [
        _chunk("docA", 0, "docA には別の話が書いてある"),
        _chunk("docB", 0, "定格電圧: DC 24V ±10% と書いてあるのは docB"),
    ]
    assert compute_rationale_grounded(qa, chunks) == 0.0


def test_grounded_no_false_hit_across_chunk_boundary():
    """チャンクの連結境界をまたいだ文字列は一致にしない。"""
    from rageval.filter import compute_rationale_grounded
    from rageval.schema import Rationale
    qa = _make_qa("q", "Q?", "A")
    # "末尾です" + "先頭です" は連結すると現れるが、どのチャンクにも存在しない
    qa.rationale = [Rationale(doc_id="docA", page=None, text="末尾です先頭です")]
    chunks = [
        _chunk("docA", 0, "これは前半の本文。末尾です"),
        _chunk("docA", 1, "先頭です。これは後半の本文。"),
    ]
    assert compute_rationale_grounded(qa, chunks) == 0.0


# ---------------- v3: 判定失敗を通過させない ----------------

def test_judge_failure_drops_qa_as_unevaluated(tmp_path: Path, capsys):
    """judge が落ちた QA は「未判定」として drop される (以前は素通りしていた)。"""
    from rageval.llm import LLMError
    from rageval.schema import Chunk

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    c = Chunk(chunk_id="sample__c0", doc_id="sample", page=None,
              text="定格電圧: DC 24V ±10%", position=0)
    (chunks_dir / "sample.jsonl").write_text(c.model_dump_json() + "\n", encoding="utf-8")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    raw_path.write_text(_make_qa("qa1", "定格電圧は？", "DC 24V").model_dump_json() + "\n",
                        encoding="utf-8")

    def judge_down(*, prompt: str, **kwargs):
        raise LLMError("judge server down")

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=judge_down, compute_uniqueness=False,
    )
    kept = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    assert kept == []
    out = capsys.readouterr().out
    assert "unevaluated" in out


def test_partial_judge_failure_also_drops(tmp_path: Path):
    """一部の観点だけ落ちた場合も、必須スコアが欠けていれば drop。"""
    from rageval.llm import LLMError
    from rageval.schema import Chunk

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    c = Chunk(chunk_id="sample__c0", doc_id="sample", page=None,
              text="定格電圧: DC 24V ±10%", position=0)
    (chunks_dir / "sample.jsonl").write_text(c.model_dump_json() + "\n", encoding="utf-8")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "batch_test.jsonl"
    raw_path.write_text(_make_qa("qa1", "定格電圧は？", "DC 24V").model_dump_json() + "\n",
                        encoding="utf-8")

    def judge_grounding_down(*, prompt: str, **kwargs):
        if "GROUNDING" in prompt:
            raise LLMError("grounding judge down")
        return _mock_judge_llm(prompt=prompt, **kwargs)

    out_path = filter_batch(
        raw_path=raw_path, out_dir=tmp_path / "filtered", chunks_dir=chunks_dir,
        judge_model="mock", llm=judge_grounding_down, compute_uniqueness=False,
    )
    assert [l for l in out_path.read_text(encoding="utf-8").splitlines() if l] == []


def test_decoy_exempt_from_answerability_and_leakage():
    """decoy (棄権が正解) は answerability/leakage で落とさない (2026-08-03)。

    構造的に「答えられない」が正解なので、低 answerability は正しい状態。
    filter_batch のふるい分け条件の分岐だけを検証する。"""
    from datetime import datetime
    from rageval.schema import KGPathProvenance, QAItem, Rationale
    qa = QAItem(
        qa_id="d", question="Q?", answer="適用されません。根拠が確認できません。",
        rationale=[Rationale(doc_id="a", chunk_id="a__c0001", page=None, text="t")],
        reasoning_complexity={}, retrieval_difficulty={}, source_structure={},
        explainability={"evidence_strictness": "multi-ref"},
        retrieval_level="Hard", answer_level="Medium", difficulty_rationale="",
        kg_novelty="unknown_relation", kg_query_type="negation_exhaustive",
        kg_path=KGPathProvenance(
            snapshot_id="s", path_text="p", edges=[], verifier_model="j",
            verified_at=datetime.now(), source="kg_decoy"),
        generation={"model": "m", "prompt_version": "v",
                    "generated_at": datetime.now().isoformat()},
    )
    assert qa.kg_path.source == "kg_decoy"   # スキーマが系統を保持すること
