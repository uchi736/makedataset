"""Tests for vector RAG ground-truth verification (rag-verify)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from rageval.llm import LLMError
from rageval.rag_verify import (
    _format_chunks_block,
    _is_unable_answer,
    _normalize_question_vec,
    _retrieval_hit,
    _retrieval_hits,
    _retrieve_topk,
    rag_verify_batch,
    rag_verify_qa,
)
from rageval.schema import (
    Chunk,
    Explainability,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)


# ---------- fixtures ----------

def _make_qa(qa_id: str = "qa1", question: str = "Q?", answer: str = "A",
             rationale_doc: str = "docA") -> QAItem:
    return QAItem(
        qa_id=qa_id,
        question=question,
        answer=answer,
        rationale=[Rationale(doc_id=rationale_doc, page=None, text="dummy")],
        category=["Reasoning"],
        aspect=["quantitative_calc"],
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="必要チャンク=1",
        generation=GenerationInfo(model="mock", prompt_version="v0", generated_at=datetime.now()),
    )


def _make_chunks() -> tuple[list[Chunk], np.ndarray]:
    """3チャンク、それぞれを別方向の単位ベクトル(L2正規化済)に。"""
    chunks = [
        Chunk(chunk_id="docA__c0", doc_id="docA", text="rationale here"),
        Chunk(chunk_id="docB__c0", doc_id="docB", text="something else"),
        Chunk(chunk_id="docC__c0", doc_id="docC", text="third one"),
    ]
    emb = np.array([
        [1.0, 0.0, 0.0],   # docA に近いベクトル
        [0.0, 1.0, 0.0],   # docB
        [0.0, 0.0, 1.0],   # docC
    ], dtype=float)
    return chunks, emb


# ---------- helper-unit tests ----------

def test_retrieval_hit_doc_id_match():
    qa = _make_qa(rationale_doc="docA")
    chunks, _ = _make_chunks()
    assert _retrieval_hit(qa, chunks[:1]) is True
    assert _retrieval_hit(qa, chunks[1:]) is False


def test_retrieval_hit_chunk_id_match():
    """rationale.doc_id にチャンクID('docA__c0') が入ってきても拾える。"""
    qa = _make_qa(rationale_doc="docA__c0")
    chunks, _ = _make_chunks()
    assert _retrieval_hit(qa, [chunks[0]]) is True
    assert _retrieval_hit(qa, [chunks[1]]) is False


def _make_qa_with_rationale_text(rationale_doc: str, rationale_text: str) -> QAItem:
    """根拠本文の逐語一致を確かめたいときに使う雛形。"""
    return QAItem(
        qa_id="qa_chunk",
        question="Q?",
        answer="A",
        rationale=[Rationale(doc_id=rationale_doc, page=None, text=rationale_text)],
        category=["Reasoning"],
        aspect=["quantitative_calc"],
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="必要チャンク=1",
        generation=GenerationInfo(model="mock", prompt_version="v0", generated_at=datetime.now()),
    )


def test_retrieval_hits_doc_only_when_wrong_chunk_of_same_doc():
    """同一文書の別チャンクが上位に来たケース。文書一致は真、チャンク一致は偽。

    技報1本が50を超えるチャンクに割れるとき、根拠とは無関係な章のチャンクが
    拾われるだけで文書一致は真になってしまう。本命指標である「チャンク一致」が
    偽になることを担保する。
    """
    qa = _make_qa_with_rationale_text(
        rationale_doc="docA",
        rationale_text="定格電圧はDC24Vである。",
    )
    # 同じ docA だが別チャンク (根拠本文を含まない)
    other_chunk_same_doc = Chunk(
        chunk_id="docA__c0099",
        doc_id="docA",
        text="筐体の塗装色はマンセル N7 とする。",
    )
    hit_doc, hit_chunk = _retrieval_hits(qa, [other_chunk_same_doc])
    assert hit_doc is True
    assert hit_chunk is False


def test_retrieval_hits_chunk_match_on_verbatim_text():
    """根拠本文を含むチャンクが上位に居れば文書一致もチャンク一致も真。"""
    qa = _make_qa_with_rationale_text(
        rationale_doc="docA",
        rationale_text="定格電圧はDC24V",
    )
    right_chunk = Chunk(
        chunk_id="docA__c0023",
        doc_id="docA",
        text="本機の定格電圧はDC24Vであり、許容変動は±10%。",
    )
    hit_doc, hit_chunk = _retrieval_hits(qa, [right_chunk])
    assert hit_doc is True
    assert hit_chunk is True


def test_retrieval_hits_chunk_match_tolerates_whitespace_reflow():
    """空白の入り方が違っても根拠本文を含むと見なせること。

    LLM が rationale を整形して空白を挟むことがあるので、
    filter._normalize_for_match と同じ規則で吸収する。
    """
    qa = _make_qa_with_rationale_text(
        rationale_doc="docA",
        rationale_text="定格電圧 は DC 24 V",
    )
    right_chunk = Chunk(
        chunk_id="docA__c0023",
        doc_id="docA",
        text="定格電圧はDC24Vである。",
    )
    _, hit_chunk = _retrieval_hits(qa, [right_chunk])
    assert hit_chunk is True


def test_retrieval_hits_empty_rationale_returns_both_false():
    qa = _make_qa_with_rationale_text(rationale_doc="docA", rationale_text="x")
    qa.rationale = []
    chunks, _ = _make_chunks()
    hit_doc, hit_chunk = _retrieval_hits(qa, chunks)
    assert hit_doc is False
    assert hit_chunk is False


def test_retrieve_topk_picks_nearest():
    chunks, emb = _make_chunks()
    def fake_embed(texts):
        return [[1.0, 0.0, 0.0]]   # docA 方向と完全一致
    top = _retrieve_topk("Q?", chunks, emb, top_k=2, embed=fake_embed)
    assert [c.doc_id for c in top] == ["docA", "docB"]


def test_retrieve_topk_handles_empty_chunks():
    arr = np.zeros((0, 3), dtype=float)
    assert _retrieve_topk("Q?", [], arr, top_k=3, embed=lambda t: [[1, 0, 0]]) == []


def test_normalize_question_vec_raises_on_zero_norm():
    """全成分0の埋め込みは無言の誤りに倒れるので例外で止める。"""
    with pytest.raises(LLMError):
        _normalize_question_vec([0.0, 0.0, 0.0])


def test_retrieve_topk_zero_norm_propagates():
    """埋め込み API が全0ベクトルを返したら例外が伝播し、先頭チャンクを返さない。"""
    chunks, emb = _make_chunks()
    with pytest.raises(LLMError):
        _retrieve_topk(
            "Q?", chunks, emb, top_k=2, embed=lambda texts: [[0.0, 0.0, 0.0]]
        )


def test_rag_verify_qa_zero_norm_yields_no_match():
    """全0埋め込み → retrieved=[] → answer_match='no_match', retrieval_hit=False。"""
    qa = _make_qa(question="", answer="期待値", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_match,
        embed=lambda texts: [[0.0, 0.0, 0.0]],
    )
    assert result.answer_match == "no_match"
    assert result.retrieval_hit is False
    assert result.retrieved_chunk_ids == []
    assert result.rag_answer == ""


def test_format_chunks_block_includes_meta():
    chunks = [Chunk(chunk_id="d__c0", doc_id="d", page=3, text="body text",
                    section_path=["第1章", "1.2 概要"])]
    block = _format_chunks_block(chunks)
    assert "d__c0" in block
    assert "p.3" in block
    assert "第1章 > 1.2 概要" in block
    assert "body text" in block


# ---------- end-to-end with mock LLM + mock embedder ----------

def _mock_llm_match(*, prompt: str, model: str, **kwargs):
    if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
        return json.dumps({"answer": "DC 24V"}, ensure_ascii=False)
    # judge
    return json.dumps({"match": "match", "reason": "意味一致"}, ensure_ascii=False)


def _mock_llm_no_match(*, prompt: str, model: str, **kwargs):
    if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
        return json.dumps({"answer": "回答不能"}, ensure_ascii=False)
    return json.dumps({"match": "no_match", "reason": "ng"}, ensure_ascii=False)


def _fake_embed_question(texts):
    # 質問は docA 方向に倒す
    return [[1.0, 0.0, 0.0]]


def test_rag_verify_qa_match_path():
    qa = _make_qa(question="定格電圧は?", answer="DC 24V", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_match, embed=_fake_embed_question,
    )
    assert result.answer_match == "match"
    assert result.retrieval_hit is True
    # 文書一致は真。チャンク本文一致は根拠本文('dummy') が
    # チャンク本文('rationale here') に逐語で含まれないので偽。
    assert result.retrieval_hit_doc is True
    assert result.retrieval_hit_chunk is False
    assert result.rag_answer == "DC 24V"
    assert result.top_k == 2
    assert "docA__c0" in result.retrieved_chunk_ids


def test_rag_verify_qa_no_match_when_rag_says_unable():
    qa = _make_qa(question="X?", answer="期待値", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_no_match, embed=_fake_embed_question,
    )
    assert result.answer_match == "no_match"
    assert result.rag_answer == "回答不能"
    # judge は走らないが retrieval は走るので retrieval_hit は計算される
    assert result.retrieval_hit is True


@pytest.mark.parametrize(
    "raw",
    [
        "回答不能",
        "回答不能。",
        " 回答不能 ",
        "不明",
        "不明。",
        "わからない",
        "分からない。",
        "判断できない",
        "判断できません。",
        "該当なし",
        "回答できません。",
        "記載なし",
        "情報なし",
        "記述なし",
        "文書からは特定できません。",
        "文書からは特定できない",
        "「回答不能」",
        "  ",
        "",
    ],
)
def test_is_unable_answer_catches_known_phrasings(raw):
    """指示追従の揺れ ('不明' '該当なし' 等) を「答えなし」に倒せること。"""
    assert _is_unable_answer(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "DC 24V",
        "不明な点は監督者に確認すること",  # 「不明」を含むが実体のある回答
        "判断できない場合は再測定する",   # 「判断できない」を含むが手順を述べている
        "該当なしと判定する基準は次のとおり",
    ],
)
def test_is_unable_answer_keeps_substantive_answers(raw):
    """部分文字列だけで弾かないこと (実体ある回答が judge に流れることを保証)。"""
    assert _is_unable_answer(raw) is False


def test_rag_verify_qa_unable_synonym_skips_judge_call():
    """Gemma が『不明』と返した場合も judge を呼ばずに no_match に倒すこと。"""
    qa = _make_qa(question="X?", answer="期待値", rationale_doc="docA")
    chunks, emb = _make_chunks()

    judge_calls: list[str] = []

    def _mock_llm_unable_synonym(*, prompt: str, model: str, **kwargs):
        if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
            # 『回答不能』の指示を守らず別表現で返してくるケース
            return json.dumps({"answer": "不明"}, ensure_ascii=False)
        judge_calls.append(prompt)
        # judge が呼ばれてしまったら誤って match を返す不愉快な世界線を再現
        return json.dumps({"match": "match", "reason": "意味一致"}, ensure_ascii=False)

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_unable_synonym, embed=_fake_embed_question,
    )
    assert result.answer_match == "no_match"
    assert result.rag_answer == "不明"
    assert judge_calls == [], "『不明』は『回答不能』と同様に judge を呼ばずに弾くべき"


def test_rag_verify_qa_unexpected_verdict_warns_and_records(capsys):
    """judge が想定外の verdict を返したら警告を出し、judge_raw に生値を残す。

    黙って no_match に倒すと、--require-rag-fail のような後段ふるい分けが
    「judge が崩れた QA」を正規の no_match と同列に通してしまう。
    """
    qa = _make_qa(question="Q?", answer="DC 24V", rationale_doc="docA")
    chunks, emb = _make_chunks()

    def _mock_llm_unexpected_verdict(*, prompt: str, model: str, **kwargs):
        if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
            return json.dumps({"answer": "DC 24V"}, ensure_ascii=False)
        # judge が想定外の語を返す
        return json.dumps({"match": "matched", "reason": "ng"}, ensure_ascii=False)

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_unexpected_verdict, embed=_fake_embed_question,
    )
    # 想定外なので no_match に倒れるが、judge_raw に生値を残して事後検証可能にする
    assert result.answer_match == "no_match"
    assert result.judge_raw == "matched"
    # 標準出力に警告が出ていること
    captured = capsys.readouterr()
    assert "unexpected judge verdict" in captured.out
    assert "'matched'" in captured.out


def test_rag_verify_qa_match_path_records_judge_raw():
    """正常に match が返ったときも judge_raw に小文字化した生値を残す。"""
    qa = _make_qa(question="定格電圧は?", answer="DC 24V", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_match, embed=_fake_embed_question,
    )
    assert result.answer_match == "match"
    assert result.judge_raw == "match"


def test_rag_verify_qa_no_judge_call_leaves_judge_raw_none():
    """rag_answer が空 / 回答不能のときは judge を呼ばず、judge_raw は None のまま。

    後段で「judge が判定して no_match」と「そもそも judge を呼ばずに no_match」を
    区別するための入口。
    """
    qa = _make_qa(question="X?", answer="期待値", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_no_match, embed=_fake_embed_question,
    )
    assert result.answer_match == "no_match"
    assert result.judge_raw is None


def test_rag_verify_qa_forces_json_on_both_calls():
    """RAG 回答も judge も force_json=True で呼ぶこと。

    Gemma 系 vLLM はフェンス外文字が混じる事故が出るので、
    response_format={'type':'json_object'} を必ず効かせる。
    """
    qa = _make_qa(question="定格電圧は?", answer="DC 24V", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    captured: list[dict] = []

    def _capturing_llm(*, prompt: str, model: str, **kwargs):
        captured.append({"model": model, **kwargs})
        if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
            return json.dumps({"answer": "DC 24V"}, ensure_ascii=False)
        return json.dumps({"match": "match", "reason": "ok"}, ensure_ascii=False)

    rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_capturing_llm, embed=_fake_embed_question,
    )

    assert len(captured) == 2, "RAG 回答と judge の2回呼ばれるはず"
    assert all(call.get("force_json") is True for call in captured), (
        f"force_json=True が両方に渡っていない: {captured}"
    )


def test_rag_verify_batch_writes_field(tmp_path: Path):
    # 1つのチャンクファイルを書き出し
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    c0 = Chunk(chunk_id="docA__c0", doc_id="docA", text="rationale here")
    c1 = Chunk(chunk_id="docB__c0", doc_id="docB", text="another")
    (chunks_dir / "docA.jsonl").write_text(
        c0.model_dump_json() + "\n", encoding="utf-8"
    )
    (chunks_dir / "docB.jsonl").write_text(
        c1.model_dump_json() + "\n", encoding="utf-8"
    )

    # QA を1件
    in_path = tmp_path / "batch.jsonl"
    qa = _make_qa(question="Q?", answer="A", rationale_doc="docA")
    in_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    out_path = tmp_path / "out.jsonl"
    # 埋め込みは事前計算した numpy 配列を inject (実 endpoint を叩かない)
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    rag_verify_batch(
        in_path=in_path, chunks_dir=chunks_dir, top_k=1,
        rag_model="mock-rag", judge_model="mock-judge",
        llm=_mock_llm_match, embed=lambda t: [[1.0, 0.0]],
        chunk_embeddings=emb,
        out_path=out_path,
    )

    out_qa = QAItem.model_validate_json(out_path.read_text(encoding="utf-8").strip())
    assert out_qa.rag_verification is not None
    assert out_qa.rag_verification.answer_match == "match"
    assert out_qa.rag_verification.retrieval_hit is True
    assert out_qa.rag_verification.rag_model == "mock-rag"


def test_rag_verify_batch_incremental_flush_preserves_partial_work(tmp_path: Path):
    """途中で例外が出ても、それまでに処理した QA は out_path に書かれている。

    long-running バッチが Ctrl-C / Azure 全断 / OOM 等で死んでも既処理分が失われない
    ことを保証する。1問ずつ append する設計の回帰テスト。
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    c0 = Chunk(chunk_id="docA__c0", doc_id="docA", text="rationale here")
    (chunks_dir / "docA.jsonl").write_text(c0.model_dump_json() + "\n", encoding="utf-8")

    # 2問: 2問目で必ず raise する mock を作って KeyboardInterrupt を投げる
    in_path = tmp_path / "batch.jsonl"
    qa1 = _make_qa(qa_id="qa1", question="Q1", answer="A1", rationale_doc="docA")
    qa2 = _make_qa(qa_id="qa2", question="Q2", answer="A2", rationale_doc="docA")
    in_path.write_text(
        qa1.model_dump_json() + "\n" + qa2.model_dump_json() + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "out.jsonl"
    emb = np.array([[1.0, 0.0]], dtype=float)

    call_count = {"n": 0}
    def crashing_llm(*, prompt, **kw):
        call_count["n"] += 1
        # qa1: RAG回答 + judge で2回呼ばれる → call 1,2 は正常応答
        # qa2: RAG回答で例外 → call 3 で KeyboardInterrupt
        if call_count["n"] >= 3:
            raise KeyboardInterrupt("simulated Ctrl-C during qa2")
        return _mock_llm_match(prompt=prompt, **kw)

    import pytest as _pt
    with _pt.raises(KeyboardInterrupt):
        rag_verify_batch(
            in_path=in_path, chunks_dir=chunks_dir, top_k=1,
            rag_model="mock-rag", judge_model="mock-judge",
            llm=crashing_llm, embed=lambda t: [[1.0, 0.0]],
            chunk_embeddings=emb,
            out_path=out_path,
        )

    # qa1 だけ書かれている (qa2 は死んだ時点で未追記)
    written = [
        QAItem.model_validate_json(l)
        for l in out_path.read_text(encoding="utf-8").splitlines() if l
    ]
    assert len(written) == 1
    assert written[0].qa_id == "qa1"
    assert written[0].rag_verification is not None


def test_rag_verify_batch_rejects_row_count_mismatch(tmp_path: Path):
    """注入された埋め込みの行数がチャンク数と合わないときは即座に止める。

    チャンクを追加してから埋め込みだけ古いキャッシュを使い回すと、
    要素対応がずれたまま hit が常に偽になる無言の不具合になる。
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    # チャンクは2件
    c0 = Chunk(chunk_id="docA__c0", doc_id="docA", text="one")
    c1 = Chunk(chunk_id="docB__c0", doc_id="docB", text="two")
    (chunks_dir / "docA.jsonl").write_text(c0.model_dump_json() + "\n", encoding="utf-8")
    (chunks_dir / "docB.jsonl").write_text(c1.model_dump_json() + "\n", encoding="utf-8")

    in_path = tmp_path / "batch.jsonl"
    qa = _make_qa(question="Q?", answer="A", rationale_doc="docA")
    in_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    # 埋め込みは古い1件分しかない (キャッシュ流用を模擬)
    stale_emb = np.array([[1.0, 0.0]], dtype=float)

    with pytest.raises(RuntimeError, match="chunk_embeddings rows=1"):
        rag_verify_batch(
            in_path=in_path, chunks_dir=chunks_dir, top_k=1,
            rag_model="mock-rag", judge_model="mock-judge",
            llm=_mock_llm_match, embed=lambda t: [[1.0, 0.0]],
            chunk_embeddings=stale_emb,
            out_path=tmp_path / "out.jsonl",
        )


# ---------- 監査で指摘された未網羅のはじっこ条件 ----------

def test_retrieve_topk_dim_mismatch_raises():
    """質問埋め込みの次元とチャンク埋め込みの列数が食い違うと LLMError で止める。

    実運用では埋め込みモデルを途中で差し替えたのに古い行列を使い回す事故が起きる。
    黙って無意味な内積を取らせず、必ず例外で気づかせる。
    """
    chunks, emb = _make_chunks()  # emb は (3, 3)
    with pytest.raises(LLMError):
        _retrieve_topk(
            "Q?", chunks, emb, top_k=2,
            embed=lambda texts: [[1.0, 0.0, 0.0, 0.0]],  # 4次元を返す壊れた埋め込み
        )


def test_retrieve_topk_zero_k_returns_empty():
    """top_k=0 のときは検索結果なしで空のリストを返す。"""
    chunks, emb = _make_chunks()
    assert _retrieve_topk(
        "Q?", chunks, emb, top_k=0, embed=lambda texts: [[1.0, 0.0, 0.0]]
    ) == []


def _mock_llm_always_raises(*, prompt: str, model: str, **kwargs):
    """RAG 回答も judge も例外で落とすモック。"""
    raise RuntimeError("simulated LLM outage")


def test_rag_verify_qa_handles_llm_exception():
    """LLM が常に例外を投げても、rag_answer='' のまま no_match に倒れる。

    検索だけは成功しているので retrieved_chunk_ids は埋まり、
    retrieval_hit の真偽だけは判定できる。
    """
    qa = _make_qa(question="定格電圧は?", answer="DC 24V", rationale_doc="docA")
    chunks, emb = _make_chunks()

    from rageval.prompts import load_prompt
    from rageval.rag_verify import _split_sections
    _, body = load_prompt("prompts/rag_verify.md")
    sections = _split_sections(body)

    result = rag_verify_qa(
        qa, chunks=chunks, chunk_embeddings=emb, top_k=2,
        rag_model="mock-rag", judge_model="mock-judge",
        rag_section=sections["RAG_ANSWER"], judge_section=sections["JUDGE_MATCH"],
        llm=_mock_llm_always_raises, embed=_fake_embed_question,
    )
    assert result.rag_answer == ""
    assert result.answer_match == "no_match"
    assert result.retrieval_hit is True
    assert "docA__c0" in result.retrieved_chunk_ids


def test_rag_verify_batch_empty_chunks_dir_raises(tmp_path: Path):
    """チャンクが1件もない置き場を渡したら RuntimeError で止める。

    空のまま判定に進むと「検索結果は常に空 → no_match」が量産されて、
    黙ってデータセットが壊れるので、入口で必ず気づかせる。
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()  # 中身は空

    in_path = tmp_path / "batch.jsonl"
    qa = _make_qa(question="Q?", answer="A", rationale_doc="docA")
    in_path.write_text(qa.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        rag_verify_batch(
            in_path=in_path, chunks_dir=chunks_dir, top_k=1,
            rag_model="mock-rag", judge_model="mock-judge",
            llm=_mock_llm_match, embed=_fake_embed_question,
        )


def _mock_llm_by_question(*, prompt: str, model: str, **kwargs):
    """質問文を見て QA ごとに match/partial/no_match を打ち分けるモック。"""
    if "RAG_ANSWER" in prompt or "[検索結果]" in prompt:
        # qa1→DC 24V, qa2→AC 100V, qa3→回答不能 で no_match に倒す
        if "qa3-question" in prompt:
            return json.dumps({"answer": "回答不能"}, ensure_ascii=False)
        if "qa2-question" in prompt:
            return json.dumps({"answer": "AC 100V"}, ensure_ascii=False)
        return json.dumps({"answer": "DC 24V"}, ensure_ascii=False)
    # judge: candidate の中身で打ち分け
    if "DC 24V" in prompt:
        return json.dumps({"match": "match", "reason": "一致"}, ensure_ascii=False)
    if "AC 100V" in prompt:
        return json.dumps({"match": "partial", "reason": "部分一致"}, ensure_ascii=False)
    return json.dumps({"match": "no_match", "reason": "不一致"}, ensure_ascii=False)


def test_rag_verify_batch_counts_match_partial_no_match(tmp_path: Path, capsys):
    """複数 QA を流したとき、match/partial/no_match の数え上げが正しい。

    最終行 '[rag-verify] done: match=1 partial=1 no_match=1 ...' を検算する。
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    c0 = Chunk(chunk_id="docA__c0", doc_id="docA", text="rationale here")
    (chunks_dir / "docA.jsonl").write_text(
        c0.model_dump_json() + "\n", encoding="utf-8"
    )

    in_path = tmp_path / "batch.jsonl"
    qa1 = _make_qa(qa_id="qa1", question="qa1-question", answer="DC 24V", rationale_doc="docA")
    qa2 = _make_qa(qa_id="qa2", question="qa2-question", answer="AC 100V", rationale_doc="docA")
    qa3 = _make_qa(qa_id="qa3", question="qa3-question", answer="期待値", rationale_doc="docA")
    in_path.write_text(
        "\n".join(q.model_dump_json() for q in (qa1, qa2, qa3)) + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "out.jsonl"
    emb = np.array([[1.0, 0.0]], dtype=float)

    rag_verify_batch(
        in_path=in_path, chunks_dir=chunks_dir, top_k=1,
        rag_model="mock-rag", judge_model="mock-judge",
        llm=_mock_llm_by_question, embed=lambda t: [[1.0, 0.0]],
        chunk_embeddings=emb,
        out_path=out_path,
    )

    captured = capsys.readouterr().out
    assert "match=1 partial=1 no_match=1" in captured

    # 結果ファイルの中身も確認 (順序は入力どおり)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    results = [QAItem.model_validate_json(ln) for ln in lines]
    verdicts = [r.rag_verification.answer_match for r in results]
    assert verdicts == ["match", "partial", "no_match"]
