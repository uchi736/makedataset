"""Tests for Stage 1 generation pipeline using a mock LLM (R1 schema)."""

from __future__ import annotations

import json
from pathlib import Path

from rageval.chunker import chunk_directory
from rageval.generate import generate_batch
from rageval.schema import QAItem


MOCK_QA_JSON = {
    "qa_id": "auto_deadbeef",
    "question": "製品Xの動作電圧範囲は何ボルトですか？",
    "answer": "DC 21.6V〜26.4V (24V±10%) です。",
    "rationale": [
        {"doc_id": "sample", "page": None, "text": "製品Xの定格電圧は DC 24V ±10% で、許容範囲は 21.6V〜26.4V とする"}
    ],
    "category": ["Reasoning"],
    "aspect": ["quantitative_calc"],
    "reasoning_complexity": {
        "multi_step": True, "quantitative": True, "negation": False,
        "cause_effect": False, "comparison": False, "temporal": False,
        "output_type": "none",
    },
    "retrieval_difficulty": {
        "multi_doc": False, "multi_chunk": False, "low_locality": False,
        "remote_reference": False, "doc_volume_large": False, "chunk_size_large": False,
        "abstraction_discrepancy": False, "vocabulary_mismatch": False,
    },
    "source_structure": {
        "tables_charts": False, "complex_layout": False, "specific_area_ref": False,
        "logical_nesting": False, "large_enumeration": False, "redundancy": False,
    },
    "explainability": {"evidence_strictness": "hier-ref"},
    "retrieval_level": "Easy",
    "answer_level": "Medium",
    "difficulty_rationale": "必要チャンク=1, 推論ステップ=1",
    "business_scenario": "工程設計レビュー",
}


def _mock_llm(**kwargs):
    # Echo the anchor_chunks_block content into the answer so we can verify the
    # prompt is being built with the multi-chunk template.
    prompt = kwargs.get("prompt", "")
    n_anchors = prompt.count("## チャンク")
    qa = dict(MOCK_QA_JSON)
    qa["answer"] = f"{MOCK_QA_JSON['answer']} (anchors={n_anchors})"
    return json.dumps(qa, ensure_ascii=False)


def test_parse_kg_mix_valid():
    from rageval.generate import parse_kg_mix
    out = parse_kg_mix("multi_hop:unknown_relation=5, traceability:procedural_relation=3")
    assert out == [
        ("multi_hop", "unknown_relation", 5),
        ("traceability", "procedural_relation", 3),
    ]


def test_parse_kg_mix_invalid_query_type():
    import pytest
    from rageval.generate import parse_kg_mix
    with pytest.raises(ValueError, match="unknown kg_query_type"):
        parse_kg_mix("invalid_qt:unknown_relation=5")


def test_parse_kg_mix_invalid_format():
    import pytest
    from rageval.generate import parse_kg_mix
    with pytest.raises(ValueError, match="format"):
        parse_kg_mix("multi_hop unknown_relation 5")


def test_build_kg_cell_queue_with_mix():
    import random
    from rageval.generate import _build_kg_cell_queue
    mix = [("multi_hop", "unknown_relation", 3), ("traceability", "procedural_relation", 2)]
    queue = _build_kg_cell_queue(5, mix, random.Random(0))
    assert len(queue) == 5
    assert queue.count(("multi_hop", "unknown_relation")) == 3
    assert queue.count(("traceability", "procedural_relation")) == 2


def test_build_kg_cell_queue_equal_default():
    import random
    from rageval.generate import _build_kg_cell_queue
    queue = _build_kg_cell_queue(15, None, random.Random(0))
    assert len(queue) == 15
    # 15 cells × 1 each
    from collections import Counter
    counts = Counter(queue)
    assert all(v == 1 for v in counts.values())


def test_detect_fabricated_terms_flags_neologism():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="本規程は振替休日を定める。")]
    fab = detect_fabricated_terms(
        "ディスミッション・エクスクルージョン条件として振替休日を設定する",
        chunks,
    )
    assert "ディスミッション" in " ".join(fab)
    assert "エクスクルージョン" in " ".join(fab)


def test_detect_fabricated_terms_passes_when_in_chunk():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="リストストラップを装着すること。")]
    fab = detect_fabricated_terms(
        "リストストラップの装着順序は?",
        chunks,
    )
    assert fab == []   # word appears in chunk → not fabricated


def test_detect_fabricated_terms_ignores_short_katakana():
    from rageval.generate import detect_fabricated_terms
    from rageval.schema import Chunk
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="ボルトのトルクは…")]
    # 3-char katakana words shouldn't trip the filter (too noisy)
    fab = detect_fabricated_terms("ナット を締める", chunks)
    assert fab == []


# ---------- 多 doc アンカー × 単一 doc rationale 検出 ----------

def _make_minimal_qa(rationale_doc_ids: list[str]) -> "QAItem":
    """rationale.doc_id だけを変えた最小 QAItem (他はテストに無関係)"""
    from datetime import datetime
    from rageval.schema import (
        QAItem, Rationale, ReasoningComplexity, RetrievalDifficulty,
        SourceStructure, Explainability, GenerationInfo,
    )
    return QAItem(
        qa_id="qx", question="Q?", answer="A",
        rationale=[Rationale(doc_id=d, page=None, text="x") for d in rationale_doc_ids],
        category=["Reasoning"], aspect=["multi_hop"],
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy", answer_level="Easy",
        difficulty_rationale="test",
        generation=GenerationInfo(model="m", prompt_version="v", generated_at=datetime.now()),
    )


def test_single_doc_rationale_with_multi_anchor_flags_violation():
    from rageval.generate import detect_single_doc_rationale_with_multi_anchor
    from rageval.schema import Chunk
    # アンカーは 2 doc (docA + docB) だが rationale は docA のみ → 違反
    anchors = [
        Chunk(chunk_id="docA__c0", doc_id="docA", text="t"),
        Chunk(chunk_id="docB__c0", doc_id="docB", text="t"),
    ]
    qa = _make_minimal_qa(rationale_doc_ids=["docA"])
    result = detect_single_doc_rationale_with_multi_anchor(qa, anchors)
    assert result is not None
    anchor_docs, rat_docs = result
    assert anchor_docs == {"docA", "docB"}
    assert rat_docs == {"docA"}


def test_single_doc_rationale_with_multi_anchor_passes_when_rationale_spans():
    from rageval.generate import detect_single_doc_rationale_with_multi_anchor
    from rageval.schema import Chunk
    # アンカー 2 doc、rationale も 2 doc → OK
    anchors = [
        Chunk(chunk_id="docA__c0", doc_id="docA", text="t"),
        Chunk(chunk_id="docB__c0", doc_id="docB", text="t"),
    ]
    qa = _make_minimal_qa(rationale_doc_ids=["docA", "docB"])
    assert detect_single_doc_rationale_with_multi_anchor(qa, anchors) is None


def test_single_doc_rationale_with_multi_anchor_passes_single_doc_anchor():
    from rageval.generate import detect_single_doc_rationale_with_multi_anchor
    from rageval.schema import Chunk
    # アンカーが元から 1 doc なら検査対象外
    anchors = [Chunk(chunk_id="docA__c0", doc_id="docA", text="t")]
    qa = _make_minimal_qa(rationale_doc_ids=["docA"])
    assert detect_single_doc_rationale_with_multi_anchor(qa, anchors) is None


# ---------- 場所当て検出 ----------

def test_detect_place_finding_question_flags_dono_jobun():
    from rageval.generate import detect_place_finding_question
    found = detect_place_finding_question(
        "液化ガス設備で、どの条文の規定に適合していればよいか教えてください。"
    )
    assert "どの条文" in found


def test_detect_place_finding_question_flags_nani_jou():
    from rageval.generate import detect_place_finding_question
    found = detect_place_finding_question("規定は第何条に書かれていますか?")
    assert "第何条" in found


def test_detect_place_finding_question_does_not_flag_content_question():
    """「どの試験」「どの基準」のような中身を問う表現は弾かない。"""
    from rageval.generate import detect_place_finding_question
    found = detect_place_finding_question(
        "管寄せの形状を評価したい。どの試験を実施してどの基準を満たせばよいか教えてください。"
    )
    assert found == []


def test_detect_place_finding_question_does_not_flag_dono_kitei():
    """「どの規定 (内容を問う)」は場所当てではない。"""
    from rageval.generate import detect_place_finding_question
    found = detect_place_finding_question("どの規定に従えばよいですか")
    assert found == []


# ---------- 条文番号列挙の検出 (v1.8) ----------

def test_detect_article_enumeration_flags_3_or_more():
    """質問に第N条が 3 個以上は試験文体の徴候。"""
    from rageval.generate import detect_article_enumeration
    n = detect_article_enumeration(
        "省令の第5条、第10条、第11条、第12条、第15条に関連する様式は何ですか"
    )
    assert n == 5


def test_detect_article_enumeration_allows_single_article():
    """1 個なら実務者が知っている根拠引用の範囲、許容。"""
    from rageval.generate import detect_article_enumeration
    n = detect_article_enumeration("第59条に関連する設計をしている")
    assert n == 1


def test_detect_article_enumeration_counts_distinct_only():
    """同じ「第59条」が複数回出ても 1 と数える。"""
    from rageval.generate import detect_article_enumeration
    n = detect_article_enumeration("第59条の規定について、第59条に紐づく様式は")
    assert n == 1


# ---------- 経歴ピボットの検出 (v1.8) ----------

def test_detect_career_pivot_flags_experience_then_pivot():
    """「Xの経験者がYに従事する場合」を捉える。"""
    from rageval.generate import detect_career_pivot
    found = detect_career_pivot(
        "防毒呼吸具の研究や設計に従事した経験がある者が、液化ガス容器の設計に関わる場合、許容応力は…"
    )
    assert len(found) > 0


def test_detect_career_pivot_flags_keiken_settei():
    from rageval.generate import detect_career_pivot
    found = detect_career_pivot(
        "ボイラーの設計に従事した経験がある技術者が、新しい圧力容器を設計するときに参照すべき規格は"
    )
    assert len(found) > 0


def test_detect_career_pivot_does_not_flag_normal_question():
    """普通の実務質問は弾かない。"""
    from rageval.generate import detect_career_pivot
    found = detect_career_pivot(
        "圧力容器のふた板を設計している。守るべき規格や附属書を教えてほしい"
    )
    assert found == []


# ---------- 既知前提引用の検出 (v1.9) ----------

def test_detect_known_premise_quote_flags_kanren_jiko():
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "圧力容器の設計において、第59条に関連する事項を確認したいと考えています。第59条に関連する様式はどれですか。"
    )
    assert len(found) > 0


def test_detect_known_premise_quote_flags_kanren_youshiki():
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "第14条に関連する様式は何ですか"
    )
    assert len(found) > 0


def test_detect_known_premise_quote_does_not_flag_natural_phrase():
    """「第N条の規定により」「第N条の場合は」など状況記述は弾かない。"""
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "圧力容器のふた板を設計している。第59条の規定により補強が必要な場合の判定基準は"
    )
    assert found == []


# ---------- v1.9 既知前提引用の隠れ変種 ----------

def test_detect_known_premise_quote_flags_kakko_article():
    """質問文の括弧書き条文番号 (第77条) を弾く。"""
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "計測装置に関する規定（第77条）に関連して使用すべき様式はどれですか"
    )
    assert len(found) > 0


def test_detect_known_premise_quote_flags_kitei_ni_jun_jite():
    """「第N条の規定に準じて」を弾く。"""
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "穴の径が規定のdの値の0.5倍以下である場合、第59条第3項の規定に準じて補強を行いたい"
    )
    assert len(found) > 0


def test_detect_known_premise_quote_flags_motozuite():
    """「第N条に基づいて」を弾く。"""
    from rageval.generate import detect_known_premise_quote
    found = detect_known_premise_quote(
        "第6条第5項の規定に基づいて補強計算を行いたい"
    )
    assert len(found) > 0


# ---------- v1.9 断片 rationale 検出 ----------

def _make_test_qa(rationale_list):
    """テスト用 QAItem を最小フィールドで作るヘルパ。"""
    from rageval.schema import (
        QAItem, Rationale, ReasoningComplexity, RetrievalDifficulty,
        SourceStructure, Explainability, GenerationInfo,
    )
    return QAItem(
        qa_id="test", question="?", answer="!",
        rationale=[Rationale(**r) for r in rationale_list],
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Hard", answer_level="Hard", difficulty_rationale="x",
        generation=GenerationInfo(model="t", prompt_version="v", generated_at="2026-06-27T00:00:00"),
    )


def test_detect_fragment_rationale_flags_article_number_only():
    """rationale.text が「第１０７条, 第１項」のような番号断片だけのものを弾く。"""
    from rageval.generate import detect_fragment_rationale
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー等の容器又は管の溶接部の耐圧試験は、第5条の規定を準用する"},
        {"doc_id": "d2", "text": "第１０７条, 第１項"},
    ])
    found = detect_fragment_rationale(qa)
    assert len(found) == 1
    assert found[0][0] == 1


def test_detect_fragment_rationale_passes_real_sentence():
    """まともな本文の rationale は弾かない。"""
    from rageval.generate import detect_fragment_rationale
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー等の容器又は管の溶接部の耐圧試験は、第5条の規定を準用する"},
        {"doc_id": "d2", "text": "技術基準の解釈において、第107条第1項に規定する試験に適合した技能を有する溶接士"},
    ])
    found = detect_fragment_rationale(qa)
    assert found == []


def test_detect_fragment_rationale_flags_short_text():
    """極端に短い rationale も実体不足とみなす。"""
    from rageval.generate import detect_fragment_rationale
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー等の容器又は管の溶接部の耐圧試験は、第5条の規定を準用する"},
        {"doc_id": "d2", "text": "（ロ）第１０７条"},
    ])
    found = detect_fragment_rationale(qa)
    assert len(found) == 1


# ---------- v2.0 短すぎる rationale の機械検出 ----------

def test_detect_short_rationale_flags_under_30_chars():
    from rageval.generate import detect_short_rationale
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー等の容器又は管の溶接部の耐圧試験は、第5条の規定を準用する"},
        {"doc_id": "d2", "text": "第１０７条第１項"},  # 8 文字
    ])
    bad = detect_short_rationale(qa, min_chars=30)
    assert len(bad) == 1
    assert bad[0][0] == 1


def test_detect_short_rationale_passes_long_text():
    from rageval.generate import detect_short_rationale
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー等の容器又は管の溶接部の耐圧試験は、第5条の規定を準用する"},
        {"doc_id": "d2", "text": "技術基準解釈第107条第1項に規定する試験を実施し、当該試験に合格すること"},
    ])
    assert detect_short_rationale(qa, min_chars=30) == []


# ---------- v2.0 answer 固有値の逐語裏取り ----------

def test_detect_ungrounded_answer_tokens_flags_missing_form_number():
    """answer に「様式第6号」と書いてあるのに rationale にその文字列が無いケース。"""
    from rageval.generate import detect_ungrounded_answer_tokens
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー附属設備の検査結果は様式に記録する"},
        {"doc_id": "d2", "text": "様式第7号（第60条関係）"},  # 第6号 ではなく第7号
    ])
    qa.answer = "様式第6号を使用します"
    missing = detect_ungrounded_answer_tokens(qa)
    assert "様式第6号" in missing


def test_detect_ungrounded_answer_tokens_passes_when_grounded():
    """answer の固有値が rationale に逐語存在する場合は OK。"""
    from rageval.generate import detect_ungrounded_answer_tokens
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "ボイラー附属設備の検査結果は様式第6号に記録する。第60条関係。"},
        {"doc_id": "d2", "text": "様式第6号（第60条関係）使用方法..."},
    ])
    qa.answer = "様式第6号を使用します"
    assert detect_ungrounded_answer_tokens(qa) == []


def test_detect_ungrounded_answer_tokens_handles_fullwidth_digits():
    """全角数字でも半角に正規化して照合する。"""
    from rageval.generate import detect_ungrounded_answer_tokens
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "第１０７条第１項に規定する試験"},  # 全角
        {"doc_id": "d2", "text": "技術基準解釈において定める"},
    ])
    qa.answer = "第107条第1項の試験を実施"  # 半角
    assert detect_ungrounded_answer_tokens(qa) == []


def test_detect_ungrounded_answer_tokens_no_tokens_in_answer():
    """answer に固有値が無いケースは pass。"""
    from rageval.generate import detect_ungrounded_answer_tokens
    qa = _make_test_qa([
        {"doc_id": "d1", "text": "適切な手順を守ってください"},
        {"doc_id": "d2", "text": "適切な記録を残してください"},
    ])
    qa.answer = "適切な手順を守ってください"
    assert detect_ungrounded_answer_tokens(qa) == []


def test_render_prompt_injects_aspect_examples():
    """Aspect の ✓良い例 / ✗悪い例 がプロンプト本文に埋め込まれることを確認。"""
    from rageval.generate import _render_generate_prompt
    from rageval.aspects import ASPECT_BAD_PATTERNS, ASPECT_GOOD_EXAMPLES
    from rageval.prompts import load_prompt
    from rageval.schema import Chunk

    _, body = load_prompt("prompts/generate.md")
    anchors = [Chunk(chunk_id="c0", doc_id="d", text="本規程は所定労働時間を定める。")]
    spec = {
        "aspect": "complex_layout",
        "category": "Figure",
        "retrieval_level": "Easy",
        "answer_level": "Medium",
    }
    rendered = _render_generate_prompt(body, anchors, spec, [])

    # 良い例(少なくとも1つ)が含まれる
    for good in ASPECT_GOOD_EXAMPLES["complex_layout"]:
        assert good in rendered
    # 悪い例の代表(月数を数える型)が含まれる
    bad = ASPECT_BAD_PATTERNS["complex_layout"]
    assert any(b in rendered for b in bad)
    # 共通禁止ルールも残っている
    assert "列挙を数えるだけ" in rendered


def test_generate_batch_with_mock_llm(tmp_path: Path, monkeypatch):
    # Stub embeddings so we don't try to hit a real endpoint.
    monkeypatch.setattr("rageval.generate.compute_embeddings", lambda chunks: None)
    # Pin to a single-chunk aspect: multi-chunk aspects would skip on a 1-chunk
    # corpus (sampler returns None → no QA written), making len(lines)==3 flaky.
    monkeypatch.setattr(
        "rageval.generate._build_spec",
        lambda rng: {"aspect": "quantitative_calc", "category": "Reasoning"},
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text(
        "製品X仕様書\n定格電圧: DC 24V ±10%\n定格電流: 2A\n",
        encoding="utf-8",
    )
    chunks_dir = tmp_path / "chunks"
    chunk_directory(docs_dir, chunks_dir)

    out_dir = tmp_path / "raw"
    seeds = tmp_path / "seeds.json"
    seeds.write_text("[]", encoding="utf-8")

    out_path = generate_batch(
        chunks_dir=chunks_dir, out_dir=out_dir, n=3, model="mock",
        seeds_path=seeds, llm=_mock_llm,
    )

    assert out_path.exists()
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    for line in lines:
        qa = QAItem.model_validate_json(line)
        assert qa.generation.model == "mock"
        assert qa.review_status == "pending"
        assert qa.aspect == ["quantitative_calc"]
        assert qa.business_scenario
        # Confirm prompt was assembled with anchor_chunks_block
        assert "anchors=" in qa.answer


# ---------------- v2.1: 接続根拠ブロック ----------------

def _mk_chunk_v21(doc, pos, text, refs=None, sect=None):
    from rageval.schema import Chunk
    return Chunk(chunk_id=f"{doc}__c{pos:04d}", doc_id=doc, page=None, text=text,
                 position=pos, section_path=sect or [], references=refs or [])


def test_describe_join_ledger_and_standard_and_alias():
    from rageval.sampling import CrossDocPolicy
    a = _mk_chunk_v21("law_a", 0, "JIS B 1083 に従う", refs=["JIS B 1083"])
    b = _mk_chunk_v21("law_b", 0, "JIS B 1083 第3表", refs=["JIS B 1083"])
    assert "規格" in CrossDocPolicy().describe_join(a, b)
    c = _mk_chunk_v21("law_c", 0, "ボイラー則第百一条に掲げる者")
    d = _mk_chunk_v21("boiler", 0, "第百一条 一級ボイラー技士…")
    pol = CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]})
    assert "名指し" in pol.describe_join(c, d)
    pol2 = CrossDocPolicy(allowed_chunk_pairs={frozenset((a.chunk_id, d.chunk_id))})
    assert "台帳" in pol2.describe_join(a, d)
    assert CrossDocPolicy().describe_join(c, d) is None  # 根拠なしは None


def test_connection_block_rendered(monkeypatch):
    import rageval.sampling as sampling
    from rageval.generate import _format_connection_block
    monkeypatch.setattr(sampling, "CROSS_DOC_POLICY",
                        sampling.CrossDocPolicy(doc_aliases={"boiler": ["ボイラー則"]}))
    a = _mk_chunk_v21("law_c", 0, "ボイラー則第百一条に掲げる者", refs=["第101条"])
    b = _mk_chunk_v21("boiler", 0, "第百一条 一級ボイラー技士…")
    block = _format_connection_block([a, b])
    assert "接続根拠" in block and "名指し" in block and "connection" in block
    # 同一文書の遠隔参照
    c1 = _mk_chunk_v21("d", 0, "第100条の規定を準用する", refs=["第100条"])
    c2 = _mk_chunk_v21("d", 20, "第100条 省令第63条に…")
    block2 = _format_connection_block([c1, c2])
    assert "同一文書内の遠隔参照" in block2 and "第100条" in block2
    assert _format_connection_block([c1]) == ""
