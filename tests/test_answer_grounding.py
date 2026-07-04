"""Unit tests for answer_grounding module (v2.0 leave-one-out)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rageval.answer_grounding import (
    grounding_batch,
    grounding_for_qa,
    is_all_docs_required,
    judge_one_doc,
)
from rageval.schema import AnswerGrounding


def _make_qa(rationale_list, question="?", answer="!"):
    from rageval.schema import (
        QAItem, Rationale, ReasoningComplexity, RetrievalDifficulty,
        SourceStructure, Explainability, GenerationInfo,
    )
    return QAItem(
        qa_id="test", question=question, answer=answer,
        rationale=[Rationale(**r) for r in rationale_list],
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Hard", answer_level="Hard", difficulty_rationale="x",
        generation=GenerationInfo(model="t", prompt_version="v", generated_at="2026-06-27T00:00:00"),
    )


# Mock judge LLM that returns different verdicts based on which doc is hidden
class _MockJudge:
    def __init__(self, verdicts: dict[str, bool]):
        """verdicts: doc_id (= hidden) -> sufficient (True = remaining alone is enough)."""
        self.verdicts = verdicts
        self.calls: list[dict] = []

    def __call__(self, *, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        # Pick hidden doc_id from prompt
        for doc_id, suff in self.verdicts.items():
            if doc_id in prompt:
                return json.dumps({"sufficient": suff, "reason": "test"})
        return json.dumps({"sufficient": False, "reason": "default"})


def test_judge_one_doc_calls_llm_with_correct_substitutions():
    qa = _make_qa([
        {"doc_id": "d1", "text": "本文 A"},
        {"doc_id": "d2", "text": "本文 B"},
    ], question="Q?", answer="A.")
    judge = _MockJudge({"d1": True})
    sufficient, raw = judge_one_doc(
        qa, "d1",
        judge_section="Q={question} A={answer} hid={hidden_doc_id} hid_text={hidden_text} rem={remaining_text}",
        judge_model="m",
        llm=judge,
    )
    assert sufficient is True
    assert len(judge.calls) == 1
    p = judge.calls[0]["prompt"]
    assert "Q=Q?" in p
    assert "A=A." in p
    assert "hid=d1" in p
    assert "本文 A" in p  # hidden
    assert "本文 B" in p  # remaining


def test_judge_one_doc_no_remaining_returns_false():
    """rationale が 1 doc しかない場合、それを伏せたら何も残らないので false。"""
    qa = _make_qa([{"doc_id": "only", "text": "本文"}])
    judge = _MockJudge({"only": True})
    sufficient, _ = judge_one_doc(
        qa, "only",
        judge_section="dummy",
        judge_model="m",
        llm=judge,
    )
    assert sufficient is False
    assert len(judge.calls) == 0  # 呼ばずに早期 return


def test_grounding_for_qa_returns_per_doc_dict():
    qa = _make_qa([
        {"doc_id": "d1", "text": "A"},
        {"doc_id": "d2", "text": "B"},
        {"doc_id": "d1", "text": "A2"},  # 同じ doc_id 2 件
    ])
    judge = _MockJudge({"d1": True, "d2": False})
    ag = grounding_for_qa(qa, judge_section="hid={hidden_doc_id}", judge_model="m", llm=judge)
    assert ag.per_doc_sufficient == {"d1": True, "d2": False}
    assert len(judge.calls) == 2  # unique doc_id 数だけ呼ばれる


def test_is_all_docs_required_only_when_no_doc_is_sufficient():
    assert is_all_docs_required(AnswerGrounding(
        per_doc_sufficient={"d1": False, "d2": False},
        judge_model="m",
        verified_at=datetime.now(),
    )) is True
    assert is_all_docs_required(AnswerGrounding(
        per_doc_sufficient={"d1": True, "d2": False},  # d1 を伏せても OK = d1 不要
        judge_model="m",
        verified_at=datetime.now(),
    )) is False


# ---------- チャンク単位 leave-one-out (2026-07-01 本丸のチャンク単位化) ----------

def test_grounding_uses_chunk_id_as_unit_when_present():
    """同一 doc_id でも chunk_id が異なれば 2 単位として leave-one-out する。"""
    from rageval.schema import Rationale
    qa = _make_qa([
        {"doc_id": "モデル就業規則", "chunk_id": "モデル就業規則__c0037", "text": "本文 A"},
        {"doc_id": "モデル就業規則", "chunk_id": "モデル就業規則__c0190", "text": "本文 B"},
    ])
    judge = _MockJudge({"c0037": True, "c0190": False})
    ag = grounding_for_qa(qa, judge_section="hid={hidden_doc_id}", judge_model="m", llm=judge)
    assert ag.per_doc_sufficient == {
        "モデル就業規則__c0037": True,
        "モデル就業規則__c0190": False,
    }
    assert len(judge.calls) == 2


def test_grounding_falls_back_to_doc_id_without_chunk_id():
    """chunk_id が無い旧データは従来どおり doc_id 単位。"""
    qa = _make_qa([
        {"doc_id": "d1", "text": "本文 A"},
        {"doc_id": "d2", "text": "本文 B"},
    ])
    judge = _MockJudge({"d1": False, "d2": False})
    ag = grounding_for_qa(qa, judge_section="hid={hidden_doc_id}", judge_model="m", llm=judge)
    assert set(ag.per_doc_sufficient.keys()) == {"d1", "d2"}


def test_evidence_unit_on_rationale():
    from rageval.schema import Rationale
    assert Rationale(doc_id="d", chunk_id="d__c0001", text="x").evidence_unit() == "d__c0001"
    assert Rationale(doc_id="d", text="x").evidence_unit() == "d"


def test_grounding_batch_writes_grounded_jsonl(tmp_path):
    qa1 = _make_qa([
        {"doc_id": "d1", "text": "本文 A"},
        {"doc_id": "d2", "text": "本文 B"},
    ], question="Q1", answer="A1")
    qa1.qa_id = "q1"

    in_path = tmp_path / "in.jsonl"
    in_path.write_text(qa1.model_dump_json() + "\n", encoding="utf-8")

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(
        "## [GROUND_CHECK]\n"
        "Q={question} A={answer} hid={hidden_doc_id}\n",
        encoding="utf-8",
    )

    judge = _MockJudge({"d1": False, "d2": True})
    out_path = grounding_batch(
        in_path,
        judge_model="m",
        prompt_path=prompt_path,
        llm=judge,
    )
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["answer_grounding"]["per_doc_sufficient"] == {"d1": False, "d2": True}
