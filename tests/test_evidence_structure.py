"""classify_evidence_structure (chained / parallel 判別) と数値正規化のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rageval.evidence import (
    EvidenceStructure,
    classify_evidence_structure,
    kanji_to_int,
    normalize_numbers,
    salient_tokens,
)
from rageval.schema import QAItem, Rationale


# ---------- 数値の表記ゆれ正規化 ----------

def test_kanji_to_int_basic():
    assert kanji_to_int("五") == 5
    assert kanji_to_int("二十") == 20
    assert kanji_to_int("百一") == 101
    assert kanji_to_int("千二百三十四") == 1234
    assert kanji_to_int("二〇二六") == 2026   # 位取りなしの桁読み
    assert kanji_to_int("あ") is None


def test_normalize_numbers_kanji_years():
    """回帰: manual_v6_01 の「二十年未満のものは五年」↔「5年」。"""
    assert "20年" in normalize_numbers("二十年未満のものは五年")
    assert "5年" in normalize_numbers("二十年未満のものは五年")


def test_normalize_numbers_decimal_and_units():
    assert "1.3倍" in normalize_numbers("最高使用圧力の一・三倍の圧力")
    assert "0.2MPa" in normalize_numbers("〇・二メガパスカル")
    assert "0.1MPa" in normalize_numbers("0.1 MPa 以上")   # 空白除去


def test_salient_tokens_extracts_articles_forms_values():
    toks = salient_tokens("第百一条に基づき様式第６号を用い、保存年限は五年とする。")
    assert "第101条" in toks
    assert any(t.startswith("様式第6") for t in toks)
    assert "5年" in toks


# ---------- chained / parallel 判別 ----------

def _qa(answer: str, rationales: list[tuple[str, str]]) -> QAItem:
    """(chunk_id, text) のリストから最小の QAItem を作る。"""
    from datetime import datetime
    return QAItem(
        qa_id="t", question="Q?", answer=answer,
        rationale=[
            Rationale(doc_id=cid.split("__")[0], chunk_id=cid, page=None, text=text)
            for cid, text in rationales
        ],
        reasoning_complexity={}, retrieval_difficulty={}, source_structure={},
        explainability={"evidence_strictness": "multi-ref"},
        retrieval_level="Hard", answer_level="Medium", difficulty_rationale="",
        generation={"model": "m", "prompt_version": "v",
                    "generated_at": datetime.now().isoformat()},
    )


def test_chained_when_values_concentrate_in_terminal():
    """答えの固有値が終端チャンクだけに載っている → chained。"""
    qa = _qa(
        "経過年数二十年未満のものは六年、二十年以上のものは二年です。",
        [("11_kouatsu__c0001", "容器再検査を受け、経済産業省令で定める期間を経過した容器"),
         ("14_youki__c0043", "経過年数二十年未満のものは六年、経過年数二十年以上のものは二年")],
    )
    s = classify_evidence_structure(qa)
    assert s.label == "chained"
    assert s.covering_unit == "14_youki__c0043"


def test_chained_tolerates_kanji_vs_ascii():
    """表記ゆれ (回答は算用数字・原文は漢数字) を跨いで一致する。"""
    qa = _qa(
        "20年未満なら5年間の保存が必要です。",
        [("a__c0001", "保存年限は台帳に定める"),
         ("b__c0002", "経過年数二十年未満のものは五年とする")],
    )
    s = classify_evidence_structure(qa)
    assert s.label == "chained"
    assert s.covering_unit == "b__c0002"


def test_parallel_when_values_spread_over_units():
    """答えの固有値が2つの根拠に分かれて載っている → parallel (束ね問い)。"""
    qa = _qa(
        "点検周期は1年、保存年限は5年です。",
        [("a__c0001", "点検は1年ごとに行う"),
         ("b__c0002", "記録の保存年限は5年とする")],
    )
    assert classify_evidence_structure(qa).label == "parallel"


def test_unverifiable_when_answer_has_no_salient_values():
    qa = _qa("学科講習が行われます。",
             [("a__c0001", "学科講習を行う"), ("b__c0002", "実技講習を行う")])
    assert classify_evidence_structure(qa).label == "unverifiable"


def test_unverifiable_when_values_not_found_in_rationales():
    qa = _qa("保存年限は5年です。",
             [("a__c0001", "台帳に記録する"), ("b__c0002", "様式は別に定める")])
    assert classify_evidence_structure(qa).label == "unverifiable"


# ---------- 実データ回帰 ----------

_V25 = Path("data/raw_kg_plant_v25/batch_20260801_095407.jsonl")
_MANUAL = Path("data/kg_manual_v6/batch_manual_20260729.filtered.jsonl")


def _load(path: Path) -> list[QAItem]:
    return [QAItem.model_validate_json(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not _V25.exists(), reason="v25 データ無し")
def test_regression_bundled_question_is_parallel():
    """回帰: auto_f9e3662a (独立2問の束ね) が chained と誤認されないこと。"""
    qa = next(q for q in _load(_V25) if q.qa_id == "auto_f9e3662a")
    assert classify_evidence_structure(qa).label != "chained"


@pytest.mark.skipif(not _MANUAL.exists(), reason="手書きデータ無し")
def test_regression_handwritten_not_parallel():
    """回帰: 手書き4問 (本物の連鎖) が parallel と誤判定されないこと。

    手書き問は答えの言い換えを含むため chained か unverifiable は許容し、
    parallel (束ね) 判定だけを禁止する。
    """
    for qa in _load(_MANUAL):
        s = classify_evidence_structure(qa)
        assert s.label != "parallel", f"{qa.qa_id} が parallel 判定: {s.token_units}"
