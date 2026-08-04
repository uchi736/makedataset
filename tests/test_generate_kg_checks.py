"""generate_kg の機械検品 (重複根拠・無内容回答) のテスト。"""

from __future__ import annotations

from datetime import datetime

from rageval.generate_kg import _rationale_cross_overlap_error, _vacuous_answer
from rageval.kg import KGPath, PathEdge
from rageval.schema import Chunk, QAItem, Rationale


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid.split("__")[0], page=None,
                 text=text, position=0)


def _qa(rationales: list[tuple[str, str]]) -> QAItem:
    return QAItem(
        qa_id="t", question="Q?", answer="A",
        rationale=[Rationale(doc_id=cid.split("__")[0], chunk_id=cid,
                             page=None, text=text) for cid, text in rationales],
        reasoning_complexity={}, retrieval_difficulty={}, source_structure={},
        explainability={"evidence_strictness": "multi-ref"},
        retrieval_level="Hard", answer_level="Medium", difficulty_rationale="",
        generation={"model": "m", "prompt_version": "v",
                    "generated_at": datetime.now().isoformat()},
    )


def test_cross_overlap_rejects_chunker_overlap_duplicate():
    """回帰: チャンク重複で同じ文が2チャンクに載り、両方から引いた問い。"""
    shared = "吹出し管であって吹出し弁又は吹出しコックを取り付けたものを備えなければならない。"
    chunks = {
        "d__c0027": _chunk("d__c0027", "蒸気ボイラーには、" + shared),
        "d__c0034": _chunk("d__c0034", shared + " ３　吹出し弁は沈殿物がたまらない構造とする。"),
    }
    qa = _qa([("d__c0027", "蒸気ボイラーには、" + shared),
              ("d__c0034", shared)])
    err = _rationale_cross_overlap_error(qa, ["d__c0027", "d__c0034"], chunks)
    assert err is not None and "重複" in err


def test_cross_overlap_passes_distinct_evidence():
    chunks = {
        "a__c0001": _chunk("a__c0001", "容器再検査を受けなければならない。"),
        "b__c0002": _chunk("b__c0002", "経過年数二十年未満のものは六年とする。"),
    }
    qa = _qa([("a__c0001", "容器再検査を受けなければならない。"),
              ("b__c0002", "経過年数二十年未満のものは六年とする。")])
    assert _rationale_cross_overlap_error(qa, ["a__c0001", "b__c0002"], chunks) is None


def _path(terminal_name: str, terminal_forms: list[str] | None = None) -> KGPath:
    return KGPath(edges=[
        PathEdge("免許", "REQUIRES", "講習", ["c1"], 2, ["免許"], ["講習"]),
        PathEdge("講習", "HAS_STEP", terminal_name, ["c2"], 2, ["講習"],
                 terminal_forms or [terminal_name]),
    ])


def test_vacuous_answer_detects_node_name_echo():
    """回帰: 「学科講習が行われます。」型 (終端ノード名 + 挨拶だけ)。"""
    assert _vacuous_answer(_path("学科講習"), "学科講習が行われます。") is True
    assert _vacuous_answer(_path("学科講習"), "学科講習があります。") is True


def test_vacuous_answer_passes_substantive_content():
    """終端ノード名を含んでいても、事実の中身が続く回答は通す。"""
    assert _vacuous_answer(
        _path("学科講習"),
        "学科講習として、酸素欠乏症及び救急そ生に関する知識を修める必要があります。",
    ) is False


def test_lexical_leak_check_catches_terminal_heading_in_question():
    """終端チャンクの見出し語を質問に写した問いは、BM25 だけで直撃される → 棄却。

    逆に、終端が一般規定 (質問の状況語を含まない) なら通過する。
    質問の状況語が終端にも載っている素材は、どう問うても語彙で当たるため
    この検品で落ちる — それは素材が single_fact 相当ということ (仕様)。
    """
    from rageval.generate_kg import CheckContext, _check_lexical_leak
    from rageval.rag_verify import BM25Index

    fillers = [
        _chunk(f"f__c{i:04d}", f"第{i}条 これは無関係な一般条文で、対象{i}の管理を定める。")
        for i in range(58)
    ]
    start = _chunk("s__c0000", "容器再検査を受けなければならない。")
    # 見出し語を含む終端 (語彙が濃い) と、一般規定の終端 (状況語を含まない)
    t_leaky = _chunk("t__c0001",
                     "液化石油ガス自動車燃料装置用容器（溶接容器に限る。）については、"
                     "経過年数二十年未満のものは六年とする。")
    t_generic = _chunk("t__c0002",
                       "前項の規定による当該期間は、経過した区分に応じ、それぞれ同表に定める。")
    corpus = fillers + [start, t_leaky, t_generic]
    bm25 = BM25Index(corpus)
    chunk_row = {c.chunk_id: i for i, c in enumerate(corpus)}

    def ctx(question: str, terminal: str) -> CheckContext:
        qa = _qa([("s__c0000", "容器再検査を受けなければならない。"),
                  (terminal, "ダミー引用")])
        qa.question = question
        return CheckContext(qa=qa, path=None, support=["s__c0000", terminal],
                            chunks_by_id={c.chunk_id: c for c in corpus},
                            bm25=bm25, chunk_row=chunk_row)

    # 悪い例: 終端の見出し語をそのまま質問に書く → 直撃 → 棄却
    leak = _check_lexical_leak(ctx(
        "液化石油ガス自動車燃料装置用容器（溶接容器に限る）の再検査の周期は？",
        "t__c0001"))
    assert leak is not None and "語彙" in leak

    # 良い例: 終端が一般規定で、質問と語彙を共有しない → 通過
    ok = _check_lexical_leak(ctx(
        "自動車に固定して使うガスの入れ物を使い続けたい。点検の頻度を教えてほしい。",
        "t__c0002"))
    assert ok is None


# ---------- traceability / negation の型別検品 ----------

def test_trace_identifiers_requires_ordered_citations():
    """traceability: 各根拠の識別子が回答に経路順で現れることを要求。"""
    from rageval.generate_kg import CheckContext, _check_trace_identifiers
    chunks = {
        "a__c0001": _chunk("a__c0001", "第八十二条 溶接自主検査は技術基準に適合すること。"),
        "b__c0002": _chunk("b__c0002", "第三条 検査の基準は平成9年通商産業省令第51号に適合するものとする。"),
    }
    def ctx(answer: str) -> CheckContext:
        qa = _qa([("a__c0001", "x"), ("b__c0002", "y")])
        qa.answer = answer
        return CheckContext(qa=qa, path=None, support=["a__c0001", "b__c0002"],
                            chunks_by_id=chunks)
    # 両方の識別子が出る → 通過 (順序は不問: 自然な説明文は前後する)
    ok = _check_trace_identifiers(ctx(
        "まず第82条が溶接自主検査を求め、その基準は第3条により省令第51号と定められている。"))
    assert ok is None
    ok2 = _check_trace_identifiers(ctx(
        "省令第51号への適合が第3条で求められており、それは第82条の溶接自主検査に基づく。"))
    assert ok2 is None
    # 片方の識別子が無い → 棄却
    ng = _check_trace_identifiers(ctx("第82条により行う必要があります。"))
    assert ng is not None and "識別子" in ng


def test_exclusion_wording_required_for_negation():
    """negation: 引用に明示的な除外文言が無ければ棄却 (捏造否定の防止)。"""
    from rageval.generate_kg import CheckContext, _check_exclusion_wording
    qa_ok = _qa([("a__c0001", "電気事業法の適用を受けるボイラーについては、この限りでない。")])
    qa_ng = _qa([("a__c0001", "ボイラーの安全管理について定める。")])
    assert _check_exclusion_wording(CheckContext(
        qa=qa_ok, path=None, support=["a__c0001"], chunks_by_id={})) is None
    err = _check_exclusion_wording(CheckContext(
        qa=qa_ng, path=None, support=["a__c0001"], chunks_by_id={}))
    assert err is not None and "除外文言" in err


def test_relation_echo_rejected():
    """回帰: 段1の関係文をそのまま回答に写した QA を棄却。"""
    from rageval.generate_kg import CheckContext, _check_relation_echo
    rel = "高性能吸収液、熱マネジメント技術が必要であり、文脈的に並列されている"
    qa = _qa([("a__c0001", "x")])
    qa.answer = "CO2分離回収には" + rel
    assert _check_relation_echo(CheckContext(
        qa=qa, path=None, support=[], chunks_by_id={}, relation=rel)) is not None
    qa.answer = "高性能吸収液と熱マネジメント技術が必要です。"
    assert _check_relation_echo(CheckContext(
        qa=qa, path=None, support=[], chunks_by_id={}, relation=rel)) is None
