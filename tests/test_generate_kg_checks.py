"""generate_kg の機械検品 (重複根拠・無内容回答) のテスト。"""

from __future__ import annotations

from datetime import datetime

import pytest

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


def test_trace_identifiers_ignore_number_unit_and_skip_chunks_without_identifiers():
    """回帰 (監査 rank 13): 識別子は条・項・様式・別表の番号だけ。

    - 数値+単位 (「六年」) は識別子ではないので、それだけでは合格にしない
    - 識別子を1つも含まない根拠は検査対象外 (棄却しない)
    """
    from rageval.generate_kg import CheckContext, _check_trace_identifiers
    chunks = {
        "a__c0001": _chunk("a__c0001", "第八十二条 溶接自主検査は六年ごとに行うこと。"),
        # 識別子を持たない根拠 (chunks_plant で 614/3175 がこの形)
        "b__c0002": _chunk("b__c0002", "溶接部は割れその他の欠陥がない状態とする。"),
    }

    def ctx(answer: str) -> CheckContext:
        qa = _qa([("a__c0001", "x"), ("b__c0002", "y")])
        qa.answer = answer
        return CheckContext(qa=qa, path=None, support=["a__c0001", "b__c0002"],
                            chunks_by_id=chunks)

    # 数値+単位しか書いていない回答は不合格 (旧実装は「6年」の一致で通していた)
    ng = _check_trace_identifiers(ctx("六年ごとに実施する必要があります。"))
    assert ng is not None and "識別子" in ng
    # 条番号を書けば合格。識別子を持たない b__c0002 は棄却理由にならない
    assert _check_trace_identifiers(
        ctx("第82条により、六年ごとに実施します。")) is None


def test_trace_identifiers_pass_when_no_chunk_has_identifier(capsys):
    """識別子がどの根拠にも無い QA は棄却せず、検査対象外だったことを1行残す。"""
    from rageval.generate_kg import CheckContext, _check_trace_identifiers
    chunks = {"a__c0001": _chunk("a__c0001", "溶接部は良好な状態とする。")}
    qa = _qa([("a__c0001", "x")])
    qa.answer = "溶接部を良好な状態に保つ必要があります。"
    assert _check_trace_identifiers(CheckContext(
        qa=qa, path=None, support=["a__c0001"], chunks_by_id=chunks)) is None
    assert "識別子系列の検品は対象外" in capsys.readouterr().out


def test_intermediate_exposure_catches_two_character_node():
    """回帰 (監査 rank 9): 2文字の中間ノードも露出として落とす (下限は共通定数)。"""
    from rageval.generate_kg import (
        MIN_EXPOSED_FORM_LEN, _question_exposes_intermediate,
    )
    assert MIN_EXPOSED_FORM_LEN == 2
    path = KGPath(edges=[
        PathEdge("貯槽設備", "REQUIRES", "配管", ["c1"], 2, ["貯槽設備"], ["配管"]),
        PathEdge("配管", "REQUIRES", "肉厚測定", ["c2"], 2, ["配管"], ["肉厚測定"]),
    ])
    assert _question_exposes_intermediate(path, "配管について必要な測定は何ですか。") == "配管"
    assert _question_exposes_intermediate(path, "貯槽設備で必要な測定は何ですか。") is None


def test_lexical_leak_skip_on_small_corpus_is_logged(capsys):
    """回帰 (監査 rank 19): 小さいコーパスで検品が外れたら理由を1行出す。"""
    from rageval.generate_kg import (
        MIN_CORPUS_FOR_LEAK_CHECK, CheckContext, _check_lexical_leak,
    )
    from rageval.rag_verify import BM25Index

    corpus = [_chunk(f"d__c{i:04d}", f"第{i}条 対象{i}の管理を定める。") for i in range(3)]
    qa = _qa([("d__c0000", "x"), ("d__c0002", "y")])
    ctx = CheckContext(qa=qa, path=None, support=["d__c0000", "d__c0002"],
                       chunks_by_id={c.chunk_id: c for c in corpus},
                       bm25=BM25Index(corpus),
                       chunk_row={c.chunk_id: i for i, c in enumerate(corpus)})
    assert _check_lexical_leak(ctx) is None
    out = capsys.readouterr().out
    assert "語彙漏れ検品を省略" in out
    assert str(MIN_CORPUS_FOR_LEAK_CHECK) in out


def test_vacuous_answer_margin_comes_from_named_constant(monkeypatch):
    """回帰 (監査 rank 20): 余白の直値をやめ、定数で挙動が動くこと。"""
    import rageval.generate_kg as g
    assert g.EMPTY_ANSWER_MARGIN == 12
    monkeypatch.setattr(g, "EMPTY_ANSWER_MARGIN", 0)
    assert g._vacuous_answer(_path("学科講習"), "学科講習が行われます。") is False


def test_docid_leak_uses_loaded_doc_ids_not_naming_convention():
    """回帰 (監査 rank 7): 露出の照合は読み込んだ doc_id の断片から作る。

    旧実装は技報 vNNnN_NN / 法令 NN_ローマ字 の正規表現だったため、同時に
    稼働している chunks_synth (SYN01_…) では一度も一致しなかった。
    逆に日本語の断片 (実務者が知っている文書名) は露出とみなさない。
    """
    from rageval.generate_kg import CheckContext, _check_docid_leak
    corpus = {
        "SYN01_白波_設備保全管理規程__c0000": _chunk(
            "SYN01_白波_設備保全管理規程__c0000", "本規程は保全の基準を定める。"),
        "33_denji_kisoku__c0001": _chunk(
            "33_denji_kisoku__c0001", "第三条 電気工作物の基準を定める。"),
    }

    def ctx(question: str) -> CheckContext:
        qa = _qa([("33_denji_kisoku__c0001", "x")])
        qa.question = question
        qa.answer = "第3条によります。"
        return CheckContext(qa=qa, path=None,
                            support=["33_denji_kisoku__c0001"], chunks_by_id=corpus)

    # 旧実装が空振りしていた命名 (SYN01_…) の前半露出を捕まえる
    leak = _check_docid_leak(ctx("SYN01 の規定では点検はどう定められていますか。"))
    assert leak is not None and "断片" in leak
    # 完全一致は従来どおり
    assert _check_docid_leak(ctx("33_denji_kisoku では何を定めていますか。")) is not None
    # 日本語の文書名は内部名の露出ではない (過剰棄却しない)
    assert _check_docid_leak(ctx("設備保全管理規程では点検はどう定められていますか。")) is None


def test_meta_leak_matches_machine_labels_and_path_notation_only():
    """回帰 (監査 rank 8): 一般名詞「始点」「終端」で decoy を棄却しないこと。"""
    from rageval.generate_kg import CheckContext, _check_meta_leak
    path = KGPath(edges=[
        PathEdge("A規程の規定", "SIMILAR_BUT_UNRELATED", "B規程の規定",
                 ["a__c0001"], 0, ["A規程"], ["B規程"]),
        PathEdge("B規程の規定", "NO_RELATION", "B規程の内容",
                 ["b__c0002"], 0, ["B規程"], ["B規程の内容"]),
    ])

    def ctx(question: str, answer: str) -> CheckContext:
        qa = _qa([("a__c0001", "x")])
        qa.question, qa.answer = question, answer
        return CheckContext(qa=qa, path=path, support=["a__c0001"], chunks_by_id={})

    # 原文語をそのまま書いた回答は通す (data/kg_decoy_v1 で 4/4 件を壊した形)
    assert _check_meta_leak(ctx(
        "溶接線の始点から十五メートルごとの検査は求められますか。",
        "終端までの間隔を定めた規定は確認できない。")) is None
    # 機械語と経路記法は棄却する
    assert _check_meta_leak(ctx(
        "SIMILAR_BUT_UNRELATED の対について答えよ。", "確認できない。")) is not None
    assert _check_meta_leak(ctx(
        "**始点**: A規程 について答えよ。", "確認できない。")) is not None
    assert _check_meta_leak(ctx("辺1: の関係は？", "確認できない。")) is not None


def test_strip_internal_labels_keeps_general_nouns():
    """回帰 (監査 rank 8): 段2へ渡す素材から抜くのは機械語と経路記法だけ。"""
    from rageval.generate_kg import _strip_internal_labels
    path = KGPath(edges=[
        PathEdge("A規程の規定", "SIMILAR_BUT_UNRELATED", "B規程の規定",
                 ["a__c0001"], 0, ["A規程"], ["B規程"]),
    ])
    reason = "SIMILAR_BUT_UNRELATED とあるが、始点の設備と終端の規定は所管が異なる"
    stripped = _strip_internal_labels(reason, path)
    assert "始点" in stripped and "終端" in stripped
    assert "SIMILAR_BUT_UNRELATED" not in stripped


def test_check_tables_are_fail_closed():
    """回帰 (監査 rank 22): 表に無いキーは素通りでなく例外。"""
    from rageval.generate_kg import (
        ALL_CHECK_KEYS, CHECKS_BY_TYPE, CheckContext, run_checks,
    )
    assert set(ALL_CHECK_KEYS) <= set(CHECKS_BY_TYPE)
    # decoy の規則は表の定義位置に載っている (後付け登録に戻さない)
    assert [n for n, _ in CHECKS_BY_TYPE["kg_decoy"]] == ["棄権明示", "内部用語"]
    ctx = CheckContext(qa=_qa([]), path=None, support=[], chunks_by_id={})
    with pytest.raises(ValueError):
        run_checks(ctx, "kg_missng")   # 綴り違い = 型別規則ゼロで通してはいけない


def test_wordlists_are_configurable(monkeypatch):
    """回帰 (監査 rank 21): 受理語彙をプロンプト側の言い回しへ追随させられること。"""
    from rageval.generate_kg import (
        ABSTAIN_WORDS_ENV, EXCLUSION_WORDS_ENV, CheckContext,
        _check_abstention_answer, _check_exclusion_wording,
    )
    qa = _qa([("a__c0001", "該当する規定は見当たりません。")])
    qa.answer = "該当する規定は見当たりません。"
    ctx = CheckContext(qa=qa, path=None, support=["a__c0001"], chunks_by_id={})

    monkeypatch.delenv(ABSTAIN_WORDS_ENV, raising=False)
    monkeypatch.delenv(EXCLUSION_WORDS_ENV, raising=False)
    assert _check_abstention_answer(ctx) is not None      # 既定語彙には無い言い回し
    assert _check_exclusion_wording(ctx) is not None

    monkeypatch.setenv(ABSTAIN_WORDS_ENV, "見当たりません、確認できない")
    monkeypatch.setenv(EXCLUSION_WORDS_ENV, "見当たりません")
    assert _check_abstention_answer(ctx) is None
    assert _check_exclusion_wording(ctx) is None


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
