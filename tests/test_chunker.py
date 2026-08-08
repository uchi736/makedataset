"""Tests for chunker section-path + reference extraction."""

from __future__ import annotations

from pathlib import Path

from rageval.chunker import (
    chunk_directory,
    chunk_text,
    extract_references,
    load_chunks,
)


def test_extract_references_jis_and_chapter():
    refs = extract_references("本締結は JIS Z 2241 と JIS B 1083-1 に従う。詳細は第3章 別表2 参照。")
    assert "JIS Z 2241" in refs
    assert "JIS B 1083-1" in refs
    assert "第3章" in refs
    assert "別表2" in refs


def test_extract_references_legal_article():
    """法令・規定の第N条/第N項を抽出できる"""
    refs = extract_references("本規定は第15条 および 第2項 を参照する。")
    assert "第15条" in refs
    assert "第2項" in refs


def test_extract_references_kanji_article():
    """回帰: 漢数字の条項番号を取りこぼさない (2026-08-08)。

    算用数字だけを見ていたため、条番号が漢数字支配の chunks_plant では
    158文書中129文書の references が 0 件になり、参照付きチャンクの 89.1% が
    算用数字表記の2文書に集中していた (ReferenceFollow の素材供給が実質その
    2文書のコーパスに縮退)。
    """
    refs = extract_references("ボイラー則第六十三条第二項および別表第七、様式第十一号による。")
    assert "第六十三条" in refs
    assert "第二項" in refs
    assert "別表第七" in refs
    assert "様式第十一号" in refs


def test_extract_references_fullwidth_article():
    """全角の算用数字表記 (09_kagi 系) も拾う。"""
    refs = extract_references("第１０７条 の規定により、別表第７を適用する。")
    assert "第１０７条" in refs
    assert "別表第７" in refs


def test_extract_references_keeps_surface_form():
    """抽出した表記は原文のまま返す (算用数字へ揃えない)。

    references は他チャンク本文への逐語照合に使われるので、算用数字へ
    正規化すると漢数字の文書では逆に当たらなくなる。
    """
    refs = extract_references("第六十三条による")
    assert refs == ["第六十三条"]


def test_extract_references_ignores_common_words():
    """「第三条件」「第一項目」のような一般語を参照と誤認しない。"""
    refs = extract_references("第三条件を満たすこと。第一項目は対象外とする。")
    assert refs == []


def test_extract_references_dedup_preserves_order():
    refs = extract_references("JIS Z 2241 を参照。再度 JIS Z 2241 と ISO 9001。")
    assert refs == ["JIS Z 2241", "ISO 9001"]


def test_extract_references_empty():
    assert extract_references("普通の文章で参照はない。") == []


def test_chunk_text_md_section_path():
    md = """# 第1章 安全規定

## 1.1 基本

ここは基本のテキスト。

## 1.2 詳細

ここは詳細のテキスト。

# 第2章 締結

第2章の本文。
"""
    chunks = chunk_text(md, doc_id="m", is_markdown=True, chunk_size=200)
    paths = [tuple(c.section_path) for c in chunks]
    assert ("第1章 安全規定", "1.1 基本") in paths
    assert ("第1章 安全規定", "1.2 詳細") in paths
    assert ("第2章 締結",) in paths


def test_chunk_text_txt_no_section_path():
    chunks = chunk_text("ただの本文です。", doc_id="t", is_markdown=False)
    assert all(c.section_path == [] for c in chunks)


def test_chunk_text_position_increments():
    md = "# A\nfoo\n# B\nbar\n# C\nbaz\n"
    chunks = chunk_text(md, doc_id="m", is_markdown=True, chunk_size=100)
    positions = [c.position for c in chunks]
    assert positions == sorted(positions)
    assert positions[0] == 0


def test_chunk_text_references_filled():
    chunks = chunk_text("JIS Z 2241 に従って測定する。", doc_id="d", is_markdown=False, chunk_size=200)
    assert chunks
    assert "JIS Z 2241" in chunks[0].references


def test_chunk_directory_roundtrip(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.md").write_text("# 第1章\nJIS B 1083 を準用する。\n", encoding="utf-8")
    out = tmp_path / "chunks"
    stats = chunk_directory(docs, out)
    assert stats["spec"] >= 1
    loaded = load_chunks(out)
    assert loaded[0].section_path == ["第1章"]
    assert "JIS B 1083" in loaded[0].references


def test_chunk_pdf_basic(tmp_path: Path):
    """A small PDF is extracted page-by-page with page numbers preserved."""
    pytest = __import__("pytest")
    try:
        from pypdf import PdfWriter
    except ImportError:
        pytest.skip("pypdf not installed")

    # Build a minimal 2-page PDF on the fly using pypdf's writer + reportlab is
    # overkill; instead we synthesize one via a known empty PDF and skip if the
    # writer can't make text pages. Use a real file if available, else skip.
    sample = Path("data/docs/モデル就業規則.pdf")
    if not sample.exists():
        pytest.skip("sample PDF not present")

    from rageval.chunker import chunk_pdf

    chunks = chunk_pdf(sample, doc_id="reg_test", chunk_size=400)
    assert len(chunks) > 0
    assert all(c.page is not None and c.page >= 1 for c in chunks)
    assert all(c.position >= 0 for c in chunks)


def test_chunk_directory_warns_documents_without_references(tmp_path: Path, capsys):
    """参照が1件も取れなかった文書を一覧で警告する (素材供給が無言で痩せないため)。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "with_ref.txt").write_text("第六十三条の規定による。\n", encoding="utf-8")
    (docs / "no_ref.txt").write_text("参照識別子を含まない普通の説明文です。\n", encoding="utf-8")
    chunk_directory(docs, tmp_path / "chunks")
    err = capsys.readouterr().err
    assert "参照が1件も取れなかった文書" in err
    assert "no_ref" in err
    assert "with_ref" not in err


# ---------------- chunk_id の分解 (doc_id / position の正本) ----------------

def test_doc_id_of_keeps_double_underscore_in_doc_id():
    """回帰: doc_id 自体が '__' を含んでも文書名を壊さない。

    split('__') 実装だと 'a__b__c0012' が 'a' になり、別文書が同一視される。
    """
    from rageval.chunker import doc_id_of
    assert doc_id_of("a__b__c0012") == "a__b"
    assert doc_id_of("09_kagi_kaishaku_240315__c0154") == "09_kagi_kaishaku_240315"
    assert doc_id_of("no_suffix") == "no_suffix"


def test_position_of_reads_trailing_number():
    from rageval.chunker import position_of
    assert position_of("a__b__c0012") == 12
    assert position_of("d__c0000") == 0
    assert position_of("no_suffix") is None


def test_doc_id_of_matches_chunk_id_built_by_chunk_text():
    """chunk_text が組み立てた chunk_id を、そのまま復元できること。"""
    from rageval.chunker import doc_id_of, position_of
    chunks = chunk_text("本文です。", doc_id="a__b", is_markdown=False)
    c = chunks[0]
    assert doc_id_of(c.chunk_id) == c.doc_id
    assert position_of(c.chunk_id) == c.position


def test_discover_patterns_accepts_valid(tmp_path: Path):
    """Mock LLM returns a corpus-specific pattern; it should be persisted and
    applied to subsequent extract_references calls."""
    import json

    from rageval.chunker import (
        DISCOVERED_PATTERNS_PATH,
        discover_patterns,
        extract_references,
    )
    from rageval.schema import Chunk

    chunks = [
        Chunk(chunk_id="c0", doc_id="d", text="本規定は DS-MEC-104 を引用する。"),
        Chunk(chunk_id="c1", doc_id="d", text="フォーム QMS-FORM-301 に記載のこと。"),
    ]

    def mock_llm(*, prompt, **kwargs):
        return json.dumps({
            "patterns": [
                {"regex": r"DS-[A-Z]+-\d+", "kind": "internal",
                 "example": "DS-MEC-104", "rationale": "社内文書ID"},
                {"regex": r"QMS-FORM-\d+", "kind": "form",
                 "example": "QMS-FORM-301", "rationale": "フォーム番号"},
                {"regex": r"[invalid regex(", "kind": "other",
                 "example": "?", "rationale": "壊れた正規表現は捨てるはず"},
            ]
        })

    # Redirect the discovered-pattern file to tmp so we don't litter the repo.
    out_path = tmp_path / "_discovered_patterns.json"
    accepted = discover_patterns(chunks, model="mock", out_path=out_path, llm=mock_llm)
    assert len(accepted) == 2  # invalid regex rejected
    assert any(p["regex"] == r"DS-[A-Z]+-\d+" for p in accepted)

    # Verify extract_references picks up the discovered patterns when passed explicitly
    refs = extract_references(
        "DS-MEC-104 と QMS-FORM-301 を参照",
        extra_patterns=[p["regex"] for p in accepted],
    )
    assert "DS-MEC-104" in refs
    assert "QMS-FORM-301" in refs


def test_discover_patterns_rejects_overly_greedy(tmp_path: Path):
    """Patterns that match > 5% of a real-sized sample are rejected."""
    import json

    from rageval.chunker import discover_patterns
    from rageval.schema import Chunk

    # Big enough sample to trigger the 5% rule (>= 500 chars)
    chunks = [Chunk(chunk_id="c0", doc_id="d", text="a" * 1000)]

    def mock_llm(*, prompt, **kwargs):
        return json.dumps({
            "patterns": [{"regex": r"a", "kind": "other", "example": "a", "rationale": "greedy"}]
        })

    out_path = tmp_path / "_p.json"
    accepted = discover_patterns(chunks, model="mock", out_path=out_path, llm=mock_llm)
    assert accepted == []


# ---------------- 発見パターン台帳は処理対象のチャンク置き場に紐づく ----------------

def _write_ledger(chunks_dir: Path, regex: str) -> None:
    import json

    from rageval.chunker import discovered_patterns_path

    p = discovered_patterns_path(chunks_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"patterns": [{"regex": regex, "kind": "internal"}]}),
                 encoding="utf-8")


def test_chunk_directory_uses_ledger_under_its_own_out_dir(tmp_path: Path):
    """台帳は出力先の直下から読む (data/chunks 固定の共有をやめた)。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.txt").write_text("本規定は DS-MEC-104 を引用する。\n", encoding="utf-8")

    out = tmp_path / "chunks_a"
    _write_ledger(out, r"DS-[A-Z]+-\d+")
    chunk_directory(docs, out)
    assert "DS-MEC-104" in load_chunks(out)[0].references


def test_chunk_directory_does_not_share_ledger_across_corpora(tmp_path: Path):
    """回帰: 片方のコーパスの台帳がもう片方の references を書き換えないこと。

    以前は data/chunks 固定の台帳を全呼び出しが暗黙に読み込んでいたため、
    退役コーパスで発見した規則が稼働中コーパスの参照台帳に混入していた
    (実測: chunks_plant の 33 件が退役コーパス由来の型番だった)。
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.txt").write_text("本規定は DS-MEC-104 を引用する。\n", encoding="utf-8")

    corpus_a = tmp_path / "chunks_a"
    _write_ledger(corpus_a, r"DS-[A-Z]+-\d+")
    chunk_directory(docs, corpus_a)

    corpus_b = tmp_path / "chunks_b"        # 台帳なし
    chunk_directory(docs, corpus_b)
    assert load_chunks(corpus_b)[0].references == []


def test_extract_references_does_not_autoload_any_ledger(tmp_path: Path, monkeypatch):
    """extra_patterns を渡さない限り、どの台帳も暗黙に読み込まない。"""
    from rageval import chunker

    ledger = tmp_path / "_discovered_patterns.json"
    _write_ledger(tmp_path, r"DS-[A-Z]+-\d+")
    monkeypatch.setattr(chunker, "DISCOVERED_PATTERNS_PATH", ledger)
    assert extract_references("本規定は DS-MEC-104 を引用する。") == []
