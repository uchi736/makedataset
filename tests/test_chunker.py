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
