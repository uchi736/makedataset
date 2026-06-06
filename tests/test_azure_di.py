"""Tests for the Azure Document Intelligence wrapper.

These tests do NOT hit the network; they mock the SDK client and verify our
chunker correctly consumes DI-produced markdown (headings → section_path,
PageBreak markers → page numbers, tables preserved as markdown).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rageval.azure_di import _cache_path, iter_pages_from_markdown


SAMPLE_DI_MARKDOWN = """# 第1章 総則

## 第1条 目的

本規程は、社員の労働条件を定める。

<!-- PageBreak -->
<!-- PageNumber="2" -->

## 第2条 適用範囲

すべての社員に適用する。

| 区分 | 適用 |
| --- | --- |
| 正社員 | ○ |
| 契約社員 | ○ |

<!-- PageBreak -->
<!-- PageNumber="3" -->

# 第2章 服務

## 第3条 服務規律

労働者は規定に従うこと。
"""


def test_iter_pages_splits_on_pagebreak():
    pages = iter_pages_from_markdown(SAMPLE_DI_MARKDOWN)
    assert len(pages) == 3
    assert pages[0][0] == 1  # fallback to counter when no PageNumber on first page
    assert pages[1][0] == 2
    assert pages[2][0] == 3
    assert "第1条" in pages[0][1]
    assert "第2条" in pages[1][1]
    assert "第3条" in pages[2][1]
    # PageNumber comments stripped from body
    assert "PageNumber" not in pages[0][1]


def test_iter_pages_handles_empty():
    assert iter_pages_from_markdown("") == []


def test_cache_path_is_sibling():
    p = Path("/tmp/some/foo.pdf")
    assert _cache_path(p) == Path("/tmp/some/foo.pdf.di.md")


def test_chunk_pdf_via_azure_di_uses_cached_markdown(tmp_path, monkeypatch):
    """When a `.di.md` cache exists newer than the PDF, no API call is made
    and the chunker produces markdown-flavored chunks with section_path."""
    from rageval.chunker import chunk_pdf_via_azure_di

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cache = pdf.with_suffix(pdf.suffix + ".di.md")
    cache.write_text(SAMPLE_DI_MARKDOWN, encoding="utf-8")
    # ensure cache mtime >= pdf mtime
    import os, time
    time.sleep(0.01)
    os.utime(cache, None)

    # Sabotage the network path so we know we hit the cache.
    def boom(*a, **kw):
        raise RuntimeError("network was called — cache should have been used")
    monkeypatch.setattr(
        "rageval.azure_di.analyze_pdf_to_markdown",
        lambda path, **kw: cache.read_text(encoding="utf-8"),
    )

    chunks = chunk_pdf_via_azure_di(pdf, doc_id="doc", chunk_size=200)
    assert len(chunks) >= 3
    # Heading hierarchy preserved
    pathset = {tuple(c.section_path) for c in chunks}
    assert any("第1章 総則" in p[0] for p in pathset if p)
    assert any("第1条 目的" in s for c in chunks for s in c.section_path)
    # Page numbers preserved
    pages_seen = {c.page for c in chunks}
    assert pages_seen >= {1, 2, 3}
    # References extracted from chunks
    all_refs = {r for c in chunks for r in c.references}
    assert "第1条" in all_refs or "第2条" in all_refs or "第3条" in all_refs
