"""Azure Document Intelligence (prebuilt-layout) wrapper.

Returns markdown for a PDF (headings, tables, page-break markers preserved).
Results are cached next to the source PDF as `<name>.di.md` so we don't burn
the API quota on every chunk run.

Env vars:
  AZURE_DI_ENDPOINT
  AZURE_DI_API_KEY
  AZURE_DI_MODEL (default: prebuilt-layout)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def is_configured() -> bool:
    return bool(os.getenv("AZURE_DI_ENDPOINT") and os.getenv("AZURE_DI_API_KEY"))


def _cache_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(pdf_path.suffix + ".di.md")


def analyze_pdf_to_markdown(
    pdf_path: Path,
    *,
    use_cache: bool = True,
    cache_path: Optional[Path] = None,
) -> str:
    """Call Azure DI prebuilt-layout and return markdown content.

    Caches result to a sibling `.pdf.di.md` file. If a fresher cache exists
    (modified after the PDF), it is returned without an API call.
    """
    cache = cache_path or _cache_path(pdf_path)
    if use_cache and cache.exists() and cache.stat().st_mtime >= pdf_path.stat().st_mtime:
        return cache.read_text(encoding="utf-8")

    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    api_key = os.getenv("AZURE_DI_API_KEY")
    model_id = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")
    if not (endpoint and api_key):
        raise RuntimeError(
            "AZURE_DI_ENDPOINT / AZURE_DI_API_KEY not set. "
            "Either configure .env or pass --pdf-backend pypdf."
        )

    # Lazy import so the dep is only required when actually used.
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    with pdf_path.open("rb") as f:
        body = f.read()
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=AnalyzeDocumentRequest(bytes_source=body),
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()
    markdown = result.content or ""

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(markdown, encoding="utf-8")
    return markdown


# ---------------- Page boundary parsing ----------------

# Azure DI inserts HTML comments like:
#   <!-- PageNumber="3" -->
#   <!-- PageBreak -->
_PAGE_NUMBER_RE = re.compile(r"<!--\s*PageNumber=\"(\d+)\"\s*-->")
_PAGE_BREAK_RE = re.compile(r"<!--\s*PageBreak\s*-->")


def iter_pages_from_markdown(md: str) -> list[tuple[int, str]]:
    """Walk DI-produced markdown and yield (page_number_1indexed, page_body).

    Strategy: split on `<!-- PageBreak -->` markers; tag each chunk with the
    PageNumber that appears just before/inside it. Falls back to incremental
    counter if PageNumber annotations are absent.
    """
    if not md.strip():
        return []
    # Some DI outputs put PageNumber right after a PageBreak; others omit it.
    parts = _PAGE_BREAK_RE.split(md)
    out: list[tuple[int, str]] = []
    counter = 1
    for part in parts:
        if not part.strip():
            continue
        m = _PAGE_NUMBER_RE.search(part)
        page = int(m.group(1)) if m else counter
        # Strip the PageNumber comment for cleaner body
        body = _PAGE_NUMBER_RE.sub("", part).strip()
        if body:
            out.append((page, body))
        counter = page + 1
    return out
