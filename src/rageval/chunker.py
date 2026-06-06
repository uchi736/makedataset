"""Character-based document chunker with markdown heading + reference tracking.

Reads `.txt` and `.md` from a directory, splits with a recursive character-based
strategy, and writes one JSONL of Chunks per file. For `.md`, the markdown
heading stack at each chunk's position becomes `section_path`. For all files,
references like "JIS Z 2241" / "第3章" / "別表2" are extracted into `references`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .schema import Chunk

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Split priorities: paragraph → sentence → newline → space → character.
_SPLIT_PATTERNS = [r"\n\n+", r"(?<=[。．.!?！？])\s+", r"\n", r"\s+", r""]


# ---------------- Reference extraction ----------------

_REFERENCE_PATTERNS: list[str] = [
    r"JIS\s*[A-Z]\s*\d{4}(?:-\d+)?",
    r"ISO\s*\d{4,5}(?:-\d+)?",
    r"IEC\s*\d{4,5}",
    r"ASME\s*[A-Z]+(?:\.\d+)?",
    r"第\s*\d+\s*章",
    r"第\s*\d+\s*節",
    r"第\s*\d+\s*条",            # 法令・規定 (第15条 等)
    r"第\s*\d+\s*項",            # 法令・規定 (第2項 等)
    r"別表\s*\d+",
    r"附属書\s*[A-Z]",
    r"\d+\.\d+(?:\.\d+)?\s*項",
]
_REFERENCE_RE = re.compile("|".join(f"(?:{p})" for p in _REFERENCE_PATTERNS))


# Path where corpus-discovered patterns are cached. The chunker auto-loads
# this and ORs them into the built-in regex on each chunk_text() call.
DISCOVERED_PATTERNS_PATH = Path("data/chunks/_discovered_patterns.json")


def _load_discovered_patterns() -> list[str]:
    """Load extra regex patterns previously discovered via discover_patterns()."""
    if not DISCOVERED_PATTERNS_PATH.exists():
        return []
    try:
        data = json.loads(DISCOVERED_PATTERNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for p in data.get("patterns", []):
        rgx = p.get("regex")
        if not rgx:
            continue
        try:
            re.compile(rgx)
        except re.error:
            continue
        out.append(rgx)
    return out


def _build_reference_re(extra_patterns: list[str] | None = None) -> re.Pattern[str]:
    """Compile built-in + discovered patterns into one regex."""
    all_patterns = list(_REFERENCE_PATTERNS) + (extra_patterns or _load_discovered_patterns())
    return re.compile("|".join(f"(?:{p})" for p in all_patterns))


def extract_references(text: str, extra_patterns: list[str] | None = None) -> list[str]:
    """Return unique references found in text, preserving discovery order."""
    pattern = _build_reference_re(extra_patterns)
    seen: dict[str, None] = {}
    for m in pattern.finditer(text):
        v = re.sub(r"\s+", " ", m.group(0)).strip()
        seen.setdefault(v, None)
    return list(seen.keys())


# ---------------- Markdown heading tracking ----------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def _split_md_into_sections(text: str) -> list[tuple[list[str], str]]:
    """Walk markdown text, yielding (section_path, body) chunks per heading."""
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    last_end = 0
    last_path: list[str] = []
    for m in _MD_HEADING_RE.finditer(text):
        body = text[last_end : m.start()]
        if body.strip():
            sections.append((list(last_path), body))
        level = len(m.group(1))
        title = m.group(2).strip()
        # Adjust stack to current level
        while len(stack) >= level:
            stack.pop()
        # Pad if jump (e.g., # → ###)
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)
        last_path = list(stack)
        last_end = m.end()
    tail = text[last_end:]
    if tail.strip():
        sections.append((list(last_path), tail))
    if not sections:
        sections = [([], text)]
    return sections


# ---------------- Recursive splitter (unchanged) ----------------

def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    for pattern in _SPLIT_PATTERNS:
        if pattern == "":
            parts = list(text)
        else:
            parts = re.split(pattern, text)
            parts = [p for p in parts if p]
        if not parts or max(len(p) for p in parts) >= size:
            continue
        return _merge(parts, size, overlap)
    return _merge(list(text), size, overlap)


def _merge(parts: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) + 1 <= size:
            buf = f"{buf} {part}".strip() if buf else part
        else:
            if buf:
                chunks.append(buf)
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                buf = f"{tail} {part}".strip()
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


# ---------------- Public API ----------------

def chunk_text(
    text: str,
    doc_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    is_markdown: bool = False,
) -> list[Chunk]:
    """Split text into chunks. Markdown mode tracks heading stack per chunk."""
    sections: list[tuple[list[str], str]]
    if is_markdown:
        sections = _split_md_into_sections(text)
    else:
        sections = [([], text)]

    out: list[Chunk] = []
    position = 0
    for section_path, body in sections:
        for piece in _recursive_split(body, chunk_size, chunk_overlap):
            out.append(Chunk(
                chunk_id=f"{doc_id}__c{position:04d}",
                doc_id=doc_id,
                page=None,
                text=piece,
                position=position,
                section_path=list(section_path),
                references=extract_references(piece + " " + " ".join(section_path)),
            ))
            position += 1
    return out


def _iter_doc_files(docs_dir: Path) -> Iterable[Path]:
    for ext in ("*.txt", "*.md", "*.pdf"):
        yield from sorted(docs_dir.rglob(ext))


def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number_1indexed, text)] for a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[chunker] page {i} of {path.name} failed: {e}")
            text = ""
        out.append((i, text))
    return out


def chunk_pdf(
    path: Path,
    doc_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """pypdf-based per-page text extraction + recursive split."""
    pages = _extract_pdf_pages(path)
    out: list[Chunk] = []
    position = 0
    for page_no, page_text in pages:
        if not page_text.strip():
            continue
        for piece in _recursive_split(page_text, chunk_size, chunk_overlap):
            out.append(Chunk(
                chunk_id=f"{doc_id}__c{position:04d}",
                doc_id=doc_id,
                page=page_no,
                text=piece,
                position=position,
                section_path=[],
                references=extract_references(piece),
            ))
            position += 1
    return out


def chunk_pdf_via_azure_di(
    path: Path,
    doc_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Azure DI prebuilt-layout: produces markdown, then chunks per-section
    while tracking page numbers from PageBreak markers."""
    from .azure_di import analyze_pdf_to_markdown, iter_pages_from_markdown

    md = analyze_pdf_to_markdown(path)
    pages = iter_pages_from_markdown(md)
    if not pages:
        # Fall back: chunk whole markdown as one block
        pages = [(1, md)]

    out: list[Chunk] = []
    position = 0
    for page_no, page_md in pages:
        sections = _split_md_into_sections(page_md)
        for section_path, body in sections:
            for piece in _recursive_split(body, chunk_size, chunk_overlap):
                out.append(Chunk(
                    chunk_id=f"{doc_id}__c{position:04d}",
                    doc_id=doc_id,
                    page=page_no,
                    text=piece,
                    position=position,
                    section_path=list(section_path),
                    references=extract_references(piece + " " + " ".join(section_path)),
                ))
                position += 1
    return out


PdfBackend = str  # "pypdf" | "azure_di" | "auto"


def chunk_directory(
    docs_dir: Path,
    out_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    pdf_backend: PdfBackend = "auto",
) -> dict[str, int]:
    """Chunk every text/markdown/pdf file in docs_dir, one JSONL per doc.

    pdf_backend:
      - "auto"     : Azure DI if env configured, else pypdf
      - "pypdf"    : force local pypdf
      - "azure_di" : force Azure DI (raises if env missing)
    """
    from .azure_di import is_configured as _di_configured

    if pdf_backend == "auto":
        pdf_backend = "azure_di" if _di_configured() else "pypdf"

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    for path in _iter_doc_files(docs_dir):
        doc_id = path.stem
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if pdf_backend == "azure_di":
                chunks = chunk_pdf_via_azure_di(
                    path, doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            else:
                chunks = chunk_pdf(
                    path, doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
        else:
            text = path.read_text(encoding="utf-8")
            chunks = chunk_text(
                text, doc_id,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                is_markdown=(suffix == ".md"),
            )
        out_path = out_dir / f"{doc_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        stats[doc_id] = len(chunks)
    return stats


def load_chunks(chunks_dir: Path) -> list[Chunk]:
    """Load all chunks from JSONL files under chunks_dir."""
    out: list[Chunk] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue  # skip metadata files like _discovered_patterns.json
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(Chunk.model_validate_json(line))
    return out


# ---------------- Corpus-driven pattern discovery (one-shot) ----------------

_DISCOVERY_PROMPT = """あなたは技術文書から「参照識別子のパターン」を発見する専門家です。

以下のサンプルテキストから、他の文書/章節/規格/帳票/型番を参照する **識別子の正規表現** を抽出してください。

# 抽出対象の例
- 規格番号: `JIS Z 2241`, `ISO 9001`
- 社内文書ID: `DS-MEC-104`, `QMS-FORM-301`, `TS-001-A`
- 章節番号: `5.2.3項`, `第3章`
- 図表番号: `別表2`, `Fig. 3-1`
- 製品型番: `XR-200`, `Model-A123`
- 改訂番号: `Rev.B`, `R2.1`

# 抽出しないもの
- 普通の数値・年号・温度・寸法
- 人名・地名・部署名
- 一般用語

# サンプルテキスト
---
{sample_text}
---

# 出力(JSONのみ。説明文禁止)
{{
  "patterns": [
    {{
      "regex": "<Python re モジュールでコンパイル可能な正規表現>",
      "kind": "standards|internal|section|form|product|revision|other",
      "example": "<コーパス内で実際に見つかった例>",
      "rationale": "<参照識別子とみなす根拠 1行>"
    }}
  ]
}}

# 重複禁止 (既に組み込み済み)
JIS規格, ISO規格, IEC規格, ASME規格, `第N章`, `第N節`, `別表N`, `附属書X`, `N.N.N項`
"""


def _build_discovery_sample(chunks: list[Chunk], max_chars: int = 12000) -> str:
    """Concatenate chunk texts up to max_chars (small corpus = all, large = truncated)."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        block = f"### {c.doc_id} / {c.chunk_id}\n{c.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def discover_patterns(
    chunks: list[Chunk],
    *,
    model: str,
    out_path: Path = DISCOVERED_PATTERNS_PATH,
    max_chars: int = 12000,
    llm: object | None = None,
) -> list[dict]:
    """Use an LLM to discover corpus-specific reference patterns. Writes JSON
    to `out_path` so subsequent `extract_references()` calls pick them up.

    Returns the list of {regex, kind, example, rationale} dicts that were saved.
    Patterns that don't compile or that match nothing in the sample are dropped.
    """
    from datetime import datetime
    from .llm import LLMError, _parse_json, generate as llm_generate

    sample = _build_discovery_sample(chunks, max_chars=max_chars)
    if not sample:
        raise RuntimeError("no chunks available for discovery")

    prompt = _DISCOVERY_PROMPT.format(sample_text=sample)
    caller = llm or llm_generate
    raw = caller(prompt=prompt, model=model, response_model=None, temperature=0.0, max_tokens=2048)
    if isinstance(raw, str):
        data = _parse_json(raw)
    else:
        data = raw
    if not isinstance(data, dict) or "patterns" not in data:
        raise LLMError(f"discovery response missing 'patterns' field: {raw!r}")

    accepted: list[dict] = []
    for entry in data.get("patterns", []):
        rgx = entry.get("regex")
        if not isinstance(rgx, str) or not rgx:
            continue
        try:
            compiled = re.compile(rgx)
        except re.error:
            continue
        # Reject patterns that don't match anything in the sample (likely garbage)
        if not compiled.search(sample):
            continue
        # Reject overly greedy patterns. Skip the check for very small samples
        # (test fixtures); for real corpora the 5% rule weeds out e.g. `\d+`.
        if len(sample) >= 500:
            match_chars = sum(len(m.group(0)) for m in compiled.finditer(sample))
            if match_chars > len(sample) * 0.05:
                continue
        accepted.append({
            "regex": rgx,
            "kind": entry.get("kind", "other"),
            "example": entry.get("example", ""),
            "rationale": entry.get("rationale", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "discovered_at": datetime.now().isoformat(),
        "model": model,
        "sample_chars": len(sample),
        "n_chunks_used": len([c for c in chunks if f"### {c.doc_id}" in sample]),
        "patterns": accepted,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return accepted
