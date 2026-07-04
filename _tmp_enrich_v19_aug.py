"""Post-hoc injection of relation_edges into a KG-PoC QA JSONL.

Logic identical to v20_batch2/batch_20260621_0005.enriched.jsonl:

1. rageval.chunker.load_chunks で data/chunks_plant 読み込み
2. 各 QA の rationale.text を含むチャンクを doc_id 内で探す (空白正規化込み)
3. 異 doc 跨ぎペアを抽出
4. anchor.references の各 ref token が target.text/section_path に含まれるか照合
5. 含まれれば (anchor.chunk_id, ref, target.chunk_id) のトリプルを relation_edges に追加
6. {path}.jsonl -> {path}.enriched.jsonl

No LLM. Deterministic.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rageval.chunker import load_chunks  # noqa: E402
from rageval.schema import Chunk  # noqa: E402


WS_RE = re.compile(r"\s+")


def _ws_normalize(s: str) -> str:
    return WS_RE.sub("", s)


def _find_anchor_chunks(rationale: list[dict], chunks_by_doc: dict[str, list[Chunk]]) -> list[Chunk]:
    """For each rationale entry, find a chunk whose text contains rationale.text
    (whitespace-insensitive) within rationale.doc_id."""
    anchors: list[Chunk] = []
    for r in rationale:
        doc_id = r.get("doc_id")
        text = r.get("text", "")
        if not doc_id or not text:
            continue
        needle = _ws_normalize(text)
        if not needle:
            continue
        candidates = chunks_by_doc.get(doc_id, [])
        matched: Chunk | None = None
        for c in candidates:
            haystack = _ws_normalize(c.text)
            if needle in haystack:
                matched = c
                break
        if matched is not None:
            anchors.append(matched)
    return anchors


def _enrich_one(qa: dict, chunks_by_doc: dict[str, list[Chunk]]) -> list[dict]:
    rationale = qa.get("rationale") or []
    anchors = _find_anchor_chunks(rationale, chunks_by_doc)
    if len(anchors) < 2:
        return []

    # 異 doc 跨ぎペア
    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    n = len(anchors)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = anchors[i]
            t = anchors[j]
            if a.doc_id == t.doc_id:
                continue
            blob = t.text + " " + " ".join(t.section_path)
            for ref in a.references:
                if ref in blob:
                    key = (a.chunk_id, ref, t.chunk_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "subject_chunk_id": a.chunk_id,
                        "predicate": ref,
                        "object_chunk_id": t.chunk_id,
                    })
    return edges


def main(src_path: str, chunks_dir: str = "data/chunks_plant") -> dict:
    src = Path(src_path)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    dst = Path(str(src).replace(".jsonl", ".enriched.jsonl"))

    chunks = load_chunks(Path(chunks_dir))
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)
    # Stable ordering by position
    for k in chunks_by_doc:
        chunks_by_doc[k].sort(key=lambda c: c.position)

    total = 0
    with_edges = 0
    total_edges = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            qa = json.loads(line)
            edges = _enrich_one(qa, chunks_by_doc)
            qa["relation_edges"] = edges
            total += 1
            if edges:
                with_edges += 1
                total_edges += len(edges)
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")

    return {
        "enriched_path": str(dst),
        "total": total,
        "with_edges": with_edges,
        "total_edges": total_edges,
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw_kg_plant_v19_aug.jsonl"
    chunks = sys.argv[2] if len(sys.argv) > 2 else "data/chunks_plant"
    result = main(src, chunks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
