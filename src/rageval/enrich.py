"""relation_edges の機械付与 (決定論・LLM不要)。

QA の rationale が指すチャンク間の参照関係を (subject, predicate, object) の
トリプルとして QAItem.relation_edges に書き込む。旧 scripts/legacy_manual_qa/
enrich_v19_aug.py の後継で、v3 の文書スコープ規則を適用する:

- 同一文書内: anchor.references のローカル参照 (第N条・別表N等) が target の
  本文/章立てに現れれば辿ってよい
- 異文書: CrossDocPolicy.join_ok を満たす対 (エッジ台帳・規格番号・明示参照)
  のみエッジにする。裸の条番号一致では繋がない
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .chunker import load_chunks
from .evidence import match_rationales
from .sampling import CrossDocPolicy, _is_global_ref, load_cross_doc_policy
from .schema import Chunk, QAItem, RelationEdge

_WS_RE = re.compile(r"\s+")


def _ws_normalize(s: str) -> str:
    return _WS_RE.sub("", s)


def _resolve_rationale_chunks(qa: QAItem, chunks_by_doc: dict[str, list[Chunk]]) -> list[Chunk]:
    """rationale ごとに、引用を逐語で含むチャンクを選ぶ (evidence に委譲)。

    見つかった場合は rationale.chunk_id も埋める。これにより後段の
    answer-grounding (leave-one-out) が同一文書内の複数チャンクを別々の
    根拠単位として扱える (chunk_id 無しだと doc 単位に潰れて検査が素通りする)。
    """
    all_chunks = [c for lst in chunks_by_doc.values() for c in lst]
    matches = match_rationales(qa.rationale, all_chunks)
    out: list[Chunk] = []
    seen: set[str] = set()
    for m in matches:
        if not m.matched or m.chunk is None:
            continue
        r = qa.rationale[m.index]
        if not r.chunk_id:
            r.chunk_id = m.chunk.chunk_id
        if m.chunk.chunk_id not in seen:
            out.append(m.chunk)
            seen.add(m.chunk.chunk_id)
    return out


def annotate_relation_edges(
    qa: QAItem,
    chunks_by_doc: dict[str, list[Chunk]],
    policy: CrossDocPolicy,
) -> list[RelationEdge]:
    """QA の根拠チャンク間の参照エッジを列挙して qa.relation_edges に書き込む。"""
    anchors = _resolve_rationale_chunks(qa, chunks_by_doc)
    edges: list[RelationEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for a in anchors:
        for t in anchors:
            if a.chunk_id == t.chunk_id:
                continue
            same_doc = a.doc_id == t.doc_id
            if not same_doc and not policy.join_ok(a, t):
                continue  # 文書スコープ規則で正当化されない異文書対
            blob = t.text + " " + " ".join(t.section_path)
            for ref in a.references:
                # 異文書ではローカル番号 (第N条等) を述語にしない。
                # 別文書の同番号は別物なので、規格番号だけが文書を跨げる。
                if not same_doc and not _is_global_ref(ref):
                    continue
                if ref in blob:
                    key = (a.chunk_id, ref, t.chunk_id)
                    if key not in seen:
                        seen.add(key)
                        edges.append(RelationEdge(
                            subject_chunk_id=a.chunk_id,
                            predicate=ref,
                            object_chunk_id=t.chunk_id,
                        ))
    qa.relation_edges = edges
    return edges


def enrich_batch(
    in_path: Path,
    *,
    chunks_dir: Path,
    out_path: Optional[Path] = None,
    anchor_pairs_path: Optional[Path] = Path("data/seeds/anchor_pairs_manual_v1.jsonl"),
    doc_aliases_path: Optional[Path] = Path("data/seeds/doc_aliases.json"),
) -> Path:
    """JSONL 全 QA に relation_edges を付与する。out_path 省略時は in-place。"""
    chunks = load_chunks(chunks_dir)
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)
    for lst in chunks_by_doc.values():
        lst.sort(key=lambda c: c.position)
    policy = load_cross_doc_policy(anchor_pairs_path, doc_aliases_path)

    qas: list[QAItem] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qas.append(QAItem.model_validate_json(line))

    with_edges = 0
    total_edges = 0
    for qa in qas:
        edges = annotate_relation_edges(qa, chunks_by_doc, policy)
        if edges:
            with_edges += 1
            total_edges += len(edges)

    out = out_path or in_path
    with out.open("w", encoding="utf-8") as f:
        for qa in qas:
            f.write(qa.model_dump_json() + "\n")
    print(f"[enrich] relation_edges: {with_edges}/{len(qas)} 問にエッジ計 {total_edges} 本 → {out}")
    return out
