"""Web UI 用データ層。

Streamlit 版 review_app.py と同じ規約で動く:
- 読み込みは data/ 配下のディレクトリ・ファイル・glob
- data/reviewed/<元ファイル名> があればそちらを優先して読む (レビュー済みの続きから)
- レビュー保存は data/reviewed/<元ファイル名> へのミラー書き出し
- 根拠照合は「空白を無視した逐語一致」(filter.py / review_app.py と同じ規則)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rageval.chunker import load_chunks
from rageval.evidence import find_spans, match_rationales, matched_chunks
from rageval.schema import Chunk, QAItem

REVIEWED_DIR_NAME = "reviewed"

_CHUNK_SUFFIX_RE = re.compile(r"__c\d+$")
_WS_RE = re.compile(r"\s+")


class ServiceError(Exception):
    """API がそのまま 4xx にして返す業務エラー。"""


def _norm(s: str) -> str:
    return _WS_RE.sub("", s)


# ---------------- パス解決 ----------------

def _is_within(path: Path, root: Path) -> bool:
    """path が root の中にあるか。

    文字列の接頭辞比較 (str.startswith) だと、data に対して data_backup のような
    **兄弟ディレクトリが通ってしまう** (実測済み)。パス要素単位で判定する。
    """
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def resolve_paths(data_root: Path, rel: str) -> list[Path]:
    """data_root 配下の相対パス (dir / file / glob) を JSONL ファイル列に解決する。

    data_root の外を指すパスは拒否する。
    """
    rel = (rel or "").strip().replace("\\", "/")
    if not rel:
        raise ServiceError("path が空です")
    root = data_root.resolve()
    if "*" in rel or "?" in rel:
        matches = sorted(root.glob(rel))
        files = [p for p in matches if p.is_file() and p.suffix == ".jsonl"]
    else:
        p = (root / rel).resolve()
        if not _is_within(p, root):
            raise ServiceError(f"data ディレクトリの外は参照できません: {rel}")
        if p.is_dir():
            files = sorted(p.glob("*.jsonl"))
        elif p.is_file():
            files = [p]
        else:
            files = []
    files = [f for f in files if _is_within(f.resolve(), root)]
    if not files:
        raise ServiceError(f"JSONL が見つかりません: {rel}")
    return files


def list_sources(data_root: Path) -> dict[str, Any]:
    """データセット候補 (JSONL を含むディレクトリ) とチャンクディレクトリの一覧。"""
    root = data_root.resolve()
    datasets: list[dict[str, Any]] = []
    chunk_dirs: list[str] = []
    if not root.exists():
        return {"datasets": [], "chunk_dirs": []}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        jsonls = sorted(d.glob("*.jsonl"))
        if d.name.startswith("chunks"):
            if jsonls:
                chunk_dirs.append(d.name)
            continue
        if not jsonls:
            continue
        datasets.append({
            "name": d.name,
            "files": [f.name for f in jsonls],
        })
    return {"datasets": datasets, "chunk_dirs": chunk_dirs}


# ---------------- 読み込み ----------------

def _is_kg(item: dict[str, Any]) -> bool:
    return item.get("kg_query_type") is not None or item.get("kg_novelty") is not None


def review_mirror_path(data_root: Path, src: Path) -> Path:
    """レビュー結果の保存先 (元ファイルのデータセットごとに分ける)。

    以前は `data/reviewed/<ファイル名>` だけで決めていたため、別ディレクトリの
    同名ファイル (raw_x/batch_1.jsonl と filtered_x/batch_1.jsonl のような、
    このリポジトリでは標準的な命名) が同じミラーを共有し、一方のレビュー結果を
    もう一方が読み込む・上書きするという破壊が起きていた。
    データセット (data からの相対ディレクトリ) をミラー側にも掘る。
    """
    root = data_root.resolve()
    reviewed_dir = root / REVIEWED_DIR_NAME
    try:
        rel_dir = src.resolve().parent.relative_to(root)
    except ValueError:
        rel_dir = Path()
    if rel_dir == Path(REVIEWED_DIR_NAME) or str(rel_dir).startswith(REVIEWED_DIR_NAME + "/"):
        return src  # すでに reviewed/ 配下
    return reviewed_dir / rel_dir / src.name


def _legacy_mirror_path(data_root: Path, src: Path) -> Path:
    """旧レイアウト (data/reviewed/<ファイル名>)。読み込み時のみ後方互換で見る。"""
    return data_root.resolve() / REVIEWED_DIR_NAME / src.name


def load_items(
    data_root: Path, rel: str, track: str = "all",
) -> dict[str, Any]:
    """QA を読み込んで JSON 化する。

    - 各ファイルについてレビュー済みミラーがあればそちらを読む
      (新レイアウト data/reviewed/<データセット>/<名前> を優先し、
       無ければ旧レイアウト data/reviewed/<名前> にフォールバック)。
    - パースに失敗した行はスキップして件数だけ返す。
    """
    files = resolve_paths(data_root, rel)
    reviewed_dir = data_root.resolve() / REVIEWED_DIR_NAME
    items: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    reviewed_override: list[str] = []
    id_counts: dict[str, int] = {}
    for f in files:
        source = f
        if f.parent != reviewed_dir:
            mirror = review_mirror_path(data_root, f)
            legacy = _legacy_mirror_path(data_root, f)
            if mirror.exists():
                source = mirror
                reviewed_override.append(f.name)
            elif legacy.exists():
                source = legacy
                reviewed_override.append(f.name)
        with source.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    qa = QAItem.model_validate_json(line)
                except Exception:
                    skipped[f.name] = skipped.get(f.name, 0) + 1
                    continue
                is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
                if track == "general" and is_kg:
                    continue
                if track == "kg_poc" and not is_kg:
                    continue
                row = qa.model_dump(mode="json")
                row["_source"] = f.name
                items.append(row)
                id_counts[qa.qa_id] = id_counts.get(qa.qa_id, 0) + 1
    return {
        "items": items,
        "skipped": skipped,
        "reviewed_override": reviewed_override,
        "files": [f.name for f in files],
        # 同一バッチの派生ファイル (.enriched / .rag_verified 等) を丸ごと読むと
        # 同じ問いが重複する。UI が注意を出せるよう個数を返す。
        "duplicate_qa_ids": sum(1 for n in id_counts.values() if n > 1),
    }


# ---------------- チャンク索引と根拠照合 ----------------

_chunk_cache: dict[str, tuple[float, dict, dict, dict]] = {}


def _dir_signature(path: Path) -> float:
    sig = 0.0
    for f in path.glob("*.jsonl"):
        try:
            sig = max(sig, f.stat().st_mtime)
        except OSError:
            continue
    return sig


def _load_chunk_index(chunks_dir: Path) -> tuple[dict, dict, dict]:
    """(doc_id, page) → [Chunk], chunk_id → Chunk, doc_id → [Chunk] の3索引。"""
    key = str(chunks_dir.resolve())
    sig = _dir_signature(chunks_dir)
    cached = _chunk_cache.get(key)
    if cached and cached[0] == sig:
        return cached[1], cached[2], cached[3]
    if not chunks_dir.exists():
        empty: tuple[dict, dict, dict] = ({}, {}, {})
        _chunk_cache[key] = (sig, *empty)
        return empty
    chunks = load_chunks(chunks_dir)
    by_doc_page: dict[tuple, list[Chunk]] = {}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc_page.setdefault((c.doc_id, c.page), []).append(c)
        by_doc.setdefault(c.doc_id, []).append(c)
    by_chunk_id = {c.chunk_id: c for c in chunks}
    _chunk_cache[key] = (sig, by_doc_page, by_chunk_id, by_doc)
    return by_doc_page, by_chunk_id, by_doc


def _resolve_anchor_chunks(rationales: list[dict], chunks_dir: Path) -> list[Chunk]:
    """rationale ごとに、対応するチャンクを1つ選ぶ (review_app と同じ方針)。

    候補が複数あるときは引用 (text) を逐語で含むチャンクを優先する。
    """
    by_doc_page, by_chunk_id, by_doc = _load_chunk_index(chunks_dir)
    out: list[Chunk] = []
    seen: set[str] = set()
    for r in rationales:
        doc_id = r.get("doc_id") or ""
        page = r.get("page")
        chunk_id = r.get("chunk_id")
        norm_doc = _CHUNK_SUFFIX_RE.sub("", doc_id)
        candidates: list[Chunk] = (
            ([by_chunk_id[chunk_id]] if chunk_id and chunk_id in by_chunk_id else [])
            or by_doc_page.get((doc_id, page))
            or by_doc_page.get((norm_doc, page))
            or ([by_chunk_id[doc_id]] if doc_id in by_chunk_id else [])
            or by_doc.get(norm_doc, [])
        )
        if not candidates:
            continue
        needle = _norm(r.get("text") or "")
        best = next(
            (c for c in candidates if needle and needle in _norm(c.text)),
            candidates[0],
        )
        if best.chunk_id not in seen:
            out.append(best)
            seen.add(best.chunk_id)
    return out


def _stripped_to_orig_map(text: str) -> list[int]:
    """空白除去後の位置 k → 元テキストでの位置、の対応表。"""
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            mapping.append(i)
    return mapping


def _find_spans(chunk_text: str, rationales: list[dict]) -> list[list[int]]:
    """rationale.text が chunk_text 中に現れる範囲 [start, end) を列挙する。

    まず逐語一致、無ければ空白無視の一致で探す。重なりはマージ。
    """
    spans: list[list[int]] = []
    stripped = _norm(chunk_text)
    mapping = _stripped_to_orig_map(chunk_text)
    for r in rationales:
        needle_raw = r.get("text") or ""
        if len(needle_raw) < 4:
            continue
        idx = chunk_text.find(needle_raw)
        if idx >= 0:
            spans.append([idx, idx + len(needle_raw)])
            continue
        needle = _norm(needle_raw)
        if not needle:
            continue
        sidx = stripped.find(needle)
        if sidx < 0:
            continue
        start = mapping[sidx]
        end_stripped = sidx + len(needle) - 1
        end = mapping[end_stripped] + 1 if end_stripped < len(mapping) else len(chunk_text)
        spans.append([start, end])
    spans.sort()
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(list(s))
    return merged


def resolve_evidence(
    data_root: Path, chunks_rel: str, rationales: list[dict],
) -> dict[str, Any]:
    """根拠チャンクの解決 + 逐語一致チェック + ハイライト範囲の計算。

    chunks_rel が "auto" のときは data/ 配下の chunks* を順に照合し、
    引用が最も多く一致するディレクトリを自動で選ぶ。明示指定でも全滅した
    場合は他のディレクトリへフォールバックする (どれを使ったかは
    used_chunks_dir で返す)。
    """
    root = data_root.resolve()
    candidates: list[str] = []
    if chunks_rel and chunks_rel != "auto":
        candidates.append(chunks_rel)
    if root.exists():
        for d in sorted(root.iterdir()):
            if (
                d.is_dir()
                and d.name.startswith("chunks")
                and d.name not in candidates
                and any(d.glob("*.jsonl"))
            ):
                candidates.append(d.name)
    if not candidates:
        candidates.append(chunks_rel or "chunks")

    best: dict[str, Any] | None = None
    best_key = (-1, -1)
    for name in candidates:
        result = _resolve_evidence_in_dir(root, name, rationales)
        matched = sum(1 for r in result["rationales"] if r["matched"])
        key = (matched, len(result["chunks"]))
        if key > best_key:
            best, best_key = result, key
        if result["all_match"]:
            best = result
            break
    assert best is not None
    return best


def _resolve_evidence_in_dir(
    root: Path, chunks_rel: str, rationales: list[dict],
) -> dict[str, Any]:
    chunks_dir = (root / chunks_rel).resolve()
    if not _is_within(chunks_dir, root):
        raise ServiceError(f"data ディレクトリの外は参照できません: {chunks_rel}")

    # 照合は evidence.match_rationales に一本化する。引用ごとに、その引用が
    # 名乗る文書の中だけで探すので、(a) doc A の引用が doc B の本文にあるだけで
    # 合格表示になる、(b) チャンクを連結した境界での偶然一致、の2つを防ぐ
    # (どちらも実データで再現していた)。
    _, by_chunk_id, _ = _load_chunk_index(chunks_dir)
    all_chunks = list(by_chunk_id.values())
    matches = match_rationales(rationales, all_chunks)
    anchors = matched_chunks(matches)

    rationale_results = []
    all_match = bool(matches)
    for m in matches:
        if not m.matched:
            all_match = False
        rationale_results.append({
            "doc_id": m.doc_id,
            "page": rationales[m.index].get("page") if m.index < len(rationales) else None,
            "text": m.text,
            "matched": m.matched,
        })

    # ハイライトは「そのチャンクに一致した引用」だけを当てる (他文書の引用で
    # 黄色が付く誤表示を防ぐ)。
    texts_by_chunk: dict[str, list[str]] = {}
    for m in matches:
        if m.matched and m.chunk is not None:
            texts_by_chunk.setdefault(m.chunk.chunk_id, []).append(m.text)

    chunk_payload = []
    for c in anchors:
        chunk_payload.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "page": c.page,
            "section_path": c.section_path,
            "text": c.text,
            "spans": find_spans(c.text, texts_by_chunk.get(c.chunk_id, [])),
        })
    return {
        "chunks": chunk_payload,
        "rationales": rationale_results,
        "all_match": all_match,
        "chunks_found": bool(anchors),
        "used_chunks_dir": chunks_rel,
    }


# ---------------- レビュー保存 ----------------

def save_review(
    data_root: Path,
    rel: str,
    source_name: str,
    qa_id: str,
    status: str,
    reviewer: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> dict[str, Any]:
    """1件のレビュー判定を data/reviewed/<データセット>/<元ファイル名> に書き出す。

    - 既にミラーがあればそれを土台に、無ければ元ファイルを土台にする
      (旧レイアウト data/reviewed/<元ファイル名> があればそれも土台候補にする)
    - 対象行だけを書き換え、他の行 (旧スキーマ行も含む) はそのまま写す
    """
    if status not in ("accepted", "edited", "rejected", "pending"):
        raise ServiceError(f"不正な status: {status}")
    files = resolve_paths(data_root, rel)
    target_file = next((f for f in files if f.name == source_name), None)
    if target_file is None:
        raise ServiceError(f"対象ファイルが見つかりません: {source_name}")

    reviewed_dir = data_root.resolve() / REVIEWED_DIR_NAME
    mirror = review_mirror_path(data_root, target_file)
    legacy = _legacy_mirror_path(data_root, target_file)
    if target_file.parent == reviewed_dir:
        base = target_file
    elif mirror.exists():
        base = mirror
    elif legacy.exists():
        base = legacy      # 旧レイアウトの続きから (書き出しは新レイアウトへ)
    else:
        base = target_file

    out_lines: list[str] = []
    updated: Optional[dict[str, Any]] = None
    with base.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                out_lines.append(raw)
                continue
            if obj.get("qa_id") != qa_id or updated is not None:
                out_lines.append(raw)
                continue
            qa = QAItem.model_validate_json(raw)
            qa.review_status = status  # type: ignore[assignment]
            if question is not None:
                qa.question = question
            if answer is not None:
                qa.answer = answer
            qa.reviewed_by = reviewer
            qa.reviewed_at = datetime.now()
            updated = qa.model_dump(mode="json")
            out_lines.append(qa.model_dump_json())
    if updated is None:
        raise ServiceError(f"qa_id が見つかりません: {qa_id} (in {base.name})")

    # 書き込み先は常にミラー (元ファイルは書き換えない)。
    # ただし reviewed/ 内のファイルを直接開いている場合はそのまま上書き。
    dest = target_file if target_file.parent == reviewed_dir else mirror
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for ln in out_lines:
            fh.write(ln + "\n")
    updated["_source"] = target_file.name
    return updated
