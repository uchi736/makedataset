"""Stage 3: Single-screen Streamlit review UI.

Usage:
    streamlit run src/rageval/review_app.py -- --in data/filtered/batch_*.jsonl

The `--in` argument accepts a concrete path or a glob (in which case the latest
match is used). Accepted/edited/rejected items are written to a mirror file in
`data/reviewed/<データセット>/<元ファイル名>` — 保存先の決定は
`webapi.service.review_mirror_path` 1本に集約してある (Web 版と同じ場所を読み書き
する)。平たい旧配置 `data/reviewed/<元ファイル名>` は読み込みのみ後方互換。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from rageval.aspects import ASPECT_DESCRIPTIONS, ASPECT_LABELS
from rageval.chunker import load_chunks
from rageval.schema import Chunk, QAItem

# 表示ラベル・合格線・チェックリストの正本は webapi/labels.py。同じ表を
# review_app / stats_app / labels.py に写していたため、Streamlit と Web で
# 指摘できる観点が食い違っていた (監査 rank 37/38)。ここでは引くだけにする。
from rageval.webapi.labels import (
    ANSWERABILITY_PASS,
    CATEGORY_LABELS as _CATEGORY_LABELS,
    DIAG_LABELS as _DIAG_LABELS,
    EVIDENCE_LABELS as _EVIDENCE_LABEL,
    GROUNDING_PASS,
    OUTPUT_TYPE_LABELS as _OUTPUT_TYPE_LABEL,
    checklist_for,
)
from rageval.webapi.service import (
    data_root_for_reviewed_dir,
    legacy_review_mirror_path,
    review_mirror_path,
)


def _check_pass(value, threshold: float) -> str:
    if value is None:
        return ":material/help: 未判定"
    return ":material/check_circle: 合格" if value >= threshold else ":material/cancel: 不合格"


def _render_checkpoints(qa: QAItem) -> None:
    """5項目の品質チェックリスト + 人手で見るべきポイント。"""
    fs = qa.filter_scores
    # 合格線は filter が実際に落とす値と同一 (labels 経由で filter.DEFAULT_* を引く)。
    # 画面が「合格」と描くのに filter は落とす、という食い違いを作らないため。
    rows = [
        ("答えられるか (answerability)", fs.answerability, ANSWERABILITY_PASS, "根拠だけで一意に答えられるか"),
        ("根拠妥当性 (grounding)",     fs.grounding,     GROUNDING_PASS, "根拠が回答を支えているか"),
    ]
    for label, val, thr, hint in rows:
        col1, col2 = st.columns([1, 3])
        col1.metric(label, f"{val:.1f}/5" if val is not None else "—", _check_pass(val, thr).split()[-1])
        col2.caption(hint)

    # leakage
    leak = fs.leakage
    leak_text = {"pass": ":material/check_circle: 合格", "fail": ":material/cancel: 不合格 (質問に答えが含まれる)"}.get(leak, ":material/help: 未判定")
    st.markdown(f"- **リーク判定 (leakage)**: {leak_text}")

    # difficulty_match
    dm = fs.difficulty_match
    dm_text = {
        "aligned":  ":material/check_circle: 一致",
        "too_easy": ":material/warning: 易しすぎる",
        "too_hard": ":material/warning: 難しすぎる",
        None:       ":material/help: 未判定",
    }.get(dm, ":material/help: 未判定")
    st.markdown(f"- **難易度整合 (difficulty_match)**: {dm_text} (宣言 vs 実態)")

    # uniqueness
    if fs.uniqueness is not None:
        st.markdown(f"- **独自性 (uniqueness)**: {fs.uniqueness:.2f} (1.0=完全独自, 0=既出と完全一致)")

    st.markdown(
        ":material/checklist: **人手で最終確認すること**:\n"
        "- 質問は曖昧でなく、業務上意味のある問いか\n"
        "- 回答は事実として正しく、過不足ないか\n"
        "- 根拠が質問・回答と整合しているか\n"
        "- 観点 (aspect) の意図に合った質問になっているか"
    )


def _summarize_diagnostic_tags(qa: QAItem) -> str:
    """Compact one-line summary of true diagnostic flags, returns markdown."""
    lines: list[str] = []
    pairs = [
        ("推論", qa.reasoning_complexity.model_dump(), _DIAG_LABELS["reasoning_complexity"]),
        ("検索", qa.retrieval_difficulty.model_dump(), _DIAG_LABELS["retrieval_difficulty"]),
        ("構造", qa.source_structure.model_dump(), _DIAG_LABELS["source_structure"]),
    ]
    for axis, dump, labels in pairs:
        tags = [labels[k] for k, v in dump.items() if k in labels and v is True]
        if tags:
            lines.append(f"- {axis}: " + " / ".join(f"`{t}`" for t in tags))
    ot = qa.reasoning_complexity.output_type
    if ot and ot != "none":
        lines.append(f"- 出力タイプ: `{_OUTPUT_TYPE_LABEL.get(ot, ot)}`")
    ev = qa.explainability.evidence_strictness
    lines.append(f"- 根拠厳密性: `{_EVIDENCE_LABEL.get(ev, ev)}`")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", dest="output", default="data/reviewed")
    parser.add_argument(
        "--data-root", dest="data_root", default=None,
        help="データ置き場 (省略時は --out の親から逆算。保存先を "
             "<保存先>/<データセット>/<元ファイル名> に掘るために使う)",
    )
    parser.add_argument("--chunks", dest="chunks_dir", default="data/chunks")
    parser.add_argument("--reviewer", default="unknown")
    parser.add_argument(
        "--track", default="all",
        choices=["all", "general", "kg_poc"],
        help="どのトラックの QA だけを表示するか (all=両方)",
    )
    # Streamlit passes extra args; strip them.
    known, _ = parser.parse_known_args()
    return known


# Reuse the lookup logic from filter.py for consistency
_CHUNK_SUFFIX_RE = __import__("re").compile(r"__c\d+$")


@st.cache_data(show_spinner=False)
def _load_chunk_index(chunks_dir: str) -> tuple[dict, dict, dict]:
    """Return (by_doc_page, by_chunk_id, by_doc) for anchor resolution.

    by_doc_page は (doc_id, page) → list[Chunk]。同一ページに複数チャンク
    (PDF は最大6個、.txt は page=None で文書全体) があるため、1個に潰すと
    引用を含む正しいチャンクを取り違える。
    """
    path = Path(chunks_dir)
    if not path.exists():
        return {}, {}, {}
    chunks = load_chunks(path)
    by_doc_page: dict[tuple, list[Chunk]] = {}
    for c in chunks:
        by_doc_page.setdefault((c.doc_id, c.page), []).append(c)
    by_chunk_id = {c.chunk_id: c for c in chunks}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    return by_doc_page, by_chunk_id, by_doc


def _resolve_anchor_chunks(qa: QAItem, chunks_dir: str) -> list[Chunk]:
    """For each rationale entry, return the matching Chunk (with full text).

    候補が複数あるときは **引用 (rationale.text) を逐語で含むチャンク** を選ぶ
    (filter.py の照合と同じ方針)。含むものが無ければ先頭候補。
    """
    import re as _re
    by_doc_page, by_chunk_id, by_doc = _load_chunk_index(chunks_dir)

    def _norm(s: str) -> str:
        return _re.sub(r"\s+", "", s)

    out: list[Chunk] = []
    seen: set[str] = set()
    for r in qa.rationale:
        norm_doc = _CHUNK_SUFFIX_RE.sub("", r.doc_id)
        candidates: list[Chunk] = (
            by_doc_page.get((r.doc_id, r.page))
            or by_doc_page.get((norm_doc, r.page))
            or ([by_chunk_id[r.doc_id]] if r.doc_id in by_chunk_id else [])
            or by_doc.get(norm_doc, [])
        )
        if not candidates:
            continue
        needle = _norm(r.text)
        best = next(
            (c for c in candidates if needle and needle in _norm(c.text)),
            candidates[0],
        )
        if best.chunk_id not in seen:
            out.append(best)
            seen.add(best.chunk_id)
    return out


def _resolve_review_paths(
    input_path: Path, out_dir: Path, data_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """(レビュー結果の保存先, 旧平置きの保存先) を返す。

    決めているのは webapi.service.review_mirror_path 1本。以前はここが
    「保存先/入力ファイル名」の平たい規約で、Web 側だけ入れ子規約へ移っていた
    ため、data/raw_x/batch.jsonl と data/filtered_x/batch.jsonl のような同名
    ファイルが保存先を奪い合い、片方を保存した瞬間にもう片方の人手判定が
    無警告で消えていた (監査 rank 3)。
    旧平置きは**読み込みの土台としてだけ**返す (既存の判定を失わないため)。
    """
    out_dir = Path(out_dir)
    root = Path(data_root) if data_root else data_root_for_reviewed_dir(out_dir)
    return (
        review_mirror_path(root, input_path, reviewed_dir=out_dir),
        legacy_review_mirror_path(root, input_path, reviewed_dir=out_dir),
    )


def _resolve_input(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        st.error(f"No files match: {pattern}")
        st.stop()
    return Path(matches[-1])


def _load_items(path: Path, track_filter: str = "all") -> list[QAItem]:
    """Load QAs from JSONL, optionally filtering by track.

    track_filter:
      - "all"     : all items
      - "general" : items WITHOUT kg_query_type
      - "kg_poc"  : items WITH kg_query_type
    """
    items: list[QAItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa = QAItem.model_validate_json(line)
            is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
            if track_filter == "general" and is_kg:
                continue
            if track_filter == "kg_poc" and not is_kg:
                continue
            items.append(qa)
    return items


def _filter_items_by_track(items: list[QAItem], track_filter: str) -> list[QAItem]:
    """Return a view of ``items`` without copying the QA objects."""
    if track_filter == "all":
        return list(items)
    want_kg = track_filter == "kg_poc"
    return [
        qa for qa in items
        if (qa.kg_query_type is not None or qa.kg_novelty is not None) == want_kg
    ]


def _merge_reviewed_items(
    source_items: list[QAItem], reviewed_items: list[QAItem],
) -> list[QAItem]:
    """Overlay prior review results while keeping the source file's full shape.

    Older versions could write only the currently visible track to the mirror.
    Treating the mirror as the entire dataset would therefore hide the other
    track forever.  The source file remains authoritative for membership/order;
    a matching reviewed item replaces only that source item.
    """
    reviewed_by_id = {qa.qa_id: qa for qa in reviewed_items}
    return [reviewed_by_id.get(qa.qa_id, qa) for qa in source_items]


def _atomic_write_lines(out_path: Path, lines: list[str]) -> None:
    """Atomically replace ``out_path`` with UTF-8 JSONL ``lines``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line.rstrip("\r\n") + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def _save_items(items: list[QAItem], out_path: Path) -> None:
    _atomic_write_lines(out_path, [qa.model_dump_json() for qa in items])


def _save_review_item(
    qa: QAItem, source_path: Path, out_path: Path,
    legacy_path: Optional[Path] = None,
) -> None:
    """Patch one QA into a complete mirror of its source JSONL.

    The source file defines membership and order. Existing review results are
    overlaid by qa_id, then ``qa`` is applied. This prevents track/category/file
    filters from dropping rows and also repairs mirrors produced by older
    partial-view saves. Unparseable legacy lines are preserved verbatim.

    保存先が新レイアウトへ移った直後は、まだ入れ子のミラーが存在せず、判定は
    平たい旧配置 (`legacy_path`) にしか無い。土台をそこから引かないと、画面で
    1問保存した瞬間に「その1問以外すべて未判定」のミラーが新規作成され、
    以後そちらが優先されて過去の人手判定が消える (監査 rank 3 の検収で発覚)。
    """
    source_path = source_path.resolve()
    out_path = out_path.resolve()

    # 土台の優先順: 新レイアウトのミラー → 平たい旧配置 → 元ファイル
    base_path = out_path
    if not out_path.exists() and legacy_path is not None:
        legacy_resolved = Path(legacy_path).resolve()
        if legacy_resolved.exists():
            base_path = legacy_resolved

    reviewed_by_id: dict[str, str] = {}
    if base_path.exists() and base_path != source_path:
        with base_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\r\n")
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                qa_id = obj.get("qa_id") if isinstance(obj, dict) else None
                if isinstance(qa_id, str):
                    reviewed_by_id[qa_id] = line

    updated_line = qa.model_dump_json()
    out_lines: list[str] = []
    found = False
    with source_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            try:
                obj = json.loads(line)
            except Exception:
                out_lines.append(line)
                continue
            qa_id = obj.get("qa_id") if isinstance(obj, dict) else None
            if qa_id == qa.qa_id and not found:
                out_lines.append(updated_line)
                found = True
            elif isinstance(qa_id, str) and qa_id in reviewed_by_id:
                out_lines.append(reviewed_by_id[qa_id])
            else:
                out_lines.append(line)

    if not found:
        raise ValueError(f"qa_id が元ファイルに見つかりません: {qa.qa_id} ({source_path})")
    _atomic_write_lines(out_path, out_lines)


def _status_counts(items: list[QAItem]) -> dict[str, int]:
    out = {"accepted": 0, "edited": 0, "rejected": 0, "pending": 0}
    for it in items:
        out[it.review_status] = out.get(it.review_status, 0) + 1
    return out


def _highlight_rationale_in_chunk(chunk_text: str, rationales: list) -> str:
    """Return HTML where rationale.text substrings are wrapped in <mark>.
    Whitespace-insensitive search (LLMが空白変えても拾う)."""
    import html
    import re as _re

    escaped = html.escape(chunk_text)
    for r in rationales:
        if not r.text or len(r.text) < 4:
            continue
        # Try exact first
        if r.text in chunk_text:
            esc_r = html.escape(r.text)
            escaped = escaped.replace(esc_r, f"<mark>{esc_r}</mark>", 1)
            continue
        # Try whitespace-stripped match
        needle_stripped = _re.sub(r"\s+", "", r.text)
        haystack_stripped = _re.sub(r"\s+", "", chunk_text)
        if needle_stripped and needle_stripped in haystack_stripped:
            # Find character span in original by walking
            stripped_idx = haystack_stripped.find(needle_stripped)
            # Walk original to find span
            orig_start = _find_orig_pos(chunk_text, stripped_idx)
            orig_end = _find_orig_pos(chunk_text, stripped_idx + len(needle_stripped))
            if 0 <= orig_start < orig_end <= len(chunk_text):
                before = html.escape(chunk_text[:orig_start])
                mid = html.escape(chunk_text[orig_start:orig_end])
                after = html.escape(chunk_text[orig_end:])
                escaped = f"{before}<mark>{mid}</mark>{after}"
    return escaped


def _find_orig_pos(text: str, stripped_target_pos: int) -> int:
    """Walk text and return the index in original at which the
    whitespace-stripped position equals stripped_target_pos."""
    import re as _re
    pos = 0
    for i, ch in enumerate(text):
        if pos == stripped_target_pos:
            return i
        if not _re.match(r"\s", ch):
            pos += 1
    return len(text)


def _checklist_groups(qa: QAItem) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Return [(group_name, [(key, label, description), ...]), ...].

    **減点式**: 問題があるときだけチェックを入れる。チェック0件 = 指摘なし =
    そのまま採用できる。General track と KG-PoC track で観点を変える。

    項目そのものは labels.CHECKLIST_* が正本。以前はここと Web 側に同じ表を
    手書きしており、誤結合 (別文書の同番号の同一視) を指摘する f_join が Web に
    しか無く、同じ QA でも見る画面で指摘できる観点が違っていた (監査 rank 37)。
    """
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
    return [
        (g["group"], [(it["key"], it["label"], it["desc"]) for it in g["items"]])
        for g in checklist_for(is_kg=is_kg)
    ]


_STATUS_DOT = {"pending": "⚪", "accepted": "🟢", "edited": "🔵", "rejected": "🔴"}


def _next_pending_index(view: list[QAItem], start: int) -> int | None:
    """view の start より後 (循環) で最初の未判定の位置を返す。無ければ None。"""
    n = len(view)
    for step in range(1, n + 1):
        i = (start + step) % n
        if view[i].review_status == "pending":
            return i
    return None


def render_review_panel(
    items: list[QAItem],
    out_path: Path,
    chunks_dir: str,
    reviewer: str,
    *,
    key_prefix: str = "rv",
    show_header: bool = True,
    track_label: str = "",
    input_label: str = "",
    save_items: list[QAItem] | None = None,
    on_save: Callable[[QAItem], None] | None = None,
) -> None:
    """Render the per-QA review panel (header + 左ナビ + body + action bar).

    Reusable from both standalone review_app and integrated stats dashboard.
    State is kept in `st.session_state[f'{key_prefix}_idx']`.
    On Accept/Edit/Reject, mutates the QAItem in place and writes JSONL to out_path.
    """
    total_all = len(items)
    if total_all == 0:
        st.warning("レビュー対象の QA がありません。データセットや絞り込みを確認してください。")
        return

    counts = _status_counts(items)
    done = total_all - counts["pending"]
    idx_key = f"{key_prefix}_idx"
    pending_key = f"{key_prefix}_pending_only"
    st.session_state.setdefault(idx_key, 0)

    # ---- 上部ストリップ: 進捗 + 集計 + 導線 (未判定のみ / 次の未判定へ) ----
    pending_only = st.session_state.get(pending_key, False)
    if show_header:
        c_prog, c_count, c_toggle, c_next = st.columns([3, 3, 2, 2])
        with c_prog:
            st.progress(
                done / total_all if total_all else 0,
                text=f"査読済み {done} / {total_all} 問 ({done / total_all * 100:.0f}%)",
            )
        with c_count:
            st.caption(
                f"🟢 採用 {counts['accepted']}　🔵 修正 {counts['edited']}　"
                f"🔴 却下 {counts['rejected']}　⚪ 未判定 {counts['pending']}"
            )
        with c_toggle:
            pending_only = st.checkbox("未判定のみ表示", key=pending_key)
        with c_next:
            if st.button(
                "次の未判定へ", width="stretch", type="primary",
                disabled=counts["pending"] == 0 or pending_only,
                key=f"{key_prefix}_nextpending",
                help="未判定の問いへ飛ぶ (未判定のみ表示中は不要)",
            ):
                nxt = _next_pending_index(items, st.session_state[idx_key])
                if nxt is not None:
                    st.session_state[idx_key] = nxt
                    st.rerun()

    view = [it for it in items if it.review_status == "pending"] if pending_only else items
    if not view:
        st.success(
            "全問レビュー済みです。data/reviewed/ のファイルがコミット対象になります。"
            "判定を見直す場合は『未判定のみ表示』を外してください。",
            icon=":material/check_circle:",
        )
        return

    idx = max(0, min(st.session_state[idx_key], len(view) - 1))
    st.session_state[idx_key] = idx
    qa = view[idx]

    # ---- 左: ナビ一覧 / 右: 本体 ----
    nav_col, body_col = st.columns([1, 3], gap="medium")
    with nav_col:
        st.caption((("未判定 " if pending_only else "全 ") + f"{len(view)} 問 — クリックで移動"))
        with st.container(height=560, border=False):
            for i, it in enumerate(view):
                dot = _STATUS_DOT.get(it.review_status, "⚪")
                label = f"{dot} {i + 1}. {it.question[:24]}"
                if st.button(
                    label, key=f"{key_prefix}_nav_{i}", width="stretch",
                    type="primary" if i == idx else "secondary",
                ):
                    st.session_state[idx_key] = i
                    st.rerun()
    with body_col:
        n1, n2, n3 = st.columns([1, 1, 4])
        with n1:
            if st.button("← 前へ", disabled=idx == 0, width="stretch", key=f"{key_prefix}_prev"):
                st.session_state[idx_key] = idx - 1
                st.rerun()
        with n2:
            if st.button("次へ →", disabled=idx >= len(view) - 1, width="stretch", key=f"{key_prefix}_next"):
                st.session_state[idx_key] = idx + 1
                st.rerun()
        with n3:
            st.caption(f"{idx + 1} / {len(view)} 問")
        _render_review_body(
            qa, save_items if save_items is not None else items, view, idx,
            out_path, chunks_dir, reviewer,
            key_prefix=key_prefix, pending_only=pending_only,
            on_save=on_save,
        )


def main() -> None:
    args = _parse_args()
    input_path = _resolve_input(args.input)
    out_path, legacy_path = _resolve_review_paths(
        input_path, Path(args.output), getattr(args, "data_root", None),
    )

    st.set_page_config(page_title="rageval review", layout="wide")

    track_filter = getattr(args, "track", "all")
    # When session was started with a different filter, reload.
    if (
        "qa_items_all" not in st.session_state
        or st.session_state.get("track_filter") != track_filter
    ):
        source_items = _load_items(input_path, track_filter="all")
        # 新規約のミラーを優先し、無ければ旧平置きを土台にする (書き出しは新規約)
        prior = next(
            (p for p in (out_path, legacy_path)
             if p.exists() and p.resolve() != input_path.resolve()),
            None,
        )
        if prior is not None:
            reviewed_items = _load_items(prior, track_filter="all")
            source_items = _merge_reviewed_items(source_items, reviewed_items)
        st.session_state.qa_items_all = source_items
        st.session_state.rv_idx = 0
        st.session_state.track_filter = track_filter

    all_items: list[QAItem] = st.session_state.qa_items_all
    items = _filter_items_by_track(all_items, track_filter)
    track_label_text = {
        "all":     "[全track]",
        "general": "[general (25観点)]",
        "kg_poc":  "[KG-PoC (3軸)]",
    }.get(track_filter, "")
    input_label_text = f"src <b>{input_path.name}</b> → out <b>{out_path}</b>"

    st.markdown(
        f"#### RAG評価レビュー <span style='font-size:11px;color:#6B6F76;'>{track_label_text}</span>",
        unsafe_allow_html=True,
    )
    render_review_panel(
        items=items,
        out_path=out_path,
        chunks_dir=args.chunks_dir,
        reviewer=args.reviewer,
        key_prefix="rv",
        show_header=True,
        track_label=track_label_text,
        input_label=input_label_text,
        save_items=all_items,
    )
    return


def _render_review_body(
    qa: QAItem,
    full_items: list[QAItem],
    view: list[QAItem],
    idx: int,
    out_path: Path,
    chunks_dir: str,
    reviewer: str,
    *,
    key_prefix: str = "rv",
    pending_only: bool = False,
    on_save: Callable[[QAItem], None] | None = None,
) -> None:
    """Per-QA body: identity strip + Q/A + evidence + checklist + actions.

    view はナビ対象の (絞り込み後の) リスト、full_items は保存対象の全件。
    qa は両方に共有される同一オブジェクトなので、qa を書き換えるだけで
    full_items にも反映される。State keys are namespaced with key_prefix so
    multiple instances (standalone review_app vs integrated stats tab) don't collide.
    """
    total = len(view)
    idx_key = f"{key_prefix}_idx"

    # 現在の判定バッジ + ID。主役は質問と回答なので識別情報は1行に抑える
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
    badge = {
        "pending":  "⚪ 未判定",
        "accepted": "🟢 採用",
        "edited":   "🟡 修正済",
        "rejected": "🔴 却下",
    }.get(qa.review_status, qa.review_status)
    st.markdown(f"**{badge}**　`{qa.qa_id}`　`track={'kg_poc' if is_kg else 'general'}`")

    edit_mode = st.session_state.get(f"{key_prefix}_edit_mode_{idx}", False)
    anchor_chunks = _resolve_anchor_chunks(qa, chunks_dir)

    # 機械照合 (決定論): 引用が原文と一字一句一致するか
    import re as _re
    stripped_all = _re.sub(r"\s+", "", "".join(c.text for c in anchor_chunks))
    all_match = True
    rationale_results = []
    for r in qa.rationale:
        needle = _re.sub(r"\s+", "", r.text)
        hit = bool(needle) and needle in stripped_all
        rationale_results.append((r, hit))
        if not hit:
            all_match = False

    # ============================================================
    # ① 読む — 質問と回答だけを大きく。メタ情報は控えめなチップに
    # ============================================================
    st.markdown("#### ① 質問")
    if edit_mode:
        new_q = st.text_area(
            "q", qa.question, height=90,
            key=f"{key_prefix}_q_{idx}", label_visibility="collapsed",
        )
    else:
        st.info(qa.question)
        new_q = qa.question

    st.markdown("#### 回答")
    if edit_mode:
        new_a = st.text_area(
            "a", qa.answer, height=120,
            key=f"{key_prefix}_a_{idx}", label_visibility="collapsed",
        )
    else:
        st.write(qa.answer)
        new_a = qa.answer

    chips: dict[str, str] = {}
    if is_kg:
        from rageval.tracks.kg_poc import (
            KG_NOVELTY_LABELS,
            KG_QUERY_TYPE_LABELS,
            LLM_KNOWLEDGE_LABELS,
        )
        qt, nov = qa.kg_query_type or "", qa.kg_novelty or ""
        chips["クエリ型"] = KG_QUERY_TYPE_LABELS.get(qt, qt or "(未指定)")
        chips["未知性"] = KG_NOVELTY_LABELS.get(nov, nov or "(未指定)")
        if qa.llm_knowledge:
            k_label = LLM_KNOWLEDGE_LABELS.get(qa.llm_knowledge, qa.llm_knowledge)
            chips["LLM既知性"] = f"{k_label} (RAG必須)" if qa.llm_knowledge == "unknown" else k_label
        else:
            chips["LLM既知性"] = "未付与"
    else:
        chips["分類"] = ", ".join(_CATEGORY_LABELS.get(c, c) for c in qa.category) or "(なし)"
        chips["観点"] = ", ".join(ASPECT_LABELS.get(a, a) for a in qa.aspect) or "(なし)"
    chips["検索"] = qa.retrieval_level
    chips["回答"] = qa.answer_level
    st.caption("　".join(f"`{k}: {v}`" for k, v in chips.items()))

    # ============================================================
    # ② 裏取り — 合格バッジ + 引用はエクスパンダ (不合格時のみ自動展開)
    # ============================================================
    st.markdown("#### ② 裏取り")
    if not anchor_chunks:
        st.warning(
            f"原文チャンクが見つかりません (参照先: {chunks_dir})。"
            "起動時の --chunks がこのQAのコーパスと合っているか確認してください "
            "(例: --chunks data/chunks_plant)"
        )
    elif all_match:
        st.success(
            "機械チェック合格：引用はすべて原文と一致。"
            "黄色の箇所が回答と合っているか目視確認だけでOK。",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            "機械チェック未合格：原文に見つからない引用があります。"
            "下の ✗ 印の引用を原文と突き合わせてください。",
            icon=":material/error:",
        )

    with st.expander("引用を表示して根拠を確認する", expanded=not all_match):
        st.markdown("**引用一覧** (✓=原文と一致 / ✗=原文に見つからない)")
        for r, hit in rationale_results:
            mark = ":green[✓]" if hit else ":red[✗]"
            st.markdown(f"- {mark} `{r.doc_id}` (p.{r.page}) — {r.text}")
        for ch in anchor_chunks:
            section = " > ".join(ch.section_path) if ch.section_path else "(見出しなし)"
            st.caption(f"**{ch.doc_id}** ・ p.{ch.page} ・ `{ch.chunk_id}` ・ {section}")
            highlighted = _highlight_rationale_in_chunk(ch.text, qa.rationale)
            st.markdown(
                f"<div style='font-size:12.5px;line-height:1.7;"
                f"max-height:240px;overflow:auto;white-space:pre-wrap;"
                f"padding:6px 8px;border-left:3px solid #45637A;background:#FAFAFA;'>"
                f"{highlighted}</div>",
                unsafe_allow_html=True,
            )

    # ============================================================
    # ③ 決める — 減点式: 問題があるときだけ指摘。指摘ゼロなら追加クリック不要
    # ============================================================
    st.markdown("#### ③ 判定")
    groups = _checklist_groups(qa)
    total_checks = sum(len(gitems) for _, gitems in groups)
    n_flags = sum(
        1 for _, gitems in groups for k, _, _ in gitems
        if st.session_state.get(f"{key_prefix}_{k}_{idx}", False)
    )
    if n_flags == 0:
        st.markdown(f"✅ **指摘なし（{total_checks}項目すべてクリア）** — このまま「採用」できます")
    else:
        st.markdown(f"⚠️ **{n_flags} 件の指摘あり** — 「修正」または「却下」を検討してください")

    with st.expander(f"問題を指摘する（{n_flags} 件選択中）", expanded=n_flags > 0):
        ck_cols = st.columns(len(groups))
        for col, (gname, gitems) in zip(ck_cols, groups):
            with col:
                st.markdown(f"**{gname}**")
                for key, label, desc in gitems:
                    st.checkbox(label, key=f"{key_prefix}_{key}_{idx}", help=desc)

    # ============================================================
    # 判定ボタン — ゲートなし (減点式なので指摘の有無に関わらず押せる)
    # ============================================================
    st.write("")
    now = datetime.now()

    def _mark(status: str, question: str, answer: str) -> None:
        qa.review_status = status  # type: ignore[assignment]
        qa.question = question
        qa.answer = answer
        qa.reviewed_by = reviewer
        qa.reviewed_at = now
        # qa は full_items 内の同一オブジェクト。書き換えは自動で反映される。
        if on_save is not None:
            on_save(qa)
        else:
            _save_items(full_items, out_path)

    def _advance() -> None:
        # 未判定のみ表示中は、判定した問いが view から抜けて次が繰り上がるので
        # idx は据え置き (clamp で末尾も吸収)。全件表示なら通常どおり1つ進む。
        if not pending_only and idx < total - 1:
            st.session_state[idx_key] = idx + 1

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🟢 採用", width="stretch", type="primary", key=f"{key_prefix}_accept",
                     help="評価セットに入れる (自動保存して次の問いへ)"):
            _mark("accepted", qa.question, qa.answer)
            _advance()
            st.rerun()
    with b2:
        if not edit_mode:
            if st.button("🟡 修正", width="stretch", key=f"{key_prefix}_edit",
                         help="質問・回答を書き直す (① が編集欄に変わる)"):
                st.session_state[f"{key_prefix}_edit_mode_{idx}"] = True
                st.rerun()
        else:
            if st.button("💾 修正を保存", width="stretch", key=f"{key_prefix}_save",
                         help="書き直した内容で確定して次の問いへ"):
                _mark("edited", new_q, new_a)
                st.session_state[f"{key_prefix}_edit_mode_{idx}"] = False
                _advance()
                st.rerun()
    with b3:
        if st.button("🔴 却下", width="stretch", key=f"{key_prefix}_reject",
                     help="評価セットに入れない (自動保存して次の問いへ)"):
            _mark("rejected", qa.question, qa.answer)
            _advance()
            st.rerun()

    # ============================================================
    # 詳細 (折りたたみ): 難易度根拠・タグの定義・判定LLMスコア・生JSON
    # ============================================================
    with st.expander("詳細（難易度根拠・タグの定義・判定LLMスコア・生JSON）"):
        if qa.difficulty_rationale:
            st.markdown("**難易度根拠**")
            st.write(qa.difficulty_rationale)
        if is_kg:
            from rageval.tracks.kg_poc import (
                KG_NOVELTY_DESCRIPTIONS,
                KG_QUERY_TYPE_DESCRIPTIONS,
            )
            if qa.kg_query_type:
                st.markdown(f"**クエリ型の定義**: {KG_QUERY_TYPE_DESCRIPTIONS.get(qa.kg_query_type, '')}")
            if qa.kg_novelty:
                st.markdown(f"**未知性の定義**: {KG_NOVELTY_DESCRIPTIONS.get(qa.kg_novelty, '')}")
        else:
            for a in qa.aspect:
                st.markdown(f"**観点「{ASPECT_LABELS.get(a, a)}」の定義**: {ASPECT_DESCRIPTIONS.get(a, '(定義なし)')}")
        diag = _summarize_diagnostic_tags(qa)
        if diag:
            st.markdown("**診断タグ**")
            st.markdown(diag)
        fs = qa.filter_scores
        st.markdown("**判定LLMスコア**")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("答えられるか", f"{fs.answerability:.1f}/5" if fs.answerability else "—")
        s2.metric("根拠妥当性", f"{fs.grounding:.1f}/5" if fs.grounding else "—")
        s3.metric("漏れ判定", str(fs.leakage) if fs.leakage else "—")
        s4.metric("独自性", f"{fs.uniqueness:.2f}" if fs.uniqueness else "—")
        s5.metric("逐語一致率", f"{fs.rationale_grounded:.0%}" if fs.rationale_grounded is not None else "—")
        st.markdown("**生JSON**")
        st.json(qa.model_dump(mode="json"))


if __name__ == "__main__":
    # Allow running via `python src/rageval/review_app.py` for quick tests.
    main()
