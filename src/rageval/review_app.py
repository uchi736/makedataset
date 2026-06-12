"""Stage 3: Single-screen Streamlit review UI.

Usage:
    streamlit run src/rageval/review_app.py -- --in data/filtered/batch_*.jsonl

The `--in` argument accepts a concrete path or a glob (in which case the latest
match is used). Accepted/edited/rejected items are written to a mirror file in
`data/reviewed/`.
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from rageval.aspects import ASPECT_DESCRIPTIONS, ASPECT_LABELS
from rageval.chunker import load_chunks
from rageval.schema import Chunk, QAItem


# Human-readable labels for diagnostic flags (true value only is shown)
_DIAG_LABELS: dict[str, dict[str, str]] = {
    "reasoning_complexity": {
        "multi_step":   "多段推論",
        "quantitative": "数値計算",
        "negation":     "否定推論",
        "cause_effect": "因果推論",
        "comparison":   "比較",
        "temporal":     "時系列",
    },
    "retrieval_difficulty": {
        "multi_doc":               "複数文書",
        "multi_chunk":             "複数チャンク",
        "low_locality":            "局所性低",
        "remote_reference":        "遠隔参照",
        "doc_volume_large":        "大量文書",
        "chunk_size_large":        "大チャンク",
        "abstraction_discrepancy": "抽象度の乖離",
        "vocabulary_mismatch":     "語彙ミスマッチ",
    },
    "source_structure": {
        "tables_charts":     "表・図",
        "complex_layout":    "複雑レイアウト",
        "specific_area_ref": "領域参照",
        "logical_nesting":   "階層構造",
        "large_enumeration": "大量列挙",
        "redundancy":        "冗長性",
    },
}
_OUTPUT_TYPE_LABEL = {"summary": "要約", "trans": "翻訳", "list": "列挙", "none": "なし"}
_EVIDENCE_LABEL = {
    "no-evidence": "根拠なし",
    "hier-ref":    "単一根拠",
    "coord-ref":   "並列根拠",
    "multi-ref":   "複数根拠厳密",
}


def _check_pass(value, threshold: float) -> str:
    if value is None:
        return ":material/help: 未判定"
    return ":material/check_circle: 合格" if value >= threshold else ":material/cancel: 不合格"


def _render_checkpoints(qa: QAItem) -> None:
    """5項目の品質チェックリスト + 人手で見るべきポイント。"""
    fs = qa.filter_scores
    rows = [
        ("答えられるか (answerability)", fs.answerability, 4.0, "根拠だけで一意に答えられるか"),
        ("根拠妥当性 (grounding)",     fs.grounding,     4.0, "根拠が回答を支えているか"),
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


_CATEGORY_LABELS = {
    "Integration": "統合",
    "Reasoning":   "推論",
    "Logic":       "論理",
    "Figure":      "図表",
    "Abstention":  "棄権",
}


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


def _save_items(items: list[QAItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qa in items:
            f.write(qa.model_dump_json() + "\n")


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
    """
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None

    if is_kg:
        return [
            ("根拠の検証", [
                ("f_inf", "回答が引用範囲を超えて推測している", "引用に無い事実・数値・結論を回答が含む"),
                ("f_mis", "引用が原文と一致していない", "✗印の引用がある / 黄色の箇所と食い違う"),
                ("f_irr", "根拠チャンクが質問と無関係", "別の話題のチャンクを根拠にしている"),
            ]),
            ("KG適性の検証", [
                ("f_single", "単一チャンク／キーワード一致で解けてしまう",
                 "ベクター検索だけで届く問い (基準点用の single_fact なら指摘不要)"),
                ("f_norel", "エンティティ間の関係をたどる必要がない",
                 "規定→例外→条件のような辿りが無い"),
                ("f_noagg", "集約・多段推論の要素がない",
                 "クエリ型タグ (集約/マルチホップ等) に実態が伴わない"),
            ]),
            ("品質の検証", [
                ("f_amb", "質問が一意に解釈できない", "複数の読み方ができる / 対象が曖昧"),
                ("f_ans", "回答に過不足がある", "誤り・抜け・余計な付け足し"),
                ("f_diff", "難易度根拠が妥当でない", "難易度タグや説明が実態と合わない"),
            ]),
        ]

    # general track
    return [
        ("根拠の検証", [
            ("f_inf", "回答が引用範囲を超えて推測している", "引用に無い事実・数値・結論を回答が含む"),
            ("f_mis", "引用が原文と一致していない", "✗印の引用がある / 黄色の箇所と食い違う"),
            ("f_irr", "根拠チャンクが質問と無関係", "別の話題のチャンクを根拠にしている"),
        ]),
        ("分類の検証", [
            ("f_asp", "観点・分類タグの意図に合っていない", "タグ通りの問いの構造になっていない"),
        ]),
        ("品質の検証", [
            ("f_amb", "質問が一意に解釈できない", "複数の読み方ができる / 対象が曖昧"),
            ("f_ans", "回答に過不足がある", "誤り・抜け・余計な付け足し"),
            ("f_leak", "質問に答えそのものが書かれている", "読むだけで解けてしまう (漏れ)"),
            ("f_diff", "難易度タグが妥当でない", "検索/回答難易度が実態と合わない"),
        ]),
    ]


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
) -> None:
    """Render the per-QA review panel (header + body + action bar).

    Reusable from both standalone review_app and integrated stats dashboard.
    State is kept in `st.session_state[f'{key_prefix}_idx']`.
    On Accept/Edit/Reject, mutates items[idx] and writes JSONL to out_path.
    """
    total = len(items)
    if total == 0:
        st.warning("No items to review.")
        return

    idx_key = f"{key_prefix}_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    idx = st.session_state[idx_key]
    idx = max(0, min(idx, total - 1))
    qa = items[idx]

    # Header: 進捗バー + 判定の集計 + ナビ
    if show_header:
        counts = _status_counts(items)
        done = total - counts["pending"]
        st.progress(done / total if total else 0, text=f"査読済み {done} / {total} 問")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("採用", counts["accepted"])
        m2.metric("修正", counts["edited"])
        m3.metric("却下", counts["rejected"])
        m4.metric("未判定", counts["pending"])
        nav1, nav2, nav3 = st.columns([1, 1, 4])
        with nav1:
            if st.button("← 前へ", disabled=idx == 0, width="stretch", key=f"{key_prefix}_prev"):
                st.session_state[idx_key] = idx - 1
                st.rerun()
        with nav2:
            if st.button("次へ →", disabled=idx >= total - 1, width="stretch", key=f"{key_prefix}_next"):
                st.session_state[idx_key] = idx + 1
                st.rerun()
        with nav3:
            st.caption(f"{idx + 1} 問目 / 全 {total} 問")
        st.divider()

    _render_review_body(qa, items, idx, out_path, chunks_dir, reviewer, key_prefix=key_prefix)


def main() -> None:
    args = _parse_args()
    input_path = _resolve_input(args.input)
    out_path = Path(args.output) / input_path.name

    st.set_page_config(page_title="rageval review", layout="wide")

    track_filter = getattr(args, "track", "all")
    # When session was started with a different filter, reload.
    if (
        "qa_items" not in st.session_state
        or st.session_state.get("track_filter") != track_filter
    ):
        if out_path.exists():
            st.session_state.qa_items = _load_items(out_path, track_filter=track_filter)
        else:
            st.session_state.qa_items = _load_items(input_path, track_filter=track_filter)
        st.session_state.rv_idx = 0
        st.session_state.track_filter = track_filter

    items: list[QAItem] = st.session_state.qa_items
    track_label_text = {
        "all":     "[全track]",
        "general": "[general (25観点)]",
        "kg_poc":  "[KG-PoC (3軸)]",
    }.get(track_filter, "")
    input_label_text = f"src <b>{input_path.name}</b> → out <b>{out_path.name}</b>"

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
    )
    return


def _render_review_body(
    qa: QAItem,
    items: list[QAItem],
    idx: int,
    out_path: Path,
    chunks_dir: str,
    reviewer: str,
    *,
    key_prefix: str = "rv",
) -> None:
    """Per-QA body: identity strip + Q/A + evidence + checklist + actions.

    State keys are namespaced with key_prefix so multiple instances
    (standalone review_app vs integrated stats tab) don't collide.
    """
    total = len(items)
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
        items[idx] = qa
        _save_items(items, out_path)

    def _advance() -> None:
        if idx < total - 1:
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
