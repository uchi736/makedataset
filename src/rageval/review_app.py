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

from rageval.aspects import ASPECT_DESCRIPTIONS, ASPECT_EXAMPLES, ASPECT_LABELS
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


_DIFF_BADGE_COLOR = {"Easy": "green", "Medium": "orange", "Hard": "red"}
_CATEGORY_LABELS = {
    "Integration": "統合",
    "Reasoning":   "推論",
    "Logic":       "論理",
    "Figure":      "図表",
    "Abstention":  "棄権",
}


def _render_framing_banner(qa: QAItem) -> None:
    """Big top banner: what aspect/category/difficulty this QA tests.

    KG-PoC track の QA (`kg_query_type` がセットされてる) は、
    観点・カテゴリではなく KG用 3軸 (クエリ型 / 未知性 / LLM既知性) を表示する。
    """
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None

    if is_kg:
        _render_framing_banner_kg(qa)
    else:
        _render_framing_banner_general(qa)

    # 難易度根拠 + 診断軸タグ (両 track 共通)
    diag = _summarize_diagnostic_tags(qa)
    info_parts = []
    if qa.difficulty_rationale:
        info_parts.append(f":material/info: **難易度根拠**: {qa.difficulty_rationale}")
    if diag:
        info_parts.append(f":material/insights: **期待される推論/検索/構造**:\n{diag}")
    if info_parts:
        st.markdown("  \n".join(info_parts))


def _render_framing_banner_general(qa: QAItem) -> None:
    """General track (25観点)向けのフレーミング."""
    aspects_display = " / ".join(f"**{ASPECT_LABELS.get(a, a)}**" for a in qa.aspect)
    cat_display = " · ".join(f"{_CATEGORY_LABELS.get(c, c)} ({c})" for c in qa.category)

    st.markdown("##### このQAは何を測るために作られたのか  `[general track]`")
    cols = st.columns([2, 1, 1, 1])
    cols[0].markdown(f"**観点**\n\n{aspects_display or '(指定なし)'}")
    cols[1].markdown(f"**カテゴリ**\n\n{cat_display or '(指定なし)'}")
    cols[2].markdown(
        f"**検索難易度**\n\n:{_DIFF_BADGE_COLOR.get(qa.retrieval_level, 'gray')}[{qa.retrieval_level}]"
    )
    cols[3].markdown(
        f"**回答難易度**\n\n:{_DIFF_BADGE_COLOR.get(qa.answer_level, 'gray')}[{qa.answer_level}]"
    )
    st.caption(f"qa_id: `{qa.qa_id}`")

    for a in qa.aspect:
        label = ASPECT_LABELS.get(a, a)
        desc = ASPECT_DESCRIPTIONS.get(a, "(定義なし)")
        ex = ASPECT_EXAMPLES.get(a, "")
        st.info(
            f"**観点「{label}」 (`{a}`) を満たす質問になっているか?**  \n"
            f":material/lightbulb: **定義**: {desc}  \n"
            f":material/business_center: **現場例**: {ex}",
            icon=":material/target:",
        )


def _render_framing_banner_kg(qa: QAItem) -> None:
    """KG-PoC track 向けのフレーミング (3軸タグ表示)."""
    from rageval.tracks.kg_poc import (
        KG_NOVELTY_DESCRIPTIONS,
        KG_NOVELTY_LABELS,
        KG_QUERY_TYPE_DESCRIPTIONS,
        KG_QUERY_TYPE_LABELS,
        LLM_KNOWLEDGE_LABELS,
    )

    qt = qa.kg_query_type or ""
    nov = qa.kg_novelty or ""
    qt_label = KG_QUERY_TYPE_LABELS.get(qt, qt or "(未指定)")
    nov_label = KG_NOVELTY_LABELS.get(nov, nov or "(未指定)")
    llm_k = qa.llm_knowledge
    llm_k_label = LLM_KNOWLEDGE_LABELS.get(llm_k, "(未プロービング)") if llm_k else "(未プロービング)"

    st.markdown(f"##### このQAは何を測るために作られたのか  `[kg_poc track]`")
    cols = st.columns([2, 2, 1, 1, 1])
    cols[0].markdown(f"**クエリ型**\n\n**{qt_label}** (`{qt}`)")
    cols[1].markdown(f"**未知性**\n\n**{nov_label}** (`{nov}`)")
    cols[2].markdown(
        f"**LLM既知性**\n\n{'**:red[' + llm_k_label + ']**' if llm_k == 'unknown' else llm_k_label}"
    )
    cols[3].markdown(
        f"**検索難易度**\n\n:{_DIFF_BADGE_COLOR.get(qa.retrieval_level, 'gray')}[{qa.retrieval_level}]"
    )
    cols[4].markdown(
        f"**回答難易度**\n\n:{_DIFF_BADGE_COLOR.get(qa.answer_level, 'gray')}[{qa.answer_level}]"
    )
    st.caption(f"qa_id: `{qa.qa_id}`")

    if qt:
        st.info(
            f"**クエリ型「{qt_label}」 (`{qt}`) の意図に合った質問か?**  \n"
            f":material/lightbulb: {KG_QUERY_TYPE_DESCRIPTIONS.get(qt, '')}",
            icon=":material/help:",
        )
    if nov:
        st.info(
            f"**未知性「{nov_label}」 (`{nov}`) を満たすか?**  \n"
            f":material/lightbulb: {KG_NOVELTY_DESCRIPTIONS.get(nov, '')}",
            icon=":material/key:",
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


def _render_diagnostic_tags(qa: QAItem) -> None:
    """4軸の bool タグのうち true のものだけバッジ表示。"""
    blocks = [
        ("Reasoning", qa.reasoning_complexity.model_dump(), _DIAG_LABELS["reasoning_complexity"]),
        ("Retrieval", qa.retrieval_difficulty.model_dump(), _DIAG_LABELS["retrieval_difficulty"]),
        ("Structure", qa.source_structure.model_dump(), _DIAG_LABELS["source_structure"]),
    ]
    any_tag = False
    for axis_name, dump, labels in blocks:
        tags = [labels[k] for k, v in dump.items() if k in labels and v is True]
        if not tags:
            continue
        any_tag = True
        st.markdown(f"- **{axis_name}**: " + " / ".join(f"`{t}`" for t in tags))
    # output_type は文字列(noneでなければ表示)
    ot = qa.reasoning_complexity.output_type
    if ot and ot != "none":
        st.markdown(f"- **出力タイプ**: `{_OUTPUT_TYPE_LABEL.get(ot, ot)}`")
        any_tag = True
    # explainability
    ev = qa.explainability.evidence_strictness
    st.markdown(f"- **根拠厳密性**: `{_EVIDENCE_LABEL.get(ev, ev)}`")
    if not any_tag:
        st.caption("(Reasoning/Retrieval/Structure 軸に該当タグなし — 単純な抽出型QAの可能性)")


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
    """Return (by_doc_page, by_chunk_id, by_doc) for anchor resolution."""
    path = Path(chunks_dir)
    if not path.exists():
        return {}, {}, {}
    chunks = load_chunks(path)
    by_doc_page = {(c.doc_id, c.page): c for c in chunks}
    by_chunk_id = {c.chunk_id: c for c in chunks}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    return by_doc_page, by_chunk_id, by_doc


def _resolve_anchor_chunks(qa: QAItem, chunks_dir: str) -> list[Chunk]:
    """For each rationale entry, return the matching Chunk (with full text)."""
    by_doc_page, by_chunk_id, by_doc = _load_chunk_index(chunks_dir)
    out: list[Chunk] = []
    seen: set[str] = set()
    for r in qa.rationale:
        # (doc_id, page) exact
        c = by_doc_page.get((r.doc_id, r.page))
        if c and c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)
            continue
        # normalize doc_id
        norm = _CHUNK_SUFFIX_RE.sub("", r.doc_id)
        c = by_doc_page.get((norm, r.page))
        if c and c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)
            continue
        # rationale.doc_id IS a chunk_id
        c = by_chunk_id.get(r.doc_id)
        if c and c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)
            continue
        # any chunk in normalized doc
        if norm in by_doc and by_doc[norm]:
            c = by_doc[norm][0]
            if c.chunk_id not in seen:
                out.append(c)
                seen.add(c.chunk_id)
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


def _render_identity_strip(qa: QAItem) -> None:
    """Identify the QA via native Streamlit: caption with key-value pairs.

    HTML chip 装飾はやめて、Streamlit の :color[text] と st.caption だけで
    ダッシュボードのタブ感に揃える。
    """
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
    diff_color = {"Easy": "green", "Medium": "orange", "Hard": "red"}
    status_marker = {
        "pending":  ":gray[● pending]",
        "accepted": ":green[✓ accepted]",
        "edited":   ":orange[✎ edited]",
        "rejected": ":red[✕ rejected]",
    }.get(qa.review_status, qa.review_status)

    parts: list[str] = [
        f"`{qa.qa_id}`",
        f"track=`{'kg_poc' if is_kg else 'general'}`",
    ]
    if is_kg:
        from rageval.tracks.kg_poc import (
            KG_NOVELTY_LABELS,
            KG_QUERY_TYPE_LABELS,
            LLM_KNOWLEDGE_LABELS,
        )
        qt, nov = qa.kg_query_type or "", qa.kg_novelty or ""
        parts.append(f"クエリ型: **{KG_QUERY_TYPE_LABELS.get(qt, qt)}**")
        parts.append(f"未知性: :red[**{KG_NOVELTY_LABELS.get(nov, nov)}**]")
        if qa.llm_knowledge:
            llm_label = LLM_KNOWLEDGE_LABELS.get(qa.llm_knowledge, qa.llm_knowledge)
            if qa.llm_knowledge == "unknown":
                parts.append(f"LLM既知性: :red[**{llm_label}**]")
            else:
                parts.append(f"LLM既知性: {llm_label}")
        else:
            parts.append("LLM既知性: :gray[未プロービング]")
    else:
        cat = ", ".join(qa.category) if qa.category else "(なし)"
        asp = ", ".join(qa.aspect) if qa.aspect else "(なし)"
        parts.append(f"カテゴリ: **{cat}**")
        parts.append(f"観点: **{asp}**")

    parts.append(f"検索: :{diff_color.get(qa.retrieval_level, 'gray')}[{qa.retrieval_level}]")
    parts.append(f"回答: :{diff_color.get(qa.answer_level, 'gray')}[{qa.answer_level}]")
    parts.append(f"status: {status_marker}")

    st.caption(" ・ ".join(parts))


def _checklist_groups(qa: QAItem) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Return [(group_name, [(key, label, description), ...]), ...].

    General track と KG-PoC track で観点を変える。
    """
    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None

    if is_kg:
        # KG-PoC専用チェックリスト: グラフ性 / 関係性 / RAG-差別性 を問う
        return [
            ("根拠の検証", [
                ("c_q",  "質問が元チャンクから一意に導ける", "チャンクを読めば質問が成立"),
                ("c_a",  "回答が元チャンクに事実根拠を持つ", "推測・外部知識に依存しない"),
                ("c_r",  "根拠が元チャンクの部分文字列として現れる", "黄色ハイライト確認"),
            ]),
            ("KG適性の検証", [
                ("c_qt", "クエリ型タグが質問の構造と一致", "single/multi_hop/aggregation等が正しい"),
                ("c_kg", "グラフ的アクセスが必要 (RAGの素朴検索では届かない)",
                 "ベクター検索だけでは弱い問いになっている"),
                ("c_ent","エンティティ・関係が文書中の実体である",
                 "造語ではなく文書に登場する具体的な対象"),
                ("c_nov","未知性タグが実態と合っている",
                 "unknown_term/relation/procedural の定義通りか"),
            ]),
            ("品質の検証", [
                ("c_l",  "回答リーク・将来情報の混入なし", "質問に答え本体が含まれない"),
                ("c_d",  "難易度タグ(検索/回答)が妥当", "根拠数や推論ステップと整合"),
            ]),
        ]

    # general track
    return [
        ("根拠の検証", [
            ("c_q",  "質問が元チャンクから一意に導ける", "チャンクを読めば質問が成立"),
            ("c_a",  "回答が元チャンクに事実根拠を持つ", "推測・外部知識に依存しない"),
            ("c_r",  "根拠が元チャンクの部分文字列として現れる", "黄色ハイライト確認"),
        ]),
        ("分類の検証", [
            ("c_p",  "観点/タグの意図に合致している", "タグ通りの問いの構造になっている"),
        ]),
        ("品質の検証", [
            ("c_l",  "回答リーク・将来情報の混入なし", "質問に答え本体が含まれない"),
            ("c_d",  "難易度タグ(検索/回答)が妥当", "根拠数や推論ステップと整合"),
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

    # Header (native): nav buttons + progress + status counts
    if show_header:
        counts = _status_counts(items)
        h1, h2, h3 = st.columns([1, 4, 1])
        with h1:
            if st.button("← prev", disabled=idx == 0, width="stretch", key=f"{key_prefix}_prev"):
                st.session_state[idx_key] = idx - 1
                st.rerun()
        with h2:
            st.progress(
                (idx + 1) / total,
                text=(
                    f"{idx + 1} / {total}    "
                    f":green[✓ {counts['accepted']}]  "
                    f":orange[✎ {counts['edited']}]  "
                    f":red[✕ {counts['rejected']}]  "
                    f":gray[● {counts['pending']}]"
                ),
            )
        with h3:
            if st.button("next →", disabled=idx >= total - 1, width="stretch", key=f"{key_prefix}_next"):
                st.session_state[idx_key] = idx + 1
                st.rerun()

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

    _render_identity_strip(qa)

    # ============================================================
    # 2-column main: Q/A on left, evidence chunks on right
    # ============================================================
    edit_mode = st.session_state.get(f"{key_prefix}_edit_mode_{idx}", False)
    anchor_chunks = _resolve_anchor_chunks(qa, chunks_dir)

    # Compute rationale match (deterministic)
    import re as _re
    all_chunks_text = "".join(c.text for c in anchor_chunks)
    stripped_all = _re.sub(r"\s+", "", all_chunks_text)
    all_match = True
    rationale_results = []
    for r in qa.rationale:
        needle = _re.sub(r"\s+", "", r.text)
        hit = bool(needle) and needle in stripped_all
        rationale_results.append((r, hit))
        if not hit:
            all_match = False

    col_qa, col_evi = st.columns(2)

    with col_qa:
        st.markdown("**質問**")
        if edit_mode:
            new_q = st.text_area(
                "q", qa.question, height=90,
                key=f"{key_prefix}_q_{idx}", label_visibility="collapsed",
            )
        else:
            st.info(qa.question)
            new_q = qa.question

        st.markdown("**回答**")
        if edit_mode:
            new_a = st.text_area(
                "a", qa.answer, height=110,
                key=f"{key_prefix}_a_{idx}", label_visibility="collapsed",
            )
        else:
            st.markdown("> " + qa.answer.replace("\n", "\n> "))
            new_a = qa.answer

        if qa.difficulty_rationale:
            st.caption(f"難易度根拠: `{qa.difficulty_rationale}`")

    with col_evi:
        st.markdown("**元チャンク** (根拠を黄色でハイライト)")
        if all_match:
            st.success("✓ 根拠はすべて元チャンクに含まれる", icon=":material/check_circle:")
        else:
            st.error("✗ 根拠の不一致あり — 要確認", icon=":material/error:")

        if not anchor_chunks:
            st.warning(f"chunks に該当なし (chunks_dir={chunks_dir})")

        for ch in anchor_chunks:
            section = " > ".join(ch.section_path) if ch.section_path else "(no section)"
            st.caption(f"**{ch.doc_id}** ・ p.{ch.page} ・ `{ch.chunk_id}` ・ {section}")
            highlighted = _highlight_rationale_in_chunk(ch.text, qa.rationale)
            st.markdown(
                f"<div style='font-size:12.5px;line-height:1.7;"
                f"max-height:200px;overflow:auto;white-space:pre-wrap;"
                f"padding:6px 8px;border-left:3px solid #45637A;background:#FAFAFA;'>"
                f"{highlighted}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**根拠** (生成時の引用)")
        for r, hit in rationale_results:
            mark = ":green[✓]" if hit else ":red[✗]"
            st.markdown(f"- {mark} `{r.doc_id}` (p.{r.page}) — {r.text}")

    # ============================================================
    # Checklist — N columns with native progress bar
    # ============================================================
    st.markdown("")
    groups = _checklist_groups(qa)
    total_checks = sum(len(items_) for _, items_ in groups)

    n_checked = sum(
        1 for _, gitems in groups for k, _, _ in gitems
        if st.session_state.get(f"{key_prefix}_{k}_{idx}", False)
    )

    st.markdown("**人手チェックリスト** :gray[→ 全項目 ✓ で Accept ボタンが解放されます]")
    st.progress(
        n_checked / total_checks if total_checks else 0,
        text=f"{n_checked} / {total_checks}",
    )

    ck_cols = st.columns(len(groups))
    check_values: list[bool] = []
    for col, (gname, gitems) in zip(ck_cols, groups):
        with col:
            st.caption(gname)
            for key, label, desc in gitems:
                v = st.checkbox(label, key=f"{key_prefix}_{key}_{idx}", help=desc)
                check_values.append(v)

    all_checked = all(check_values)

    # ============================================================
    # Collapsibles: judge scores + raw JSON
    # ============================================================
    fs = qa.filter_scores
    score_summary = (
        f"答えられるか={fs.answerability or '-'} 根拠妥当性={fs.grounding or '-'} "
        f"漏れ判定={fs.leakage or '-'} 根拠一致={fs.rationale_grounded if fs.rationale_grounded is not None else '-'}"
    )
    with st.expander(f"参考: 判定LLMスコア ({score_summary})", expanded=False):
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("答えられるか", f"{fs.answerability:.1f}/5" if fs.answerability else "—")
        s2.metric("根拠妥当性", f"{fs.grounding:.1f}/5" if fs.grounding else "—")
        s3.metric("回答漏れ判定", str(fs.leakage) if fs.leakage else "—")
        s4.metric("独自性", f"{fs.uniqueness:.2f}" if fs.uniqueness else "—")
        s5.metric("根拠の逐語一致率", f"{fs.rationale_grounded:.0%}" if fs.rationale_grounded is not None else "—")

    with st.expander("生 JSON (デバッグ用)"):
        st.json(qa.model_dump(mode="json"))

    # ============================================================
    # Action bar (bottom)
    # ============================================================
    st.markdown("---")
    now = datetime.now()

    def _mark(status: str, question: str, answer: str) -> None:
        qa.review_status = status  # type: ignore[assignment]
        qa.question = question
        qa.answer = answer
        qa.reviewed_by = reviewer
        qa.reviewed_at = now
        items[idx] = qa
        _save_items(items, out_path)

    gate_msg = (
        ":material/check_circle: チェック完了 → Accept できます"
        if all_checked
        else f":material/info: Accept には残り **{total_checks - n_checked}** 項目のチェックが必要"
    )
    ga1, ga2, ga3, ga4, ga5 = st.columns([3, 1, 1, 1, 1])
    with ga1:
        if all_checked:
            st.success(gate_msg)
        else:
            st.info(gate_msg)
    with ga2:
        if st.button("Reject", width="stretch", type="secondary", key=f"{key_prefix}_reject"):
            _mark("rejected", qa.question, qa.answer)
            if idx < total - 1:
                st.session_state[idx_key] = idx + 1
            st.rerun()
    with ga3:
        if not edit_mode:
            if st.button("Edit", width="stretch", type="secondary", key=f"{key_prefix}_edit"):
                st.session_state[f"{key_prefix}_edit_mode_{idx}"] = True
                st.rerun()
        else:
            if st.button("Save Edit", width="stretch", type="secondary", key=f"{key_prefix}_save"):
                _mark("edited", new_q, new_a)
                st.session_state[f"{key_prefix}_edit_mode_{idx}"] = False
                if idx < total - 1:
                    st.session_state[idx_key] = idx + 1
                st.rerun()
    with ga4:
        if st.button("Accept", width="stretch", type="primary",
                     disabled=not all_checked, key=f"{key_prefix}_accept"):
            _mark("accepted", qa.question, qa.answer)
            if idx < total - 1:
                st.session_state[idx_key] = idx + 1
            st.rerun()
    with ga5:
        if st.button("Save snapshot", width="stretch", key=f"{key_prefix}_snapshot"):
            _save_items(items, out_path)
            st.toast("Saved snapshot")


if __name__ == "__main__":
    # Allow running via `python src/rageval/review_app.py` for quick tests.
    main()
