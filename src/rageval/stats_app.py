"""Streamlit dashboard for visualizing a QA JSONL dataset.

Design principles (避けたい問題に基づく):
- チャートタイトル/凡例の日本語は Vega で文字化けする → st.markdown で書く
- カテゴリ数が少ないとバーが太すぎてみづらい → 横向き(mark_bar)で固定行高にする
- 軸単位の混在(% vs count)はやめる → ぜんぶ件数+割合をテキスト表記
- 1画面詰め込みすぎ → タブ + 1タブ最大3チャート

Usage:
    streamlit run src/rageval/stats_app.py -- --in data/raw
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from rageval.aspects import (
    ALL_ASPECTS,
    ASPECT_DESCRIPTIONS,
    ASPECT_EXAMPLES,
    ASPECT_LABELS,
    ASPECT_TO_CATEGORY,
)
from rageval.schema import QAItem

# ---------------- Plan targets ----------------

CATEGORY_ORDER = ["Integration", "Reasoning", "Logic", "Figure", "Abstention"]
DIFF_ORDER = ["Easy", "Medium", "Hard"]

TARGET_SEARCH_DIFF = {"Easy": 0.375, "Medium": 0.375, "Hard": 0.25}
TARGET_ANSWER_DIFF = {"Easy": 0.175, "Medium": 0.625, "Hard": 0.2}
TARGET_ABSTENTION = (0.10, 0.15)
PILOT_SIZE_RANGE = (30, 50)
ASPECT_MIN_PER = 3  # 各観点 最低3問 (R1 plan)

ANSWERABILITY_PASS = 4.0
GROUNDING_PASS = 4.0
UNIQUENESS_PASS = 1.0 - 0.92


JP_FONT = "Yu Gothic UI, Meiryo, Hiragino Sans, sans-serif"

# ---------------- Color palettes ----------------

# 信号色: 易=緑, 中=黄, 難=赤
DIFFICULTY_COLORS = {"Easy": "#22c55e", "Medium": "#f59e0b", "Hard": "#ef4444"}
CATEGORY_COLORS = {
    "Integration": "#3b82f6",
    "Reasoning":   "#8b5cf6",
    "Logic":       "#06b6d4",
    "Figure":      "#f59e0b",
    "Abstention":  "#6b7280",
}
LEAKAGE_COLORS = {"pass": "#22c55e", "fail": "#ef4444"}
DIFFICULTY_MATCH_COLORS = {
    "aligned":  "#22c55e",
    "too_easy": "#3b82f6",
    "too_hard": "#ef4444",
}
REVIEW_STATUS_COLORS = {
    "pending":  "#9ca3af",
    "accepted": "#22c55e",
    "edited":   "#3b82f6",
    "rejected": "#ef4444",
}
EVIDENCE_STRICTNESS_COLORS = {
    "no-evidence": "#9ca3af",
    "hier-ref":    "#3b82f6",
    "coord-ref":   "#06b6d4",
    "multi-ref":   "#8b5cf6",
}

# ---------------- Display labels (内部値=英語、表示=日本語) ----------------

CATEGORY_LABELS = {
    "Integration": "統合 (Integration)",
    "Reasoning":   "推論 (Reasoning)",
    "Logic":       "論理 (Logic)",
    "Figure":      "図表 (Figure)",
    "Abstention":  "棄権 (Abstention)",
}
DIFFICULTY_LABELS = {
    "Easy":   "易 (Easy)",
    "Medium": "中 (Medium)",
    "Hard":   "難 (Hard)",
}
LEAKAGE_LABELS = {"pass": "合格", "fail": "不合格"}
DIFFICULTY_MATCH_LABELS = {
    "aligned":  "一致",
    "too_easy": "易しすぎる",
    "too_hard": "難しすぎる",
}
EVIDENCE_STRICTNESS_LABELS = {
    "no-evidence": "根拠なし",
    "hier-ref":    "単一根拠",
    "coord-ref":   "並列根拠",
    "multi-ref":   "複数根拠厳密",
}

# ---------------- IO ----------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", default="data/raw")
    parser.add_argument(
        "--track", default="general",
        choices=["all", "general", "kg_poc"],
        help="どの track の QA を表示するか",
    )
    # Review-tab inputs (when 'レビュー' tab is used inside the dashboard)
    parser.add_argument("--reviewed-out", default="data/reviewed",
                        help="レビュー保存先ディレクトリ")
    parser.add_argument("--chunks", default="data/chunks",
                        help="チャンクディレクトリ (元チャンク表示用)")
    parser.add_argument("--reviewer", default="unknown")
    known, _ = parser.parse_known_args()
    return known


def _resolve_inputs(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if p.is_file():
        return [p]
    return [Path(x) for x in sorted(glob.glob(pattern))]


@st.cache_data(show_spinner=False)
def _load_items(
    paths: tuple[str, ...], track_filter: str = "all",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load QAs. Returns (items, skipped_per_file).

    track_filter:
      - "all"     : 全件
      - "general" : KG-PoC (kg_query_type付き) を除外
      - "kg_poc"  : KG-PoC のみ
    """
    items: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    qa = QAItem.model_validate_json(line)
                    is_kg = qa.kg_query_type is not None or qa.kg_novelty is not None
                    if track_filter == "general" and is_kg:
                        continue
                    if track_filter == "kg_poc" and not is_kg:
                        continue
                    row = qa.model_dump(mode="json")
                    row["__source"] = path.name
                    items.append(row)
                except Exception:
                    skipped[path.name] = skipped.get(path.name, 0) + 1
    return items, skipped


def _to_dataframe(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    rows = []
    for it in items:
        fs = it.get("filter_scores") or {}
        gen = it.get("generation") or {}
        ss = it.get("source_structure") or {}
        expl = it.get("explainability") or {}
        rd = it.get("retrieval_difficulty") or {}
        rows.append({
            "qa_id": it.get("qa_id"),
            "source": it.get("__source"),
            "question": it.get("question"),
            "answer": it.get("answer"),
            "business_scenario": it.get("business_scenario"),
            "kg_query_type": it.get("kg_query_type"),
            "kg_novelty": it.get("kg_novelty"),
            "llm_knowledge": it.get("llm_knowledge"),
            "category": ", ".join(it.get("category") or []),
            "category_primary": (it.get("category") or ["unknown"])[0],
            "aspect_list": it.get("aspect") or [],
            "aspect_primary": (it.get("aspect") or ["unknown"])[0],
            "retrieval_level": it.get("retrieval_level"),
            "answer_level": it.get("answer_level"),
            "difficulty_rationale": it.get("difficulty_rationale"),
            "evidence_strictness": expl.get("evidence_strictness"),
            "uses_tables_charts": bool(ss.get("tables_charts")),
            "vocabulary_mismatch": bool(rd.get("vocabulary_mismatch")),
            "doc_id": (it.get("rationale") or [{}])[0].get("doc_id"),
            "n_rationale": len(it.get("rationale") or []),
            "review_status": it.get("review_status"),
            "answerability": fs.get("answerability"),
            "grounding": fs.get("grounding"),
            "uniqueness": fs.get("uniqueness"),
            "leakage": fs.get("leakage"),
            "difficulty_match": fs.get("difficulty_match"),
            "model": gen.get("model"),
            "prompt_version": gen.get("prompt_version"),
        })
    return pd.DataFrame(rows)


# ---------------- Chart primitives ----------------

def _row_bar(
    df: pd.DataFrame,
    column: str,
    order: list[str],
    *,
    targets: dict[str, float] | None = None,
    color_map: dict[str, str] | None = None,
    label_map: dict[str, str] | None = None,
    height_per_row: int = 44,
    show_legend: bool = False,
) -> alt.LayerChart:
    """Horizontal bar with count + percent label inside; optional target tick.

    color_map / label_map: keyed by ORIGINAL value (English). label_map controls
    the Japanese display string used on axis + legend.
    """
    total = len(df) or 1
    counts = df[column].value_counts(dropna=False)

    def disp(v: str) -> str:
        return (label_map or {}).get(v, v)

    rows = []
    for v in order:
        n = int(counts.get(v, 0))
        rows.append({
            "value": str(v),
            "display": disp(str(v)),
            "count": n,
            "ratio": n / total,
            "label": f"{n}件 ({n / total:.0%})",
        })
    d = pd.DataFrame(rows)
    display_order = [disp(str(v)) for v in order]

    x_axis = alt.Axis(
        format="%",
        title="割合",
        titleFont=JP_FONT,
        grid=False,
        values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    y_axis = alt.Axis(
        labelLimit=240,
        labelFont=JP_FONT,
        labelFontSize=12,
        labelOverlap=False,
        labelPadding=6,
    )

    # Use a band-scale step so each row is guaranteed `height_per_row` pixels.
    y_scale = alt.Scale(domain=display_order, paddingInner=0.35, paddingOuter=0.2)

    base = alt.Chart(d).encode(
        y=alt.Y("display:N", sort=display_order, title=None, axis=y_axis, scale=y_scale),
    )

    if color_map:
        domain_disp = [disp(str(v)) for v in order if str(v) in color_map]
        range_ = [color_map[v] for v in order if str(v) in color_map]
        for v in order:
            if str(v) not in color_map:
                domain_disp.append(disp(str(v)))
                range_.append("#94a3b8")
        color_enc = alt.Color(
            "display:N",
            scale=alt.Scale(domain=domain_disp, range=range_),
            legend=alt.Legend(orient="top", title=None, labelFont=JP_FONT, labelFontSize=12) if show_legend else None,
        )
        bars = base.mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("ratio:Q", axis=x_axis, scale=alt.Scale(domain=[0, 1])),
            color=color_enc,
            tooltip=[alt.Tooltip("display:N", title="値"), "count:Q", alt.Tooltip("ratio:Q", format=".1%")],
        )
    else:
        bars = base.mark_bar(color="#3b82f6", cornerRadiusEnd=4).encode(
            x=alt.X("ratio:Q", axis=x_axis, scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("display:N", title="値"), "count:Q", alt.Tooltip("ratio:Q", format=".1%")],
        )

    labels = base.mark_text(
        align="left", baseline="middle", dx=6, color="#111", font=JP_FONT, fontSize=12,
    ).encode(
        x="ratio:Q",
        text="label:N",
        opacity=alt.condition("datum.count > 0", alt.value(1.0), alt.value(0.35)),
    )
    layers = [bars, labels]
    if targets:
        td = pd.DataFrame([{"display": disp(str(v)), "target": targets.get(v, 0.0)} for v in order])
        target_ticks = (
            alt.Chart(td)
            .mark_tick(color="crimson", thickness=3, size=height_per_row - 10)
            .encode(
                x="target:Q",
                y=alt.Y("display:N", sort=display_order, scale=y_scale),
                tooltip=[alt.Tooltip("target:Q", format=".0%", title="目標")],
            )
        )
        layers.append(target_ticks)

    # Use `step` instead of fixed height so each row gets `height_per_row` reliably.
    chart = alt.layer(*layers).properties(
        height=alt.Step(height_per_row),
    )
    return chart


def _coverage_heatmap(df: pd.DataFrame) -> alt.LayerChart:
    """カテゴリ × 検索難易度のヒートマップ。"""
    cats = CATEGORY_ORDER
    diffs = DIFF_ORDER
    cats_disp = [CATEGORY_LABELS.get(c, c) for c in cats]
    diffs_disp = [DIFFICULTY_LABELS.get(d, d) for d in diffs]
    grid = []
    for c in cats:
        for d in diffs:
            n = int(((df["category_primary"] == c) & (df["retrieval_level"] == d)).sum())
            grid.append({
                "category": CATEGORY_LABELS.get(c, c),
                "retrieval_level": DIFFICULTY_LABELS.get(d, d),
                "count": n,
            })
    g = pd.DataFrame(grid)
    max_count = max(1, int(g["count"].max()))
    rects = (
        alt.Chart(g)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("retrieval_level:N", sort=diffs_disp,
                    title="検索難易度", axis=alt.Axis(labelFont=JP_FONT, titleFont=JP_FONT, labelAngle=0)),
            y=alt.Y("category:N", sort=cats_disp,
                    title="カテゴリ", axis=alt.Axis(labelFont=JP_FONT, titleFont=JP_FONT)),
            color=alt.condition(
                "datum.count == 0",
                alt.value("#fde2e2"),
                alt.Color("count:Q",
                          scale=alt.Scale(scheme="yelloworangered", domain=[0, max_count]),
                          legend=alt.Legend(title="件数", titleFont=JP_FONT, orient="right")),
            ),
            tooltip=["category", "retrieval_level", "count"],
        )
    )
    text = (
        alt.Chart(g)
        .mark_text(fontSize=18, font=JP_FONT, fontWeight="bold")
        .encode(
            x=alt.X("retrieval_level:N", sort=diffs_disp),
            y=alt.Y("category:N", sort=cats_disp),
            text="count:Q",
            color=alt.condition(
                f"datum.count > {max_count * 0.6}",
                alt.value("white"),
                alt.value("#111"),
            ),
        )
    )
    return (rects + text).properties(height=300)


def _aspect_coverage_chart(df: pd.DataFrame) -> alt.LayerChart:
    """25観点それぞれの件数。目標(各3問)を満たすかを赤線で示す。"""
    # 各QAの aspect_list を展開
    exploded = df.explode("aspect_list").dropna(subset=["aspect_list"])
    counts = exploded["aspect_list"].value_counts()
    rows = []
    for asp in ALL_ASPECTS:
        n = int(counts.get(asp, 0))
        cat = ASPECT_TO_CATEGORY.get(asp, "?")
        rows.append({
            "aspect": asp,
            "display": f"[{cat}] {ASPECT_LABELS.get(asp, asp)}",
            "count": n,
            "category": cat,
            "label": f"{n}",
        })
    d = pd.DataFrame(rows)
    # カテゴリ順に並べる
    cat_order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    d = d.sort_values(by=["category", "aspect"], key=lambda s: s.map(cat_order) if s.name == "category" else s)
    display_order = d["display"].tolist()

    y_axis = alt.Axis(labelLimit=320, labelFont=JP_FONT, labelFontSize=11,
                       labelOverlap=False, labelPadding=6)
    y_scale = alt.Scale(domain=display_order, paddingInner=0.3, paddingOuter=0.2)
    base = alt.Chart(d).encode(
        y=alt.Y("display:N", sort=display_order, title=None, axis=y_axis, scale=y_scale),
    )
    color_enc = alt.Color(
        "category:N",
        scale=alt.Scale(domain=CATEGORY_ORDER, range=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER]),
        legend=alt.Legend(orient="top", title=None, labelFont=JP_FONT),
    )
    max_count = max(int(d["count"].max()), ASPECT_MIN_PER + 1)
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("count:Q", title="件数", axis=alt.Axis(titleFont=JP_FONT)),
        color=color_enc,
        tooltip=["aspect", "category", "count"],
    )
    labels = base.mark_text(align="left", baseline="middle", dx=4, font=JP_FONT, fontSize=11).encode(
        x="count:Q", text="label:N",
        opacity=alt.condition("datum.count > 0", alt.value(1.0), alt.value(0.35)),
    )
    threshold = (
        alt.Chart(pd.DataFrame({"t": [ASPECT_MIN_PER]}))
        .mark_rule(color="crimson", strokeDash=[6, 4], size=2)
        .encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", title="目標下限")])
    )
    return alt.layer(bars, labels, threshold).properties(height=alt.Step(24)).resolve_scale(color="independent")


def _hist_threshold(df: pd.DataFrame, column: str, threshold: float) -> alt.LayerChart:
    data = df[[column]].dropna().copy()
    if data.empty:
        return alt.Chart(pd.DataFrame({column: [], "count": []})).mark_bar()
    data["pass"] = data[column] >= threshold
    hist = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=10), title=None),
            y=alt.Y("count():Q", title="件数", axis=alt.Axis(titleFont=JP_FONT)),
            color=alt.Color(
                "pass:N",
                scale=alt.Scale(domain=[True, False], range=["#22c55e", "#ef4444"]),
                legend=alt.Legend(
                    orient="top", title=None,
                    labelExpr="datum.label == 'true' ? '合格' : '不合格'",
                    labelFont=JP_FONT,
                ),
            ),
            tooltip=["count()", alt.Tooltip("pass:N", title="合格?")],
        )
    )
    line = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#111", strokeDash=[6, 4], size=2)
        .encode(x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="閾値")])
    )
    return alt.layer(hist, line).properties(height=220)


# ---------------- Status verbalizer ----------------

def _status_lines(df: pd.DataFrame) -> list[tuple[str, str]]:
    n = len(df)
    out: list[tuple[str, str]] = []
    if n == 0:
        return [("warn", "QA が 0 件です。先に `rageval generate` を回してください。")]

    lo, hi = PILOT_SIZE_RANGE
    if n < lo:
        out.append(("warn", f"件数 {n} 件 — パイロット目標 {lo}-{hi} に未達"))
    elif n <= hi:
        out.append(("ok", f"件数 {n} 件 — パイロット規模 {lo}-{hi} に到達"))
    else:
        out.append(("ok", f"件数 {n} 件 — パイロット規模超過(標準フェーズへ)"))

    abstention_n = int((df["category_primary"] == "Abstention").sum())
    ar = abstention_n / n
    lo_a, hi_a = TARGET_ABSTENTION
    if lo_a <= ar <= hi_a:
        out.append(("ok", f"Abstention {abstention_n}件 ({ar:.0%}) — 目標 {lo_a:.0%}-{hi_a:.0%} 内"))
    else:
        msg = "不足" if ar < lo_a else "過多"
        out.append(("warn", f"Abstention {abstention_n}件 ({ar:.0%}) — 目標 {lo_a:.0%}-{hi_a:.0%} に対し{msg}"))

    gaps = [
        f"{c}×{d}"
        for c in CATEGORY_ORDER for d in DIFF_ORDER
        if int(((df["category_primary"] == c) & (df["retrieval_level"] == d)).sum()) == 0
    ]
    if gaps:
        out.append(("warn", f"未カバーセル {len(gaps)}個: {', '.join(gaps[:6])}{' …' if len(gaps) > 6 else ''}"))
    else:
        out.append(("ok", "カテゴリ×検索難易度のセルは全埋め"))

    # 25観点カバレッジ
    aspect_counts = df.explode("aspect_list")["aspect_list"].value_counts() if not df.empty else pd.Series(dtype=int)
    uncovered = [a for a in ALL_ASPECTS if int(aspect_counts.get(a, 0)) < ASPECT_MIN_PER]
    if uncovered:
        out.append((
            "warn",
            f"25観点中 {len(uncovered)}個が目標(各{ASPECT_MIN_PER}問)に未達: "
            f"{', '.join(uncovered[:5])}{' …' if len(uncovered) > 5 else ''}",
        ))
    else:
        out.append(("ok", f"25観点すべて目標(各{ASPECT_MIN_PER}問)を達成"))

    if df["answerability"].notna().any():
        scored = int(df["answerability"].notna().sum())
        passed = int(((df["answerability"] >= ANSWERABILITY_PASS) & (df["leakage"] != "fail")).sum())
        rate = passed / scored if scored else 0
        out.append(("info", f"判定済 {scored}件中、しきい値クリア {passed}件 ({rate:.0%})"))
    return out


# ---------------- Tab body renderers (extracted so both general and kg_poc
# track-specific dashboards can reuse the bodies) ----------------

def _render_balance_tab(df: pd.DataFrame) -> None:
    st.caption("色: 緑=易 / 黄=中 / 赤=難。赤縦線=目標値。バーが赤線に近いほど良い配分。")
    st.markdown("#### 検索難易度 (目標: 易 37.5% / 中 37.5% / 難 25%)")
    st.altair_chart(
        _row_bar(df, "retrieval_level", DIFF_ORDER, targets=TARGET_SEARCH_DIFF,
                 color_map=DIFFICULTY_COLORS, label_map=DIFFICULTY_LABELS, show_legend=True),
        width="stretch",
    )
    st.markdown("#### 回答難易度 (目標: 易 17.5% / 中 62.5% / 難 20%)")
    st.altair_chart(
        _row_bar(df, "answer_level", DIFF_ORDER, targets=TARGET_ANSWER_DIFF,
                 color_map=DIFFICULTY_COLORS, label_map=DIFFICULTY_LABELS, show_legend=True),
        width="stretch",
    )
    if "category_primary" in df.columns and df["category_primary"].notna().any():
        st.markdown("#### 棄権(Abstention)比率 (目標: 10%-15%)")
        absten_n = int((df["category_primary"] == "Abstention").sum())
        absten_r = absten_n / len(df) if len(df) else 0
        st.progress(min(absten_r / 0.15, 1.0), text=f"{absten_n} 件 / {len(df)} 件 = {absten_r:.1%}")


def _render_quality_tab(df: pd.DataFrame) -> None:
    if not df["answerability"].notna().any():
        st.info("filter_scores がまだありません。`rageval filter` を実行してください。")
        return
    st.caption("赤点線=しきい値。右側に山があれば良好。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**答えやすさ (answerability)** しきい値 ≥ {ANSWERABILITY_PASS}")
        st.altair_chart(_hist_threshold(df, "answerability", ANSWERABILITY_PASS), width="stretch")
    with c2:
        st.markdown(f"**根拠妥当性 (grounding)** しきい値 ≥ {GROUNDING_PASS}")
        st.altair_chart(_hist_threshold(df, "grounding", GROUNDING_PASS), width="stretch")
    with c3:
        st.markdown(f"**独自性 (uniqueness)** しきい値 ≥ {UNIQUENESS_PASS:.2f}")
        st.altair_chart(_hist_threshold(df, "uniqueness", UNIQUENESS_PASS), width="stretch")

    st.markdown("#### 棄却理由内訳")
    reasons: list[dict] = []
    for _, row in df.iterrows():
        if pd.notna(row["answerability"]) and row["answerability"] < ANSWERABILITY_PASS:
            reasons.append({"reason": "answerability < 4"})
        if row["leakage"] == "fail":
            reasons.append({"reason": "leakage = fail"})
        if pd.notna(row["grounding"]) and row["grounding"] < GROUNDING_PASS:
            reasons.append({"reason": "grounding < 4"})
        if pd.notna(row["uniqueness"]) and row["uniqueness"] < UNIQUENESS_PASS:
            reasons.append({"reason": "uniqueness < 0.08 (重複疑い)"})
    if reasons:
        rdf = pd.DataFrame(reasons)
        reason_order = sorted(rdf["reason"].unique(), key=lambda r: -int((rdf["reason"] == r).sum()))
        st.altair_chart(_row_bar(rdf, "reason", reason_order), width="stretch")
    else:
        st.success("棄却対象なし。全件しきい値クリア。")

    st.markdown("#### リーク判定 / 難易度整合")
    lc1, lc2 = st.columns(2)
    with lc1:
        if df["leakage"].notna().any():
            st.altair_chart(
                _row_bar(df.dropna(subset=["leakage"]), "leakage", ["pass", "fail"],
                         color_map=LEAKAGE_COLORS, label_map=LEAKAGE_LABELS, show_legend=True),
                width="stretch",
            )
    with lc2:
        if df["difficulty_match"].notna().any():
            st.altair_chart(
                _row_bar(df.dropna(subset=["difficulty_match"]), "difficulty_match",
                         ["aligned", "too_easy", "too_hard"],
                         color_map=DIFFICULTY_MATCH_COLORS, label_map=DIFFICULTY_MATCH_LABELS,
                         show_legend=True),
                width="stretch",
            )


def _render_list_tab(df: pd.DataFrame) -> None:
    st.caption(
        "カラムヘッダクリックでソート。レビュー状況はサイドバーで絞り込み・"
        "上部 KPI で件数確認。詳細は『個別表示』タブへ。"
    )
    is_kg_df = df["kg_query_type"].notna().any() if "kg_query_type" in df.columns else False

    def _diff(r) -> str:
        return f"{r['retrieval_level'] or '?'} / {r['answer_level'] or '?'}"

    def _score(r) -> str:
        a, g = r.get("answerability"), r.get("grounding")
        if pd.isna(a) and pd.isna(g):
            return "—"
        a_s = f"{a:.1f}" if pd.notna(a) else "—"
        g_s = f"{g:.1f}" if pd.notna(g) else "—"
        leak = r.get("leakage") or ""
        return f"{a_s}/{g_s}" + (" ⚠リーク" if leak == "fail" else "")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "質問": r.get("question") or "",
            "タグ": r.get("kg_query_type") if is_kg_df else r.get("category_primary"),
            "難易度 (検索/回答)": _diff(r),
            "判定スコア (ans/grd)": _score(r),
        })
    list_df = pd.DataFrame(rows)
    st.dataframe(
        list_df, width="stretch", hide_index=True, height=520,
        column_config={
            "質問": st.column_config.TextColumn(width="large"),
            "タグ": st.column_config.TextColumn(
                width="small",
                help="general track: カテゴリ / KG-PoC track: クエリ型",
            ),
            "難易度 (検索/回答)": st.column_config.TextColumn(width="small"),
            "判定スコア (ans/grd)": st.column_config.TextColumn(
                width="small",
                help="answerability / grounding (各 5点満点)。⚠リークは judge が leakage=fail と判定",
            ),
        },
    )


def _render_detail_tab(df: pd.DataFrame, items: list[dict]) -> None:
    if len(df) == 0:
        st.info("対象がありません。")
        return
    ids = df["qa_id"].tolist()
    sel = st.selectbox("qa_id", ids, label_visibility="collapsed")
    row = df[df["qa_id"] == sel].iloc[0]
    is_kg = bool(row.get("kg_query_type"))

    # トップに識別子と分類タグを caption で一行
    diff_color = {"Easy": "green", "Medium": "orange", "Hard": "red"}
    parts: list[str] = [f"`{sel}`"]
    if is_kg:
        parts.append(f"track=`kg_poc` クエリ型=**{row['kg_query_type']}** 未知性=**{row['kg_novelty']}**")
        if row.get("llm_knowledge"):
            parts.append(f"LLM既知性=**{row['llm_knowledge']}**")
    else:
        aspects_str = ", ".join(row["aspect_list"]) if isinstance(row["aspect_list"], list) else ""
        parts.append(f"カテゴリ=**{row.get('category_primary', '?')}** 観点=**{aspects_str}**")
    parts.append(f"検索=:{diff_color.get(row['retrieval_level'], 'gray')}[{row['retrieval_level']}]")
    parts.append(f"回答=:{diff_color.get(row['answer_level'], 'gray')}[{row['answer_level']}]")
    st.caption(" ・ ".join(parts))

    # Q/A をレビューパネルと同じスタイル
    col_qa, col_meta = st.columns([2, 1])
    with col_qa:
        st.markdown("**質問**")
        st.info(row["question"])
        st.markdown("**回答**")
        st.markdown("> " + (row["answer"] or "").replace("\n", "\n> "))
        if row.get("difficulty_rationale"):
            st.caption(f"難易度根拠: `{row['difficulty_rationale']}`")
    with col_meta:
        st.markdown("**判定スコア**")
        m1, m2 = st.columns(2)
        m1.metric("answerability", f"{row['answerability']:.1f}" if pd.notna(row['answerability']) else "—")
        m2.metric("grounding", f"{row['grounding']:.1f}" if pd.notna(row['grounding']) else "—")
        m3, m4 = st.columns(2)
        m3.metric("uniqueness", f"{row['uniqueness']:.2f}" if pd.notna(row['uniqueness']) else "—")
        m4.metric("leakage", str(row['leakage']) if pd.notna(row['leakage']) else "—")
        st.caption(f"source: `{row['source']}` ・ review: **{row['review_status']}**")

    # rationale を本文として出す (chunk への参照)
    original = next((it for it in items if it.get("qa_id") == sel), None)
    if original and (rats := original.get("rationale") or []):
        st.markdown("**根拠** (生成時の引用)")
        for r in rats:
            st.markdown(f"- `{r.get('doc_id')}` (p.{r.get('page')}) — {r.get('text')}")

    if original:
        with st.expander("生 JSON"):
            st.json(original)


# ---------------- Main ----------------

def main() -> None:
    args = _parse_args()
    track_filter = getattr(args, "track", "general")
    st.set_page_config(page_title="rageval stats", layout="wide", page_icon=":bar_chart:")
    track_badge = {
        "all":     ("全track",          "#1F2329"),
        "general": ("general (25観点)", "#45637A"),
        "kg_poc":  ("KG-PoC (3軸)",    "#C25239"),
    }.get(track_filter, (track_filter, "#1F2329"))
    st.markdown(
        f"## RAG評価ダッシュボード "
        f"<span style='font-size:12px;padding:3px 10px;background:{track_badge[1]};"
        f"color:white;margin-left:8px;vertical-align:4px;'>{track_badge[0]}</span>",
        unsafe_allow_html=True,
    )

    # Sidebar input
    st.sidebar.header("入力")
    pattern = st.sidebar.text_input("path / dir / glob", args.input)
    paths = _resolve_inputs(pattern)
    if not paths:
        st.error(f"No files match: {pattern}")
        return
    st.sidebar.caption(f"track: **{track_badge[0]}**")
    st.sidebar.caption("読み込みファイル:")
    for p in paths:
        st.sidebar.caption(f"• {p.name}")

    items, skipped = _load_items(tuple(str(p) for p in paths), track_filter=track_filter)
    if skipped:
        with st.sidebar.expander(f"スキップ {sum(skipped.values())} 件 (旧スキーマ)", expanded=False):
            for name, n in skipped.items():
                st.caption(f"• {name}: {n} 件")
    df_all = _to_dataframe(items)
    if df_all.empty:
        st.warning("No valid QA items.")
        return

    # Sidebar filters
    st.sidebar.divider()
    st.sidebar.header("絞り込み")
    cats_avail = sorted(df_all["category_primary"].dropna().unique())
    status_avail = sorted(df_all["review_status"].dropna().unique())
    src_avail = sorted(df_all["source"].dropna().unique())
    sel_cat = st.sidebar.multiselect("カテゴリ", cats_avail, default=cats_avail)
    sel_status = st.sidebar.multiselect("review_status", status_avail, default=status_avail)
    sel_source = st.sidebar.multiselect("source file", src_avail, default=src_avail)

    df = df_all[
        df_all["category_primary"].isin(sel_cat)
        & df_all["review_status"].isin(sel_status)
        & df_all["source"].isin(sel_source)
    ]

    # Top status
    st.caption(f"対象 {len(df)} / {len(df_all)} 件")
    for kind, msg in _status_lines(df):
        if kind == "ok":
            st.success(msg, icon=":material/check_circle:")
        elif kind == "warn":
            st.warning(msg, icon=":material/warning:")
        else:
            st.info(msg, icon=":material/info:")

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("QA総数", len(df))
    k2.metric("ドキュメント数", df["doc_id"].nunique())
    k3.metric("採点済QA", int(df["answerability"].notna().sum()))
    k4.metric("未レビュー", int((df["review_status"] == "pending").sum()))
    k5.metric("承認済", int((df["review_status"] == "accepted").sum()))
    k6.metric("却下", int((df["review_status"] == "rejected").sum()))

    st.divider()

    # レビュー用 in-memory QAItem リストを構築 (df 経由ではなく元の items から)
    from rageval.review_app import render_review_panel
    from rageval.schema import QAItem
    # 現在の絞り込み (df 由来) に含まれる qa_id だけを取り出す
    df_ids = set(df["qa_id"].tolist())
    review_qas = [QAItem.model_validate(it) for it in items if it.get("qa_id") in df_ids]
    # session に持つ (フィルタやファイル切替で reset)
    review_state_key = f"review_qas_{paths[0].name}_{track_filter}"
    if st.session_state.get("_review_state_key") != review_state_key:
        st.session_state.review_qas = review_qas
        st.session_state.rv_idx = 0
        st.session_state._review_state_key = review_state_key
    reviewed_out = Path(args.reviewed_out) / paths[0].name

    # KG-PoC専用ダッシュボード
    if track_filter == "kg_poc":
        kg_labels = ["KG-PoC概要", "バランス", "品質スコア", "一覧", "レビュー", "個別表示"]
        kg_tabs = st.tabs(kg_labels)
        with kg_tabs[0]:
            _render_kg_tab(df)
        with kg_tabs[1]:
            _render_balance_tab(df)
        with kg_tabs[2]:
            _render_quality_tab(df)
        with kg_tabs[3]:
            _render_list_tab(df)
        with kg_tabs[4]:
            render_review_panel(
                items=st.session_state.review_qas,
                out_path=reviewed_out,
                chunks_dir=args.chunks,
                reviewer=args.reviewer,
                key_prefix="rv",
                show_header=True,
                input_label=f"reviewed → <b>{reviewed_out}</b>",
                track_label="[KG-PoC]",
            )
        with kg_tabs[5]:
            _render_detail_tab(df, items)
        return

    # general/all track: 概要にバランスを統合し、独立タブは廃止
    has_kg = df["kg_query_type"].notna().any() or df["kg_novelty"].notna().any()
    tab_labels = ["概要", "カバレッジ", "品質スコア", "一覧", "レビュー", "個別表示"]
    if has_kg:
        tab_labels.append("KG-PoC")
    tabs = st.tabs(tab_labels)
    tab_overview, tab_coverage, tab_quality, tab_list, tab_review, tab_detail = tabs[:6]
    tab_kg = tabs[6] if has_kg else None
    with tab_review:
        render_review_panel(
            items=st.session_state.review_qas,
            out_path=reviewed_out,
            chunks_dir=args.chunks,
            reviewer=args.reviewer,
            key_prefix="rv",
            show_header=True,
            input_label=f"reviewed → <b>{reviewed_out}</b>",
            track_label=f"[{track_filter}]",
        )

    # ===== 概要 (Executive Summary) =====
    with tab_overview:
        # --- 1. 健全性 KPI ---
        st.markdown("### 健全性スコアカード")
        lo, hi = PILOT_SIZE_RANGE
        n_total = len(df)
        size_rate = min(n_total / lo, 1.0) if lo else 0

        aspect_counts = df.explode("aspect_list")["aspect_list"].value_counts() if not df.empty else pd.Series(dtype=int)
        covered_aspects = int((aspect_counts >= 1).sum())
        well_covered = int((aspect_counts >= ASPECT_MIN_PER).sum())

        n_reviewed = int((df["review_status"].isin(["accepted", "edited", "rejected"])).sum())
        review_rate = n_reviewed / n_total if n_total else 0

        scored = df.dropna(subset=["answerability"])
        if len(scored):
            passed = int(((scored["answerability"] >= ANSWERABILITY_PASS) & (scored["leakage"] != "fail")).sum())
            quality_rate = passed / len(scored)
            quality_text = f"{quality_rate:.0%} ({passed}/{len(scored)})"
        else:
            quality_rate = None
            quality_text = "未判定"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            f"件数達成 (目標 {lo}-{hi})",
            f"{n_total} 件",
            delta=f"{n_total - lo:+d} 対パイロット下限" if n_total else None,
            delta_color="normal" if n_total >= lo else "inverse",
        )
        k2.metric(
            "観点カバレッジ (≥1問)",
            f"{covered_aspects} / 25",
            delta=f"目標達成(≥{ASPECT_MIN_PER}問): {well_covered}/25",
        )
        k3.metric(
            "レビュー進捗",
            f"{review_rate:.0%}",
            delta=f"{n_reviewed}/{n_total} 件",
        )
        k4.metric(
            "品質ゲート通過率",
            quality_text,
            delta=None if quality_rate is None else (
                "✓ 健全" if quality_rate >= 0.7 else "要改善"
            ),
            delta_color="normal" if (quality_rate or 0) >= 0.7 else "inverse",
        )

        st.divider()

        # --- 2. 分布の俯瞰 ---
        st.markdown("### 分布の俯瞰")
        st.caption("赤縦線=目標値。バーが赤線に近いほど良い配分。")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown("**カテゴリ**")
            st.altair_chart(
                _row_bar(df, "category_primary", CATEGORY_ORDER,
                         color_map=CATEGORY_COLORS, label_map=CATEGORY_LABELS),
                width="stretch",
            )
        with oc2:
            st.markdown("**検索難易度** (目標: 易 37.5% / 中 37.5% / 難 25%)")
            st.altair_chart(
                _row_bar(df, "retrieval_level", DIFF_ORDER, targets=TARGET_SEARCH_DIFF,
                         color_map=DIFFICULTY_COLORS, label_map=DIFFICULTY_LABELS),
                width="stretch",
            )
        with oc3:
            st.markdown("**回答難易度** (目標: 易 17.5% / 中 62.5% / 難 20%)")
            st.altair_chart(
                _row_bar(df, "answer_level", DIFF_ORDER, targets=TARGET_ANSWER_DIFF,
                         color_map=DIFFICULTY_COLORS, label_map=DIFFICULTY_LABELS),
                width="stretch",
            )

        # 棄権比率 (元バランスタブから移動)
        if "category_primary" in df.columns and df["category_primary"].notna().any():
            st.markdown("**棄権 (Abstention) 比率** (目標: 10%-15%)")
            absten_n = int((df["category_primary"] == "Abstention").sum())
            absten_r = absten_n / len(df) if len(df) else 0
            st.progress(min(absten_r / 0.15, 1.0), text=f"{absten_n} 件 / {len(df)} 件 = {absten_r:.1%}")

        st.divider()

        # --- 3. 25観点リファレンス ---
        st.markdown(f"### 25観点リファレンス (目標: 各観点 ≥ {ASPECT_MIN_PER} 問)")
        st.caption("カテゴリごとに展開。件数列は現在のデータセットでの登場数。")

        for cat in CATEGORY_ORDER:
            aspects_in_cat = [a for a in ALL_ASPECTS if ASPECT_TO_CATEGORY.get(a) == cat]
            n_achieved = sum(1 for a in aspects_in_cat if int(aspect_counts.get(a, 0)) >= ASPECT_MIN_PER)
            cat_label = CATEGORY_LABELS.get(cat, cat)
            header = f"**{cat_label}**  ({n_achieved} / {len(aspects_in_cat)} 観点が目標達成)"
            with st.expander(header, expanded=(cat == "Integration")):
                rows = []
                for a in aspects_in_cat:
                    cnt = int(aspect_counts.get(a, 0))
                    status = "✓" if cnt >= ASPECT_MIN_PER else ("△" if cnt > 0 else "✗")
                    rows.append({
                        "状態": status,
                        "観点": ASPECT_LABELS.get(a, a),
                        "件数": cnt,
                        "観点ID": a,
                        "定義": ASPECT_DESCRIPTIONS.get(a, ""),
                        "現場例": ASPECT_EXAMPLES.get(a, ""),
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch", hide_index=True,
                    column_config={
                        "状態": st.column_config.TextColumn(width="small"),
                        "件数": st.column_config.NumberColumn(width="small"),
                        "観点ID": st.column_config.TextColumn(width="small"),
                        "観点": st.column_config.TextColumn(width="medium"),
                        "定義": st.column_config.TextColumn(width="large"),
                        "現場例": st.column_config.TextColumn(width="large"),
                    },
                )

    # ===== カバレッジ (general track only - uses 25 aspects) =====
    with tab_coverage:
        st.markdown("#### カテゴリ × 検索難易度")
        st.caption("赤背景セル = 0件 (Phase 2 の補充対象)")
        st.altair_chart(_coverage_heatmap(df), width="stretch")

        st.markdown(f"#### 25観点カバレッジ (目標: 各観点 最低 {ASPECT_MIN_PER}問)")
        st.caption("赤縦線が目標下限。バーが赤線より右なら達成。")
        st.altair_chart(_aspect_coverage_chart(df), width="stretch")

        st.markdown("#### ドキュメント別件数")
        doc_order = sorted(df["doc_id"].dropna().unique())
        st.altair_chart(_row_bar(df, "doc_id", doc_order), width="stretch")

    # ===== 品質スコア =====
    with tab_quality:
        _render_quality_tab(df)

    # ===== 一覧 =====
    with tab_list:
        _render_list_tab(df)

    # ===== 個別表示 =====
    with tab_detail:
        _render_detail_tab(df, items)

    # ===== KG-PoC (only when KG fields present) =====
    if tab_kg is not None:
        with tab_kg:
            _render_kg_tab(df)


def _render_kg_tab(df: pd.DataFrame) -> None:
    """KG-PoC dashboard: 3-axis matrix + LLM-knowledge breakdown."""
    from rageval.tracks.kg_poc import (
        ALL_KG_NOVELTY,
        ALL_KG_QUERY_TYPES,
        KG_NOVELTY_LABELS,
        KG_QUERY_TYPE_LABELS,
        LLM_KNOWLEDGE_LABELS,
    )

    st.markdown("### KG-PoC 評価設計フレーム")
    st.caption("3軸タグ (クエリ型 × 未知性 × LLM既知性) の配分可視化")

    kg_df = df.dropna(subset=["kg_query_type", "kg_novelty"])
    if kg_df.empty:
        st.warning("KG-PoC track の QA が見つかりません。`rageval generate --track kg_poc` で生成してください。")
        return

    # KPIs
    n_total = len(kg_df)
    n_probed = int(kg_df["llm_knowledge"].notna().sum())
    n_unknown = int((kg_df["llm_knowledge"] == "unknown").sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("KG-PoC 総数", n_total)
    k2.metric("プロービング済", f"{n_probed}/{n_total}")
    k3.metric("LLM未知 (RAG必須)", n_unknown)

    if n_probed == 0:
        st.info("`rageval probe --in <jsonl>` を実行すると、LLM既知性が判定されます。")

    st.markdown("#### クエリ型 × 未知性 マトリクス")
    st.caption("セル数 = QA件数。0件セルは薄い色で表示。")

    rows = []
    for qt in ALL_KG_QUERY_TYPES:
        for nov in ALL_KG_NOVELTY:
            n = int(((kg_df["kg_query_type"] == qt) & (kg_df["kg_novelty"] == nov)).sum())
            n_unk = int(((kg_df["kg_query_type"] == qt) & (kg_df["kg_novelty"] == nov)
                         & (kg_df["llm_knowledge"] == "unknown")).sum())
            rows.append({
                "query_type":    KG_QUERY_TYPE_LABELS[qt],
                "novelty":       KG_NOVELTY_LABELS[nov],
                "count":         n,
                "count_unknown": n_unk,
            })
    mdf = pd.DataFrame(rows)

    # Domain は 0 を起点にして、低件数でも色のグラデーションが見える
    max_count = max(1, int(mdf["count"].max()))
    rects = (
        alt.Chart(mdf)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("novelty:N", sort=[KG_NOVELTY_LABELS[n] for n in ALL_KG_NOVELTY],
                    title="未知性", axis=alt.Axis(labelFont=JP_FONT, titleFont=JP_FONT)),
            y=alt.Y("query_type:N", sort=[KG_QUERY_TYPE_LABELS[q] for q in ALL_KG_QUERY_TYPES],
                    title="クエリ型", axis=alt.Axis(labelFont=JP_FONT, titleFont=JP_FONT)),
            color=alt.condition(
                "datum.count == 0",
                alt.value("#F0F0F0"),  # 0件 = 薄いグレー
                alt.Color(
                    "count:Q",
                    scale=alt.Scale(
                        scheme="yelloworangered",
                        domain=[0, max_count],
                    ),
                    legend=alt.Legend(title="件数", titleFont=JP_FONT),
                ),
            ),
            tooltip=["query_type", "novelty", "count", "count_unknown"],
        )
    )
    # 件数によって自動的に色のコントラスト確保。高件数=濃赤、低件数=黄色
    text = (
        alt.Chart(mdf)
        .mark_text(fontSize=16, font=JP_FONT, fontWeight="bold")
        .encode(
            x=alt.X("novelty:N", sort=[KG_NOVELTY_LABELS[n] for n in ALL_KG_NOVELTY]),
            y=alt.Y("query_type:N", sort=[KG_QUERY_TYPE_LABELS[q] for q in ALL_KG_QUERY_TYPES]),
            text="count:Q",
            color=alt.condition(
                f"datum.count > {max_count * 0.6}",
                alt.value("white"),
                alt.value("#111"),
            ),
        )
    )
    st.altair_chart((rects + text).properties(height=320), width="stretch")

    # LLM-knowledge breakdown by axis
    if n_probed > 0:
        st.markdown("#### LLM既知性内訳 (プロービング結果)")
        st.caption("「未知」セルが多いほど、KG-RAG 検証用データとして価値が高い (RAG 無しで答えられないため)")

        order = ["unknown", "known"]
        knowledge_labels = {k: LLM_KNOWLEDGE_LABELS[k] for k in order}
        st.altair_chart(
            _row_bar(
                kg_df.dropna(subset=["llm_knowledge"]),
                "llm_knowledge",
                order,
                color_map={"unknown": "#ef4444", "known": "#94a3b8"},
                label_map=knowledge_labels,
                show_legend=True,
            ),
            width="stretch",
        )


if __name__ == "__main__":
    main()
