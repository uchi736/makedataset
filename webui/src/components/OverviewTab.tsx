import { CATEGORY_COLORS, DIFFICULTY_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { countBy, isKg, pct, statusCounts } from "../util";
import { DistBar, Legend, StatTile } from "./charts";
import type { DistRow } from "./charts";

interface StatusLine { kind: "ok" | "warn" | "info"; msg: string }

// 目標値は配信 (labels.targets) が正本。説明文・帯の色・帯の満杯基準を同じ値から
// 組み立てる — 直値で書くと、目標を動かしたとき上段の文言だけが新しい目標を語り、
// 真下の帯は旧区間で色を決める (緑の帯の上に「過多」と書かれる) 自己矛盾が起きる。
function fmtPct(v: number): string {
  return `${Math.round(v * 1000) / 10}%`;
}

function targetText(labels: Labels, targets: Record<string, number>): string {
  return labels.diff_order
    .map((d) => `${labels.difficulty_labels[d] ?? d} ${fmtPct(targets[d] ?? 0)}`)
    .join(" / ");
}

function buildStatusLines(items: QAItem[], labels: Labels): StatusLine[] {
  const n = items.length;
  const out: StatusLine[] = [];
  if (n === 0) return [{ kind: "warn", msg: "QA が 0 件です。データセットの選択と絞り込みを確認してください。" }];

  const [lo, hi] = labels.targets.pilot_size;
  if (n < lo) out.push({ kind: "warn", msg: `件数 ${n} 件 — パイロット目標 ${lo}-${hi} に未達` });
  else if (n <= hi) out.push({ kind: "ok", msg: `件数 ${n} 件 — パイロット規模 ${lo}-${hi} に到達` });
  else out.push({ kind: "ok", msg: `件数 ${n} 件 — パイロット規模超過 (標準フェーズへ)` });

  const generalItems = items.filter((it) => !isKg(it));
  if (generalItems.length > 0) {
    const abstN = generalItems.filter((it) => (it.category[0] ?? "") === "Abstention").length;
    const ar = abstN / n;
    const [loA, hiA] = labels.targets.abstention;
    if (loA <= ar && ar <= hiA) {
      out.push({ kind: "ok", msg: `棄権 (Abstention) ${abstN}件 (${pct(ar)}) — 目標 ${fmtPct(loA)}-${fmtPct(hiA)} 内` });
    } else {
      out.push({
        kind: "warn",
        msg: `棄権 (Abstention) ${abstN}件 (${pct(ar)}) — 目標 ${fmtPct(loA)}-${fmtPct(hiA)} に対し${ar < loA ? "不足" : "過多"}`,
      });
    }

    const gaps: string[] = [];
    for (const c of labels.category_order) {
      for (const d of labels.diff_order) {
        const hit = generalItems.some((it) => (it.category[0] ?? "") === c && it.retrieval_level === d);
        if (!hit) gaps.push(`${labels.category_labels[c] ?? c}×${labels.difficulty_labels[d] ?? d}`);
      }
    }
    if (gaps.length) {
      out.push({
        kind: "warn",
        msg: `未カバーセル ${gaps.length}個: ${gaps.slice(0, 6).join(", ")}${gaps.length > 6 ? " …" : ""}`,
      });
    } else {
      out.push({ kind: "ok", msg: "カテゴリ×検索難易度のセルは全埋め" });
    }

    const aspectCounts = new Map<string, number>();
    for (const it of generalItems) for (const a of it.aspect) aspectCounts.set(a, (aspectCounts.get(a) ?? 0) + 1);
    const min = labels.targets.aspect_min;
    const uncovered = labels.all_aspects.filter((a) => (aspectCounts.get(a) ?? 0) < min);
    // 観点の数も配信 (all_aspects) から数える。観点表を増減したときに
    // 文言だけ 25 のまま残ると、未達の分母が画面から読めなくなる。
    const nAspects = labels.all_aspects.length;
    if (uncovered.length) {
      out.push({
        kind: "warn",
        msg: `${nAspects}観点中 ${uncovered.length}個が目標 (各${min}問) に未達: ${uncovered.slice(0, 5).map((a) => labels.aspect_labels[a] ?? a).join(", ")}${uncovered.length > 5 ? " …" : ""}`,
      });
    } else {
      out.push({ kind: "ok", msg: `${nAspects}観点すべて目標 (各${min}問) を達成` });
    }
  }

  const scored = items.filter((it) => it.filter_scores.answerability != null);
  if (scored.length > 0) {
    const passed = scored.filter(
      (it) =>
        (it.filter_scores.answerability ?? 0) >= labels.targets.answerability_pass &&
        it.filter_scores.leakage !== "fail",
    ).length;
    out.push({ kind: "info", msg: `判定済 ${scored.length}件中、しきい値クリア ${passed}件 (${pct(passed / scored.length)})` });
  }
  return out;
}

export default function OverviewTab({ items, labels }: { items: QAItem[]; labels: Labels }) {
  const n = items.length;
  const counts = statusCounts(items);
  const nDocs = new Set(items.map((it) => it.rationale[0]?.doc_id).filter(Boolean)).size;
  const nScored = items.filter((it) => it.filter_scores.answerability != null).length;
  const generalItems = items.filter((it) => !isKg(it));

  const diffRows = (key: "retrieval_level" | "answer_level", targets: Record<string, number>): DistRow[] => {
    const m = countBy(items, (it) => it[key] as string);
    return labels.diff_order.map((d) => ({
      key: d,
      label: labels.difficulty_labels[d] ?? d,
      count: m.get(d) ?? 0,
      color: DIFFICULTY_COLORS[d],
      target: targets[d],
    }));
  };

  const catRows: DistRow[] = labels.category_order.map((c) => {
    const m = countBy(generalItems, (it) => it.category[0]);
    return {
      key: c,
      label: labels.category_labels[c] ?? c,
      count: m.get(c) ?? 0,
      color: CATEGORY_COLORS[c] ?? "var(--cat-1)",
    };
  });

  const abstN = generalItems.filter((it) => (it.category[0] ?? "") === "Abstention").length;
  const abstR = n ? abstN / n : 0;
  const [abstLo, abstHi] = labels.targets.abstention;

  const statusLines = buildStatusLines(items, labels);

  return (
    <div>
      <div className="tile-row">
        <StatTile label="QA総数" value={n} />
        <StatTile label="ドキュメント数" value={nDocs} />
        <StatTile label="採点済QA" value={nScored} />
        <StatTile
          label="未レビュー"
          value={counts.pending}
          delta={n ? `残り ${pct(counts.pending / n)}` : undefined}
          tone={counts.pending === 0 ? "good" : "plain"}
        />
        <StatTile label="採用" value={counts.accepted} tone="good" />
        <StatTile label="却下" value={counts.rejected} />
      </div>

      <div className="card">
        <h3>健全性チェック</h3>
        <p className="caption">rag_eval_dataset_r1_plan の目標値に対する現在地。</p>
        {statusLines.map((s, i) => (
          <div className={`status-line ${s.kind}`} key={i}>
            <span className="icon">{s.kind === "ok" ? "●" : s.kind === "warn" ? "▲" : "◆"}</span>
            <span>{s.msg}</span>
          </div>
        ))}
      </div>

      <div className={generalItems.length > 0 ? "grid-3" : "grid-2"}>
        {generalItems.length > 0 && (
          <div className="card">
            <h3>カテゴリ</h3>
            <p className="caption">general トラックの主分類 (先頭カテゴリ)。</p>
            <DistBar rows={catRows} total={generalItems.length} />
          </div>
        )}
        <div className="card">
          <h3>検索難易度</h3>
          <p className="caption">目標: {targetText(labels, labels.targets.search_diff)} (赤い刻みが目標)。</p>
          <Legend keys={labels.diff_order.map((d) => ({ label: labels.difficulty_labels[d] ?? d, color: DIFFICULTY_COLORS[d] }))} />
          <DistBar rows={diffRows("retrieval_level", labels.targets.search_diff)} total={n} />
        </div>
        <div className="card">
          <h3>回答難易度</h3>
          <p className="caption">目標: {targetText(labels, labels.targets.answer_diff)} (赤い刻みが目標)。</p>
          <Legend keys={labels.diff_order.map((d) => ({ label: labels.difficulty_labels[d] ?? d, color: DIFFICULTY_COLORS[d] }))} />
          <DistBar rows={diffRows("answer_level", labels.targets.answer_diff)} total={n} />
        </div>
      </div>

      {generalItems.length > 0 && (
        <div className="card">
          <h3>棄権 (Abstention) 比率</h3>
          <p className="caption">
            目標: {fmtPct(abstLo)}〜{fmtPct(abstHi)}。答えられない問いを一定割合入れて、無理に答えない能力を測る。
          </p>
          <div className="progress-track" style={{ maxWidth: 480 }}>
            <div
              className="progress-fill"
              style={{
                width: pct(abstHi > 0 ? Math.min(abstR / abstHi, 1) : 0),
                background: abstR >= abstLo && abstR <= abstHi ? "var(--status-good)" : "var(--status-warning)",
              }}
            />
          </div>
          <span style={{ fontSize: 13 }}>{abstN} 件 / {n} 件 = {pct(abstR, 1)}</span>
        </div>
      )}

    </div>
  );
}
