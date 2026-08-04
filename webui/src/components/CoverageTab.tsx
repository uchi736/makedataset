import { CATEGORY_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { countBy, isKg } from "../util";
import { DistBar, Heatmap, Legend } from "./charts";

export default function CoverageTab({ items, labels }: { items: QAItem[]; labels: Labels }) {
  const generalItems = items.filter((it) => !isKg(it));
  if (generalItems.length === 0) {
    return <div className="empty-note">general トラックの QA がありません (25観点カバレッジは general 専用)。</div>;
  }

  // カテゴリ × 検索難易度
  const yLabels = labels.category_order.map((c) => labels.category_labels[c] ?? c);
  const xLabels = labels.diff_order.map((d) => labels.difficulty_labels[d] ?? d);
  const values = labels.category_order.map((c) =>
    labels.diff_order.map(
      (d) => generalItems.filter((it) => (it.category[0] ?? "") === c && it.retrieval_level === d).length,
    ),
  );

  // 25観点カバレッジ
  const aspectCounts = new Map<string, number>();
  for (const it of generalItems) for (const a of it.aspect) aspectCounts.set(a, (aspectCounts.get(a) ?? 0) + 1);
  const min = labels.targets.aspect_min;

  // ドキュメント別
  const docCounts = countBy(generalItems, (it) => it.rationale[0]?.doc_id);
  const docs = [...docCounts.keys()].sort();

  return (
    <div>
      <div className="card">
        <h3>カテゴリ × 検索難易度</h3>
        <p className="caption">赤味のセル = 0件 (Phase 2 の補充対象)。色の濃さ = 件数。</p>
        <Heatmap xLabels={xLabels} yLabels={yLabels} values={values} flagZero />
      </div>

      <div className="card">
        <h3>25観点カバレッジ (目標: 各観点 最低 {min} 問)</h3>
        <p className="caption">赤い刻みが目標下限。バーの色はカテゴリ。</p>
        <Legend
          keys={labels.category_order.map((c) => ({
            label: labels.category_labels[c] ?? c,
            color: CATEGORY_COLORS[c] ?? "var(--cat-1)",
          }))}
        />
        {labels.category_order.map((cat) => {
          const aspectsInCat = labels.all_aspects.filter((a) => labels.aspect_to_category[a] === cat);
          const catMax = Math.max(min + 1, ...aspectsInCat.map((a) => aspectCounts.get(a) ?? 0));
          return (
            <div key={cat} style={{ marginBottom: 14 }}>
              <h4 style={{ margin: "8px 0 6px", fontSize: 13 }}>
                {labels.category_labels[cat] ?? cat}
                <span style={{ color: "var(--muted)", fontWeight: 400, marginLeft: 8, fontSize: 12 }}>
                  目標達成 {aspectsInCat.filter((a) => (aspectCounts.get(a) ?? 0) >= min).length} / {aspectsInCat.length} 観点
                </span>
              </h4>
              <DistBar
                rows={aspectsInCat.map((a) => ({
                  key: a,
                  label: labels.aspect_labels[a] ?? a,
                  count: aspectCounts.get(a) ?? 0,
                  color: CATEGORY_COLORS[cat] ?? "var(--cat-1)",
                  target: min / catMax,
                  hint: labels.aspect_descriptions[a],
                }))}
                total={catMax}
              />
            </div>
          );
        })}
        <p className="caption" style={{ marginTop: 4 }}>
          ※ バーの長さはカテゴリ内の最大件数を基準にした相対値。赤い刻みが「各観点 {min} 問」の位置。
        </p>
      </div>

      <div className="card">
        <h3>ドキュメント別件数</h3>
        <p className="caption">根拠 (rationale) の先頭ドキュメントで集計。</p>
        <DistBar
          rows={docs.map((d) => ({
            key: d,
            label: d,
            count: docCounts.get(d) ?? 0,
            color: "var(--cat-1)",
          }))}
          total={generalItems.length}
        />
      </div>
    </div>
  );
}
