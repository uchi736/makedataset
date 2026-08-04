import { DIFFICULTY_MATCH_COLORS, LEAKAGE_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { DistBar, Histogram, Legend } from "./charts";

export default function QualityTab({ items, labels }: { items: QAItem[]; labels: Labels }) {
  const scored = items.filter((it) => it.filter_scores.answerability != null);
  if (scored.length === 0) {
    return (
      <div className="empty-note">
        filter_scores がまだありません。<span className="mono">uv run rageval filter</span> を実行してください。
      </div>
    );
  }

  const t = labels.targets;
  const answerability = items.map((it) => it.filter_scores.answerability).filter((v): v is number => v != null);
  const grounding = items.map((it) => it.filter_scores.grounding).filter((v): v is number => v != null);
  const uniqueness = items.map((it) => it.filter_scores.uniqueness).filter((v): v is number => v != null);

  // 棄却理由の内訳 (stats_app と同じ規則)
  const reasons = new Map<string, number>();
  const bump = (k: string) => reasons.set(k, (reasons.get(k) ?? 0) + 1);
  for (const it of items) {
    const fs = it.filter_scores;
    if (fs.answerability != null && fs.answerability < t.answerability_pass) bump(`answerability < ${t.answerability_pass}`);
    if (fs.leakage === "fail") bump("leakage = fail (質問に答えが漏れている)");
    if (fs.grounding != null && fs.grounding < t.grounding_pass) bump(`grounding < ${t.grounding_pass}`);
    if (fs.uniqueness != null && fs.uniqueness < t.uniqueness_pass) bump(`uniqueness < ${t.uniqueness_pass.toFixed(2)} (重複疑い)`);
  }
  const reasonRows = [...reasons.entries()].sort((a, b) => b[1] - a[1]);

  const leak = items.filter((it) => it.filter_scores.leakage != null);
  const dm = items.filter((it) => it.filter_scores.difficulty_match != null);

  return (
    <div>
      <div className="grid-3">
        <div className="card">
          <h3>答えやすさ (answerability)</h3>
          <p className="caption">根拠だけで一意に答えられるか。しきい値 ≥ {t.answerability_pass} (点線)。</p>
          <Histogram values={answerability} threshold={t.answerability_pass} domain={[1, 5]} binWidth={0.5} />
        </div>
        <div className="card">
          <h3>根拠妥当性 (grounding)</h3>
          <p className="caption">根拠が回答を支えているか。しきい値 ≥ {t.grounding_pass} (点線)。</p>
          <Histogram values={grounding} threshold={t.grounding_pass} domain={[1, 5]} binWidth={0.5} />
        </div>
        <div className="card">
          <h3>独自性 (uniqueness)</h3>
          <p className="caption">既出との違い。しきい値 ≥ {t.uniqueness_pass.toFixed(2)} (点線)。1.0 = 完全独自。</p>
          <Histogram values={uniqueness} threshold={t.uniqueness_pass} domain={[0, 1]} binWidth={0.05} />
        </div>
      </div>

      <div className="card">
        <h3>棄却理由内訳</h3>
        <p className="caption">1件の QA が複数の理由に該当することもある (延べ数)。</p>
        {reasonRows.length === 0 ? (
          <div className="banner ok">棄却対象なし。全件しきい値クリア。</div>
        ) : (
          <DistBar
            rows={reasonRows.map(([r, c]) => ({
              key: r,
              label: r,
              count: c,
              color: "var(--status-critical)",
            }))}
            total={items.length}
          />
        )}
      </div>

      <div className="grid-2">
        {leak.length > 0 && (
          <div className="card">
            <h3>リーク判定 (leakage)</h3>
            <p className="caption">質問文に答えそのものが書かれていないか。</p>
            <Legend keys={["pass", "fail"].map((v) => ({ label: labels.leakage_labels[v] ?? v, color: LEAKAGE_COLORS[v] }))} />
            <DistBar
              rows={["pass", "fail"].map((v) => ({
                key: v,
                label: labels.leakage_labels[v] ?? v,
                count: leak.filter((it) => it.filter_scores.leakage === v).length,
                color: LEAKAGE_COLORS[v],
              }))}
              total={leak.length}
            />
          </div>
        )}
        {dm.length > 0 && (
          <div className="card">
            <h3>難易度整合 (difficulty_match)</h3>
            <p className="caption">宣言した難易度と実態が合っているか (判定LLMの見立て)。</p>
            <Legend
              keys={["aligned", "too_easy", "too_hard"].map((v) => ({
                label: labels.difficulty_match_labels[v] ?? v,
                color: DIFFICULTY_MATCH_COLORS[v],
              }))}
            />
            <DistBar
              rows={["aligned", "too_easy", "too_hard"].map((v) => ({
                key: v,
                label: labels.difficulty_match_labels[v] ?? v,
                count: dm.filter((it) => it.filter_scores.difficulty_match === v).length,
                color: DIFFICULTY_MATCH_COLORS[v],
              }))}
              total={dm.length}
            />
          </div>
        )}
      </div>
    </div>
  );
}
