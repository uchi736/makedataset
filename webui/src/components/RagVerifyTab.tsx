import { useMemo, useState } from "react";
import { RAG_MATCH_COLORS, RETRIEVAL_HIT_COLORS } from "../colors";
import type { Labels, QAItem, RagAnswerMatch, RetrievalMode } from "../types";
import { ALL_RAG_ANSWER_MATCH, ALL_RETRIEVAL_MODE, enumOrder, isKgCandidate } from "../types";
import { countBy, itemKey, pct, truncate } from "../util";
import DetailPanel from "./DetailPanel";
import { DistBar, Legend, StackRow, StatTile } from "./charts";

/** 検索方式の表示名。方式を混ぜて読ませないための注記なので短く出す。 */
const MODE_LABELS: Record<RetrievalMode, string> = {
  dense: "密ベクトルのみ",
  hybrid: "密+BM25",
  hybrid_rerank: "密+BM25→リランク",
};

export default function RagVerifyTab(props: {
  items: QAItem[];
  labels: Labels;
  onGoReview: (qaId: string) => void;
}) {
  const { items, labels, onGoReview } = props;
  const [selected, setSelected] = useState<string | null>(null);
  const verified = items.filter((it) => it.rag_verification != null);
  const verdicts = useMemo(
    () => enumOrder<RagAnswerMatch>(labels.rag_answer_match_labels, ALL_RAG_ANSWER_MATCH),
    [labels],
  );

  const aspectAgg = useMemo(() => {
    const agg = new Map<string, { total: number; counts: Record<string, number> }>();
    const zeroCounts = () => Object.fromEntries(verdicts.map((v) => [v, 0])) as Record<string, number>;
    for (const it of verified) {
      for (const a of it.aspect) {
        const entry = agg.get(a) ?? { total: 0, counts: zeroCounts() };
        entry.total += 1;
        const v = it.rag_verification!.answer_match;
        entry.counts[v] = (entry.counts[v] ?? 0) + 1;
        agg.set(a, entry);
      }
    }
    return [...agg.entries()].sort(
      (x, y) => (y[1].counts.no_match ?? 0) / y[1].total - (x[1].counts.no_match ?? 0) / x[1].total,
    );
  }, [verified, verdicts]);

  if (verified.length === 0) {
    return (
      <div className="empty-note">
        このデータには rag_verification がありません。<span className="mono">uv run rageval rag-verify</span> を先に実行してください。
      </div>
    );
  }

  const nBy = (v: string) => verified.filter((it) => it.rag_verification?.answer_match === v).length;
  // 配信に新しい判定値が増えても棒と凡例が消えないよう、色には既定を持たせる。
  const verdictColor = (v: string) => RAG_MATCH_COLORS[v] ?? "var(--muted)";
  const hitItems = verified.filter((it) => it.rag_verification?.retrieval_hit_chunk != null || it.rag_verification?.retrieval_hit != null);
  const chunkHit = (it: QAItem) => it.rag_verification?.retrieval_hit_chunk ?? it.rag_verification?.retrieval_hit ?? false;
  // 根拠の逐語命中率。配信は Pydantic を通るので、項目を持たない古い JSONL でも
  // 既定の 0.0 が実体化されて届く (?? では古い行を判別できない)。命中している
  // のに比率が 0 という組み合わせは比率が記録されていない行なので、そこだけ
  // 2値から復元する。
  const hitRatio = (it: QAItem) => {
    const ratio = it.rag_verification?.retrieval_hit_ratio ?? 0;
    return ratio > 0 ? ratio : (chunkHit(it) ? 1 : 0);
  };

  // 検索方式は行ごとに違いうる (dense → hybrid → hybrid_rerank と切り替えてきた)。
  // 方式で命中の基準が変わるので、混在は分母をまとめる前に画面へ出す。
  const modeCounts = countBy(verified, (it) => it.rag_verification?.retrieval_mode ?? "dense");
  const modeRows = ALL_RETRIEVAL_MODE
    .filter((m) => (modeCounts.get(m) ?? 0) > 0)
    .map((m) => ({ mode: m, count: modeCounts.get(m) ?? 0 }));

  // 照合不能 (judge_ok=false) は「RAG が解けなかった」ではないので KG候補から外す。
  const judgeNg = verified.filter((it) => it.rag_verification?.judge_ok === false).length;
  const answerabilityPass = labels.targets.answerability_pass;
  const candidates = items.filter((it) => isKgCandidate(it, answerabilityPass));

  const selectedQa = selected ? items.find((it) => itemKey(it) === selected) ?? null : null;

  return (
    <div>
      <div className="tile-row">
        <StatTile label="検証済QA" value={verified.length} />
        <StatTile label="一致 (vector RAGが正答)" value={nBy("match")} />
        <StatTile label="部分一致" value={nBy("partial")} />
        <StatTile
          label="KG候補 (不一致・照合成立)"
          value={candidates.length}
          delta={`不一致 ${nBy("no_match")}件のうち`}
          tone={candidates.length > 0 ? "bad" : "plain"}
        />
        <StatTile
          label="照合不能 (候補から除外)"
          value={judgeNg}
          delta={judgeNg > 0 ? "判定失敗・回答生成の障害" : undefined}
          tone={judgeNg > 0 ? "bad" : "plain"}
        />
      </div>

      <div className="banner info">
        検索方式: {modeRows.map((r) => `${MODE_LABELS[r.mode]} ${r.count}件`).join(" / ")}
        {modeRows.length > 1 && " — 方式ごとに命中の基準が違うため、下の命中率はまとめて読まないこと。"}
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>一致判定の分布</h3>
          <p className="caption">
            不一致が多いほど vector RAG では拾えていない問い (KG候補)。
            {judgeNg > 0 && ` 照合不能 ${judgeNg}件はこの分布では不一致に見えるが、KG候補には数えていない。`}
          </p>
          <Legend keys={verdicts.map((v) => ({ label: labels.rag_answer_match_labels[v] ?? v, color: verdictColor(v) }))} />
          <DistBar
            rows={verdicts.map((v) => ({
              key: v,
              label: labels.rag_answer_match_labels[v] ?? v,
              count: nBy(v),
              color: verdictColor(v),
            }))}
            total={verified.length}
          />
        </div>
        <div className="card">
          <h3>検索命中 (チャンク粒度)</h3>
          <p className="caption">
            根拠本文を逐語で含むチャンクが top-k に入ったか。根拠が複数ある問いは
            一部だけ届く場合があるので、全命中と部分命中を分けて出す。
          </p>
          <Legend
            keys={[
              { label: "全命中", color: RETRIEVAL_HIT_COLORS.true },
              { label: "部分命中", color: RAG_MATCH_COLORS.partial },
              { label: "外れ", color: RETRIEVAL_HIT_COLORS.false },
            ]}
          />
          <DistBar
            rows={[
              { key: "hit", label: "全命中", count: hitItems.filter((it) => hitRatio(it) >= 1).length, color: RETRIEVAL_HIT_COLORS.true },
              {
                key: "partial",
                label: "部分命中",
                count: hitItems.filter((it) => hitRatio(it) > 0 && hitRatio(it) < 1).length,
                color: RAG_MATCH_COLORS.partial,
                hint: "根拠の一部だけが top-k に入った (aggregation の実測はここに出る)",
              },
              { key: "miss", label: "外れ", count: hitItems.filter((it) => hitRatio(it) <= 0).length, color: RETRIEVAL_HIT_COLORS.false },
            ]}
            total={hitItems.length}
          />
        </div>
      </div>

      {aspectAgg.length > 0 && (
        <div className="card">
          <h3>観点別の判定割合</h3>
          <p className="caption">不一致率の高い観点が上 (= ベクトル検索が苦手な観点)。</p>
          <Legend keys={verdicts.map((v) => ({ label: labels.rag_answer_match_labels[v] ?? v, color: verdictColor(v) }))} />
          {aspectAgg.map(([aspect, agg]) => (
            <StackRow
              key={aspect}
              name={labels.aspect_labels[aspect] ?? aspect}
              total={agg.total}
              segments={verdicts.map((v) => ({
                key: v,
                label: labels.rag_answer_match_labels[v] ?? v,
                count: agg.counts[v] ?? 0,
                color: verdictColor(v),
              }))}
            />
          ))}
        </div>
      )}

      <div className="card">
        <h3>KG候補リスト (vector RAG が落とした問い)</h3>
        <p className="caption">
          判定LLMが答えあり (答えやすさ ≥{answerabilityPass}) または未判定で、vector RAG の回答が
          不一致だった問い。判定そのものが失敗した行 (照合不能) は障害由来なので含めない。行クリックで詳細。
        </p>
        {candidates.length === 0 ? (
          <div className="banner ok">KG候補に該当する問いはありません。vector RAG が十分にカバーしています。</div>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>QA-ID</th>
                  <th>観点 / クエリ型</th>
                  <th>質問</th>
                  <th>正答</th>
                  <th>RAGの回答</th>
                  <th>検索方式</th>
                  <th>検索命中</th>
                  <th>答えやすさ</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((it) => (
                  <tr key={itemKey(it)} onClick={() => setSelected(itemKey(it))}>
                    <td className="mono">{it.qa_id}</td>
                    <td>
                      {it.aspect.length
                        ? it.aspect.map((a) => labels.aspect_labels[a] ?? a).join(", ")
                        : labels.kg_query_type_labels[it.kg_query_type ?? ""] ?? "—"}
                    </td>
                    <td>{truncate(it.question, 60)}</td>
                    <td>{truncate(it.answer, 60)}</td>
                    <td>{truncate(it.rag_verification?.rag_answer || "(回答不能)", 60)}</td>
                    <td>{MODE_LABELS[it.rag_verification?.retrieval_mode ?? "dense"]}</td>
                    <td>{hitRatio(it) >= 1 ? "◯" : hitRatio(it) > 0 ? `△ ${pct(hitRatio(it))}` : "✗"}</td>
                    <td className="num">{it.filter_scores.answerability?.toFixed(1) ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedQa && (
        <DetailPanel
          qa={selectedQa}
          labels={labels}
          onClose={() => setSelected(null)}
          onGoReview={(id) => {
            setSelected(null);
            onGoReview(id);
          }}
        />
      )}
    </div>
  );
}
