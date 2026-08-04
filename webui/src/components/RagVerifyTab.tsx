import { useMemo, useState } from "react";
import { RAG_MATCH_COLORS, RETRIEVAL_HIT_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { itemKey, truncate } from "../util";
import DetailPanel from "./DetailPanel";
import { DistBar, Legend, StackRow, StatTile } from "./charts";

const VERDICTS = ["match", "partial", "no_match"] as const;

export default function RagVerifyTab(props: {
  items: QAItem[];
  labels: Labels;
  onGoReview: (qaId: string) => void;
}) {
  const { items, labels, onGoReview } = props;
  const [selected, setSelected] = useState<string | null>(null);
  const verified = items.filter((it) => it.rag_verification != null);

  const aspectAgg = useMemo(() => {
    const agg = new Map<string, { total: number; counts: Record<string, number> }>();
    for (const it of verified) {
      for (const a of it.aspect) {
        const entry = agg.get(a) ?? { total: 0, counts: { match: 0, partial: 0, no_match: 0 } };
        entry.total += 1;
        entry.counts[it.rag_verification!.answer_match] += 1;
        agg.set(a, entry);
      }
    }
    return [...agg.entries()].sort(
      (x, y) => y[1].counts.no_match / y[1].total - x[1].counts.no_match / x[1].total,
    );
  }, [verified]);

  if (verified.length === 0) {
    return (
      <div className="empty-note">
        このデータには rag_verification がありません。<span className="mono">uv run rageval rag-verify</span> を先に実行してください。
      </div>
    );
  }

  const nBy = (v: string) => verified.filter((it) => it.rag_verification?.answer_match === v).length;
  const hitItems = verified.filter((it) => it.rag_verification?.retrieval_hit_chunk != null || it.rag_verification?.retrieval_hit != null);
  const chunkHit = (it: QAItem) => it.rag_verification?.retrieval_hit_chunk ?? it.rag_verification?.retrieval_hit ?? false;

  // KG候補: judge が答えありと認めた (>=4) または未判定、かつ vector RAG が落とした問い
  const candidates = items.filter(
    (it) =>
      (it.filter_scores.answerability == null || it.filter_scores.answerability >= 4) &&
      it.rag_verification?.answer_match === "no_match",
  );

  const selectedQa = selected ? items.find((it) => itemKey(it) === selected) ?? null : null;

  return (
    <div>
      <div className="tile-row">
        <StatTile label="検証済QA" value={verified.length} />
        <StatTile label="一致 (vector RAGが正答)" value={nBy("match")} />
        <StatTile label="部分一致" value={nBy("partial")} />
        <StatTile label="不一致 (KG候補)" value={nBy("no_match")} tone={nBy("no_match") > 0 ? "bad" : "plain"} />
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>一致判定の分布</h3>
          <p className="caption">不一致が多いほど vector RAG では拾えていない問い (KG候補)。</p>
          <Legend keys={VERDICTS.map((v) => ({ label: labels.rag_answer_match_labels[v] ?? v, color: RAG_MATCH_COLORS[v] }))} />
          <DistBar
            rows={VERDICTS.map((v) => ({
              key: v,
              label: labels.rag_answer_match_labels[v] ?? v,
              count: nBy(v),
              color: RAG_MATCH_COLORS[v],
            }))}
            total={verified.length}
          />
        </div>
        <div className="card">
          <h3>検索命中 (チャンク粒度)</h3>
          <p className="caption">根拠本文を逐語で含むチャンクが top-k に入ったか。外れが多いと検索段階で取りこぼしている。</p>
          <Legend keys={[{ label: "命中", color: RETRIEVAL_HIT_COLORS.true }, { label: "外れ", color: RETRIEVAL_HIT_COLORS.false }]} />
          <DistBar
            rows={[
              { key: "hit", label: "命中", count: hitItems.filter(chunkHit).length, color: RETRIEVAL_HIT_COLORS.true },
              { key: "miss", label: "外れ", count: hitItems.filter((it) => !chunkHit(it)).length, color: RETRIEVAL_HIT_COLORS.false },
            ]}
            total={hitItems.length}
          />
        </div>
      </div>

      {aspectAgg.length > 0 && (
        <div className="card">
          <h3>観点別の判定割合</h3>
          <p className="caption">不一致率の高い観点が上 (= ベクトル検索が苦手な観点)。</p>
          <Legend keys={VERDICTS.map((v) => ({ label: labels.rag_answer_match_labels[v] ?? v, color: RAG_MATCH_COLORS[v] }))} />
          {aspectAgg.map(([aspect, agg]) => (
            <StackRow
              key={aspect}
              name={labels.aspect_labels[aspect] ?? aspect}
              total={agg.total}
              segments={VERDICTS.map((v) => ({
                key: v,
                label: labels.rag_answer_match_labels[v] ?? v,
                count: agg.counts[v],
                color: RAG_MATCH_COLORS[v],
              }))}
            />
          ))}
        </div>
      )}

      <div className="card">
        <h3>KG候補リスト (vector RAG が落とした問い)</h3>
        <p className="caption">判定LLMが答えあり (≥4) または未判定で、vector RAG の回答が不一致だった問い。行クリックで詳細。</p>
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
                    <td>{chunkHit(it) ? "◯" : "✗"}</td>
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
