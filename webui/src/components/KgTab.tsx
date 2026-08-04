import { LLM_KNOWLEDGE_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { isKg } from "../util";
import { DistBar, Heatmap, Legend, StatTile } from "./charts";

export default function KgTab({ items, labels }: { items: QAItem[]; labels: Labels }) {
  const kgItems = items.filter(isKg);
  if (kgItems.length === 0) {
    return (
      <div className="empty-note">
        KG-PoC トラックの QA が見つかりません。<span className="mono">uv run rageval generate --track kg_poc</span> で生成してください。
      </div>
    );
  }

  const nProbed = kgItems.filter((it) => it.llm_knowledge != null).length;
  const nUnknown = kgItems.filter((it) => it.llm_knowledge === "unknown").length;

  const xLabels = labels.all_kg_novelty.map((n) => labels.kg_novelty_labels[n] ?? n);
  const yLabels = labels.all_kg_query_types.map((q) => labels.kg_query_type_labels[q] ?? q);
  const values = labels.all_kg_query_types.map((qt) =>
    labels.all_kg_novelty.map(
      (nov) => kgItems.filter((it) => it.kg_query_type === qt && it.kg_novelty === nov).length,
    ),
  );
  const unknownValues = labels.all_kg_query_types.map((qt) =>
    labels.all_kg_novelty.map(
      (nov) =>
        kgItems.filter(
          (it) => it.kg_query_type === qt && it.kg_novelty === nov && it.llm_knowledge === "unknown",
        ).length,
    ),
  );

  return (
    <div>
      <div className="tile-row">
        <StatTile label="KG-PoC 総数" value={kgItems.length} />
        <StatTile label="プロービング済" value={`${nProbed}/${kgItems.length}`} />
        <StatTile
          label="LLM未知 (RAG必須)"
          value={nUnknown}
          delta="KG-RAG の主戦場"
          tone={nUnknown > 0 ? "good" : "plain"}
        />
      </div>

      {nProbed === 0 && (
        <div className="banner info">
          <span className="mono">uv run rageval probe --in &lt;jsonl&gt;</span> を実行すると LLM既知性 (AXIS-3) が判定されます。
        </div>
      )}

      <div className="card">
        <h3>クエリ型 × 未知性 マトリクス</h3>
        <p className="caption">セルの数字 = QA件数。色の濃さ = 件数。ホバーで「LLM未知」の内数を表示。</p>
        <Heatmap
          xLabels={xLabels}
          yLabels={yLabels}
          values={values}
          cellTip={(xi, yi, v) => (
            <div>
              {yLabels[yi]} × {xLabels[xi]}: <b>{v}件</b>
              <div>うち LLM未知: {unknownValues[yi][xi]}件</div>
            </div>
          )}
        />
      </div>

      {nProbed > 0 && (
        <div className="card">
          <h3>LLM既知性内訳 (プロービング結果)</h3>
          <p className="caption">「未知」が多いほど、KG-RAG 検証用データとして価値が高い (RAG 無しで答えられないため)。</p>
          <Legend
            keys={["unknown", "known"].map((k) => ({
              label: labels.llm_knowledge_labels[k] ?? k,
              color: LLM_KNOWLEDGE_COLORS[k],
            }))}
          />
          <DistBar
            rows={["unknown", "known"].map((k) => ({
              key: k,
              label: labels.llm_knowledge_labels[k] ?? k,
              count: kgItems.filter((it) => it.llm_knowledge === k).length,
              color: LLM_KNOWLEDGE_COLORS[k],
            }))}
            total={nProbed}
          />
        </div>
      )}

      <div className="card">
        <h3>タグの定義</h3>
        <div className="grid-2">
          <div>
            <h4 style={{ fontSize: 13, margin: "4px 0" }}>クエリ型 (AXIS-1)</h4>
            {labels.all_kg_query_types.map((qt) => (
              <p key={qt} style={{ fontSize: 12.5, margin: "4px 0" }}>
                <b>{labels.kg_query_type_labels[qt] ?? qt}</b> — {labels.kg_query_type_descriptions[qt]}
              </p>
            ))}
          </div>
          <div>
            <h4 style={{ fontSize: 13, margin: "4px 0" }}>未知性 (AXIS-2)</h4>
            {labels.all_kg_novelty.map((nov) => (
              <p key={nov} style={{ fontSize: 12.5, margin: "4px 0" }}>
                <b>{labels.kg_novelty_labels[nov] ?? nov}</b> — {labels.kg_novelty_descriptions[nov]}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
