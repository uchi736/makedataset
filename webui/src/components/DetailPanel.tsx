import { REVIEW_STATUS_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { fmtScore, isKg } from "../util";

export function QAChips({ qa, labels }: { qa: QAItem; labels: Labels }) {
  const kg = isKg(qa);
  return (
    <div style={{ margin: "4px 0" }}>
      {kg ? (
        <>
          <span className="chip">クエリ型: <b>{labels.kg_query_type_labels[qa.kg_query_type ?? ""] ?? qa.kg_query_type ?? "(未指定)"}</b></span>
          <span className="chip">未知性: <b>{labels.kg_novelty_labels[qa.kg_novelty ?? ""] ?? qa.kg_novelty ?? "(未指定)"}</b></span>
          <span className="chip">
            LLM既知性: <b>{qa.llm_knowledge ? labels.llm_knowledge_labels[qa.llm_knowledge] ?? qa.llm_knowledge : "未付与"}</b>
          </span>
        </>
      ) : (
        <>
          <span className="chip">分類: <b>{qa.category.map((c) => labels.category_labels[c] ?? c).join(", ") || "(なし)"}</b></span>
          <span className="chip">観点: <b>{qa.aspect.map((a) => labels.aspect_labels[a] ?? a).join(", ") || "(なし)"}</b></span>
        </>
      )}
      <span className="chip">検索: <b>{labels.difficulty_labels[qa.retrieval_level] ?? qa.retrieval_level}</b></span>
      <span className="chip">回答: <b>{labels.difficulty_labels[qa.answer_level] ?? qa.answer_level}</b></span>
    </div>
  );
}

export function StatusBadge({ status, labels }: { status: string; labels: Labels }) {
  return (
    <span className="badge">
      <span className="dot" style={{ background: REVIEW_STATUS_COLORS[status] ?? "var(--muted)" }} />
      {labels.review_status_labels[status] ?? status}
    </span>
  );
}

/** 診断タグ (真のものだけ) を1行ずつまとめる。 */
export function DiagTags({ qa, labels }: { qa: QAItem; labels: Labels }) {
  const rows: { axis: string; tags: string[] }[] = [];
  const pairs: [string, Record<string, unknown>][] = [
    ["reasoning_complexity", qa.reasoning_complexity],
    ["retrieval_difficulty", qa.retrieval_difficulty],
    ["source_structure", qa.source_structure],
  ];
  const axisNames: Record<string, string> = {
    reasoning_complexity: "推論",
    retrieval_difficulty: "検索",
    source_structure: "構造",
  };
  for (const [axis, dump] of pairs) {
    const dict = labels.diag_labels[axis] ?? {};
    const tags = Object.entries(dump)
      .filter(([k, v]) => v === true && dict[k])
      .map(([k]) => dict[k]);
    if (tags.length) rows.push({ axis: axisNames[axis], tags });
  }
  const ot = qa.reasoning_complexity["output_type"];
  const ev = qa.explainability?.evidence_strictness;
  return (
    <div style={{ fontSize: 12.5 }}>
      {rows.map((r) => (
        <div key={r.axis}>
          <span style={{ color: "var(--muted)" }}>{r.axis}: </span>
          {r.tags.map((tg) => <span className="chip" key={tg}>{tg}</span>)}
        </div>
      ))}
      {typeof ot === "string" && ot !== "none" && (
        <div><span style={{ color: "var(--muted)" }}>出力タイプ: </span><span className="chip">{labels.output_type_labels[ot] ?? ot}</span></div>
      )}
      {ev && (
        <div><span style={{ color: "var(--muted)" }}>根拠厳密性: </span><span className="chip">{labels.evidence_labels[ev] ?? ev}</span></div>
      )}
    </div>
  );
}

export default function DetailPanel(props: {
  qa: QAItem;
  labels: Labels;
  onClose: () => void;
  onGoReview: (qaId: string) => void;
}) {
  const { qa, labels, onClose, onGoReview } = props;
  const fs = qa.filter_scores;
  const rv = qa.rag_verification;
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" role="dialog" aria-label="QA詳細">
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
          <StatusBadge status={qa.review_status} labels={labels} />
          <span className="mono" style={{ color: "var(--muted)" }}>{qa.qa_id}</span>
          <span className="chip">{qa._source}</span>
          <span style={{ flex: 1 }} />
          <button className="btn primary" onClick={() => onGoReview(qa.qa_id)}>この問いをレビュー</button>
          <button className="icon-btn" onClick={onClose} aria-label="閉じる">✕</button>
        </div>

        <h2>質問</h2>
        <div className="qa-question">{qa.question}</div>
        <h2>回答</h2>
        <div className="qa-answer">{qa.answer}</div>
        <QAChips qa={qa} labels={labels} />

        {qa.difficulty_rationale && (
          <p style={{ fontSize: 12.5, color: "var(--ink-2)" }}>
            <b>難易度根拠:</b> {qa.difficulty_rationale}
          </p>
        )}

        <div className="card" style={{ marginTop: 12 }}>
          <h3>判定LLMスコア</h3>
          <div className="tile-row" style={{ marginBottom: 0 }}>
            <div className="stat-tile"><div className="label">答えやすさ</div><div className="value">{fmtScore(fs.answerability)}</div></div>
            <div className="stat-tile"><div className="label">根拠妥当性</div><div className="value">{fmtScore(fs.grounding)}</div></div>
            <div className="stat-tile"><div className="label">独自性</div><div className="value">{fmtScore(fs.uniqueness, 2)}</div></div>
            <div className="stat-tile"><div className="label">リーク判定</div><div className="value" style={{ fontSize: 18 }}>{fs.leakage ? labels.leakage_labels[fs.leakage] ?? fs.leakage : "—"}</div></div>
            <div className="stat-tile"><div className="label">逐語一致率</div><div className="value">{fs.rationale_grounded != null ? `${Math.round(fs.rationale_grounded * 100)}%` : "—"}</div></div>
          </div>
        </div>

        <div className="card">
          <h3>根拠 (生成時の引用)</h3>
          {qa.rationale.map((r, i) => (
            <div className="rationale-line" key={i}>
              <span className="mono" style={{ color: "var(--muted)" }}>{r.doc_id}{r.page != null ? ` p.${r.page}` : ""}</span> — {r.text}
            </div>
          ))}
        </div>

        {rv && (
          <div className="card">
            <h3>RAG検証</h3>
            <p style={{ fontSize: 13 }}>
              判定: <b>{labels.rag_answer_match_labels[rv.answer_match] ?? rv.answer_match}</b>
              　/　文書命中: <b>{rv.retrieval_hit_doc ?? rv.retrieval_hit ? "あり" : "なし"}</b>
              　/　チャンク命中: <b>{rv.retrieval_hit_chunk ? "あり" : "なし"}</b>
              　/　top-k: {rv.top_k}
            </p>
            <p style={{ fontSize: 12.5, color: "var(--ink-2)", whiteSpace: "pre-wrap" }}>
              <b>RAGの回答:</b> {rv.rag_answer || "(回答不能)"}
            </p>
            <p className="caption">rag: {rv.rag_model} / judge: {rv.judge_model}</p>
          </div>
        )}

        <div className="card">
          <h3>診断タグ</h3>
          <DiagTags qa={qa} labels={labels} />
        </div>

        <details className="raw-json">
          <summary>生 JSON</summary>
          <pre>{JSON.stringify(qa, null, 2)}</pre>
        </details>
      </div>
    </>
  );
}
