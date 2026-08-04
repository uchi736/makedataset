import { useMemo, useState } from "react";
import { REVIEW_STATUS_COLORS } from "../colors";
import type { Labels, QAItem } from "../types";
import { fmtScore, isKg, itemKey, truncate } from "../util";
import DetailPanel from "./DetailPanel";

type SortKey = "question" | "tag" | "difficulty" | "score" | "rag" | "status";

interface Row {
  qa: QAItem;
  question: string;
  tag: string;
  difficulty: string;
  score: number;
  scoreText: string;
  rag: string;
  status: string;
}

export default function ListTab(props: {
  items: QAItem[];
  labels: Labels;
  onGoReview: (qaId: string) => void;
}) {
  const { items, labels, onGoReview } = props;
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("question");
  const [sortDir, setSortDir] = useState<1 | -1>(1);
  const [selected, setSelected] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<Set<string>>(
    new Set(["pending", "accepted", "edited", "rejected"]),
  );

  const rows: Row[] = useMemo(() => {
    return items.map((qa) => {
      const kg = isKg(qa);
      const tag = kg
        ? labels.kg_query_type_labels[qa.kg_query_type ?? ""] ?? qa.kg_query_type ?? "—"
        : labels.category_labels[qa.category[0] ?? ""] ?? qa.category[0] ?? "—";
      const a = qa.filter_scores.answerability;
      const g = qa.filter_scores.grounding;
      const leak = qa.filter_scores.leakage === "fail";
      const scoreText =
        a == null && g == null ? "—" : `${fmtScore(a)}/${fmtScore(g)}${leak ? " ⚠漏れ" : ""}`;
      const rag = qa.rag_verification
        ? labels.rag_answer_match_labels[qa.rag_verification.answer_match] ?? qa.rag_verification.answer_match
        : "—";
      return {
        qa,
        question: qa.question,
        tag,
        difficulty: `${labels.difficulty_labels[qa.retrieval_level] ?? "?"} / ${labels.difficulty_labels[qa.answer_level] ?? "?"}`,
        score: a ?? -1,
        scoreText,
        rag,
        status: qa.review_status,
      };
    });
  }, [items, labels]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const byStatus = rows.filter((r) => statusFilter.has(r.status));
    const base = q
      ? byStatus.filter(
          (r) =>
            r.question.toLowerCase().includes(q) ||
            r.qa.answer.toLowerCase().includes(q) ||
            r.qa.qa_id.toLowerCase().includes(q),
        )
      : byStatus;
    const sorted = [...base].sort((x, y) => {
      const kx = sortKey === "score" ? x.score : String(x[sortKey]);
      const ky = sortKey === "score" ? y.score : String(y[sortKey]);
      if (kx < ky) return -1 * sortDir;
      if (kx > ky) return 1 * sortDir;
      return 0;
    });
    return sorted;
  }, [rows, query, sortKey, sortDir, statusFilter]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir((d) => (d === 1 ? -1 : 1));
    else {
      setSortKey(key);
      setSortDir(1);
    }
  };

  const arrow = (key: SortKey) => (sortKey === key ? (sortDir === 1 ? " ↑" : " ↓") : "");
  const selectedQa = selected ? items.find((it) => itemKey(it) === selected) ?? null : null;

  return (
    <div>
      <div style={{ display: "flex", gap: 10, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="text"
          placeholder="質問・回答・qa_id を検索…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ maxWidth: 380 }}
        />
        <div className="chip-row">
          {(["pending", "accepted", "edited", "rejected"] as const).map((s) => (
            <button
              key={s}
              className={`filter-chip ${statusFilter.has(s) ? "on" : ""}`}
              onClick={() =>
                setStatusFilter((cur) => {
                  const next = new Set(cur);
                  if (next.has(s)) next.delete(s);
                  else next.add(s);
                  return next;
                })
              }
            >
              <span
                style={{
                  display: "inline-block", width: 7, height: 7, borderRadius: "50%",
                  background: REVIEW_STATUS_COLORS[s], marginRight: 4,
                }}
              />
              {labels.review_status_labels[s] ?? s}
            </button>
          ))}
        </div>
        <span style={{ fontSize: 12.5, color: "var(--muted)" }}>
          {filtered.length} / {items.length} 件 — 行クリックで詳細
        </span>
      </div>

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th onClick={() => toggleSort("status")}>状態{arrow("status")}</th>
              <th onClick={() => toggleSort("question")}>質問{arrow("question")}</th>
              <th onClick={() => toggleSort("tag")}>タグ{arrow("tag")}</th>
              <th onClick={() => toggleSort("difficulty")} title="検索難易度 / 回答難易度">難易度 (検索/回答){arrow("difficulty")}</th>
              <th onClick={() => toggleSort("score")} title="answerability / grounding (各5点満点)">判定 (ans/grd){arrow("score")}</th>
              <th onClick={() => toggleSort("rag")} title="vector RAG 実走の照合結果">RAG検証{arrow("rag")}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr
                key={itemKey(r.qa)}
                className={selected === itemKey(r.qa) ? "selected" : ""}
                onClick={() => setSelected(itemKey(r.qa))}
              >
                <td>
                  <span className="badge" style={{ fontSize: 11 }}>
                    <span className="dot" style={{ background: REVIEW_STATUS_COLORS[r.status] ?? "var(--muted)" }} />
                    {labels.review_status_labels[r.status] ?? r.status}
                  </span>
                </td>
                <td>{truncate(r.question, 90)}</td>
                <td style={{ whiteSpace: "nowrap" }}>{r.tag}</td>
                <td className="num" style={{ whiteSpace: "nowrap" }}>{r.difficulty}</td>
                <td className="num" style={{ whiteSpace: "nowrap" }}>{r.scoreText}</td>
                <td style={{ whiteSpace: "nowrap" }}>{r.rag}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && <div className="empty-note">該当する QA がありません。</div>}
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
