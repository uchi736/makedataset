import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { fetchEvidence, postReview } from "../api";
import { REVIEW_STATUS_COLORS } from "../colors";
import type { EvidenceResponse, Labels, QAItem, ReviewStatus } from "../types";
import { isKg, itemKey, pct, statusCounts, truncate } from "../util";
import { DiagTags, QAChips, StatusBadge } from "./DetailPanel";

function HighlightedText({ text, spans }: { text: string; spans: [number, number][] }) {
  if (!spans.length) return <>{text}</>;
  const parts: ReactNode[] = [];
  let cur = 0;
  spans.forEach(([s, e], i) => {
    if (s > cur) parts.push(<span key={`t${i}`}>{text.slice(cur, s)}</span>);
    parts.push(<mark key={`m${i}`}>{text.slice(s, e)}</mark>);
    cur = e;
  });
  if (cur < text.length) parts.push(<span key="tail">{text.slice(cur)}</span>);
  return <>{parts}</>;
}

/** チャンク本文。最初のハイライトが見える位置まで自動スクロールする。 */
function EvidenceText({ text, spans }: { text: string; spans: [number, number][] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const box = ref.current;
    const mark = box?.querySelector("mark");
    if (box && mark) {
      box.scrollTop = Math.max(0, (mark as HTMLElement).offsetTop - 48);
    }
  }, [text, spans]);
  return (
    <div className="evidence-text" ref={ref}>
      <HighlightedText text={text} spans={spans} />
    </div>
  );
}

export default function ReviewTab(props: {
  items: QAItem[];
  labels: Labels;
  datasetPath: string;
  chunksDir: string;
  reviewer: string;
  onItemUpdated: (item: QAItem) => void;
  jumpTo: string | null;
  onJumpConsumed: () => void;
}) {
  const { items, labels, datasetPath, chunksDir, reviewer, onItemUpdated, jumpTo, onJumpConsumed } = props;
  const [idx, setIdx] = useState(0);
  const [showPendingOnly, setShowPendingOnly] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [editQ, setEditQ] = useState("");
  const [editA, setEditA] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flags, setFlags] = useState<Map<string, Set<string>>>(new Map());
  const [evidence, setEvidence] = useState<Map<string, EvidenceResponse>>(new Map());
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // 表示対象 (未判定のみ表示のとき、判定した問いはリストから抜けて次が繰り上がる)
  const working = useMemo(
    () => (showPendingOnly ? items.filter((it) => it.review_status === "pending") : items),
    [items, showPendingOnly],
  );

  const total = working.length;
  const clamped = Math.max(0, Math.min(idx, total - 1));
  const qa: QAItem | undefined = working[clamped];

  // 一覧・詳細からのジャンプ。未判定のみ表示で隠れている問いなら表示を全件に戻す
  useEffect(() => {
    if (jumpTo == null) return;
    const i = working.findIndex((it) => it.qa_id === jumpTo);
    if (i >= 0) {
      setIdx(i);
      onJumpConsumed();
    } else if (items.some((it) => it.qa_id === jumpTo)) {
      setShowPendingOnly(false);
    } else {
      onJumpConsumed();
    }
  }, [jumpTo, working, items, onJumpConsumed]);

  // QA が変わったら編集モードを解除
  useEffect(() => {
    setEditMode(false);
    setError(null);
  }, [clamped, qa?.qa_id]);

  // 根拠チャンクの取得 (qa_id + 照合先 単位でキャッシュ)
  // 同一 qa_id でもファイルが違えば rationale が異なりうる (raw と filtered など)。
  // キャッシュキーにファイル名を含めないと、先に開いた方の根拠が表示され続ける。
  const evKey = qa ? `${chunksDir}::${itemKey(qa)}` : "";
  useEffect(() => {
    if (!qa || evidence.has(evKey)) return;
    let cancelled = false;
    setEvidenceLoading(true);
    fetchEvidence(chunksDir, qa.rationale)
      .then((res) => {
        if (cancelled) return;
        setEvidence((m) => new Map(m).set(evKey, res));
      })
      .catch((e: Error) => {
        if (!cancelled) setError(`根拠チャンクの取得に失敗: ${e.message}`);
      })
      .finally(() => {
        if (!cancelled) setEvidenceLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [qa, evKey, chunksDir, evidence]);

  const ev = evidence.get(evKey);

  // 合格時は本文をすっきり保ち、不一致や照合不能のときだけ根拠を自動展開する。
  useEffect(() => {
    setEvidenceOpen(ev ? !ev.all_match : false);
  }, [evKey, ev]);

  const gotoNextPending = useCallback(() => {
    if (!working.length) return;
    for (let step = 1; step <= working.length; step++) {
      const i = (clamped + step) % working.length;
      if (working[i].review_status === "pending") {
        setIdx(i);
        return;
      }
    }
  }, [working, clamped]);

  const mark = useCallback(
    async (status: ReviewStatus, question?: string, answer?: string) => {
      if (!qa || saving) return;
      setSaving(true);
      setError(null);
      try {
        const updated = await postReview({
          path: datasetPath,
          source: qa._source,
          qa_id: qa.qa_id,
          status,
          reviewer,
          question,
          answer,
        });
        onItemUpdated(updated);
        setEditMode(false);
        // 未判定のみ表示なら判定済みが自動で抜けて次が繰り上がる。全件表示なら1つ進む
        if (!showPendingOnly) setIdx((i) => Math.min(i + 1, total - 1));
      } catch (e) {
        setError(`保存に失敗しました: ${(e as Error).message}`);
      } finally {
        setSaving(false);
      }
    },
    [qa, saving, datasetPath, reviewer, onItemUpdated, showPendingOnly, total],
  );

  const startEdit = useCallback(() => {
    if (!qa) return;
    setEditQ(qa.question);
    setEditA(qa.answer);
    setEditMode(true);
  }, [qa]);

  // キーボードショートカット (入力中は無効)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      // 文字を打つ欄だけショートカット無効。チェックボックス等にフォーカスが
      // 残っていても A/R/N は効くようにする
      const isTextEntry =
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable ||
        (target.tagName === "INPUT" &&
          !["checkbox", "radio", "button"].includes((target as HTMLInputElement).type));
      if (isTextEntry) {
        if (editMode && e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          void mark("edited", editQ, editA);
        }
        if (editMode && e.key === "Escape") setEditMode(false);
        return;
      }
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, i - 1));
      else if (e.key === "ArrowRight") setIdx((i) => Math.min(total - 1, i + 1));
      else if (e.key === "a" || e.key === "A") void mark("accepted");
      else if (e.key === "r" || e.key === "R") void mark("rejected");
      else if (e.key === "e" || e.key === "E") startEdit();
      else if (e.key === "n" || e.key === "N") gotoNextPending();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [editMode, editQ, editA, mark, startEdit, total, gotoNextPending]);

  // 現在行を左リストの表示範囲に保つ
  useEffect(() => {
    const el = listRef.current?.querySelector(".review-item.active");
    el?.scrollIntoView({ block: "nearest" });
  }, [clamped]);

  const counts = useMemo(() => statusCounts(items), [items]);
  const allTotal = items.length;
  const done = allTotal - counts.pending;

  // ==================== 上部: 進捗と導線 ====================
  const headerStrip = (
    <div className="review-strip">
      <div className="strip-progress">
        <b>{done}</b> / {allTotal} 問 査読済み ({allTotal ? pct(done / allTotal) : "—"})
        <div className="progress-track" style={{ width: 180 }}>
          <div className="progress-fill" style={{ width: pct(allTotal ? done / allTotal : 0) }} />
        </div>
      </div>
      <div className="mini-counts">
        {(["accepted", "edited", "rejected", "pending"] as const).map((s) => (
          <span key={s}>
            <span className="dot" style={{ background: REVIEW_STATUS_COLORS[s] }} />
            {labels.review_status_labels[s]} {counts[s]}
          </span>
        ))}
      </div>
      <span style={{ flex: 1 }} />
      <label className="pending-toggle">
        <input
          type="checkbox"
          checked={showPendingOnly}
          onChange={(e) => {
            setShowPendingOnly(e.target.checked);
            setIdx(0);
          }}
        />
        未判定のみ表示
      </label>
      <button className="btn primary" disabled={counts.pending === 0} onClick={gotoNextPending}>
        次の未判定へ <span className="kbd">N</span>
      </button>
    </div>
  );

  if (allTotal === 0) {
    return (
      <div className="empty-note">
        レビュー対象の QA がありません。サイドバーでデータセットを選んでください
        (まだ生成していなければ <span className="mono">uv run rageval generate</span>)。
      </div>
    );
  }

  if (counts.pending === 0 && showPendingOnly) {
    return (
      <div>
        {headerStrip}
        <div className="banner ok" style={{ marginTop: 12 }}>
          全問レビュー済みです。data/reviewed/ のファイルがコミット対象になります。
          判定を見直す場合は「未判定のみ表示」を外してください。
        </div>
      </div>
    );
  }

  if (!qa) {
    return <div className="empty-note">表示できる問いがありません。絞り込みを確認してください。</div>;
  }

  const kg = isKg(qa);
  const groups = kg ? labels.checklist_kg : labels.checklist_general;
  const totalChecks = groups.reduce((s, g) => s + g.items.length, 0);
  const flagKey = itemKey(qa);
  const qaFlags = flags.get(flagKey) ?? new Set<string>();
  const nFlags = qaFlags.size;

  const toggleFlag = (key: string) => {
    setFlags((m) => {
      const next = new Map(m);
      const set = new Set(next.get(flagKey) ?? []);
      if (set.has(key)) set.delete(key);
      else set.add(key);
      next.set(flagKey, set);
      return next;
    });
  };

  return (
    <div>
      {headerStrip}
      <div className="review-layout">
        {/* 左: ナビゲーション */}
        <div className="review-list">
          <div className="head">
            <div style={{ fontSize: 12, color: "var(--muted)" }}>
              {showPendingOnly ? `未判定 ${total} 問` : `全 ${total} 問`} — クリックで移動
            </div>
          </div>
          <div className="review-items" ref={listRef}>
            {working.map((it, i) => (
              <button
                key={itemKey(it)}
                className={`review-item ${i === clamped ? "active" : ""}`}
                onClick={() => setIdx(i)}
              >
                <span className="dot" style={{ background: REVIEW_STATUS_COLORS[it.review_status] }} />
                <span className="num" style={{ color: "var(--muted)" }}>{i + 1}</span>
                <span className="q">{truncate(it.question, 42)}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 右: レビュー本体 */}
        <div className="review-main">
          {error && <div className="banner warn">{error}</div>}

          <div className="review-head">
            <StatusBadge status={qa.review_status} labels={labels} />
            <span className="mono" style={{ color: "var(--muted)" }}>{qa.qa_id}</span>
            <span className="chip">track: <b>{kg ? "kg_poc" : "general"}</b></span>
            <span className="chip">{qa._source}</span>
            <span style={{ flex: 1 }} />
            <span style={{ fontSize: 12, color: "var(--muted)" }}>
              {clamped + 1} / {total} 問　<kbd className="hint">←</kbd> <kbd className="hint">→</kbd> で移動
            </span>
          </div>

          <div className="review-document">
            <section className="review-section">
              <h2 className="review-section-title">■質問</h2>
              {editMode ? (
                <div className="review-q">
                  <textarea rows={3} value={editQ} onChange={(e) => setEditQ(e.target.value)} aria-label="質問" />
                </div>
              ) : (
                <div className="qa-question">{qa.question}</div>
              )}

              <h2 className="review-section-title review-answer-title">■回答</h2>
              {editMode ? (
                <div className="review-a">
                  <textarea rows={5} value={editA} onChange={(e) => setEditA(e.target.value)} aria-label="回答" />
                  <p className="caption review-edit-hint">
                    <kbd className="hint">Ctrl+Enter</kbd> 保存 / <kbd className="hint">Esc</kbd> キャンセル
                  </p>
                </div>
              ) : (
                <div className="qa-answer">{qa.answer}</div>
              )}
              <div className="review-meta-line">
                <QAChips qa={qa} labels={labels} />
              </div>
            </section>

            <section className="review-section">
              <h2 className="review-section-title">■引用元</h2>
            {evidenceLoading && !ev && <p className="caption">原文チャンクを照合中…</p>}
            {ev && !ev.chunks_found && (
              <div className="banner warn">
                原文チャンクが見つかりません。全チャンクディレクトリを探しても引用に合う原文が
                ありませんでした。サイドバーの詳細設定で照合先を確認してください。
              </div>
            )}
            {ev && ev.chunks_found && ev.all_match && (
              <div className="banner ok">
                機械チェック合格: 引用はすべて原文と一致 (照合先: {ev.used_chunks_dir})。
                黄色の箇所が回答と合っているか目視確認だけで OK。
              </div>
            )}
            {ev && ev.chunks_found && !ev.all_match && (
              <div className="banner warn">
                機械チェック未合格: 原文に見つからない引用があります (照合先: {ev.used_chunks_dir})。
                下の ✗ 印の引用を原文と突き合わせてください。
              </div>
            )}
            {ev && (
              <>
                <button className="icon-btn review-evidence-toggle" onClick={() => setEvidenceOpen((o) => !o)}>
                  {evidenceOpen ? "▼" : "▶"} 引用と原文チャンク ({ev.rationales.length}件の引用 / {ev.chunks.length}チャンク)
                </button>
                {evidenceOpen && (
                  <div style={{ marginTop: 8 }}>
                    {ev.rationales.map((r, i) => (
                      <div className="rationale-line" key={i}>
                        <span className={r.matched ? "ok" : "ng"}>{r.matched ? "✓" : "✗"}</span>{" "}
                        <span className="mono" style={{ color: "var(--muted)" }}>
                          {r.doc_id}{r.page != null ? ` p.${r.page}` : ""}
                        </span>{" "}
                        — {r.text}
                      </div>
                    ))}
                    {ev.chunks.map((ch) => (
                      <div className="evidence-chunk" key={ch.chunk_id}>
                        <div className="meta">
                          <b>{ch.doc_id}</b>
                          {ch.page != null && <> ・ p.{ch.page}</>} ・ <span className="mono">{ch.chunk_id}</span>
                          {ch.section_path.length > 0 && <> ・ {ch.section_path.join(" > ")}</>}
                        </div>
                        <EvidenceText text={ch.text} spans={ch.spans} />
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
            </section>

            <section className="review-section review-decision-section">
              <h2 className="review-section-title">■ 判定</h2>
            {nFlags === 0 ? (
              <div className="banner ok">指摘なし ({totalChecks}項目) — このまま「採用」できます。問題があるときだけ下でチェックを入れる。</div>
            ) : (
              <div className="banner warn">{nFlags} 件の指摘あり — 「修正」または「却下」を検討してください。</div>
            )}
            <div className="checklist-groups">
              {groups.map((g) => (
                <div key={g.group}>
                  <h4>{g.group}</h4>
                  {g.items.map((ci) => (
                    <label className="check-item" key={ci.key} title={ci.desc}>
                      <input
                        type="checkbox"
                        checked={qaFlags.has(ci.key)}
                        onChange={() => toggleFlag(ci.key)}
                      />
                      <span>
                        {ci.label}
                        <span className="desc">{ci.desc}</span>
                      </span>
                    </label>
                  ))}
                </div>
              ))}
            </div>

            <p className="caption review-save-caption">
              判定は即座に data/reviewed/{qa._source} に保存されます (レビュー者: {reviewer})。
            </p>
            <div className="action-bar review-action-bar">
              {!editMode ? (
                <>
                  <button className="btn primary" disabled={saving} onClick={() => void mark("accepted")}>
                    採用 <span className="kbd">A</span>
                  </button>
                  <button className="btn warn" disabled={saving} onClick={startEdit}>
                    修正 <span className="kbd">E</span>
                  </button>
                  <button className="btn danger" disabled={saving} onClick={() => void mark("rejected")}>
                    却下 <span className="kbd">R</span>
                  </button>
                </>
              ) : (
                <>
                  <button className="btn primary" disabled={saving} onClick={() => void mark("edited", editQ, editA)}>
                    修正を保存 <span className="kbd">Ctrl+Enter</span>
                  </button>
                  <button className="btn" disabled={saving} onClick={() => setEditMode(false)}>
                    キャンセル <span className="kbd">Esc</span>
                  </button>
                </>
              )}
              {saving && <span style={{ fontSize: 12.5, color: "var(--muted)" }}>保存中…</span>}
              <div className="nav">
                <button className="btn" disabled={clamped === 0} onClick={() => setIdx(clamped - 1)}>← 前へ</button>
                <button className="btn" disabled={clamped >= total - 1} onClick={() => setIdx(clamped + 1)}>次へ →</button>
              </div>
            </div>
            </section>
          </div>

          <details className="raw-json">
            <summary>詳細 (難易度根拠・診断タグ・判定LLMスコア・生JSON)</summary>
            <div className="card" style={{ marginTop: 8 }}>
              {qa.difficulty_rationale && (
                <p style={{ fontSize: 12.5 }}><b>難易度根拠:</b> {qa.difficulty_rationale}</p>
              )}
              {kg && qa.kg_query_type && (
                <p style={{ fontSize: 12.5 }}>
                  <b>クエリ型の定義:</b> {labels.kg_query_type_descriptions[qa.kg_query_type] ?? ""}
                </p>
              )}
              {kg && qa.kg_novelty && (
                <p style={{ fontSize: 12.5 }}>
                  <b>未知性の定義:</b> {labels.kg_novelty_descriptions[qa.kg_novelty] ?? ""}
                </p>
              )}
              {!kg &&
                qa.aspect.map((a) => (
                  <p style={{ fontSize: 12.5 }} key={a}>
                    <b>観点「{labels.aspect_labels[a] ?? a}」:</b> {labels.aspect_descriptions[a] ?? "(定義なし)"}
                  </p>
                ))}
              <DiagTags qa={qa} labels={labels} />
              <p style={{ fontSize: 12.5, marginTop: 8 }}>
                <b>判定LLMスコア:</b> 答えやすさ {qa.filter_scores.answerability?.toFixed(1) ?? "—"} / 根拠妥当性 {qa.filter_scores.grounding?.toFixed(1) ?? "—"} / 独自性 {qa.filter_scores.uniqueness?.toFixed(2) ?? "—"} / リーク {qa.filter_scores.leakage ?? "—"}
              </p>
              <pre style={{ background: "var(--plane)", border: "1px solid var(--border)", borderRadius: 8, padding: 10, overflow: "auto", maxHeight: 320, fontSize: 11.5 }}>
                {JSON.stringify(qa, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
