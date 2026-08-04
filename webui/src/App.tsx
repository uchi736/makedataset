import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchItems, fetchMeta } from "./api";
import AnalysisTab from "./components/AnalysisTab";
import ListTab from "./components/ListTab";
import ReviewTab from "./components/ReviewTab";
import Sidebar from "./components/Sidebar";
import type { ItemsResponse, Meta, QAItem, Track } from "./types";
import { ThemeContext, TooltipProvider } from "./ui";
import { isKg, itemKey } from "./util";

type TabKey = "review" | "list" | "analysis";

function pickDefaultDataset(meta: Meta): string {
  const names = meta.sources.datasets.map((d) => d.name);
  for (const cand of ["filtered", "reviewed", "raw"]) {
    if (names.includes(cand)) return cand;
  }
  return names[0] ?? "";
}

function HelpPanel({ onClose }: { onClose: () => void }) {
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="help-panel" role="dialog" aria-label="使い方">
        <div style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
          <h2 style={{ margin: 0, fontSize: 15 }}>使い方 — 3ステップ</h2>
          <span style={{ flex: 1 }} />
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>
        <ol className="help-steps">
          <li>
            <b>データセットを選ぶ</b> (左上)。
            raw = 生成直後 / filtered = 判定LLM通過 / reviewed = レビュー済み。
            ふだんは filtered を開いてレビューする。
          </li>
          <li>
            <b>「分析」で現在地を確かめる</b>。
            件数・難易度の配分・品質ゲート通過率が目標に届いているかを見る。
          </li>
          <li>
            <b>「レビュー」で1問ずつ判定する</b>。
            ①質問と回答を読む → ②根拠 (黄色の箇所) を確かめる → ③採用/修正/却下。
            判定は即座に <span className="mono">data/reviewed/</span> に保存され、
            そのファイルが納品物になる。
          </li>
        </ol>
        <h3 style={{ fontSize: 13, margin: "14px 0 6px" }}>キーボード操作 (レビュー画面)</h3>
        <table className="help-keys">
          <tbody>
            <tr><td><kbd className="hint">A</kbd></td><td>採用</td><td><kbd className="hint">E</kbd></td><td>修正モード</td></tr>
            <tr><td><kbd className="hint">R</kbd></td><td>却下</td><td><kbd className="hint">N</kbd></td><td>次の未判定へ</td></tr>
            <tr><td><kbd className="hint">←</kbd> <kbd className="hint">→</kbd></td><td>前へ / 次へ</td><td><kbd className="hint">Ctrl+Enter</kbd></td><td>修正を保存</td></tr>
          </tbody>
        </table>
        <p className="caption" style={{ marginTop: 12 }}>
          根拠の照合先チャンクは自動で探します。狙いを固定したいときはサイドバーの詳細設定から。
        </p>
      </div>
    </>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);

  const [datasetDir, setDatasetDir] = useState("");
  const [datasetFile, setDatasetFile] = useState("");
  const [track, setTrack] = useState<Track>("all");
  const [chunksDir, setChunksDir] = useState("auto");

  const [itemsResp, setItemsResp] = useState<ItemsResponse | null>(null);
  const [itemsError, setItemsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [tab, setTab] = useState<TabKey>("review");
  const [catFilter, setCatFilter] = useState<Set<string>>(new Set());
  const [sourceFilter, setSourceFilter] = useState<Set<string> | null>(null);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("rageval.reviewer") ?? "");
  const [jumpTo, setJumpTo] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(() => localStorage.getItem("rageval.helpSeen") !== "1");
  const [sideOpen, setSideOpen] = useState(() => localStorage.getItem("rageval.sidebar") !== "0");

  useEffect(() => {
    localStorage.setItem("rageval.sidebar", sideOpen ? "1" : "0");
  }, [sideOpen]);

  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("rageval.theme");
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("rageval.theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    localStorage.setItem("rageval.reviewer", reviewer);
  }, [reviewer]);

  const closeHelp = useCallback(() => {
    setHelpOpen(false);
    localStorage.setItem("rageval.helpSeen", "1");
  }, []);

  // 初期ロード: メタ情報
  useEffect(() => {
    fetchMeta()
      .then((m) => {
        setMeta(m);
        setCatFilter(new Set(m.labels.category_order));
        setDatasetDir((cur) => cur || pickDefaultDataset(m));
        setReviewer((cur) => cur || m.defaults.reviewer);
      })
      .catch((e: Error) => setMetaError(e.message));
  }, []);

  const datasetPath = datasetFile ? `${datasetDir}/${datasetFile}` : datasetDir;

  // データセット / トラック切替でアイテムを再取得
  useEffect(() => {
    if (!datasetPath) return;
    let cancelled = false;
    setLoading(true);
    setItemsError(null);
    fetchItems(datasetPath, track)
      .then((res) => {
        if (cancelled) return;
        setItemsResp(res);
        setSourceFilter(null);
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setItemsResp(null);
          setItemsError(e.message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetPath, track]);

  const items = itemsResp?.items ?? [];

  const filteredItems = useMemo(() => {
    return items.filter((it) => {
      if (sourceFilter != null && !sourceFilter.has(it._source)) return false;
      if (!isKg(it) && meta) {
        const cat = it.category[0];
        if (cat && !catFilter.has(cat)) return false;
      }
      return true;
    });
  }, [items, sourceFilter, catFilter, meta]);

  const onItemUpdated = useCallback((updated: QAItem) => {
    setItemsResp((resp) => {
      if (!resp) return resp;
      // 同一 qa_id が複数ファイルにあり得る (raw と filtered など) ので、
      // 更新は「ファイル + qa_id」が一致する行だけに当てる。qa_id だけで
      // 突き合わせると、保存していないファイルの行まで「レビュー済み」に
      // 見え、_source 表示まで書き換わってしまう。
      const key = itemKey(updated);
      return {
        ...resp,
        items: resp.items.map((it) => (itemKey(it) === key ? { ...it, ...updated } : it)),
      };
    });
  }, []);

  const goReview = useCallback((qaId: string) => {
    setJumpTo(qaId);
    setTab("review");
  }, []);

  if (metaError) {
    return (
      <div className="loading">
        <p>バックエンドに接続できません: {metaError}</p>
        <p className="mono">uv run rageval web でサーバを起動してからリロードしてください。</p>
        <button className="btn" onClick={() => location.reload()}>再読み込み</button>
      </div>
    );
  }

  if (!meta) return <div className="loading">読み込み中…</div>;

  const pendingCount = filteredItems.filter((i) => i.review_status === "pending").length;
  const tabs: { key: TabKey; label: string; count?: number; hint: string }[] = [
    { key: "review", label: "レビュー", count: pendingCount, hint: "1問ずつ採否を判定する (主作業)" },
    { key: "list", label: "一覧", count: filteredItems.length, hint: "検索して個別の問いを確かめる" },
    { key: "analysis", label: "分析", hint: "件数・配分・品質が目標に届いているか見る" },
  ];

  return (
    <ThemeContext.Provider value={{ dark }}>
      <TooltipProvider>
        <div className="app">
          <header className="topbar">
            <button
              className="icon-btn side-toggle"
              onClick={() => setSideOpen((o) => !o)}
              title={sideOpen ? "設定パネルをたたむ (画面を広く使う)" : "設定パネルを開く"}
              aria-label="設定パネルの開閉"
            >
              {sideOpen ? "◀" : "☰"}
            </button>
            <h1>rageval</h1>
            <span className="sub">RAG評価データセットのレビューと分析</span>
            <span className="spacer" />
            <span className="sub">
              data/{datasetPath || "—"} ・ {items.length}件{loading ? " (読込中…)" : ""}
            </span>
            <button className="icon-btn" onClick={() => setHelpOpen(true)}>？ 使い方</button>
            <button
              className="icon-btn"
              onClick={() => setDark((d) => !d)}
              title="画面の明暗を切り替え"
            >
              {dark ? "☀" : "🌙"}
            </button>
          </header>

          <div className={`body ${sideOpen ? "" : "no-side"}`}>
            {sideOpen && (
            <Sidebar
              meta={meta}
              itemsResp={itemsResp}
              datasetDir={datasetDir}
              datasetFile={datasetFile}
              track={track}
              chunksDir={chunksDir}
              catFilter={catFilter}
              sourceFilter={sourceFilter}
              reviewer={reviewer}
              onDatasetDir={(v) => {
                setDatasetDir(v);
                setDatasetFile("");
              }}
              onDatasetFile={setDatasetFile}
              onTrack={setTrack}
              onChunksDir={setChunksDir}
              onCatFilter={setCatFilter}
              onSourceFilter={setSourceFilter}
              onReviewer={setReviewer}
            />
            )}

            <main className="main">
              <nav className="tabbar">
                {tabs.map((t) => (
                  <button
                    key={t.key}
                    className={`tab ${tab === t.key ? "active" : ""}`}
                    title={t.hint}
                    onClick={() => setTab(t.key)}
                  >
                    {t.label}
                    {t.count != null && <span className="count">{t.count}</span>}
                  </button>
                ))}
              </nav>

              <div className="content">
                {itemsError && (
                  <div className="banner warn">読み込みエラー: {itemsError}</div>
                )}
                {itemsResp && itemsResp.duplicate_qa_ids > 0 && (
                  <div className="banner warn">
                    同じ qa_id が複数ファイルに重複しています ({itemsResp.duplicate_qa_ids} 問)。
                    .enriched / .rag_verified は同一バッチの派生ファイルです。
                    サイドバーの「ファイル」で1つに絞ってください。
                  </div>
                )}
                {loading && !itemsResp && <div className="loading">読み込み中…</div>}
                {itemsResp && filteredItems.length === 0 && items.length > 0 && (
                  <div className="banner info">絞り込みで全件が除外されています。サイドバーの詳細設定を緩めてください。</div>
                )}
                {itemsResp && tab === "review" && (
                  <ReviewTab
                    key={`${datasetPath}::${track}`}
                    items={filteredItems}
                    labels={meta.labels}
                    datasetPath={datasetPath}
                    chunksDir={chunksDir}
                    reviewer={reviewer || "unknown"}
                    onItemUpdated={onItemUpdated}
                    jumpTo={jumpTo}
                    onJumpConsumed={() => setJumpTo(null)}
                  />
                )}
                {itemsResp && tab === "list" && (
                  <ListTab items={filteredItems} labels={meta.labels} onGoReview={goReview} />
                )}
                {itemsResp && tab === "analysis" && (
                  <AnalysisTab items={filteredItems} labels={meta.labels} onGoReview={goReview} />
                )}
              </div>
            </main>
          </div>

          {helpOpen && <HelpPanel onClose={closeHelp} />}
        </div>
      </TooltipProvider>
    </ThemeContext.Provider>
  );
}
