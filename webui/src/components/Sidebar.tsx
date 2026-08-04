import type { ItemsResponse, Meta, Track } from "../types";

const TRACKS: { value: Track; label: string }[] = [
  { value: "all", label: "自動 (全 track 表示)" },
  { value: "general", label: "general (25観点) のみ" },
  { value: "kg_poc", label: "KG-PoC (3軸) のみ" },
];

export default function Sidebar(props: {
  meta: Meta;
  itemsResp: ItemsResponse | null;
  datasetDir: string;
  datasetFile: string;
  track: Track;
  chunksDir: string;
  catFilter: Set<string>;
  sourceFilter: Set<string> | null;
  reviewer: string;
  onDatasetDir: (v: string) => void;
  onDatasetFile: (v: string) => void;
  onTrack: (v: Track) => void;
  onChunksDir: (v: string) => void;
  onCatFilter: (v: Set<string>) => void;
  onSourceFilter: (v: Set<string> | null) => void;
  onReviewer: (v: string) => void;
}) {
  const { meta, itemsResp } = props;
  const labels = meta.labels;
  const dirs = meta.sources.datasets;
  const currentDirFiles = dirs.find((d) => d.name === props.datasetDir)?.files ?? [];
  const sources = itemsResp?.files ?? [];
  const skippedTotal = itemsResp ? Object.values(itemsResp.skipped).reduce((a, b) => a + b, 0) : 0;

  const toggleIn = (set: Set<string>, key: string, cb: (v: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    cb(next);
  };

  return (
    <div className="sidebar">
      <div className="side-section">
        <label>① データセットを選ぶ</label>
        <select value={props.datasetDir} onChange={(e) => props.onDatasetDir(e.target.value)}>
          {dirs.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.files.length}ファイル)
            </option>
          ))}
        </select>
        {currentDirFiles.length > 1 && (
          <select
            style={{ marginTop: 6 }}
            value={props.datasetFile}
            onChange={(e) => props.onDatasetFile(e.target.value)}
          >
            <option value="">(全ファイル)</option>
            {currentDirFiles.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        )}
        <div className="side-note">
          raw = 生成直後 / filtered = 判定通過 / reviewed = 人手レビュー済み
        </div>
        {itemsResp && itemsResp.reviewed_override.length > 0 && (
          <div className="side-note">
            ✎ レビューの続きを表示中 ({itemsResp.reviewed_override.join(", ")})
          </div>
        )}
        {skippedTotal > 0 && (
          <div className="side-note">旧スキーマ {skippedTotal} 件をスキップ</div>
        )}
      </div>

      <div className="side-section">
        <label>② レビュー者 (reviewed_by)</label>
        <input
          type="text"
          value={props.reviewer}
          placeholder="名前を入力"
          onChange={(e) => props.onReviewer(e.target.value)}
        />
      </div>

      <details className="side-advanced">
        <summary>詳細設定</summary>

        <div className="side-section">
          <label>トラックの表示</label>
          <select value={props.track} onChange={(e) => props.onTrack(e.target.value as Track)}>
            {TRACKS.map((tr) => (
              <option key={tr.value} value={tr.value}>{tr.label}</option>
            ))}
          </select>
        </div>

        <div className="side-section">
          <label>根拠の照合先チャンク</label>
          <select value={props.chunksDir} onChange={(e) => props.onChunksDir(e.target.value)}>
            <option value="auto">自動 (引用が一致する場所を探す)</option>
            {meta.sources.chunk_dirs.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div className="side-section">
          <label>カテゴリで絞り込み (general)</label>
          <div className="chip-row">
            {labels.category_order.map((c) => (
              <button
                key={c}
                className={`filter-chip ${props.catFilter.has(c) ? "on" : ""}`}
                onClick={() => toggleIn(props.catFilter, c, props.onCatFilter)}
              >
                {labels.category_labels[c] ?? c}
              </button>
            ))}
          </div>
        </div>

        {sources.length > 1 && (
          <div className="side-section">
            <label>ファイルで絞り込み</label>
            <div className="chip-row">
              {sources.map((f) => {
                const active = props.sourceFilter == null || props.sourceFilter.has(f);
                return (
                  <button
                    key={f}
                    className={`filter-chip ${active ? "on" : ""}`}
                    onClick={() => {
                      const cur = props.sourceFilter ?? new Set(sources);
                      const next = new Set(cur);
                      if (next.has(f)) next.delete(f);
                      else next.add(f);
                      props.onSourceFilter(next.size === sources.length ? null : next);
                    }}
                  >
                    {f}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </details>
    </div>
  );
}
