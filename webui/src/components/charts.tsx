import { Fragment } from "react";
import type { ReactNode } from "react";
import { seqColor } from "../colors";
import { useDark, useTooltip } from "../ui";
import { pct } from "../util";

// ---------------- KPI タイル ----------------

export function StatTile(props: {
  label: string;
  value: ReactNode;
  delta?: ReactNode;
  tone?: "good" | "bad" | "plain";
}) {
  const toneClass = props.tone === "good" ? "good" : props.tone === "bad" ? "bad" : "";
  return (
    <div className="stat-tile">
      <div className="label">{props.label}</div>
      <div className="value">{props.value}</div>
      {props.delta != null && <div className={`delta ${toneClass}`}>{props.delta}</div>}
    </div>
  );
}

// ---------------- 横向き分布バー (件数 + 割合の直接ラベル、目標の赤刻み) ----------------

export interface DistRow {
  key: string;
  label: string;
  count: number;
  color: string;
  target?: number; // 0..1
  hint?: string;
}

export function DistBar({ rows, total }: { rows: DistRow[]; total: number }) {
  const tip = useTooltip();
  const denom = total || 1;
  return (
    <div className="dist">
      {rows.map((r) => {
        const ratio = r.count / denom;
        // ラベルはバーの右外に置く。バーが長いときだけ内側 (白) に。
        const inside = ratio > 0.72;
        return (
          <div className="dist-row" key={r.key}>
            <div className="name">{r.label}</div>
            <div
              className="dist-track"
              onMouseMove={(e) =>
                tip.show(e, (
                  <div>
                    <b>{r.label}</b>: {r.count}件 ({pct(ratio, 1)})
                    {r.target != null && <div>目標 {pct(r.target)}</div>}
                    {r.hint && <div>{r.hint}</div>}
                  </div>
                ))
              }
              onMouseLeave={tip.hide}
            >
              <div className="dist-fill" style={{ width: pct(ratio), background: r.color }} />
              <div
                className="dist-label"
                style={
                  inside
                    ? { right: `calc(${pct(1 - ratio)} + 6px)`, color: "#fff", paddingLeft: 0 }
                    : { left: pct(ratio), opacity: r.count === 0 ? 0.5 : 1 }
                }
              >
                {r.count}件 ({pct(ratio)})
              </div>
              {r.target != null && (
                <div className="dist-target" style={{ left: `calc(${pct(r.target)} - 1px)` }} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Legend({ keys }: { keys: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {keys.map((k) => (
        <span className="key" key={k.label}>
          <span className="sw" style={{ background: k.color }} />
          {k.label}
        </span>
      ))}
    </div>
  );
}

// ---------------- ヒートマップ (sequential 青ランプ / 0件セルは対象外表示) ----------------

export interface HeatmapProps {
  xLabels: string[];
  yLabels: string[];
  /** values[y][x] */
  values: number[][];
  /** 0件セルを「補充が要る欠け」として赤味で示すか */
  flagZero?: boolean;
  cellTip?: (x: number, y: number, v: number) => ReactNode;
}

export function Heatmap({ xLabels, yLabels, values, flagZero, cellTip }: HeatmapProps) {
  const tip = useTooltip();
  const dark = useDark();
  const max = Math.max(1, ...values.flat());
  return (
    <div
      className="heatmap"
      style={{ gridTemplateColumns: `minmax(90px, 160px) repeat(${xLabels.length}, 1fr)` }}
    >
      <div className="hm-corner" />
      {xLabels.map((x) => (
        <div className="hm-xlabel" key={x}>{x}</div>
      ))}
      {yLabels.map((y, yi) => (
        <Fragment key={y}>
          <div className="hm-ylabel">{y}</div>
          {xLabels.map((x, xi) => {
            const v = values[yi][xi];
            const zero = v === 0;
            const { bg, fg } = seqColor(v / max, dark);
            const style = zero
              ? { background: flagZero ? "var(--gap-tint)" : "var(--plane)", color: "var(--muted)" }
              : { background: bg, color: fg };
            return (
              <div
                className="hm-cell"
                key={`${y}-${x}`}
                style={style}
                onMouseMove={(e) =>
                  tip.show(
                    e,
                    cellTip ? cellTip(xi, yi, v) : (
                      <div>{y} × {x}: <b>{v}件</b>{zero && flagZero ? " (未カバー)" : ""}</div>
                    ),
                  )
                }
                onMouseLeave={tip.hide}
              >
                {v}
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}

// ---------------- しきい値付きヒストグラム ----------------

export interface HistogramProps {
  values: number[];
  threshold: number;
  domain: [number, number];
  binWidth: number;
  /** true: しきい値以上が合格 (右側が緑) */
  passAbove?: boolean;
}

export function Histogram({ values, threshold, domain, binWidth, passAbove = true }: HistogramProps) {
  const tip = useTooltip();
  const [lo, hi] = domain;
  const nBins = Math.max(1, Math.round((hi - lo) / binWidth));
  const bins = new Array(nBins).fill(0) as number[];
  for (const v of values) {
    if (v == null || Number.isNaN(v)) continue;
    const idx = Math.min(nBins - 1, Math.max(0, Math.floor((v - lo) / binWidth)));
    bins[idx] += 1;
  }
  const maxBin = Math.max(1, ...bins);
  const W = 320;
  const H = 150;
  const padB = 18;
  const padT = 6;
  const barW = W / nBins;
  const tx = ((threshold - lo) / (hi - lo)) * W;
  return (
    <div className="hist">
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {bins.map((b, i) => {
          const binLo = lo + i * binWidth;
          const binHi = binLo + binWidth;
          const mid = (binLo + binHi) / 2;
          const isPass = passAbove ? mid >= threshold : mid < threshold;
          const h = (b / maxBin) * (H - padB - padT);
          return (
            <rect
              key={i}
              x={i * barW + 1}
              y={H - padB - h}
              width={Math.max(1, barW - 2)}
              height={Math.max(b > 0 ? 2 : 0, h)}
              rx={2}
              fill={isPass ? "var(--status-good)" : "var(--status-critical)"}
              onMouseMove={(e) =>
                tip.show(e, (
                  <div>
                    {binLo.toFixed(2)} 〜 {binHi.toFixed(2)}: <b>{b}件</b> ({isPass ? "合格域" : "不合格域"})
                  </div>
                ))
              }
              onMouseLeave={tip.hide}
            />
          );
        })}
        <line x1={0} y1={H - padB} x2={W} y2={H - padB} stroke="var(--baseline)" strokeWidth={1} />
        <line
          x1={tx} y1={padT} x2={tx} y2={H - padB}
          stroke="var(--ink)" strokeWidth={1.5} strokeDasharray="5 4"
        />
        <text className="rule-label" x={Math.min(tx + 4, W - 60)} y={padT + 10}>
          しきい値 {threshold}
        </text>
        <text className="axis-text" x={0} y={H - 4}>{lo}</text>
        <text className="axis-text" x={W - 18} y={H - 4}>{hi}</text>
      </svg>
    </div>
  );
}

// ---------------- 100% 積み上げバー (観点別の判定割合など) ----------------

export interface StackSegment { key: string; label: string; count: number; color: string }

export function StackRow(props: { name: string; total: number; segments: StackSegment[] }) {
  const tip = useTooltip();
  const { name, total, segments } = props;
  const denom = total || 1;
  return (
    <div className="stack-row">
      <div className="name" title={name}>{name}</div>
      <div className="stack-track">
        {segments.filter((s) => s.count > 0).map((s) => {
          const ratio = s.count / denom;
          return (
            <div
              key={s.key}
              className="stack-seg"
              style={{ width: pct(ratio, 2), background: s.color }}
              onMouseMove={(e) =>
                tip.show(e, (
                  <div>
                    <b>{name}</b>
                    <div>{s.label}: {s.count}件 ({pct(ratio, 1)})</div>
                  </div>
                ))
              }
              onMouseLeave={tip.hide}
            >
              {ratio >= 0.14 ? s.count : ""}
            </div>
          );
        })}
      </div>
      <div className="total num">{total}件</div>
    </div>
  );
}
