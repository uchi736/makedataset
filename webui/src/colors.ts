// 色の割り当て規則:
//  - カテゴリ (系列の識別) → categorical スロット (CSS 変数、固定順で割当)
//  - 難易度・合否・一致 (状態) → status パレット
//  - ヒートマップ (量) → sequential 青ランプ (ここは実 hex が要るため TS に持つ)

export const CAT_VARS = [
  "var(--cat-1)", "var(--cat-2)", "var(--cat-3)", "var(--cat-4)",
  "var(--cat-5)", "var(--cat-6)", "var(--cat-7)", "var(--cat-8)",
];

export const CATEGORY_COLORS: Record<string, string> = {
  Integration: "var(--cat-1)",
  Reasoning:   "var(--cat-5)",
  Logic:       "var(--cat-2)",
  Figure:      "var(--cat-3)",
  Abstention:  "var(--muted)",
};

export const DIFFICULTY_COLORS: Record<string, string> = {
  Easy:   "var(--status-good)",
  Medium: "var(--status-warning)",
  Hard:   "var(--status-critical)",
};

export const LEAKAGE_COLORS: Record<string, string> = {
  pass: "var(--status-good)",
  fail: "var(--status-critical)",
};

export const DIFFICULTY_MATCH_COLORS: Record<string, string> = {
  aligned:  "var(--status-good)",
  too_easy: "var(--cat-1)",
  too_hard: "var(--status-critical)",
};

export const REVIEW_STATUS_COLORS: Record<string, string> = {
  pending:  "var(--muted)",
  accepted: "var(--status-good)",
  edited:   "var(--cat-1)",
  rejected: "var(--status-critical)",
};

export const RAG_MATCH_COLORS: Record<string, string> = {
  match:    "var(--status-good)",
  partial:  "var(--status-warning)",
  no_match: "var(--status-critical)",
};

export const RETRIEVAL_HIT_COLORS: Record<string, string> = {
  true:  "var(--status-good)",
  false: "var(--status-critical)",
};

export const LLM_KNOWLEDGE_COLORS: Record<string, string> = {
  unknown: "var(--status-critical)",
  known:   "var(--muted)",
};

export const EVIDENCE_COLORS: Record<string, string> = {
  "no-evidence": "var(--muted)",
  "hier-ref":    "var(--cat-1)",
  "coord-ref":   "var(--cat-2)",
  "multi-ref":   "var(--cat-5)",
};

// sequential 青ランプ (palette.md の 100→700)。ヒートマップのセル塗りに使う。
// dark モードでは「多い = 明るい」に反転させる (暗い面では明るさが強度として読める)。
const SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const SEQ_DARK = [...SEQ_LIGHT].reverse();

/** count/max (0..1) をランプの hex とセル内文字色に変換する。 */
export function seqColor(ratio: number, dark: boolean): { bg: string; fg: string } {
  const ramp = dark ? SEQ_DARK : SEQ_LIGHT;
  const idx = Math.min(ramp.length - 1, Math.max(0, Math.floor(ratio * ramp.length)));
  // light: 濃い青 (idx>=4) は白文字 / dark: 明るい青 (idx>=4) は黒文字
  const fg = idx >= 4 ? (dark ? "#0b0b0b" : "#ffffff") : dark ? "#ffffff" : "#0b0b0b";
  return { bg: ramp[idx], fg };
}
