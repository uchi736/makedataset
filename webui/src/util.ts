import type { QAItem, ReviewStatus } from "./types";

export function isKg(item: QAItem): boolean {
  return item.kg_query_type != null || item.kg_novelty != null;
}

/** 行の一意識別子。同一 qa_id が複数ファイル (.enriched / .rag_verified 等) に
    現れても衝突しないよう、元ファイル名と組にする。 */
export function itemKey(item: QAItem): string {
  return `${item._source}::${item.qa_id}`;
}

export function pct(n: number, digits = 0): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtScore(v: number | null | undefined, digits = 1): string {
  return v == null ? "—" : v.toFixed(digits);
}

export function statusCounts(items: QAItem[]): Record<ReviewStatus, number> {
  const out: Record<ReviewStatus, number> = { pending: 0, accepted: 0, edited: 0, rejected: 0 };
  for (const it of items) out[it.review_status] = (out[it.review_status] ?? 0) + 1;
  return out;
}

/** 値ごとの件数を数える。null/undefined は除外。 */
export function countBy<T>(items: T[], key: (t: T) => string | null | undefined): Map<string, number> {
  const m = new Map<string, number>();
  for (const it of items) {
    const k = key(it);
    if (k == null) continue;
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

export function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
