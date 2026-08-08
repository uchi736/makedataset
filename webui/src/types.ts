// バックエンド (rageval.webapi) が返す JSON の型定義。
// Python 側 schema.py / labels.py と対応する。

// ===== Python の Literal に対応する列挙 =====
// 並びの正本は schema.py の Literal で、画面はそれを配信ラベル表の鍵順で受け取る
// (enumOrder)。2026-08-08: 同じ並びが画面の各所へ手写しされており、Python 側に値を
// 足しても画面は増えた値を数えないまま全件を分母にする (棒の合計が 100% に届かない)
// 事故が起きていたため、控えの並びもここ1箇所に集約した。

export type Difficulty = "Easy" | "Medium" | "Hard";
export type RagAnswerMatch = "match" | "partial" | "no_match";
export type LeakageVerdict = "pass" | "fail";
export type DifficultyMatch = "aligned" | "too_easy" | "too_hard";
export type ReviewStatus = "pending" | "accepted" | "edited" | "rejected";
export type LLMKnowledge = "known" | "unknown";
export type RetrievalMode = "dense" | "hybrid" | "hybrid_rerank";

/** 列挙値の並び。Record<T, true> の鍵で網羅を型検査に確かめさせているので、
    上の型に値を足して表に書き忘れると型エラーになる (黙って落ちない)。 */
function enumValues<T extends string>(table: Record<T, true>): T[] {
  return Object.keys(table) as T[];
}

export const ALL_RAG_ANSWER_MATCH = enumValues<RagAnswerMatch>({
  match: true, partial: true, no_match: true,
});
export const ALL_LEAKAGE = enumValues<LeakageVerdict>({ pass: true, fail: true });
export const ALL_DIFFICULTY_MATCH = enumValues<DifficultyMatch>({
  aligned: true, too_easy: true, too_hard: true,
});
export const ALL_REVIEW_STATUS = enumValues<ReviewStatus>({
  pending: true, accepted: true, edited: true, rejected: true,
});
export const ALL_LLM_KNOWLEDGE = enumValues<LLMKnowledge>({ known: true, unknown: true });
export const ALL_RETRIEVAL_MODE = enumValues<RetrievalMode>({
  dense: true, hybrid: true, hybrid_rerank: true,
});

/** 列挙の並びは配信されたラベル表の鍵を正とする — Python 側に値を足したとき、
    画面を直さなくても分布と絞り込みに現れるようにするため。配信が古い
    (ラベルが空) ときだけ上の控えに落ちる。 */
export function enumOrder<T extends string>(
  labels: Record<string, string> | undefined,
  fallback: T[],
): T[] {
  const keys = Object.keys(labels ?? {}) as T[];
  return keys.length > 0 ? keys : fallback;
}

export interface Rationale {
  doc_id: string;
  chunk_id?: string | null;
  page?: number | null;
  text: string;
}

export interface FilterScores {
  answerability?: number | null;
  leakage?: LeakageVerdict | null;
  grounding?: number | null;
  uniqueness?: number | null;
  difficulty_match?: DifficultyMatch | null;
  rationale_grounded?: number | null;
}

export interface RagVerification {
  top_k: number;
  retrieved_chunk_ids: string[];
  // 検索方式 (schema.RAGVerification.retrieval_mode)。dense=密のみ (旧) /
  // hybrid=密+BM25 / hybrid_rerank=交差エンコーダで並べ直す2段。方式ごとに
  // 命中の基準が違うので、混ざった行を同じ分母で読むと命中率の意味が変わる。
  // 古い JSONL には項目が無いため、欠けは Python 側の既定と同じ dense として扱う。
  retrieval_mode?: RetrievalMode;
  retrieval_hit_doc?: boolean;
  retrieval_hit_chunk?: boolean;
  // 逐語一致した根拠の割合。1.0=全命中、0<x<1 が部分命中。命中/外れの2値に
  // 潰すと aggregation (要素が複数チャンクに散る型) の実測が読めなくなる。
  retrieval_hit_ratio?: number;
  retrieval_hit?: boolean;
  rag_answer: string;
  answer_match: RagAnswerMatch;
  judge_raw?: string | null;
  // judge が正常に判定を返したか。false は「照合できなかった」であって
  // 「vector RAG が解けなかった」ではない。障害由来の no_match を KG候補に
  // 数えないための旗 (schema.py と同じ既定 true = 正常)。
  judge_ok?: boolean;
  rag_model: string;
  judge_model: string;
  verified_at: string;
}

export type Track = "all" | "general" | "kg_poc";

export interface QAItem {
  qa_id: string;
  question: string;
  answer: string;
  rationale: Rationale[];
  category: string[];
  aspect: string[];
  reasoning_complexity: Record<string, boolean | string>;
  retrieval_difficulty: Record<string, boolean>;
  source_structure: Record<string, boolean>;
  explainability: { evidence_strictness: string };
  retrieval_level: Difficulty;
  answer_level: Difficulty;
  difficulty_rationale: string;
  business_scenario?: string | null;
  kg_query_type?: string | null;
  kg_novelty?: string | null;
  llm_knowledge?: LLMKnowledge | null;
  rag_verification?: RagVerification | null;
  generation: { model: string; prompt_version: string; generated_at: string };
  filter_scores: FilterScores;
  review_status: ReviewStatus;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  _source: string;
  [key: string]: unknown;
}

/** KG候補 = 判定LLMが「答えあり」と認めた (または未判定の) 問いのうち、
    vector RAG が落としたもの。判定規則を Python 側と揃えるためここ1箇所に置く。
    judge_ok=false を外すのが要点で、これは「照合できなかった」であって RAG が
    解けなかった証拠ではない (rag_verify.py の注記・cli の pipeline も同じ扱い)。
    混ぜると推論サーバの一時障害が KG-RAG の優位の証拠として読まれる
    (2026-08-08 の監査指摘)。合格線は配信された目標値を渡す。 */
export function isKgCandidate(item: QAItem, answerabilityPass: number): boolean {
  const v = item.rag_verification;
  if (v == null || v.answer_match !== "no_match") return false;
  if (v.judge_ok === false) return false;
  const score = item.filter_scores.answerability;
  return score == null || score >= answerabilityPass;
}

export interface ChecklistItem { key: string; label: string; desc: string }
export interface ChecklistGroup { group: string; items: ChecklistItem[] }

export interface Labels {
  all_aspects: string[];
  aspect_labels: Record<string, string>;
  aspect_descriptions: Record<string, string>;
  aspect_examples: Record<string, string>;
  aspect_to_category: Record<string, string>;
  category_order: string[];
  category_labels: Record<string, string>;
  diff_order: Difficulty[];
  difficulty_labels: Record<string, string>;
  leakage_labels: Record<string, string>;
  difficulty_match_labels: Record<string, string>;
  review_status_labels: Record<string, string>;
  rag_answer_match_labels: Record<string, string>;
  output_type_labels: Record<string, string>;
  evidence_labels: Record<string, string>;
  diag_labels: Record<string, Record<string, string>>;
  all_kg_query_types: string[];
  kg_query_type_labels: Record<string, string>;
  kg_query_type_descriptions: Record<string, string>;
  all_kg_novelty: string[];
  kg_novelty_labels: Record<string, string>;
  kg_novelty_descriptions: Record<string, string>;
  llm_knowledge_labels: Record<string, string>;
  targets: {
    search_diff: Record<string, number>;
    answer_diff: Record<string, number>;
    abstention: [number, number];
    pilot_size: [number, number];
    aspect_min: number;
    answerability_pass: number;
    grounding_pass: number;
    uniqueness_pass: number;
  };
  checklist_general: ChecklistGroup[];
  checklist_kg: ChecklistGroup[];
}

export interface Meta {
  sources: {
    datasets: { name: string; files: string[] }[];
    chunk_dirs: string[];
  };
  labels: Labels;
  defaults: { reviewer: string };
}

export interface ItemsResponse {
  items: QAItem[];
  skipped: Record<string, number>;
  reviewed_override: string[];
  files: string[];
  /** 複数ファイルにまたがって重複していた qa_id の個数 */
  duplicate_qa_ids: number;
}

export interface EvidenceChunk {
  chunk_id: string;
  doc_id: string;
  page?: number | null;
  section_path: string[];
  text: string;
  spans: [number, number][];
}

export interface EvidenceResponse {
  chunks: EvidenceChunk[];
  rationales: { doc_id: string; page?: number | null; text: string; matched: boolean }[];
  all_match: boolean;
  chunks_found: boolean;
  used_chunks_dir: string;
}
