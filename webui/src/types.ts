// バックエンド (rageval.webapi) が返す JSON の型定義。
// Python 側 schema.py / labels.py と対応する。

export interface Rationale {
  doc_id: string;
  chunk_id?: string | null;
  page?: number | null;
  text: string;
}

export interface FilterScores {
  answerability?: number | null;
  leakage?: "pass" | "fail" | null;
  grounding?: number | null;
  uniqueness?: number | null;
  difficulty_match?: "aligned" | "too_easy" | "too_hard" | null;
  rationale_grounded?: number | null;
}

export interface RagVerification {
  top_k: number;
  retrieved_chunk_ids: string[];
  retrieval_hit_doc?: boolean;
  retrieval_hit_chunk?: boolean;
  retrieval_hit?: boolean;
  rag_answer: string;
  answer_match: "match" | "partial" | "no_match";
  judge_raw?: string | null;
  rag_model: string;
  judge_model: string;
  verified_at: string;
}

export type ReviewStatus = "pending" | "accepted" | "edited" | "rejected";
export type Difficulty = "Easy" | "Medium" | "Hard";
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
  llm_knowledge?: "known" | "unknown" | null;
  rag_verification?: RagVerification | null;
  generation: { model: string; prompt_version: string; generated_at: string };
  filter_scores: FilterScores;
  review_status: ReviewStatus;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  _source: string;
  [key: string]: unknown;
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
