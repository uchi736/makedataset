import type { EvidenceResponse, ItemsResponse, Meta, QAItem, Rationale, ReviewStatus, Track } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* JSON でないエラーはステータス行のまま */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function fetchMeta(): Promise<Meta> {
  return request<Meta>("/api/meta");
}

export function fetchItems(path: string, track: Track): Promise<ItemsResponse> {
  const q = new URLSearchParams({ path, track });
  return request<ItemsResponse>(`/api/items?${q.toString()}`);
}

export function fetchEvidence(chunksDir: string, rationale: Rationale[]): Promise<EvidenceResponse> {
  return request<EvidenceResponse>("/api/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunks_dir: chunksDir, rationale }),
  });
}

export interface ReviewPayload {
  path: string;
  source: string;
  qa_id: string;
  status: ReviewStatus;
  reviewer: string;
  question?: string;
  answer?: string;
}

export function postReview(payload: ReviewPayload): Promise<QAItem> {
  return request<QAItem>("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
