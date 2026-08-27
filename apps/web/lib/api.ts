// Typed client for the rag-integrity-guard API (apps/api). Mirrors the
// Pydantic schemas in apps/api/schemas.py and rag_guard/eval/report.py.

export type StrictnessMode = "strict" | "filter" | "log_only";

export type AttackType = "none" | "exact_mutation" | "payload_injection" | "semantic_drift";

export type ValidationStatus =
  | "valid"
  | "hash_mismatch"
  | "semantic_drift"
  | "missing_fingerprint";

export interface IngestChunkInput {
  text: string;
  metadata?: Record<string, unknown>;
}

export interface IngestedChunk {
  chunk_id: string;
  content_hash: string;
}

export interface IngestResponse {
  document_id: string;
  ingested: IngestedChunk[];
}

export interface QueryResult {
  chunk_id: string;
  text: string;
  integrity_status: ValidationStatus;
  integrity_similarity: number | null;
}

export interface QueryResponse {
  query: string;
  strictness: StrictnessMode;
  blocked: boolean;
  violations: string[];
  results: QueryResult[];
}

export interface AttackResponse {
  chunk_id: string;
  attack_type: AttackType;
  original_excerpt: string;
  tampered_excerpt: string;
  resulting_status: ValidationStatus;
  resulting_similarity: number | null;
}

export interface ConfusionCounts {
  true_positive: number;
  false_negative: number;
  false_positive: number;
  true_negative: number;
}

export interface AttackTypeMetrics {
  attack_type: AttackType;
  sample_count: number;
  true_positive_rate: number | null;
  counts: ConfusionCounts;
}

export interface BenchmarkReport {
  dataset: string;
  split: string;
  sample_size: number;
  similarity_threshold: number;
  overall_true_positive_rate: number;
  overall_false_positive_rate: number;
  per_attack_type: AttackTypeMetrics[];
  verification_overhead_ms_mean: number;
  verification_overhead_ms_p95: number;
  generated_at: string;
}

// Unset in production: Vercel routes /api/* straight to the Python function
// (see the root vercel.json), so a relative fetch is correct there. For
// local dev, point this at your running `uvicorn index:app` (see
// apps/web/.env.local.example).
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      0,
      `Couldn't reach the API${API_BASE_URL ? ` at ${API_BASE_URL}` : ""}. ` +
        "Is apps/api running (uvicorn index:app --port 8000)?"
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

export function ingestDocument(
  documentId: string,
  chunks: IngestChunkInput[]
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ document_id: documentId, chunks }),
  });
}

export function runQuery(
  queryText: string,
  options: { k?: number; strictness?: StrictnessMode; similarity_threshold?: number } = {}
): Promise<QueryResponse> {
  return apiFetch<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify({ query: queryText, ...options }),
  });
}

export function launchAttack(chunkId: string, attackType: AttackType): Promise<AttackResponse> {
  return apiFetch<AttackResponse>("/api/attack", {
    method: "POST",
    body: JSON.stringify({ chunk_id: chunkId, attack_type: attackType }),
  });
}

export function getBenchmarkReport(): Promise<BenchmarkReport> {
  return apiFetch<BenchmarkReport>("/api/benchmark");
}
