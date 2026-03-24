import type {
  AlgorithmsResponse,
  DatasetInfo,
  DatasetPreview,
  JobRecord,
  PredictResponse,
  PreprocessPayload,
  TrainResponse,
  UnsupervisedFamily,
  UploadResponse,
} from "./types";

/** Default FastAPI origin when env is missing (must match uvicorn bind). */
const DEFAULT_BACKEND = "http://localhost:8000";

function configuredApiUrl(): string | null {
  const raw = process.env.NEXT_PUBLIC_API_URL;
  if (raw == null) return null;
  const t = String(raw).trim();
  if (!t) return null;
  return t.replace(/\/$/, "");
}

/**
 * Base URL used for `fetch`.
 * - If `NEXT_PUBLIC_API_URL` is set (non-empty): call that origin directly.
 * - If unset in the browser: use same-origin paths `/api/...` so Next can rewrite to FastAPI.
 */
export function getApiBaseUrl(): string {
  const c = configuredApiUrl();
  if (c) return c;
  if (typeof window !== "undefined") return "";
  return DEFAULT_BACKEND;
}

/** Absolute origin for OpenAPI (/docs) and debugging. */
export function getBackendPublicOrigin(): string {
  return configuredApiUrl() ?? DEFAULT_BACKEND;
}

/** Human-readable hint for the dashboard. */
export function getApiRequestLabel(): string {
  const base = getApiBaseUrl();
  if (base) return `${base} (direct)`;
  if (typeof window !== "undefined") {
    return `${window.location.origin}/api/* → ${DEFAULT_BACKEND} (Next rewrite)`;
  }
  return DEFAULT_BACKEND;
}

function apiUrl(path: string): string {
  const base = getApiBaseUrl();
  const p = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${p}` : p;
}

export type ErrorWithHint = Error & { hint: string | null };

/** Parse FastAPI `detail` (string, validation array, or `{ message, hint }`). */
export function parseApiFailure(
  data: unknown,
  fallback: string,
): { message: string; hint: string | null } {
  if (data && typeof data === "object" && "detail" in data) {
    const d = (data as { detail: unknown }).detail;
    if (typeof d === "string") {
      return { message: d, hint: null };
    }
    if (d && typeof d === "object" && !Array.isArray(d)) {
      const o = d as Record<string, unknown>;
      const message =
        typeof o.message === "string" ? o.message : JSON.stringify(d);
      const hint =
        typeof o.hint === "string" && o.hint.trim() ? o.hint : null;
      return { message, hint };
    }
    if (Array.isArray(d)) {
      return {
        message: d
          .map((e) =>
            typeof e === "object" && e && "msg" in e
              ? String((e as { msg: string }).msg)
              : JSON.stringify(e),
          )
          .join("; "),
        hint: null,
      };
    }
  }
  return { message: fallback, hint: null };
}

export function getErrorHint(err: unknown): string | null {
  if (err instanceof Error && "hint" in err) {
    const h = (err as ErrorWithHint).hint;
    return typeof h === "string" && h.trim() ? h : null;
  }
  return null;
}

async function parseResponse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: unknown = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }
  }
  if (!res.ok) {
    const { message, hint } = parseApiFailure(
      data,
      res.statusText || "Request failed",
    );
    const err = new Error(message) as ErrorWithHint;
    err.hint = hint;
    throw err;
  }
  return data as T;
}

export async function listAlgorithms(): Promise<AlgorithmsResponse> {
  const res = await fetch(apiUrl("/api/algorithms"));
  return parseResponse<AlgorithmsResponse>(res);
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(apiUrl("/api/datasets/upload"), {
    method: "POST",
    body: fd,
  });
  return parseResponse<UploadResponse>(res);
}

export async function listDatasets(): Promise<{ datasets: DatasetInfo[] }> {
  const res = await fetch(apiUrl("/api/datasets"));
  return parseResponse(res);
}

export async function getDataset(datasetId: string): Promise<DatasetInfo> {
  const res = await fetch(
    apiUrl(`/api/datasets/${encodeURIComponent(datasetId)}`),
  );
  return parseResponse<DatasetInfo>(res);
}

export async function getDatasetPreview(
  datasetId: string,
  opts?: { sampleRows?: number; targetColumn?: string },
): Promise<DatasetPreview> {
  const params = new URLSearchParams();
  const n = opts?.sampleRows ?? 15;
  if (n >= 5 && n <= 20) params.set("sample_rows", String(n));
  if (opts?.targetColumn?.trim())
    params.set("target_column", opts.targetColumn.trim());
  const q = params.toString();
  const path = `/api/datasets/${encodeURIComponent(datasetId)}/preview${q ? `?${q}` : ""}`;
  const res = await fetch(apiUrl(path));
  return parseResponse<DatasetPreview>(res);
}

export async function startTrain(body: {
  dataset_id: string;
  mode: "supervised" | "unsupervised";
  algorithm: string;
  /** Train and compare multiple algorithms (one job each; supervised or unsupervised). */
  algorithms?: string[];
  target_column?: string;
  test_size?: number;
  unsupervised_family?: UnsupervisedFamily;
  exclude_columns?: string[];
  preprocessing?: PreprocessPayload | null;
}): Promise<TrainResponse> {
  const res = await fetch(apiUrl("/api/jobs/train"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: body.dataset_id,
      mode: body.mode,
      algorithm: body.algorithm,
      algorithms: body.algorithms ?? null,
      target_column: body.target_column ?? null,
      test_size: body.test_size ?? 0.2,
      unsupervised_family: body.unsupervised_family ?? null,
      exclude_columns: body.exclude_columns ?? [],
      preprocessing: body.preprocessing ?? null,
    }),
  });
  return parseResponse<TrainResponse>(res);
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const res = await fetch(
    apiUrl(`/api/jobs/${encodeURIComponent(jobId)}`),
  );
  return parseResponse<JobRecord>(res);
}

export async function listJobs(): Promise<{ jobs: JobRecord[] }> {
  const res = await fetch(apiUrl("/api/jobs"));
  return parseResponse(res);
}

export function modelBundleDownloadUrl(jobId: string): string {
  return apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/download`);
}

export async function predict(
  jobId: string,
  rows: Record<string, unknown>[],
): Promise<PredictResponse> {
  const res = await fetch(apiUrl("/api/predict"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, rows }),
  });
  return parseResponse<PredictResponse>(res);
}

/** @deprecated use getApiBaseUrl */
export function getBaseUrl(): string {
  return getApiBaseUrl() || DEFAULT_BACKEND;
}
