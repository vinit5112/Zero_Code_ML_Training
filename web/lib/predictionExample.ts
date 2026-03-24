import type { DatasetColumnProfile, JobRecord } from "./types";

/** Feature names the predict API expects (same order as training / metadata). */
export function resolvePredictionFeatureNames(
  job: JobRecord,
  fallbackFeatures: string[],
): string[] {
  const f = job.feature_columns;
  if (f && f.length > 0) return [...f];
  return [...fallbackFeatures];
}

/** Example row stored on the job and in downloaded metadata.json (if trained with current backend). */
export function predictionExampleFromJob(
  job: JobRecord,
): Record<string, unknown> | null {
  const row = job.prediction_example_row;
  if (!row || typeof row !== "object" || Array.isArray(row)) return null;
  return row as Record<string, unknown>;
}

function placeholderForProfile(
  col: DatasetColumnProfile | undefined,
): string | number | boolean | null {
  if (!col) return 0;
  const d = col.dtype.toLowerCase();
  if (d === "bool" || d === "boolean") return false;
  if (
    d.includes("int") ||
    d.includes("float") ||
    d.includes("number") ||
    d.includes("double")
  ) {
    return 0;
  }
  return "example";
}

/**
 * When metadata has no example row (e.g. old jobs), build one from preview or placeholders.
 */
export function buildFallbackExampleRow(
  featureNames: string[],
  sampleRow: Record<string, unknown> | undefined,
  columnProfiles: DatasetColumnProfile[] | undefined,
): Record<string, unknown> {
  const byName = new Map(columnProfiles?.map((c) => [c.name, c]));
  const out: Record<string, unknown> = {};
  for (const name of featureNames) {
    if (sampleRow && Object.prototype.hasOwnProperty.call(sampleRow, name)) {
      const v = sampleRow[name];
      out[name] = v === undefined ? null : v;
    } else {
      out[name] = placeholderForProfile(byName.get(name));
    }
  }
  return out;
}

/** One object per feature column, in training column order (stable, readable JSON). */
export function orderRowToFeatureColumns(
  featureNames: string[],
  row: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const name of featureNames) {
    out[name] = Object.prototype.hasOwnProperty.call(row, name)
      ? row[name]
      : null;
  }
  return out;
}

export function formatPredictJsonArray(row: Record<string, unknown>): string {
  return JSON.stringify([row], null, 2);
}
