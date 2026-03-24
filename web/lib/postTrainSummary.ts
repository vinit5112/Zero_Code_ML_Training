import type { JobRecord, LearningMode, UnsupervisedFamily } from "./types";

function n(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

function fmtNum(x: number, d = 4): string {
  if (Math.abs(x) >= 1000 || (Math.abs(x) < 0.0001 && x !== 0))
    return x.toExponential(3);
  return x.toFixed(d).replace(/\.?0+$/, "") || "0";
}

/** Plain-language headline metrics from training job + optional UI fallbacks. */
export function buildPostTrainSummary(
  job: JobRecord,
  opts: {
    learningMode: LearningMode | null;
    targetColumn: string;
    fallbackFeatureColumns: string[];
  },
): {
  modelLine: string;
  metricLine: string | null;
  featuresLine: string;
} {
  const modelName = job.algorithm_label ?? job.algorithm_id ?? "Model";
  const feats =
    job.feature_columns?.length ? job.feature_columns : opts.fallbackFeatureColumns;

  const task = job.task;
  let taskPhrase: string;
  if (task === "classification") {
    taskPhrase = "classification (predict categories)";
  } else if (task === "regression") {
    taskPhrase = "regression (predict numbers)";
  } else if (task === "unsupervised") {
    const fam = job.unsupervised_family as UnsupervisedFamily | undefined;
    const famLabel =
      fam === "clustering"
        ? "clustering"
        : fam === "decomposition"
          ? "dimensionality reduction"
          : fam === "anomaly"
            ? "anomaly detection"
            : "unsupervised";
    taskPhrase = `${famLabel} (no target column)`;
  } else {
    taskPhrase = task ?? "unknown";
  }

  const modelLine = `Model: ${modelName} · Task: ${taskPhrase}`;

  const m = job.metrics ?? {};
  let metricLine: string | null = null;

  if (task === "regression") {
    const rmse = n(m.rmse);
    const r2 = n(m.r2);
    const mae = n(m.mae);
    const parts: string[] = [];
    if (rmse != null) parts.push(`test RMSE: ${fmtNum(rmse)}`);
    if (mae != null) parts.push(`MAE: ${fmtNum(mae)}`);
    if (r2 != null) parts.push(`R²: ${fmtNum(r2, 3)}`);
    if (parts.length) metricLine = parts.join(" · ");
  } else if (task === "classification") {
    const acc = n(m.accuracy);
    const f1 = n(m.f1_macro);
    const parts: string[] = [];
    if (acc != null) parts.push(`test accuracy: ${(acc * 100).toFixed(1)}%`);
    if (f1 != null) parts.push(`macro F1: ${fmtNum(f1, 3)}`);
    if (parts.length) metricLine = parts.join(" · ");
  } else if (task === "unsupervised") {
    const sil = n(m.silhouette);
    const ev = n(m.explained_variance_ratio_sum);
    const outF = n(m.outlier_fraction_fit_data);
    const parts: string[] = [];
    if (sil != null) parts.push(`silhouette: ${fmtNum(sil, 3)}`);
    if (ev != null) parts.push(`variance explained (sum): ${fmtNum(ev, 3)}`);
    if (outF != null) parts.push(`outlier share (on training rows): ${(outF * 100).toFixed(1)}%`);
    if (m.n_rows != null && typeof m.n_rows === "number")
      parts.push(`${Math.round(m.n_rows)} rows used`);
    metricLine = parts.length ? parts.join(" · ") : null;
  }

  const nFeat = feats.length;
  let featuresLine: string;
  if (nFeat === 0) {
    featuresLine = "Features used: (see downloaded metadata.json)";
  } else if (nFeat <= 5) {
    featuresLine = `Features used (${nFeat}): ${feats.join(", ")}`;
  } else {
    featuresLine = `Features used (${nFeat}): ${feats.slice(0, 4).join(", ")} … +${nFeat - 4} more`;
  }

  if (opts.learningMode === "supervised" && opts.targetColumn) {
    featuresLine += ` · Target: ${opts.targetColumn}`;
  }

  return { modelLine, metricLine, featuresLine };
}
