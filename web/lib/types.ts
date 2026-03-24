export type JobStatusValue = "queued" | "running" | "completed" | "failed";

export type AlgorithmKind = "classification" | "regression" | "both";

export type LearningMode = "supervised" | "unsupervised";

export type UnsupervisedFamily = "clustering" | "decomposition" | "anomaly";

export type UnsupervisedInference = "predict" | "transform" | "none";

export interface SupervisedAlgorithmMeta {
  id: string;
  label: string;
  kind: AlgorithmKind;
  scale_numeric: boolean;
}

export interface UnsupervisedAlgorithmMeta {
  id: string;
  label: string;
  scale_numeric: boolean;
  inference: UnsupervisedInference;
}

export interface AlgorithmsResponse {
  supervised: SupervisedAlgorithmMeta[];
  unsupervised: Record<UnsupervisedFamily, UnsupervisedAlgorithmMeta[]>;
}

export interface DatasetInfo {
  dataset_id: string;
  filename: string;
  columns: string[];
  approx_rows: number;
}

export interface UploadResponse {
  dataset_id: string;
  columns: string[];
}

export interface ColumnTopValue {
  value: string;
  count: number;
}

/** Per-column EDA from the preview API. */
export type ColumnSummary =
  | {
      kind: "numeric";
      min?: number | null;
      max?: number | null;
      mean?: number | null;
      median?: number | null;
      std?: number | null;
      zero_pct?: number;
      non_null_count?: number;
      note?: string;
    }
  | {
      kind: "categorical";
      top_values: ColumnTopValue[];
      non_null_count?: number;
    };

export interface DatasetColumnProfile {
  name: string;
  dtype: string;
  missing_count: number;
  missing_pct: number;
  n_unique: number;
  summary?: ColumnSummary;
}

/** Pearson correlation among numeric columns: full heatmap or top pairs when many numerics. */
export type NumericCorrelationBlock =
  | {
      format: "matrix";
      columns: string[];
      matrix: (number | null)[][];
    }
  | {
      format: "top_pairs";
      pairs: Array<{ column_a: string; column_b: string; r: number }>;
      note?: string;
    };

/** Univariate association with the supervised target (MI ± Pearson where computed). */
export interface FeatureTargetAssociationRow {
  column: string;
  pearson_r: number | null;
  mutual_info: number | null;
  note?: string;
}

/** Aggregated histogram + KDE (numeric) or value counts (categorical) for charting. */
export interface DistributionNumericChart {
  kind: "numeric";
  non_null_count: number;
  points: Array<{
    x_lo: number;
    x_hi: number;
    count: number;
    kde: number | null;
  }>;
  kde_x: number[];
  kde_y: number[];
  x_min: number;
  x_max: number;
}

export interface DistributionCategoricalChart {
  kind: "categorical";
  non_null_count: number;
  bars: Array<{ label: string; count: number }>;
}

export type ColumnDistributionChart =
  | DistributionNumericChart
  | DistributionCategoricalChart;

export interface DistributionChartsPayload {
  columns: Record<string, ColumnDistributionChart>;
  target: ColumnDistributionChart | null;
  truncated: boolean;
  note?: string | null;
}

/** Cheap data-quality hints (heuristic; not a substitute for domain review). */
export interface DataQualityPayload {
  duplicate_row_count: number;
  duplicate_row_pct: number;
  constant_columns: Array<{ column: string; value_sample: string }>;
  high_cardinality_categoricals: Array<{
    column: string;
    n_unique: number;
    unique_to_rows_ratio: number;
    note: string;
  }>;
  id_like_columns: Array<{
    column: string;
    n_unique: number;
    non_null_count: number;
    unique_ratio: number;
  }>;
  leaky_column_hints: Array<{
    column: string;
    name_pattern: string;
    note: string;
  }>;
  heuristic_warning: string;
}

export interface DatasetPreview {
  n_rows: number;
  n_columns: number;
  sample_rows: Record<string, unknown>[];
  sample_row_count: number;
  columns: DatasetColumnProfile[];
  target_column?: string;
  inferred_task?: "classification" | "regression";
  target_task_explanation?: string;
  target_n_unique?: number;
  target_missing_count?: number;
  target_missing_pct?: number;
  class_distribution?: Record<string, number> | null;
  imbalance_ratio_max_to_min?: number | null;
  minority_class_pct?: number | null;
  imbalance_warning?: boolean;
  numeric_outlier_summary?: Array<{
    column: string;
    iqr_outlier_pct?: number | null;
    note?: string;
  }>;
  numeric_correlation?: NumericCorrelationBlock | null;
  feature_target_association?: FeatureTargetAssociationRow[];
  distribution_charts?: DistributionChartsPayload;
  data_quality?: DataQualityPayload;
}

export interface JobRecord {
  job_id: string;
  status: JobStatusValue;
  message?: string | null;
  metrics?: Record<string, unknown> | null;
  task?: string | null;
  dataset_id?: string | null;
  target_column?: string | null;
  algorithm_id?: string | null;
  algorithm_label?: string | null;
  learning_mode?: LearningMode | null;
  unsupervised_family?: UnsupervisedFamily | null;
  inference_method?: UnsupervisedInference | string | null;
  feature_columns?: string[] | null;
  /** First training feature row; same as metadata.json `prediction_example_row`. */
  prediction_example_row?: Record<string, unknown> | null;
  /** Short fix suggestion when status is failed (or null). */
  user_hint?: string | null;
  preprocess_applied?: Record<string, unknown> | null;
}

export type ScalingOption = "auto" | "none" | "standard" | "minmax" | "robust";

export interface PreprocessPayload {
  feature_columns?: string[] | null;
  scaling: ScalingOption;
  numeric_outliers: "none" | "clip_iqr";
  imbalance: "none" | "smote";
  numeric_bins: Array<{
    column: string;
    n_bins: number;
    strategy: "uniform" | "quantile";
  }>;
}

export interface TrainResponse {
  job_id: string;
  /** Present when multiple supervised models were trained in one run. */
  job_ids?: string[];
}

export interface PredictResponse {
  predictions: unknown[];
  inference_method?: string;
}
