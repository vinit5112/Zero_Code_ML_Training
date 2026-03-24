"use client";

import { DataPreviewPanel } from "@/components/data-preview";
import {
  getApiRequestLabel,
  getBackendPublicOrigin,
  getDataset,
  getDatasetPreview,
  getErrorHint,
  getJob,
  listAlgorithms,
  listDatasets,
  listJobs,
  modelBundleDownloadUrl,
  predict,
  startTrain,
  uploadDataset,
} from "@/lib/api";
import {
  buildFallbackExampleRow,
  formatPredictJsonArray,
  orderRowToFeatureColumns,
  predictionExampleFromJob,
  resolvePredictionFeatureNames,
} from "@/lib/predictionExample";
import { buildPostTrainSummary } from "@/lib/postTrainSummary";
import type {
  AlgorithmsResponse,
  DatasetColumnProfile,
  DatasetInfo,
  DatasetPreview,
  JobRecord,
  LearningMode,
  PreprocessPayload,
  ScalingOption,
  UnsupervisedFamily,
} from "@/lib/types";
import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

function cn(...parts: (string | false | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

function Badge({ children, tone }: { children: ReactNode; tone: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold tracking-wide",
        tone,
      )}
    >
      {children}
    </span>
  );
}

function jobTone(status: JobRecord["status"]): string {
  switch (status) {
    case "completed":
      return "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/25";
    case "failed":
      return "bg-rose-500/15 text-rose-300 ring-1 ring-rose-500/25";
    case "running":
      return "bg-teal-500/15 text-teal-300 ring-1 ring-teal-400/30";
    default:
      return "bg-amber-500/15 text-amber-200 ring-1 ring-amber-500/25";
  }
}

function groupSupervised(list: AlgorithmsResponse["supervised"]) {
  const by = (k: "classification" | "regression" | "both") =>
    list.filter((a) => a.kind === k).sort((a, b) => a.label.localeCompare(b.label));
  return {
    classification: by("classification"),
    regression: by("regression"),
    both: by("both"),
  };
}

function supervisedIdsForTask(
  grouped: ReturnType<typeof groupSupervised>,
  task: "classification" | "regression" | undefined,
): string[] {
  if (task === "classification") {
    return [...grouped.classification, ...grouped.both].map((a) => a.id);
  }
  if (task === "regression") {
    return [...grouped.regression, ...grouped.both].map((a) => a.id);
  }
  const merged = [
    ...grouped.classification,
    ...grouped.regression,
    ...grouped.both,
  ];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const a of merged.sort((x, y) => x.label.localeCompare(y.label))) {
    if (!seen.has(a.id)) {
      seen.add(a.id);
      out.push(a.id);
    }
  }
  return out;
}

function formatJobMetricsLine(job: JobRecord): string {
  const m = job.metrics;
  if (!m || typeof m !== "object") return "—";
  if (job.task === "classification") {
    const acc = m.accuracy;
    const f1 = m.f1_macro;
    const parts: string[] = [];
    if (typeof acc === "number") parts.push(`acc ${(acc * 100).toFixed(1)}%`);
    if (typeof f1 === "number") parts.push(`F1 ${f1.toFixed(3)}`);
    return parts.join(" · ") || "—";
  }
  if (job.task === "regression") {
    const r2 = m.r2;
    const rmse = m.rmse;
    const parts: string[] = [];
    if (typeof r2 === "number") parts.push(`R² ${r2.toFixed(4)}`);
    if (typeof rmse === "number") parts.push(`RMSE ${rmse.toFixed(4)}`);
    return parts.join(" · ") || "—";
  }
  if (job.task === "unsupervised" || job.learning_mode === "unsupervised") {
    const parts: string[] = [];
    if (typeof m.silhouette === "number")
      parts.push(`silhouette ${m.silhouette.toFixed(3)}`);
    if (typeof m.reconstruction_error === "number")
      parts.push(`recon ${Number(m.reconstruction_error).toExponential(2)}`);
    if (typeof m.explained_variance_ratio_sum === "number")
      parts.push(`EV Σ ${m.explained_variance_ratio_sum.toFixed(3)}`);
    if (typeof m.rows_flagged_outlier === "number")
      parts.push(`outliers ${m.rows_flagged_outlier}`);
    return parts.join(" · ") || "—";
  }
  return "—";
}

const WIZARD_STEP_COUNT = 7;

function stepTitle(step: number, mode: LearningMode | null): string {
  const titles = [
    "Choose your goal",
    "Upload your CSV",
    mode === "unsupervised"
      ? "Columns & analysis type"
      : "Choose target column",
    "Features & preprocessing",
    "Pick algorithm(s)",
    "Train model(s)",
    "Results & next steps",
  ];
  return titles[step] ?? "";
}

/** One-line hint for progress pills (hover to read). */
function stepTooltip(step: number, mode: LearningMode | null): string {
  const tips = [
    "Decide if you have a target column to predict (supervised) or not (unsupervised).",
    "Load a CSV file or reuse a dataset id you already uploaded in this session.",
    mode === "unsupervised"
      ? "Choose clustering, dimension reduction, or anomaly detection; optionally exclude columns."
      : "Pick the column the model should learn to predict; everything else can be used as input.",
    "Toggle which columns are features and set scaling, outlier handling, resampling, and binning.",
    "Supervised: pick one or more models and holdout %. Unsupervised: pick one or more models in the family you chose earlier.",
    "Confirm your settings; each selected model becomes its own server job (compare on the results step).",
    "Read metrics, download the bundle for local prediction, or try the API from this page.",
  ];
  return tips[step] ?? "";
}

function isNumericProfile(p: DatasetColumnProfile): boolean {
  const d = p.dtype.toLowerCase();
  if (d.includes("bool")) return false;
  return (
    d.includes("int") ||
    d.includes("float") ||
    d.includes("number") ||
    d.includes("double")
  );
}

export default function Dashboard() {
  const [error, setError] = useState<string | null>(null);
  const [errorHint, setErrorHint] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [wizardStep, setWizardStep] = useState(0);
  const [learningMode, setLearningMode] = useState<LearningMode | null>(null);

  const [algorithms, setAlgorithms] = useState<AlgorithmsResponse | null>(null);
  const [unsupervisedFamily, setUnsupervisedFamily] =
    useState<UnsupervisedFamily>("clustering");
  const [algorithm, setAlgorithm] = useState("random_forest");
  /** Supervised: one or more models to train in parallel for comparison. */
  const [selectedSupervisedAlgorithms, setSelectedSupervisedAlgorithms] = useState<
    string[]
  >(["random_forest"]);
  const [selectedUnsupervisedAlgorithms, setSelectedUnsupervisedAlgorithms] =
    useState<string[]>([]);
  const [trainJobIds, setTrainJobIds] = useState<string[]>([]);
  const [batchJobs, setBatchJobs] = useState<JobRecord[]>([]);

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);

  const [datasetId, setDatasetId] = useState("");
  const [columns, setColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [excludeColumnsRaw, setExcludeColumnsRaw] = useState("");
  const [testSize, setTestSize] = useState(0.2);

  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [scalingMode, setScalingMode] = useState<ScalingOption>("auto");
  const [numericOutliers, setNumericOutliers] = useState<"none" | "clip_iqr">(
    "none",
  );
  const [imbalanceMode, setImbalanceMode] = useState<"none" | "smote">("none");
  const [binRows, setBinRows] = useState<
    Array<{
      id: string;
      column: string;
      n_bins: number;
      strategy: "quantile" | "uniform";
    }>
  >([]);

  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);

  const [predictJson, setPredictJson] = useState(
    '[\n  { "example_feature": 1 }\n]',
  );
  const [predictOut, setPredictOut] = useState<string | null>(null);
  const [lastInferenceMethod, setLastInferenceMethod] = useState<string | null>(
    null,
  );

  const [showLibrary, setShowLibrary] = useState(false);

  const [dataPreview, setDataPreview] = useState<DatasetPreview | null>(null);
  const [dataPreviewLoading, setDataPreviewLoading] = useState(false);
  const [dataPreviewError, setDataPreviewError] = useState<string | null>(null);

  /** Dataset preview for the completed job's CSV (fallback example row if job lacks metadata example). */
  const [completionPreview, setCompletionPreview] =
    useState<DatasetPreview | null>(null);
  const [predictHelperCopied, setPredictHelperCopied] = useState(false);

  function clearErrors() {
    setError(null);
    setErrorHint(null);
  }

  function surfaceClientError(message: string) {
    setError(message);
    setErrorHint(null);
  }

  function surfaceThrownError(e: unknown, fallback: string) {
    if (e instanceof Error) {
      setError(e.message);
      setErrorHint(getErrorHint(e));
      return;
    }
    setError(fallback);
    setErrorHint(null);
  }

  const supervisedGrouped = useMemo(
    () =>
      algorithms ? groupSupervised(algorithms.supervised) : null,
    [algorithms],
  );

  const unsupervisedList = useMemo(() => {
    if (!algorithms) return [];
    return algorithms.unsupervised[unsupervisedFamily] ?? [];
  }, [algorithms, unsupervisedFamily]);

  const allowedSupervisedIds = useMemo(() => {
    if (!supervisedGrouped) return [] as string[];
    return supervisedIdsForTask(
      supervisedGrouped,
      dataPreview?.inferred_task,
    );
  }, [supervisedGrouped, dataPreview?.inferred_task]);

  useEffect(() => {
    if (!supervisedGrouped || allowedSupervisedIds.length === 0) return;
    const allowed = new Set(allowedSupervisedIds);
    setSelectedSupervisedAlgorithms((prev) => {
      const next = prev.filter((id) => allowed.has(id));
      if (next.length > 0) return next;
      const prefer =
        allowed.has("random_forest") ? "random_forest" : allowedSupervisedIds[0];
      return prefer ? [prefer] : [];
    });
  }, [supervisedGrouped, allowedSupervisedIds]);

  const featureCandidates = useMemo(() => {
    if (learningMode === "supervised" && targetColumn) {
      return columns.filter((c) => c !== targetColumn);
    }
    if (learningMode === "unsupervised") {
      const ex = new Set(
        excludeColumnsRaw
          .split(/[,;\n]/)
          .map((s) => s.trim())
          .filter(Boolean),
      );
      return columns.filter((c) => !ex.has(c));
    }
    return [];
  }, [learningMode, targetColumn, columns, excludeColumnsRaw]);

  const numericColumnOptions = useMemo(() => {
    if (!dataPreview?.columns) return [] as string[];
    const numericNames = new Set(
      dataPreview.columns.filter(isNumericProfile).map((c) => c.name),
    );
    return featureCandidates.filter((c) => numericNames.has(c));
  }, [dataPreview?.columns, featureCandidates]);

  useEffect(() => {
    if (learningMode !== "unsupervised" || unsupervisedList.length === 0) return;
    const allowed = new Set(unsupervisedList.map((a) => a.id));
    setSelectedUnsupervisedAlgorithms((prev) => {
      const next = prev.filter((id) => allowed.has(id));
      if (next.length > 0) return next;
      return [unsupervisedList[0].id];
    });
  }, [learningMode, unsupervisedList]);

  useEffect(() => {
    if (learningMode !== "unsupervised" || selectedUnsupervisedAlgorithms.length === 0)
      return;
    const head = selectedUnsupervisedAlgorithms[0];
    if (head !== algorithm) setAlgorithm(head);
  }, [learningMode, selectedUnsupervisedAlgorithms, algorithm]);

  const refreshLists = useCallback(async () => {
    try {
      const [d, j] = await Promise.all([listDatasets(), listJobs()]);
      setDatasets(d.datasets);
      setJobs(j.jobs);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const data = await listAlgorithms();
        setAlgorithms(data);
        setAlgorithm("random_forest");
        setSelectedSupervisedAlgorithms(["random_forest"]);
      } catch {
        surfaceClientError(
          "Could not load algorithms from API. Is the backend running?",
        );
      }
    })();
  }, []);

  useEffect(() => {
    void refreshLists();
  }, [refreshLists]);

  useEffect(() => {
    if (!datasetId.trim() || wizardStep < 2 || wizardStep > 5) {
      if (wizardStep < 2) {
        setDataPreview(null);
        setDataPreviewError(null);
      }
      return;
    }
    let cancelled = false;
    (async () => {
      setDataPreviewLoading(true);
      setDataPreviewError(null);
      try {
        const p = await getDatasetPreview(datasetId.trim(), {
          sampleRows: 15,
          targetColumn:
            learningMode === "supervised" && targetColumn.trim()
              ? targetColumn.trim()
              : undefined,
        });
        if (!cancelled) setDataPreview(p);
      } catch (e) {
        if (!cancelled) {
          setDataPreview(null);
          setDataPreviewError(
            e instanceof Error ? e.message : "Could not load preview",
          );
        }
      } finally {
        if (!cancelled) setDataPreviewLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [datasetId, wizardStep, learningMode, targetColumn]);

  useEffect(() => {
    if (wizardStep !== 6 || !job?.dataset_id || job.status !== "completed") {
      if (wizardStep !== 6) setCompletionPreview(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const tc = job.target_column?.trim();
        const p = await getDatasetPreview(job.dataset_id!.trim(), {
          sampleRows: 15,
          targetColumn:
            job.task === "classification" || job.task === "regression"
              ? tc || undefined
              : undefined,
        });
        if (!cancelled) setCompletionPreview(p);
      } catch {
        if (!cancelled) setCompletionPreview(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [wizardStep, job?.dataset_id, job?.status, job?.task, job?.target_column]);

  useEffect(() => {
    const ids =
      trainJobIds.length > 0
        ? trainJobIds
        : jobId.trim()
          ? [jobId.trim()]
          : [];
    if (ids.length === 0) {
      setBatchJobs([]);
      setJob(null);
      return;
    }
    let cancelled = false;
    const tick = async () => {
      const rows = await Promise.all(
        ids.map((i) => getJob(i).catch(() => null)),
      );
      if (cancelled) return;
      const ok = rows.filter((x): x is JobRecord => x != null);
      setBatchJobs(ok);
      const jid = jobId.trim();
      const pick =
        (jid ? ok.find((j) => j.job_id === jid) : undefined) ?? ok[0] ?? null;
      setJob(pick);
    };
    void tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [trainJobIds, jobId]);

  function resetWizard() {
    setWizardStep(0);
    setLearningMode(null);
    setDatasetId("");
    setColumns([]);
    setTargetColumn("");
    setExcludeColumnsRaw("");
    setJobId("");
    setJob(null);
    setTrainJobIds([]);
    setBatchJobs([]);
    setPredictOut(null);
    setLastInferenceMethod(null);
    setPredictJson('[{ "your_feature_column": 0 }]');
    setAlgorithm("random_forest");
    setSelectedSupervisedAlgorithms(["random_forest"]);
    setSelectedUnsupervisedAlgorithms([]);
    clearErrors();
    setDataPreview(null);
    setDataPreviewError(null);
    setCompletionPreview(null);
    setPredictHelperCopied(false);
    setSelectedFeatures([]);
    setScalingMode("auto");
    setNumericOutliers("none");
    setImbalanceMode("none");
    setBinRows([]);
  }

  function choosePath(mode: LearningMode) {
    setLearningMode(mode);
    clearErrors();
    if (mode === "supervised") {
      setAlgorithm("random_forest");
      setSelectedSupervisedAlgorithms(["random_forest"]);
    } else if (algorithms) {
      const first = algorithms.unsupervised[unsupervisedFamily][0];
      if (first) {
        setAlgorithm(first.id);
        setSelectedUnsupervisedAlgorithms([first.id]);
      }
    }
    setWizardStep(1);
  }

  async function onUpload(file: File | null) {
    clearErrors();
    if (!file) return;
    setBusy("upload");
    try {
      const res = await uploadDataset(file);
      setDatasetId(res.dataset_id);
      setColumns(res.columns);
      setTargetColumn((prev) =>
        prev && res.columns.includes(prev) ? prev : (res.columns[0] ?? ""),
      );
      await refreshLists();
      setWizardStep(2);
    } catch (e) {
      surfaceThrownError(e, "Upload failed");
    } finally {
      setBusy(null);
    }
  }

  function parseExcludeColumns(): string[] {
    return excludeColumnsRaw
      .split(/[,;\n]/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function toggleSupervisedAlgorithm(id: string) {
    setSelectedSupervisedAlgorithms((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 1) return prev;
        return prev.filter((x) => x !== id);
      }
      if (prev.length >= 12) return prev;
      return [...prev, id];
    });
  }

  function selectAllSupervisedForTask() {
    if (allowedSupervisedIds.length === 0) return;
    setSelectedSupervisedAlgorithms([...allowedSupervisedIds]);
  }

  function clearSupervisedToSingle() {
    const keep =
      selectedSupervisedAlgorithms.find((id) =>
        allowedSupervisedIds.includes(id),
      ) ?? allowedSupervisedIds[0];
    if (keep) setSelectedSupervisedAlgorithms([keep]);
  }

  function toggleUnsupervisedAlgorithm(id: string) {
    setSelectedUnsupervisedAlgorithms((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 1) return prev;
        return prev.filter((x) => x !== id);
      }
      if (prev.length >= 12) return prev;
      return [...prev, id];
    });
  }

  function selectAllUnsupervisedInFamily() {
    if (unsupervisedList.length === 0) return;
    setSelectedUnsupervisedAlgorithms(unsupervisedList.map((a) => a.id));
  }

  function clearUnsupervisedToSingle() {
    const keep =
      selectedUnsupervisedAlgorithms.find((id) =>
        unsupervisedList.some((a) => a.id === id),
      ) ?? unsupervisedList[0]?.id;
    if (keep) setSelectedUnsupervisedAlgorithms([keep]);
  }

  const pendingTrainCount =
    learningMode === "supervised"
      ? selectedSupervisedAlgorithms.length
      : learningMode === "unsupervised"
        ? selectedUnsupervisedAlgorithms.length
        : 1;

  function canProceedFrom(step: number): boolean {
    if (!learningMode && step > 0) return false;
    switch (step) {
      case 0:
        return false;
      case 1:
        return !!datasetId.trim();
      case 2:
        if (learningMode === "supervised") return !!targetColumn;
        {
          const ex = parseExcludeColumns();
          const left = columns.filter((c) => !ex.includes(c));
          return left.length > 0;
        }
      case 3:
        return selectedFeatures.length > 0;
      case 4:
        if (!algorithms) return false;
        if (learningMode === "supervised")
          return selectedSupervisedAlgorithms.length > 0;
        if (learningMode === "unsupervised")
          return selectedUnsupervisedAlgorithms.length > 0;
        return false;
      case 5:
        return true;
      default:
        return false;
    }
  }

  async function goNext() {
    clearErrors();
    if (wizardStep === 1 && datasetId.trim() && columns.length === 0) {
      setBusy("dataset");
      try {
        const d = await getDataset(datasetId.trim());
        setColumns(d.columns);
        setTargetColumn((prev) =>
          prev && d.columns.includes(prev) ? prev : (d.columns[0] ?? ""),
        );
      } catch (e) {
        surfaceThrownError(
          e,
          "Could not load that dataset ID. Upload a CSV or pick one from the library.",
        );
        setBusy(null);
        return;
      } finally {
        setBusy(null);
      }
    }
    if (!canProceedFrom(wizardStep)) {
      if (wizardStep === 2 && learningMode === "unsupervised") {
        surfaceClientError(
          "After exclusions there are no feature columns. Remove some names from the exclude list.",
        );
      } else if (wizardStep === 2 && learningMode === "supervised") {
        surfaceClientError(
          "Pick the column that contains your labels / outcome.",
        );
      } else if (wizardStep === 3) {
        surfaceClientError("Select at least one feature column.");
      }
      return;
    }
    if (wizardStep === 2) {
      setSelectedFeatures([...featureCandidates]);
    }
    if (wizardStep < WIZARD_STEP_COUNT - 1) {
      setWizardStep((s) => s + 1);
    }
  }

  function goBack() {
    clearErrors();
    if (wizardStep <= 0) return;
    if (wizardStep === 6) return;
    setWizardStep((s) => Math.max(0, s - 1));
  }

  async function onTrain() {
    clearErrors();
    if (!datasetId.trim() || !learningMode) {
      surfaceClientError("Missing dataset or workflow.");
      return;
    }
    if (learningMode === "supervised" && !targetColumn) {
      surfaceClientError("Select a target column.");
      return;
    }
    if (learningMode === "unsupervised") {
      const ex = parseExcludeColumns();
      const left = columns.filter((c) => !ex.includes(c));
      if (left.length === 0) {
        surfaceClientError("No feature columns left after exclusions.");
        return;
      }
    }
    if (selectedFeatures.length === 0) {
      surfaceClientError("No feature columns selected. Go back to preprocessing.");
      return;
    }

    const sameAll =
      selectedFeatures.length === featureCandidates.length &&
      featureCandidates.every((c) => selectedFeatures.includes(c));

    const imbalanceForTrain =
      learningMode === "unsupervised"
        ? "none"
        : dataPreview?.inferred_task === "regression"
          ? "none"
          : imbalanceMode;

    const preprocessing: PreprocessPayload = {
      feature_columns: sameAll ? null : [...selectedFeatures],
      scaling: scalingMode,
      numeric_outliers: numericOutliers,
      imbalance: imbalanceForTrain,
      numeric_bins: binRows
        .filter((b) => b.column && selectedFeatures.includes(b.column))
        .map((b) => ({
          column: b.column,
          n_bins: Math.min(50, Math.max(2, Math.round(b.n_bins))),
          strategy: b.strategy,
        })),
    };

    setBusy("train");
    try {
      const res = await startTrain({
        dataset_id: datasetId.trim(),
        mode: learningMode,
        algorithm:
          learningMode === "supervised"
            ? selectedSupervisedAlgorithms[0] ?? "random_forest"
            : selectedUnsupervisedAlgorithms[0] ?? algorithm,
        algorithms:
          learningMode === "supervised"
            ? selectedSupervisedAlgorithms
            : learningMode === "unsupervised"
              ? selectedUnsupervisedAlgorithms
              : undefined,
        target_column:
          learningMode === "supervised" ? targetColumn : undefined,
        test_size: testSize,
        unsupervised_family:
          learningMode === "unsupervised" ? unsupervisedFamily : undefined,
        exclude_columns:
          learningMode === "unsupervised" ? parseExcludeColumns() : undefined,
        preprocessing,
      });
      const ids =
        res.job_ids && res.job_ids.length > 0 ? res.job_ids : [res.job_id];
      setTrainJobIds(ids);
      setJobId(res.job_id);
      setLastInferenceMethod(null);
      await refreshLists();
      setWizardStep(6);
    } catch (e) {
      surfaceThrownError(e, "Training failed");
    } finally {
      setBusy(null);
    }
  }

  async function onPredict() {
    clearErrors();
    setPredictOut(null);
    setLastInferenceMethod(null);
    if (!jobId.trim()) {
      surfaceClientError("Job ID missing.");
      return;
    }
    let rows: Record<string, unknown>[];
    try {
      rows = JSON.parse(predictJson) as Record<string, unknown>[];
      if (!Array.isArray(rows)) rows = [rows as Record<string, unknown>];
    } catch {
      surfaceClientError("JSON must be an array of row objects.");
      return;
    }
    setBusy("predict");
    try {
      const res = await predict(jobId.trim(), rows);
      setPredictOut(JSON.stringify(res.predictions, null, 2));
      if (res.inference_method) setLastInferenceMethod(res.inference_method);
    } catch (e) {
      surfaceThrownError(e, "Predict failed");
    } finally {
      setBusy(null);
    }
  }

  const modeLabel =
    learningMode === "supervised"
      ? "Supervised (I have labels)"
      : learningMode === "unsupervised"
        ? "Unsupervised (no labels)"
        : "";

  const fallbackFeatureColumns = useMemo(() => {
    const excluded = excludeColumnsRaw
      .split(/[,;\n]/)
      .map((s) => s.trim())
      .filter(Boolean);
    const ex = new Set(excluded);
    if (learningMode === "supervised" && targetColumn) {
      return columns.filter((c) => c !== targetColumn);
    }
    if (learningMode === "unsupervised") {
      return columns.filter((c) => !ex.has(c));
    }
    return columns;
  }, [learningMode, targetColumn, columns, excludeColumnsRaw]);

  const predictionFeatures = useMemo(() => {
    if (!job || job.status !== "completed") return [];
    return resolvePredictionFeatureNames(job, fallbackFeatureColumns);
  }, [job, fallbackFeatureColumns]);

  const predictExampleJson = useMemo(() => {
    if (!job || job.status !== "completed" || predictionFeatures.length === 0) {
      return null;
    }
    const fromJob = predictionExampleFromJob(job);
    const sample = completionPreview?.sample_rows?.[0];
    const raw =
      fromJob && Object.keys(fromJob).length > 0
        ? fromJob
        : buildFallbackExampleRow(
            predictionFeatures,
            sample,
            completionPreview?.columns,
          );
    const ordered = orderRowToFeatureColumns(predictionFeatures, raw);
    return formatPredictJsonArray(ordered);
  }, [job, completionPreview, predictionFeatures]);

  async function copyPredictExampleJson() {
    if (!predictExampleJson) return;
    try {
      await navigator.clipboard.writeText(predictExampleJson);
      setPredictHelperCopied(true);
      window.setTimeout(() => setPredictHelperCopied(false), 2000);
    } catch {
      surfaceClientError("Could not copy to clipboard.");
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-10 px-5 py-12 sm:px-8 md:py-16 lg:max-w-5xl lg:px-12">
      <header className="animate-page-enter border-b border-[var(--border)] pb-10">
        <p className="text-xs font-bold uppercase tracking-[0.28em] text-[var(--muted)]">
        Guided training
        </p>
        <h1 className="mt-3 text-4xl font-extrabold leading-[1.08] tracking-tight md:text-5xl lg:text-[3.35rem]">
          Zero-code Machine Learning
        </h1>
        <div className="accent-rule mt-5" aria-hidden />
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)] md:text-lg">
          Follow the steps below. You can go back anytime before training
          finishes. Each step only asks for one kind of choice; hover a step
          number above for a short explanation of what that step is for.
        </p>
        <p className="mt-4 font-mono text-sm text-[var(--highlight-muted)]">
          API: {getApiRequestLabel()}
        </p>
      </header>

      {error && (
        <div
          className="animate-page-enter-delay-1 rounded-2xl border-2 border-rose-500/40 bg-rose-500/10 px-5 py-4 text-base text-rose-100"
          role="alert"
        >
          <p>{error}</p>
          {errorHint && (
            <p className="mt-2 border-t border-rose-500/20 pt-2 text-xs leading-relaxed text-rose-50/90">
              <span className="font-semibold text-amber-200/95">Tip: </span>
              {errorHint}
            </p>
          )}
        </div>
      )}

      {/* Progress */}
      <nav
        aria-label="Progress"
        className="animate-page-enter-delay-1 space-y-3"
      >
        <ol className="flex flex-wrap gap-2.5 md:gap-3">
          {Array.from({ length: WIZARD_STEP_COUNT }, (_, i) => {
            const done = wizardStep > i;
            const current = wizardStep === i;
            return (
              <li key={i}>
                <span
                  title={stepTooltip(i, learningMode)}
                  className={cn(
                    "inline-flex cursor-help items-center gap-2 rounded-full border-2 border-transparent px-4 py-2 text-sm font-semibold transition hover:brightness-110",
                    done &&
                      "border-emerald-500/25 bg-emerald-500/15 text-emerald-200",
                    current &&
                      "border-[var(--accent)]/50 bg-[var(--accent-dim)] text-[var(--foreground)] shadow-[0_0_24px_-4px_rgba(255,77,42,0.45)]",
                    !done &&
                      !current &&
                      "border-[var(--border)] bg-[var(--input-bg)] text-[var(--muted)]",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold",
                      current && "bg-[var(--accent)] text-[var(--accent-fg)]",
                      done && !current && "bg-emerald-500/30 text-emerald-100",
                      !done &&
                        !current &&
                        "bg-[var(--card-elevated)] text-[var(--muted)]",
                    )}
                  >
                    {i + 1}
                  </span>
                  <span className="max-w-[10.5rem] leading-tight sm:max-w-none">
                    {stepTitle(i, learningMode)}
                  </span>
                </span>
              </li>
            );
          })}
        </ol>
        {learningMode && wizardStep > 0 && (
          <p className="text-sm text-[var(--muted)]">
            Flow:{" "}
            <span className="font-semibold text-[var(--foreground)]">
              {modeLabel}
            </span>
          </p>
        )}
      </nav>

      {/* Step panels */}
      <div className="animate-page-enter-delay-2 rounded-[2rem] border-2 border-[var(--border)] bg-[var(--card)] p-7 shadow-[0_28px_90px_-20px_rgba(0,0,0,0.65)] sm:p-10 lg:p-12">
        {wizardStep >= 2 &&
          wizardStep <= 5 &&
          datasetId.trim() &&
          learningMode && (
            <div className="mb-10">
              <DataPreviewPanel
                preview={dataPreview}
                loading={dataPreviewLoading}
                error={dataPreviewError}
                title="Data preview"
                footer={
                  learningMode === "supervised" && !targetColumn ? (
                    <p className="mt-3 text-xs text-[var(--muted)]">
                      Select a target column below to see whether training will run as{" "}
                      <strong>classification</strong> or <strong>regression</strong>{" "}
                      (same rule as the trainer).
                    </p>
                  ) : null
                }
              />
            </div>
          )}

        {wizardStep === 0 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              What kind of problem is this?
            </h2>
            <p className="text-base leading-relaxed text-[var(--muted)] md:text-lg">
              Supervised learning needs a column that holds the correct answer
              (class or number). Unsupervised learning uses only the features —
              for grouping, compression, or finding outliers.
            </p>
            <div className="grid gap-5 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => choosePath("supervised")}
                className="group rounded-[1.75rem] border-2 border-[var(--border)] bg-[var(--input-bg)] p-8 text-left transition hover:border-[var(--accent)]/55 hover:bg-[var(--card-elevated)] hover:shadow-[0_20px_50px_-12px_rgba(255,77,42,0.2)]"
              >
                <span className="text-xl font-bold text-[var(--foreground)] md:text-2xl">
                  I have labels
                </span>
                <p className="mt-3 text-base text-[var(--muted)]">
                  Predict a target column (classification or regression). Example:
                  house price, spam vs not spam.
                </p>
                <span className="mt-6 inline-block text-base font-bold text-[var(--accent)] group-hover:underline">
                  Continue →
                </span>
              </button>
              <button
                type="button"
                onClick={() => choosePath("unsupervised")}
                className="group rounded-[1.75rem] border-2 border-[var(--border)] bg-[var(--input-bg)] p-8 text-left transition hover:border-[var(--highlight-muted)] hover:bg-[var(--card-elevated)] hover:shadow-[0_20px_50px_-12px_rgba(58,232,201,0.12)]"
              >
                <span className="text-xl font-bold text-[var(--foreground)] md:text-2xl">
                  I don&apos;t have labels
                </span>
                <p className="mt-3 text-base text-[var(--muted)]">
                  Cluster rows, reduce dimensions, or flag anomalies — no target
                  column required.
                </p>
                <span className="mt-6 inline-block text-base font-bold text-[var(--highlight)] group-hover:underline">
                  Continue →
                </span>
              </button>
            </div>
          </div>
        )}

        {wizardStep === 1 && learningMode && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              Upload your CSV
            </h2>
            <p className="text-sm text-[var(--muted)]">
              First row must be column names; each following row is one training
              example. The app stores the file under a new{" "}
              <span className="font-mono text-[var(--foreground)]">dataset_id</span>{" "}
              you will use for training and preview. After a successful upload,
              the wizard advances automatically.
            </p>
            <label className="block">
              <input
                type="file"
                accept=".csv"
                disabled={busy === "upload"}
                onChange={(e) => void onUpload(e.target.files?.[0] ?? null)}
                className="block w-full cursor-pointer rounded-2xl border-2 border-dashed border-[var(--border)] bg-[var(--input-bg)] px-6 py-14 text-center text-base file:mr-4 file:rounded-xl file:border-0 file:bg-[var(--accent)] file:px-6 file:py-3 file:text-base file:font-bold file:text-[var(--accent-fg)]"
              />
            </label>
            <div>
              <label className="mb-1 block text-xs text-[var(--muted)]">
                Or paste existing dataset ID
              </label>
              <p className="mb-1 text-[11px] leading-relaxed text-[var(--muted)]">
                Use this if you already uploaded in this browser session and
                copied the id, or you are resuming without re-uploading the same
                CSV.
              </p>
              <input
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                placeholder="dataset_id"
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-sm"
              />
            </div>
            {datasetId && (
              <p className="font-mono text-xs text-[var(--muted)]">
                Current ID:{" "}
                <span className="text-[var(--foreground)]">{datasetId}</span>
                {columns.length > 0 && (
                  <span className="block mt-1 text-[var(--muted)]">
                    {columns.length} columns detected
                  </span>
                )}
              </p>
            )}
          </div>
        )}

        {wizardStep === 2 && learningMode === "supervised" && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              Which column is the target?
            </h2>
            <p className="text-sm text-[var(--muted)]">
              The <strong>target</strong> is the correct answer for each row: a
              category (classification) or a number (regression). The trainer
              automatically picks classification when the target has few enough
              distinct values; everything else becomes an input feature unless
              you turn it off later.
            </p>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full rounded-2xl border-2 border-[var(--border)] bg-[var(--input-bg)] px-4 py-3.5 text-base"
            >
              <option value="">Select column…</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        )}

        {wizardStep === 2 && learningMode === "unsupervised" && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              Columns &amp; analysis type
            </h2>
            <p className="text-sm text-[var(--muted)]">
              Unsupervised mode never uses a target: the model only looks at the
              feature columns you leave in. Exclusions are for identifiers or
              columns you do not want in the math (for example raw text ids).
            </p>
            <div>
              <label className="mb-1 block text-sm text-[var(--muted)]">
                What do you want to do?
              </label>
              <select
                value={unsupervisedFamily}
                onChange={(e) =>
                  setUnsupervisedFamily(e.target.value as UnsupervisedFamily)
                }
                className="w-full rounded-2xl border-2 border-[var(--border)] bg-[var(--input-bg)] px-4 py-3.5 text-base"
              >
                <option value="clustering">Group similar rows (clustering)</option>
                <option value="decomposition">
                  Compress / embed columns (PCA, NMF, …)
                </option>
                <option value="anomaly">Find unusual rows (outliers)</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm text-[var(--muted)]">
                Columns to exclude (optional)
              </label>
              <input
                value={excludeColumnsRaw}
                onChange={(e) => setExcludeColumnsRaw(e.target.value)}
                placeholder="id, row_id, …"
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-sm"
              />
              <p className="mt-1 text-xs text-[var(--muted)]">
                Everything else in the CSV is used as a feature.
              </p>
            </div>
          </div>
        )}

        {wizardStep === 3 && learningMode && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
                Features &amp; preprocessing
              </h2>
              <p className="mt-2 text-sm text-[var(--muted)]">
                <strong>Features</strong> are the inputs the model may use.
                Unchecked columns are ignored for training and must not be sent
                at prediction time. Text-like columns are ordinal-encoded inside
                the saved pipeline (same encoding when you run predict). Every
                choice in this step is stored in the bundle so local prediction
                matches what you trained here.
              </p>
            </div>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold text-[var(--foreground)]">
                Feature columns
              </h3>
              <p className="text-xs text-[var(--muted)]">
                {selectedFeatures.length} of {featureCandidates.length} selected.
                Use Select all / Clear, or tick individual columns. Leave out
                identifiers and leaky fields you identified in the data preview.
              </p>
              <div className="max-h-48 space-y-2 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-3">
                {featureCandidates.length === 0 ? (
                  <p className="text-sm text-[var(--muted)]">
                    No candidate columns. Go back and set target / exclusions.
                  </p>
                ) : (
                  featureCandidates.map((c) => (
                    <label
                      key={c}
                      className="flex cursor-pointer items-center gap-2 text-sm"
                    >
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(c)}
                        onChange={() => {
                          setSelectedFeatures((prev) =>
                            prev.includes(c)
                              ? prev.filter((x) => x !== c)
                              : [...prev, c],
                          );
                        }}
                        className="rounded border-[var(--border)]"
                      />
                      <span className="font-mono text-xs">{c}</span>
                    </label>
                  ))
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedFeatures([...featureCandidates])}
                  className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs"
                >
                  Select all
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedFeatures([])}
                  className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs"
                >
                  Clear
                </button>
              </div>
            </section>

            {learningMode === "supervised" &&
              dataPreview?.inferred_task === "classification" && (
                <div
                  className={cn(
                    "rounded-xl border p-4 text-sm",
                    dataPreview.imbalance_warning
                      ? "border-amber-500/35 bg-amber-500/10"
                      : "border-[var(--border)] bg-[var(--input-bg)]",
                  )}
                >
                  <p
                    className={cn(
                      "font-medium",
                      dataPreview.imbalance_warning
                        ? "text-amber-100"
                        : "text-[var(--foreground)]",
                    )}
                  >
                    {dataPreview.imbalance_warning
                      ? "Imbalanced classes detected"
                      : "Class balance (optional resampling)"}
                  </p>
                  <p
                    className={cn(
                      "mt-1 text-[11px] leading-relaxed",
                      dataPreview.imbalance_warning
                        ? "text-amber-100/80"
                        : "text-[var(--muted)]",
                    )}
                  >
                    Shows how often each target class appears in the uploaded CSV.
                    SMOTE synthesizes minority examples on the <strong>training</strong>{" "}
                    split only to reduce bias toward frequent classes; it does not
                    change your uploaded file.
                  </p>
                  {dataPreview.minority_class_pct != null && (
                    <p
                      className={cn(
                        "mt-1 text-xs",
                        dataPreview.imbalance_warning
                          ? "text-amber-100/85"
                          : "text-[var(--muted)]",
                      )}
                    >
                      Smallest class ~{dataPreview.minority_class_pct}% of rows
                      {dataPreview.imbalance_ratio_max_to_min != null &&
                        ` · max:min ratio ${dataPreview.imbalance_ratio_max_to_min}`}
                      .
                    </p>
                  )}
                  <fieldset className="mt-3 space-y-2 text-xs">
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="imb"
                        checked={imbalanceMode === "none"}
                        onChange={() => setImbalanceMode("none")}
                      />
                      Keep as-is (no resampling)
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="imb"
                        checked={imbalanceMode === "smote"}
                        onChange={() => setImbalanceMode("smote")}
                      />
                      Apply SMOTENC / SMOTE on the training split (before
                      encoding; categorical columns supported)
                    </label>
                  </fieldset>
                  <p className="mt-2 text-[11px] text-[var(--muted)]">
                    SMOTE runs only on the training split, before encoding. Needs
                    at least 2 rows per class there.
                  </p>
                </div>
              )}

            {numericColumnOptions.length > 0 &&
              dataPreview?.numeric_outlier_summary &&
              dataPreview.numeric_outlier_summary.some(
                (o) =>
                  o.iqr_outlier_pct != null &&
                  o.iqr_outlier_pct > 5 &&
                  numericColumnOptions.includes(o.column),
              ) && (
                <div className="rounded-2xl border-2 border-teal-500/35 bg-teal-500/10 p-5 text-sm text-teal-50/95">
                  <p className="text-base font-bold text-teal-100">
                    Possible numeric outliers (IQR rule on full file)
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-teal-100/80">
                    For each numeric feature, counts how many values fall outside
                    1.5× the inter-quartile range (classic box-plot fences). This
                    is the same idea as optional clipping in preprocessing; high
                    percentages flag heavy tails or data entry issues.
                  </p>
                  <p className="mt-2 text-sm text-teal-200/70">
                    Informational unless you enable clipping in the dropdown below.
                  </p>
                </div>
              )}

            <section className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium">
                  Numeric scaling
                </label>
                <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)]">
                  Rescales numeric inputs so different units are comparable.
                  Tree models are less sensitive; distance-based models care more.
                  Auto follows the algorithm default from the server.
                </p>
                <select
                  value={scalingMode}
                  onChange={(e) =>
                    setScalingMode(e.target.value as ScalingOption)
                  }
                  className="w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2 text-sm"
                >
                  <option value="auto">
                    Auto (follow algorithm default)
                  </option>
                  <option value="none">None</option>
                  <option value="standard">StandardScaler (zero mean)</option>
                  <option value="minmax">MinMaxScaler</option>
                  <option value="robust">RobustScaler (IQR-based)</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">
                  Numeric outliers in pipeline
                </label>
                <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)]">
                  Optional clipping of extreme numeric values using the same IQR
                  rule as in the preview. Applied per column during training (and
                  baked into the saved pipeline), not just for display.
                </p>
                <select
                  value={numericOutliers}
                  onChange={(e) =>
                    setNumericOutliers(e.target.value as "none" | "clip_iqr")
                  }
                  className="w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2 text-sm"
                >
                  <option value="none">No change</option>
                  <option value="clip_iqr">
                    Clip to IQR fences (1.5×IQR per column, before imputation)
                  </option>
                </select>
              </div>
            </section>

            <section className="space-y-3">
              <p className="text-[11px] leading-relaxed text-[var(--muted)]">
                Turn a numeric column into ordered categories (bins). Useful for
                heavy-tailed fields or when you want trees to split on ranges
                without linear scaling. Binned columns skip the usual numeric
                scaler for those inputs.
              </p>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold">
                  Bin numeric → ordinal buckets
                </h3>
                <button
                  type="button"
                  onClick={() =>
                    setBinRows((rows) => [
                      ...rows,
                      {
                        id: `${Date.now()}-${rows.length}`,
                        column: numericColumnOptions[0] ?? "",
                        n_bins: 5,
                        strategy: "quantile" as const,
                      },
                    ])
                  }
                  disabled={numericColumnOptions.length === 0}
                  className="rounded-lg bg-[var(--accent-dim)] px-3 py-1.5 text-xs font-medium text-[var(--accent)] disabled:opacity-40"
                >
                  Add binning rule
                </button>
              </div>
              {binRows.length === 0 ? (
                <p className="text-xs text-[var(--muted)]">
                  Optional. Binned columns skip linear scaling (trees still work
                  well).
                </p>
              ) : (
                <ul className="space-y-3">
                  {binRows.map((row) => (
                    <li
                      key={row.id}
                      className="flex flex-wrap items-end gap-2 rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-3"
                    >
                      <div className="min-w-[140px] flex-1">
                        <label className="text-xs text-[var(--muted)]">
                          Column
                        </label>
                        <select
                          value={row.column}
                          onChange={(e) =>
                            setBinRows((rs) =>
                              rs.map((r) =>
                                r.id === row.id
                                  ? { ...r, column: e.target.value }
                                  : r,
                              ),
                            )
                          }
                          className="mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--card)] px-2 py-1.5 font-mono text-xs"
                        >
                          <option value="">Select…</option>
                          {numericColumnOptions.map((c) => (
                            <option key={c} value={c}>
                              {c}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="w-20">
                        <label className="text-xs text-[var(--muted)]">
                          Bins
                        </label>
                        <input
                          type="number"
                          min={2}
                          max={50}
                          value={row.n_bins}
                          onChange={(e) =>
                            setBinRows((rs) =>
                              rs.map((r) =>
                                r.id === row.id
                                  ? {
                                      ...r,
                                      n_bins: Number(e.target.value) || 2,
                                    }
                                  : r,
                              ),
                            )
                          }
                          className="mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--card)] px-2 py-1.5 text-xs"
                        />
                      </div>
                      <div className="min-w-[120px]">
                        <label className="text-xs text-[var(--muted)]">
                          Strategy
                        </label>
                        <select
                          value={row.strategy}
                          onChange={(e) =>
                            setBinRows((rs) =>
                              rs.map((r) =>
                                r.id === row.id
                                  ? {
                                      ...r,
                                      strategy: e.target.value as
                                        | "quantile"
                                        | "uniform",
                                    }
                                  : r,
                              ),
                            )
                          }
                          className="mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--card)] px-2 py-1.5 text-xs"
                        >
                          <option value="quantile">Quantile</option>
                          <option value="uniform">Uniform width</option>
                        </select>
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          setBinRows((rs) => rs.filter((r) => r.id !== row.id))
                        }
                        className="rounded-lg border border-rose-500/40 px-2 py-1 text-xs text-rose-200"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        )}

        {wizardStep === 4 && learningMode && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              Choose algorithm(s)
            </h2>
            <p className="text-sm text-[var(--muted)]">
              {learningMode === "supervised" ? (
                <>
                  Pick <strong>one or more</strong> estimators. Each selection
                  starts a separate training job with the same data and
                  preprocessing so you can <strong>compare metrics</strong> on
                  the results screen and run <strong>predictions</strong> against
                  any model that finished training. Only algorithms that match
                  your inferred task (classification vs regression) are shown.
                </>
              ) : (
                <>
                  Pick <strong>one or more</strong> methods from the{" "}
                  <strong>{unsupervisedFamily}</strong> family you chose earlier.
                  Each runs as its own job on the same feature matrix so you can
                  compare metrics (e.g. silhouette, reconstruction error) and use{" "}
                  <strong>predict</strong> / <strong>transform</strong> where the
                  model supports it.
                </>
              )}
            </p>
            {learningMode === "supervised" && supervisedGrouped && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={selectAllSupervisedForTask}
                    disabled={!algorithms || allowedSupervisedIds.length === 0}
                    className="rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-semibold disabled:opacity-40"
                  >
                    Select all ({allowedSupervisedIds.length})
                  </button>
                  <button
                    type="button"
                    onClick={clearSupervisedToSingle}
                    disabled={selectedSupervisedAlgorithms.length <= 1}
                    className="rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-semibold disabled:opacity-40"
                  >
                    Keep one only
                  </button>
                  <span className="text-xs text-[var(--muted)]">
                    {selectedSupervisedAlgorithms.length} model
                    {selectedSupervisedAlgorithms.length !== 1 ? "s" : ""}{" "}
                    selected · max 12
                  </span>
                </div>
                <div className="max-h-80 space-y-5 overflow-y-auto rounded-2xl border border-[var(--border)] bg-[var(--input-bg)]/80 p-4">
                  {(
                    [
                      ["Classification", supervisedGrouped.classification],
                      ["Regression", supervisedGrouped.regression],
                      [
                        "Works for class or regression",
                        supervisedGrouped.both,
                      ],
                    ] as const
                  ).map(([title, items]) => {
                    const opts = items.filter((a) =>
                      allowedSupervisedIds.includes(a.id),
                    );
                    if (opts.length === 0) return null;
                    return (
                      <div key={title}>
                        <p className="mb-2 text-[11px] font-bold uppercase tracking-wide text-[var(--muted)]">
                          {title}
                        </p>
                        <ul className="space-y-2">
                          {opts.map((a) => (
                            <li key={a.id}>
                              <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2.5 transition hover:border-[var(--accent)]/30">
                                <input
                                  type="checkbox"
                                  checked={selectedSupervisedAlgorithms.includes(
                                    a.id,
                                  )}
                                  onChange={() => toggleSupervisedAlgorithm(a.id)}
                                  className="mt-1 accent-[var(--accent)]"
                                />
                                <span className="min-w-0">
                                  <span className="font-medium text-[var(--foreground)]">
                                    {a.label}
                                  </span>
                                  {a.scale_numeric ? (
                                    <span className="ml-2 text-[10px] text-[var(--muted)]">
                                      scales numerics by default
                                    </span>
                                  ) : null}
                                </span>
                              </label>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {learningMode === "unsupervised" && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={selectAllUnsupervisedInFamily}
                    disabled={!algorithms || unsupervisedList.length === 0}
                    className="rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-semibold disabled:opacity-40"
                  >
                    Select all ({unsupervisedList.length})
                  </button>
                  <button
                    type="button"
                    onClick={clearUnsupervisedToSingle}
                    disabled={selectedUnsupervisedAlgorithms.length <= 1}
                    className="rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-semibold disabled:opacity-40"
                  >
                    Keep one only
                  </button>
                  <span className="text-xs text-[var(--muted)]">
                    {selectedUnsupervisedAlgorithms.length} model
                    {selectedUnsupervisedAlgorithms.length !== 1 ? "s" : ""}{" "}
                    selected · max 12
                  </span>
                </div>
                <div className="max-h-80 space-y-2 overflow-y-auto rounded-2xl border border-[var(--border)] bg-[var(--input-bg)]/80 p-4">
                  <ul className="space-y-2">
                    {unsupervisedList.map((a) => (
                      <li key={a.id}>
                        <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-[var(--border)] bg-[var(--card)] px-3 py-2.5 transition hover:border-[var(--accent)]/30">
                          <input
                            type="checkbox"
                            checked={selectedUnsupervisedAlgorithms.includes(
                              a.id,
                            )}
                            onChange={() => toggleUnsupervisedAlgorithm(a.id)}
                            className="mt-1 accent-[var(--accent)]"
                          />
                          <span className="min-w-0">
                            <span className="font-medium text-[var(--foreground)]">
                              {a.label}
                            </span>
                            <span className="ml-2 font-mono text-[10px] text-[var(--muted)]">
                              · {a.inference}
                            </span>
                            {a.scale_numeric ? (
                              <span className="ml-2 text-[10px] text-[var(--muted)]">
                                scales numerics by default
                              </span>
                            ) : null}
                          </span>
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="text-xs text-[var(--muted)]">
                  <strong>predict</strong> = assign cluster / outlier flag ·{" "}
                  <strong>transform</strong> = vector output ·{" "}
                  <strong>none</strong> = no API for new rows
                </p>
              </div>
            )}
            {learningMode === "supervised" && (
              <div>
                <label className="mb-1 flex justify-between text-sm text-[var(--muted)]">
                  <span>Fraction of data for testing</span>
                  <span className="font-mono text-[var(--foreground)]">
                    {(testSize * 100).toFixed(0)}%
                  </span>
                </label>
                <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)]">
                  That share of rows is held out after a random split. Metrics on
                  the results screen are computed on that holdout set (and SMOTE,
                  if enabled, runs only on the training portion).
                </p>
                <input
                  type="range"
                  min={5}
                  max={40}
                  step={1}
                  value={Math.round(testSize * 100)}
                  onChange={(e) => setTestSize(Number(e.target.value) / 100)}
                  className="w-full accent-[var(--accent)]"
                />
              </div>
            )}
          </div>
        )}

        {wizardStep === 5 && learningMode && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">
              Ready to train
            </h2>
            <p className="text-sm text-[var(--muted)]">
              Summary of what will be sent to the server. Each selected model
              becomes its own job (same dataset and preprocessing). Hover a label
              for a short explanation.
            </p>
            <dl className="space-y-2 rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-4 text-sm">
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="Supervised = predict a target column; unsupervised = patterns without a target."
                >
                  Mode
                </dt>
                <dd>{modeLabel}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="Server-side id of the uploaded CSV used for this train run."
                >
                  Dataset
                </dt>
                <dd className="truncate font-mono text-xs">{datasetId}</dd>
              </div>
              {learningMode === "supervised" && (
                <div className="flex justify-between gap-4">
                  <dt
                    className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                    title="Column the model learns to output; not used as an input feature."
                  >
                    Target
                  </dt>
                  <dd>{targetColumn}</dd>
                </div>
              )}
              {learningMode === "unsupervised" && (
                <>
                  <div className="flex justify-between gap-4">
                    <dt
                      className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                      title="Which unsupervised task: clusters, components, or anomaly score."
                    >
                      Family
                    </dt>
                    <dd>{unsupervisedFamily}</dd>
                  </div>
                  {parseExcludeColumns().length > 0 && (
                    <div className="flex justify-between gap-4">
                      <dt
                        className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                        title="Columns removed from the feature matrix before fitting."
                      >
                        Excluded
                      </dt>
                      <dd className="text-right text-xs">
                        {parseExcludeColumns().join(", ")}
                      </dd>
                    </div>
                  )}
                </>
              )}
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="Concrete sklearn-style estimator trained on your data."
                >
                  {(learningMode === "supervised" &&
                    selectedSupervisedAlgorithms.length > 1) ||
                  (learningMode === "unsupervised" &&
                    selectedUnsupervisedAlgorithms.length > 1)
                    ? "Algorithms"
                    : "Algorithm"}
                </dt>
                <dd className="max-w-[min(100%,20rem)] text-right text-xs leading-relaxed">
                  {learningMode === "supervised" ? (
                    <>
                      {selectedSupervisedAlgorithms
                        .map(
                          (id) =>
                            algorithms?.supervised.find((a) => a.id === id)
                              ?.label ?? id,
                        )
                        .join(", ")}
                      {selectedSupervisedAlgorithms.length > 1 && (
                        <span className="mt-1 block text-[10px] text-[var(--muted)]">
                          {selectedSupervisedAlgorithms.length} separate training
                          jobs
                        </span>
                      )}
                    </>
                  ) : (
                    <>
                      {selectedUnsupervisedAlgorithms
                        .map(
                          (id) =>
                            algorithms?.unsupervised[unsupervisedFamily]?.find(
                              (a) => a.id === id,
                            )?.label ?? id,
                        )
                        .join(", ")}
                      {selectedUnsupervisedAlgorithms.length > 1 && (
                        <span className="mt-1 block text-[10px] text-[var(--muted)]">
                          {selectedUnsupervisedAlgorithms.length} separate training
                          jobs
                        </span>
                      )}
                    </>
                  )}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="Input columns checked in the previous step; unchecked columns are not trained or expected at predict."
                >
                  Features
                </dt>
                <dd className="text-right text-xs">
                  {selectedFeatures.length} column
                  {selectedFeatures.length !== 1 ? "s" : ""}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="How numeric columns are rescaled inside the pipeline before the model."
                >
                  Scaling
                </dt>
                <dd className="text-right text-xs">{scalingMode}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt
                  className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                  title="Whether extreme numeric values are clipped using IQR fences in the saved pipeline."
                >
                  Outliers
                </dt>
                <dd className="text-right text-xs">{numericOutliers}</dd>
              </div>
              {learningMode === "supervised" &&
                dataPreview?.inferred_task === "classification" && (
                  <div className="flex justify-between gap-4">
                    <dt
                      className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                      title="Whether to oversample minority classes on the training split only (SMOTE / SMOTENC)."
                    >
                      Imbalance
                    </dt>
                    <dd className="text-right text-xs">{imbalanceMode}</dd>
                  </div>
                )}
              {binRows.some((b) => b.column) && (
                <div className="flex justify-between gap-4">
                  <dt
                    className="cursor-help text-[var(--muted)] underline decoration-dotted decoration-[var(--muted)]/50 underline-offset-2"
                    title="Numeric columns converted to discrete bins before modeling."
                  >
                    Binning
                  </dt>
                  <dd className="text-right text-xs">
                    {binRows.filter((b) => b.column).length} rule(s)
                  </dd>
                </div>
              )}
            </dl>
            <button
              type="button"
              disabled={busy === "train" || !algorithms}
              onClick={() => void onTrain()}
              className="w-full rounded-2xl bg-[var(--accent)] py-4 text-base font-bold text-[var(--accent-fg)] shadow-[0_12px_40px_-8px_rgba(255,77,42,0.55)] transition hover:brightness-110 disabled:opacity-50"
            >
              {busy === "train"
                ? pendingTrainCount > 1
                  ? `Starting ${pendingTrainCount} jobs…`
                  : "Starting training…"
                : pendingTrainCount > 1
                  ? `Train ${pendingTrainCount} models`
                  : "Start training"}
            </button>
          </div>
        )}

        {wizardStep === 6 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold tracking-tight md:text-[1.85rem] md:leading-snug">Results</h2>
            <p className="text-sm text-[var(--muted)]">
              After training completes you get metrics per model, optional bundle
              downloads, and (when supported) in-browser prediction. If you
              trained several models at once, use the table to compare metrics,
              then pick which model to inspect, download, or score with below.
            </p>
            {batchJobs.length > 1 && (
              <div className="overflow-x-auto rounded-2xl border border-[var(--border)] bg-[var(--input-bg)]/90 p-4">
                <p className="text-sm font-semibold text-[var(--foreground)]">
                  Compare models
                </p>
                <p className="mt-1 text-xs text-[var(--muted)]">
                  Same dataset and preprocessing; one row per training job.
                </p>
                <table className="mt-4 w-full min-w-[32rem] border-collapse text-left text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border)] text-[var(--muted)]">
                      <th className="py-2 pr-3 font-semibold">Model</th>
                      <th className="py-2 pr-3 font-semibold">Status</th>
                      <th className="py-2 pr-3 font-semibold">Metrics</th>
                      <th className="py-2 font-semibold">Bundle</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchJobs.map((j) => (
                      <tr
                        key={j.job_id}
                        className="border-b border-[var(--border)]/50 last:border-0"
                      >
                        <td className="py-2.5 pr-3 align-top font-medium text-[var(--foreground)]">
                          {j.algorithm_label ?? j.algorithm_id}
                        </td>
                        <td className="py-2.5 pr-3 align-top">
                          <Badge tone={jobTone(j.status)}>{j.status}</Badge>
                        </td>
                        <td className="max-w-[14rem] py-2.5 pr-3 align-top font-mono text-[11px] leading-relaxed text-[var(--muted)]">
                          {j.status === "completed"
                            ? formatJobMetricsLine(j)
                            : "—"}
                        </td>
                        <td className="py-2.5 align-top">
                          {j.status === "completed" ? (
                            <a
                              href={modelBundleDownloadUrl(j.job_id)}
                              className="font-semibold text-emerald-400 underline-offset-2 hover:underline"
                              target="_blank"
                              rel="noreferrer"
                            >
                              Download
                            </a>
                          ) : (
                            <span className="text-[var(--muted)]">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {batchJobs.length > 1 && (
              <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-4">
                <label className="block text-sm font-medium text-[var(--foreground)]">
                  Model for details, download focus &amp; prediction
                  <select
                    value={jobId}
                    onChange={(e) => setJobId(e.target.value)}
                    className="mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2.5 text-sm"
                  >
                    {batchJobs.map((j) => (
                      <option key={j.job_id} value={j.job_id}>
                        {j.algorithm_label ?? j.algorithm_id} · {j.status}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="mt-2 text-[11px] leading-relaxed text-[var(--muted)]">
                  The summary card and &ldquo;Try a prediction&rdquo; block follow
                  this selection. Each job still has its own id and bundle.
                </p>
              </div>
            )}
            {job && (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={jobTone(job.status)}>{job.status}</Badge>
                  {batchJobs.length > 1 && (
                    <span className="text-xs text-[var(--muted)]">
                      Selected: {job.algorithm_label ?? job.algorithm_id}
                    </span>
                  )}
                </div>
                {job.message && (
                  <div className="space-y-2">
                    <p className="text-sm text-rose-300">{job.message}</p>
                    {job.user_hint?.trim() && (
                      <p className="text-xs leading-relaxed text-amber-100/90">
                        <span className="font-semibold text-amber-200/95">
                          Tip:{" "}
                        </span>
                        {job.user_hint}
                      </p>
                    )}
                  </div>
                )}

                {job.status === "completed" && (
                  <div className="rounded-2xl border border-[var(--border)] bg-[var(--input-bg)] p-5">
                    <p className="text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
                      What you trained
                    </p>
                    <p className="mt-1 text-[11px] leading-relaxed text-[var(--muted)]">
                      Short human summary of model type, headline metric, and
                      feature count. Use it to confirm the job matches what you
                      intended before downloading or integrating.
                    </p>
                    {(() => {
                      const sum = buildPostTrainSummary(job, {
                        learningMode,
                        targetColumn,
                        fallbackFeatureColumns,
                      });
                      const plainBlock = [
                        sum.modelLine,
                        sum.metricLine,
                        sum.featuresLine,
                      ]
                        .filter(Boolean)
                        .join(" · ");
                      return (
                        <div className="mt-3 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                          <p className="min-w-0 flex-1 text-sm leading-relaxed text-[var(--foreground)]">
                            {plainBlock}
                          </p>
                          <div className="flex shrink-0 flex-wrap gap-2 sm:justify-end">
                            {jobId && (
                              <a
                                href={modelBundleDownloadUrl(jobId)}
                                className="inline-flex rounded-xl bg-emerald-500/90 px-4 py-2.5 text-sm font-semibold text-emerald-950 hover:bg-emerald-400"
                                target="_blank"
                                rel="noreferrer"
                              >
                                Download
                              </a>
                            )}
                            {job.inference_method !== "none" && (
                              <button
                                type="button"
                                onClick={() =>
                                  document
                                    .getElementById("wizard-try-predict")
                                    ?.scrollIntoView({
                                      behavior: "smooth",
                                      block: "start",
                                    })
                                }
                                className="inline-flex rounded-xl border border-[var(--accent)]/50 bg-[var(--accent-dim)] px-4 py-2.5 text-sm font-semibold text-[var(--accent)]"
                              >
                                Try prediction
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })()}
                    <p className="mt-4 text-xs text-[var(--muted)]">
                      <strong>Download</strong> adds{" "}
                      <code className="rounded bg-black/30 px-1">predict_local.py</code>{" "}
                      and{" "}
                      <code className="rounded bg-black/30 px-1">
                        requirements-predict.txt
                      </code>{" "}
                      next to the model—unpack, run{" "}
                      <code className="rounded bg-black/30 px-1">
                        pip install -r requirements-predict.txt
                      </code>
                      , then{" "}
                      <code className="rounded bg-black/30 px-1">
                        python predict_local.py
                      </code>{" "}
                      (see script header for JSON/CSV usage).{" "}
                      <strong>Try prediction</strong> tests in the browser (features
                      only—no target).
                    </p>
                  </div>
                )}

                {job.status === "completed" && job.preprocess_applied && (
                  <details className="rounded-xl border border-[var(--border)] bg-black/20">
                    <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-[var(--muted)]">
                      Applied preprocessing (saved in model bundle)
                    </summary>
                    <p className="border-t border-[var(--border)] px-4 pb-0 pt-3 text-[11px] leading-relaxed text-[var(--muted)]">
                      Exact parameters the server embedded in the job: imputers,
                      encoders, scalers, and any SMOTE or binning flags. Matches
                      what the downloaded pipeline will apply at predict time.
                    </p>
                    <pre className="max-h-40 overflow-auto border-t border-[var(--border)] p-4 font-mono text-xs">
                      {JSON.stringify(job.preprocess_applied, null, 2)}
                    </pre>
                  </details>
                )}

                {job.metrics && (
                  <details className="rounded-xl border border-[var(--border)] bg-black/20">
                    <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-[var(--muted)]">
                      Technical details (raw metrics)
                    </summary>
                    <p className="border-t border-[var(--border)] px-4 pb-0 pt-3 text-[11px] leading-relaxed text-[var(--muted)]">
                      Unprocessed metric dict from the trainer (accuracy, R²,
                      silhouette, etc., depending on task). For debugging or
                      comparing runs; the summary above is easier to read first.
                    </p>
                    <pre className="max-h-56 overflow-auto border-t border-[var(--border)] p-4 font-mono text-xs">
                      {JSON.stringify(job.metrics, null, 2)}
                    </pre>
                  </details>
                )}

                {job.inference_method === "none" &&
                  job.status === "completed" && (
                    <p className="text-xs text-amber-200/90">
                      This model cannot score new rows via the API. Download the
                      bundle or pick an algorithm with predict/transform.
                    </p>
                  )}
              </>
            )}
            {!job && jobId && (
              <p className="text-sm text-[var(--muted)]">Loading job status…</p>
            )}

            {job?.status === "completed" &&
              job.inference_method !== "none" && (
                <div
                  id="wizard-try-predict"
                  className="scroll-mt-6 space-y-3 border-t border-[var(--border)] pt-6"
                >
                  <h3 className="text-sm font-semibold">Try a prediction</h3>
                  <p className="text-xs text-[var(--muted)]">
                    Calls the deployed inference endpoint with the same feature
                    columns the model was trained on. Send a JSON array of row
                    objects; property names must match training exactly
                    (spelling and case). Never include the target column. The
                    response is the model output for each row (class, score,
                    cluster id, etc., depending on the algorithm).
                  </p>

                  {predictionFeatures.length > 0 && (
                    <div className="rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-4">
                      <p className="text-xs font-medium text-[var(--foreground)]">
                        Required feature columns
                      </p>
                      <ul className="mt-2 list-inside list-disc text-xs leading-relaxed text-[var(--muted)]">
                        {predictionFeatures.map((name) => (
                          <li key={name} className="font-mono text-[var(--foreground)]">
                            {name}
                          </li>
                        ))}
                      </ul>
                      {predictExampleJson && (
                        <>
                          <p className="mt-4 text-xs font-medium text-[var(--foreground)]">
                            Example JSON (from training metadata
                            {job && predictionExampleFromJob(job)
                              ? ""
                              : " + dataset preview"}
                            )
                          </p>
                          <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-black/35 p-3 font-mono text-[11px] leading-relaxed text-emerald-100/95">
                            {predictExampleJson}
                          </pre>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={() => void copyPredictExampleJson()}
                              className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-medium"
                            >
                              {predictHelperCopied ? "Copied" : "Copy example"}
                            </button>
                            <button
                              type="button"
                              onClick={() => setPredictJson(predictExampleJson)}
                              className="rounded-lg bg-[var(--accent)] px-3 py-2 text-xs font-semibold text-[var(--accent-fg)]"
                            >
                              Use as input
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  )}

                  <textarea
                    value={predictJson}
                    onChange={(e) => setPredictJson(e.target.value)}
                    rows={6}
                    spellCheck={false}
                    className="w-full rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-3 font-mono text-xs"
                  />
                  <button
                    type="button"
                    disabled={busy === "predict"}
                    onClick={() => void onPredict()}
                    className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--accent-fg)] disabled:opacity-50"
                  >
                    {busy === "predict" ? "Running…" : "Run predict / transform"}
                  </button>
                  {lastInferenceMethod && (
                    <p className="text-xs text-[var(--muted)]">
                      Method: {lastInferenceMethod}
                    </p>
                  )}
                  {predictOut && (
                    <pre className="max-h-40 overflow-auto rounded-xl bg-black/35 p-3 font-mono text-xs">
                      {predictOut}
                    </pre>
                  )}
                </div>
              )}

            <button
              type="button"
              onClick={resetWizard}
              className="w-full rounded-xl border border-[var(--border)] py-3 text-sm font-medium"
            >
              Start a new run
            </button>
          </div>
        )}
      </div>

      {/* Nav footer */}
      {wizardStep > 0 && wizardStep < 6 && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={goBack}
            className="rounded-xl border border-[var(--border)] px-5 py-2.5 text-sm font-medium"
          >
            Back
          </button>
          {wizardStep < 5 && (
            <button
              type="button"
              onClick={() => void goNext()}
              disabled={
                !canProceedFrom(wizardStep) ||
                busy === "dataset" ||
                busy === "upload"
              }
              className="rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-fg)] disabled:opacity-40"
            >
              {busy === "dataset" ? "Loading…" : "Continue"}
            </button>
          )}
        </div>
      )}

      {showLibrary && (
        <section className="animate-page-enter-delay-3 grid gap-8 rounded-[2rem] border-2 border-[var(--border)] bg-[var(--card)] p-8 sm:grid-cols-2">
          <div>
            <h3 className="text-base font-bold text-[var(--foreground)]">
              Datasets ({datasets.length})
            </h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Click a row to load that dataset id and jump to target selection.
            </p>
            <ul className="mt-3 max-h-52 space-y-1.5 overflow-auto text-sm">
              {datasets.map((d) => (
                <li
                  key={d.dataset_id}
                  className="cursor-pointer rounded-xl px-3 py-2 hover:bg-[var(--input-bg)]"
                  onClick={() => {
                    setDatasetId(d.dataset_id);
                    setColumns(d.columns);
                    setTargetColumn(d.columns[0] ?? "");
                    if (wizardStep < 2) setWizardStep(2);
                  }}
                >
                  <span className="font-mono text-sm">{d.dataset_id}</span>
                  <span className="ml-2 text-sm text-[var(--muted)]">
                    {d.approx_rows} rows
                  </span>
                </li>
              ))}
              {datasets.length === 0 && (
                <li className="text-[var(--muted)]">None yet.</li>
              )}
            </ul>
          </div>
          <div>
            <h3 className="text-base font-bold text-[var(--foreground)]">
              Jobs ({jobs.length})
            </h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Open a job to view results and download the trained bundle.
            </p>
            <ul className="mt-3 max-h-52 space-y-1.5 overflow-auto text-sm">
              {jobs.map((j) => (
                <li key={j.job_id}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-2 rounded-xl px-3 py-2 text-left hover:bg-[var(--input-bg)]"
                    onClick={() => {
                      setTrainJobIds([j.job_id]);
                      setJobId(j.job_id);
                      setWizardStep(6);
                    }}
                  >
                    <span className="truncate font-mono text-sm">{j.job_id}</span>
                    <Badge tone={jobTone(j.status)}>{j.status}</Badge>
                  </button>
                </li>
              ))}
              {jobs.length === 0 && (
                <li className="text-[var(--muted)]">None yet.</li>
              )}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
