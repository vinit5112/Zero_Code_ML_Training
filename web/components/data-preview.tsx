"use client";

import type {
  ColumnSummary,
  DataQualityPayload,
  DatasetColumnProfile,
  DatasetPreview,
  NumericCorrelationBlock,
} from "@/lib/types";
import dynamic from "next/dynamic";
import type { CSSProperties, ReactNode } from "react";

const PreviewDistributionCharts = dynamic(
  () =>
    import("@/components/preview-distribution-charts").then(
      (m) => m.PreviewDistributionCharts,
    ),
  {
    ssr: false,
    loading: () => (
      <p className="px-1 py-2 text-xs text-[var(--muted)]">Loading charts…</p>
    ),
  },
);

function cn(...parts: (string | false | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

function MissingBar({ pct }: { pct: number }) {
  if (pct <= 0) return <span className="text-[var(--muted)]">0%</span>;
  return (
    <span
      className={cn(
        "font-mono text-xs",
        pct >= 50 && "text-rose-300",
        pct >= 15 && pct < 50 && "text-amber-200",
        pct < 15 && "text-[var(--muted)]",
      )}
    >
      {pct.toFixed(1)}%
    </span>
  );
}

function MutedHelp({ children }: { children: ReactNode }) {
  return (
    <p className="text-sm leading-relaxed text-[var(--muted)]">{children}</p>
  );
}

function ThHint({ label, hint }: { label: string; hint: string }) {
  return (
    <th
      title={hint}
      className="cursor-help px-3 py-2.5 text-sm font-bold underline decoration-dotted decoration-[var(--muted)]/45 underline-offset-2"
    >
      {label}
    </th>
  );
}

export function DataPreviewPanel({
  preview,
  loading,
  error,
  title = "Data preview",
  footer,
}: {
  preview: DatasetPreview | null;
  loading: boolean;
  error: string | null;
  title?: string;
  footer?: ReactNode;
}) {
  if (loading) {
    return (
      <div className="rounded-2xl border-2 border-[var(--border)] bg-[var(--input-bg)] px-6 py-10 text-center text-base text-[var(--muted)]">
        Loading preview…
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-2xl border-2 border-rose-500/40 bg-rose-500/10 px-5 py-4 text-base text-rose-100">
        Preview failed: {error}
      </div>
    );
  }
  if (!preview) return null;

  const sampleKeys =
    preview.sample_rows[0] != null
      ? Object.keys(preview.sample_rows[0] as object)
      : preview.columns.map((c) => c.name);

  return (
    <div className="space-y-6 rounded-[1.75rem] border-2 border-[var(--border)] bg-[var(--input-bg)] p-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h3 className="text-lg font-bold tracking-tight text-[var(--foreground)] md:text-xl">
          {title}
        </h3>
        <p className="font-mono text-sm text-[var(--highlight-muted)]">
          {preview.n_rows.toLocaleString()} rows × {preview.n_columns} columns · showing
          first {preview.sample_row_count} rows
        </p>
      </div>
      <MutedHelp>
        This panel summarizes your CSV before training: a row sample, column
        types and missingness, exploratory stats, relationships between columns,
        chartable distributions, and quick data-quality flags. Open each section
        below for detail; hover dotted table headers where you see them for
        definitions.
      </MutedHelp>

      {preview.inferred_task && preview.target_column && (
        <div className="rounded-2xl border-2 border-teal-500/35 bg-teal-500/10 px-4 py-3 md:px-5 md:py-4">
          <p className="text-sm font-bold text-teal-200">
            Inferred task for{" "}
            <span className="font-mono text-[var(--foreground)]">
              {preview.target_column}
            </span>
            :{" "}
            <span className="uppercase tracking-wide">
              {preview.inferred_task}
            </span>
          </p>
          {preview.target_task_explanation && (
            <p className="mt-2 text-sm text-[var(--muted)]">
              {preview.target_task_explanation}
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-4 text-sm text-[var(--muted)]">
            <span>
              Unique values:{" "}
              <span className="text-[var(--foreground)]">
                {preview.target_n_unique?.toLocaleString()}
              </span>
            </span>
            <span>
              Missing:{" "}
              <span
                className={cn(
                  (preview.target_missing_count ?? 0) > 0 && "text-amber-200",
                )}
              >
                {preview.target_missing_count ?? 0} (
                {preview.target_missing_pct?.toFixed(1)}%)
              </span>
            </span>
          </div>
          <MutedHelp>
            <strong>Unique values</strong> counts distinct non-null entries in the
            target (helps spot how many classes or how continuous the target looks).{" "}
            <strong>Missing</strong> is how many rows have no target; those rows
            are usually unusable for supervised training.
          </MutedHelp>
        </div>
      )}

      <div>
        <p className="mb-1.5 text-sm font-bold text-[var(--muted)]">
          Sample rows (first {preview.sample_row_count})
        </p>
        <MutedHelp>
          Raw cell values from the beginning of the file. Use this to sanity-check
          delimiters, column names, and whether values look like numbers or text.
        </MutedHelp>
      </div>
      <div className="overflow-x-auto rounded-2xl border-2 border-[var(--border)]">
        <table className="w-full min-w-[520px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b-2 border-[var(--border)] bg-[var(--card-elevated)]">
              {sampleKeys.map((k) => (
                <th
                  key={k}
                  className={cn(
                    "whitespace-nowrap px-3 py-3 text-sm font-bold",
                    k === preview.target_column && "text-teal-300",
                  )}
                >
                  {k}
                  {k === preview.target_column && (
                    <span className="ml-1.5 text-xs font-semibold uppercase tracking-wider text-teal-400/90">
                      target
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.sample_rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-[var(--border)]/60 hover:bg-white/[0.03]"
              >
                {sampleKeys.map((k) => (
                  <td
                    key={k}
                    className="max-w-[160px] truncate px-3 py-2.5 font-mono text-sm"
                  >
                    {formatCell((row as Record<string, unknown>)[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {preview.columns.some((c) => c.missing_pct >= 25) && (
        <p className="text-xs text-amber-200/90">
          One or more columns have many missing values. Fix the CSV, impute, or
          exclude ID-like columns before training.
        </p>
      )}

      <div>
        <p className="mb-1 text-sm font-bold text-[var(--muted)]">
          Column overview (dtype · missing · distinct values)
        </p>
        <MutedHelp>
          <span className="font-mono">dtype</span> is how pandas read the column.{" "}
          <strong>Missing</strong> is null/empty count and percent of all rows.{" "}
          <strong>Unique</strong> is how many different non-null values appear
          (high counts often mean ids or free text).
        </MutedHelp>
        <ul className="mt-2 max-h-56 space-y-2 overflow-y-auto text-sm">
          {preview.columns.map((c: DatasetColumnProfile) => (
            <li
              key={c.name}
              className={cn(
                "flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-transparent px-3 py-2 hover:border-[var(--border)]",
                c.name === preview.target_column && "bg-teal-500/10 ring-1 ring-teal-500/20",
              )}
            >
              <span className="font-mono text-[var(--foreground)]">{c.name}</span>
              <span className="text-[var(--muted)]">{c.dtype}</span>
              <span className="text-[var(--muted)]">
                missing: {c.missing_count}{" "}
                <MissingBar pct={c.missing_pct} />
              </span>
              <span className="text-[var(--muted)]">
                unique: {c.n_unique.toLocaleString()}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {preview.data_quality && (
        <details className="rounded-2xl border-2 border-[var(--border)] bg-black/25">
          <summary className="cursor-pointer list-none px-4 py-3.5 text-sm font-bold text-[var(--foreground)] marker:content-none [&::-webkit-details-marker]:hidden">
            Data quality checks
          </summary>
          <div className="space-y-3 border-t border-[var(--border)] px-3 py-3 text-xs">
            <MutedHelp>
              Automated sanity checks for common issues (duplicates, useless
              constants, very high-cardinality categories, near-unique id columns,
              and a strict &quot;maybe leakage&quot; rule for classification).
              They are rules of thumb, not proof: always interpret with your
              problem context.
            </MutedHelp>
            <DataQualityBlock dq={preview.data_quality} />
          </div>
        </details>
      )}

      {preview.columns.some((c) => c.summary) && (
        <details className="rounded-2xl border-2 border-[var(--border)] bg-black/25">
          <summary className="cursor-pointer list-none px-4 py-3.5 text-sm font-bold text-[var(--foreground)] marker:content-none [&::-webkit-details-marker]:hidden">
            Column summaries (EDA)
          </summary>
          <div className="space-y-2 border-t border-[var(--border)] px-3 py-3">
            <MutedHelp>
              Per-column exploratory stats. Numeric columns show spread and central
              tendency; categorical columns show the five most frequent values.
              These are computed on the full file, not only the sample table above.
            </MutedHelp>
          </div>
          <div className="overflow-x-auto border-t border-[var(--border)]">
            <table className="w-full min-w-[720px] border-collapse text-left text-[11px]">
              <thead>
                <tr className="border-b border-[var(--border)] bg-black/25">
                  <th
                    title="CSV column name; target is marked."
                    className="sticky left-0 z-[1] cursor-help bg-black/40 px-2 py-2 font-medium underline decoration-dotted decoration-white/25 underline-offset-2"
                  >
                    Column
                  </th>
                  <ThHint
                    label="Kind"
                    hint="Whether the column is treated as numeric metrics or categorical counts for the summary row."
                  />
                  <ThHint
                    label="Details"
                    hint="min/max/mean/median/std/zero% for numbers; top 5 value counts for categories."
                  />
                </tr>
              </thead>
              <tbody>
                {preview.columns.map((c) => (
                  <tr
                    key={c.name}
                    className={cn(
                      "border-b border-[var(--border)]/50",
                      c.name === preview.target_column && "bg-teal-500/5",
                    )}
                  >
                    <td className="sticky left-0 z-[1] whitespace-nowrap bg-[var(--input-bg)] px-2 py-2 font-mono text-[var(--foreground)]">
                      {c.name}
                      {c.name === preview.target_column && (
                        <span className="ml-1 text-[10px] text-teal-400">target</span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-2 py-2 text-[var(--muted)]">
                      {c.summary?.kind ?? "—"}
                    </td>
                    <td className="px-2 py-2 text-[var(--muted)]">
                      <SummaryCell summary={c.summary} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      <details className="rounded-lg border border-[var(--border)] bg-black/15">
        <summary className="cursor-pointer px-3 py-2.5 text-xs font-medium text-[var(--foreground)]">
          Pairwise relationships
        </summary>
        <div className="space-y-4 border-t border-[var(--border)] px-3 py-3 text-xs">
          <MutedHelp>
            How columns relate to each other: linear correlation among numeric
            features, and (when a target is set) how strongly each feature aligns
            with that target on its own. Pairwise correlation ignores the target;
            the association table is supervised and includes categorical features
            via mutual information.
          </MutedHelp>
          <div>
            <p className="mb-1 font-medium text-[var(--muted)]">
              Numeric Pearson correlation
            </p>
            <MutedHelp>
              Measures straight-line relationship between two numeric columns (-1 to
              1). Near 0 means little linear link; extreme values mean they rise or
              fall together (positive) or opposite (negative). Only columns stored
              as numbers are included; booleans are skipped.
            </MutedHelp>
            {preview.numeric_correlation == null ? (
              <p className="text-[var(--muted)]">
                Not enough numeric columns for a correlation summary (need at least
                two).
              </p>
            ) : (
              <CorrelationBlock block={preview.numeric_correlation} />
            )}
          </div>

          {preview.target_column &&
            preview.feature_target_association &&
            preview.feature_target_association.length > 0 && (
              <div>
                <p className="mb-1 font-medium text-[var(--muted)]">
                  Univariate association with target{" "}
                  <span className="font-mono text-[var(--foreground)]">
                    {preview.target_column}
                  </span>
                </p>
                <MutedHelp>
                  Each row is one feature examined alone against the target (no
                  interaction effects). Rows sort by mutual information (larger =
                  more informative), then by absolute Pearson r for regression ties.{" "}
                  <strong>MI</strong> works for nonlinear links and categories;{" "}
                  <strong>Pearson r</strong> only makes sense for numeric target and
                  numeric feature and appears for regression.
                </MutedHelp>
                <div className="mt-2 overflow-x-auto rounded-md border border-[var(--border)]">
                  <table className="w-full min-w-[480px] border-collapse text-left text-[11px]">
                    <thead>
                      <tr className="border-b border-[var(--border)] bg-black/25">
                        <ThHint
                          label="Column"
                          hint="Feature name from the CSV (not the target)."
                        />
                        <ThHint
                          label="Pearson r"
                          hint="Linear correlation with the numeric target, when applicable; em dash if not computed (e.g. classification or non-numeric feature)."
                        />
                        <ThHint
                          label="Mutual info"
                          hint="How much knowing this feature reduces uncertainty about the target; higher is stronger. Comparable within the same table, not across datasets."
                        />
                        <ThHint
                          label="Note"
                          hint="Why a score is missing or unreliable (too few values, numeric failure, etc.)."
                        />
                      </tr>
                    </thead>
                    <tbody>
                      {preview.feature_target_association.map((row) => (
                        <tr
                          key={row.column}
                          className={cn(
                            "border-b border-[var(--border)]/50",
                            row.column === preview.target_column &&
                              "bg-teal-500/5",
                          )}
                        >
                          <td className="whitespace-nowrap px-2 py-1.5 font-mono text-[var(--foreground)]">
                            {row.column}
                          </td>
                          <td className="px-2 py-1.5 font-mono text-[var(--muted)]">
                            {row.pearson_r != null ? fmtNum(row.pearson_r) : "—"}
                          </td>
                          <td className="px-2 py-1.5 font-mono text-[var(--muted)]">
                            {row.mutual_info != null ? fmtNum(row.mutual_info) : "—"}
                          </td>
                          <td className="px-2 py-1.5 text-[var(--muted)]">
                            {associationNoteLabel(row.note)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
        </div>
      </details>

      {preview.distribution_charts && (
        <details className="rounded-2xl border-2 border-[var(--border)] bg-black/25">
          <summary className="cursor-pointer list-none px-4 py-3.5 text-sm font-bold text-[var(--foreground)] marker:content-none [&::-webkit-details-marker]:hidden">
            Distribution charts
          </summary>
          <div className="space-y-2 border-t border-[var(--border)] px-3 py-3">
            <MutedHelp>
              Shape of each variable: where values pile up, heavy tails, or dominant
              categories. Charts use <strong>aggregated counts</strong> from the
              server (bins and category totals), so the browser never needs every
              row. A cap may apply to how many columns get a chart; see any warning
              inside the section.
            </MutedHelp>
            <PreviewDistributionCharts
              payload={preview.distribution_charts}
              targetColumn={preview.target_column}
            />
          </div>
        </details>
      )}

      {footer}
    </div>
  );
}

function correlationCellStyle(r: number | null): CSSProperties {
  if (r === null)
    return { backgroundColor: "rgba(255,255,255,0.04)", color: "var(--muted)" };
  const t = Math.max(-1, Math.min(1, r));
  if (t >= 0) {
    const a = 0.12 + Math.abs(t) * 0.5;
    return {
      backgroundColor: `rgba(56, 189, 248, ${a})`,
      color: t > 0.65 ? "#0c1222" : "var(--foreground)",
    };
  }
  const a = 0.12 + Math.abs(t) * 0.5;
  return {
    backgroundColor: `rgba(251, 113, 133, ${a})`,
    color: t < -0.55 ? "#0c1222" : "var(--foreground)",
  };
}

function CorrelationBlock({ block }: { block: NumericCorrelationBlock }) {
  if (block.format === "matrix") {
    const cols = block.columns;
    return (
      <div className="space-y-2">
        <div className="overflow-x-auto rounded-md border border-[var(--border)]">
        <table className="border-collapse text-left text-[10px]">
          <thead>
            <tr>
              <th className="sticky left-0 z-[1] bg-[var(--input-bg)] px-1.5 py-1" />
              {cols.map((c) => (
                <th
                  key={c}
                  className="max-w-[72px] truncate px-1 py-1 font-mono font-medium text-[var(--muted)]"
                  title={c}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cols.map((rowName, i) => (
              <tr key={rowName}>
                <th
                  className="sticky left-0 z-[1] max-w-[100px] truncate bg-[var(--input-bg)] px-1.5 py-1 text-left font-mono font-normal text-[var(--foreground)]"
                  title={rowName}
                >
                  {rowName}
                </th>
                {cols.map((_, j) => {
                  const r = block.matrix[i]?.[j] ?? null;
                  return (
                    <td
                      key={`${i}-${j}`}
                      className="px-1 py-0.5 text-center font-mono tabular-nums"
                      style={correlationCellStyle(r)}
                      title={r != null ? `r = ${r}` : undefined}
                    >
                      {r != null ? r.toFixed(2) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        <p className="text-[10px] leading-relaxed text-[var(--muted)]">
          Diagonal entries are always 1 (a column with itself). Cell color: sky =
          positive correlation, rose = negative; stronger color means |r| closer
          to 1. Hover a cell for the exact value.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-[10px] leading-relaxed text-[var(--muted)]">
        Too many numeric columns for a full matrix; listing the strongest linear
        pairs by absolute correlation instead.
      </p>
      {block.note && (
        <p className="text-[11px] text-[var(--muted)]">{block.note}</p>
      )}
      <ul className="max-h-56 space-y-1 overflow-y-auto font-mono text-[11px] leading-relaxed text-[var(--foreground)]">
        {block.pairs.map((p) => (
          <li key={`${p.column_a}|${p.column_b}`}>
            <span className="text-[var(--muted)]">{p.column_a}</span>
            {" · "}
            <span className="text-[var(--muted)]">{p.column_b}</span>
            <span className="ml-2 text-teal-300/90">r = {p.r.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function dataQualityPatternLabel(pattern: string): string {
  const m: Record<string, string> = {
    uuid_or_guid: "UUID / GUID",
    row_or_record_id: "row / record id",
    suffix_id: "ends with _id / _uuid",
    id_token: "id",
    index_like: "index",
    datetime_like: "date / time",
  };
  return m[pattern] ?? pattern.replace(/_/g, " ");
}

function DataQualityBlock({ dq }: { dq: DataQualityPayload }) {
  const hasLists =
    dq.constant_columns.length > 0 ||
    dq.high_cardinality_categoricals.length > 0 ||
    dq.id_like_columns.length > 0 ||
    dq.leaky_column_hints.length > 0;

  return (
    <>
      <p className="text-[11px] leading-relaxed text-[var(--muted)]">
        {dq.heuristic_warning}
      </p>

      <div>
        <p className="mb-1 font-medium text-[var(--muted)]">Duplicate rows</p>
        <p className="mb-1.5 text-[10px] leading-relaxed text-[var(--muted)]">
          Rows that match an earlier row on <strong>every column</strong> (full-row
          duplicate). The count is how many extra copies exist if you keep the first
          occurrence only.
        </p>
        {dq.duplicate_row_count > 0 ? (
          <p className="text-amber-200/90">
            <span className="font-mono">{dq.duplicate_row_count.toLocaleString()}</span>{" "}
            rows are exact duplicates of an earlier row (
            {dq.duplicate_row_pct.toFixed(1)}% of the table). Consider deduplicating.
          </p>
        ) : (
          <p className="text-[var(--muted)]">No fully duplicate rows detected.</p>
        )}
      </div>

      <div>
        <p className="mb-1 font-medium text-[var(--muted)]">Constant columns</p>
        <p className="mb-1.5 text-[10px] leading-relaxed text-[var(--muted)]">
          Columns where all non-null values are identical. They carry no information
          for learning (only one level); safe to drop unless you rely on them
          downstream.
        </p>
        {dq.constant_columns.length === 0 ? (
          <p className="text-[var(--muted)]">None (among non-null values).</p>
        ) : (
          <ul className="list-inside list-disc space-y-1 font-mono text-[11px] text-[var(--foreground)]">
            {dq.constant_columns.map((r) => (
              <li key={r.column}>
                <span className="text-teal-200/80">{r.column}</span>
                <span className="text-[var(--muted)]"> · sample value </span>
                {r.value_sample}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <p className="mb-1 font-medium text-[var(--muted)]">
          High-cardinality categoricals
        </p>
        <p className="mb-1.5 text-[10px] leading-relaxed text-[var(--muted)]">
          Text or object columns with many distinct values. One-hot encoding can
          explode width; tree models may grow large or overfit. Consider hashing,
          grouping rare levels, or dropping if it is really an identifier.
        </p>
        {dq.high_cardinality_categoricals.length === 0 ? (
          <p className="text-[var(--muted)]">No columns flagged (object/string; thresholds ~50+ levels or &gt;12% of row count).</p>
        ) : (
          <ul className="space-y-1 font-mono text-[11px] text-[var(--foreground)]">
            {dq.high_cardinality_categoricals.map((r) => (
              <li key={r.column}>
                <span className="text-amber-200/90">{r.column}</span>
                <span className="text-[var(--muted)]">
                  {" "}
                  · {r.n_unique.toLocaleString()} distinct ·{" "}
                  {(r.unique_to_rows_ratio * 100).toFixed(1)}% of rows — one-hot or
                  trees may be costly.
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <p className="mb-1 font-medium text-[var(--muted)]">ID-like (almost unique)</p>
        <p className="mb-1.5 text-[10px] leading-relaxed text-[var(--muted)]">
          Feature columns (target excluded) where almost every non-null value is
          different from the others. Usually row keys or names, not causes of the
          outcome; they often hurt generalization if included.
        </p>
        {dq.id_like_columns.length === 0 ? (
          <p className="text-[var(--muted)]">
            No feature columns look near-unique on non-null values (target excluded).
          </p>
        ) : (
          <ul className="space-y-1 font-mono text-[11px] text-[var(--foreground)]">
            {dq.id_like_columns.map((r) => (
              <li key={r.column}>
                <span className="text-teal-200/80">{r.column}</span>
                <span className="text-[var(--muted)]">
                  {" "}
                  · {r.n_unique.toLocaleString()} / {r.non_null_count.toLocaleString()}{" "}
                  distinct (ratio {r.unique_ratio.toFixed(3)}) — consider excluding as a
                  feature.
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <p className="mb-1 font-medium text-[var(--muted)]">
          Possible leakage (name + 1:1 with target)
        </p>
        <p className="mb-1 text-[11px] leading-relaxed text-[var(--muted)]">
          Only when a <strong>classification</strong> target is set: the column
          name looks like id/date/index <em>and</em>, on rows where both feature and
          target are non-null, each distinct feature value appears with exactly one
          class and each class appears with exactly one feature value (perfect
          pairing). That can indicate a row key or accidental duplicate of the
          label, but it can also misfire on small or curated data. Use as a review
          trigger, not automatic proof of leakage.
        </p>
        {dq.leaky_column_hints.length === 0 ? (
          <p className="text-[var(--muted)]">No columns matched this strict heuristic.</p>
        ) : (
          <ul className="space-y-2 text-[11px]">
            {dq.leaky_column_hints.map((r) => (
              <li
                key={r.column}
                className="rounded-md border border-amber-500/25 bg-amber-500/10 px-2 py-1.5"
              >
                <span className="font-mono text-amber-100">{r.column}</span>
                <span className="text-[var(--muted)]">
                  {" "}
                  · name hint: {dataQualityPatternLabel(r.name_pattern)}
                </span>
                <p className="mt-1 text-[var(--muted)]">{r.note}</p>
              </li>
            ))}
          </ul>
        )}
      </div>

      {!hasLists && dq.duplicate_row_count === 0 && (
        <p className="text-[11px] text-emerald-200/70">
          No column-level issues flagged beyond duplicates.
        </p>
      )}
    </>
  );
}

function associationNoteLabel(note: string | undefined): string {
  if (!note) return "—";
  if (note === "too_few_values")
    return "Too few non-null rows to score this feature safely.";
  if (note === "mi_failed")
    return "Mutual information could not be computed for this column.";
  if (note === "failed") return "Association metrics failed for this column.";
  return note;
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e6 || (a > 0 && a < 1e-4)) return n.toExponential(3);
  return Number.isInteger(n) ? String(n) : n.toFixed(4).replace(/\.?0+$/, "");
}

function SummaryCell({ summary }: { summary: ColumnSummary | undefined }) {
  if (!summary) return <span>—</span>;
  if (summary.kind === "numeric") {
    if ("note" in summary && summary.note === "no_numeric_values") {
      return <span className="text-amber-200/80">No numeric values</span>;
    }
    return (
      <span
        title="min/max: range; mean: average; med: median (50th percentile); std: spread around mean; zeros: percent of rows that are exactly zero (among non-null numbers)."
        className="font-mono leading-relaxed text-[var(--foreground)]/90"
      >
        min {fmtNum(summary.min)} · max {fmtNum(summary.max)} · mean{" "}
        {fmtNum(summary.mean)} · med {fmtNum(summary.median)} · std{" "}
        {fmtNum(summary.std)} · zeros {summary.zero_pct?.toFixed(1) ?? "—"}%
      </span>
    );
  }
  if (summary.kind === "categorical" && summary.top_values?.length) {
    const parts = summary.top_values.map(
      (t) => `"${t.value}" (${t.count.toLocaleString()})`,
    );
    return (
      <span
        title="Most frequent values in the column and how many rows each appears in (among non-null); only the top five are listed."
        className="font-mono leading-relaxed text-[var(--foreground)]/90"
      >
        {parts.join(" · ")}
        <span className="text-[var(--muted)]"> · top 5</span>
      </span>
    );
  }
  return <span>—</span>;
}
