"use client";

import type {
  ColumnDistributionChart,
  DistributionChartsPayload,
  DistributionCategoricalChart,
  DistributionNumericChart,
} from "@/lib/types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = "#94a3b8";
const GRID = "rgba(148, 163, 184, 0.15)";
const BAR = "#3ae8c9";
const KDE = "#ff7a5c";
const TOOLTIP_BG = "rgba(15, 23, 42, 0.95)";

function fmtTick(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e6 || (a > 0 && a < 1e-3)) return n.toExponential(1);
  return Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, "");
}

function numericChartRows(chart: DistributionNumericChart) {
  return chart.points.map((p, i) => ({
    key: `b${i}`,
    binLabel: fmtTick((p.x_lo + p.x_hi) / 2),
    count: p.count,
    kde: p.kde,
  }));
}

function NumericDistChart({
  title,
  chart,
}: {
  title: string;
  chart: DistributionNumericChart;
}) {
  const rows = numericChartRows(chart);
  const showKde = rows.some((r) => r.kde != null && Number.isFinite(r.kde));
  const kdeData = rows.map((r) => ({
    ...r,
    kdeLine: r.kde != null && Number.isFinite(r.kde) ? r.kde : 0,
  }));

  return (
    <div className="rounded-lg border border-[var(--border)] bg-black/20 p-3">
      <p className="mb-2 font-mono text-[11px] font-medium text-[var(--foreground)]">
        {title}
        <span className="ml-2 font-sans text-[10px] font-normal text-[var(--muted)]">
          n = {chart.non_null_count.toLocaleString()}
        </span>
      </p>
      <p className="mb-2 text-[10px] leading-relaxed text-[var(--muted)]">
        X-axis: bin center (range of values grouped together). Bars: how many rows
        fall in each bin. Pink line: KDE, a smooth estimate of the same shape
        (scaled to match the tallest bar for comparison only).
      </p>
      <div className="h-[260px] w-full min-w-0 md:h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={kdeData} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
            <XAxis
              dataKey="binLabel"
              tick={{ fill: AXIS, fontSize: 10 }}
              interval="preserveStartEnd"
              angle={rows.length > 14 ? -35 : 0}
              textAnchor={rows.length > 14 ? "end" : "middle"}
              height={rows.length > 14 ? 52 : 28}
            />
            <YAxis tick={{ fill: AXIS, fontSize: 10 }} width={36} />
            <Tooltip
              contentStyle={{
                background: TOOLTIP_BG,
                border: "1px solid rgba(148,163,184,0.35)",
                borderRadius: 8,
                fontSize: 11,
              }}
              labelStyle={{ color: "#e2e8f0" }}
              formatter={(value: number, name: string) => [
                typeof value === "number" ? value.toLocaleString() : value,
                name === "count" ? "Count" : "KDE (scaled)",
              ]}
            />
            {showKde && (
              <Legend
                wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                formatter={(v) =>
                  v === "kdeLine" ? "KDE (smoothed)" : "Histogram"
                }
              />
            )}
            <Bar dataKey="count" name="count" fill={BAR} radius={[2, 2, 0, 0]} />
            {showKde && (
              <Line
                type="monotone"
                dataKey="kdeLine"
                name="kdeLine"
                stroke={KDE}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CategoricalDistChart({
  title,
  chart,
}: {
  title: string;
  chart: DistributionCategoricalChart;
}) {
  const data = chart.bars.map((b, i) => ({
    key: `c${i}`,
    label: b.label,
    count: b.count,
  }));

  return (
    <div className="rounded-lg border border-[var(--border)] bg-black/20 p-3">
      <p className="mb-2 font-mono text-[11px] font-medium text-[var(--foreground)]">
        {title}
        <span className="ml-2 font-sans text-[10px] font-normal text-[var(--muted)]">
          n = {chart.non_null_count.toLocaleString()}
        </span>
      </p>
      <p className="mb-2 text-[10px] leading-relaxed text-[var(--muted)]">
        Each bar is one category (or &quot;Other&quot; if many rare levels were
        combined). Bar length is the row count for that value; rare labels may be
        truncated in the chart label.
      </p>
      <div
        className="w-full min-w-0"
        style={{ height: Math.min(360, 48 + data.length * 22) }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 4, right: 12, left: 4, bottom: 4 }}
          >
            <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fill: AXIS, fontSize: 10 }} />
            <YAxis
              type="category"
              dataKey="label"
              width={108}
              tick={{ fill: AXIS, fontSize: 9 }}
              interval={0}
            />
            <Tooltip
              contentStyle={{
                background: TOOLTIP_BG,
                border: "1px solid rgba(148,163,184,0.35)",
                borderRadius: 8,
                fontSize: 11,
              }}
              formatter={(v: number) => [v.toLocaleString(), "Count"]}
            />
            <Bar dataKey="count" fill={BAR} radius={[0, 2, 2, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function OneChart({
  name,
  chart,
}: {
  name: string;
  chart: ColumnDistributionChart;
}) {
  if (chart.kind === "numeric") {
    return <NumericDistChart title={name} chart={chart} />;
  }
  return <CategoricalDistChart title={name} chart={chart} />;
}

export function PreviewDistributionCharts({
  payload,
  targetColumn,
}: {
  payload: DistributionChartsPayload;
  targetColumn?: string;
}) {
  const entries = Object.entries(payload.columns);
  const hasAny = entries.length > 0 || payload.target != null;

  if (!hasAny) {
    return (
      <p className="text-xs text-[var(--muted)]">
        No distribution charts (not enough values in numeric columns, or nothing to
        plot).
      </p>
    );
  }

  return (
    <div className="space-y-5 text-sm">
      <p className="leading-relaxed text-[var(--muted)]">
        Visual summaries of how values are spread. <strong>Numeric</strong>{" "}
        columns use a histogram (bucketed counts) plus an optional KDE curve
        (Gaussian kernel density, scaled to the tallest bar so you can overlay
        shape on counts). <strong>Non-numeric</strong> columns use horizontal bar
        charts of value frequencies. Only aggregated counts are sent to the
        browser, which keeps large files fast. A limited number of columns may be
        charted per dataset; see the note below if some columns are skipped.
      </p>
      {payload.note && (
        <p className="text-[11px] text-amber-200/90">{payload.note}</p>
      )}

      {payload.target && targetColumn && (
        <div>
          <p className="mb-1 text-sm font-bold text-teal-200">
            Target distribution
          </p>
          <p className="mb-2 text-[10px] leading-relaxed text-[var(--muted)]">
            How the outcome is distributed across rows: category counts for
            classification, histogram (+ KDE when applicable) for regression.
            Compare this to feature charts to see whether classes or ranges are
            balanced.
          </p>
          <OneChart name={targetColumn} chart={payload.target} />
        </div>
      )}

      {entries.length > 0 && (
        <div>
          <p className="mb-1 text-[11px] font-medium text-[var(--muted)]">
            Features ({entries.length} column{entries.length === 1 ? "" : "s"})
          </p>
          <p className="mb-2 text-[10px] leading-relaxed text-[var(--muted)]">
            Input columns only (target omitted). Chart type follows column type:
            numbers get histogram/KDE, text/bool/category get bar counts.
          </p>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {entries.map(([col, chart]) => (
              <OneChart key={col} name={col} chart={chart} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
