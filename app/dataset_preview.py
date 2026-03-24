"""CSV profile for UI: sample rows, dtypes, missing counts, inferred supervised task."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder

from app.automl import infer_task


def _finite_float(x: Any, nd: int = 6) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _column_summary(s: pd.Series) -> dict[str, Any]:
    """EDA stats: numeric (min/max/mean/median/std/zero %) or categorical top values."""
    n_non_null = int(s.notna().sum())

    if pd.api.types.is_bool_dtype(s):
        vc = s.value_counts(dropna=False).head(5)
        top: list[dict[str, Any]] = []
        for k, cnt in vc.items():
            key = "(null)" if pd.isna(k) else str(bool(k))
            top.append({"value": key, "count": int(cnt)})
        return {"kind": "categorical", "top_values": top}

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        vn = v.dropna()
        if len(vn) == 0:
            return {
                "kind": "numeric",
                "note": "no_numeric_values",
            }
        zeros = int((vn == 0).sum())
        zero_pct = round(100.0 * zeros / len(vn), 2) if len(vn) else 0.0
        return {
            "kind": "numeric",
            "min": _finite_float(vn.min()),
            "max": _finite_float(vn.max()),
            "mean": _finite_float(vn.mean()),
            "median": _finite_float(vn.median()),
            "std": _finite_float(vn.std(ddof=0), nd=6),
            "zero_pct": zero_pct,
            "non_null_count": int(len(vn)),
        }

    vc = s.astype(str).value_counts(dropna=False).head(5)
    top_cat: list[dict[str, Any]] = []
    for k, cnt in vc.items():
        if pd.isna(k) or str(k).lower() == "nan":
            label = "(null)"
        else:
            label = str(k) if len(str(k)) <= 80 else str(k)[:77] + "…"
        top_cat.append({"value": label, "count": int(cnt)})
    return {
        "kind": "categorical",
        "top_values": top_cat,
        "non_null_count": n_non_null,
    }


def _numeric_column_names(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def _numeric_correlation_block(
    df: pd.DataFrame,
    *,
    max_full_matrix: int = 18,
    max_pairs: int = 60,
) -> dict[str, Any] | None:
    """Pearson correlation between numeric columns: small matrix or top |r| pairs."""
    num_cols = _numeric_column_names(df)
    if len(num_cols) < 2:
        return None
    sub = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson", min_periods=3)

    def cell(a: str, b: str) -> float | None:
        v = corr.loc[a, b]
        if pd.isna(v):
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        return round(f, 4)

    if len(num_cols) <= max_full_matrix:
        matrix: list[list[float | None]] = []
        for a in num_cols:
            row = [cell(a, b) for b in num_cols]
            matrix.append(row)
        return {
            "format": "matrix",
            "columns": num_cols,
            "matrix": matrix,
        }

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1 :]:
            r = cell(a, b)
            if r is None:
                continue
            pairs.append({"column_a": a, "column_b": b, "r": r})
    pairs.sort(key=lambda x: -abs(x["r"]))
    return {
        "format": "top_pairs",
        "pairs": pairs[:max_pairs],
        "note": f"{len(num_cols)} numeric columns; showing top {max_pairs} pairs by |r|.",
    }


def _feature_target_association(
    df: pd.DataFrame,
    target: str,
    task: str,
) -> list[dict[str, Any]]:
    """Univariate association: Pearson (regression, numeric features) + mutual information."""
    feats = [c for c in df.columns if c != target]
    y_series = df[target]
    rows: list[dict[str, Any]] = []

    if task == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_series.astype(str).fillna("__na__").to_numpy())
        for c in feats:
            s = df[c]
            try:
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(
                    s
                ):
                    x = (
                        pd.to_numeric(s, errors="coerce")
                        .to_numpy(dtype=float)
                        .reshape(-1, 1)
                    )
                    mask = ~np.isnan(x[:, 0])
                    if int(mask.sum()) < 5:
                        rows.append(
                            {
                                "column": c,
                                "pearson_r": None,
                                "mutual_info": None,
                                "note": "too_few_values",
                            }
                        )
                        continue
                    mi = mutual_info_classif(
                        x[mask],
                        y[mask],
                        discrete_features=[False],
                        random_state=0,
                    )[0]
                    rows.append(
                        {
                            "column": c,
                            "pearson_r": None,
                            "mutual_info": round(float(mi), 6),
                        }
                    )
                else:
                    le_x = LabelEncoder()
                    x = le_x.fit_transform(
                        s.astype(str).fillna("__na__").to_numpy()
                    ).reshape(-1, 1)
                    mi = mutual_info_classif(
                        x,
                        y,
                        discrete_features=[True],
                        random_state=0,
                    )[0]
                    rows.append(
                        {
                            "column": c,
                            "pearson_r": None,
                            "mutual_info": round(float(mi), 6),
                        }
                    )
            except Exception:
                rows.append(
                    {
                        "column": c,
                        "pearson_r": None,
                        "mutual_info": None,
                        "note": "mi_failed",
                    }
                )
    else:
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
        for c in feats:
            s = df[c]
            try:
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(
                    s
                ):
                    x = (
                        pd.to_numeric(s, errors="coerce")
                        .to_numpy(dtype=float)
                        .reshape(-1, 1)
                    )
                    mask = ~np.isnan(x[:, 0]) & ~np.isnan(y)
                    if int(mask.sum()) < 5:
                        rows.append(
                            {
                                "column": c,
                                "pearson_r": None,
                                "mutual_info": None,
                                "note": "too_few_values",
                            }
                        )
                        continue
                    xv, yv = x[mask, 0], y[mask]
                    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                        pr = None
                    else:
                        m = np.corrcoef(xv, yv)[0, 1]
                        pr = (
                            round(float(m), 4)
                            if math.isfinite(float(m))
                            else None
                        )
                    mi = mutual_info_regression(
                        x[mask],
                        y[mask],
                        random_state=0,
                    )[0]
                    rows.append(
                        {
                            "column": c,
                            "pearson_r": pr,
                            "mutual_info": round(float(mi), 6),
                        }
                    )
                else:
                    le_x = LabelEncoder()
                    x = le_x.fit_transform(
                        s.astype(str).fillna("__na__").to_numpy()
                    ).reshape(-1, 1)
                    mask = ~np.isnan(y)
                    if int(mask.sum()) < 5:
                        rows.append(
                            {
                                "column": c,
                                "pearson_r": None,
                                "mutual_info": None,
                                "note": "too_few_values",
                            }
                        )
                        continue
                    mi = mutual_info_regression(
                        x[mask],
                        y[mask],
                        random_state=0,
                    )[0]
                    rows.append(
                        {
                            "column": c,
                            "pearson_r": None,
                            "mutual_info": round(float(mi), 6),
                        }
                    )
            except Exception:
                rows.append(
                    {
                        "column": c,
                        "pearson_r": None,
                        "mutual_info": None,
                        "note": "failed",
                    }
                )

    rows.sort(
        key=lambda r: (
            -(r.get("mutual_info") or 0.0),
            -(abs(r["pearson_r"]) if r.get("pearson_r") is not None else 0.0),
        )
    )
    return rows


def _task_explanation(task: str) -> str:
    if task == "classification":
        return (
            "The target column looks categorical or has few distinct numeric values "
            "(20 or fewer unique). Training will use classification metrics."
        )
    return (
        "The target column looks numeric with many distinct values. "
        "Training will use regression metrics."
    )


def _truncate_chart_label(s: str, max_len: int = 40) -> str:
    t = str(s)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _numeric_distribution_chart(s: pd.Series) -> dict[str, Any] | None:
    """Histogram bin counts + KDE curve (scaled to histogram peak) for client charts."""
    v_raw = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    v = v_raw[np.isfinite(v_raw)]
    n = int(len(v))
    if n < 3:
        return None
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return None

    if abs(vmax - vmin) < 1e-15:
        return {
            "kind": "numeric",
            "non_null_count": n,
            "points": [
                {
                    "x_lo": round(vmin, 6),
                    "x_hi": round(vmax, 6),
                    "count": n,
                    "kde": None,
                }
            ],
            "kde_x": [],
            "kde_y": [],
            "x_min": vmin,
            "x_max": vmax,
        }

    try:
        edges = np.histogram_bin_edges(v, bins="auto")
    except (ValueError, TypeError):
        edges = np.linspace(vmin, vmax, num=min(25, max(11, int(np.sqrt(n)) + 1)))
    if len(edges) < 2:
        return None
    if len(edges) > 42:
        edges = np.histogram_bin_edges(v, bins=40)
    counts, edges = np.histogram(v, bins=edges)
    peak = float(np.max(counts)) if len(counts) else 0.0

    kde_x_list: list[float] = []
    kde_y_list: list[float] = []
    kde_y_at_mids: list[float | None] = []

    if n >= 5:
        std = float(np.std(v))
        q75, q25 = np.percentile(v, [75, 25])
        iqr = float(q75 - q25)
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if sigma <= 0 or not math.isfinite(sigma):
            sigma = (vmax - vmin) / 10.0
        bw = sigma * (n ** (-1.0 / 5.0))
        if bw <= 0 or not math.isfinite(bw):
            bw = (vmax - vmin) / 20.0
        try:
            kde_m = KernelDensity(kernel="gaussian", bandwidth=bw).fit(v.reshape(-1, 1))
            n_x = min(96, max(48, n))
            xs = np.linspace(vmin, vmax, n_x)
            log_d = kde_m.score_samples(xs.reshape(-1, 1))
            dens = np.exp(log_d)
            dmax = float(np.max(dens)) if len(dens) else 0.0
            if dmax > 0 and peak > 0:
                dens_scaled = dens / dmax * peak
            else:
                dens_scaled = dens
            kde_x_list = [round(float(x), 6) for x in xs]
            kde_y_list = [round(float(y), 6) for y in dens_scaled]
            mids = (edges[:-1] + edges[1:]) / 2.0
            kde_y_at_mids = [
                round(float(np.interp(float(mid), xs, dens_scaled)), 6) for mid in mids
            ]
        except Exception:
            kde_y_at_mids = [None] * len(counts)
    else:
        kde_y_at_mids = [None] * len(counts)

    points: list[dict[str, Any]] = []
    for i in range(len(counts)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        ky = kde_y_at_mids[i] if i < len(kde_y_at_mids) else None
        points.append(
            {
                "x_lo": round(lo, 6),
                "x_hi": round(hi, 6),
                "count": int(counts[i]),
                "kde": ky,
            }
        )

    return {
        "kind": "numeric",
        "non_null_count": n,
        "points": points,
        "kde_x": kde_x_list,
        "kde_y": kde_y_list,
        "x_min": vmin,
        "x_max": vmax,
    }


def _categorical_distribution_chart(
    s: pd.Series,
    *,
    max_bars: int = 16,
) -> dict[str, Any] | None:
    vc = s.value_counts(dropna=False)
    if len(vc) == 0:
        return None
    bars: list[dict[str, Any]] = []
    other = 0
    for i, (k, cnt) in enumerate(vc.items()):
        if i < max_bars:
            if pd.isna(k):
                label = "(null)"
            else:
                label = _truncate_chart_label(str(k))
            bars.append({"label": label, "count": int(cnt)})
        else:
            other += int(cnt)
    if other > 0:
        bars.append({"label": "Other", "count": other})
    return {
        "kind": "categorical",
        "non_null_count": int(s.notna().sum()),
        "bars": bars,
    }


def _distribution_charts_payload(
    df: pd.DataFrame,
    *,
    target_column: str | None,
    task: str | None,
    max_numeric: int = 20,
    max_categorical: int = 20,
) -> dict[str, Any]:
    """Aggregated counts only (no raw rows) for client-side histogram / bar / KDE plots."""
    columns_dict: dict[str, Any] = {}
    numeric_added = 0
    categorical_added = 0
    truncated = False

    for c in df.columns:
        if c == target_column:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if numeric_added >= max_numeric:
                truncated = True
                continue
            ch = _numeric_distribution_chart(s)
            if ch is not None:
                columns_dict[c] = ch
                numeric_added += 1
        else:
            if categorical_added >= max_categorical:
                truncated = True
                continue
            ch = _categorical_distribution_chart(s)
            if ch is not None:
                columns_dict[c] = ch
                categorical_added += 1

    tgt: dict[str, Any] | None = None
    if target_column and target_column in df.columns and task in (
        "classification",
        "regression",
    ):
        ts = df[target_column]
        if task == "classification":
            tgt = _categorical_distribution_chart(ts, max_bars=24)
        else:
            tgt = _numeric_distribution_chart(ts)

    note: str | None = None
    if truncated:
        note = (
            f"Feature charts limited to the first {max_numeric} numeric and "
            f"{max_categorical} non-numeric columns in file order."
        )

    return {
        "columns": columns_dict,
        "target": tgt,
        "truncated": truncated,
        "note": note,
    }


_LEAKY_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(uuid|guid)\b", re.I), "uuid_or_guid"),
    (re.compile(r"\b(row_?id|record_?id)\b", re.I), "row_or_record_id"),
    (re.compile(r"(?:_id|_uuid|_guid)$", re.I), "suffix_id"),
    (re.compile(r"\bid\b", re.I), "id_token"),
    (
        re.compile(r"^index$|^_index_$|\bidx\b|\brow_?index\b", re.I),
        "index_like",
    ),
    (
        re.compile(
            r"\b(timestamp|datetime)\b|created|updated|modified|_date_|_at$|_date$|^date$",
            re.I,
        ),
        "datetime_like",
    ),
]


def _leaky_column_name_pattern(column: str) -> str | None:
    for rx, label in _LEAKY_NAME_PATTERNS:
        if rx.search(column):
            return label
    return None


def _feature_target_bijection(
    sub: pd.DataFrame,
    feature: str,
    target: str,
    *,
    min_rows: int = 30,
    min_levels: int = 3,
) -> bool:
    """True if feature and target label each other 1:1 on non-null rows (classification)."""
    if len(sub) < min_rows:
        return False
    gc = sub.groupby(feature, dropna=True)[target].nunique(dropna=False)
    if int(gc.max()) != 1:
        return False
    gt = sub.groupby(target, dropna=True)[feature].nunique(dropna=False)
    if int(gt.max()) != 1:
        return False
    nu_f = int(sub[feature].nunique(dropna=True))
    nu_t = int(sub[target].nunique(dropna=True))
    return nu_f == nu_t and nu_f >= min_levels


def _data_quality_checks(
    df: pd.DataFrame,
    target_column: str | None,
    task: str | None,
) -> dict[str, Any]:
    """Lightweight EDA warnings: duplicates, constants, cardinality, IDs, leaky-name hints."""
    n = len(df)
    dup = int(df.duplicated().sum()) if n else 0
    dup_pct = round(100.0 * dup / n, 2) if n else 0.0

    constant_columns: list[dict[str, Any]] = []
    high_cardinality: list[dict[str, Any]] = []
    id_like_columns: list[dict[str, Any]] = []
    leaky_column_hints: list[dict[str, Any]] = []

    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        nu = int(s.nunique(dropna=True))
        if nn > 0 and nu <= 1:
            sample = s.dropna().iloc[0]
            sample_s = (
                "(null)"
                if pd.isna(sample)
                else (str(sample) if len(str(sample)) <= 64 else str(sample)[:61] + "…")
            )
            constant_columns.append({"column": c, "value_sample": sample_s})

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        nu = int(s.nunique(dropna=True))
        if nu < 2:
            continue
        ratio = nu / n if n else 0.0
        warn = False
        if nu >= 80:
            warn = True
        elif nu >= 50 and n >= 80:
            warn = True
        elif n >= 40 and ratio >= 0.12 and nu >= 25:
            warn = True
        if warn:
            high_cardinality.append(
                {
                    "column": c,
                    "n_unique": nu,
                    "unique_to_rows_ratio": round(ratio, 4),
                    "note": "many_levels",
                }
            )

    for c in df.columns:
        if c == target_column:
            continue
        s = df[c]
        nn = int(s.notna().sum())
        if nn < 10:
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        nu = int(s.nunique(dropna=True))
        u_ratio = nu / nn if nn else 0.0
        if u_ratio >= 0.97 or (nu == nn and nn >= 20):
            id_like_columns.append(
                {
                    "column": c,
                    "n_unique": nu,
                    "non_null_count": nn,
                    "unique_ratio": round(u_ratio, 4),
                }
            )

    if (
        target_column
        and target_column in df.columns
        and task == "classification"
    ):
        for c in df.columns:
            if c == target_column:
                continue
            pattern = _leaky_column_name_pattern(c)
            if not pattern:
                continue
            sub = df[[c, target_column]].dropna()
            if not _feature_target_bijection(sub, c, target_column):
                continue
            leaky_column_hints.append(
                {
                    "column": c,
                    "name_pattern": pattern,
                    "note": (
                        "Column name suggests id/date/index and values line up 1:1 with "
                        "target classes on non-null rows - possible leakage or row key; "
                        "verify before using as a feature."
                    ),
                }
            )

    return {
        "duplicate_row_count": dup,
        "duplicate_row_pct": dup_pct,
        "constant_columns": constant_columns,
        "high_cardinality_categoricals": high_cardinality,
        "id_like_columns": id_like_columns,
        "leaky_column_hints": leaky_column_hints,
        "heuristic_warning": (
            "Heuristic checks only — confirm in your domain before dropping columns."
        ),
    }


def build_preview(
    csv_path: Path,
    sample_rows: int = 15,
    target_column: str | None = None,
) -> dict[str, Any]:
    sample_rows = max(5, min(20, int(sample_rows)))
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}") from e

    if df.empty:
        raise ValueError("CSV has no data rows.")

    n = len(df)
    sample_df = df.head(sample_rows)
    try:
        sample_json: list[dict[str, Any]] = json.loads(
            sample_df.to_json(orient="records", date_format="iso")
        )
    except Exception:
        sample_json = sample_df.astype(object).where(pd.notna(sample_df), None).to_dict(
            orient="records"
        )

    miss = df.isna().sum()
    col_profiles: list[dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        mc = int(miss[c])
        prof: dict[str, Any] = {
            "name": c,
            "dtype": str(s.dtype),
            "missing_count": mc,
            "missing_pct": round(float(mc / n) * 100, 2) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
            "summary": _column_summary(s),
        }
        col_profiles.append(prof)

    out: dict[str, Any] = {
        "n_rows": n,
        "n_columns": len(df.columns),
        "sample_rows": sample_json,
        "sample_row_count": len(sample_json),
        "columns": col_profiles,
    }

    out["numeric_correlation"] = _numeric_correlation_block(df)

    if target_column is not None:
        t = str(target_column).strip()
        if not t:
            pass
        elif t not in df.columns:
            raise ValueError(f"Target column {t!r} is not in the CSV.")
        else:
            y = df[t]
            task = infer_task(y)
            out["target_column"] = t
            out["inferred_task"] = task
            out["target_task_explanation"] = _task_explanation(task)
            out["target_n_unique"] = int(y.nunique(dropna=True))
            out["target_missing_count"] = int(y.isna().sum())
            out["target_missing_pct"] = (
                round(float(y.isna().sum() / n) * 100, 2) if n else 0.0
            )

            if out.get("inferred_task") == "classification":
                vc = y.astype(str).value_counts()
                mins = int(vc.min())
                majs = int(vc.max())
                ratio = float(majs / mins) if mins > 0 else None
                minority_pct = float(mins / n * 100) if n else 0.0
                out["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
                out["imbalance_ratio_max_to_min"] = (
                    round(ratio, 2) if ratio is not None else None
                )
                out["minority_class_pct"] = round(minority_pct, 2)
                out["imbalance_warning"] = minority_pct < 5.0 or (
                    ratio is not None and ratio > 10
                )
            else:
                out["class_distribution"] = None
                out["imbalance_ratio_max_to_min"] = None
                out["minority_class_pct"] = None
                out["imbalance_warning"] = False

            feat_cols = [c for c in df.columns if c != t]
            out["numeric_outlier_summary"] = _numeric_outlier_summary(df, feat_cols)
            out["feature_target_association"] = _feature_target_association(
                df, t, task
            )

    if "numeric_outlier_summary" not in out:
        out["numeric_outlier_summary"] = _numeric_outlier_summary(
            df, list(df.columns)
        )

    tcol = out.get("target_column")
    ttask = out.get("inferred_task")
    out["distribution_charts"] = _distribution_charts_payload(
        df,
        target_column=tcol if isinstance(tcol, str) else None,
        task=ttask if ttask in ("classification", "regression") else None,
    )

    out["data_quality"] = _data_quality_checks(
        df,
        tcol if isinstance(tcol, str) else None,
        ttask if ttask in ("classification", "regression") else None,
    )

    return out


def _numeric_outlier_summary(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """IQR-based outlier share per numeric column (for UI guidance only)."""
    rows: list[dict] = []
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c]
        if not (
            pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        ):
            continue
        v = pd.to_numeric(s, errors="coerce").dropna()
        if len(v) < 10:
            rows.append({"column": c, "iqr_outlier_pct": None, "note": "too_few_values"})
            continue
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = float(q3 - q1)
        if iqr <= 0:
            rows.append({"column": c, "iqr_outlier_pct": 0.0, "note": "zero_iqr"})
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct = float(((v < lo) | (v > hi)).mean() * 100)
        rows.append({"column": c, "iqr_outlier_pct": round(pct, 2)})
    return rows
