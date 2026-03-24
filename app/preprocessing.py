"""
Production-style preprocessing: column selection, scaling, IQR clip, KBinsDiscretizer,
and optional SMOTENC/SMOTE before ColumnTransformer (classification only).

All steps are embedded in the fitted Pipeline saved to model.joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    FunctionTransformer,
)

from app.algorithms import resolve_estimator

ScalingMode = Literal["auto", "none", "standard", "minmax", "robust"]
OutlierMode = Literal["none", "clip_iqr"]
ImbalanceMode = Literal["none", "smote"]
BinStrategy = Literal["uniform", "quantile"]


@dataclass
class NumericBinSpec:
    column: str
    n_bins: int = 5
    strategy: BinStrategy = "quantile"


@dataclass
class PreprocessConfig:
    """Resolved preprocessing options (from API / defaults)."""

    feature_columns: list[str] | None = None
    scaling: ScalingMode = "auto"
    numeric_outliers: OutlierMode = "none"
    imbalance: ImbalanceMode = "none"
    numeric_bins: list[NumericBinSpec] = field(default_factory=list)


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _clip_iqr_2d(X: np.ndarray) -> np.ndarray:
    """Per-column IQR clipping; NaNs preserved."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    out = np.array(X, copy=True, dtype=float)
    factor = 1.5
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 4:
            continue
        vals = col[mask]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        out[:, j] = np.clip(col, lo, hi)
    return out


def _make_clipper() -> FunctionTransformer:
    return FunctionTransformer(
        func=_clip_iqr_2d,
        feature_names_out="one-to-one",
        validate=False,
    )


def _scaler_for_mode(mode: ScalingMode, algorithm_wants_scale: bool) -> Any | None:
    if mode == "none":
        return None
    if mode == "auto":
        return StandardScaler() if algorithm_wants_scale else None
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler()
    return None


def _cat_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def build_column_transformer(
    X: pd.DataFrame,
    *,
    scaling: ScalingMode,
    algorithm_wants_scale: bool,
    numeric_outliers: OutlierMode,
    bin_specs: dict[str, NumericBinSpec],
) -> ColumnTransformer:
    """
    Build ColumnTransformer for feature frame X (already column-filtered).
    bin_specs: column -> spec for numeric columns to discretize.
    """
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in X.columns:
        s = X[col]
        if _is_numeric_series(s):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    binned_set = set(bin_specs.keys()) if bin_specs else set()
    for c in binned_set:
        if c not in num_cols:
            raise ValueError(
                f"Binning column {c!r} is not numeric or not in feature columns."
            )

    plain_num = [c for c in num_cols if c not in binned_set]
    binned_num = [c for c in num_cols if c in binned_set]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    scaler = _scaler_for_mode(scaling, algorithm_wants_scale)

    for col in binned_num:
        spec = bin_specs[col]
        n_bins = max(2, min(50, int(spec.n_bins)))
        strat: BinStrategy = spec.strategy if spec.strategy in ("uniform", "quantile") else "quantile"
        steps: list[tuple[str, Any]] = []
        if numeric_outliers == "clip_iqr":
            steps.append(("clip", _make_clipper()))
        steps.append(("imputer", SimpleImputer(strategy="median")))
        steps.append(
            (
                "kbd",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="ordinal",
                    strategy=strat,
                    subsample=None,
                ),
            )
        )
        transformers.append((f"num_bin_{col}", Pipeline(steps=steps), [col]))

    if plain_num:
        steps_plain: list[tuple[str, Any]] = []
        if numeric_outliers == "clip_iqr":
            steps_plain.append(("clip", _make_clipper()))
        steps_plain.append(("imputer", SimpleImputer(strategy="median")))
        if scaler is not None:
            steps_plain.append(("scaler", scaler))
        transformers.append(
            ("num_plain", Pipeline(steps=steps_plain), plain_num),
        )

    if cat_cols:
        transformers.append(("cat", _cat_pipeline(), cat_cols))

    if not transformers:
        raise ValueError("No feature columns after preprocessing split.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def wrap_with_imbalance_sampler(
    X: pd.DataFrame,
    *,
    preprocess: ColumnTransformer,
    model: Any,
    task: str,
    imbalance: ImbalanceMode,
    random_state: int,
) -> Pipeline:
    """Optionally prepend SMOTENC/SMOTE (raw feature space) before preprocess + model."""
    if imbalance != "smote" or task != "classification":
        return Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as e:
        raise ValueError(
            "SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn"
        ) from e

    cat_idx: list[int] = []
    for i, c in enumerate(X.columns):
        if not _is_numeric_series(X[c]):
            cat_idx.append(i)

    if cat_idx:
        smote = SMOTENC(
            categorical_features=cat_idx,
            random_state=random_state,
        )
    else:
        smote = SMOTE(random_state=random_state)

    return ImbPipeline(
        steps=[
            ("sampler", smote),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def preprocess_config_from_dict(d: dict[str, Any] | None) -> PreprocessConfig:
    if not d:
        return PreprocessConfig()
    bins_raw = d.get("numeric_bins") or []
    bins: list[NumericBinSpec] = []
    for b in bins_raw:
        if isinstance(b, dict) and b.get("column"):
            strat = b.get("strategy", "quantile")
            if strat not in ("uniform", "quantile"):
                strat = "quantile"
            bins.append(
                NumericBinSpec(
                    column=str(b["column"]),
                    n_bins=int(b.get("n_bins", 5)),
                    strategy=strat,  # type: ignore[arg-type]
                )
            )
    fc = d.get("feature_columns")
    feature_columns = [str(x) for x in fc] if isinstance(fc, list) else None
    scaling = d.get("scaling", "auto")
    if scaling not in ("auto", "none", "standard", "minmax", "robust"):
        scaling = "auto"
    no = d.get("numeric_outliers", "none")
    if no not in ("none", "clip_iqr"):
        no = "none"
    imb = d.get("imbalance", "none")
    if imb not in ("none", "smote"):
        imb = "none"
    return PreprocessConfig(
        feature_columns=feature_columns,
        scaling=scaling,  # type: ignore[arg-type]
        numeric_outliers=no,  # type: ignore[arg-type]
        imbalance=imb,  # type: ignore[arg-type]
        numeric_bins=bins,
    )


def config_to_jsonable(cfg: PreprocessConfig) -> dict[str, Any]:
    return {
        "feature_columns": cfg.feature_columns,
        "scaling": cfg.scaling,
        "numeric_outliers": cfg.numeric_outliers,
        "imbalance": cfg.imbalance,
        "numeric_bins": [
            {"column": b.column, "n_bins": b.n_bins, "strategy": b.strategy}
            for b in cfg.numeric_bins
        ],
    }
