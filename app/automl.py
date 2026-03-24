from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from app.algorithms import default_algorithm_id, resolve_estimator
from app.preprocessing import (
    NumericBinSpec,
    build_column_transformer,
    config_to_jsonable,
    preprocess_config_from_dict,
    wrap_with_imbalance_sampler,
)
from app.unsupervised_algorithms import (
    Family,
    resolve_unsupervised,
    unsupervised_label_for,
)

InferenceMethod = Literal["predict", "transform", "none"]


@dataclass
class TrainResult:
    task: str
    metrics: dict[str, Any]
    feature_columns: list[str]
    label_classes: list[str] | None
    algorithm_id: str
    algorithm_label: str
    learning_mode: Literal["supervised", "unsupervised"] = "supervised"
    unsupervised_family: str | None = None
    inference_method: InferenceMethod = "predict"
    target_column: str | None = None
    # First training row's feature values, JSON-safe, for predict API examples.
    prediction_example_row: dict[str, Any] | None = None
    preprocess_applied: dict[str, Any] | None = None


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _json_safe_prediction_row(row: pd.Series) -> dict[str, Any]:
    """Single raw feature row for metadata / UI (matches /api/predict row shape)."""
    out: dict[str, Any] = {}
    for key in row.index:
        v = row[key]
        k = str(key)
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            fv = float(v)
            out[k] = None if np.isnan(fv) else fv
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif isinstance(v, str):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def infer_task(y: pd.Series) -> str:
    if not _is_numeric_series(y):
        return "classification"
    n_unique = int(y.nunique(dropna=True))
    if n_unique <= 20:
        return "classification"
    return "regression"


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in X.columns:
        s = X[col]
        if _is_numeric_series(s):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def build_feature_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        num_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
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
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    task: str,
    algorithm_id: str,
) -> Pipeline:
    model, scale_numeric = resolve_estimator(algorithm_id, task)  # type: ignore[arg-type]
    preprocess = build_feature_preprocessor(num_cols, cat_cols, scale_numeric)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def build_unsupervised_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    algorithm_id: str,
    family: Family,
) -> tuple[Pipeline, InferenceMethod]:
    model, scale_numeric, inference = resolve_unsupervised(algorithm_id, family)
    preprocess = build_feature_preprocessor(num_cols, cat_cols, scale_numeric)
    return (
        Pipeline(steps=[("preprocess", preprocess), ("model", model)]),
        inference,
    )


def _algorithm_label_supervised(algorithm_id: str) -> str:
    from app.algorithms import list_algorithms_public

    for row in list_algorithms_public():
        if row["id"] == algorithm_id:
            return row["label"]
    return algorithm_id


def _adapt_estimator_to_n_samples(model: Any, n: int) -> None:
    """Shrink discrete cluster/component counts so small CSVs still fit."""
    if n < 2:
        return
    cap = max(1, n - 1)
    params: dict[str, Any] = {}
    p = model.get_params(deep=False)
    if "n_clusters" in p:
        v = p["n_clusters"]
        if isinstance(v, int) and v > cap:
            params["n_clusters"] = max(1, cap)
    if "n_components" in p:
        v = p["n_components"]
        if isinstance(v, int) and v > cap:
            params["n_components"] = max(1, cap)
    if params:
        model.set_params(**params)


def _compute_unsupervised_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    family: str,
) -> dict[str, Any]:
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_t = pre.transform(X)
    metrics: dict[str, Any] = {
        "n_samples": int(len(X)),
        "n_features_transformed": int(X_t.shape[1]),
    }

    if family == "clustering":
        labels = None
        if hasattr(model, "labels_"):
            labels = np.asarray(model.labels_)
        elif hasattr(model, "predict"):
            labels = np.asarray(model.predict(X_t))
        if labels is not None:
            valid = labels >= 0
            uniq = np.unique(labels[valid]) if np.any(valid) else np.array([])
            metrics["n_label_values"] = int(len(uniq))
            if len(uniq) >= 2 and np.sum(valid) >= len(uniq) + 1:
                try:
                    metrics["silhouette"] = float(
                        silhouette_score(X_t[valid], labels[valid])
                    )
                    metrics["calinski_harabasz"] = float(
                        calinski_harabasz_score(X_t[valid], labels[valid])
                    )
                    metrics["davies_bouldin"] = float(
                        davies_bouldin_score(X_t[valid], labels[valid])
                    )
                except Exception as e:
                    metrics["cluster_metric_note"] = str(e)
        return metrics

    if family == "decomposition":
        evr = getattr(model, "explained_variance_ratio_", None)
        if evr is not None:
            ev = np.asarray(evr)
            metrics["explained_variance_ratio_sum"] = float(np.sum(ev))
            metrics["n_components"] = int(ev.shape[0])
        rec = getattr(model, "reconstruction_err_", None)
        if rec is not None:
            metrics["reconstruction_error"] = float(rec)
        return metrics

    if family == "anomaly":
        if hasattr(model, "score_samples"):
            try:
                s = model.score_samples(X_t)
                metrics["anomaly_score_mean"] = float(np.mean(s))
                metrics["anomaly_score_std"] = float(np.std(s))
            except Exception as e:
                metrics["anomaly_metric_note"] = str(e)
        if hasattr(model, "predict"):
            try:
                yh = np.asarray(model.predict(X_t))
                n_out = int(np.sum(yh == -1))
                metrics["rows_flagged_outlier"] = n_out
                if len(yh):
                    metrics["outlier_fraction_fit_data"] = float(n_out / len(yh))
            except Exception:
                pass
        return metrics

    return metrics


def train_unsupervised_tabular(
    csv_path: Path,
    algorithm_id: str,
    family: Family,
    exclude_columns: list[str],
    preprocess_opts: dict[str, Any] | None = None,
) -> tuple[Pipeline, TrainResult, None]:
    df = pd.read_csv(csv_path)
    cfg = preprocess_config_from_dict(preprocess_opts)
    if cfg.imbalance != "none":
        raise ValueError(
            "Imbalance handling (SMOTE) is only for supervised classification."
        )
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown feature columns: {missing}")
        X = df[cfg.feature_columns].copy()
    else:
        drop_set = {c for c in exclude_columns if c in df.columns}
        X = df.drop(columns=list(drop_set), errors="ignore")
    feature_columns = list(X.columns)

    if not feature_columns:
        raise ValueError("No feature columns left after exclusions.")

    if len(X) < 5:
        raise ValueError("Need at least 5 rows for unsupervised training.")

    num_cols, cat_cols = _split_columns(X)
    if not num_cols and not cat_cols:
        raise ValueError("No usable columns after preprocessing split.")

    model, scale_numeric, inference = resolve_unsupervised(algorithm_id, family)
    bin_map: dict[str, NumericBinSpec] = {b.column: b for b in cfg.numeric_bins}
    preprocess = build_column_transformer(
        X,
        scaling=cfg.scaling,
        algorithm_wants_scale=scale_numeric,
        numeric_outliers=cfg.numeric_outliers,
        bin_specs=bin_map,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    _adapt_estimator_to_n_samples(pipeline.named_steps["model"], len(X))
    pipeline.fit(X)

    metrics = _compute_unsupervised_metrics(pipeline, X, family)

    result = TrainResult(
        task="unsupervised",
        metrics=metrics,
        feature_columns=feature_columns,
        label_classes=None,
        algorithm_id=algorithm_id,
        algorithm_label=unsupervised_label_for(algorithm_id) or algorithm_id,
        learning_mode="unsupervised",
        unsupervised_family=family,
        inference_method=inference,
        target_column=None,
        prediction_example_row=_json_safe_prediction_row(X.iloc[0]),
        preprocess_applied=config_to_jsonable(cfg),
    )
    return pipeline, result, None


def train_tabular(
    csv_path: Path,
    target_column: str,
    test_size: float,
    algorithm_id: str | None = None,
    random_state: int = 42,
    preprocess_opts: dict[str, Any] | None = None,
) -> tuple[Pipeline, TrainResult, LabelEncoder | None]:
    aid = algorithm_id or default_algorithm_id()
    cfg = preprocess_config_from_dict(preprocess_opts)

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset.")

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Unknown feature columns: {missing}")
        X = X[cfg.feature_columns].copy()
    feature_columns = list(X.columns)

    task = infer_task(y_raw)
    if task == "classification":
        n_unique_target = int(y_raw.nunique(dropna=True))
        if n_unique_target > 400:
            raise ValueError(
                f"Target has {n_unique_target} unique values—too many distinct classes for classification."
            )
    label_encoder: LabelEncoder | None = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            mask = y.notna()
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)
        y = y.to_numpy(dtype=float)

    if len(X) < 10:
        raise ValueError("Need at least 10 rows after cleaning.")

    if task != "classification" and cfg.imbalance == "smote":
        raise ValueError(
            "SMOTE applies only to classification. Set imbalance to none for regression."
        )

    num_cols, cat_cols = _split_columns(X)
    if not num_cols and not cat_cols:
        raise ValueError("No usable columns after preprocessing split.")

    estimator, algorithm_wants_scale = resolve_estimator(aid, task)  # type: ignore[arg-type]
    bin_map = {b.column: b for b in cfg.numeric_bins}
    preprocess = build_column_transformer(
        X,
        scaling=cfg.scaling,
        algorithm_wants_scale=algorithm_wants_scale,
        numeric_outliers=cfg.numeric_outliers,
        bin_specs=bin_map,
    )
    pipeline = wrap_with_imbalance_sampler(
        X,
        preprocess=preprocess,
        model=estimator,
        task=task,
        imbalance=cfg.imbalance,
        random_state=random_state,
    )

    stratify = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    sampler_step = pipeline.named_steps.get("sampler")
    if sampler_step is not None and task == "classification":
        _, counts = np.unique(y_train, return_counts=True)
        m = int(counts.min())
        if m < 2:
            raise ValueError(
                "SMOTE needs at least 2 training samples per class. "
                "Use imbalance none or collect more data."
            )
        k_smote = min(5, max(1, m - 1))
        sampler_step.set_params(k_neighbors=k_smote)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(
                f1_score(y_test, preds, average="macro", zero_division=0)
            ),
            "test_rows": int(len(y_test)),
            "train_rows": int(len(y_train)),
        }
        label_classes = (
            label_encoder.classes_.tolist() if label_encoder is not None else None
        )
    else:
        rmse = float(root_mean_squared_error(y_test, preds))
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": rmse,
            "test_rows": int(len(y_test)),
            "train_rows": int(len(y_train)),
        }
        label_classes = None

    result = TrainResult(
        task=task,
        metrics=metrics,
        feature_columns=feature_columns,
        label_classes=label_classes,
        algorithm_id=aid,
        algorithm_label=_algorithm_label_supervised(aid),
        learning_mode="supervised",
        unsupervised_family=None,
        inference_method="predict",
        target_column=target_column,
        prediction_example_row=_json_safe_prediction_row(X.iloc[0]),
        preprocess_applied=config_to_jsonable(cfg),
    )
    return pipeline, result, label_encoder


def save_artifact(
    out_dir: Path,
    pipeline: Pipeline,
    result: TrainResult,
    label_encoder: LabelEncoder | None,
    target_column: str | None,
    dataset_id: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")
    meta = {
        "task": result.task,
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "prediction_example_row": result.prediction_example_row,
        "label_classes": result.label_classes,
        "target_column": target_column,
        "dataset_id": dataset_id,
        "algorithm_id": result.algorithm_id,
        "algorithm_label": result.algorithm_label,
        "learning_mode": result.learning_mode,
        "unsupervised_family": result.unsupervised_family,
        "inference_method": result.inference_method,
        "preprocess_applied": result.preprocess_applied,
    }
    if label_encoder is not None:
        joblib.dump(label_encoder, out_dir / "label_encoder.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_predict_bundle(artifact_dir: Path) -> tuple[Pipeline, dict[str, Any], LabelEncoder | None]:
    pipeline = joblib.load(artifact_dir / "model.joblib")
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    le_path = artifact_dir / "label_encoder.joblib"
    label_encoder = joblib.load(le_path) if le_path.exists() else None
    return pipeline, meta, label_encoder


def infer_rows(
    artifact_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Any]:
    pipeline, meta, label_encoder = load_predict_bundle(artifact_dir)
    features = meta["feature_columns"]
    X = pd.DataFrame(rows)
    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    X = X[features]

    mode = meta.get("learning_mode", "supervised")
    method = meta.get("inference_method", "predict")

    if mode == "unsupervised":
        if method == "none":
            raise ValueError(
                "This unsupervised model was fit without a sklearn API for new rows "
                "(e.g. DBSCAN, agglomerative, spectral). Use an algorithm with "
                "predict or transform (KMeans, GMM, PCA, isolation forest, …)."
            )
        if method == "transform":
            arr = pipeline.transform(X)
            return np.asarray(arr).tolist()
        raw = pipeline.predict(X)
        return np.asarray(raw).tolist()

    raw = pipeline.predict(X)
    if meta["task"] == "classification" and label_encoder is not None:
        return label_encoder.inverse_transform(np.asarray(raw)).tolist()
    return np.asarray(raw).tolist()


def predict_rows(
    artifact_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Any]:
    return infer_rows(artifact_dir, rows)
