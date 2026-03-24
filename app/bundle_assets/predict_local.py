#!/usr/bin/env python3
"""
Local inference for a Zero-code ML export (same folder as this file).

Unpack the zip so these files sit next to this script:
  - model.joblib
  - metadata.json
  - label_encoder.joblib   (only for some classification models)

Install dependencies:
  pip install -r requirements-predict.txt

Usage:
  python predict_local.py
      Run one demo row using prediction_example_row from metadata.json (if present).

  python predict_local.py rows.json
      rows.json = JSON array of objects, e.g. [{"feature_a": 1, "feature_b": "x"}, ...]
      Prints JSON array of predictions to stdout.

  python predict_local.py --csv input.csv --out output.csv
      Reads CSV with all required feature columns; writes CSV with a new column
      "prediction" (or "embedding" for unsupervised transform outputs, as JSON strings).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent


def _load_bundle() -> tuple[Any, dict[str, Any], Any]:
    model_path = BUNDLE_DIR / "model.joblib"
    meta_path = BUNDLE_DIR / "metadata.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "model.joblib and metadata.json must be in the same folder as this script."
        )
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    le_path = BUNDLE_DIR / "label_encoder.joblib"
    label_encoder = joblib.load(le_path) if le_path.is_file() else None
    return pipeline, meta, label_encoder


def predict_rows(rows: list[dict[str, Any]]) -> list[Any]:
    """Same rules as the hosted API: exact feature column names, correct types."""
    pipeline, meta, label_encoder = _load_bundle()
    features: list[str] = meta["feature_columns"]
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
                "This model cannot score new rows (e.g. DBSCAN / agglomerative). "
                "Train with predict or transform (KMeans, GMM, PCA, Isolation Forest, …)."
            )
        if method == "transform":
            arr = pipeline.transform(X)
            return np.asarray(arr).tolist()
        raw = pipeline.predict(X)
        return np.asarray(raw).tolist()

    raw = pipeline.predict(X)
    if meta.get("task") == "classification" and label_encoder is not None:
        return label_encoder.inverse_transform(np.asarray(raw)).tolist()
    return np.asarray(raw).tolist()


def _demo_row(meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    ex = meta.get("prediction_example_row")
    if isinstance(ex, dict) and ex:
        return [dict(ex)]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local predictions with exported model.")
    parser.add_argument(
        "json_path",
        nargs="?",
        help="Path to JSON file: array of row dicts with feature columns",
    )
    parser.add_argument("--csv", dest="csv_in", help="Input CSV path (feature columns)")
    parser.add_argument(
        "--out",
        dest="csv_out",
        help="Output CSV path (default: print predictions only for --csv)",
    )
    args = parser.parse_args()

    try:
        _, meta, _ = _load_bundle()
    except Exception as e:
        print(f"Error loading bundle: {e}", file=sys.stderr)
        return 1

    if args.csv_in:
        path = Path(args.csv_in)
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
        preds = predict_rows(rows)
        out_df = df.copy()
        col = "prediction"
        if meta.get("learning_mode") == "unsupervised" and meta.get("inference_method") == "transform":
            col = "embedding"
            out_df[col] = [json.dumps(p) if isinstance(p, list) else p for p in preds]
        else:
            out_df[col] = preds
        if args.csv_out:
            out_df.to_csv(args.csv_out, index=False)
            print(f"Wrote {args.csv_out}", file=sys.stderr)
        else:
            print(out_df.to_csv(index=False))
        return 0

    if args.json_path:
        raw = Path(args.json_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print("JSON must be an array of objects or one object.", file=sys.stderr)
            return 1
        preds = predict_rows(data)
        print(json.dumps(preds, indent=2))
        return 0

    demo = _demo_row(meta)
    if not demo:
        print(
            "No prediction_example_row in metadata.json. Pass rows.json or use --csv.",
            file=sys.stderr,
        )
        return 1
    preds = predict_rows(demo)
    print(json.dumps(preds, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
