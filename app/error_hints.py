"""Map raw exception messages to short, actionable user hints."""

from __future__ import annotations

import re
from typing import Callable

from fastapi import HTTPException

Matcher = tuple[Callable[[str], bool], str]


def _has(sub: str) -> Callable[[str], bool]:
    sub_l = sub.lower()

    def check(msg: str) -> bool:
        return sub_l in msg.lower()

    return check


def _rx(pattern: str) -> Callable[[str], bool]:
    cre = re.compile(pattern, re.IGNORECASE | re.DOTALL)

    def check(msg: str) -> bool:
        return bool(cre.search(msg))

    return check


# More specific patterns first.
_MATCHERS: list[Matcher] = [
    (
        _has("Missing columns:"),
        "Use the exact feature names from training (copy the list or example JSON). "
        "Include every listed column; spelling and case must match.",
    ),
    (
        _rx(
            r"This unsupervised model was fit without|sklearn API for new rows|DBSCAN|agglomerative|spectral"
        ),
        "This algorithm does not score new rows in the API. Train again with one that supports "
        "predict or transform (e.g. K-Means, Gaussian Mixture, PCA, Isolation Forest).",
    ),
    (
        _rx(r"Target column '[^']+' not in dataset"),
        "Pick a target that appears exactly in your CSV header (watch spaces and spelling).",
    ),
    (
        _has("Target column not in dataset"),
        "Choose a column that exists in this dataset.",
    ),
    (
        _rx(r"Target column .+ is not in the CSV"),
        "That column name is not in the file—match the header row exactly.",
    ),
    (
        _has("exclude_columns not in dataset"),
        "Only exclude names that exist in the CSV. Remove typos or columns from another file.",
    ),
    (
        _has("No feature columns left after exclusions"),
        "Shorten the exclude list so at least one column remains for features.",
    ),
    (
        _has("No usable columns after preprocessing split"),
        "Features may be all-empty or the wrong type—check the CSV has real values in those columns.",
    ),
    (
        _has("Need at least 5 rows for unsupervised"),
        "Add more data rows (unsupervised needs at least 5).",
    ),
    (
        _has("Need at least 10 rows after cleaning"),
        "Too many rows were dropped for missing targets—fix gaps in the target column or add more rows.",
    ),
    (
        _rx(r"for regression only.*inferred as classification"),
        "The app treated your target as classification. Pick a classification algorithm, or use fewer "
        "categories / bucket labels; if the target is really numeric, check column types in the CSV.",
    ),
    (
        _rx(r"for classification only.*inferred as regression"),
        "The app treated your target as regression. Pick a regression algorithm, or turn the target into clear categories.",
    ),
    (
        _rx(r"too many distinct classes for classification"),
        "Use regression if the target is numeric, or bucket labels into fewer groups (e.g. top categories plus an 'Other' bucket).",
    ),
    (
        lambda m: "too many unique" in m.lower() and "class" in m.lower(),
        "Too many categories makes training hard. Merge rare labels, bucket into groups, or use regression if the target is numeric.",
    ),
    (
        _has("least populated class"),
        "Each class needs enough examples (often 2+ per class for a test split). Merge rare labels or collect more data.",
    ),
    (
        _rx(r"test_size|stratify|Cannot stratify|minimum number of groups"),
        "Train/test split failed—try a larger dataset, fewer classes, or more rows per class.",
    ),
    (
        lambda m: "Unknown algorithm_id" in m or "Unknown unsupervised algorithm_id" in m,
        "Refresh algorithms and pick an option from the list (unknown id).",
    ),
    (
        _rx(r"Algorithm .+ is for '.+', not"),
        "Pick an algorithm that matches the analysis family (clustering vs decomposition vs anomaly).",
    ),
    (
        _has("Could not read CSV"),
        "Save as UTF-8 CSV with a standard delimiter; check for corrupt or binary content.",
    ),
    (
        _has("CSV has no data rows"),
        "Add at least one row of data under the header.",
    ),
    (
        _has("Could not profile dataset"),
        "Check for odd quoting, wrong delimiter, or merged cells—open the file in a spreadsheet app.",
    ),
    (
        _has("Could not parse CSV"),
        "The upload is not a valid CSV—try exporting again with comma separation and UTF-8.",
    ),
    (
        _has("Upload a .csv file"),
        "Choose a file whose name ends in .csv.",
    ),
    (
        lambda m: "Unknown dataset" in m or "CSV file missing on disk" in m,
        "Re-upload the file or pick a dataset from the library after refreshing.",
    ),
    (
        _has("Unknown job"),
        "That job id is not on this server—start training again or pick a job from the list.",
    ),
    (
        _has("Job must be completed"),
        "Wait until training finishes (or fix a failed run) before download or prediction.",
    ),
    (
        lambda m: "Artifacts missing" in m or "Model artifacts missing" in m,
        "The saved model files are missing—train again on this server.",
    ),
    (
        _has("target_column is required"),
        "Supervised training needs a target column—select which column to predict.",
    ),
    (
        _has("unsupervised_family is required"),
        "Pick clustering, decomposition, or anomaly before training.",
    ),
    (
        _rx(r"SMOTE|SMOTENC|imblearn|k_neighbors"),
        "Resampling failed—ensure each class has at least 2 rows in the training split, or turn off SMOTE.",
    ),
    (
        _rx(r"KBinsDiscretizer|bin edges|unique values"),
        "Binning failed—use fewer bins or drop rows with too few distinct values in that column.",
    ),
    (
        _has("imbalanced-learn"),
        "Install imbalanced-learn on the server: pip install imbalanced-learn",
    ),
]


def hint_for_error_text(message: str) -> str | None:
    if not message or not str(message).strip():
        return None
    msg = str(message)
    for pred, hint in _MATCHERS:
        try:
            if pred(msg):
                return hint
        except Exception:
            continue
    return None


def api_http_exception(status_code: int, message: str) -> HTTPException:
    """HTTP error with optional structured hint for the web UI."""
    hint = hint_for_error_text(message)
    if hint:
        return HTTPException(
            status_code=status_code,
            detail={"message": message, "hint": hint},
        )
    return HTTPException(status_code=status_code, detail=message)
