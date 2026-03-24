from __future__ import annotations

import io
import json
import os
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.algorithms import (
    algorithm_label_for,
    assert_registered_algorithm,
    list_algorithms_public,
)
from app.automl import predict_rows as run_predict
from app.automl import save_artifact, train_tabular, train_unsupervised_tabular
from app.schemas import JobStatus, PredictRequest, TrainRequest
from app.dataset_preview import build_preview as build_dataset_preview
from app.error_hints import api_http_exception, hint_for_error_text
from app.unsupervised_algorithms import (
    assert_unsupervised_algorithm,
    list_unsupervised_grouped,
    unsupervised_label_for,
)

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
BUNDLE_ASSETS_DIR = APP_DIR / "bundle_assets"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

_jobs: dict[str, dict] = {}
_datasets: dict[str, dict] = {}
_lock = threading.Lock()


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_dirs()
    yield


app = FastAPI(
    title="Zero-Code ML",
    description="Supervised and unsupervised sklearn pipelines: train, metrics, download, predict/transform.",
    version="0.3.0",
    lifespan=lifespan,
)

_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_supervised_train_job(
    job_id: str,
    dataset_id: str,
    target_column: str,
    test_size: float,
    algorithm: str,
    preprocess: dict | None,
) -> None:
    csv_path = UPLOADS_DIR / f"{dataset_id}.csv"
    out_dir = ARTIFACTS_DIR / job_id
    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["message"] = "Training model…"
        pipeline, result, label_encoder = train_tabular(
            csv_path,
            target_column,
            test_size=test_size,
            algorithm_id=algorithm,
            preprocess_opts=preprocess,
        )
        save_artifact(
            out_dir,
            pipeline,
            result,
            label_encoder,
            target_column=target_column,
            dataset_id=dataset_id,
        )
        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["message"] = None
            _jobs[job_id]["metrics"] = result.metrics
            _jobs[job_id]["task"] = result.task
            _jobs[job_id]["algorithm_id"] = result.algorithm_id
            _jobs[job_id]["algorithm_label"] = result.algorithm_label
            _jobs[job_id]["learning_mode"] = result.learning_mode
            _jobs[job_id]["unsupervised_family"] = result.unsupervised_family
            _jobs[job_id]["inference_method"] = result.inference_method
            _jobs[job_id]["feature_columns"] = result.feature_columns
            _jobs[job_id]["prediction_example_row"] = result.prediction_example_row
            _jobs[job_id]["user_hint"] = None
            _jobs[job_id]["preprocess_applied"] = result.preprocess_applied
    except Exception as e:
        msg = str(e)
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["message"] = msg
            _jobs[job_id]["user_hint"] = hint_for_error_text(msg)


def _run_unsupervised_train_job(
    job_id: str,
    dataset_id: str,
    algorithm: str,
    family: str,
    exclude_columns: list[str],
    preprocess: dict | None,
) -> None:
    csv_path = UPLOADS_DIR / f"{dataset_id}.csv"
    out_dir = ARTIFACTS_DIR / job_id
    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["message"] = "Fitting unsupervised model…"
        pipeline, result, _le = train_unsupervised_tabular(
            csv_path,
            algorithm_id=algorithm,
            family=family,  # type: ignore[arg-type]
            exclude_columns=exclude_columns,
            preprocess_opts=preprocess,
        )
        save_artifact(
            out_dir,
            pipeline,
            result,
            None,
            target_column=None,
            dataset_id=dataset_id,
        )
        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["message"] = None
            _jobs[job_id]["metrics"] = result.metrics
            _jobs[job_id]["task"] = result.task
            _jobs[job_id]["algorithm_id"] = result.algorithm_id
            _jobs[job_id]["algorithm_label"] = result.algorithm_label
            _jobs[job_id]["learning_mode"] = result.learning_mode
            _jobs[job_id]["unsupervised_family"] = result.unsupervised_family
            _jobs[job_id]["inference_method"] = result.inference_method
            _jobs[job_id]["feature_columns"] = result.feature_columns
            _jobs[job_id]["prediction_example_row"] = result.prediction_example_row
            _jobs[job_id]["user_hint"] = None
            _jobs[job_id]["preprocess_applied"] = result.preprocess_applied
    except Exception as e:
        msg = str(e)
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["message"] = msg
            _jobs[job_id]["user_hint"] = hint_for_error_text(msg)


@app.get("/")
async def root():
    return {
        "service": "zero-code-ml",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "algorithms": "/api/algorithms",
    }


@app.get("/api/algorithms")
async def get_algorithms():
    return {
        "supervised": list_algorithms_public(),
        "unsupervised": list_unsupervised_grouped(),
    }


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise api_http_exception(400, "Upload a .csv file.")
    dataset_id = _new_id()
    dest = UPLOADS_DIR / f"{dataset_id}.csv"
    content = await file.read()
    dest.write_bytes(content)
    try:
        import pandas as pd

        df = pd.read_csv(dest, nrows=5)
        columns = list(df.columns)
    except Exception:
        dest.unlink(missing_ok=True)
        raise api_http_exception(400, "Could not parse CSV.")
    row_hint = len(content.splitlines()) - 1
    with _lock:
        _datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "columns": columns,
            "approx_rows": max(row_hint, 0),
        }
    return {"dataset_id": dataset_id, "columns": columns}


@app.get("/api/datasets")
async def list_datasets():
    with _lock:
        return {"datasets": list(_datasets.values())}


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    with _lock:
        d = _datasets.get(dataset_id)
    if not d:
        raise api_http_exception(404, "Unknown dataset.")
    return d


@app.get("/api/datasets/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    sample_rows: int = 15,
    target_column: str | None = None,
):
    """Sample rows, dtypes, missing counts, and inferred classification vs regression for a target."""
    with _lock:
        if dataset_id not in _datasets:
            raise api_http_exception(404, "Unknown dataset.")
    path = UPLOADS_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise api_http_exception(404, "CSV file missing on disk.")
    try:
        return build_dataset_preview(
            path,
            sample_rows=sample_rows,
            target_column=target_column,
        )
    except ValueError as e:
        raise api_http_exception(400, str(e)) from e
    except Exception as e:
        raise api_http_exception(400, f"Could not profile dataset: {e}") from e


def _dedupe_preserve_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        t = str(x).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


@app.post("/api/jobs/train")
async def start_train(body: TrainRequest, background: BackgroundTasks):
    with _lock:
        if body.dataset_id not in _datasets:
            raise api_http_exception(404, "Unknown dataset.")
        cols = set(_datasets[body.dataset_id]["columns"])

    if body.mode == "supervised":
        assert body.target_column is not None
        if body.target_column not in cols:
            raise api_http_exception(400, "Target column not in dataset.")
        raw_algos = (
            _dedupe_preserve_order(list(body.algorithms))
            if body.algorithms
            else [body.algorithm]
        )
        if not raw_algos:
            raise api_http_exception(400, "Select at least one algorithm.")
        for aid in raw_algos:
            try:
                assert_registered_algorithm(aid)
            except ValueError as e:
                raise api_http_exception(400, str(e)) from e
    else:
        assert body.unsupervised_family is not None
        raw_algos = (
            _dedupe_preserve_order(list(body.algorithms))
            if body.algorithms
            else [body.algorithm]
        )
        if not raw_algos:
            raise api_http_exception(400, "Select at least one algorithm.")
        unknown_ex = [c for c in body.exclude_columns if c not in cols]
        if unknown_ex:
            raise api_http_exception(
                400,
                f"exclude_columns not in dataset: {unknown_ex}",
            )
        for aid in raw_algos:
            try:
                assert_unsupervised_algorithm(body.unsupervised_family, aid)
            except ValueError as e:
                raise api_http_exception(400, str(e)) from e

    preprocess_dump = (
        body.preprocessing.model_dump(mode="json") if body.preprocessing else None
    )
    if body.preprocessing and body.preprocessing.feature_columns:
        fc = body.preprocessing.feature_columns
        unk = [c for c in fc if c not in cols]
        if unk:
            raise api_http_exception(400, f"Unknown feature columns: {unk}")
        if body.mode == "supervised" and body.target_column in fc:
            raise api_http_exception(
                400,
                "feature_columns must not include the target column.",
            )

    job_ids: list[str] = []

    for aid in raw_algos:
        job_id = _new_id()
        job_ids.append(job_id)
        with _lock:
            if body.mode == "supervised":
                assert body.target_column is not None
                _jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "message": None,
                    "metrics": None,
                    "task": None,
                    "dataset_id": body.dataset_id,
                    "target_column": body.target_column,
                    "algorithm_id": aid,
                    "algorithm_label": algorithm_label_for(aid),
                    "learning_mode": "supervised",
                    "unsupervised_family": None,
                    "inference_method": None,
                }
            else:
                _jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "message": None,
                    "metrics": None,
                    "task": None,
                    "dataset_id": body.dataset_id,
                    "target_column": None,
                    "algorithm_id": aid,
                    "algorithm_label": unsupervised_label_for(aid),
                    "learning_mode": "unsupervised",
                    "unsupervised_family": body.unsupervised_family,
                    "inference_method": None,
                }

        if body.mode == "supervised":
            assert body.target_column is not None
            background.add_task(
                _run_supervised_train_job,
                job_id,
                body.dataset_id,
                body.target_column,
                body.test_size,
                aid,
                preprocess_dump,
            )
        else:
            background.add_task(
                _run_unsupervised_train_job,
                job_id,
                body.dataset_id,
                aid,
                body.unsupervised_family,
                body.exclude_columns,
                preprocess_dump,
            )

    first = job_ids[0]
    return {"job_id": first, "job_ids": job_ids}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
    if not j:
        raise api_http_exception(404, "Unknown job.")
    return JobStatus(
        job_id=j["job_id"],
        status=j["status"],
        message=j.get("message"),
        metrics=j.get("metrics"),
        task=j.get("task"),
        dataset_id=j.get("dataset_id"),
        target_column=j.get("target_column"),
        algorithm_id=j.get("algorithm_id"),
        algorithm_label=j.get("algorithm_label"),
        learning_mode=j.get("learning_mode"),
        unsupervised_family=j.get("unsupervised_family"),
        inference_method=j.get("inference_method"),
        feature_columns=j.get("feature_columns"),
        prediction_example_row=j.get("prediction_example_row"),
        user_hint=j.get("user_hint"),
        preprocess_applied=j.get("preprocess_applied"),
    )


@app.get("/api/jobs")
async def list_jobs():
    with _lock:
        return {"jobs": list(_jobs.values())}


@app.get("/api/jobs/{job_id}/download")
async def download_model_bundle(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
    if not j:
        raise api_http_exception(404, "Unknown job.")
    if j["status"] != "completed":
        raise api_http_exception(400, "Job must be completed to download the model.")
    artifact_dir = ARTIFACTS_DIR / job_id
    model_path = artifact_dir / "model.joblib"
    meta_path = artifact_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise api_http_exception(404, "Artifacts missing.")
    buf = io.BytesIO()
    predict_py = BUNDLE_ASSETS_DIR / "predict_local.py"
    req_txt = BUNDLE_ASSETS_DIR / "requirements-predict.txt"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "model.joblib")
        zf.write(meta_path, "metadata.json")
        le_path = artifact_dir / "label_encoder.joblib"
        if le_path.exists():
            zf.write(le_path, "label_encoder.joblib")
        if predict_py.is_file():
            zf.write(predict_py, "predict_local.py")
        if req_txt.is_file():
            zf.write(req_txt, "requirements-predict.txt")
    buf.seek(0)
    filename = f"zero-code-ml-model-{job_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/predict")
async def predict(body: PredictRequest):
    with _lock:
        j = _jobs.get(body.job_id)
    if not j:
        raise api_http_exception(404, "Unknown job.")
    if j["status"] != "completed":
        raise api_http_exception(400, "Job must be completed before prediction.")
    artifact_dir = ARTIFACTS_DIR / body.job_id
    if not artifact_dir.exists() or not (artifact_dir / "model.joblib").exists():
        raise api_http_exception(404, "Model artifacts missing.")
    try:
        preds = run_predict(artifact_dir, body.rows)
    except ValueError as e:
        raise api_http_exception(400, str(e)) from e
    except Exception as e:
        raise api_http_exception(500, str(e)) from e
    method = j.get("inference_method") or "predict"
    return {"predictions": preds, "inference_method": method}
