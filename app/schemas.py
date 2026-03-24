from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.algorithms import default_algorithm_id


class NumericBinSpecModel(BaseModel):
    column: str
    n_bins: int = Field(default=5, ge=2, le=50)
    strategy: Literal["uniform", "quantile"] = "quantile"


class PreprocessOptions(BaseModel):
    """User-controlled preprocessing embedded in the saved Pipeline."""

    feature_columns: list[str] | None = None
    scaling: Literal["auto", "none", "standard", "minmax", "robust"] = "auto"
    numeric_outliers: Literal["none", "clip_iqr"] = "none"
    imbalance: Literal["none", "smote"] = "none"
    numeric_bins: list[NumericBinSpecModel] = Field(default_factory=list)


class TrainRequest(BaseModel):
    dataset_id: str
    mode: Literal["supervised", "unsupervised"] = "supervised"
    algorithm: str = Field(default_factory=default_algorithm_id)
    """When set, train one job per id (supervised or unsupervised — compare models)."""
    algorithms: list[str] | None = None
    target_column: str | None = None
    unsupervised_family: Literal["clustering", "decomposition", "anomaly"] | None = (
        None
    )
    exclude_columns: list[str] = Field(default_factory=list)
    test_size: float = Field(default=0.2, ge=0.05, le=0.4)
    preprocessing: PreprocessOptions | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        if self.algorithms is not None and len(self.algorithms) > 12:
            raise ValueError("At most 12 algorithms per train request.")
        if self.mode == "supervised":
            if not self.target_column or not str(self.target_column).strip():
                raise ValueError("target_column is required for supervised mode.")
        else:
            if self.unsupervised_family is None:
                raise ValueError(
                    "unsupervised_family is required when mode is unsupervised."
                )
        return self


class PredictRequest(BaseModel):
    job_id: str
    rows: list[dict[str, Any]]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str | None = None
    metrics: dict[str, Any] | None = None
    task: str | None = None
    dataset_id: str | None = None
    target_column: str | None = None
    algorithm_id: str | None = None
    algorithm_label: str | None = None
    learning_mode: str | None = None
    unsupervised_family: str | None = None
    inference_method: str | None = None
    feature_columns: list[str] | None = None
    prediction_example_row: dict[str, Any] | None = None
    user_hint: str | None = None
    preprocess_applied: dict[str, Any] | None = None
