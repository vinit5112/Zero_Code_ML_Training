"""
Unsupervised scikit-learn estimators: clustering, decomposition, anomaly detection.

``inference`` describes how new rows are scored after training:
- ``predict``: ``pipeline.predict(X)`` (cluster label, inlier/outlier, etc.)
- ``transform``: ``pipeline.transform(X)`` (embeddings / reduced features)
- ``none``: no supported API for new rows (e.g. DBSCAN, agglomerative clustering)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sklearn.base import BaseEstimator

Family = Literal["clustering", "decomposition", "anomaly"]
Inference = Literal["predict", "transform", "none"]
EstimatorFactory = Callable[[], BaseEstimator]


@dataclass(frozen=True)
class UnsupervisedSpec:
    id: str
    label: str
    family: Family
    scale_numeric: bool
    inference: Inference
    build: EstimatorFactory


def _rs() -> dict[str, Any]:
    return {"random_state": 42}


def _nj() -> dict[str, Any]:
    return {"n_jobs": -1, "random_state": 42}


def _specs() -> list[UnsupervisedSpec]:
    from sklearn.cluster import (
        AffinityPropagation,
        AgglomerativeClustering,
        Birch,
        DBSCAN,
        FeatureAgglomeration,
        KMeans,
        MeanShift,
        MiniBatchKMeans,
        OPTICS,
        SpectralClustering,
    )
    from sklearn.covariance import EllipticEnvelope
    from sklearn.decomposition import (
        DictionaryLearning,
        FactorAnalysis,
        FastICA,
        IncrementalPCA,
        KernelPCA,
        LatentDirichletAllocation,
        NMF,
        PCA,
        SparsePCA,
        TruncatedSVD,
    )
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    from sklearn.svm import OneClassSVM

    return [
        # --- Clustering ---
        UnsupervisedSpec(
            "kmeans",
            "K-means",
            "clustering",
            True,
            "predict",
            lambda: KMeans(n_clusters=8, n_init="auto", **_rs()),
        ),
        UnsupervisedSpec(
            "mini_batch_kmeans",
            "Mini-batch K-means",
            "clustering",
            True,
            "predict",
            lambda: MiniBatchKMeans(n_clusters=8, n_init=3, batch_size=256, **_rs()),
        ),
        UnsupervisedSpec(
            "birch",
            "BIRCH",
            "clustering",
            True,
            "predict",
            lambda: Birch(n_clusters=8, threshold=0.5, branching_factor=50),
        ),
        UnsupervisedSpec(
            "gaussian_mixture",
            "Gaussian mixture",
            "clustering",
            True,
            "predict",
            lambda: GaussianMixture(n_components=8, **_rs()),
        ),
        UnsupervisedSpec(
            "bayesian_gaussian_mixture",
            "Bayesian Gaussian mixture",
            "clustering",
            True,
            "predict",
            lambda: BayesianGaussianMixture(
                n_components=10,
                max_iter=300,
                **_rs(),
            ),
        ),
        UnsupervisedSpec(
            "dbscan",
            "DBSCAN",
            "clustering",
            True,
            "none",
            lambda: DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
        ),
        UnsupervisedSpec(
            "optics",
            "OPTICS",
            "clustering",
            True,
            "none",
            lambda: OPTICS(min_samples=5, n_jobs=-1),
        ),
        UnsupervisedSpec(
            "agglomerative",
            "Agglomerative clustering",
            "clustering",
            True,
            "none",
            lambda: AgglomerativeClustering(n_clusters=8),
        ),
        UnsupervisedSpec(
            "spectral",
            "Spectral clustering",
            "clustering",
            True,
            "none",
            lambda: SpectralClustering(
                n_clusters=8,
                affinity="rbf",
                assign_labels="kmeans",
                **_rs(),
            ),
        ),
        UnsupervisedSpec(
            "mean_shift",
            "Mean shift",
            "clustering",
            True,
            "none",
            lambda: MeanShift(bandwidth=None, n_jobs=-1),
        ),
        UnsupervisedSpec(
            "affinity_propagation",
            "Affinity propagation",
            "clustering",
            True,
            "none",
            lambda: AffinityPropagation(random_state=42, max_iter=300),
        ),
        UnsupervisedSpec(
            "feature_agglomeration",
            "Feature agglomeration (embedding)",
            "clustering",
            True,
            "transform",
            lambda: FeatureAgglomeration(n_clusters=8),
        ),
        # --- Decomposition / reduction ---
        UnsupervisedSpec(
            "pca",
            "PCA",
            "decomposition",
            True,
            "transform",
            lambda: PCA(n_components=0.95, **_rs()),
        ),
        UnsupervisedSpec(
            "incremental_pca",
            "Incremental PCA",
            "decomposition",
            True,
            "transform",
            lambda: IncrementalPCA(n_components=10, batch_size=64),
        ),
        UnsupervisedSpec(
            "kernel_pca",
            "Kernel PCA",
            "decomposition",
            True,
            "transform",
            lambda: KernelPCA(
                n_components=10,
                kernel="rbf",
                **_rs(),
                n_jobs=-1,
            ),
        ),
        UnsupervisedSpec(
            "truncated_svd",
            "Truncated SVD (LSA)",
            "decomposition",
            False,
            "transform",
            lambda: TruncatedSVD(n_components=10, random_state=42),
        ),
        UnsupervisedSpec(
            "nmf",
            "NMF",
            "decomposition",
            False,
            "transform",
            lambda: NMF(n_components=10, max_iter=500, random_state=42, init="nndsvda"),
        ),
        UnsupervisedSpec(
            "fast_ica",
            "FastICA",
            "decomposition",
            True,
            "transform",
            lambda: FastICA(n_components=10, random_state=42, max_iter=400, whiten="unit-variance"),
        ),
        UnsupervisedSpec(
            "factor_analysis",
            "Factor analysis",
            "decomposition",
            True,
            "transform",
            lambda: FactorAnalysis(n_components=10, random_state=42, max_iter=500),
        ),
        UnsupervisedSpec(
            "sparse_pca",
            "Sparse PCA",
            "decomposition",
            True,
            "transform",
            lambda: SparsePCA(n_components=10, random_state=42, max_iter=500),
        ),
        UnsupervisedSpec(
            "dictionary_learning",
            "Dictionary learning",
            "decomposition",
            True,
            "transform",
            lambda: DictionaryLearning(
                n_components=15,
                random_state=42,
                max_iter=500,
                fit_algorithm="lars",
            ),
        ),
        UnsupervisedSpec(
            "lda_topic",
            "Latent Dirichlet allocation (topics)",
            "decomposition",
            False,
            "transform",
            lambda: LatentDirichletAllocation(
                n_components=10,
                random_state=42,
                max_iter=50,
                learning_method="batch",
            ),
        ),
        UnsupervisedSpec(
            "gaussian_random_projection",
            "Gaussian random projection",
            "decomposition",
            True,
            "transform",
            lambda: GaussianRandomProjection(n_components=10, **_rs()),
        ),
        UnsupervisedSpec(
            "sparse_random_projection",
            "Sparse random projection",
            "decomposition",
            False,
            "transform",
            lambda: SparseRandomProjection(n_components=10, **_rs()),
        ),
        # --- Anomaly detection ---
        UnsupervisedSpec(
            "isolation_forest",
            "Isolation forest",
            "anomaly",
            False,
            "predict",
            lambda: IsolationForest(n_estimators=200, contamination="auto", **_nj()),
        ),
        UnsupervisedSpec(
            "local_outlier_factor",
            "Local outlier factor (novelty)",
            "anomaly",
            True,
            "predict",
            lambda: LocalOutlierFactor(
                n_neighbors=20,
                novelty=True,
                contamination="auto",
                n_jobs=-1,
            ),
        ),
        UnsupervisedSpec(
            "one_class_svm",
            "One-class SVM",
            "anomaly",
            True,
            "predict",
            lambda: OneClassSVM(kernel="rbf", nu=0.05),
        ),
        UnsupervisedSpec(
            "elliptic_envelope",
            "Elliptic envelope",
            "anomaly",
            True,
            "predict",
            lambda: EllipticEnvelope(contamination=0.05, random_state=42),
        ),
    ]


_SPECS: list[UnsupervisedSpec] | None = None
_BY_ID: dict[str, UnsupervisedSpec] | None = None


def _load_registry() -> tuple[list[UnsupervisedSpec], dict[str, UnsupervisedSpec]]:
    global _SPECS, _BY_ID
    if _SPECS is None:
        _SPECS = _specs()
        _BY_ID = {s.id: s for s in _SPECS}
    return _SPECS, _BY_ID


def list_unsupervised_grouped() -> dict[str, list[dict[str, Any]]]:
    specs, _ = _load_registry()
    out: dict[str, list[dict[str, Any]]] = {
        "clustering": [],
        "decomposition": [],
        "anomaly": [],
    }
    for s in specs:
        out[s.family].append(
            {
                "id": s.id,
                "label": s.label,
                "scale_numeric": s.scale_numeric,
                "inference": s.inference,
            }
        )
    return out


def assert_unsupervised_algorithm(family: Family, algorithm_id: str) -> None:
    _, by_id = _load_registry()
    spec = by_id.get(algorithm_id)
    if spec is None:
        raise ValueError(f"Unknown unsupervised algorithm_id: {algorithm_id!r}")
    if spec.family != family:
        raise ValueError(
            f"Algorithm {algorithm_id!r} is for {spec.family!r}, not {family!r}."
        )


def resolve_unsupervised(algorithm_id: str, family: Family) -> tuple[BaseEstimator, bool, Inference]:
    assert_unsupervised_algorithm(family, algorithm_id)
    _, by_id = _load_registry()
    spec = by_id[algorithm_id]
    return spec.build(), spec.scale_numeric, spec.inference


def unsupervised_label_for(algorithm_id: str) -> str | None:
    _, by_id = _load_registry()
    s = by_id.get(algorithm_id)
    return s.label if s else None
