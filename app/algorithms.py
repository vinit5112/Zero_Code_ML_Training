"""
Sklearn supervised estimators exposed by id for zero-code training.

Each algorithm has a stable ``id``, human ``label``, which tasks it supports,
and whether numeric columns are standard-scaled after imputation (recommended
for linear models, SVM, k-NN, MLP; tree/naive Bayes typically omit scaling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sklearn.base import BaseEstimator

Task = Literal["classification", "regression"]
AlgoKind = Literal["classification", "regression", "both"]

EstimatorFactory = Callable[[], BaseEstimator]


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    label: str
    kind: AlgoKind
    scale_numeric: bool
    build_classification: EstimatorFactory | None
    build_regression: EstimatorFactory | None


def _rs() -> dict[str, Any]:
    return {"random_state": 42}


def _nj() -> dict[str, Any]:
    return {"n_jobs": -1, "random_state": 42}


def _specs() -> list[AlgorithmSpec]:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (
        ARDRegression,
        BayesianRidge,
        ElasticNet,
        GammaRegressor,
        HuberRegressor,
        Lars,
        Lasso,
        LassoLars,
        LinearRegression,
        LogisticRegression,
        OrthogonalMatchingPursuit,
        PassiveAggressiveClassifier,
        PassiveAggressiveRegressor,
        Perceptron,
        PoissonRegressor,
        QuantileRegressor,
        RANSACRegressor,
        Ridge,
        RidgeClassifier,
        SGDClassifier,
        SGDRegressor,
        TheilSenRegressor,
        TweedieRegressor,
    )
    from sklearn.naive_bayes import (
        BernoulliNB,
        CategoricalNB,
        ComplementNB,
        GaussianNB,
        MultinomialNB,
    )
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    )

    stump_c = lambda: DecisionTreeClassifier(max_depth=2, random_state=42)
    stump_r = lambda: DecisionTreeRegressor(max_depth=3, random_state=42)

    return [
        AlgorithmSpec(
            "logistic_regression",
            "Logistic regression",
            "classification",
            True,
            lambda: LogisticRegression(
                max_iter=2000,
                random_state=42,
                solver="lbfgs",
            ),
            None,
        ),
        AlgorithmSpec(
            "ridge_classifier",
            "Ridge classifier",
            "classification",
            True,
            lambda: RidgeClassifier(**_rs()),
            None,
        ),
        AlgorithmSpec(
            "sgd_classifier",
            "SGD classifier (log loss)",
            "classification",
            True,
            lambda: SGDClassifier(
                loss="log_loss",
                max_iter=8000,
                tol=1e-3,
                random_state=42,
                early_stopping=True,
            ),
            None,
        ),
        AlgorithmSpec(
            "passive_aggressive_classifier",
            "Passive aggressive classifier",
            "classification",
            True,
            lambda: PassiveAggressiveClassifier(random_state=42, max_iter=2000),
            None,
        ),
        AlgorithmSpec(
            "perceptron",
            "Perceptron",
            "classification",
            True,
            lambda: Perceptron(random_state=42, max_iter=2000),
            None,
        ),
        AlgorithmSpec(
            "linear_regression",
            "Linear / multiple regression (OLS)",
            "regression",
            True,
            None,
            lambda: LinearRegression(),
        ),
        AlgorithmSpec(
            "multiple_linear_regression",
            "Multiple linear regression (alias: OLS on all features)",
            "regression",
            True,
            None,
            lambda: LinearRegression(),
        ),
        AlgorithmSpec(
            "ridge",
            "Ridge regression",
            "regression",
            True,
            None,
            lambda: Ridge(**_rs()),
        ),
        AlgorithmSpec(
            "lasso",
            "Lasso",
            "regression",
            True,
            None,
            lambda: Lasso(random_state=42, max_iter=5000),
        ),
        AlgorithmSpec(
            "elastic_net",
            "Elastic net",
            "regression",
            True,
            None,
            lambda: ElasticNet(random_state=42, max_iter=5000),
        ),
        AlgorithmSpec(
            "lars",
            "LARS",
            "regression",
            True,
            None,
            lambda: Lars(**_rs()),
        ),
        AlgorithmSpec(
            "lasso_lars",
            "Lasso LARS",
            "regression",
            True,
            None,
            lambda: LassoLars(**_rs()),
        ),
        AlgorithmSpec(
            "bayesian_ridge",
            "Bayesian ridge",
            "regression",
            True,
            None,
            lambda: BayesianRidge(),
        ),
        AlgorithmSpec(
            "ard_regression",
            "ARD regression",
            "regression",
            True,
            None,
            lambda: ARDRegression(),
        ),
        AlgorithmSpec(
            "sgd_regressor",
            "SGD regressor",
            "regression",
            True,
            None,
            lambda: SGDRegressor(
                max_iter=8000,
                tol=1e-3,
                random_state=42,
                early_stopping=True,
            ),
        ),
        AlgorithmSpec(
            "passive_aggressive_regressor",
            "Passive aggressive regressor",
            "regression",
            True,
            None,
            lambda: PassiveAggressiveRegressor(random_state=42, max_iter=2000),
        ),
        AlgorithmSpec(
            "omp",
            "Orthogonal matching pursuit",
            "regression",
            True,
            None,
            lambda: OrthogonalMatchingPursuit(),
        ),
        AlgorithmSpec(
            "huber",
            "Huber regressor",
            "regression",
            True,
            None,
            lambda: HuberRegressor(max_iter=200),
        ),
        AlgorithmSpec(
            "ransac",
            "RANSAC regression",
            "regression",
            True,
            None,
            lambda: RANSACRegressor(
                estimator=LinearRegression(),
                random_state=42,
                max_trials=200,
            ),
        ),
        AlgorithmSpec(
            "theil_sen",
            "Theil–Sen regressor",
            "regression",
            True,
            None,
            lambda: TheilSenRegressor(random_state=42, max_iter=256),
        ),
        AlgorithmSpec(
            "poisson",
            "Poisson regressor",
            "regression",
            False,
            None,
            lambda: PoissonRegressor(max_iter=500),
        ),
        AlgorithmSpec(
            "gamma",
            "Gamma regressor",
            "regression",
            False,
            None,
            lambda: GammaRegressor(max_iter=500),
        ),
        AlgorithmSpec(
            "tweedie",
            "Tweedie regressor",
            "regression",
            False,
            None,
            lambda: TweedieRegressor(max_iter=500, power=1.5),
        ),
        AlgorithmSpec(
            "quantile",
            "Quantile regressor (median)",
            "regression",
            True,
            None,
            lambda: QuantileRegressor(quantile=0.5, alpha=1e-6, solver="highs"),
        ),
        AlgorithmSpec(
            "kernel_ridge",
            "Kernel ridge",
            "regression",
            True,
            None,
            lambda: KernelRidge(alpha=1.0, kernel="rbf"),
        ),
        AlgorithmSpec(
            "knn",
            "k-nearest neighbors",
            "both",
            True,
            lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
            lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
        ),
        AlgorithmSpec(
            "svc_rbf",
            "SVM (RBF)",
            "both",
            True,
            lambda: SVC(kernel="rbf", **_rs()),
            lambda: SVR(kernel="rbf"),
        ),
        AlgorithmSpec(
            "svc_linear",
            "Linear SVM",
            "both",
            True,
            lambda: LinearSVC(dual="auto", max_iter=5000, random_state=42),
            lambda: LinearSVR(max_iter=5000, random_state=42),
        ),
        AlgorithmSpec(
            "nu_svc",
            "Nu-SVM",
            "both",
            True,
            lambda: NuSVC(**_rs()),
            lambda: NuSVR(),
        ),
        AlgorithmSpec(
            "lda",
            "Linear discriminant analysis",
            "classification",
            True,
            lambda: LinearDiscriminantAnalysis(),
            None,
        ),
        AlgorithmSpec(
            "qda",
            "Quadratic discriminant analysis",
            "classification",
            True,
            lambda: QuadraticDiscriminantAnalysis(reg_param=0.0),
            None,
        ),
        AlgorithmSpec(
            "decision_tree",
            "Decision tree",
            "both",
            False,
            lambda: DecisionTreeClassifier(**_rs()),
            lambda: DecisionTreeRegressor(**_rs()),
        ),
        AlgorithmSpec(
            "extra_tree",
            "Extra-tree (single)",
            "both",
            False,
            lambda: ExtraTreeClassifier(**_rs()),
            lambda: ExtraTreeRegressor(**_rs()),
        ),
        AlgorithmSpec(
            "random_forest",
            "Random forest",
            "both",
            False,
            lambda: RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=1,
                **_nj(),
            ),
            lambda: RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=1,
                **_nj(),
            ),
        ),
        AlgorithmSpec(
            "extra_trees",
            "Extra trees",
            "both",
            False,
            lambda: ExtraTreesClassifier(n_estimators=200, **_nj()),
            lambda: ExtraTreesRegressor(n_estimators=200, **_nj()),
        ),
        AlgorithmSpec(
            "gradient_boosting",
            "Gradient boosting",
            "both",
            False,
            lambda: GradientBoostingClassifier(random_state=42),
            lambda: GradientBoostingRegressor(random_state=42),
        ),
        AlgorithmSpec(
            "hist_gradient_boosting",
            "Hist gradient boosting",
            "both",
            False,
            lambda: HistGradientBoostingClassifier(**_rs()),
            lambda: HistGradientBoostingRegressor(**_rs()),
        ),
        AlgorithmSpec(
            "adaboost",
            "AdaBoost",
            "both",
            False,
            lambda: AdaBoostClassifier(
                estimator=stump_c(),
                n_estimators=100,
                random_state=42,
            ),
            lambda: AdaBoostRegressor(
                estimator=stump_r(),
                n_estimators=100,
                random_state=42,
            ),
        ),
        AlgorithmSpec(
            "bagging",
            "Bagging",
            "both",
            False,
            lambda: BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
                n_estimators=30,
                n_jobs=-1,
                random_state=42,
            ),
            lambda: BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=None, random_state=42),
                n_estimators=30,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        AlgorithmSpec(
            "mlp",
            "Multi-layer perceptron",
            "both",
            True,
            lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
            lambda: MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
        ),
        AlgorithmSpec(
            "gaussian_nb",
            "Gaussian naive Bayes",
            "classification",
            False,
            lambda: GaussianNB(),
            None,
        ),
        AlgorithmSpec(
            "multinomial_nb",
            "Multinomial naive Bayes",
            "classification",
            False,
            lambda: MultinomialNB(),
            None,
        ),
        AlgorithmSpec(
            "bernoulli_nb",
            "Bernoulli naive Bayes",
            "classification",
            False,
            lambda: BernoulliNB(),
            None,
        ),
        AlgorithmSpec(
            "complement_nb",
            "Complement naive Bayes",
            "classification",
            False,
            lambda: ComplementNB(),
            None,
        ),
        AlgorithmSpec(
            "categorical_nb",
            "Categorical naive Bayes",
            "classification",
            False,
            lambda: CategoricalNB(),
            None,
        ),
    ]


_SPECS: list[AlgorithmSpec] | None = None
_BY_ID: dict[str, AlgorithmSpec] | None = None


def _load_registry() -> tuple[list[AlgorithmSpec], dict[str, AlgorithmSpec]]:
    global _SPECS, _BY_ID
    if _SPECS is None:
        _SPECS = _specs()
        _BY_ID = {s.id: s for s in _SPECS}
    return _SPECS, _BY_ID


def list_algorithms_public() -> list[dict[str, str]]:
    specs, _ = _load_registry()
    out: list[dict[str, str]] = []
    for s in specs:
        out.append(
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "scale_numeric": s.scale_numeric,
            }
        )
    return out


def resolve_estimator(algorithm_id: str, task: Task) -> tuple[BaseEstimator, bool]:
    _, by_id = _load_registry()
    spec = by_id.get(algorithm_id)
    if spec is None:
        raise ValueError(f"Unknown algorithm_id: {algorithm_id!r}")

    if task == "classification":
        if spec.kind == "regression":
            raise ValueError(
                f"Algorithm {algorithm_id!r} is for regression only; "
                "this dataset was inferred as classification."
            )
        if spec.build_classification is None:
            raise ValueError(f"Algorithm {algorithm_id!r} has no classification estimator.")
        return spec.build_classification(), spec.scale_numeric

    if spec.kind == "classification":
        raise ValueError(
            f"Algorithm {algorithm_id!r} is for classification only; "
            "this dataset was inferred as regression."
        )
    if spec.build_regression is None:
        raise ValueError(f"Algorithm {algorithm_id!r} has no regression estimator.")
    return spec.build_regression(), spec.scale_numeric


def default_algorithm_id() -> str:
    return "random_forest"


def assert_registered_algorithm(algorithm_id: str) -> None:
    _, by_id = _load_registry()
    if algorithm_id not in by_id:
        raise ValueError(f"Unknown algorithm_id: {algorithm_id!r}")


def algorithm_label_for(algorithm_id: str) -> str | None:
    _, by_id = _load_registry()
    spec = by_id.get(algorithm_id)
    return spec.label if spec else None
