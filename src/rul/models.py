"""The regressors compared in the benchmark. All are scikit-learn estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: str  # "current" or "window", see rul.features
    make: Callable[..., RegressorMixin]
    description: str


def _ridge() -> RegressorMixin:
    return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))


def _hgb(max_iter: int = 400, seed: int = 0) -> RegressorMixin:
    # No early stopping: its internal validation split is by row, so rows of the same engine
    # would sit on both sides and the stopping point would be chosen on leaked data.
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "mean", "current", lambda **_: DummyRegressor(), "predicts the mean training label"
        ),
        ModelSpec(
            "ridge_current", "current", lambda **_: _ridge(), "ridge on current sensor values"
        ),
        ModelSpec("ridge_window", "window", lambda **_: _ridge(), "ridge on 30-cycle features"),
        ModelSpec(
            "hgb_window",
            "window",
            lambda max_iter=400, seed=0: _hgb(max_iter, seed),
            "gradient boosting on 30-cycle features",
        ),
    )
}

DEFAULT_MODEL = "hgb_window"


def make_model(name: str, **kwargs: object) -> RegressorMixin:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name].make(**kwargs)
