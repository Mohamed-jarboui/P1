"""Training: out-of-fold predictions by engine, conformal calibration, final fit, artifact."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import GroupKFold

from rul import __version__
from rul.config import RUL_CAP, SENSORS, SETTINGS, WINDOW
from rul.conformal import MondrianConformal
from rul.data import add_train_rul
from rul.features import build_features, feature_columns, last_rows
from rul.models import DEFAULT_MODEL, MODELS, make_model
from rul.monitoring import GuardResult, RangeGuard, check
from rul.preprocess import Preprocessor


def fit_predict(
    spec_name: str,
    train: pd.DataFrame,
    apply: pd.DataFrame,
    estimator: RegressorMixin | None = None,
    window: int = WINDOW,
) -> tuple[np.ndarray, pd.DataFrame, Preprocessor, RegressorMixin]:
    """Fit preprocessing + model on `train` (with labels), predict every cycle of `apply`."""
    spec = MODELS[spec_name]
    pre = Preprocessor().fit(train)
    cols = feature_columns(pre.sensors, spec.features)
    f_train = build_features(pre.transform(train), pre.sensors, spec.features, window)
    labels = train.set_index(["unit", "cycle"])["rul"]
    y = labels.loc[list(zip(f_train["unit"], f_train["cycle"], strict=True))].to_numpy()
    model = clone(estimator) if estimator is not None else make_model(spec_name)
    model.fit(f_train[cols], y)
    f_apply = build_features(pre.transform(apply), pre.sensors, spec.features, window)
    pred = np.clip(model.predict(f_apply[cols]), 0.0, RUL_CAP)
    return pred, f_apply[["unit", "cycle"]], pre, model


def out_of_fold(
    train: pd.DataFrame,
    spec_name: str = DEFAULT_MODEL,
    folds: int = 5,
    seed: int = 0,
    estimator: RegressorMixin | None = None,
) -> pd.DataFrame:
    """Predictions for every cycle of every training engine, from a model that never saw it.

    Folds are by engine (GroupKFold): splitting by row would put cycles of the same engine in
    both the fitting and the scoring data.
    """
    labelled = add_train_rul(train)
    units = labelled["unit"].to_numpy()
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    parts = []
    for k, (fit_idx, val_idx) in enumerate(splitter.split(labelled, groups=units)):
        fit_rows, val_rows = labelled.iloc[fit_idx], labelled.iloc[val_idx]
        pred, keys, _, _ = fit_predict(spec_name, fit_rows, val_rows, estimator)
        truth = val_rows.set_index(["unit", "cycle"]).loc[
            list(zip(keys["unit"], keys["cycle"], strict=True)), ["rul", "rul_true"]
        ]
        part = keys.copy()
        part["fold"] = k
        part["rul"] = truth["rul"].to_numpy()
        part["rul_true"] = truth["rul_true"].to_numpy()
        part["pred"] = pred
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["unit", "cycle"], ignore_index=True)


@dataclass
class Policy:
    """Schedule maintenance when the conformal lower bound drops to `threshold` cycles."""

    threshold: float
    lead_time: int
    cost_planned: float
    cost_failure: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "lead_time": self.lead_time,
            "cost_planned": self.cost_planned,
            "cost_failure": self.cost_failure,
        }


@dataclass
class RULModel:
    """Everything needed to score raw C-MAPSS cycles: the artifact saved per subset."""

    subset: str
    model_name: str
    feature_set: str
    window: int
    preprocessor: Preprocessor
    estimator: RegressorMixin
    conformal: MondrianConformal
    guard: RangeGuard
    policy: Policy | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return feature_columns(self.preprocessor.sensors, self.feature_set)

    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Point prediction and interval at every cycle of every unit in `raw`."""
        norm = self.preprocessor.transform(raw)
        feats = build_features(norm, self.preprocessor.sensors, self.feature_set, self.window)
        pred = np.clip(self.estimator.predict(feats[self.columns]), 0.0, RUL_CAP)
        lo, hi = self.conformal.interval(pred)
        out = feats[["unit", "cycle"]].copy()
        out["pred"], out["lo"], out["hi"] = pred, lo, hi
        return out

    def predict_last(self, raw: pd.DataFrame) -> pd.DataFrame:
        """One row per unit, at its last observed cycle (how the test set is scored)."""
        norm = self.preprocessor.transform(raw)
        feats = last_rows(
            build_features(norm, self.preprocessor.sensors, self.feature_set, self.window)
        )
        pred = np.clip(self.estimator.predict(feats[self.columns]), 0.0, RUL_CAP)
        lo, hi = self.conformal.interval(pred)
        out = feats[["unit", "cycle"]].copy()
        out["pred"], out["lo"], out["hi"] = pred, lo, hi
        return out

    def guard_check(self, raw: pd.DataFrame) -> GuardResult:
        return check(self.preprocessor.transform(raw), self.guard, window=self.window)

    def decide(self, lower_bound: float, guard: GuardResult) -> tuple[str, list[str]]:
        if not guard.reliable:
            return "inspect_manually", guard.reasons()
        if self.policy is not None and lower_bound <= self.policy.threshold:
            return "schedule_maintenance", [
                f"lower bound {lower_bound:.0f} cycles <= policy threshold "
                f"{self.policy.threshold:.0f} (lead time {self.policy.lead_time} cycles)"
            ]
        return "continue_operation", []

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> RULModel:
        model = joblib.load(path)
        if not isinstance(model, RULModel):
            raise TypeError(f"{path} does not hold a RULModel")
        return model


def train_model(
    subset: str,
    train: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    folds: int = 5,
    seed: int = 0,
    alpha: float = 0.1,
    estimator: RegressorMixin | None = None,
    oof: pd.DataFrame | None = None,
) -> tuple[RULModel, pd.DataFrame]:
    """Out-of-fold predictions -> conformal calibration -> final fit on all training engines."""
    started = time.perf_counter()
    if oof is None:
        oof = out_of_fold(train, model_name, folds, seed, estimator)
    conformal = MondrianConformal(alpha=alpha).fit(oof["pred"], oof["rul"])
    lo, hi = conformal.interval(oof["pred"].to_numpy())
    oof = oof.assign(lo=lo, hi=hi)

    labelled = add_train_rul(train)
    _, _, pre, fitted = fit_predict(model_name, labelled, labelled.head(1), estimator)
    guard = RangeGuard().fit(pre.transform(labelled), pre.sensors)
    model = RULModel(
        subset=subset,
        model_name=model_name,
        feature_set=MODELS[model_name].features,
        window=WINDOW,
        preprocessor=pre,
        estimator=fitted,
        conformal=conformal,
        guard=guard,
        metadata={
            "version": __version__,
            "trained_units": int(train["unit"].nunique()),
            "trained_rows": int(len(train)),
            "sensors_used": pre.sensors,
            "sensors_dropped": pre.dropped,
            "conditions": pre.n_conditions,
            "folds": folds,
            "seed": seed,
            "train_seconds": round(time.perf_counter() - started, 1),
            "inputs": {"settings": list(SETTINGS), "sensors": list(SENSORS)},
        },
    )
    return model, oof
