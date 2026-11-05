"""Error metrics, including the asymmetric score of the PHM 2008 data challenge."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum over engines of exp(-d/13) - 1 if early (d < 0), exp(d/10) - 1 if late (d >= 0).

    d = predicted - true RUL. Predicting too much remaining life (late) is punished harder,
    because it means the engine can fail before the planned maintenance.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def interval_stats(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, float)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "lower_bound_coverage": float(np.mean(y >= lo)),
        "mean_width": float(np.mean(np.asarray(hi) - np.asarray(lo))),
    }


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "late_share": float(np.mean(d > 0)),
        "n": int(len(d)),
    }
