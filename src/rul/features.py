"""Features at every cycle, computed only from that cycle and the ones before it.

`window` features summarise the last `WINDOW` cycles of each normalised sensor: the current
value, the window mean, the window standard deviation, the least-squares slope over the window,
and an exponentially weighted mean over the whole history. `current` features keep only the
current values (an ablation that shows what the history adds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul.config import WINDOW

FEATURE_SETS = ("current", "window")
EWM_SPAN = 10


def feature_columns(sensors: list[str], kind: str) -> list[str]:
    if kind == "current":
        return ["cycle", *(f"{s}_last" for s in sensors)]
    if kind == "window":
        stats = ("last", "mean", "std", "slope", "ewm")
        return ["cycle", *(f"{s}_{k}" for s in sensors for k in stats)]
    raise ValueError(f"unknown feature set {kind!r}; expected one of {FEATURE_SETS}")


def _per_unit(grouped_result: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Drop the group level that pandas adds to grouped rolling/ewm results and realign."""
    return grouped_result.reset_index(level=0, drop=True).reindex(index)


def build_features(
    norm: pd.DataFrame, sensors: list[str], kind: str = "window", window: int = WINDOW
) -> pd.DataFrame:
    """`norm` is the output of `Preprocessor.transform`. Returns unit, cycle and the features."""
    df = norm.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    x = df[sensors]
    cols: dict[str, pd.Series | np.ndarray] = {"unit": df["unit"], "cycle": df["cycle"]}
    if kind == "current":
        for s in sensors:
            cols[f"{s}_last"] = x[s]
        return pd.DataFrame(cols)[["unit", *feature_columns(sensors, kind)]]
    if kind != "window":
        raise ValueError(f"unknown feature set {kind!r}; expected one of {FEATURE_SETS}")

    t = df["cycle"].astype(float)
    helper = pd.DataFrame({"unit": df["unit"], "t": t, "tt": t * t})
    for s in sensors:
        helper[f"tx_{s}"] = t * x[s]
    stacked = pd.concat([helper, x], axis=1)
    g = stacked.groupby("unit", sort=False)
    value_cols = ["t", "tt", *(f"tx_{s}" for s in sensors), *sensors]
    mean = _per_unit(g[value_cols].rolling(window, min_periods=1).mean(), df.index)
    std = _per_unit(g[sensors].rolling(window, min_periods=1).std(ddof=0), df.index)
    ewm = _per_unit(g[sensors].ewm(span=EWM_SPAN, adjust=False).mean(), df.index)

    var_t = (mean["tt"] - mean["t"] ** 2).to_numpy()
    for s in sensors:
        cov = (mean[f"tx_{s}"] - mean["t"] * mean[s]).to_numpy()
        slope = np.divide(cov, var_t, out=np.zeros_like(cov), where=var_t > 1e-9)
        cols[f"{s}_last"] = x[s]
        cols[f"{s}_mean"] = mean[s]
        cols[f"{s}_std"] = std[s].fillna(0.0)
        cols[f"{s}_slope"] = slope
        cols[f"{s}_ewm"] = ewm[s]
    return pd.DataFrame(cols)[["unit", *feature_columns(sensors, kind)]]


def last_rows(features: pd.DataFrame) -> pd.DataFrame:
    """The row at the last observed cycle of each unit, in unit order."""
    idx = features.groupby("unit", sort=True)["cycle"].idxmax()
    return features.loc[idx.to_numpy()].reset_index(drop=True)
