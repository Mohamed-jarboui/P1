"""Operating conditions, sensor selection and per-condition normalisation.

FD002 and FD004 mix six flight conditions (altitude, Mach number, throttle angle). A sensor's
raw value depends far more on the condition than on wear, so each sensor is z-scored with the
mean and standard deviation of its own condition. FD001 and FD003 have one condition, where
this reduces to ordinary standardisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rul.config import SENSORS, SETTINGS

# Rounding that maps the noisy setting values onto the discrete flight conditions.
_ROUNDING = {"setting_1": 0, "setting_2": 2, "setting_3": 0}
# How far a setting may be from a known condition before the row counts as an unknown condition.
# Within a known condition the largest deviation in the training data is below 0.01 / 0.002 / 0.
SETTING_TOLERANCE = np.array([1.0, 0.05, 1.0])


@dataclass
class Preprocessor:
    """Fitted on training rows only; applied unchanged to validation, test and API inputs."""

    min_unique: int = 5
    centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    sensors: list[str] = field(default_factory=list)
    mean: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (condition, sensor)
    std: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    dropped: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        settings = df[list(SETTINGS)].to_numpy(dtype=float)
        rounded = pd.DataFrame(
            {c: df[c].round(d) + 0.0 for c, d in _ROUNDING.items()}
        )  # + 0.0 turns -0.0 into 0.0
        keys = rounded.astype(str).agg("|".join, axis=1)
        counts = keys.value_counts()
        min_rows = max(10, int(0.005 * len(df)))
        kept = sorted(k for k, n in counts.items() if n >= min_rows)
        if not kept:
            raise ValueError("no operating condition with enough rows")
        self.centroids = np.vstack([settings[(keys == k).to_numpy()].mean(axis=0) for k in kept])

        cond, _ = self.assign(df)
        self.sensors, self.dropped = [], []
        for s in SENSORS:
            unique_per_cond = df.groupby(cond)[s].nunique()
            if unique_per_cond.min() < self.min_unique:
                self.dropped.append(s)
            else:
                self.sensors.append(s)
        values = df[self.sensors].to_numpy(dtype=float)
        k = len(self.centroids)
        self.mean = np.zeros((k, len(self.sensors)))
        self.std = np.ones((k, len(self.sensors)))
        for c in range(k):
            rows = values[cond == c]
            if len(rows):
                self.mean[c] = rows.mean(axis=0)
                sd = rows.std(axis=0)
                self.std[c] = np.where(sd > 1e-12, sd, 1.0)
        return self

    @property
    def n_conditions(self) -> int:
        return len(self.centroids)

    def assign(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Nearest known condition for every row, and the scaled distance to it (>1 = unknown)."""
        x = df[list(SETTINGS)].to_numpy(dtype=float)
        dist = np.abs(x[:, None, :] - self.centroids[None, :, :]) / SETTING_TOLERANCE
        dist = dist.max(axis=2)  # (rows, conditions)
        cond = dist.argmin(axis=1)
        return cond, dist[np.arange(len(x)), cond]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cond, dist = self.assign(df)
        values = df[self.sensors].to_numpy(dtype=float)
        z = (values - self.mean[cond]) / self.std[cond]
        out = pd.DataFrame(z, columns=self.sensors, index=df.index)
        out.insert(0, "condition_distance", dist)
        out.insert(0, "condition", cond)
        out.insert(0, "cycle", df["cycle"].to_numpy())
        out.insert(0, "unit", df["unit"].to_numpy())
        return out
