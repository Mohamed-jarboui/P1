"""Split conformal intervals for RUL, with one set of quantiles per band of predicted RUL.

Errors are not the same everywhere: an engine predicted to be at the cap (healthy) and one
predicted to have 15 cycles left have very different error spreads. A Mondrian (binned) version
of split conformal learns the residual quantiles separately in each band of predicted RUL.

The finite-sample guarantee assumes calibration and test points are exchangeable. Calibration
residuals here come from out-of-fold predictions at every cycle of the training engines, which
are correlated within an engine; so the coverage is checked on the official test set, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rul.config import RUL_CAP


def conformal_quantile(values: np.ndarray, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest value: the split conformal correction."""
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0:
        raise ValueError("no calibration values")
    k = math.ceil((n + 1) * level)
    return float(v[min(max(k, 1), n) - 1])


@dataclass
class MondrianConformal:
    alpha: float = 0.1  # 1 - alpha = target coverage of the two-sided interval
    edges: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
    min_per_bin: int = 50
    cap: float = RUL_CAP
    q_lo: list[float] = field(default_factory=list)  # residual quantiles, residual = true - pred
    q_hi: list[float] = field(default_factory=list)
    n_per_bin: list[int] = field(default_factory=list)

    def _bins(self, pred: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges), np.asarray(pred, float), side="right")

    def fit(self, pred: np.ndarray, true: np.ndarray) -> MondrianConformal:
        pred, true = np.asarray(pred, float), np.asarray(true, float)
        resid = true - pred
        bins = self._bins(pred)
        lo_level, hi_level = self.alpha / 2, 1 - self.alpha / 2
        global_lo = -conformal_quantile(-resid, hi_level)
        global_hi = conformal_quantile(resid, hi_level)
        self.q_lo, self.q_hi, self.n_per_bin = [], [], []
        for b in range(len(self.edges) + 1):
            r = resid[bins == b]
            self.n_per_bin.append(int(len(r)))
            if len(r) < self.min_per_bin:  # too few points: fall back to the pooled quantiles
                self.q_lo.append(global_lo)
                self.q_hi.append(global_hi)
            else:
                # lower quantile at level alpha/2 = minus the upper (1 - alpha/2) quantile of -r
                self.q_lo.append(-conformal_quantile(-r, 1 - lo_level))
                self.q_hi.append(conformal_quantile(r, hi_level))
        # Where the model is biased in a band, both quantiles can sit on the same side of zero
        # and the interval would exclude the point prediction. Widening it to include the point
        # can only raise coverage, and keeps the reported numbers consistent with each other.
        self.q_lo = [min(q, 0.0) for q in self.q_lo]
        self.q_hi = [max(q, 0.0) for q in self.q_hi]
        return self

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.q_lo:
            raise RuntimeError("fit the calibrator first")
        pred = np.asarray(pred, float)
        b = self._bins(pred)
        lo = np.clip(pred + np.asarray(self.q_lo)[b], 0.0, self.cap)
        hi = np.clip(pred + np.asarray(self.q_hi)[b], 0.0, self.cap)
        return lo, np.maximum(hi, lo)

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "edges": list(self.edges),
            "q_lo": [round(q, 3) for q in self.q_lo],
            "q_hi": [round(q, 3) for q in self.q_hi],
            "n_per_bin": self.n_per_bin,
        }
