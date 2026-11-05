"""Input guards and drift statistics.

A model trained on one fleet should not silently score data from another. Two cheap guards run
on every request:

- unknown operating condition: the settings are far from every condition seen in training;
- out-of-range sensors: normalised values outside the range seen in training.

`psi` (population stability index) compares a batch with the training distribution, for
fleet-level monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
PSI_MAJOR = 0.25


@dataclass
class RangeGuard:
    quantile: float = 0.001
    margin: float = 0.5  # extra slack, in standard deviations, on both sides
    low: dict[str, float] = field(default_factory=dict)
    high: dict[str, float] = field(default_factory=dict)

    def fit(self, norm: pd.DataFrame, sensors: list[str]) -> RangeGuard:
        for s in sensors:
            lo, hi = np.quantile(norm[s].to_numpy(float), [self.quantile, 1 - self.quantile])
            self.low[s], self.high[s] = float(lo - self.margin), float(hi + self.margin)
        return self

    def out_of_range(self, norm: pd.DataFrame) -> np.ndarray:
        """Boolean per row: at least one sensor outside its training range."""
        flags = np.zeros(len(norm), dtype=bool)
        for s, lo in self.low.items():
            v = norm[s].to_numpy(float)
            flags |= (v < lo) | (v > self.high[s])
        return flags


@dataclass(frozen=True)
class GuardResult:
    unknown_condition_rate: float
    out_of_range_rate: float
    max_condition_distance: float

    @property
    def reliable(self) -> bool:
        return self.unknown_condition_rate == 0.0 and self.out_of_range_rate <= 0.1

    def reasons(self) -> list[str]:
        out = []
        if self.unknown_condition_rate > 0:
            out.append(
                f"{self.unknown_condition_rate:.0%} of cycles are at an operating condition "
                "not seen in training"
            )
        if self.out_of_range_rate > 0.1:
            out.append(
                f"{self.out_of_range_rate:.0%} of cycles have sensors outside the training range"
            )
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "reliable": self.reliable,
            "unknown_condition_rate": round(self.unknown_condition_rate, 4),
            "out_of_range_rate": round(self.out_of_range_rate, 4),
            "max_condition_distance": round(self.max_condition_distance, 4),
            "reasons": self.reasons(),
        }


def check(norm: pd.DataFrame, guard: RangeGuard, window: int | None = None) -> GuardResult:
    """Guard one engine's history (optionally only its last `window` cycles)."""
    part = norm.sort_values("cycle").tail(window) if window else norm
    dist = part["condition_distance"].to_numpy(float)
    return GuardResult(
        unknown_condition_rate=float(np.mean(dist > 1.0)),
        out_of_range_rate=float(np.mean(guard.out_of_range(part))),
        max_condition_distance=float(dist.max()) if len(dist) else 0.0,
    )


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with quantile bins taken from the reference sample."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / max(len(cur), 1)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))
