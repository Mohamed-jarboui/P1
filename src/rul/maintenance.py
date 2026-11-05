"""Turn RUL predictions into maintenance decisions and compare policies by cost.

Setting (assumptions, all configurable):

- every engine is checked after every cycle; a policy decides when to plan maintenance;
- planned maintenance happens `lead_time` cycles after the decision (parts, hangar slot);
- if the engine reaches the end of its life first, it fails: an unplanned removal;
- costs: `planned` per planned removal, `failure` per failure (only the ratio matters).

Policies are compared by cost per operating cycle, the long-run cost rate of a renewal process
(sum of costs / sum of cycles flown). Replacing early wastes life, which raises the rate;
failures raise it more. The oracle replaces every engine exactly at its last cycle.

Simulation uses the out-of-fold predictions on the training engines, because only those run to
failure (test engines are cut off before it). Thresholds are tuned on 4 folds of engines and
applied to the 5th, so no engine is scored by a threshold tuned on itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rul.config import RUL_CAP
from rul.conformal import MondrianConformal


@dataclass(frozen=True)
class Costs:
    planned: float = 1.0
    failure: float = 5.0


@dataclass(frozen=True)
class Engine:
    unit: int
    fold: int
    life: int  # last cycle, at which true RUL is 0
    cycles: np.ndarray
    signals: dict[str, np.ndarray]  # e.g. "pred" and "lo", one value per cycle


def engines_from_oof(oof: pd.DataFrame, signals: tuple[str, ...] = ("pred", "lo")) -> list[Engine]:
    out = []
    for unit, g in oof.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        cycles = g["cycle"].to_numpy()
        out.append(
            Engine(
                unit=int(unit),
                fold=int(g["fold"].iloc[0]),
                life=int(cycles[-1] + g["rul_true"].iloc[-1]),
                cycles=cycles,
                signals={s: g[s].to_numpy(float) for s in signals if s in g},
            )
        )
    return out


def first_trigger(cycles: np.ndarray, signal: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """For each threshold h, the first cycle with signal <= h (or a huge value if never)."""
    running_min = np.minimum.accumulate(signal)
    idx = np.searchsorted(-running_min, -np.asarray(thresholds, float), side="left")
    never = idx >= len(cycles)
    return np.where(never, np.iinfo(np.int64).max // 4, cycles[np.minimum(idx, len(cycles) - 1)])


def outcomes(maint_at: np.ndarray, life: np.ndarray, costs: Costs) -> dict[str, np.ndarray]:
    """Arrays with one value per engine (and per threshold, if 2-D)."""
    planned = maint_at <= life
    flown = np.where(planned, maint_at, life)
    return {
        "planned": planned,
        "flown": flown,
        "cost": np.where(planned, costs.planned, costs.failure),
        "unused": np.where(planned, life - maint_at, 0),
    }


def cost_rate(o: dict[str, np.ndarray], axis: int = 0) -> np.ndarray:
    return o["cost"].sum(axis=axis) / o["flown"].sum(axis=axis)


def _maint_matrix(
    engines: list[Engine], signal: str, thresholds: np.ndarray, lead_time: int
) -> np.ndarray:
    """(engines, thresholds) matrix of the cycle at which maintenance would happen."""
    return np.vstack(
        [first_trigger(e.cycles, e.signals[signal], thresholds) + lead_time for e in engines]
    )


THRESHOLDS = np.arange(0, RUL_CAP + 1, dtype=float)


def tune_threshold(
    engines: list[Engine], signal: str, costs: Costs, lead_time: int
) -> tuple[float, float]:
    life = np.array([e.life for e in engines])[:, None]
    rates = cost_rate(outcomes(_maint_matrix(engines, signal, THRESHOLDS, lead_time), life, costs))
    best = int(np.argmin(rates))
    return float(THRESHOLDS[best]), float(rates[best])


def tune_age(engines: list[Engine], costs: Costs) -> tuple[int, float]:
    life = np.array([e.life for e in engines])
    ages = np.arange(1, life.max() + 1)
    o = outcomes(np.broadcast_to(ages, (len(life), len(ages))), life[:, None], costs)
    rates = cost_rate(o)
    best = int(np.argmin(rates))
    return int(ages[best]), float(rates[best])


def _summary(o: dict[str, np.ndarray], costs: Costs, oracle_rate: float) -> dict[str, float]:
    rate = float(cost_rate(o))
    planned = o["planned"]
    return {
        "cost_rate": rate,
        "vs_oracle": rate / oracle_rate,
        "failures": int((~planned).sum()),
        "failure_rate": float((~planned).mean()),
        "mean_unused_life": float(o["unused"][planned].mean()) if planned.any() else 0.0,
        "n": int(len(planned)),
    }


def _recalibrate(oof: pd.DataFrame, fit_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Lower bounds for every row from a calibrator fitted only on rows in `fit_mask`."""
    cal = MondrianConformal(alpha=alpha).fit(oof.loc[fit_mask, "pred"], oof.loc[fit_mask, "rul"])
    return cal.interval(oof["pred"].to_numpy())[0]


def compare_policies(
    oof: pd.DataFrame, costs: Costs | None = None, lead_time: int = 10, alpha: float = 0.1
) -> dict[str, object]:
    """Cross-fitted comparison of reactive, fixed-age, predictive and oracle policies."""
    costs = costs or Costs()
    folds = sorted(oof["fold"].unique())
    rows: dict[str, list[dict[str, np.ndarray]]] = {
        "reactive": [],
        "fixed_age": [],
        "predictive_point": [],
        "predictive_lower_bound": [],
        "oracle": [],
    }
    chosen: dict[str, list[float]] = {"fixed_age": [], "point": [], "lower_bound": []}
    for k in folds:
        fit_mask = (oof["fold"] != k).to_numpy()
        frame = oof.assign(lo=_recalibrate(oof, fit_mask, alpha))
        engines = engines_from_oof(frame)
        tune = [e for e in engines if e.fold != k]
        held = [e for e in engines if e.fold == k]
        life = np.array([e.life for e in held])

        age, _ = tune_age(tune, costs)
        h_point, _ = tune_threshold(tune, "pred", costs, lead_time)
        h_lo, _ = tune_threshold(tune, "lo", costs, lead_time)
        chosen["fixed_age"].append(age)
        chosen["point"].append(h_point)
        chosen["lower_bound"].append(h_lo)

        rows["reactive"].append(outcomes(life + 1, life, costs))
        rows["oracle"].append(outcomes(life, life, costs))
        rows["fixed_age"].append(outcomes(np.full_like(life, age), life, costs))
        for name, signal, h in (
            ("predictive_point", "pred", h_point),
            ("predictive_lower_bound", "lo", h_lo),
        ):
            m = _maint_matrix(held, signal, np.array([h]), lead_time)[:, 0]
            rows[name].append(outcomes(m, life, costs))

    merged = {
        name: {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        for name, parts in rows.items()
    }
    oracle_rate = float(cost_rate(merged["oracle"]))
    return {
        "costs": {"planned": costs.planned, "failure": costs.failure},
        "lead_time": lead_time,
        "policies": {name: _summary(o, costs, oracle_rate) for name, o in merged.items()},
        "chosen_per_fold": chosen,
    }
