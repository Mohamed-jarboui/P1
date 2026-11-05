import numpy as np
import pandas as pd
import pytest

from rul.maintenance import Costs, compare_policies, cost_rate, first_trigger, outcomes
from rul.monitoring import GuardResult, RangeGuard, psi


def test_first_trigger_uses_the_first_crossing():
    cycles = np.arange(1, 6)
    signal = np.array([100.0, 50.0, 20.0, 30.0, 5.0])
    got = first_trigger(cycles, signal, np.array([25.0, 4.0, 100.0, 50.0]))
    assert got[0] == 3 and got[2] == 1 and got[3] == 2
    assert got[1] > 10**9  # never triggered


def test_outcomes_and_cost_rate():
    life = np.array([100, 200])
    o = outcomes(np.array([90, 250]), life, Costs(1.0, 10.0))
    assert list(o["planned"]) == [True, False]
    assert list(o["flown"]) == [90, 200]
    assert list(o["unused"]) == [10, 0]
    assert cost_rate(o) == pytest.approx((1 + 10) / 290)


def _oof(n_units=40, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 300))
        cycles = np.arange(1, life + 1)
        rul_true = (life - cycles).astype(float)
        rul = np.minimum(rul_true, 125)
        pred = np.clip(rul + rng.normal(0, noise, life), 0, 125)
        parts.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": cycles,
                    "fold": unit % 5,
                    "rul": rul,
                    "rul_true": rul_true,
                    "pred": pred,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_perfect_predictions_get_close_to_the_oracle():
    r = compare_policies(_oof(), Costs(1.0, 5.0), lead_time=10)
    p = r["policies"]
    assert p["reactive"]["vs_oracle"] == pytest.approx(5.0)
    assert p["oracle"]["failures"] == 0
    assert p["predictive_point"]["failures"] == 0
    assert p["predictive_point"]["vs_oracle"] < 1.01
    assert p["predictive_point"]["vs_oracle"] < p["fixed_age"]["vs_oracle"]
    assert len(r["chosen_per_fold"]["point"]) == 5


def test_noisy_predictions_cost_more_than_perfect_ones():
    clean = compare_policies(_oof(noise=0.0), Costs(1.0, 10.0))["policies"]
    noisy = compare_policies(_oof(noise=15.0), Costs(1.0, 10.0))["policies"]
    assert noisy["predictive_point"]["cost_rate"] > clean["predictive_point"]["cost_rate"]


def test_psi_is_small_for_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_range_guard_and_guard_result():
    norm = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 2000)})
    guard = RangeGuard().fit(norm, ["a"])
    flags = guard.out_of_range(pd.DataFrame({"a": [0.0, 50.0, -50.0]}))
    assert list(flags) == [False, True, True]
    assert GuardResult(0.0, 0.05, 0.1).reliable
    assert not GuardResult(0.2, 0.0, 3.0).reliable
    assert "operating condition" in GuardResult(0.2, 0.0, 3.0).reasons()[0]
