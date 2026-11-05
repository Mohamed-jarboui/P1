import math

import numpy as np
import pytest

from rul.conformal import MondrianConformal, conformal_quantile
from rul.scoring import interval_stats, nasa_score, rmse, summary


def test_nasa_score_punishes_late_predictions_more():
    assert nasa_score(np.array([50.0]), np.array([50.0])) == 0.0
    early = nasa_score(np.array([50.0]), np.array([37.0]))  # 13 cycles early
    late = nasa_score(np.array([50.0]), np.array([60.0]))  # 10 cycles late
    assert early == pytest.approx(math.e - 1)
    assert late == pytest.approx(math.e - 1)
    assert nasa_score(np.array([50.0]), np.array([60.0])) > nasa_score(
        np.array([50.0]), np.array([40.0])
    )


def test_rmse_and_summary():
    y, p = np.array([10.0, 20.0]), np.array([13.0, 16.0])
    assert rmse(y, p) == pytest.approx(math.sqrt((9 + 16) / 2))
    s = summary(y, p)
    assert s["n"] == 2 and s["late_share"] == 0.5 and s["mae"] == 3.5


def test_interval_stats():
    s = interval_stats(np.array([1.0, 5.0, 9.0]), np.array([0, 6, 0]), np.array([2, 8, 8]))
    assert s["coverage"] == pytest.approx(1 / 3)
    assert s["lower_bound_coverage"] == pytest.approx(2 / 3)


def test_conformal_quantile_uses_finite_sample_correction():
    values = np.arange(1, 10)  # n = 9
    assert conformal_quantile(values, 0.9) == 9  # ceil(10 * 0.9) = 9th value
    assert conformal_quantile(values, 0.5) == 5


def test_mondrian_coverage_on_exchangeable_heteroscedastic_data():
    rng = np.random.default_rng(0)

    def draw(n):
        pred = rng.uniform(0, 125, n)
        noise = rng.normal(0, 2 + 0.15 * pred)  # error grows with predicted RUL
        return pred, pred + noise

    cal = MondrianConformal(alpha=0.1, cap=1e9).fit(*draw(20000))
    pred, true = draw(20000)
    lo, hi = cal.interval(pred)
    stats = interval_stats(true, lo, hi)
    assert 0.88 <= stats["coverage"] <= 0.92
    widths = np.array(cal.q_hi) - np.array(cal.q_lo)
    assert np.all(np.diff(widths) > 0)  # wider bands where the noise is larger


def test_sparse_bins_fall_back_to_pooled_quantiles():
    pred = np.r_[np.full(500, 10.0), np.full(5, 110.0)]
    true = pred + np.random.default_rng(1).normal(0, 3, len(pred))
    cal = MondrianConformal(min_per_bin=50).fit(pred, true)
    assert cal.n_per_bin[-1] == 5
    assert cal.q_hi[-1] == cal.q_hi[1] == cal.q_hi[2]  # empty/sparse bins share the pooled value


def test_interval_is_clipped_to_valid_rul():
    cal = MondrianConformal().fit(np.full(100, 5.0), np.linspace(-20, 30, 100) + 5)
    lo, hi = cal.interval(np.array([1.0, 124.0]))
    assert lo.min() >= 0 and hi.max() <= 125 and np.all(hi >= lo)
