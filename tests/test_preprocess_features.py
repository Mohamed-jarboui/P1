import numpy as np
import pandas as pd

from rul.features import build_features, feature_columns, last_rows
from rul.preprocess import Preprocessor


def test_single_condition_drops_constant_and_quantised_sensors(fleet):
    pre = Preprocessor().fit(fleet)
    assert pre.n_conditions == 1
    assert set(pre.dropped) == {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
    assert len(pre.sensors) == 14


def test_six_conditions_are_found_and_normalised_per_condition(fleet6):
    pre = Preprocessor().fit(fleet6)
    assert pre.n_conditions == 6
    norm = pre.transform(fleet6)
    assert norm["condition_distance"].max() < 0.1
    stats = norm.groupby("condition")[pre.sensors].agg(["mean", "std"])
    np.testing.assert_allclose(stats.xs("mean", axis=1, level=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(stats.xs("std", axis=1, level=1), 1.0, atol=0.05)


def test_unknown_operating_condition_is_far_from_every_centroid(fleet):
    pre = Preprocessor().fit(fleet)
    odd = fleet.head(3).copy()
    odd["setting_1"] = 15.0
    assert (pre.transform(odd)["condition_distance"] > 1.0).all()


def test_features_never_look_ahead(fleet):
    """Features at cycle t must not change when the cycles after t are removed."""
    pre = Preprocessor().fit(fleet)
    full = build_features(pre.transform(fleet), pre.sensors, "window")
    cut = fleet[fleet["cycle"] <= 50]
    partial = build_features(pre.transform(cut), pre.sensors, "window")
    expected = full[full["cycle"] <= 50].reset_index(drop=True)
    pd.testing.assert_frame_equal(partial, expected)


def test_slope_of_a_linear_signal_is_its_coefficient():
    n = 40
    norm = pd.DataFrame({"unit": 1, "cycle": np.arange(1, n + 1), "a": 3.0 * np.arange(n) + 7})
    feats = build_features(norm, ["a"], "window", window=10)
    np.testing.assert_allclose(feats["a_slope"].iloc[1:], 3.0)
    assert feats["a_slope"].iloc[0] == 0.0  # one point: no slope
    np.testing.assert_allclose(feats["a_std"].iloc[0], 0.0)
    np.testing.assert_allclose(feats["a_mean"].iloc[-1], np.mean(3.0 * np.arange(30, 40) + 7))


def test_feature_sets_and_last_rows(fleet):
    pre = Preprocessor().fit(fleet)
    norm = pre.transform(fleet)
    current = build_features(norm, pre.sensors, "current")
    assert list(current.columns) == ["unit", *feature_columns(pre.sensors, "current")]
    window = build_features(norm, pre.sensors, "window")
    assert len(feature_columns(pre.sensors, "window")) == 1 + 5 * len(pre.sensors)
    last = last_rows(window)
    assert list(last["unit"]) == sorted(fleet["unit"].unique())
    np.testing.assert_array_equal(last["cycle"], fleet.groupby("unit")["cycle"].max().to_numpy())
