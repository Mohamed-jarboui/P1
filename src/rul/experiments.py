"""The experiments behind every number in the README. Each returns a JSON-ready dict."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from rul.config import RUL_CAP, SENSOR_NAMES, SUBSETS
from rul.data import Subset, add_train_rul
from rul.features import build_features, feature_columns
from rul.maintenance import Costs, compare_policies
from rul.models import MODELS
from rul.monitoring import check, psi
from rul.pipeline import RULModel, fit_predict
from rul.scoring import interval_stats, rmse, summary

COST_RATIOS = (2.0, 5.0, 10.0, 20.0)
LEAD_TIMES = (5, 10, 20)


def _last_cycle(pred: np.ndarray, keys: pd.DataFrame) -> np.ndarray:
    idx = keys.reset_index(drop=True).groupby("unit", sort=True)["cycle"].idxmax().to_numpy()
    return pred[idx]


def benchmark(sub: Subset, main: RULModel, oof: pd.DataFrame) -> dict[str, object]:
    """Every model fitted on all training engines, scored at the last cycle of each test engine.

    `rmse` uses the true RUL capped at 125 like the labels (the usual C-MAPSS convention);
    `rmse_uncapped` uses the raw true RUL.
    """
    truth = np.minimum(sub.test_rul, RUL_CAP)
    labelled = add_train_rul(sub.train)
    out: dict[str, object] = {}
    for name in MODELS:
        if name == main.model_name:
            pred = main.predict_last(sub.test)["pred"].to_numpy()
        else:
            p, keys, _, _ = fit_predict(name, labelled, sub.test)
            pred = _last_cycle(p, keys)
        out[name] = {**summary(truth, pred), "rmse_uncapped": rmse(sub.test_rul, pred)}
    out["test_truth_above_cap"] = {
        "share": float(np.mean(sub.test_rul > RUL_CAP)),
        "max_true_rul": float(sub.test_rul.max()),
    }
    out["cv_rmse_all_cycles"] = {
        "model": main.model_name,
        "rmse": rmse(oof["rul"], oof["pred"]),
        "n_rows": int(len(oof)),
        "n_units": int(oof["unit"].nunique()),
    }
    return out


def conformal_check(sub: Subset, main: RULModel) -> dict[str, object]:
    truth = np.minimum(sub.test_rul, RUL_CAP)
    last = main.predict_last(sub.test)
    lo, hi, pred = last["lo"].to_numpy(), last["hi"].to_numpy(), last["pred"].to_numpy()
    edges = np.asarray(main.conformal.edges)
    bins = np.searchsorted(edges, pred, side="right")
    labels = [f"<{edges[0]:.0f}"]
    labels += [f"{a:.0f}-{b:.0f}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    labels += [f">={edges[-1]:.0f}"]
    per_bin = []
    for b, label in enumerate(labels):
        mask = bins == b
        if mask.any():
            per_bin.append(
                {
                    "bin": label,
                    **interval_stats(truth[mask], lo[mask], hi[mask]),
                    "n": int(mask.sum()),
                }
            )
    return {
        "target_coverage": 1 - main.conformal.alpha,
        "test": {**interval_stats(truth, lo, hi), "n": int(len(truth))},
        "per_bin": per_bin,
        "calibration": main.conformal.as_dict(),
    }


def simulate(oof: pd.DataFrame) -> dict[str, object]:
    base = compare_policies(oof, Costs(1.0, 5.0), lead_time=10)
    by_ratio = {}
    for ratio in COST_RATIOS:
        r = compare_policies(oof, Costs(1.0, ratio), lead_time=10)
        by_ratio[f"{ratio:g}"] = {k: round(v["vs_oracle"], 4) for k, v in r["policies"].items()}
    by_lead = {}
    for lead in LEAD_TIMES:
        r = compare_policies(oof, Costs(1.0, 5.0), lead_time=lead)
        by_lead[str(lead)] = {
            k: {"vs_oracle": round(v["vs_oracle"], 4), "failures": v["failures"]}
            for k, v in r["policies"].items()
        }
    return {"default": base, "vs_oracle_by_cost_ratio": by_ratio, "by_lead_time": by_lead}


def cross_subset(models: dict[str, RULModel], subsets: dict[str, Subset]) -> dict[str, object]:
    """Apply each subset's model to every test set: accuracy and how often the guards fire."""
    matrix: dict[str, dict[str, object]] = {}
    for train_name, model in models.items():
        reference = model.preprocessor.transform(subsets[train_name].train)
        row = {}
        for test_name, sub in subsets.items():
            truth = np.minimum(sub.test_rul, RUL_CAP)
            pred = model.predict_last(sub.test)["pred"].to_numpy()
            norm = model.preprocessor.transform(sub.test)
            flagged = [
                not check(g, model.guard, window=model.window).reliable
                for _, g in norm.groupby("unit", sort=True)
            ]
            psis = [psi(reference[s], norm[s]) for s in model.preprocessor.sensors]
            row[test_name] = {
                "rmse": rmse(truth, pred),
                "engines_flagged": float(np.mean(flagged)),
                "unknown_condition_rows": float(np.mean(norm["condition_distance"] > 1.0)),
                "max_sensor_psi": float(np.max(psis)),
                "n": int(len(truth)),
            }
        matrix[train_name] = row
    return {"train_on_rows_test_on_columns": matrix, "subsets": list(subsets)}


def importance(sub: Subset, model_name: str, seed: int = 0, repeats: int = 5) -> dict[str, object]:
    """Grouped permutation importance on 20% of the training engines (held out by unit).

    All features built from one sensor are shuffled together (same row permutation), so the
    score is per physical sensor, not per derived feature.
    """
    labelled = add_train_rul(sub.train)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(split.split(labelled, groups=labelled["unit"]))
    fit_rows, val_rows = labelled.iloc[fit_idx], labelled.iloc[val_idx]
    _, _, pre, model = fit_predict(model_name, fit_rows, fit_rows.head(1))
    kind = MODELS[model_name].features
    cols = feature_columns(pre.sensors, kind)
    feats = build_features(pre.transform(val_rows), pre.sensors, kind)
    y = (
        val_rows.set_index(["unit", "cycle"])
        .loc[list(zip(feats["unit"], feats["cycle"], strict=True)), "rul"]
        .to_numpy()
    )
    x = feats[cols].reset_index(drop=True)
    base = rmse(y, np.clip(model.predict(x), 0, RUL_CAP))
    rng = np.random.default_rng(seed)
    groups = {"cycle": ["cycle"]} | {
        s: [c for c in cols if c.startswith(f"{s}_")] for s in pre.sensors
    }
    scores = []
    for name, group in groups.items():
        increases = []
        for _ in range(repeats):
            shuffled = x.copy()
            perm = rng.permutation(len(x))
            shuffled[group] = x[group].to_numpy()[perm]
            increases.append(rmse(y, np.clip(model.predict(shuffled), 0, RUL_CAP)) - base)
        scores.append(
            {
                "group": name,
                "description": SENSOR_NAMES.get(name, "engine age in cycles"),
                "rmse_increase": float(np.mean(increases)),
                "std": float(np.std(increases)),
            }
        )
    scores.sort(key=lambda r: -r["rmse_increase"])
    return {
        "model": model_name,
        "baseline_rmse": base,
        "n_val_units": int(val_rows["unit"].nunique()),
        "n_val_rows": int(len(val_rows)),
        "repeats": repeats,
        "groups": scores,
    }


__all__ = ["SUBSETS", "benchmark", "conformal_check", "cross_subset", "importance", "simulate"]
