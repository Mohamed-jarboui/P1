import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rul.api import create_app
from rul.cli import build_parser
from rul.config import SENSORS, SETTINGS
from rul.pipeline import RULModel
from rul.report import update_readme

from .conftest import make_fleet


def test_out_of_fold_covers_every_cycle_once(trained, fleet):
    _, oof = trained
    assert len(oof) == len(fleet)
    assert oof.groupby("unit")["fold"].nunique().max() == 1  # an engine sits in one fold
    assert oof["pred"].between(0, 125).all()
    assert (oof["lo"] <= oof["pred"] + 1e-9).all() and (oof["hi"] >= oof["pred"] - 1e-9).all()


def test_model_learns_something_on_synthetic_engines(trained):
    _, oof = trained
    err_model = np.sqrt(np.mean((oof["pred"] - oof["rul"]) ** 2))
    err_mean = np.sqrt(np.mean((oof["rul"].mean() - oof["rul"]) ** 2))
    assert err_model < 0.5 * err_mean


def test_artifact_round_trip_and_actions(trained, tmp_path):
    model, _ = trained
    test, _ = make_fleet(n_units=5, seed=9, truncate=True)
    path = tmp_path / "FD001.joblib"
    model.save(path)
    loaded = RULModel.load(path)
    a, b = model.predict_last(test), loaded.predict_last(test)
    np.testing.assert_allclose(a["pred"], b["pred"])
    assert len(a) == 5

    guard = model.guard_check(test[test["unit"] == 1])
    assert guard.reliable
    assert model.decide(5.0, guard)[0] == "schedule_maintenance"
    assert model.decide(100.0, guard)[0] == "continue_operation"


def _payload(df, subset="FD001"):
    return {
        "subset": subset,
        "cycles": [
            {"settings": [float(r[s]) for s in SETTINGS], "sensors": [float(r[s]) for s in SENSORS]}
            for _, r in df.iterrows()
        ],
    }


@pytest.fixture()
def client(trained):
    with TestClient(create_app(models={"FD001": trained[0]})) as c:
        yield c


def test_api_health_and_model_card(client):
    assert client.get("/health").json() == {"status": "ok", "models": ["FD001"]}
    card = client.get("/models/FD001").json()
    assert card["model"] == "hgb_window" and card["policy"]["threshold"] == 20.0
    assert client.get("/models/FD003").status_code == 404


def test_api_predict(client):
    test, _ = make_fleet(n_units=1, seed=11, truncate=True)
    body = client.post("/predict", json=_payload(test)).json()
    assert body["cycles_seen"] == len(test)
    assert 0 <= body["interval"]["low"] <= body["rul"] <= body["interval"]["high"] <= 125
    assert body["action"] in {"continue_operation", "schedule_maintenance"}
    assert body["guards"]["reliable"] is True


def test_api_flags_inputs_from_an_unknown_operating_condition(client):
    test, _ = make_fleet(n_units=1, seed=12)
    test["setting_1"] = 30.0
    body = client.post("/predict", json=_payload(test)).json()
    assert body["action"] == "inspect_manually"
    assert body["guards"]["unknown_condition_rate"] == 1.0


def test_api_validates_input(client):
    test, _ = make_fleet(n_units=1, seed=13)
    payload = _payload(test.head(3))
    payload["cycles"][0]["sensors"] = payload["cycles"][0]["sensors"][:20]
    assert client.post("/predict", json=payload).status_code == 422
    assert client.post("/predict", json={"subset": "FD001", "cycles": []}).status_code == 422
    assert client.post("/predict", json=_payload(test.head(3), "FD002")).status_code == 404


def test_report_rewrites_only_marked_sections(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    row = {
        "rmse": 12.345,
        "rmse_uncapped": 13.0,
        "mae": 9.0,
        "nasa_score": 250.0,
        "late_share": 0.5,
        "n": 100,
    }
    metrics = {
        "subsets": {
            "FD001": {
                "mean": row,
                "ridge_current": row,
                "ridge_window": row,
                "hgb_window": row,
                "cv_rmse_all_cycles": {"model": "hgb_window", "rmse": 12.0},
                "test_truth_above_cap": {"share": 0.1, "max_true_rul": 145.0},
            }
        }
    }
    (results / "metrics.json").write_text(json.dumps(metrics))
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:main -->\nold\n<!-- /results:main -->\noutro\n")
    assert update_readme(readme, results)
    text = readme.read_text()
    assert "12.35" in text and "old" not in text and text.startswith("intro")
    assert not update_readme(readme, results)  # idempotent


def test_cli_parser():
    args = build_parser().parse_args(["train", "--subsets", "FD001", "FD003", "--folds", "3"])
    assert args.subsets == ["FD001", "FD003"] and args.folds == 3


def test_report_fills_empty_sections(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "importance.json").write_text(
        json.dumps({"subsets": {"FD001": {"groups": [{"group": "s11", "rmse_increase": 4.0}]}}})
    )
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:importance -->\n<!-- /results:importance -->\n")
    assert update_readme(readme, results)
    assert "s11 (+4.0)" in readme.read_text()
