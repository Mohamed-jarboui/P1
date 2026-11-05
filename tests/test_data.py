import hashlib

import numpy as np
import pytest

from rul import data
from rul.data import ChecksumError, add_train_rul, last_cycle_truth, load_subset

from .conftest import make_fleet, write_cmapss


def test_load_subset_round_trip(tmp_path):
    train, _ = make_fleet(n_units=4, seed=3)
    test, rul = make_fleet(n_units=3, seed=4, truncate=True)
    write_cmapss(train, tmp_path / "train_FD001.txt")
    write_cmapss(test, tmp_path / "test_FD001.txt")
    (tmp_path / "RUL_FD001.txt").write_text("\n".join(f"{int(r)} " for r in rul) + "\n")

    sub = load_subset("FD001", tmp_path)
    assert sub.train.shape == train.shape
    assert list(sub.train.columns) == list(train.columns)
    assert sub.train["unit"].dtype.kind == "i"
    np.testing.assert_allclose(sub.test_rul, rul)
    truth = last_cycle_truth(sub.test, sub.test_rul)
    assert list(truth["unit"]) == [1, 2, 3]


def test_load_subset_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown subset"):
        load_subset("FD009", tmp_path)


def test_train_labels_count_down_to_zero_and_are_capped():
    train, _ = make_fleet(n_units=2, seed=5, life_range=(140, 150))
    labelled = add_train_rul(train, cap=125)
    for _, g in labelled.groupby("unit"):
        assert g["rul_true"].iloc[-1] == 0
        assert g["rul_true"].iloc[0] == len(g) - 1
        assert g["rul"].max() == 125
        np.testing.assert_array_equal(np.diff(g["rul_true"]), -1)


def test_download_falls_back_to_second_mirror_and_verifies(tmp_path, monkeypatch):
    good = b"1 1 0.0\n"
    monkeypatch.setattr(data, "SHA256", {"x.txt": hashlib.sha256(good).hexdigest()})
    calls = []

    def fake_fetch(url, timeout=60.0):
        calls.append(url)
        return b"corrupted" if url.startswith("https://a") else good

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    fetched = data.download(tmp_path, mirrors=("https://a", "https://b"))
    assert fetched == ["x.txt"]
    assert (tmp_path / "x.txt").read_bytes() == good
    assert len(calls) == 2

    calls.clear()
    assert data.download(tmp_path, mirrors=("https://a", "https://b")) == []
    assert calls == []  # verified file on disk is not downloaded again


def test_download_raises_when_no_mirror_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SHA256", {"x.txt": "0" * 64})
    monkeypatch.setattr(data, "_fetch", lambda url, timeout=60.0: b"whatever")
    with pytest.raises(ChecksumError, match="x.txt"):
        data.download(tmp_path, mirrors=("https://a",))
    assert not (tmp_path / "x.txt").exists()
