"""Download, verify and parse the NASA C-MAPSS turbofan data set.

The files are fetched from GitHub mirrors pinned to a commit and checked against SHA-256 hashes.
Two independent mirrors were found to hold byte-identical files; the second is a fallback.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rul.config import COLUMNS, RUL_CAP, SUBSETS

MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-Transformer-for-RUL-Prediction/"
    "dcb137fe50277307a954f4e530e3f68bc7016070/CMAPSSData",
    "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/"
    "c6c35773b82ab10795c2d71425d95673c56b486b/C-MAPSS-Data",
)

SHA256: dict[str, str] = {
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
}


class ChecksumError(RuntimeError):
    """A downloaded file does not match its pinned SHA-256 hash."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (pinned https URLs)
        return resp.read()


def download(data_dir: Path, force: bool = False, mirrors: tuple[str, ...] = MIRRORS) -> list[str]:
    """Fetch every missing or corrupt file into `data_dir`. Returns the names downloaded."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for name, expected in SHA256.items():
        target = data_dir / name
        if not force and target.exists() and sha256_of(target) == expected:
            continue
        errors: list[str] = []
        for base in mirrors:
            try:
                payload = _fetch(f"{base}/{name}")
            except OSError as exc:  # URLError and timeouts are OSError subclasses
                errors.append(f"{base}: {exc}")
                continue
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                errors.append(f"{base}: sha256 {actual} != {expected}")
                continue
            # write to a temporary file first so an interrupted run never leaves a partial file
            with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as tmp:
                tmp.write(payload)
            os.replace(tmp.name, target)
            fetched.append(name)
            break
        else:
            raise ChecksumError(f"could not get a verified copy of {name}: " + "; ".join(errors))
    return fetched


def verify(data_dir: Path) -> dict[str, str]:
    """Return the hash of every file, raising if one is missing or does not match."""
    hashes = {}
    for name, expected in SHA256.items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `rul download`")
        actual = sha256_of(path)
        if actual != expected:
            raise ChecksumError(f"{path}: sha256 {actual} != {expected}")
        hashes[name] = actual
    return hashes


def read_split(path: Path) -> pd.DataFrame:
    """Parse one train_/test_ file: 26 whitespace-separated columns, one row per cycle."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=list(COLUMNS), engine="c")
    if df.shape[1] != len(COLUMNS) or df.isna().any().any():
        raise ValueError(f"{path}: expected {len(COLUMNS)} complete columns")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_rul(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).reshape(-1)


@dataclass(frozen=True)
class Subset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray  # true RUL at the last test cycle, one value per unit, in unit order


def load_subset(name: str, data_dir: Path) -> Subset:
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}; expected one of {SUBSETS}")
    train = read_split(data_dir / f"train_{name}.txt")
    test = read_split(data_dir / f"test_{name}.txt")
    test_rul = read_rul(data_dir / f"RUL_{name}.txt")
    if len(test_rul) != test["unit"].nunique():
        raise ValueError(f"{name}: {len(test_rul)} RUL values for {test['unit'].nunique()} units")
    return Subset(name, train, test, test_rul)


def add_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Training engines run to failure: true RUL at a cycle is (last cycle - cycle)."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_true"] = (last - out["cycle"]).astype(float)
    out["rul"] = out["rul_true"].clip(upper=cap)
    return out


def last_cycle_truth(test: pd.DataFrame, test_rul: np.ndarray) -> pd.DataFrame:
    """One row per test unit: its last observed cycle and the true RUL there."""
    last = test.groupby("unit", sort=True)["cycle"].max().reset_index()
    last["rul_true"] = test_rul
    return last
