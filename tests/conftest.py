"""Synthetic engines in the C-MAPSS format, so tests never touch the network or the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul.config import COLUMNS, SENSORS
from rul.models import make_model

CONDITIONS = np.array(
    [
        [0.0, 0.0, 100.0],
        [10.0, 0.25, 100.0],
        [20.0, 0.7, 100.0],
        [25.0, 0.62, 60.0],
        [35.0, 0.84, 100.0],
        [42.0, 0.84, 100.0],
    ]
)
CONSTANT = {"s1", "s5", "s10", "s16", "s18", "s19"}


def make_fleet(
    n_units: int = 12,
    conditions: int = 1,
    seed: int = 0,
    life_range: tuple[int, int] = (90, 170),
    truncate: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Engines whose informative sensors drift exponentially towards failure.

    Returns the cycles and, per unit, the true RUL at its last cycle (0 unless truncated).
    """
    physics = np.random.default_rng(42)  # same sensor response in every fleet
    gain = physics.uniform(0.5, 1.5, len(SENSORS)) * physics.choice([-1, 1], len(SENSORS))
    rng = np.random.default_rng(seed)
    rows, rul = [], []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(*life_range))
        last = int(rng.integers(life // 3, life - 5)) if truncate else life
        rul.append(life - last)
        for t in range(1, last + 1):
            c = int(rng.integers(conditions))
            settings = CONDITIONS[c] + rng.normal(0, [0.003, 0.0003, 0.0])
            wear = np.exp(3.0 * t / life) - 1.0
            sensors = []
            for i, s in enumerate(SENSORS):
                base = 100.0 + 10.0 * i + 7.0 * c
                if s in CONSTANT:
                    sensors.append(base)
                elif s == "s6":
                    sensors.append(base + rng.choice([0.0, 0.01]))
                else:
                    sensors.append(base + gain[i] * wear + rng.normal(0, 0.8))
            rows.append([unit, t, *settings, *sensors])
    return pd.DataFrame(rows, columns=list(COLUMNS)), np.array(rul, dtype=float)


def write_cmapss(df: pd.DataFrame, path: Path) -> None:
    lines = [
        " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " "
        for row in df.itertuples(index=False)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def fleet() -> pd.DataFrame:
    return make_fleet(n_units=16, conditions=1, seed=1)[0]


@pytest.fixture(scope="session")
def fleet6() -> pd.DataFrame:
    return make_fleet(n_units=16, conditions=6, seed=2)[0]


@pytest.fixture(scope="session")
def small_hgb():
    return make_model("hgb_window", max_iter=40)


@pytest.fixture(scope="session")
def trained(fleet, small_hgb):
    from rul.pipeline import Policy, train_model

    model, oof = train_model("FD001", fleet, folds=3, estimator=small_hgb)
    model.policy = Policy(threshold=20.0, lead_time=5, cost_planned=1.0, cost_failure=5.0)
    return model, oof
