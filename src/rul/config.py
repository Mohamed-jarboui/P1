"""Constants shared by the whole package: dataset layout, sensor names, default paths."""

from __future__ import annotations

import os
from pathlib import Path

SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

SETTINGS: tuple[str, ...] = ("setting_1", "setting_2", "setting_3")
SENSORS: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 22))
COLUMNS: tuple[str, ...] = ("unit", "cycle", *SETTINGS, *SENSORS)

# Sensor descriptions from Saxena et al. (2008), "Damage propagation modeling for aircraft
# engine run-to-failure simulation", Table 2.
SENSOR_NAMES: dict[str, str] = {
    "s1": "T2 - total temperature at fan inlet",
    "s2": "T24 - total temperature at LPC outlet",
    "s3": "T30 - total temperature at HPC outlet",
    "s4": "T50 - total temperature at LPT outlet",
    "s5": "P2 - pressure at fan inlet",
    "s6": "P15 - total pressure in bypass duct",
    "s7": "P30 - total pressure at HPC outlet",
    "s8": "Nf - physical fan speed",
    "s9": "Nc - physical core speed",
    "s10": "epr - engine pressure ratio (P50/P2)",
    "s11": "Ps30 - static pressure at HPC outlet",
    "s12": "phi - ratio of fuel flow to Ps30",
    "s13": "NRf - corrected fan speed",
    "s14": "NRc - corrected core speed",
    "s15": "BPR - bypass ratio",
    "s16": "farB - burner fuel-air ratio",
    "s17": "htBleed - bleed enthalpy",
    "s18": "Nf_dmd - demanded fan speed",
    "s19": "PCNfR_dmd - demanded corrected fan speed",
    "s20": "W31 - HPT coolant bleed",
    "s21": "W32 - LPT coolant bleed",
}

SUBSET_INFO: dict[str, dict[str, object]] = {
    "FD001": {"conditions": 1, "fault_modes": "HPC degradation"},
    "FD002": {"conditions": 6, "fault_modes": "HPC degradation"},
    "FD003": {"conditions": 1, "fault_modes": "HPC and fan degradation"},
    "FD004": {"conditions": 6, "fault_modes": "HPC and fan degradation"},
}

# Labels are capped: early in life an engine looks healthy and the exact remaining life cannot
# be read from the sensors. 125 cycles is the usual choice in the C-MAPSS literature.
RUL_CAP: int = 125

# Number of past cycles summarised by the window features. 30 is the shortest test history
# in FD001 (31 cycles), so every test engine has a full window there.
WINDOW: int = 30


def data_dir() -> Path:
    return Path(os.environ.get("RUL_DATA_DIR", "data/raw"))


def models_dir() -> Path:
    return Path(os.environ.get("RUL_MODELS_DIR", "models"))


def results_dir() -> Path:
    return Path(os.environ.get("RUL_RESULTS_DIR", "results"))
