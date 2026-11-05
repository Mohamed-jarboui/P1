"""HTTP API: score an engine's sensor history and return a maintenance recommendation."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from rul import __version__
from rul.config import SENSORS, SETTINGS, SUBSETS, models_dir
from rul.pipeline import RULModel

MAX_CYCLES = 2000


class Cycle(BaseModel):
    settings: list[float] = Field(min_length=3, max_length=3, description="3 operating settings")
    sensors: list[float] = Field(min_length=21, max_length=21, description="21 sensors s1..s21")

    @field_validator("settings", "sensors")
    @classmethod
    def _finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values):
            raise ValueError("values must be finite numbers")
        return values


class PredictRequest(BaseModel):
    subset: str = Field(description="which fleet model to use: FD001..FD004")
    cycles: list[Cycle] = Field(
        min_length=1, max_length=MAX_CYCLES, description="sensor history, oldest cycle first"
    )


class Interval(BaseModel):
    low: float
    high: float
    level: float


class PredictResponse(BaseModel):
    subset: str
    cycles_seen: int
    rul: float = Field(description="predicted remaining useful life in cycles (capped at 125)")
    interval: Interval
    action: str = Field(description="continue_operation, schedule_maintenance, inspect_manually")
    reasons: list[str]
    guards: dict[str, object]
    model_version: str


def _frame(cycles: list[Cycle]) -> pd.DataFrame:
    rows = [
        {
            "unit": 1,
            "cycle": i + 1,
            **dict(zip(SETTINGS, c.settings, strict=True)),
            **dict(zip(SENSORS, c.sensors, strict=True)),
        }
        for i, c in enumerate(cycles)
    ]
    return pd.DataFrame(rows)


def load_models(directory: Path) -> dict[str, RULModel]:
    return {
        s: RULModel.load(directory / f"{s}.joblib")
        for s in SUBSETS
        if (directory / f"{s}.joblib").exists()
    }


def create_app(models: dict[str, RULModel] | None = None, directory: Path | None = None) -> FastAPI:
    state: dict[str, RULModel] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.update(models if models is not None else load_models(directory or models_dir()))
        yield
        state.clear()

    app = FastAPI(
        title="Turbofan RUL API",
        version=__version__,
        description="Remaining useful life with a conformal interval and a maintenance action.",
        lifespan=lifespan,
    )

    def _get(subset: str) -> RULModel:
        if subset not in state:
            raise HTTPException(404, f"no model for {subset!r}; loaded: {sorted(state)}")
        return state[subset]

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok" if state else "no_models", "models": sorted(state)}

    @app.get("/models/{subset}")
    def model_card(subset: str) -> dict[str, object]:
        m = _get(subset)
        return {
            "subset": m.subset,
            "model": m.model_name,
            "feature_set": m.feature_set,
            "window": m.window,
            "interval": m.conformal.as_dict(),
            "policy": m.policy.as_dict() if m.policy else None,
            "metadata": m.metadata,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        m = _get(req.subset)
        raw = _frame(req.cycles)
        last = m.predict_last(raw).iloc[0]
        guard = m.guard_check(raw)
        action, reasons = m.decide(float(last["lo"]), guard)
        return PredictResponse(
            subset=m.subset,
            cycles_seen=len(req.cycles),
            rul=round(float(last["pred"]), 2),
            interval=Interval(
                low=round(float(last["lo"]), 2),
                high=round(float(last["hi"]), 2),
                level=1 - m.conformal.alpha,
            ),
            action=action,
            reasons=reasons,
            guards=guard.as_dict(),
            model_version=str(m.metadata.get("version", __version__)),
        )

    return app


app = create_app()
