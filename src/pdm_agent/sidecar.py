"""PdM diagnostic sidecar — FastAPI HTTP service exposing diagnose() over JSON.

Why a sidecar? See README "Architecture > Sidecar rationale". TL;DR: keeps the
PdM brain language-agnostic and process-isolated so the host EMS-demo (or any
other client) can call it via plain HTTP. We measure HTTP overhead in
scripts/measure_latency.py — for our scope (window-level diagnostics, ~1s
windows), 20-100ms HTTP overhead is acceptable.

Endpoints:
  POST /v1/diagnose      — accept window payload, return Diagnosis JSON
  GET  /v1/health        — liveness probe
  GET  /v1/info          — model + threshold metadata
  POST /v1/diagnose/batch — batch variant for evaluation
"""
from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .data import VibrationSample
from .diagnostic import diagnose, threshold_baseline

log = logging.getLogger(__name__)

app = FastAPI(
    title="PdM Agent Sidecar",
    version=__version__,
    summary="Vibration diagnostic sidecar (HTTP wrapper over diagnose()).",
)


class VibrationPayload(BaseModel):
    sample_id: str = Field(..., min_length=1, max_length=128)
    signal: list[float] = Field(..., min_length=1024)
    sample_rate_hz: int = Field(..., gt=0, le=1_000_000)
    rpm: float = Field(..., gt=0, le=100_000)
    fault_class: Literal["normal", "inner_race", "ball", "outer_race"] = "normal"
    rpm_tolerance_hz: float | None = None  # currently informational

    @field_validator("signal")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("signal contains non-finite values")
        return v

    def to_sample(self) -> VibrationSample:
        return VibrationSample(
            sample_id=self.sample_id,
            fault_class=self.fault_class,
            signal=np.asarray(self.signal, dtype=np.float32),
            sample_rate_hz=self.sample_rate_hz,
            rpm=self.rpm,
            source="synthetic",  # client could be either; we don't trust this
        )


class DiagnoseResponse(BaseModel):
    diagnosis: dict
    latency_ms: float
    server_version: str


class BatchPayload(BaseModel):
    samples: list[VibrationPayload]


@app.get("/v1/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/info")
async def info() -> dict:
    return {
        "version": __version__,
        "method": "envelope-spectrum-v2-family",
        "supports": ["inner_race", "outer_race", "ball", "normal"],
        "severity_buckets": ["normal", "watch", "alert", "critical"],
        "input_window_min_samples": 1024,
        "confidence_semantics": (
            "Reported 'confidence' is a softmax over deterministic family scores. "
            "It IS NOT a calibrated posterior — a misclassification can still "
            "report high confidence (see eval/error_analysis.md). Treat as a "
            "ranking signal between competing fault classes, not a probability."
        ),
        "honest_scope": (
            "CWRU-tuned thresholds; analog benchmark for BESS auxiliary "
            "equipment (pump/fan) bearings — NOT validated for direct BESS PdM. "
            "See repository README License & Scope sections."
        ),
    }


@app.post("/v1/diagnose", response_model=DiagnoseResponse)
async def diagnose_endpoint(payload: VibrationPayload) -> DiagnoseResponse:
    t0 = time.perf_counter()
    try:
        sample = payload.to_sample()
        d = diagnose(sample)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    latency_ms = (time.perf_counter() - t0) * 1000
    return DiagnoseResponse(
        diagnosis=d.to_dict(),
        latency_ms=round(latency_ms, 3),
        server_version=__version__,
    )


@app.post("/v1/baseline")
async def baseline_endpoint(payload: VibrationPayload) -> DiagnoseResponse:
    """RMS threshold baseline — for comparison in evaluation."""
    t0 = time.perf_counter()
    try:
        d = threshold_baseline(payload.to_sample())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return DiagnoseResponse(
        diagnosis=d.to_dict(),
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        server_version=__version__,
    )


@app.post("/v1/diagnose/batch")
async def diagnose_batch(payload: BatchPayload) -> dict:
    t0 = time.perf_counter()
    results: list[dict] = []
    for item in payload.samples:
        try:
            d = diagnose(item.to_sample())
            results.append(d.to_dict())
        except ValueError as e:
            results.append({"sample_id": item.sample_id, "error": str(e)})
    return {
        "n": len(results),
        "results": results,
        "total_latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }
