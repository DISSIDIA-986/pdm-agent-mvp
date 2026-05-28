"""Sidecar HTTP integration tests using FastAPI TestClient."""
from __future__ import annotations

from fastapi.testclient import TestClient

from pdm_agent.data import generate_synthetic
from pdm_agent.sidecar import app


def _payload(fault: str, seed: int = 42, duration_s: float = 1.0) -> dict:
    s = generate_synthetic(fault, duration_s=duration_s, snr_db=15.0, seed=seed)  # type: ignore[arg-type]
    return {
        "sample_id": s.sample_id,
        "signal": s.signal.astype(float).tolist(),
        "sample_rate_hz": s.sample_rate_hz,
        "rpm": s.rpm,
        "fault_class": "normal",
    }


def test_health_returns_ok() -> None:
    with TestClient(app) as c:
        r = c.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_info_returns_honest_scope_note() -> None:
    with TestClient(app) as c:
        r = c.get("/v1/info")
        body = r.json()
        assert "honest_scope" in body
        assert "analog benchmark" in body["honest_scope"].lower()


def test_diagnose_returns_inner_race_for_synthetic() -> None:
    with TestClient(app) as c:
        r = c.post("/v1/diagnose", json=_payload("inner_race", seed=1))
        r.raise_for_status()
        body = r.json()
        assert body["diagnosis"]["predicted_class"] == "inner_race"
        assert body["latency_ms"] >= 0
        assert body["server_version"]


def test_diagnose_rejects_short_signal() -> None:
    payload = _payload("normal")
    payload["signal"] = payload["signal"][:100]
    with TestClient(app) as c:
        r = c.post("/v1/diagnose", json=payload)
        assert r.status_code == 422  # pydantic min_length violation


def test_diagnose_rejects_nan_signal_via_validator() -> None:
    """Verify the pydantic field_validator catches non-finite values.

    Note: Strict JSON cannot transport NaN/Inf anyway (RFC 7159 §6), so by the
    time a payload reaches our service it's already well-formed floats — the
    only realistic ingress for NaN is in-process construction. We test the
    validator directly to assert defense-in-depth at the model layer.
    """
    import pytest
    from pdm_agent.sidecar import VibrationPayload
    bad_signal = [0.0] * 2048
    bad_signal[10] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        VibrationPayload(
            sample_id="bad",
            signal=bad_signal,
            sample_rate_hz=12_000,
            rpm=1797,
        )


def test_batch_endpoint_handles_mixed() -> None:
    with TestClient(app) as c:
        body = {"samples": [_payload("normal", seed=2), _payload("outer_race", seed=3)]}
        r = c.post("/v1/diagnose/batch", json=body)
        r.raise_for_status()
        result = r.json()
        assert result["n"] == 2
        classes = [x["predicted_class"] for x in result["results"]]
        assert "outer_race" in classes


def test_baseline_endpoint() -> None:
    with TestClient(app) as c:
        r = c.post("/v1/baseline", json=_payload("inner_race", seed=4))
        r.raise_for_status()
        d = r.json()["diagnosis"]
        assert d["method"] == "rms-threshold-baseline"


def _acoustic_payload(label: str = "abnormal", seed: int = 1) -> dict:
    from pdm_agent.acoustic import generate_synthetic_acoustic
    s = generate_synthetic_acoustic(label, seed=seed, snr_db=12.0)  # type: ignore[arg-type]
    return {
        "sample_id": s.sample_id,
        "signal": s.signal.astype(float).tolist(),
        "sample_rate_hz": s.sample_rate_hz,
        "machine_id": s.machine_id,
        "label": "normal",  # we don't trust client claim
    }


def test_diagnose_acoustic_endpoint_flags_abnormal() -> None:
    with TestClient(app) as c:
        r = c.post("/v1/diagnose_acoustic", json=_acoustic_payload("abnormal", seed=7))
        r.raise_for_status()
        body = r.json()
        assert body["diagnosis"]["predicted_label"] == "abnormal"
        assert body["diagnosis"]["severity"] in ("alert", "critical")


def test_diagnose_acoustic_endpoint_passes_normal() -> None:
    with TestClient(app) as c:
        r = c.post("/v1/diagnose_acoustic", json=_acoustic_payload("normal", seed=8))
        r.raise_for_status()
        body = r.json()
        assert body["diagnosis"]["predicted_label"] == "normal"
        assert body["diagnosis"]["severity"] == "normal"


def test_info_reports_both_modalities() -> None:
    with TestClient(app) as c:
        r = c.get("/v1/info")
        body = r.json()
        assert body["vibration_method"] == "envelope-spectrum-v2-family"
        assert body["acoustic_method"] == "acoustic-zscore-baseline-v1"
        assert body["acoustic_labels"] == ["normal", "abnormal"]
