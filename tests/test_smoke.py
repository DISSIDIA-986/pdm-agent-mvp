"""Smoke tests: ensure every key entry point runs end-to-end without error.

These are intentionally fast (<5s total) and run on synthetic data only so
they pass in CI without internet access or the CWRU download.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

from pdm_agent.data import generate_synthetic, validate_samples
from pdm_agent.diagnostic import diagnose, threshold_baseline
from pdm_agent.workflow import build_workflow, run_until_decision
from pdm_agent.workorder import WorkOrderStore


def test_smoke_synthetic_pipeline(tmp_path) -> None:
    """Data → diagnose → workflow → work order persisted (full in-process flow)."""
    sample = generate_synthetic("inner_race", duration_s=1.0, snr_db=15.0, seed=999)
    validate_samples([sample])
    d = diagnose(sample)
    assert d.predicted_class == "inner_race"
    store = WorkOrderStore(tmp_path / "smoke.db")
    graph = build_workflow(store, checkpoint_db=str(tmp_path / "smoke-cp.db"))
    out = run_until_decision(graph, asset_id="A", sample=sample, thread_id="smoke-1")
    assert "interrupt" in out
    assert store.get(out["interrupt"]["work_order_id"]).status == "pending_approval"


def test_smoke_baseline_runs() -> None:
    sample = generate_synthetic("outer_race", snr_db=15.0, seed=998)
    d = threshold_baseline(sample)
    assert d.method == "rms-threshold-baseline"
    assert d.severity in ("normal", "alert")


def test_smoke_each_fault_class_has_predictable_severity() -> None:
    for fc, expected_alert in [
        ("normal", False),
        ("inner_race", True),
        ("outer_race", True),
    ]:
        s = generate_synthetic(fc, snr_db=15.0, seed=hash(fc) & 0xFFFF)  # type: ignore[arg-type]
        d = diagnose(s)
        if expected_alert:
            assert d.severity in ("watch", "alert", "critical"), (fc, d.severity)
        else:
            assert d.severity == "normal", (fc, d.severity)


def test_smoke_data_module_main_doesnt_crash_offline(tmp_path, monkeypatch) -> None:
    """If download fails, the module still produces synthetic samples without crashing."""
    from pdm_agent import data as data_mod
    # Point to an empty raw dir so no CWRU files exist
    samples = data_mod.load_cwru_dataset(tmp_path)
    assert samples == []
    # Fallback synthetic generation works
    fallback = [data_mod.generate_synthetic(fc, seed=i)  # type: ignore[arg-type]
                for i, fc in enumerate(["normal", "inner_race", "outer_race", "ball"])]
    stats = data_mod.validate_samples(fallback)
    assert stats["n_samples"] == 4
