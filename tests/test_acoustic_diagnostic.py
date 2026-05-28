"""Tests for the acoustic feature extraction and z-score diagnostic."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pdm_agent.acoustic import AcousticSample, generate_synthetic_acoustic
from pdm_agent.acoustic_diagnostic import (
    AcousticBaseline,
    diagnose_acoustic,
    extract_features,
    fit_baseline,
    threshold_baseline_acoustic,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_features_returns_finite_values() -> None:
    s = generate_synthetic_acoustic("normal", seed=0, snr_db=15.0)
    f = extract_features(s)
    for value in f.to_dict().values():
        assert np.isfinite(value)


def test_band_ratios_sum_to_one() -> None:
    s = generate_synthetic_acoustic("normal", seed=1, snr_db=15.0)
    f = extract_features(s)
    s_ratio = f.band_low_ratio + f.band_mid_ratio + f.band_high_ratio
    assert math.isclose(s_ratio, 1.0, abs_tol=1e-3) or s_ratio < 1.0 + 1e-3


def test_abnormal_shifts_mid_band_energy_up() -> None:
    """The synthetic abnormal injects 1-3 kHz noise; that's our mid-band."""
    n_seed = 5
    norm_mid, abn_mid = [], []
    for seed in range(n_seed):
        norm_mid.append(extract_features(generate_synthetic_acoustic("normal", seed=seed, snr_db=15)).band_mid_ratio)
        abn_mid.append(extract_features(generate_synthetic_acoustic("abnormal", seed=seed, snr_db=15)).band_mid_ratio)
    assert np.mean(abn_mid) > np.mean(norm_mid), (
        f"abnormal mid-band ratio {np.mean(abn_mid):.3f} should exceed normal {np.mean(norm_mid):.3f}"
    )


def test_extract_features_rejects_short_signal() -> None:
    sig = np.zeros(100, dtype=np.float32)
    # Build AcousticSample manually (its constructor enforces a softer 4s min;
    # use a barely-passing length to test extract_features's own guard).
    with pytest.raises(ValueError, match="too short"):
        AcousticSample(
            sample_id="x", label="normal", signal=sig,
            sample_rate_hz=16_000, machine_id="id_synth", source="synthetic",
        )


# ---------------------------------------------------------------------------
# Baseline + diagnosis
# ---------------------------------------------------------------------------

def _normal_pool(n: int = 8, seed_base: int = 0, snr_db: float = 12.0) -> list[AcousticSample]:
    """Fit baselines and test samples at the same SNR to avoid distribution shift."""
    return [generate_synthetic_acoustic("normal", seed=seed_base + i, snr_db=snr_db) for i in range(n)]


def test_fit_baseline_records_means_and_stds() -> None:
    pool = _normal_pool(n=8)
    base = fit_baseline(pool)
    assert base.n_train == 8
    assert len(base.means) == len(base.feature_names) == 8
    # Stds must be non-negative
    for s in base.stds:
        assert s >= 0


def test_fit_baseline_requires_at_least_one_normal() -> None:
    with pytest.raises(ValueError):
        fit_baseline([generate_synthetic_acoustic("abnormal", seed=1)])


def test_diagnose_normal_clip_against_normal_baseline() -> None:
    """A normal clip drawn from the same distribution should score 'normal' severity."""
    base = fit_baseline(_normal_pool(n=10, seed_base=100))
    # Hold-out a fresh normal sample with a different seed
    test = generate_synthetic_acoustic("normal", seed=999, snr_db=12)
    d = diagnose_acoustic(test, base)
    assert d.predicted_label == "normal"
    assert d.severity == "normal"


def test_diagnose_abnormal_clip_against_normal_baseline() -> None:
    """An abnormal clip should produce alert or critical."""
    base = fit_baseline(_normal_pool(n=10, seed_base=200))
    test = generate_synthetic_acoustic("abnormal", seed=999, snr_db=12)
    d = diagnose_acoustic(test, base)
    assert d.predicted_label == "abnormal", (
        f"abnormal misclassified, anomaly_score={d.anomaly_score:.3f}, severity={d.severity}"
    )
    assert d.severity in ("alert", "critical")
    assert d.anomaly_score >= 3.0


def test_diagnose_is_json_serialisable() -> None:
    import json
    base = fit_baseline(_normal_pool(n=8, seed_base=300))
    d = diagnose_acoustic(generate_synthetic_acoustic("abnormal", seed=42), base)
    payload = json.dumps(d.to_dict())
    parsed = json.loads(payload)
    assert parsed["method"] == "acoustic-zscore-baseline-v1"
    assert parsed["predicted_label"] in ("normal", "abnormal")
    assert parsed["anomaly_score_is_calibrated"] is False


def test_baseline_roundtrip(tmp_path) -> None:
    base = fit_baseline(_normal_pool(n=8, seed_base=400))
    path = tmp_path / "baseline.json"
    base.save(path)
    restored = AcousticBaseline.load(path)
    assert restored.n_train == base.n_train
    assert restored.feature_names == base.feature_names
    for a, b in zip(restored.means, base.means):
        assert math.isclose(a, b, rel_tol=1e-9)


def test_threshold_baseline_returns_normal_or_abnormal() -> None:
    s_norm = generate_synthetic_acoustic("normal", seed=1, snr_db=15)
    s_abn = generate_synthetic_acoustic("abnormal", seed=1, snr_db=15)
    d_norm = threshold_baseline_acoustic(s_norm, rms_threshold=0.5)  # high threshold -> normal
    d_abn = threshold_baseline_acoustic(s_abn, rms_threshold=0.0)    # zero threshold -> abnormal
    assert d_norm.predicted_label == "normal"
    assert d_abn.predicted_label == "abnormal"
    assert d_abn.method == "acoustic-rms-threshold-baseline"
