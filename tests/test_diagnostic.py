"""Tests for diagnostic primitives — verify physics-derived detection works."""
from __future__ import annotations

import numpy as np
import pytest

from pdm_agent.data import generate_synthetic, _bearing_fault_frequencies
from pdm_agent.diagnostic import (
    diagnose,
    envelope_spectrum,
    threshold_baseline,
    time_features,
)


def test_time_features_basic_stats() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(4096).astype(np.float32)
    tf = time_features(sig)
    assert tf.rms > 0.5
    assert tf.peak > tf.rms
    assert tf.crest_factor > 1.0
    # Standard normal: excess kurtosis ~0
    assert abs(tf.kurtosis) < 1.0


def test_envelope_spectrum_returns_consistent_shape() -> None:
    sig = np.sin(2 * np.pi * 100 * np.arange(4096) / 12_000).astype(np.float32)
    freqs, spec = envelope_spectrum(sig, 12_000)
    assert len(freqs) == len(spec)
    assert freqs[0] == 0.0
    # Highest frequency is Nyquist
    assert pytest.approx(freqs[-1], rel=1e-3) == 6000.0


@pytest.mark.parametrize("fc", ["inner_race", "outer_race"])
def test_diagnose_detects_synthetic_fault(fc: str) -> None:
    """High-SNR synthetic faults should be detected by the diagnostic."""
    sample = generate_synthetic(fc, duration_s=2.0, snr_db=20.0, seed=11)  # type: ignore[arg-type]
    d = diagnose(sample)
    assert d.predicted_class == fc, f"expected {fc}, got {d.predicted_class}; evidence={d.evidence}"
    assert d.severity in {"watch", "alert", "critical"}
    assert d.confidence > 0.0


def test_diagnose_normal_signal_predicts_normal() -> None:
    sample = generate_synthetic("normal", duration_s=2.0, snr_db=20.0, seed=22)
    d = diagnose(sample)
    assert d.predicted_class == "normal"
    assert d.severity == "normal"


def test_diagnose_includes_evidence_for_all_classes() -> None:
    sample = generate_synthetic("inner_race", duration_s=1.5, snr_db=15.0, seed=33)
    d = diagnose(sample)
    classes_in_evidence = {e.fault_class for e in d.evidence}
    assert classes_in_evidence == {"inner_race", "outer_race", "ball"}
    # Evidence should be ranked: predicted class has highest score
    by_class = {e.fault_class: e.score for e in d.evidence}
    assert by_class["inner_race"] >= by_class["outer_race"]
    assert by_class["inner_race"] >= by_class["ball"]


def test_diagnose_evidence_peak_near_expected_frequency() -> None:
    sample = generate_synthetic("inner_race", rpm=1797, duration_s=2.0, snr_db=25.0, seed=44)
    d = diagnose(sample)
    expected = _bearing_fault_frequencies(sample.rpm, "inner_race")[0]
    inner_evidence = next(e for e in d.evidence if e.fault_class == "inner_race")
    # Peak should be within tolerance of expected fault frequency
    assert abs(inner_evidence.peak_freq_hz - expected) < 5.0, (
        f"expected ~{expected:.1f} Hz, got {inner_evidence.peak_freq_hz:.1f}"
    )


def test_threshold_baseline_returns_binary_severity() -> None:
    high = generate_synthetic("inner_race", duration_s=1.0, snr_db=3.0, seed=55)
    low = generate_synthetic("normal", duration_s=1.0, snr_db=20.0, seed=66)
    d_high = threshold_baseline(high, rms_threshold=0.3)
    d_low = threshold_baseline(low, rms_threshold=0.3)
    # Synthetic injection ensures inner-race signal has higher RMS
    assert d_high.severity in {"alert", "normal"}
    assert d_low.severity in {"alert", "normal"}
    # baseline never returns the multi-class structured prediction
    assert d_high.predicted_class in {"alert_unknown", "normal"}


def test_diagnosis_is_json_serialisable() -> None:
    import json
    sample = generate_synthetic("outer_race", duration_s=1.0, snr_db=15.0, seed=77)
    d = diagnose(sample)
    payload = json.dumps(d.to_dict())
    parsed = json.loads(payload)
    assert parsed["predicted_class"] == "outer_race"
    assert len(parsed["evidence"]) == 3
