"""Tests for the order-tracking + cepstrum module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pdm_agent.data import _bearing_fault_frequencies, generate_synthetic
from pdm_agent.order_tracking import (
    envelope_cepstrum,
    ftf_periodicity_score,
    order_envelope_spectrum,
    order_track,
)


def test_order_track_preserves_revolutions() -> None:
    """Resampled length / samples_per_rev should equal the original revolution count."""
    rpm = 1797.0
    sample_rate = 12_000
    duration_s = 1.0
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    signal = np.sin(2 * np.pi * (rpm / 60.0) * t).astype(np.float32)
    angle_signal, _ = order_track(signal, sample_rate, rpm, samples_per_rev=256)
    expected_revs = duration_s * (rpm / 60.0)
    actual_revs = len(angle_signal) / 256
    assert abs(actual_revs - expected_revs) / expected_revs < 0.01


def test_order_track_rejects_zero_rpm() -> None:
    signal = np.zeros(2048, dtype=np.float32)
    with pytest.raises(ValueError):
        order_track(signal, sample_rate_hz=12_000, rpm=0)


def test_order_envelope_spectrum_returns_consistent_shape() -> None:
    rng = np.random.default_rng(0)
    angle_signal = rng.standard_normal(4096).astype(np.float32)
    orders, spec = order_envelope_spectrum(angle_signal, samples_per_rev=256)
    assert len(orders) == len(spec)
    assert orders[0] == 0.0
    # Largest order = (N/2) * (samples_per_rev/N) = samples_per_rev/2
    assert math.isclose(orders[-1], 128.0, rel_tol=1e-3)


def test_envelope_cepstrum_returns_sane_arrays() -> None:
    sample = generate_synthetic("ball", duration_s=1.0, snr_db=15.0, seed=0)
    quef, cep = envelope_cepstrum(sample.signal, sample.sample_rate_hz)
    assert quef.shape == cep.shape
    assert quef[0] == 0.0
    # Quefrencies are in seconds, bounded by signal length / 2 / sr
    assert quef[-1] < 0.6  # ~0.5s for a 1s window


def test_ftf_score_separates_ball_from_normal() -> None:
    """Synthetic ball faults inject FTF modulation; normal signals should not."""
    rpm = 1797
    ball = generate_synthetic("ball", rpm=rpm, duration_s=2.0, snr_db=20.0, seed=1)
    normal = generate_synthetic("normal", rpm=rpm, duration_s=2.0, snr_db=20.0, seed=1)
    score_ball = ftf_periodicity_score(ball.signal, ball.sample_rate_hz, ball.rpm)
    score_normal = ftf_periodicity_score(normal.signal, normal.sample_rate_hz, normal.rpm)
    # Both should produce non-negative scores; ball must beat normal.
    assert score_ball["score"] >= 0
    assert score_normal["score"] >= 0
    assert score_ball["score"] > score_normal["score"], (
        f"ball score {score_ball['score']} not greater than normal {score_normal['score']}"
    )


def test_ftf_score_includes_harmonic_peaks_metadata() -> None:
    sample = generate_synthetic("ball", duration_s=1.0, snr_db=15.0, seed=2)
    score = ftf_periodicity_score(sample.signal, sample.sample_rate_hz, sample.rpm)
    assert "ftf_hz" in score
    assert score["ftf_hz"] > 0
    assert isinstance(score["harmonic_peaks_q_s"], list)
    # SKF 6205 FTF at 1797 RPM ≈ 11.93 Hz; quefrency ≈ 0.0838 s for the first harmonic
    assert any(0.05 < p["quefrency_s"] < 0.15 for p in score["harmonic_peaks_q_s"])


def test_ftf_score_robust_to_short_signal() -> None:
    short = np.zeros(2048, dtype=np.float32) + 0.01
    # Should not crash; may return tiny score
    score = ftf_periodicity_score(short, 12_000, 1797)
    assert 0.0 <= score["score"] < 100.0
