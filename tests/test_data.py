"""Smoke + validation tests for data loader and synthetic generator."""
from __future__ import annotations

import numpy as np
import pytest

from pdm_agent.data import (
    VibrationSample,
    _bearing_fault_frequencies,
    generate_synthetic,
    validate_samples,
)


def test_synthetic_generator_deterministic() -> None:
    a = generate_synthetic("inner_race", seed=42)
    b = generate_synthetic("inner_race", seed=42)
    assert np.array_equal(a.signal, b.signal)


@pytest.mark.parametrize("fc", ["normal", "inner_race", "ball", "outer_race"])
def test_synthetic_each_class_passes_validation(fc: str) -> None:
    s = generate_synthetic(fc, seed=1)  # type: ignore[arg-type]
    assert s.fault_class == fc
    assert s.signal.shape == (12_000,)
    assert np.isfinite(s.signal).all()


def test_validate_samples_reports_class_balance() -> None:
    samples = [generate_synthetic(fc, seed=i) for i, fc in enumerate(  # type: ignore[arg-type]
        ["normal", "normal", "inner_race", "ball", "outer_race"]
    )]
    stats = validate_samples(samples)
    assert stats["n_samples"] == 5
    assert stats["by_class"]["normal"] == 2
    assert stats["sources"] == ["synthetic"]


def test_validate_rejects_nan() -> None:
    sig = np.zeros(2048, dtype=np.float32)
    sig[100] = np.nan
    with pytest.raises(ValueError):
        s = VibrationSample(
            sample_id="bad",
            fault_class="normal",
            signal=sig,
            sample_rate_hz=12_000,
            rpm=1797,
            source="synthetic",
        )
        validate_samples([s])


def test_vibration_sample_rejects_short_signal() -> None:
    with pytest.raises(ValueError, match="signal too short"):
        VibrationSample(
            sample_id="too-short",
            fault_class="normal",
            signal=np.zeros(100, dtype=np.float32),
            sample_rate_hz=12_000,
            rpm=1797,
            source="synthetic",
        )


def test_fault_frequencies_present_in_inner_race_synthetic() -> None:
    """Sanity check: injected inner-race harmonics should show up in FFT."""
    s = generate_synthetic("inner_race", rpm=1797, duration_s=2.0, snr_db=20.0, seed=7)
    spectrum = np.abs(np.fft.rfft(s.signal))
    freqs = np.fft.rfftfreq(len(s.signal), d=1 / s.sample_rate_hz)
    expected = _bearing_fault_frequencies(s.rpm, "inner_race")
    for ef in expected:
        idx = int(np.argmin(np.abs(freqs - ef)))
        # Peak around expected fault frequency should be at least 4× the median magnitude
        median = float(np.median(spectrum))
        assert spectrum[idx] > 4 * median, f"expected peak at {ef:.1f} Hz, got {spectrum[idx]:.3f} vs median {median:.3f}"


def test_normal_synthetic_has_no_strong_fault_peaks() -> None:
    """Normal class should not have strong peaks at fault frequencies."""
    s = generate_synthetic("normal", duration_s=2.0, snr_db=20.0, seed=7)
    spectrum = np.abs(np.fft.rfft(s.signal))
    freqs = np.fft.rfftfreq(len(s.signal), d=1 / s.sample_rate_hz)
    # Check that no synthetic inner-race or outer-race peak exists
    for ef in _bearing_fault_frequencies(s.rpm, "inner_race") + _bearing_fault_frequencies(s.rpm, "outer_race"):
        idx = int(np.argmin(np.abs(freqs - ef)))
        median = float(np.median(spectrum))
        assert spectrum[idx] < 4 * median, f"unexpected fault peak in normal signal at {ef:.1f} Hz"
