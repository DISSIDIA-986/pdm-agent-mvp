"""Smoke + validation tests for the acoustic data loader and synthetic generator."""
from __future__ import annotations

import numpy as np
import pytest

from pdm_agent.acoustic import (
    AcousticSample,
    generate_synthetic_acoustic,
    load_mimii_fan_dir,
    validate_acoustic_samples,
)


def test_synthetic_acoustic_deterministic() -> None:
    a = generate_synthetic_acoustic("normal", seed=42)
    b = generate_synthetic_acoustic("normal", seed=42)
    assert np.array_equal(a.signal, b.signal)


@pytest.mark.parametrize("label", ["normal", "abnormal"])
def test_synthetic_each_label_passes_validation(label: str) -> None:
    s = generate_synthetic_acoustic(label, seed=1)  # type: ignore[arg-type]
    assert s.label == label
    assert s.signal.shape == (160_000,)  # 10 s @ 16 kHz
    assert np.isfinite(s.signal).all()
    assert np.all(np.abs(s.signal) <= 1.0)


def test_abnormal_has_higher_broadband_energy() -> None:
    """Sanity check on the synthetic generator: 'abnormal' actually injects
    extra 1-3 kHz energy."""
    n_seed = 5
    norm_powers = []
    abnorm_powers = []
    for seed in range(n_seed):
        for label in ("normal", "abnormal"):
            s = generate_synthetic_acoustic(label, seed=seed, snr_db=12.0)  # type: ignore[arg-type]
            spec = np.abs(np.fft.rfft(s.signal))
            freqs = np.fft.rfftfreq(len(s.signal), 1 / s.sample_rate_hz)
            band_mask = (freqs >= 1000) & (freqs <= 3000)
            band_power = float(np.mean(spec[band_mask] ** 2))
            if label == "normal":
                norm_powers.append(band_power)
            else:
                abnorm_powers.append(band_power)
    assert np.mean(abnorm_powers) > 2.0 * np.mean(norm_powers), (
        f"abnormal mean 1-3 kHz power {np.mean(abnorm_powers):.3g} should clearly exceed "
        f"normal {np.mean(norm_powers):.3g}"
    )


def test_validate_rejects_nan() -> None:
    sig = np.zeros(160_000, dtype=np.float32)
    sig[100] = np.nan
    with pytest.raises(ValueError):
        s = AcousticSample(
            sample_id="bad",
            label="normal",
            signal=sig,
            sample_rate_hz=16_000,
            machine_id="id_synth",
            source="synthetic",
        )
        validate_acoustic_samples([s])


def test_acoustic_sample_rejects_short_signal() -> None:
    with pytest.raises(ValueError, match="signal too short"):
        AcousticSample(
            sample_id="too-short",
            label="normal",
            signal=np.zeros(100, dtype=np.float32),
            sample_rate_hz=16_000,
            machine_id="id_synth",
            source="synthetic",
        )


def test_validate_reports_label_balance() -> None:
    samples = [
        generate_synthetic_acoustic("normal", seed=0, machine_id="id_synth"),
        generate_synthetic_acoustic("normal", seed=1, machine_id="id_synth"),
        generate_synthetic_acoustic("abnormal", seed=2, machine_id="id_synth"),
    ]
    stats = validate_acoustic_samples(samples)
    assert stats["n_samples"] == 3
    assert stats["by_label"] == {"normal": 2, "abnormal": 1}
    assert stats["by_machine"] == {"id_synth": 3}


def test_load_mimii_returns_empty_for_missing_dir(tmp_path) -> None:
    samples = load_mimii_fan_dir(tmp_path / "nonexistent")
    assert samples == []
