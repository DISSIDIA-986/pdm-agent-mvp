"""Order tracking + cepstrum: ball-fault-aware features.

Why these two operators:

1. **Order tracking** (`order_track`):
   Resample the time-domain signal so each sample corresponds to a fixed
   fraction of a shaft revolution rather than a fixed time interval. The
   resulting "angle-domain" signal has bearing fault peaks at integer
   *orders* (BPFI ≈ 5.4152, BPFO ≈ 3.5848, BSF ≈ 4.7135, FTF ≈ 0.3983)
   regardless of any speed drift. CWRU's drive-end runs at a near-constant
   ~1797 RPM, so the speed-drift benefit is marginal — we ship the operator
   anyway because (a) it is the textbook tool and (b) it makes the fault
   frequencies become integer orders that the cepstrum step can lock onto.

2. **Real cepstrum of the envelope** (`envelope_cepstrum`):
   The standard envelope-spectrum diagnostic looks for harmonic *amplitude*
   at one fault frequency. A ball defect's impulse train is modulated by
   the cage rotation (FTF) as balls move in and out of the load zone, so
   the energy is smeared across {2·BSF ± k·FTF} sidebands; the family-score
   v2 catches some of this but misses the periodic *structure* itself. The
   cepstrum (IFFT of log-magnitude spectrum) turns periodic spectral
   spacing into a spike at the corresponding *quefrency*, so a clear
   quefrency peak at `1/(2·FTF)` is a strong ball-fault signature even
   when the BSF fundamental is weak.

Honest scope: both operators are textbook signal-processing (no learning).
CWRU 0.007" ball faults are *known* to be hard for any envelope-only method;
the goal of this module is to lift the 0/10 baseline reported in
`eval/error_analysis.md`, not to claim universal ball-fault detection.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import hilbert, resample_poly


# ---------------------------------------------------------------------------
# Order tracking
# ---------------------------------------------------------------------------

def order_track(
    signal: np.ndarray,
    sample_rate_hz: float,
    rpm: float,
    *,
    samples_per_rev: int = 256,
) -> tuple[np.ndarray, float]:
    """Resample a constant-RPM time signal to fixed angular sampling.

    Returns (angle_signal, orders_per_sample). The angle-domain signal has
    `samples_per_rev` points per shaft revolution; its DFT bins are in units
    of *orders* (cycles per revolution).

    For variable-RPM scenarios this would need a tachometer phase reference
    and angle-resampling per cycle. For CWRU drive-end (constant ~1797 RPM)
    we use a single rational resampling factor — adequate, transparent, and
    fast.
    """
    if rpm <= 0 or sample_rate_hz <= 0:
        raise ValueError("rpm and sample_rate_hz must be positive")
    fr_hz = rpm / 60.0
    samples_per_rev_native = sample_rate_hz / fr_hz
    # Resample so that we land exactly on `samples_per_rev` samples / revolution.
    # We pick (up, down) such that signal length * (up/down) ≈ len * (samples_per_rev / samples_per_rev_native).
    # Use the rational approximation via numerator/denominator on a fine grid.
    from fractions import Fraction
    ratio = Fraction(samples_per_rev).limit_denominator(1024) / Fraction(samples_per_rev_native).limit_denominator(1024)
    up = ratio.numerator
    down = ratio.denominator
    # Guard against extreme ratios that would explode memory.
    if up * len(signal) > 50_000_000:
        raise ValueError(f"order_track ratio too extreme: up={up} down={down} len={len(signal)}")
    angle_signal = resample_poly(signal.astype(np.float64), up, down)
    # In the angle domain, the FFT bin index k corresponds to order =
    # k / samples_per_rev when window length == samples_per_rev. For a longer
    # window (typically the full signal), orders_per_sample is reported so
    # callers can convert FFT bin -> order accurately.
    orders_per_sample = 1.0 / samples_per_rev
    return angle_signal.astype(np.float32), orders_per_sample


def order_envelope_spectrum(
    angle_signal: np.ndarray,
    samples_per_rev: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope spectrum of an angle-domain signal.

    Returns (orders, magnitudes). `orders[i]` is the cycles-per-revolution
    at bin i.
    """
    if len(angle_signal) < 64:
        raise ValueError("angle_signal too short")
    analytic = hilbert(angle_signal.astype(np.float64))
    envelope = np.abs(analytic)
    envelope -= envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope))
    # Δorder per FFT bin = 1 / window_length_in_revolutions = 1 / (N / samples_per_rev)
    n = len(envelope)
    delta_order = samples_per_rev / n
    orders = np.arange(len(spectrum)) * delta_order
    return orders, spectrum


# ---------------------------------------------------------------------------
# Envelope cepstrum — ball-aware FTF modulation detector
# ---------------------------------------------------------------------------

def envelope_cepstrum(
    signal: np.ndarray, sample_rate_hz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Real cepstrum of the envelope spectrum.

    Returns (quefrencies_s, cepstrum). A spectral comb at spacing Δf produces
    a cepstral peak at quefrency 1/Δf — so a ball fault modulated by FTF
    (spacing ≈ 12 Hz @ 1797 RPM) shows up as a clear peak near
    quefrency 1/12 ≈ 83 ms.
    """
    if len(signal) < 64:
        raise ValueError("signal too short for envelope cepstrum")
    analytic = hilbert(signal.astype(np.float64))
    envelope = np.abs(analytic)
    envelope -= envelope.mean()
    # log-magnitude spectrum
    spec = np.abs(np.fft.rfft(envelope))
    # Floor to keep log stable in near-zero bins
    log_spec = np.log(spec + 1e-9)
    # Real cepstrum = inverse FFT of log-magnitude (symmetric input -> real output)
    cep = np.fft.irfft(log_spec, n=2 * (len(log_spec) - 1)).real
    # Quefrency axis (one-sided)
    n = 2 * (len(log_spec) - 1)
    quef = np.arange(n // 2) / sample_rate_hz
    return quef, cep[: n // 2]


def ftf_periodicity_score(
    signal: np.ndarray,
    sample_rate_hz: int,
    rpm: float,
    *,
    n_harmonics: int = 3,
    relative_band: float = 0.15,
) -> dict:
    """Quantify the FTF modulation strength in the envelope.

    Computes the envelope cepstrum and sums the cepstral peak height around
    quefrencies {1/FTF, 1/(2·FTF), ...} relative to a robust background.
    The returned score behaves like the family-score: ≥ ~3 is suggestive,
    ≥ ~6 is strong. This is the ball-fault signal we feed back to
    diagnose() v3.
    """
    fr_hz = rpm / 60.0
    ftf_hz = 0.3983 * fr_hz   # SKF 6205-2RS cage frequency
    quef, cep = envelope_cepstrum(signal, sample_rate_hz)
    # Skip the very-low-quefrency region (DC + linear trend) which has nothing to do with FTF.
    qmin = max(0.005, 0.5 / (ftf_hz * 2))  # avoid first 5 ms
    valid_mask = quef >= qmin
    valid_cep = np.abs(cep[valid_mask])
    valid_quef = quef[valid_mask]
    if valid_cep.size == 0:
        return {"score": 0.0, "ftf_hz": ftf_hz, "harmonic_peaks_q_s": [], "background": 0.0}
    background = float(np.median(valid_cep)) + 1e-9
    harmonic_peaks: list[dict] = []
    score = 0.0
    for k in range(1, n_harmonics + 1):
        q_center = 1.0 / (k * ftf_hz)
        half = relative_band * q_center
        mask = (valid_quef >= q_center - half) & (valid_quef <= q_center + half)
        if not mask.any():
            continue
        local_peak = float(valid_cep[mask].max())
        harmonic_peaks.append(
            {"k": k, "quefrency_s": q_center, "peak": local_peak}
        )
        score += (local_peak / background) / k
    return {
        "score": float(score),
        "ftf_hz": ftf_hz,
        "background": background,
        "harmonic_peaks_q_s": harmonic_peaks,
    }
