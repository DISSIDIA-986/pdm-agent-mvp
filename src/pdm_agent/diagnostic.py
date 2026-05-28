"""Vibration diagnostic core: feature extraction + envelope spectrum + fault classifier.

This module is the PdM "brain". Inspired by ISO 13374 stages 1-3 (data acquisition,
data manipulation, state detection). We deliberately keep the classifier simple
and rule-based so the audit trail is trivially explainable — this matches the
"runtime, not copilot" thesis: deterministic detection backed by physics, LLM
only for orchestration and natural-language summarisation.

Diagnostic flow (v2 — post adversarial review):
  raw signal
    └── (optional) bandpass into resonance band (default 2-4 kHz)
    └── time-domain features (RMS, kurtosis, crest factor)
    └── envelope spectrum (Hilbert → FFT)
    └── for each fault class score = sum over harmonic family
            sum_{k=1..K_HARM} ( peak_in_band(k * f_fault, half_width) +
                                sum_{m=1..M_SIDE} peak_in_band(k*f_fault ± m*FTF, half_width))
    └── softmax across fault classes; threshold + separation gate for normal
    └── severity = bucketed from top score + kurtosis impulse signal

Why harmonic+sideband family: ball faults in particular often have weak
fundamental and stronger 2x harmonic, with FTF-modulated sidebands. Looking
only at the first harmonic systematically under-detects them (which is exactly
what we observed: 0/10 ball detection on real CWRU before this fix).
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .data import VibrationSample, _bearing_fault_frequencies

Severity = Literal["normal", "watch", "alert", "critical"]


@dataclasses.dataclass(frozen=True)
class TimeFeatures:
    rms: float
    peak: float
    kurtosis: float
    crest_factor: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FaultEvidence:
    """Per-fault evidence: harmonic energy and signal-to-background ratio."""

    fault_class: str
    peak_freq_hz: float
    peak_amp: float
    background_median: float
    score: float  # peak_amp / background_median (>=1.0; >3 strong)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    sample_id: str
    predicted_class: str
    severity: Severity
    confidence: float  # softmax over family scores — see note below
    time_features: TimeFeatures
    evidence: list[FaultEvidence]
    sample_rate_hz: int
    rpm: float
    method: str = "envelope-spectrum-v2-family"

    # IMPORTANT: `confidence` is *not* a calibrated posterior probability. It is
    # a softmax over the deterministic family scores (one per fault class plus
    # an implicit "normal" channel). A miscalibrated misclassification can
    # still produce confidence close to 1.0 — eval/error_analysis.md
    # documents this. Treat it as a ranking signal between the competing
    # candidate classes; do NOT show it to operators as "% probability of
    # failure". Real calibration would require Platt scaling or isotonic
    # regression on a held-out labelled set.

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "predicted_class": self.predicted_class,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "confidence_is_calibrated": False,
            "time_features": self.time_features.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "sample_rate_hz": self.sample_rate_hz,
            "rpm": self.rpm,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def time_features(signal: np.ndarray) -> TimeFeatures:
    """Compute baseline time-domain statistics."""
    if len(signal) < 16:
        raise ValueError("signal too short for time features")
    signal = np.asarray(signal, dtype=np.float64)
    mean = signal.mean()
    centered = signal - mean
    var = centered.var()
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(np.abs(centered)))
    # Fisher kurtosis (excess); zero for Gaussian
    if var < 1e-12:
        kurt = 0.0
    else:
        m4 = np.mean(centered ** 4)
        kurt = float(m4 / (var ** 2) - 3.0)
    crest = peak / rms if rms > 1e-12 else 0.0
    return TimeFeatures(rms=rms, peak=peak, kurtosis=kurt, crest_factor=float(crest))


def _bandpass(signal: np.ndarray, sample_rate_hz: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Used to isolate bearing resonance band."""
    nyq = 0.5 * sample_rate_hz
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        return signal
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def envelope_spectrum(
    signal: np.ndarray,
    sample_rate_hz: int,
    *,
    resonance_band_hz: tuple[float, float] | None = (2000.0, 4500.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute envelope spectrum via resonance-band Hilbert envelope.

    `resonance_band_hz`: if provided, bandpass the raw signal into this band
    BEFORE the Hilbert envelope. This isolates the high-frequency carrier where
    bearing impacts excite structural resonances and demodulates fault impulses
    cleanly. Standard CWRU practice puts this band in 2-4 kHz. Pass None to
    operate on full-band signal (legacy behaviour).
    """
    if len(signal) < 64:
        raise ValueError("signal too short for envelope spectrum")
    work = signal.astype(np.float64)
    if resonance_band_hz is not None and sample_rate_hz >= 2 * resonance_band_hz[1]:
        try:
            work = _bandpass(work, sample_rate_hz, resonance_band_hz[0], resonance_band_hz[1])
        except Exception:  # noqa: BLE001 — fall back to full-band if filter fails
            work = signal.astype(np.float64)
    analytic = hilbert(work)
    envelope = np.abs(analytic)
    envelope -= envelope.mean()  # remove DC
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1 / sample_rate_hz)
    return freqs, spectrum


def _peak_in_band(freqs: np.ndarray, spectrum: np.ndarray, center_hz: float, half_width_hz: float) -> tuple[float, float]:
    """Return (peak_freq, peak_amp) in [center-half_width, center+half_width]."""
    if center_hz <= 0 or center_hz >= freqs[-1]:
        return center_hz, 0.0
    mask = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    if not mask.any():
        return center_hz, 0.0
    band_freqs = freqs[mask]
    band_amp = spectrum[mask]
    i = int(np.argmax(band_amp))
    return float(band_freqs[i]), float(band_amp[i])


def _ftf_hz(rpm: float) -> float:
    """Fundamental Train Frequency for SKF 6205-2RS (CWRU drive-end)."""
    return 0.3983 * (rpm / 60.0)


def _harmonic_family_score(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    fundamental_hz: float,
    ftf_hz: float,
    background: float,
    *,
    n_harmonics: int = 3,
    n_sidebands: int = 2,
    half_width_hz: float = 4.0,
) -> tuple[float, float, float]:
    """Aggregate evidence over harmonic family + FTF sidebands.

    Returns (family_score, fundamental_peak_freq, fundamental_peak_amp). The
    family_score is the L1 sum of (peak_amp / background) across harmonics and
    sidebands, downweighted for higher harmonics so a single big fundamental
    still dominates a noisy higher-order peak.
    """
    total = 0.0
    fund_freq, fund_amp = _peak_in_band(freqs, spectrum, fundamental_hz, half_width_hz)
    for k in range(1, n_harmonics + 1):
        weight = 1.0 / k  # 1, 1/2, 1/3 — first harmonic dominates
        center = k * fundamental_hz
        _, amp = _peak_in_band(freqs, spectrum, center, half_width_hz)
        total += weight * (amp / background)
        for m in range(1, n_sidebands + 1):
            for sign in (-1, 1):
                side = center + sign * m * ftf_hz
                _, samp = _peak_in_band(freqs, spectrum, side, half_width_hz)
                total += 0.5 * weight * (samp / background)
    return total, fund_freq, fund_amp


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _severity_from_score(score: float, kurtosis: float) -> Severity:
    """Bucket overall severity using top fault-frequency family score + kurtosis.

    Thresholds chosen on CWRU-like data after the v2 family-score rewrite;
    documented in README for honesty. Higher than v1 because family scores
    sum across multiple harmonics+sidebands so absolute magnitudes are larger.
    """
    if score >= 25.0 or kurtosis >= 8.0:
        return "critical"
    if score >= 12.0 or kurtosis >= 4.5:
        return "alert"
    if score >= 6.0 or kurtosis >= 2.5:
        return "watch"
    return "normal"


def _half_width_hz(sample_rate_hz: int, n_samples: int, fundamental_hz: float, rpm_drift_pct: float = 1.0) -> float:
    """Tolerance for peak search: max(2 FFT bins, RPM-drift-derived).

    Defaults to 1% RPM drift, which on CWRU's 1797 RPM ≈ 0.3 Hz at the BPFO
    fundamental and 0.4 Hz at BPFI. We add 2-bin floor so short windows still
    work despite coarse frequency resolution.
    """
    bin_hz = sample_rate_hz / n_samples
    drift_hz = fundamental_hz * (rpm_drift_pct / 100.0)
    return float(max(2.0 * bin_hz, drift_hz, 1.0))


def diagnose(sample: VibrationSample) -> Diagnosis:
    """Run end-to-end diagnostic on a VibrationSample. Pure function, no I/O."""
    tf = time_features(sample.signal)
    freqs, spectrum = envelope_spectrum(sample.signal, sample.sample_rate_hz)

    # Background = robust median (excluding DC bin) used for SNR-ish scoring
    background = float(np.median(spectrum[1:])) + 1e-9

    ftf = _ftf_hz(sample.rpm)
    evidences: list[FaultEvidence] = []
    for cls in ("inner_race", "outer_race", "ball"):
        fundamental_freqs = _bearing_fault_frequencies(sample.rpm, cls)  # type: ignore[arg-type]
        if not fundamental_freqs:
            continue
        fundamental = fundamental_freqs[0]
        half = _half_width_hz(sample.sample_rate_hz, len(sample.signal), fundamental)
        # Limit to 2 harmonics to avoid CWRU geometry aliasing: 3*BPFO ≈ 2*BPFI
        # for SKF 6205, which double-counts and causes cross-class contamination.
        family_score, peak_freq, peak_amp = _harmonic_family_score(
            freqs, spectrum, fundamental, ftf, background,
            n_harmonics=2, n_sidebands=2, half_width_hz=half,
        )
        evidences.append(
            FaultEvidence(
                fault_class=cls,
                peak_freq_hz=peak_freq,
                peak_amp=peak_amp,
                background_median=background,
                score=float(family_score),
            )
        )

    # Pick top-scoring fault class. If best score < threshold, predict normal.
    # We require BOTH (a) absolute score over NORMAL_THRESHOLD AND (b) clear
    # separation from runner-up (top/second >= MIN_SEPARATION). This guards
    # against spectral leakage producing weak peaks across all bands.
    evidences_sorted = sorted(evidences, key=lambda e: e.score, reverse=True)
    top = evidences_sorted[0]
    second = evidences_sorted[1] if len(evidences_sorted) > 1 else top
    NORMAL_THRESHOLD = 6.0  # family score; corresponds to ~watch threshold
    MIN_SEPARATION = 1.3  # top must beat second-place by 30%
    separation = top.score / max(second.score, 1e-6)
    has_strong_peak = top.score >= NORMAL_THRESHOLD and separation >= MIN_SEPARATION
    has_impulsive_kurtosis = tf.kurtosis >= 3.0
    if has_strong_peak or has_impulsive_kurtosis:
        predicted = top.fault_class
        top_score = top.score
    else:
        predicted = "normal"
        top_score = top.score  # still report for transparency

    # Confidence: softmax across the three fault scores (plus an implicit "normal" channel)
    scores = np.array([e.score for e in evidences] + [NORMAL_THRESHOLD], dtype=np.float64)
    scores = scores - scores.max()  # numerical stability
    probs = np.exp(scores) / np.exp(scores).sum()
    if predicted == "normal":
        conf = float(probs[-1])
    else:
        # index of predicted class in evidences order
        idx = next(i for i, e in enumerate(evidences) if e.fault_class == predicted)
        conf = float(probs[idx])

    severity = _severity_from_score(top_score, tf.kurtosis)
    if predicted == "normal":
        severity = "normal"

    return Diagnosis(
        sample_id=sample.sample_id,
        predicted_class=predicted,
        severity=severity,
        confidence=conf,
        time_features=tf,
        evidence=evidences,
        sample_rate_hz=sample.sample_rate_hz,
        rpm=sample.rpm,
    )


def threshold_baseline(sample: VibrationSample, rms_threshold: float = 0.5) -> Diagnosis:
    """Naive RMS threshold baseline used for evaluation comparison.

    Reports binary normal vs alert based purely on RMS amplitude. This is the
    "would-a-typical-PLC-do" reference so the agent's improvement is measurable.
    """
    tf = time_features(sample.signal)
    severity: Severity = "alert" if tf.rms >= rms_threshold else "normal"
    predicted = "alert_unknown" if severity == "alert" else "normal"
    return Diagnosis(
        sample_id=sample.sample_id,
        predicted_class=predicted,
        severity=severity,
        confidence=1.0 if severity == "alert" else 0.0,
        time_features=tf,
        evidence=[],
        sample_rate_hz=sample.sample_rate_hz,
        rpm=sample.rpm,
        method="rms-threshold-baseline",
    )
