"""Acoustic anomaly diagnostic — second-modality counterpart to `diagnostic.py`.

Engineering posture mirrors the vibration diagnostic:
  - Deterministic features, no learned model. Keeps the audit trail trivially
    explainable and matches the "runtime, not copilot" thesis.
  - Class-balanced features chosen for *explainability* over leaderboard
    SOTA: spectral centroid, band-energy ratios, spectral flatness, and a
    crest-factor-like envelope kurtosis.
  - A simple z-score-vs-baseline anomaly score; the baseline is fit on a
    pool of `normal` samples (synthetic or real MIMII), so the diagnostic
    is reproducible on a laptop and on CI.
  - Same Severity bucketing as `diagnostic.py` so the LangGraph workflow
    treats both modalities the same way.

Honest scope (read first):
  - MIMII fan signal characteristics ≠ a real microgrid inverter cooling
    fan. The features here are *general* enough to plausibly transfer
    (spectral concentration, band-energy ratios), but production
    deployment would require recalibration on the actual installed fan.
  - The "abnormal" injection in `generate_synthetic_acoustic` is a
    smoke-test stand-in for the wide range of real MIMII fault modes
    (bearing damage, fan imbalance, voltage anomalies, contamination).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Iterable, Literal

import numpy as np
from scipy.signal import welch

from .acoustic import AcousticSample
from .diagnostic import Severity  # reuse the four severity buckets

ACOUSTIC_METHOD_VERSION = "acoustic-zscore-baseline-v1"


@dataclasses.dataclass(frozen=True)
class AcousticFeatures:
    rms: float
    centroid_hz: float          # spectral centroid (1st moment of PSD)
    rolloff_85_hz: float        # 85 % spectral-roll-off
    flatness: float             # geometric mean / arithmetic mean of PSD bins
    band_low_ratio: float       # 0-500 Hz / total
    band_mid_ratio: float       # 500-3000 Hz / total (where MIMII fan anomalies bunch)
    band_high_ratio: float      # 3000-7000 Hz / total
    envelope_kurtosis: float    # crest-factor analogue on the envelope

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.rms,
                self.centroid_hz,
                self.rolloff_85_hz,
                self.flatness,
                self.band_low_ratio,
                self.band_mid_ratio,
                self.band_high_ratio,
                self.envelope_kurtosis,
            ],
            dtype=np.float64,
        )


@dataclasses.dataclass(frozen=True)
class AcousticBaseline:
    """Per-feature mean + std fit on a pool of normal clips."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    n_train: int
    machine_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "n_train": self.n_train,
            "machine_ids": list(self.machine_ids),
        }

    def save(self, path: pathlib.Path | str) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "AcousticBaseline":
        d = json.loads(pathlib.Path(path).read_text())
        return cls(
            feature_names=tuple(d["feature_names"]),
            means=tuple(d["means"]),
            stds=tuple(d["stds"]),
            n_train=int(d["n_train"]),
            machine_ids=tuple(d["machine_ids"]),
        )

    def zscore(self, features: AcousticFeatures) -> np.ndarray:
        v = features.to_vector()
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        # Floor stds so a single-value feature (zero variance) doesn't blow up
        floor = np.maximum(stds, 1e-6)
        return (v - means) / floor


@dataclasses.dataclass(frozen=True)
class AcousticDiagnosis:
    sample_id: str
    machine_id: str
    predicted_label: Literal["normal", "abnormal"]
    severity: Severity
    anomaly_score: float        # mean |z| across features
    features: AcousticFeatures
    baseline_n_train: int
    method: str = ACOUSTIC_METHOD_VERSION

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "machine_id": self.machine_id,
            "predicted_label": self.predicted_label,
            "severity": self.severity,
            "anomaly_score": round(self.anomaly_score, 4),
            "anomaly_score_is_calibrated": False,
            "features": self.features.to_dict(),
            "baseline_n_train": self.baseline_n_train,
            "method": self.method,
        }


FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "centroid_hz",
    "rolloff_85_hz",
    "flatness",
    "band_low_ratio",
    "band_mid_ratio",
    "band_high_ratio",
    "envelope_kurtosis",
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(sample: AcousticSample) -> AcousticFeatures:
    sig = sample.signal.astype(np.float64)
    n = len(sig)
    if n < 1024:
        raise ValueError("signal too short for acoustic features")
    sig = sig - sig.mean()
    rms = float(np.sqrt(np.mean(sig ** 2)))

    # PSD via Welch — robust to clip-to-clip variation
    nperseg = min(4096, n // 4)
    freqs, psd = welch(sig, fs=sample.sample_rate_hz, nperseg=nperseg, noverlap=nperseg // 2)
    psd_sum = float(psd.sum()) + 1e-12

    centroid_hz = float((freqs * psd).sum() / psd_sum)
    cumulative = np.cumsum(psd)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff_85_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    log_psd = np.log(psd + 1e-12)
    flatness = float(np.exp(log_psd.mean()) / (psd.mean() + 1e-12))

    band_low = float(psd[(freqs >= 0) & (freqs < 500)].sum())
    band_mid = float(psd[(freqs >= 500) & (freqs < 3000)].sum())
    band_high = float(psd[(freqs >= 3000) & (freqs < 7000)].sum())
    band_total = band_low + band_mid + band_high + 1e-12
    band_low_ratio = band_low / band_total
    band_mid_ratio = band_mid / band_total
    band_high_ratio = band_high / band_total

    # Envelope kurtosis — picks up impulsive content (bearing knock, etc.)
    from scipy.signal import hilbert
    env = np.abs(hilbert(sig))
    env -= env.mean()
    var = float(env.var())
    if var < 1e-12:
        env_kurt = 0.0
    else:
        env_kurt = float(np.mean(env ** 4) / (var ** 2) - 3.0)

    return AcousticFeatures(
        rms=rms,
        centroid_hz=centroid_hz,
        rolloff_85_hz=rolloff_85_hz,
        flatness=flatness,
        band_low_ratio=band_low_ratio,
        band_mid_ratio=band_mid_ratio,
        band_high_ratio=band_high_ratio,
        envelope_kurtosis=env_kurt,
    )


# ---------------------------------------------------------------------------
# Baseline fitting + diagnosis
# ---------------------------------------------------------------------------

def fit_baseline(samples: Iterable[AcousticSample]) -> AcousticBaseline:
    """Fit per-feature mean and std on a pool of NORMAL samples only."""
    samples = list(samples)
    normals = [s for s in samples if s.label == "normal"]
    if not normals:
        raise ValueError("fit_baseline requires at least one normal sample")
    feats = np.vstack([extract_features(s).to_vector() for s in normals])
    means = feats.mean(axis=0)
    stds = feats.std(axis=0, ddof=1) if len(normals) >= 2 else np.ones_like(means)
    machine_ids = tuple(sorted({s.machine_id for s in normals}))
    return AcousticBaseline(
        feature_names=FEATURE_NAMES,
        means=tuple(map(float, means)),
        stds=tuple(map(float, stds)),
        n_train=len(normals),
        machine_ids=machine_ids,
    )


def _severity_from_score(score: float) -> Severity:
    """Bucket anomaly score (mean |z|) into the shared 4-level severity."""
    if score >= 5.0:
        return "critical"
    if score >= 3.0:
        return "alert"
    if score >= 1.5:
        return "watch"
    return "normal"


def diagnose_acoustic(sample: AcousticSample, baseline: AcousticBaseline) -> AcousticDiagnosis:
    """Score one clip against the baseline. Pure function, no I/O."""
    features = extract_features(sample)
    z = baseline.zscore(features)
    # Use mean of |z| as the headline anomaly score — robust to one outlier feature
    score = float(np.mean(np.abs(z)))
    severity = _severity_from_score(score)
    label: Literal["normal", "abnormal"] = "abnormal" if severity in ("alert", "critical") else "normal"
    return AcousticDiagnosis(
        sample_id=sample.sample_id,
        machine_id=sample.machine_id,
        predicted_label=label,
        severity=severity,
        anomaly_score=score,
        features=features,
        baseline_n_train=baseline.n_train,
    )


def threshold_baseline_acoustic(sample: AcousticSample, *, rms_threshold: float = 0.08) -> AcousticDiagnosis:
    """Naive RMS threshold reference — same idea as the vibration baseline."""
    feats = extract_features(sample)
    label: Literal["normal", "abnormal"] = "abnormal" if feats.rms >= rms_threshold else "normal"
    severity: Severity = "alert" if label == "abnormal" else "normal"
    return AcousticDiagnosis(
        sample_id=sample.sample_id,
        machine_id=sample.machine_id,
        predicted_label=label,
        severity=severity,
        anomaly_score=feats.rms,
        features=feats,
        baseline_n_train=0,
        method="acoustic-rms-threshold-baseline",
    )
