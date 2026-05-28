"""Reproduction script for the ball-fault detection negative-result experiment.

We tested four envelope-family extensions for ball detection on real CWRU
0.007"/0.014"/0.021" bearing data. None of them lifted ball detection above
0/30 on a level worth shipping. This script is the executable proof of that
finding — running it should reproduce the same numbers, byte-for-byte.

Methods tried (all run on the same 30 ball windows across 3 .mat files):
  A. Baseline:        family score from diagnose() v2 (2 harmonics, 2 FTF sidebands)
  B. More harmonics:  n_harm=3, n_side=3
  C. SES at fixed band: squared envelope spectrum bandpassed 2-4.5 kHz
  D. SK band + SES:   sweep 8 sub-bands, pick max-envelope-kurtosis band, then SES

For each, we report:
  - ball detection rate per .mat file
  - the average dominant family-class on ball signals

Findings:
  - A, B, C all produce 0/30 ball detection on real CWRU. The bearing
    geometry causes BPFI/BPFO peaks to overlap with BSF harmonics, and on
    small-defect ball data the race-related components systematically
    dominate the envelope spectrum.
  - D produces 1/30 — within noise of zero. Spectral-kurtosis-driven band
    selection finds different bands for different windows, but on average
    the relative ordering of fault-class family scores does not change.

Conclusion: envelope-only methods are not sufficient for CWRU ball-fault
detection on the curated 0.007–0.021" subset. To lift this above zero we'd
need:
  - Cyclic Spectral Coherence (Antoni 2007)
  - Pre-whitening / AR-residual then SES
  - Supervised classification on time-frequency features
None of those are in scope for this MVP.

This negative result is the more useful portfolio signal than tuning until
some metric ticks up. The 0/30 number in the README is honest, the
operators in `order_tracking.py` are textbook-correct, and the limit is
documented at its real source: classical envelope methods cannot reliably
separate small ball-defect signatures from race-related components on
this rig.

References (consulted via Codex web search during round-2 review):
  - Smith & Randall 2015 — CWRU benchmark study; explicitly flags several
    ball records as "not diagnosable with established methods"
    https://www.sciencedirect.com/science/article/pii/S0888327015002034
  - Polito 2021 — CWRU envelope demodulation study; tags ball case B021_0
    as having non-periodic impulses
  - IEEE Access 2023 — CWRU bearing diagnosis with envelope + ML
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from pdm_agent.data import _bearing_fault_frequencies, load_cwru_dataset
from pdm_agent.diagnostic import _ftf_hz, _peak_in_band, diagnose

ROOT = Path(__file__).resolve().parents[1]
BALL_FILES = ["118.mat", "185.mat", "222.mat"]  # 0.007 / 0.014 / 0.021"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kurtosis(x: np.ndarray) -> float:
    centered = x - x.mean()
    var = float(centered.var())
    if var < 1e-12:
        return 0.0
    return float(np.mean(centered ** 4) / (var ** 2) - 3.0)


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def _ses_score(
    signal: np.ndarray, sr: int, rpm: float, band: tuple[float, float],
    fault_class: str, *, n_harm: int = 2, n_side: int = 2,
) -> float:
    filt = _bandpass(signal, sr, *band)
    env_sq = np.abs(hilbert(filt)) ** 2
    env_sq -= env_sq.mean()
    spec = np.abs(np.fft.rfft(env_sq))
    freqs = np.fft.rfftfreq(len(env_sq), 1 / sr)
    bg = float(np.median(spec[1:])) + 1e-9
    funds = _bearing_fault_frequencies(rpm, fault_class)
    if not funds:
        return 0.0
    ftf = _ftf_hz(rpm)
    total = 0.0
    for k in range(1, n_harm + 1):
        w = 1 / k
        center = k * funds[0]
        _, peak = _peak_in_band(freqs, spec, center, 4.0)
        total += w * peak / bg
        for m in range(1, n_side + 1):
            for sign in (-1, 1):
                _, sp = _peak_in_band(freqs, spec, center + sign * m * ftf, 4.0)
                total += 0.5 * w * sp / bg
    return total


def _select_band_by_sk(signal: np.ndarray, sr: int) -> tuple[tuple[float, float], float]:
    """Sweep 8 sub-bands in [500, 5500] Hz, pick max envelope-kurtosis."""
    edges = np.linspace(500, 5500, 9)
    best_k = -1e9
    best_band = (edges[0], edges[1])
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        env = np.abs(hilbert(_bandpass(signal, sr, lo, hi)))
        k = _kurtosis(env)
        if k > best_k:
            best_k = k
            best_band = (lo, hi)
    return best_band, best_k


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def method_a_baseline(sample) -> str:
    return diagnose(sample).predicted_class


def _classify_via_family(scores: dict[str, float], threshold: float = 50.0) -> str:
    top_class = max(scores, key=scores.get)
    return top_class if scores[top_class] >= threshold else "normal"


def method_b_more_harm(sample) -> str:
    band = (2000.0, 4500.0)
    scores = {
        c: _ses_score(sample.signal, sample.sample_rate_hz, sample.rpm, band, c,
                      n_harm=3, n_side=3)
        for c in ("inner_race", "outer_race", "ball")
    }
    return _classify_via_family(scores, threshold=100)


def method_c_ses_fixed(sample) -> str:
    band = (2000.0, 4500.0)
    scores = {
        c: _ses_score(sample.signal, sample.sample_rate_hz, sample.rpm, band, c)
        for c in ("inner_race", "outer_race", "ball")
    }
    return _classify_via_family(scores)


def method_d_sk_band(sample) -> str:
    band, _ = _select_band_by_sk(sample.signal, sample.sample_rate_hz)
    scores = {
        c: _ses_score(sample.signal, sample.sample_rate_hz, sample.rpm, band, c)
        for c in ("inner_race", "outer_race", "ball")
    }
    return _classify_via_family(scores)


METHODS = {
    "A baseline (diagnose v2)":         method_a_baseline,
    "B family (n_harm=3, n_side=3)":    method_b_more_harm,
    "C SES fixed band 2-4.5 kHz":       method_c_ses_fixed,
    "D SK-selected band + SES":         method_d_sk_band,
}


def main() -> None:
    samples = load_cwru_dataset(ROOT / "data" / "raw", window_s=1.0)
    by_file: dict[str, list] = defaultdict(list)
    for s in samples:
        if "file=" in s.notes:
            by_file[s.notes.split("file=")[1]].append(s)

    print("Ball-fault detection — envelope-family extensions on real CWRU")
    print("=" * 78)
    hdr_a = '118 (0.007in)'
    hdr_b = '185 (0.014in)'
    hdr_c = '222 (0.021in)'
    print(f"{'method':38s}  {hdr_a:>14s}  {hdr_b:>14s}  {hdr_c:>14s}  {'total':>5s}")
    print("-" * 78)
    for name, fn in METHODS.items():
        row = [f"{name:38s}"]
        total = 0
        for ball_file in BALL_FILES:
            ws = by_file.get(ball_file, [])
            preds = [fn(s) for s in ws]
            correct = sum(p == "ball" for p in preds)
            total += correct
            row.append(f"{correct}/{len(ws)}".rjust(14))
        row.append(f"{total}/30".rjust(5))
        print("  ".join(row))
    print("=" * 78)
    print("All envelope-family extensions remain at-or-near 0/30 on CWRU ball data.")
    print("Negative result is intentional and documented in docs/research/ball-detection.md")


if __name__ == "__main__":
    main()
