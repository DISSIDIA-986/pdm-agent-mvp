"""Confidence calibration via multinomial temperature scaling.

Background
----------
The deterministic family-score diagnostic produces a raw `score` per fault
class. Treating those as softmax logits and taking the top one gives a number
that looks like a probability but isn't — a misclassification can still
report 0.99 because the family score is not in log-likelihood units.

We calibrate with the standard temperature-scaling recipe (Guo et al. 2017),
adapted to the 4-class case where "normal" is a non-trivial channel:

    logits(c | s) = [b_normal, s_inner/T, s_outer/T, s_ball/T]
    P(c | s)      = softmax(logits)

Two learnable parameters:
  T            — temperature, shared across fault classes
  b_normal     — bias of the implicit normal logit

This avoids the one-vs-rest pathology where ``P(normal)`` was constructed as
``∏(1 - P_fault)`` and silently dominated argmax. Multinomial calibration
keeps the four classes mutually exclusive by construction and matches the
classification setup the diagnostic was already using.

Honest scope
------------
- Trained on CWRU drive-end bearings (12 kHz, ~1800 RPM, SKF 6205 geometry).
- We report leave-one-FILE-out cross-validation so the reliability number is
  on bearing runs the calibrator never saw.
- We expose BOTH a pooled ECE (over all folds) AND per-fold ECE (so we don't
  hide cross-file variance).
- The calibrator persists as a small JSON — no opaque pickled weights.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .data import VibrationSample
from .diagnostic import diagnose

FAULT_CLASSES: tuple[str, ...] = ("inner_race", "outer_race", "ball")
ALL_CLASSES: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")
CALIBRATOR_VERSION = "temperature-multinomial-v1"


@dataclasses.dataclass(frozen=True)
class Calibrator:
    """Two-parameter multinomial temperature-scaling calibrator."""

    version: str
    method: str
    temperature: float       # T > 0
    normal_bias: float       # bias for the implicit normal logit
    n_train: int
    n_train_normal: int
    n_train_fault: dict[str, int]
    training_pool_files: list[str]

    def _logits(self, family_scores: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                self.normal_bias,
                family_scores.get("inner_race", 0.0) / self.temperature,
                family_scores.get("outer_race", 0.0) / self.temperature,
                family_scores.get("ball", 0.0) / self.temperature,
            ],
            dtype=np.float64,
        )

    def calibrate(self, family_scores: dict[str, float]) -> dict[str, float]:
        """Map raw family scores to per-class probabilities (4-way softmax)."""
        logits = self._logits(family_scores)
        logits -= logits.max()  # numerical stability
        exps = np.exp(logits)
        probs = exps / exps.sum()
        return {cls: float(probs[i]) for i, cls in enumerate(ALL_CLASSES)}

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(**d)

    def save(self, path: pathlib.Path | str) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Calibrator":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))


# ---------------------------------------------------------------------------
# Fitting via scipy.optimize on multinomial NLL
# ---------------------------------------------------------------------------

def _build_feature_matrix(samples: Iterable[VibrationSample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, files) where X[i] = [s_inner, s_outer, s_ball], y[i] ∈ 0..3."""
    rows: list[list[float]] = []
    ys: list[int] = []
    files: set[str] = set()
    class_idx = {c: i for i, c in enumerate(ALL_CLASSES)}
    for s in samples:
        d = diagnose(s)
        family = {e.fault_class: e.score for e in d.evidence}
        rows.append([family.get(c, 0.0) for c in FAULT_CLASSES])
        ys.append(class_idx.get(s.fault_class, 0))
        if "file=" in s.notes:
            files.add(s.notes.split("file=")[1])
    return np.asarray(rows, dtype=np.float64), np.asarray(ys, dtype=np.int64), sorted(files)


def _nll(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of multinomial temperature scaling.

    params = [log_T, b_normal]   (we optimise log_T so T stays positive)
    """
    log_T, b_normal = params
    T = math.exp(log_T)
    n = X.shape[0]
    # Logits: column 0 = b_normal; columns 1..3 = X[:, c] / T for fault classes
    L = np.empty((n, 4), dtype=np.float64)
    L[:, 0] = b_normal
    L[:, 1:] = X / T
    L -= L.max(axis=1, keepdims=True)
    exps = np.exp(L)
    Z = exps.sum(axis=1, keepdims=True)
    log_probs = L - np.log(Z)
    # Pick the true-class column for each row
    nll = -log_probs[np.arange(n), y].mean()
    # Tiny L2 regulariser keeps the fit finite if a column is degenerate
    return float(nll + 1e-6 * (log_T ** 2 + b_normal ** 2))


def fit_calibrator(
    samples: Iterable[VibrationSample],
    *,
    training_pool_files: list[str] | None = None,
) -> Calibrator:
    """Fit T and b_normal by maximum likelihood."""
    samples_list = list(samples)
    X, y, files_in_pool = _build_feature_matrix(samples_list)
    if X.shape[0] == 0:
        raise ValueError("no samples provided to fit_calibrator")
    # Initial guess: T scales family scores into a sensible range (max ~5),
    # b_normal placed so normal class gets ~uniform weight at low scores.
    s_top = float(X.max()) if X.size else 1.0
    init = np.array([math.log(max(s_top / 5.0, 1e-3)), 0.0])
    result = minimize(_nll, init, args=(X, y), method="L-BFGS-B")
    if not result.success:
        # Fall back to BFGS without bounds if L-BFGS-B did not converge —
        # both should always succeed on a well-posed 2-D logistic NLL, but
        # we make this explicit instead of silently using whatever .x holds.
        result_fallback = minimize(_nll, init, args=(X, y), method="BFGS")
        if not result_fallback.success:
            raise RuntimeError(
                f"calibrator fit failed: L-BFGS-B={result.message!r}; "
                f"BFGS={result_fallback.message!r}"
            )
        result = result_fallback
    log_T, b_normal = result.x
    T = math.exp(log_T)
    n_train_fault = {cls: int((y == i).sum()) for i, cls in enumerate(ALL_CLASSES) if cls != "normal"}
    n_train_normal = int((y == 0).sum())
    pool = sorted(set(training_pool_files or []) | set(files_in_pool))
    return Calibrator(
        version=CALIBRATOR_VERSION,
        method="envelope-spectrum-v2-family",
        temperature=float(T),
        normal_bias=float(b_normal),
        n_train=int(X.shape[0]),
        n_train_normal=n_train_normal,
        n_train_fault=n_train_fault,
        training_pool_files=pool,
    )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: Iterable[float], correct: Iterable[bool], *, n_bins: int = 10
) -> float:
    """Top-label Expected Calibration Error."""
    confidences = np.asarray(list(confidences), dtype=np.float64)
    correct = np.asarray(list(correct), dtype=np.float64)
    n = len(confidences)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def multiclass_brier_score(
    probs: list[dict[str, float]], y_true: list[str]
) -> float:
    """Standard multiclass Brier: mean Σ_c (P(c) - 1[true==c])^2.

    Lower is better; range [0, 2] (for a 4-class problem).
    """
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, y_true):
        for cls in ALL_CLASSES:
            indicator = 1.0 if y == cls else 0.0
            total += (p.get(cls, 0.0) - indicator) ** 2
    return float(total / len(probs))


def reliability_bins(
    confidences: Iterable[float], correct: Iterable[bool], *, n_bins: int = 10
) -> list[dict]:
    confidences = np.asarray(list(confidences), dtype=np.float64)
    correct = np.asarray(list(correct), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.any():
            out.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": int(mask.sum()),
                    "mean_conf": float(confidences[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return out
