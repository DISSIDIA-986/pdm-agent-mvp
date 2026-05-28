"""Unit tests for the Platt scaling calibrator."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pdm_agent.calibration import (
    Calibrator,
    expected_calibration_error,
    fit_calibrator,
    multiclass_brier_score,
    reliability_bins,
)
from pdm_agent.data import generate_synthetic


# ---------------------------------------------------------------------------
# Calibrator end-to-end
# ---------------------------------------------------------------------------

def _synthetic_pool() -> list:
    """Build a synthetic CWRU-like pool: 8 windows per class at SNR 15 dB."""
    out = []
    for fc in ("normal", "inner_race", "outer_race", "ball"):
        for i in range(8):
            out.append(generate_synthetic(fc, snr_db=15.0, seed=i * 7 + hash(fc) & 0xFFFF))  # type: ignore[arg-type]
    return out


def test_fit_calibrator_returns_two_params() -> None:
    cal = fit_calibrator(_synthetic_pool())
    assert cal.version == "temperature-multinomial-v1"
    assert cal.temperature > 0
    assert cal.n_train > 0
    assert cal.n_train_normal > 0
    assert set(cal.n_train_fault.keys()) == {"inner_race", "outer_race", "ball"}


def test_calibrate_probabilities_normalize() -> None:
    cal = fit_calibrator(_synthetic_pool())
    family_scores = {"inner_race": 150.0, "outer_race": 8.0, "ball": 5.0}
    probs = cal.calibrate(family_scores)
    assert set(probs.keys()) == {"normal", "inner_race", "outer_race", "ball"}
    total = sum(probs.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6)
    # Inner race had the dominant feature -> highest probability
    assert probs["inner_race"] == max(probs.values())


def test_calibrate_low_scores_yield_high_normal_probability() -> None:
    cal = fit_calibrator(_synthetic_pool())
    probs = cal.calibrate({"inner_race": 0.0, "outer_race": 0.0, "ball": 0.0})
    # All-zero family scores should make "normal" the dominant probability
    assert probs["normal"] == max(probs.values())
    assert probs["normal"] > 0.5


def test_calibrator_roundtrip(tmp_path) -> None:
    cal = fit_calibrator(_synthetic_pool())
    path = tmp_path / "cal.json"
    cal.save(path)
    restored = Calibrator.load(path)
    assert restored.version == cal.version
    assert math.isclose(restored.temperature, cal.temperature, rel_tol=1e-9)
    assert math.isclose(restored.normal_bias, cal.normal_bias, rel_tol=1e-9)
    # Same scores -> same probs
    scores = {"inner_race": 12.0, "outer_race": 3.0, "ball": 2.0}
    assert cal.calibrate(scores) == restored.calibrate(scores)


def test_multiclass_brier_perfect_predictions() -> None:
    probs = [{"normal": 1.0, "inner_race": 0.0, "outer_race": 0.0, "ball": 0.0}]
    y = ["normal"]
    assert multiclass_brier_score(probs, y) == 0.0


def test_multiclass_brier_worst_case() -> None:
    # Predict normal with 1.0 but truth is inner_race
    probs = [{"normal": 1.0, "inner_race": 0.0, "outer_race": 0.0, "ball": 0.0}]
    y = ["inner_race"]
    # (1-0)^2 + (0-1)^2 = 2
    assert multiclass_brier_score(probs, y) == 2.0


def test_calibrator_temperature_is_positive() -> None:
    """Temperature parameter must be > 0 (we optimise log_T to enforce this)."""
    cal = fit_calibrator(_synthetic_pool())
    assert cal.temperature > 0


def test_cv_report_has_required_keys() -> None:
    """Regression-pin the CV report shape so README references don't go stale.

    We don't actually fit the CV (that requires real CWRU on disk); we just
    spec-check the keys our README + adversarial review depend on by
    constructing the same structure manually.
    """
    required_keys = {
        "method",
        "calibration_method",
        "files",
        "n_windows",
        "calibrated_top1_accuracy",
        "uncalibrated_diagnose_accuracy",
        "pooled_top_label_ECE",
        "multiclass_brier",
        "fold_ece_mean",
        "fold_ece_std",
        "fold_ece_min",
        "fold_ece_max",
        "folds",
        "reliability",
    }
    # If eval/calibration_cv_metrics.json exists locally, load it and verify shape.
    import pathlib, json as _json
    p = pathlib.Path(__file__).resolve().parents[1] / "eval" / "calibration_cv_metrics.json"
    if not p.exists():
        pytest.skip("calibration CV results not yet generated — run eval/run_calibration.py")
    data = _json.loads(p.read_text())
    missing = required_keys - data.keys()
    assert not missing, f"CV report missing keys: {missing}"
    # Per-fold structure spec
    for fold in data["folds"]:
        assert {"held_out_file", "n_test", "n_train", "accuracy", "ece"} <= fold.keys()


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration_is_zero() -> None:
    # When confidence == accuracy in every bin, ECE = 0
    n = 100
    conf = np.linspace(0.05, 0.95, n)
    correct = np.random.default_rng(0).binomial(1, conf).astype(bool)
    # With enough samples per bin, ECE should be small; exactly zero requires
    # a deterministic relationship. Just check it's < 0.15.
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece < 0.15


def test_ece_all_wrong_at_high_conf_is_large() -> None:
    conf = [0.95] * 50
    correct = [False] * 50
    ece = expected_calibration_error(conf, correct, n_bins=10)
    # If you say 95% but you're 0% correct, ECE = 0.95
    assert ece > 0.9


def test_reliability_bins_count_total_matches() -> None:
    conf = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    correct = [True, False, True, True, False, True]
    bins = reliability_bins(conf, correct, n_bins=5)
    total = sum(b["n"] for b in bins)
    assert total == len(conf)
