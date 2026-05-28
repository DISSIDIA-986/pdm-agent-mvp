"""Fit a Platt-scaling calibrator with leave-one-file-out cross-validation.

Why leave-one-file-out instead of random k-fold: CWRU windows from the same
.mat file share the same physical bearing run and acquisition session. A
random k-fold split would lump train and test windows from the same run
together and give optimistically calibrated probabilities. Holding out one
FILE at a time forces the calibrator to predict on a bearing run it never
saw — the right boundary for an honest reliability number.

Outputs (eval/):
  calibration.json                   final calibrator fit on all 10 files
  calibration_cv_metrics.json        per-fold ECE + brier + accuracy
  calibration_reliability.json       binned reliability for the README plot
  calibration_reliability.png        reliability diagram (rendered figure)
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pdm_agent.calibration import (
    Calibrator,
    expected_calibration_error,
    fit_calibrator,
    multiclass_brier_score,
    reliability_bins,
)
from pdm_agent.data import CWRU_DOWNLOAD_INDEX, load_cwru_dataset
from pdm_agent.diagnostic import diagnose

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _windows_grouped_by_file() -> dict[str, list]:
    raw = ROOT / "data" / "raw"
    samples = load_cwru_dataset(raw, window_s=1.0)
    if not samples:
        raise SystemExit("No CWRU data found — run `python -m pdm_agent.data` first")
    by_file: dict[str, list] = defaultdict(list)
    for s in samples:
        if "file=" in s.notes:
            fn = s.notes.split("file=")[1]
            by_file[fn].append(s)
    return by_file


def leave_one_file_out_cv() -> dict:
    by_file = _windows_grouped_by_file()
    file_names = sorted(by_file.keys())
    fold_results = []
    all_top_conf: list[float] = []
    all_correct: list[bool] = []
    all_probs: list[dict[str, float]] = []
    all_truth: list[str] = []
    # Also track the *raw* (uncalibrated) softmax accuracy as a control
    all_raw_correct: list[bool] = []

    for held_out in file_names:
        train_samples = [s for fn, ws in by_file.items() if fn != held_out for s in ws]
        test_samples = by_file[held_out]
        cal = fit_calibrator(train_samples, training_pool_files=[f for f in file_names if f != held_out])

        fold_top_conf: list[float] = []
        fold_correct: list[bool] = []
        for s in test_samples:
            d = diagnose(s, calibrator=cal)
            probs = d.calibrated_probabilities or {}
            top_class = max(probs, key=probs.get) if probs else "normal"
            top_conf = float(probs.get(top_class, 0.0))
            is_correct = (top_class == s.fault_class)
            fold_top_conf.append(top_conf)
            fold_correct.append(is_correct)
            all_top_conf.append(top_conf)
            all_correct.append(is_correct)
            all_probs.append(probs)
            all_truth.append(s.fault_class)
            # Raw control: use the diagnostic's deterministic prediction (no calibration)
            all_raw_correct.append(d.predicted_class == s.fault_class)

        fold_results.append({
            "held_out_file": held_out,
            "n_test": len(test_samples),
            "n_train": len(train_samples),
            "accuracy": float(np.mean(fold_correct)) if fold_correct else None,
            "ece": expected_calibration_error(fold_top_conf, fold_correct, n_bins=10),
            "mean_confidence_on_correct": (
                float(np.mean([c for c, ok in zip(fold_top_conf, fold_correct) if ok]))
                if any(fold_correct) else None
            ),
            "mean_confidence_on_wrong": (
                float(np.mean([c for c, ok in zip(fold_top_conf, fold_correct) if not ok]))
                if any(not ok for ok in fold_correct) else None
            ),
        })

    overall_ece = expected_calibration_error(all_top_conf, all_correct, n_bins=10)
    overall_acc = float(np.mean(all_correct))
    overall_raw_acc = float(np.mean(all_raw_correct))
    reliability = reliability_bins(all_top_conf, all_correct, n_bins=10)
    # Standard multiclass Brier (per Codex feedback — was incorrectly binary before)
    brier_multiclass = multiclass_brier_score(all_probs, all_truth)
    # Per-fold ECE distribution — exposes cross-file variance (was hidden by pooling)
    fold_eces = [f["ece"] for f in fold_results]
    return {
        "method": "leave-one-file-out",
        "calibration_method": "temperature-multinomial-v1",
        "files": file_names,
        "n_windows": len(all_top_conf),
        # Headline metrics
        "calibrated_top1_accuracy": overall_acc,
        "uncalibrated_diagnose_accuracy": overall_raw_acc,  # control
        "pooled_top_label_ECE": overall_ece,
        "multiclass_brier": brier_multiclass,
        # Fold-level distribution (Codex flagged that pooled ECE alone hides variance)
        "fold_ece_mean": float(np.mean(fold_eces)) if fold_eces else 0.0,
        "fold_ece_std": float(np.std(fold_eces)) if fold_eces else 0.0,
        "fold_ece_min": float(np.min(fold_eces)) if fold_eces else 0.0,
        "fold_ece_max": float(np.max(fold_eces)) if fold_eces else 0.0,
        "folds": fold_results,
        "reliability": reliability,
    }


def render_reliability_diagram(reliability: list[dict], out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if reliability:
        xs = [b["mean_conf"] for b in reliability]
        ys = [b["accuracy"] for b in reliability]
        ns = [b["n"] for b in reliability]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888", label="perfectly calibrated")
        sc = ax.scatter(xs, ys, s=[max(60, n * 20) for n in ns], alpha=0.7, color="#1f77b4",
                        edgecolor="black", linewidth=0.5, label="empirical (size = bin count)")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted top-class probability (calibrated)")
    ax.set_ylabel("empirical accuracy in bin")
    ax.set_title("Reliability diagram — leave-one-file-out CWRU")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_final_calibrator() -> Calibrator:
    """Final calibrator fit on ALL CWRU files (for runtime use)."""
    by_file = _windows_grouped_by_file()
    all_samples = [s for ws in by_file.values() for s in ws]
    files = sorted(by_file.keys())
    return fit_calibrator(all_samples, training_pool_files=files)


def main() -> None:
    out_dir = ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running leave-one-file-out cross-validation…")
    cv = leave_one_file_out_cv()
    (out_dir / "calibration_cv_metrics.json").write_text(json.dumps(cv, indent=2))
    print(f"  calibrated top-1 accuracy   = {cv['calibrated_top1_accuracy']:.3f}")
    print(f"  uncalibrated (raw) accuracy = {cv['uncalibrated_diagnose_accuracy']:.3f}  (control)")
    print(f"  pooled top-label ECE        = {cv['pooled_top_label_ECE']:.3f}  (lower is better)")
    print(f"  fold ECE mean ± std         = {cv['fold_ece_mean']:.3f} ± {cv['fold_ece_std']:.3f}")
    print(f"  fold ECE [min, max]         = [{cv['fold_ece_min']:.3f}, {cv['fold_ece_max']:.3f}]")
    print(f"  multiclass Brier            = {cv['multiclass_brier']:.3f}  (lower is better)")

    print("Rendering reliability diagram…")
    render_reliability_diagram(cv["reliability"], ROOT / "docs" / "figures" / "calibration_reliability.png")
    (out_dir / "calibration_reliability.json").write_text(
        json.dumps(cv["reliability"], indent=2)
    )

    print("Fitting final calibrator on the full pool…")
    final = fit_final_calibrator()
    final.save(out_dir / "calibration.json")
    print(f"  saved {out_dir / 'calibration.json'}")
    print(f"    temperature = {final.temperature:.3f}")
    print(f"    normal_bias = {final.normal_bias:.3f}")
    print(f"    n_train     = {final.n_train}  ({final.n_train_normal} normal + "
          f"{sum(final.n_train_fault.values())} fault: {final.n_train_fault})")


if __name__ == "__main__":
    main()
