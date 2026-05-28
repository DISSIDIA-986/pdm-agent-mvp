"""Evaluate the acoustic diagnostic on the synthetic test set + the RMS baseline.

Layout, intentionally parallel to `eval/run_eval.py`:
  - eval/acoustic_eval_v1.jsonl       deterministic seed list
  - eval/acoustic_metrics_zscore.json    diagnose_acoustic results
  - eval/acoustic_metrics_rms.json       threshold_baseline_acoustic results
  - eval/acoustic_confusion.txt          side-by-side confusion text

Honest scope: this eval is on SYNTHETIC MIMII-style fan clips. The acoustic
diagnostic was developed against the same synthetic generator (`generate_
synthetic_acoustic`), so the metrics below are necessarily optimistic — they
report internal-consistency, not cross-dataset generalisation. Pointing the
loader at a real MIMII fan dump (see `pdm_agent.acoustic.load_mimii_fan_dir`)
is the next step (Roadmap §2b in this MVP — out of session scope).
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable

import numpy as np

from pdm_agent.acoustic import AcousticSample, generate_synthetic_acoustic
from pdm_agent.acoustic_diagnostic import (
    AcousticBaseline,
    AcousticDiagnosis,
    diagnose_acoustic,
    fit_baseline,
    threshold_baseline_acoustic,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "eval" / "acoustic_eval_v1.jsonl"

# Deterministic eval set: 30 normal + 30 abnormal, all at SNR 12 dB (matches
# the sidecar/MCP default baseline SNR), seeds in [1000, 1060). Different
# range from the baseline-fit seeds [20260528, 20260540) so the eval is at
# least sample-disjoint from training. NOT held out at distribution level —
# that requires real MIMII.
EVAL_SEEDS_NORMAL = list(range(1000, 1030))
EVAL_SEEDS_ABNORMAL = list(range(1030, 1060))
BASELINE_SEEDS = list(range(20260528, 20260540))  # n=12, matches sidecar


def _build_eval_set() -> list[dict]:
    rows: list[dict] = []
    for seed in EVAL_SEEDS_NORMAL:
        rows.append({"id": f"acoustic-eval-normal-{seed}", "label": "normal", "seed": seed, "snr_db": 12.0})
    for seed in EVAL_SEEDS_ABNORMAL:
        rows.append({"id": f"acoustic-eval-abnormal-{seed}", "label": "abnormal", "seed": seed, "snr_db": 12.0})
    return rows


def _materialise_samples(rows: list[dict]) -> list[AcousticSample]:
    return [
        generate_synthetic_acoustic(r["label"], seed=r["seed"], snr_db=r["snr_db"])
        for r in rows
    ]


def _materialise_baseline() -> AcousticBaseline:
    pool = [generate_synthetic_acoustic("normal", seed=s, snr_db=12.0) for s in BASELINE_SEEDS]
    return fit_baseline(pool)


def _evaluate(samples: list[AcousticSample], fn: Callable[[AcousticSample], AcousticDiagnosis], method: str) -> dict:
    tp = fn_count = 0
    confusion = {
        "normal":   {"normal": 0, "abnormal": 0},
        "abnormal": {"normal": 0, "abnormal": 0},
    }
    per_case: list[dict] = []
    for s in samples:
        d = fn(s)
        actual = s.label
        pred = d.predicted_label
        confusion[actual][pred] += 1
        per_case.append({
            "id": s.sample_id,
            "actual": actual,
            "predicted": pred,
            "severity": d.severity,
            "anomaly_score": d.anomaly_score,
            "features": d.features.to_dict(),
        })

    # Treat "abnormal" as positive; classical 2x2 metrics
    tp = confusion["abnormal"]["abnormal"]
    fn_c = confusion["abnormal"]["normal"]
    fp = confusion["normal"]["abnormal"]
    tn = confusion["normal"]["normal"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn_c) if (tp + fn_c) else None
    # F1 must distinguish "metric undefined" (None) from "metric is zero".
    # Codex round-1 review caught that `if precision and recall else None`
    # silently maps precision=0 OR recall=0 to None, which would hide real
    # zero-performance cases on production MIMII data.
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / len(samples) if samples else 0.0
    return {
        "method": method,
        "n_samples": len(samples),
        "accuracy": round(acc, 4),
        "binary_anomaly_detection": {
            "true_positive": tp, "false_negative": fn_c,
            "false_positive": fp, "true_negative": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        },
        "confusion": confusion,
        "per_case": per_case,
    }


def _format_confusion(metrics: dict) -> str:
    confusion = metrics["confusion"]
    rows = ["actual \\ pred   normal      abnormal"]
    rows.append("-" * 36)
    for actual in ("normal", "abnormal"):
        n_norm = confusion[actual]["normal"]
        n_abn = confusion[actual]["abnormal"]
        rows.append(f"{actual:14s}  {n_norm:>6d}     {n_abn:>6d}")
    return "\n".join(rows)


def main() -> None:
    out_dir = ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_eval_set()
    EVAL_FILE.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    samples = _materialise_samples(rows)
    baseline = _materialise_baseline()
    print(f"Acoustic eval — {len(samples)} synthetic clips at SNR 12 dB")
    print(f"  baseline fit on {baseline.n_train} normal clips")
    print()

    # zscore-against-baseline
    zscore_metrics = _evaluate(
        samples,
        lambda s: diagnose_acoustic(s, baseline),
        method="acoustic-zscore-baseline-v1",
    )
    summary = {k: v for k, v in zscore_metrics.items() if k != "per_case"}
    (out_dir / "acoustic_metrics_zscore.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "acoustic_results_zscore.jsonl").write_text(
        "\n".join(json.dumps(c) for c in zscore_metrics["per_case"]) + "\n"
    )

    # RMS baseline
    rms_metrics = _evaluate(
        samples,
        lambda s: threshold_baseline_acoustic(s, rms_threshold=0.20),
        method="acoustic-rms-threshold-baseline",
    )
    rms_summary = {k: v for k, v in rms_metrics.items() if k != "per_case"}
    (out_dir / "acoustic_metrics_rms.json").write_text(json.dumps(rms_summary, indent=2))

    confusion_text = (
        f"# z-score baseline — accuracy {zscore_metrics['accuracy']:.3f}, "
        f"F1 {zscore_metrics['binary_anomaly_detection']['f1']}\n\n"
        + _format_confusion(zscore_metrics)
        + "\n\n# RMS threshold baseline — accuracy "
        f"{rms_metrics['accuracy']:.3f}, F1 {rms_metrics['binary_anomaly_detection']['f1']}\n\n"
        + _format_confusion(rms_metrics)
    )
    (out_dir / "acoustic_confusion.txt").write_text(confusion_text)

    print("z-score-against-baseline:")
    print(f"  accuracy = {zscore_metrics['accuracy']:.3f}")
    print(f"  F1       = {zscore_metrics['binary_anomaly_detection']['f1']}")
    print(f"  P / R    = {zscore_metrics['binary_anomaly_detection']['precision']} / "
          f"{zscore_metrics['binary_anomaly_detection']['recall']}")
    print()
    print("RMS threshold baseline (rms >= 0.20):")
    print(f"  accuracy = {rms_metrics['accuracy']:.3f}")
    print(f"  F1       = {rms_metrics['binary_anomaly_detection']['f1']}")
    print(f"  P / R    = {rms_metrics['binary_anomaly_detection']['precision']} / "
          f"{rms_metrics['binary_anomaly_detection']['recall']}")
    print()
    print(confusion_text)


if __name__ == "__main__":
    main()
