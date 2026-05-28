"""Render evaluation + diagnostic figures used in the README hero area.

Outputs (PNG, 150 dpi, embedded in README):
  docs/figures/confusion.png       diagnose v2 vs baseline side-by-side
  docs/figures/envelope_demo.png   one inner-race CWRU window: raw + envelope spectrum

These are deterministic given a fixed eval set and fixed CWRU download —
re-running the script produces identical pixels. Safe to commit.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from pdm_agent.data import load_cwru_dataset, _bearing_fault_frequencies
from pdm_agent.diagnostic import envelope_spectrum

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["normal", "inner_race", "outer_race", "ball"]


def _load_confusion(method: str) -> dict[str, dict[str, int]]:
    metrics_path = ROOT / "eval" / f"metrics_{method}.json"
    metrics = json.loads(metrics_path.read_text())
    return metrics["confusion"]


def _confusion_to_matrix(confusion: dict[str, dict[str, int]]) -> np.ndarray:
    cols = CLASSES + (["fault"] if any("fault" in v for v in confusion.values()) else [])
    matrix = np.zeros((len(CLASSES), len(cols)), dtype=int)
    for i, actual in enumerate(CLASSES):
        for j, pred in enumerate(cols):
            matrix[i, j] = confusion.get(actual, {}).get(pred, 0)
    return matrix, cols


def render_confusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, method, title in zip(
        axes,
        ("diagnose", "baseline"),
        ("Diagnose v2 (envelope-spectrum-v2-family)", "RMS-threshold baseline"),
    ):
        confusion = _load_confusion(method)
        matrix, cols = _confusion_to_matrix(confusion)
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=25, ha="right")
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title, fontsize=11)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if v > 0:
                    on_diag = (i < len(CLASSES) and j < len(CLASSES) and CLASSES[i] == cols[j])
                    color = "white" if v > matrix.max() / 2 else "black"
                    ax.text(j, i, str(v), ha="center", va="center",
                            color=color, fontweight="bold" if on_diag else "normal",
                            fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("43-case CWRU eval — diagnose v2 vs baseline", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_envelope_demo() -> None:
    """Show one inner-race CWRU window: raw waveform + envelope spectrum + fault markers."""
    raw_dir = ROOT / "data" / "raw"
    samples = load_cwru_dataset(raw_dir, window_s=1.0)
    inner = next((s for s in samples if s.fault_class == "inner_race"), None)
    if inner is None:
        print("no inner-race CWRU window available — skipping envelope demo")
        return
    sr = inner.sample_rate_hz
    n = len(inner.signal)
    t = np.arange(n) / sr
    freqs, spectrum = envelope_spectrum(inner.signal, sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5))
    ax1.plot(t, inner.signal, linewidth=0.5, color="#1f77b4")
    ax1.set_title(f"Raw vibration — CWRU inner-race fault @ {inner.rpm:.0f} RPM", fontsize=11)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("acceleration")
    ax1.grid(alpha=0.3)

    cutoff_hz = 600
    mask = freqs <= cutoff_hz
    ax2.plot(freqs[mask], spectrum[mask], linewidth=0.7, color="#1f77b4", label="envelope spectrum")
    bpfi = _bearing_fault_frequencies(inner.rpm, "inner_race")
    bpfo = _bearing_fault_frequencies(inner.rpm, "outer_race")
    bsf = _bearing_fault_frequencies(inner.rpm, "ball")
    ymax = float(spectrum[mask].max())
    for f, label, color in [
        (bpfi[0], "BPFI", "#d62728"),
        (bpfo[0], "BPFO", "#2ca02c"),
        (bsf[0], "2×BSF", "#9467bd"),
    ]:
        if f <= cutoff_hz:
            ax2.axvline(f, color=color, linestyle="--", alpha=0.6)
            ax2.text(f, ymax * 0.92, f" {label}\n {f:.0f}Hz", color=color, fontsize=9, va="top")
    ax2.set_title("Envelope spectrum with theoretical fault frequencies", fontsize=11)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("magnitude")
    ax2.set_xlim(0, cutoff_hz)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "envelope_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_workflow_trace() -> None:
    """ASCII-art–style state machine card. Generated as a clean PNG so the
    README hero is consistent across GitHub renderers."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    # Box positions
    nodes = [
        ("OPC UA\nVibration_Window", 0.05, 0.5, "#e8f0fe"),
        ("Diagnose\n(envelope v2)", 0.27, 0.5, "#fff1c4"),
        ("Severity\nrouter", 0.48, 0.5, "#fff1c4"),
        ("Draft\nWork Order", 0.66, 0.5, "#ffe3e3"),
        ("interrupt()\nHuman approval", 0.84, 0.5, "#ffd1d1"),
    ]
    for label, x, y, color in nodes:
        ax.add_patch(plt.Rectangle((x - 0.07, y - 0.12), 0.14, 0.24,
                                    facecolor=color, edgecolor="#444", linewidth=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, fontweight="bold")
    for x0, x1 in [(0.12, 0.20), (0.34, 0.41), (0.55, 0.59), (0.73, 0.77)]:
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    # Audit log spur from "Draft" and "Human approval" boxes
    for x in (0.66, 0.84):
        ax.annotate("", xy=(x, 0.18), xytext=(x, 0.38),
                    arrowprops=dict(arrowstyle="->", color="#888", linestyle=":", lw=1.0))
    ax.add_patch(plt.Rectangle((0.55, 0.05), 0.40, 0.13,
                                facecolor="#f0f0f0", edgecolor="#444", linewidth=1.0))
    ax.text(0.75, 0.115, "SQLite WAL — work_orders + audit_log",
            ha="center", va="center", fontsize=9.5)

    # Auto-resolve path for severity == normal
    ax.annotate("", xy=(0.55, 0.82), xytext=(0.48, 0.62),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
    ax.text(0.56, 0.84, "severity == normal\n→ no action", fontsize=8.5, color="#2ca02c")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("LangGraph runtime: ingest → diagnose → route → human-approval gate → audit",
                 fontsize=12, pad=12)
    out = FIG_DIR / "workflow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    render_confusion()
    render_envelope_demo()
    render_workflow_trace()


if __name__ == "__main__":
    main()
