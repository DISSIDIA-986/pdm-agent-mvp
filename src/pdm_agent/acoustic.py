"""Acoustic-modality ingestion: MIMII fan loader + deterministic synthetic generator.

This module is the second-modality counterpart to `pdm_agent.data` (vibration).
We deliberately mirror its shape — `AcousticSample` is the analog of
`VibrationSample`; `generate_synthetic_acoustic()` matches `generate_synthetic()`;
`load_mimii_fan_dir()` matches `load_cwru_dataset()`. Same engineering
patterns, different signal.

Honest scope (read first):
  - MIMII (https://zenodo.org/records/3384388) targets industrial machine
    sound: fan, pump, slider, valve. We use the **fan** subset only.
  - The MIMII fan recordings are made on a small 4-pole induction motor +
    centrifugal fan rig — they are an **analog benchmark** for a microgrid
    inverter cooling fan, NOT a vendor-validated inverter model.
  - The dataset is ~10 GB total per SNR. We do not auto-download. Instead
    the loader points at a directory the user populates manually (or via
    `scripts/download_mimii_fan.sh`) and falls back to synthetic data
    when the directory is empty — same pattern as the CWRU loader.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
from typing import Iterable, Literal

import numpy as np

log = logging.getLogger(__name__)

# MIMII canonical settings
MIMII_SAMPLE_RATE_HZ = 16_000
MIMII_CLIP_DURATION_S = 10.0

# Public Zenodo location of the fan SNR=+6 dB subset (the easiest signal-to-
# detect band). Documented for users who want to populate data/raw/mimii/.
MIMII_FAN_PLUS6DB_URL = (
    "https://zenodo.org/api/records/3384388/files/6_dB_fan.zip/content"
)
MIMII_FAN_PLUS6DB_SIZE_GB = 9.7

AcousticLabel = Literal["normal", "abnormal"]


@dataclasses.dataclass(frozen=True)
class AcousticSample:
    """A single labelled acoustic clip (~10s, 16 kHz mono)."""

    sample_id: str
    label: AcousticLabel
    signal: np.ndarray  # shape (N,), float32, range roughly ±1
    sample_rate_hz: int
    machine_id: str  # e.g. "id_00" — controls speaker variation
    source: Literal["mimii", "synthetic"]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {self.signal.shape}")
        if len(self.signal) < 4 * self.sample_rate_hz:  # < 4 s rejected
            raise ValueError(f"signal too short: {len(self.signal)} samples ({len(self.signal)/self.sample_rate_hz:.2f}s)")
        if self.label not in ("normal", "abnormal"):
            raise ValueError(f"unknown label {self.label}")

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.sample_rate_hz

    def to_metadata(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "sample_rate_hz": self.sample_rate_hz,
            "machine_id": self.machine_id,
            "source": self.source,
            "n_samples": int(len(self.signal)),
            "duration_s": self.duration_s,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Synthetic fan generator (deterministic — CI + offline development)
# ---------------------------------------------------------------------------

def _fan_carrier(t: np.ndarray, *, rpm: float, n_blades: int) -> np.ndarray:
    """Periodic fan rotation tone + first blade-pass harmonic."""
    fr = rpm / 60.0  # shaft rotation
    bpf = fr * n_blades  # blade-pass frequency
    return (
        0.12 * np.sin(2 * np.pi * fr * t)
        + 0.30 * np.sin(2 * np.pi * bpf * t)
        + 0.18 * np.sin(2 * np.pi * 2 * bpf * t)
    )


def generate_synthetic_acoustic(
    label: AcousticLabel,
    *,
    rpm: float = 2900.0,  # typical small AC fan
    n_blades: int = 7,
    duration_s: float = MIMII_CLIP_DURATION_S,
    sample_rate_hz: int = MIMII_SAMPLE_RATE_HZ,
    snr_db: float = 6.0,
    seed: int | None = None,
    machine_id: str = "id_synth",
) -> AcousticSample:
    """Deterministic synthetic fan acoustic signal.

    `label == "abnormal"` injects:
      1. A 1-3 kHz broadband noise burst (~bearing/bushing rub)
      2. A slow amplitude modulation at ~0.5 Hz (load imbalance)
    These mimic — but do NOT replicate — the failure modes MIMII labels
    abnormal. Used for smoke tests + offline pipeline development.
    """
    rng = np.random.default_rng(seed if seed is not None else 0xCAFE)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    carrier = _fan_carrier(t, rpm=rpm, n_blades=n_blades).astype(np.float64)
    base_noise = 0.04 * rng.standard_normal(n)
    signal = carrier + base_noise

    if label == "abnormal":
        # 1. Broadband 1-3 kHz noise burst (bearing-like)
        broadband = rng.standard_normal(n)
        # Simple FIR-ish filter via FFT band-pass to 1-3 kHz
        spec = np.fft.rfft(broadband)
        freqs = np.fft.rfftfreq(n, 1 / sample_rate_hz)
        mask = (freqs >= 1000) & (freqs <= 3000)
        spec[~mask] = 0
        broadband_band = np.fft.irfft(spec, n=n)
        signal += 0.25 * broadband_band
        # 2. Slow 0.5 Hz AM (load imbalance / belt slip)
        am_envelope = 1.0 + 0.35 * np.sin(2 * np.pi * 0.5 * t)
        signal *= am_envelope

    # Add Gaussian noise to hit SNR target relative to carrier power
    signal_power = np.mean(carrier ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(max(target_noise_power, 1e-9)), n)
    out = (signal + noise).astype(np.float32)
    # Clip to ±1 (matches int16 WAV range when normalised)
    np.clip(out, -1.0, 1.0, out=out)

    sample_id = "synth-" + hashlib.sha1(
        f"{label}-{rpm}-{n_blades}-{duration_s}-{sample_rate_hz}-{snr_db}-{seed}".encode()
    ).hexdigest()[:10]
    return AcousticSample(
        sample_id=sample_id,
        label=label,
        signal=out,
        sample_rate_hz=sample_rate_hz,
        machine_id=machine_id,
        source="synthetic",
        notes=f"synthetic snr_db={snr_db} rpm={rpm} seed={seed}",
    )


# ---------------------------------------------------------------------------
# MIMII fan loader (best-effort; falls back to synthetic in offline mode)
# ---------------------------------------------------------------------------

def load_mimii_wav(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Load a MIMII fan .wav. MIMII clips are 16 kHz, 16-bit, mono.

    We use the stdlib `wave` module so we don't pull in an extra dep just for
    MIMII. Returns (signal_float32_pm1, sample_rate_hz).
    """
    import wave

    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sample_width != 2:
        raise ValueError(f"{path.name}: expected 16-bit PCM, got sample_width={sample_width}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm, sample_rate


def load_mimii_fan_dir(
    fan_root: pathlib.Path,
    *,
    machine_ids: Iterable[str] | None = None,
    clip_duration_s: float = MIMII_CLIP_DURATION_S,
) -> list[AcousticSample]:
    """Walk a MIMII fan directory and return AcousticSamples.

    Expected layout (matches the official Zenodo zip after extraction):

        fan_root/
          fan/
            id_00/
              normal/   *.wav  (~991 clips)
              abnormal/ *.wav  (~407 clips)
            id_02/ ...
            id_04/ ...
            id_06/ ...

    If `fan_root` is missing or empty, returns []; callers should fall back
    to `generate_synthetic_acoustic` for development. Bug-tolerant: skips
    any .wav whose header we can't read rather than crashing the load.
    """
    if not fan_root.exists():
        log.info("MIMII fan_root %s missing — returning empty list", fan_root)
        return []
    samples: list[AcousticSample] = []
    # Allow `fan_root` to be either the parent (.../+6dB) or the fan-specific dir
    fan_dir = fan_root / "fan" if (fan_root / "fan").is_dir() else fan_root
    if not fan_dir.is_dir():
        log.info("MIMII fan dir not found under %s", fan_root)
        return []
    targets = list(fan_dir.iterdir()) if machine_ids is None else [fan_dir / m for m in machine_ids]
    for machine_dir in sorted(targets):
        if not machine_dir.is_dir():
            continue
        for label in ("normal", "abnormal"):
            label_dir = machine_dir / label
            if not label_dir.is_dir():
                continue
            for wav_path in sorted(label_dir.glob("*.wav")):
                try:
                    signal, sr = load_mimii_wav(wav_path)
                except Exception as e:  # noqa: BLE001
                    log.warning("skip unreadable %s: %s", wav_path.name, e)
                    continue
                target_len = int(clip_duration_s * sr)
                if len(signal) < int(4 * sr):
                    continue  # too short to be useful
                signal = signal[:target_len]
                samples.append(
                    AcousticSample(
                        sample_id=f"mimii-{machine_dir.name}-{label}-{wav_path.stem}",
                        label=label,  # type: ignore[arg-type]
                        signal=signal.astype(np.float32),
                        sample_rate_hz=int(sr),
                        machine_id=machine_dir.name,
                        source="mimii",
                        notes=f"mimii path={wav_path.name}",
                    )
                )
    log.info("loaded %d MIMII fan samples from %s", len(samples), fan_root)
    return samples


def validate_acoustic_samples(samples: list[AcousticSample]) -> dict:
    """Return basic stats + raise on bad data — mirrors `data.validate_samples`."""
    if not samples:
        raise ValueError("empty acoustic sample list")
    by_label: dict[str, int] = {}
    by_machine: dict[str, int] = {}
    for s in samples:
        by_label[s.label] = by_label.get(s.label, 0) + 1
        by_machine[s.machine_id] = by_machine.get(s.machine_id, 0) + 1
        if not np.all(np.isfinite(s.signal)):
            raise ValueError(f"non-finite values in {s.sample_id}")
    return {
        "n_samples": len(samples),
        "by_label": by_label,
        "by_machine": by_machine,
        "sources": sorted({s.source for s in samples}),
    }
