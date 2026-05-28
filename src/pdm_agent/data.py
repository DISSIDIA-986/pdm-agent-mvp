"""CWRU bearing vibration data: loader, synthetic generator, validation.

Honest scope note: this module supports two data sources:
1. CWRU Bearing Data Center .mat files (real measurements, electric-motor test rig)
2. Synthetic vibration generator (for CI/smoke tests + when network unavailable)

CWRU is used here as an *analog benchmark* for BESS auxiliary-equipment (pump/fan)
bearings — it is NOT validated BESS PdM data. See README License & Scope sections.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Iterable, Literal

import numpy as np

log = logging.getLogger(__name__)

# CWRU public download base (Drive Apron — Drive End bearings, 12kHz subset)
# Per https://engineering.case.edu/bearingdatacenter/download-data-file these
# files are individually downloadable .mat snapshots. We deliberately list a
# small curated subset for MVP scope.
CWRU_DOWNLOAD_INDEX: dict[str, dict] = {
    # filename -> {url, fault_class, fault_diameter_inches, load_hp, rpm_nominal, sha256_optional}
    # NB: CWRU URLs occasionally rotate; loader falls back to synthetic if 404.
    "97.mat": {
        "url": "https://engineering.case.edu/sites/default/files/97.mat",
        "fault_class": "normal",
        "fault_diameter_inches": 0.0,
        "load_hp": 0,
        "rpm_nominal": 1797,
    },
    "105.mat": {
        "url": "https://engineering.case.edu/sites/default/files/105.mat",
        "fault_class": "inner_race",
        "fault_diameter_inches": 0.007,
        "load_hp": 0,
        "rpm_nominal": 1797,
    },
    "118.mat": {
        "url": "https://engineering.case.edu/sites/default/files/118.mat",
        "fault_class": "ball",
        "fault_diameter_inches": 0.007,
        "load_hp": 0,
        "rpm_nominal": 1797,
    },
    "130.mat": {
        "url": "https://engineering.case.edu/sites/default/files/130.mat",
        "fault_class": "outer_race",
        "fault_diameter_inches": 0.007,
        "load_hp": 0,
        "rpm_nominal": 1797,
    },
}

FaultClass = Literal["normal", "inner_race", "ball", "outer_race"]
SAMPLE_RATE_HZ_DEFAULT = 12_000


@dataclasses.dataclass(frozen=True)
class VibrationSample:
    """A single labelled vibration window (raw time series + metadata)."""

    sample_id: str
    fault_class: FaultClass
    signal: np.ndarray  # shape (N,), float32
    sample_rate_hz: int
    rpm: float
    source: Literal["cwru", "synthetic"]
    fault_diameter_inches: float = 0.0
    load_hp: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {self.signal.shape}")
        if len(self.signal) < 1024:
            raise ValueError(f"signal too short: {len(self.signal)} samples")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.fault_class not in {"normal", "inner_race", "ball", "outer_race"}:
            raise ValueError(f"unknown fault_class {self.fault_class}")

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.sample_rate_hz

    def to_metadata(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "fault_class": self.fault_class,
            "sample_rate_hz": self.sample_rate_hz,
            "rpm": self.rpm,
            "source": self.source,
            "fault_diameter_inches": self.fault_diameter_inches,
            "load_hp": self.load_hp,
            "n_samples": int(len(self.signal)),
            "duration_s": self.duration_s,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Synthetic generator (deterministic, used for smoke tests and CI)
# ---------------------------------------------------------------------------

def _bearing_fault_frequencies(rpm: float, fault: FaultClass) -> list[float]:
    """Return canonical bearing fault frequencies for SKF 6205-2RS (CWRU drive-end).

    Reference: CWRU bearing geometry. Numbers are the standard CWRU formulas.
    """
    fr = rpm / 60.0
    bpfi = 5.4152 * fr  # ball-pass freq inner race
    bpfo = 3.5848 * fr  # ball-pass freq outer race
    bsf = 4.7135 * fr  # ball spin freq (×2 for ball defect harmonic visibility)
    match fault:
        case "inner_race":
            return [bpfi, 2 * bpfi]
        case "outer_race":
            return [bpfo, 2 * bpfo]
        case "ball":
            return [2 * bsf, 4 * bsf]
        case "normal":
            return []


def generate_synthetic(
    fault: FaultClass,
    rpm: float = 1797,
    duration_s: float = 1.0,
    sample_rate_hz: int = SAMPLE_RATE_HZ_DEFAULT,
    snr_db: float = 6.0,
    seed: int | None = None,
) -> VibrationSample:
    """Deterministic synthetic vibration signal injecting bearing fault harmonics.

    NOT meant for model training (too clean). Used for smoke tests and for
    confirming the diagnostic pipeline's plumbing works end-to-end.
    """
    rng = np.random.default_rng(seed if seed is not None else 0xC0FFEE)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz

    # Baseline rotational unbalance + cage noise
    fr = rpm / 60.0
    signal = 0.05 * np.sin(2 * np.pi * fr * t)

    # Inject fault harmonics
    fault_freqs = _bearing_fault_frequencies(rpm, fault)
    for f in fault_freqs:
        amp = 0.4 if fault != "ball" else 0.25
        signal += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        # Periodic impacts (impulses) at fault frequency to mimic real defects
        impulse_period = int(sample_rate_hz / f)
        impulse = np.zeros(n)
        impulse[::impulse_period] = 1.0
        # decaying impulse response (band-limited)
        kernel = np.exp(-np.arange(50) / 6) * np.sin(2 * np.pi * 2000 * np.arange(50) / sample_rate_hz)
        signal += 0.3 * np.convolve(impulse, kernel, mode="same")

    # Gaussian noise to hit target SNR
    signal_power = np.mean(signal ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(max(target_noise_power, 1e-9)), n)
    signal = (signal + noise).astype(np.float32)

    sample_id = "synth-" + hashlib.sha1(
        f"{fault}-{rpm}-{duration_s}-{sample_rate_hz}-{snr_db}-{seed}".encode()
    ).hexdigest()[:10]
    return VibrationSample(
        sample_id=sample_id,
        fault_class=fault,
        signal=signal,
        sample_rate_hz=sample_rate_hz,
        rpm=rpm,
        source="synthetic",
        notes=f"synthetic snr_db={snr_db} seed={seed}",
    )


# ---------------------------------------------------------------------------
# CWRU .mat loader (best-effort; falls back to synthetic in offline mode)
# ---------------------------------------------------------------------------

def load_cwru_mat(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Load drive-end accelerometer signal from a CWRU .mat file.

    CWRU .mat keys vary per file (e.g. X097_DE_time, X105_DE_time). We pick the
    first key ending in '_DE_time' (drive end) which is the standard convention.
    Returns (signal_float32, sample_rate_hz).
    """
    try:
        from scipy.io import loadmat  # local import to keep top-level fast
    except ImportError as e:
        raise RuntimeError("scipy is required to load .mat files") from e

    mat = loadmat(str(path), squeeze_me=True)
    de_keys = [k for k in mat if k.endswith("_DE_time") or k.endswith("DE_time")]
    if not de_keys:
        # Some normal-condition files use _BA_time (bearing accel) — fall back
        de_keys = [k for k in mat if k.endswith("BA_time")]
    if not de_keys:
        raise ValueError(f"no DE_time/BA_time key in {path.name}; keys={list(mat)}")
    raw = np.asarray(mat[de_keys[0]], dtype=np.float32).ravel()
    # CWRU drive-end was sampled at 12 kHz unless filename hints 48 kHz
    sr = 48_000 if "48k" in path.name.lower() else SAMPLE_RATE_HZ_DEFAULT
    return raw, sr


def download_cwru_subset(
    target_dir: pathlib.Path,
    filenames: Iterable[str] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, pathlib.Path]:
    """Download a curated CWRU .mat subset. Returns map filename -> local path.

    Failures are logged and skipped (caller can fall back to synthetic).
    """
    import httpx

    target_dir.mkdir(parents=True, exist_ok=True)
    filenames = list(filenames) if filenames else list(CWRU_DOWNLOAD_INDEX.keys())
    out: dict[str, pathlib.Path] = {}
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for fn in filenames:
            entry = CWRU_DOWNLOAD_INDEX.get(fn)
            if not entry:
                log.warning("skipping unknown CWRU file %s", fn)
                continue
            dest = target_dir / fn
            if dest.exists() and dest.stat().st_size > 1024:
                log.info("cached %s (%d bytes)", fn, dest.stat().st_size)
                out[fn] = dest
                continue
            try:
                log.info("downloading %s from %s", fn, entry["url"])
                r = client.get(entry["url"])
                r.raise_for_status()
                dest.write_bytes(r.content)
                log.info("wrote %s (%d bytes)", fn, len(r.content))
                out[fn] = dest
            except Exception as e:  # noqa: BLE001
                log.error("download failed for %s: %s", fn, e)
    return out


def load_cwru_dataset(raw_dir: pathlib.Path, window_s: float = 1.0) -> list[VibrationSample]:
    """Load every CWRU .mat in raw_dir into windowed VibrationSamples.

    Each .mat is sliced into non-overlapping windows of `window_s` seconds.
    """
    samples: list[VibrationSample] = []
    for fn, meta in CWRU_DOWNLOAD_INDEX.items():
        path = raw_dir / fn
        if not path.exists():
            log.debug("skip missing %s", fn)
            continue
        try:
            signal, sr = load_cwru_mat(path)
        except Exception as e:  # noqa: BLE001
            log.error("load failed for %s: %s", fn, e)
            continue
        win_n = int(window_s * sr)
        n_windows = len(signal) // win_n
        for i in range(n_windows):
            window = signal[i * win_n : (i + 1) * win_n]
            sid = f"cwru-{fn.replace('.mat', '')}-w{i:03d}"
            samples.append(
                VibrationSample(
                    sample_id=sid,
                    fault_class=meta["fault_class"],
                    signal=window,
                    sample_rate_hz=sr,
                    rpm=float(meta["rpm_nominal"]),
                    source="cwru",
                    fault_diameter_inches=float(meta["fault_diameter_inches"]),
                    load_hp=int(meta["load_hp"]),
                    notes=f"cwru file={fn}",
                )
            )
    log.info("loaded %d CWRU windows from %s", len(samples), raw_dir)
    return samples


# ---------------------------------------------------------------------------
# Quick validation utility used by smoke tests
# ---------------------------------------------------------------------------

def validate_samples(samples: list[VibrationSample]) -> dict:
    """Return basic stats; raise on bad data."""
    if not samples:
        raise ValueError("empty sample list")
    by_class: dict[str, int] = {}
    for s in samples:
        by_class[s.fault_class] = by_class.get(s.fault_class, 0) + 1
        if not np.all(np.isfinite(s.signal)):
            raise ValueError(f"non-finite values in {s.sample_id}")
    return {
        "n_samples": len(samples),
        "by_class": by_class,
        "sources": sorted({s.source for s in samples}),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    here = pathlib.Path(__file__).resolve().parents[2]
    raw = here / "data" / "raw"
    downloaded = download_cwru_subset(raw)
    print(json.dumps({k: str(v) for k, v in downloaded.items()}, indent=2))
    samples = load_cwru_dataset(raw)
    if not samples:
        # Fall back to synthetic so the pipeline is still demonstrable offline.
        print("No CWRU data available — generating synthetic samples")
        for fc in ("normal", "inner_race", "ball", "outer_race"):
            samples.append(generate_synthetic(fc, seed=hash(fc) & 0xFFFF))  # type: ignore[arg-type]
    print(json.dumps(validate_samples(samples), indent=2))
