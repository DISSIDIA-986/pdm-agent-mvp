"""Build a deterministic 43-case evaluation set from the downloaded CWRU subset.

The 4 curated CWRU files (97/105/118/130) yield exactly:
  - 97.mat  (normal, ~3.9 MB)       -> 20 windows of 1 second @ 12 kHz
  - 105.mat (inner_race, 0.007")    -> 10 windows
  - 118.mat (ball, 0.007")          -> 10 windows
  - 130.mat (outer_race, 0.007")    -> 10 windows

We subsample normal down to 13 (keeps class balance reasonable) and take all
10 fault windows, giving 13 + 10 + 10 + 10 = 43 cases. Shuffled with a fixed
seed for reproducibility. To extend, add files to data.CWRU_DOWNLOAD_INDEX
and bump CASES_PER_CLASS targets here.

Layout (eval_v1.jsonl):
  one line per case: {"id", "fault_class", "source_file", "window_idx",
                       "sample_rate_hz", "rpm", "notes"}

We mix all 4 conditions present in data/raw (97/105/118/130 -> normal/inner/ball/outer),
take 12-13 windows from each, and shuffle deterministically (seed=2026) so the
distribution is balanced and reproducible.
"""
from __future__ import annotations

import json
import pathlib
import random

from pdm_agent.data import load_cwru_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "eval_v1.jsonl"

# Target per class. Capped by what the curated CWRU subset actually yields:
# fault files give exactly 10 windows each, so 'normal' is the only knob.
CASES_PER_CLASS = {"normal": 13, "inner_race": 10, "outer_race": 10, "ball": 10}  # = 43


def main() -> None:
    raw = ROOT / "data" / "raw"
    samples = load_cwru_dataset(raw, window_s=1.0)
    by_class: dict[str, list] = {}
    for s in samples:
        by_class.setdefault(s.fault_class, []).append(s)
    rng = random.Random(2026)
    selected = []
    for cls, target in CASES_PER_CLASS.items():
        pool = by_class.get(cls, [])
        if not pool:
            print(f"WARNING: no samples for class {cls} — skipping (will produce undersized eval)")
            continue
        # If we have fewer than target, take all; else random subsample
        if len(pool) <= target:
            chosen = pool
        else:
            rng.shuffle(pool)
            chosen = pool[:target]
        selected.extend(chosen)
    rng.shuffle(selected)
    OUT.parent.mkdir(exist_ok=True, parents=True)
    with OUT.open("w") as f:
        for s in selected:
            row = {
                "id": s.sample_id,
                "fault_class": s.fault_class,
                "source_file": s.notes.split("file=")[1] if "file=" in s.notes else "",
                "sample_rate_hz": s.sample_rate_hz,
                "rpm": s.rpm,
                "n_samples": int(len(s.signal)),
                "fault_diameter_inches": s.fault_diameter_inches,
                "load_hp": s.load_hp,
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(selected)} cases to {OUT}")
    print({c: sum(1 for s in selected if s.fault_class == c) for c in CASES_PER_CLASS})


if __name__ == "__main__":
    main()
