"""Quick sanity check: run diagnose() on real CWRU windows and print confusion table.

Not the formal evaluation — that's eval/run_eval.py in P4. This is a smoke test
to confirm the diagnostic pipeline is plausibly working on real measurements
before we wire it into the agent runtime.
"""
from __future__ import annotations

import pathlib
from collections import Counter, defaultdict

from pdm_agent.data import load_cwru_dataset
from pdm_agent.diagnostic import diagnose

HERE = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    raw = HERE / "data" / "raw"
    samples = load_cwru_dataset(raw, window_s=1.0)
    if not samples:
        print("No CWRU samples found — did you run `python -m pdm_agent.data` first?")
        return
    confusion: dict[str, Counter] = defaultdict(Counter)
    severities: Counter = Counter()
    for s in samples:
        d = diagnose(s)
        confusion[s.fault_class][d.predicted_class] += 1
        severities[d.severity] += 1
    print(f"Total CWRU windows: {len(samples)}")
    print(f"Severity counts:    {dict(severities)}\n")
    classes = ["normal", "inner_race", "outer_race", "ball"]
    header = "actual \\ pred".ljust(15) + "".join(c.ljust(13) for c in classes)
    print(header)
    print("-" * len(header))
    for actual in classes:
        row = actual.ljust(15)
        for pred in classes:
            row += str(confusion[actual].get(pred, 0)).ljust(13)
        print(row)
    correct = sum(confusion[c][c] for c in classes)
    print(f"\nNaive accuracy: {correct}/{len(samples)} = {correct/len(samples):.1%}")


if __name__ == "__main__":
    main()
