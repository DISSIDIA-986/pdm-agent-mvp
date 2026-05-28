"""Measure end-to-end HTTP latency of the sidecar diagnose endpoint.

Run with the sidecar already up (uvicorn pdm_agent.sidecar:app --port 8200).
Reports p50 / p95 / p99 over N requests for two payload sizes.
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx
import numpy as np

from pdm_agent.data import generate_synthetic


def measure(url: str, n: int, duration_s: float) -> dict:
    sample = generate_synthetic("inner_race", duration_s=duration_s, snr_db=10.0, seed=999)
    payload = {
        "sample_id": sample.sample_id,
        "signal": sample.signal.astype(float).tolist(),
        "sample_rate_hz": sample.sample_rate_hz,
        "rpm": sample.rpm,
        "fault_class": "normal",  # we don't trust client's label anyway
    }
    latencies: list[float] = []
    server_latencies: list[float] = []
    with httpx.Client(timeout=30) as c:
        # Warmup
        for _ in range(3):
            r = c.post(f"{url}/v1/diagnose", json=payload)
            r.raise_for_status()
        for _ in range(n):
            t0 = time.perf_counter()
            r = c.post(f"{url}/v1/diagnose", json=payload)
            r.raise_for_status()
            wall = (time.perf_counter() - t0) * 1000
            latencies.append(wall)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "window_duration_s": duration_s,
        "n_samples_per_request": len(sample.signal),
        "n_requests": n,
        "client_p50_ms": statistics.median(latencies),
        "client_p95_ms": float(np.percentile(latencies, 95)),
        "client_p99_ms": float(np.percentile(latencies, 99)),
        "server_p50_ms": statistics.median(server_latencies),
        "server_p95_ms": float(np.percentile(server_latencies, 95)),
        "http_overhead_p50_ms": statistics.median(latencies) - statistics.median(server_latencies),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    import json
    for dur in (0.5, 1.0, 2.0):
        result = measure(args.url, args.n, dur)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
