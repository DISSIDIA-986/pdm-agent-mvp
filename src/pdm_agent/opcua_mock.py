"""Mock OPC UA server publishing simulated vibration windows for a fictional asset.

Why mock instead of real PLC: clearly inside MVP scope. Real OT integration
would require: vendor-specific gateway, certificates, network access, and a
physical sensor. The mock asset publishes deterministic vibration windows so
the agent runtime can be exercised end-to-end on any laptop.

Asset model:
  Objects/
    Microgrid/
      BESS_Site_A/
        CoolingPump_01/
          Vibration_Window         (variable: float[] — last window)
          Vibration_RMS            (variable: float — convenience scalar)
          RPM                      (variable: float)
          SampleRate               (variable: int)
          OperatingMode            (variable: str)
          UpdateCounter            (variable: int)

Honest scope note: this is a *simulation* — no real BESS data, no field PLC.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Sequence

from asyncua import Server, ua

from .data import VibrationSample, generate_synthetic

log = logging.getLogger(__name__)

# Default scenario: cycles through fault classes so a poll loop sees variety.
DEFAULT_SCENARIO: list[str] = [
    "normal",
    "normal",
    "normal",
    "watch_inner",
    "inner_race",
    "inner_race",
    "outer_race",
    "normal",
    "ball",
    "normal",
]


def _scenario_sample(label: str, tick: int) -> VibrationSample:
    """Generate a deterministic sample for a scenario step.

    `watch_inner` injects a borderline (low-SNR) inner race fault to exercise
    the diagnostic's severity bucketing.
    """
    seed = tick * 1000
    if label == "watch_inner":
        return generate_synthetic("inner_race", duration_s=1.0, snr_db=4.0, seed=seed)
    # snr_db=15 keeps the scenario clearly above the diagnostic's family-score
    # threshold so the demo deterministically traverses each fault branch.
    return generate_synthetic(label, duration_s=1.0, snr_db=15.0, seed=seed)  # type: ignore[arg-type]


class MockOpcUaServer:
    def __init__(
        self,
        endpoint: str = "opc.tcp://127.0.0.1:4842/pdm-agent-mock",
        scenario: Sequence[str] | None = None,
        tick_interval_s: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.scenario = list(scenario) if scenario else list(DEFAULT_SCENARIO)
        self.tick_interval_s = tick_interval_s
        self._server: Server | None = None
        self._task: asyncio.Task | None = None
        self._tick = 0
        # node references populated in init()
        self._n_window = None
        self._n_rms = None
        self._n_rpm = None
        self._n_sample_rate = None
        self._n_mode = None
        self._n_counter = None

    async def init(self) -> None:
        self._server = Server()
        await self._server.init()
        self._server.set_endpoint(self.endpoint)
        self._server.set_server_name("pdm-agent-mvp Mock OPC UA Server")
        # Anonymous + NoSecurity for local dev only. Production deployments
        # MUST use Basic256Sha256 with X.509 certificates (or stronger) and
        # username/password or certificate-based user identity. See README
        # "Production deployment checklist" §1.
        self._server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        ns_uri = "https://github.com/pdm-agent-mvp/mock"
        idx = await self._server.register_namespace(ns_uri)

        objects = self._server.nodes.objects
        microgrid = await objects.add_object(idx, "Microgrid")
        site = await microgrid.add_object(idx, "BESS_Site_A")
        pump = await site.add_object(idx, "CoolingPump_01")

        # Window starts as 1 second of zeros so clients always get well-formed data
        zero_window = [0.0] * 12_000
        self._n_window = await pump.add_variable(idx, "Vibration_Window", zero_window)
        self._n_rms = await pump.add_variable(idx, "Vibration_RMS", 0.0)
        self._n_rpm = await pump.add_variable(idx, "RPM", 1797.0)
        self._n_sample_rate = await pump.add_variable(idx, "SampleRate", 12_000)
        self._n_mode = await pump.add_variable(idx, "OperatingMode", "RUNNING")
        self._n_counter = await pump.add_variable(idx, "UpdateCounter", 0)
        for n in (self._n_window, self._n_rms, self._n_rpm, self._n_sample_rate, self._n_mode, self._n_counter):
            await n.set_writable()
        log.info("mock OPC UA initialised at %s", self.endpoint)

    async def _tick_loop(self) -> None:
        assert self._server is not None
        try:
            while True:
                label = self.scenario[self._tick % len(self.scenario)]
                sample = _scenario_sample(label, self._tick)
                rms = float((sample.signal ** 2).mean() ** 0.5)
                await self._n_window.write_value(sample.signal.astype(float).tolist())
                await self._n_rms.write_value(rms)
                await self._n_counter.write_value(self._tick)
                await self._n_mode.write_value("RUNNING" if label == "normal" else "RUNNING_FAULT_HINT")
                log.debug("tick %d label=%s rms=%.3f", self._tick, label, rms)
                self._tick += 1
                await asyncio.sleep(self.tick_interval_s)
        except asyncio.CancelledError:
            log.info("mock tick loop cancelled at tick %d", self._tick)
            raise

    async def start(self) -> None:
        if self._server is None:
            await self.init()
        await self._server.start()
        self._task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._server is not None:
            await self._server.stop()


async def amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    srv = MockOpcUaServer()
    await srv.start()
    log.info("mock OPC UA running — Ctrl+C to stop")
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await srv.stop()


if __name__ == "__main__":
    asyncio.run(amain())
