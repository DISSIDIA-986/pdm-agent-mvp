"""Async OPC UA client that polls vibration windows from the mock server.

Used by the LangGraph workflow as the "ingest" tool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from asyncua import Client

from .data import VibrationSample

log = logging.getLogger(__name__)


@dataclass
class OpcSnapshot:
    sample: VibrationSample
    update_counter: int
    operating_mode: str


async def _find_child_by_browse_name(parent, name: str):
    for n in await parent.get_children():
        bn = (await n.read_browse_name()).Name
        if bn == name:
            return n
    raise RuntimeError(f"node {name!r} not found under {parent}")


async def poll_once(
    endpoint: str = "opc.tcp://127.0.0.1:4842/pdm-agent-mock",
    *,
    site_name: str = "BESS_Site_A",
    asset_name: str = "CoolingPump_01",
) -> OpcSnapshot:
    """One-shot read of the latest vibration window from a named asset.

    Path is resolved by browse-name lookup at each level: Objects/Microgrid/<site>/<asset>.
    This avoids the fragility of "first child" indexing if the server's node
    order changes between deployments.
    """
    async with Client(url=endpoint) as client:
        objects = client.nodes.objects
        microgrid = await _find_child_by_browse_name(objects, "Microgrid")
        site = await _find_child_by_browse_name(microgrid, site_name)
        pump = await _find_child_by_browse_name(site, asset_name)
        children = await pump.get_children()
        readings: dict[str, object] = {}
        for n in children:
            bn = (await n.read_browse_name()).Name
            readings[bn] = await n.read_value()
        signal = np.asarray(readings["Vibration_Window"], dtype=np.float32)
        sample = VibrationSample(
            sample_id=f"opc-tick-{readings['UpdateCounter']:06d}",
            fault_class="normal",  # actual label unknown to the agent
            signal=signal,
            sample_rate_hz=int(readings["SampleRate"]),
            rpm=float(readings["RPM"]),
            source="synthetic",
            notes=f"polled via OPC UA, mode={readings['OperatingMode']}",
        )
        return OpcSnapshot(
            sample=sample,
            update_counter=int(readings["UpdateCounter"]),
            operating_mode=str(readings["OperatingMode"]),
        )
