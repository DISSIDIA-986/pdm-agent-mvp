"""End-to-end: mock OPC UA server publishing → client polling → workflow → audit trail.

This is the smoke test that proves the closed loop functions on a real
socket, not just in-process calls.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from pdm_agent.opcua_client import poll_once
from pdm_agent.opcua_mock import MockOpcUaServer
from pdm_agent.workflow import build_workflow, resume_with_decision, run_until_decision
from pdm_agent.workorder import WorkOrderStore


@asynccontextmanager
async def running_mock(scenario: list[str], port: int):
    endpoint = f"opc.tcp://127.0.0.1:{port}/pdm-agent-mock-test"
    srv = MockOpcUaServer(endpoint=endpoint, scenario=scenario, tick_interval_s=0.2)
    await srv.start()
    # Wait for at least one tick
    await asyncio.sleep(0.5)
    try:
        yield endpoint
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_e2e_inner_race_through_full_loop(tmp_path) -> None:
    """An inner-race scenario step should produce a pending_approval work order
    after one OPC UA poll → workflow run."""
    async with running_mock(scenario=["inner_race", "inner_race", "inner_race"], port=4843) as endpoint:
        snap = await poll_once(endpoint)
    assert snap.sample.signal.shape[0] > 1024
    # Run workflow (sync — LangGraph itself is sync-compatible)
    store = WorkOrderStore(tmp_path / "e2e.db")
    graph = build_workflow(store)
    out = run_until_decision(graph, asset_id="Microgrid/BESS_Site_A/CoolingPump_01",
                             sample=snap.sample, thread_id="e2e-inner")
    assert "interrupt" in out
    wo_id = out["interrupt"]["work_order_id"]
    assert store.get(wo_id).status == "pending_approval"
    # Human approves
    resume_with_decision(graph, thread_id="e2e-inner", approve=True,
                         decided_by="human:e2e-test", note="end-to-end test")
    wo = store.get(wo_id)
    assert wo.status == "approved"
    trail = store.audit_trail(wo_id)
    assert [e["event"] for e in trail] == ["created", "approval_requested", "approved"]


@pytest.mark.asyncio
async def test_e2e_normal_scenario_no_workorder(tmp_path) -> None:
    async with running_mock(scenario=["normal", "normal", "normal"], port=4844) as endpoint:
        snap = await poll_once(endpoint)
    store = WorkOrderStore(tmp_path / "e2e2.db")
    graph = build_workflow(store)
    out = run_until_decision(graph, asset_id="A1", sample=snap.sample, thread_id="e2e-normal")
    assert "final" in out
    assert out["final"]["final_status"] == "normal_no_action"
    assert store.list_by_status() == []


@pytest.mark.asyncio
async def test_opcua_publish_advances_counter(tmp_path) -> None:
    """Mock server should increment UpdateCounter every tick."""
    async with running_mock(scenario=["normal"] * 5, port=4845) as endpoint:
        snap1 = await poll_once(endpoint)
        await asyncio.sleep(0.5)
        snap2 = await poll_once(endpoint)
    assert snap2.update_counter > snap1.update_counter
