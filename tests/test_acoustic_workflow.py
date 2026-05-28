"""End-to-end acoustic workflow tests.

The acoustic state graph shares the WorkOrderStore + audit_log with the
vibration graph — these tests verify that:
  - A normal acoustic clip ends without a work order
  - An abnormal acoustic clip pauses for human approval
  - Human decision lands in the same audit_log as the vibration path
  - Vibration + acoustic incidents on the SAME asset live in the same store
"""
from __future__ import annotations

import pytest

from pdm_agent.acoustic import generate_synthetic_acoustic
from pdm_agent.acoustic_diagnostic import fit_baseline
from pdm_agent.data import generate_synthetic
from pdm_agent.workflow import (
    build_acoustic_workflow,
    build_workflow,
    resume_with_decision,
    run_until_decision,
)
from pdm_agent.workorder import WorkOrderStore


@pytest.fixture
def shared_store(tmp_path):
    """Single store representing one asset's incident history."""
    return WorkOrderStore(tmp_path / "shared.db")


def _build_baseline(seed_base: int = 1000):
    pool = [generate_synthetic_acoustic("normal", seed=seed_base + i, snr_db=12.0) for i in range(10)]
    return fit_baseline(pool)


def _run_acoustic_until_decision(graph, *, asset_id, sample, baseline, thread_id):
    """The acoustic workflow has the extra `baseline` field — small wrapper."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"asset_id": asset_id, "sample": sample, "baseline": baseline},
        config=config,
    )
    state = graph.get_state(config)
    if state.tasks and any(t.interrupts for t in state.tasks):
        intr = state.tasks[0].interrupts[0]
        return {"interrupt": intr.value, "thread_id": thread_id}
    return {"final": result, "thread_id": thread_id}


def test_acoustic_normal_clip_no_work_order(shared_store, tmp_path) -> None:
    baseline = _build_baseline()
    graph = build_acoustic_workflow(shared_store, checkpoint_db=str(tmp_path / "cp.db"))
    sample = generate_synthetic_acoustic("normal", seed=2024, snr_db=12.0)
    out = _run_acoustic_until_decision(
        graph, asset_id="Microgrid/BESS_Site_A/CoolingFan_01",
        sample=sample, baseline=baseline, thread_id="ac-normal",
    )
    assert "final" in out
    assert out["final"]["final_status"] == "normal_no_action"
    assert shared_store.list_by_status() == []


def test_acoustic_abnormal_clip_pauses_for_human(shared_store, tmp_path) -> None:
    baseline = _build_baseline()
    graph = build_acoustic_workflow(shared_store, checkpoint_db=str(tmp_path / "cp.db"))
    sample = generate_synthetic_acoustic("abnormal", seed=4242, snr_db=12.0)
    out = _run_acoustic_until_decision(
        graph, asset_id="Microgrid/BESS_Site_A/CoolingFan_01",
        sample=sample, baseline=baseline, thread_id="ac-abn",
    )
    assert "interrupt" in out
    intr = out["interrupt"]
    assert intr["modality"] == "acoustic"
    wo = shared_store.get(intr["work_order_id"])
    assert wo.status == "pending_approval"
    assert wo.evidence["modality"] == "acoustic"
    # Resume with human approval
    resume_with_decision(
        graph, thread_id="ac-abn", approve=True,
        decided_by="human:test", note="acoustic alert confirmed",
    )
    final = shared_store.get(intr["work_order_id"])
    assert final.status == "approved"
    trail = shared_store.audit_trail(intr["work_order_id"])
    assert [e["event"] for e in trail] == ["created", "approval_requested", "approved"]


def test_vibration_and_acoustic_share_same_store(shared_store, tmp_path) -> None:
    """One asset's audit history must span both modalities — the whole point
    of building the second modality on top of the same persistence layer."""
    baseline = _build_baseline()
    asset = "Microgrid/BESS_Site_A/CoolingPump_01"
    cp = str(tmp_path / "cp.db")

    # Vibration: inner_race fault
    v_graph = build_workflow(shared_store, checkpoint_db=cp + ".v")
    v_sample = generate_synthetic(
        "inner_race", snr_db=15.0, seed=11,  # type: ignore[arg-type]
    )
    v_out = run_until_decision(v_graph, asset_id=asset, sample=v_sample, thread_id="v-1")
    assert "interrupt" in v_out
    v_wo = v_out["interrupt"]["work_order_id"]
    resume_with_decision(v_graph, thread_id="v-1", approve=True, decided_by="human:ops")

    # Acoustic: abnormal fan
    a_graph = build_acoustic_workflow(shared_store, checkpoint_db=cp + ".a")
    a_sample = generate_synthetic_acoustic("abnormal", seed=22, snr_db=12.0)
    a_out = _run_acoustic_until_decision(
        a_graph, asset_id=asset, sample=a_sample, baseline=baseline, thread_id="a-1",
    )
    assert "interrupt" in a_out
    a_wo = a_out["interrupt"]["work_order_id"]
    resume_with_decision(a_graph, thread_id="a-1", approve=False, decided_by="human:ops", note="acoustic FP")

    # Same store: both work orders findable, both audit trails present
    all_wos = shared_store.list_by_status()
    asset_wos = [w for w in all_wos if w.asset_id == asset]
    assert {w.id for w in asset_wos} >= {v_wo, a_wo}
    modalities = {w.evidence.get("modality", "vibration") for w in asset_wos}
    # vibration evidence dict doesn't carry an explicit modality tag (it
    # predates the acoustic addition); acoustic is explicit. Both rows
    # exist either way.
    assert "acoustic" in modalities
