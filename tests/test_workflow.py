"""Integration tests for the LangGraph workflow: severity routing + human approval."""
from __future__ import annotations

import pytest

from pdm_agent.data import generate_synthetic
from pdm_agent.workflow import build_workflow, resume_with_decision, run_until_decision
from pdm_agent.workorder import WorkOrderStore


@pytest.fixture
def store_and_graph(tmp_path):
    store = WorkOrderStore(tmp_path / "wf.db")
    graph = build_workflow(store)
    return store, graph


def test_normal_sample_ends_without_work_order(store_and_graph) -> None:
    store, graph = store_and_graph
    sample = generate_synthetic("normal", duration_s=1.0, snr_db=20.0, seed=10)
    out = run_until_decision(
        graph, asset_id="Microgrid/BESS_Site_A/CoolingPump_01", sample=sample, thread_id="t-normal",
    )
    assert "final" in out
    assert out["final"]["final_status"] == "normal_no_action"
    assert store.list_by_status() == []


def test_alert_sample_pauses_for_human(store_and_graph) -> None:
    store, graph = store_and_graph
    sample = generate_synthetic("inner_race", duration_s=1.0, snr_db=15.0, seed=11)
    out = run_until_decision(
        graph, asset_id="Microgrid/BESS_Site_A/CoolingPump_01", sample=sample, thread_id="t-alert",
    )
    assert "interrupt" in out, f"expected interrupt, got {out}"
    intr = out["interrupt"]
    assert "work_order_id" in intr
    wo = store.get(intr["work_order_id"])
    assert wo.status == "pending_approval"


def test_resume_approve_closes_workflow(store_and_graph) -> None:
    store, graph = store_and_graph
    sample = generate_synthetic("outer_race", duration_s=1.0, snr_db=15.0, seed=12)
    first = run_until_decision(graph, asset_id="A1", sample=sample, thread_id="t-approve")
    assert "interrupt" in first
    wo_id = first["interrupt"]["work_order_id"]
    final = resume_with_decision(graph, thread_id="t-approve", approve=True, decided_by="human:alice", note="confirmed")
    assert final["final"]["final_status"] == "human_decided"
    wo = store.get(wo_id)
    assert wo.status == "approved"
    trail = store.audit_trail(wo_id)
    assert [e["event"] for e in trail] == ["created", "approval_requested", "approved"]


def test_resume_reject_closes_workflow(store_and_graph) -> None:
    store, graph = store_and_graph
    sample = generate_synthetic("inner_race", duration_s=1.0, snr_db=15.0, seed=13)
    first = run_until_decision(graph, asset_id="A1", sample=sample, thread_id="t-reject")
    assert "interrupt" in first
    wo_id = first["interrupt"]["work_order_id"]
    resume_with_decision(graph, thread_id="t-reject", approve=False, decided_by="human:bob", note="false positive")
    wo = store.get(wo_id)
    assert wo.status == "rejected"


def test_workflow_is_deterministic(store_and_graph) -> None:
    """Same sample + fresh thread => same diagnosis and interrupt payload."""
    store, graph = store_and_graph
    a = generate_synthetic("inner_race", seed=20, snr_db=15.0)
    b = generate_synthetic("inner_race", seed=20, snr_db=15.0)
    out1 = run_until_decision(graph, asset_id="A", sample=a, thread_id="t1")
    out2 = run_until_decision(graph, asset_id="A", sample=b, thread_id="t2")
    # Both interrupt with equivalent payload (modulo work_order_id which is a uuid)
    assert "interrupt" in out1 and "interrupt" in out2
    p1, p2 = out1["interrupt"], out2["interrupt"]
    assert p1["severity"] == p2["severity"]
    assert p1["predicted_class"] == p2["predicted_class"]
    assert abs(p1["confidence"] - p2["confidence"]) < 1e-6
