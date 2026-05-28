"""Persistence + thread-id safety tests added after codex round-3 review.

These verify the fixes for the two 🔴 致命 findings:
  1. SqliteSaver persists checkpoints across process boundaries
  2. thread_id reuse without proper sequencing is rejected by assert_thread_unused
"""
from __future__ import annotations

import pytest

from pdm_agent.data import generate_synthetic
from pdm_agent.workflow import (
    assert_thread_unused,
    build_workflow,
    resume_with_decision,
    run_until_decision,
)
from pdm_agent.workorder import WorkOrderStore


def _fresh(tmp_path, name: str = "wf"):
    return WorkOrderStore(tmp_path / f"{name}.db")


def test_persistent_checkpoint_survives_rebuild(tmp_path) -> None:
    """Build graph A → pause → discard → rebuild graph B with same SQLite file → resume → finish."""
    store = _fresh(tmp_path, "persist")
    cp = str(tmp_path / "checkpoints.db")
    g1 = build_workflow(store, checkpoint_db=cp)
    sample = generate_synthetic("inner_race", snr_db=15.0, seed=101)
    out = run_until_decision(g1, asset_id="A1", sample=sample, thread_id="incident-101")
    assert "interrupt" in out
    wo_id = out["interrupt"]["work_order_id"]
    # Discard the in-memory graph (simulates a process restart)
    del g1
    g2 = build_workflow(store, checkpoint_db=cp)
    # Reopened graph should still see the paused incident
    final = resume_with_decision(
        g2, thread_id="incident-101", approve=True, decided_by="human:alice",
        expected_work_order_id=wo_id,
    )
    assert final["final"]["final_status"] == "human_decided"
    assert store.get(wo_id).status == "approved"


def test_thread_id_reuse_rejected_by_assert(tmp_path) -> None:
    store = _fresh(tmp_path, "reuse")
    cp = str(tmp_path / "cp.db")
    graph = build_workflow(store, checkpoint_db=cp)
    s1 = generate_synthetic("inner_race", snr_db=15.0, seed=102)
    s2 = generate_synthetic("outer_race", snr_db=15.0, seed=103)
    assert_thread_unused(graph, "shared-thread")
    out1 = run_until_decision(graph, asset_id="A1", sample=s1, thread_id="shared-thread")
    assert "interrupt" in out1
    # Now the same thread should be flagged as in-use
    with pytest.raises(ValueError, match="already has a checkpoint"):
        assert_thread_unused(graph, "shared-thread")
    # Caller should pick a fresh thread for the new incident
    out2 = run_until_decision(graph, asset_id="A2", sample=s2, thread_id="other-thread")
    assert "interrupt" in out2
    # First incident still resumable
    resume_with_decision(graph, thread_id="shared-thread", approve=False, decided_by="human:b")
    assert store.get(out1["interrupt"]["work_order_id"]).status == "rejected"


def test_resume_wrong_work_order_rejected(tmp_path) -> None:
    store = _fresh(tmp_path, "guard")
    cp = str(tmp_path / "cp.db")
    graph = build_workflow(store, checkpoint_db=cp)
    sample = generate_synthetic("inner_race", snr_db=15.0, seed=104)
    out = run_until_decision(graph, asset_id="A1", sample=sample, thread_id="guard-1")
    real_wo = out["interrupt"]["work_order_id"]
    with pytest.raises(ValueError, match="resume mismatch"):
        resume_with_decision(
            graph, thread_id="guard-1", approve=True, decided_by="human:x",
            expected_work_order_id="bogus-id",
        )
    # And the real one still works
    resume_with_decision(
        graph, thread_id="guard-1", approve=True, decided_by="human:x",
        expected_work_order_id=real_wo,
    )
    assert store.get(real_wo).status == "approved"
