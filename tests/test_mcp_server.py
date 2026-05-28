"""Tests for the MCP server wrapper.

We exercise the underlying tool functions directly rather than via the stdio
transport — that gives the same coverage of the diagnostic and work-order
plumbing without booting a subprocess. A separate manual smoke test under
`python -m pdm_agent.mcp_server --transport sse` covers the transport layer.

Special focus areas for these tests (driven by codex round-2 review):
  - LLM clients must NOT be able to forge a `human:` identity
  - decision tools are advisory-only; they never mutate work_orders.status
  - tool schemas use Literal/typed arguments so MCP clients see proper enums
"""
from __future__ import annotations

import asyncio
import pathlib

import numpy as np
import pytest

from pdm_agent import mcp_server
from pdm_agent.data import generate_synthetic


def test_diagnose_vibration_round_trip() -> None:
    s = generate_synthetic("inner_race", snr_db=15.0, seed=1)
    result = mcp_server.diagnose_vibration(
        signal=s.signal.astype(float).tolist(),
        sample_rate_hz=s.sample_rate_hz,
        rpm=s.rpm,
        sample_id="mcp-test-1",
    )
    assert result["predicted_class"] == "inner_race"
    assert result["confidence_is_calibrated"] is False
    assert "evidence" in result


def test_diagnose_vibration_rejects_short_signal() -> None:
    result = mcp_server.diagnose_vibration(
        signal=[0.0] * 100,
        sample_rate_hz=12_000,
        rpm=1797,
        sample_id="short",
    )
    assert "error" in result


def test_diagnose_vibration_rejects_nonfinite() -> None:
    bad = [0.0] * 2048
    bad[10] = float("inf")
    result = mcp_server.diagnose_vibration(
        signal=bad,
        sample_rate_hz=12_000,
        rpm=1797,
        sample_id="bad",
    )
    assert "error" in result and "finite" in result["error"]


def test_diagnose_synthetic_reports_match() -> None:
    result = mcp_server.diagnose_synthetic(fault_class="outer_race", snr_db=15.0, seed=2)
    assert result["ground_truth"] == "outer_race"
    assert result["matched"] is True
    assert result["diagnosis"]["predicted_class"] == "outer_race"


def test_diagnose_synthetic_normal_class() -> None:
    result = mcp_server.diagnose_synthetic(fault_class="normal", snr_db=20.0, seed=3)
    assert result["matched"] is True
    assert result["diagnosis"]["severity"] == "normal"


@pytest.fixture
def mcp_db(tmp_path, monkeypatch):
    db = tmp_path / "mcp.db"
    monkeypatch.setattr(mcp_server, "DEFAULT_DB", str(db))
    return db


def _seed_pending_wo(db_path, **overrides):
    from pdm_agent.workorder import WorkOrderStore
    store = WorkOrderStore(db_path)
    defaults = dict(
        sample_id="s-1",
        asset_id="Microgrid/BESS_Site_A/CoolingPump_01",
        severity="alert",
        predicted_class="inner_race",
        confidence=0.78,
        summary="seed wo",
        evidence={},
    )
    defaults.update(overrides)
    wo = store.create_draft(**defaults)
    store.request_approval(wo.id)
    return store, wo


def test_propose_decision_advisory_only(mcp_db) -> None:
    """Advisory recommendation must NOT mutate work order status."""
    store, wo = _seed_pending_wo(mcp_db)
    result = mcp_server.propose_decision_for_work_order(
        work_order_id=wo.id,
        recommendation="approve",
        rationale="Strong BPFI peak; high family score; consistent with inner race fault.",
        proposed_by="agent:claude-haiku-4-5",
    )
    assert result["ok"] is True
    assert result["advisory_only"] is True
    # The work order must STILL be pending_approval — no mutation
    assert store.get(wo.id).status == "pending_approval"
    # And the recommendation must be retrievable
    recs = mcp_server.list_agent_recommendations(work_order_id=wo.id)
    assert recs["n"] == 1
    assert recs["items"][0]["recommendation"] == "approve"
    assert recs["items"][0]["proposed_by"] == "agent:claude-haiku-4-5"


def test_propose_decision_rejects_human_impersonation(mcp_db) -> None:
    """An LLM client must not be able to forge a human identity."""
    store, wo = _seed_pending_wo(mcp_db)
    for forged in ("human:alice", "human:bob", "user:carol", "operator", "alice"):
        r = mcp_server.propose_decision_for_work_order(
            work_order_id=wo.id,
            recommendation="approve",
            rationale="x" * 20,
            proposed_by=forged,
        )
        assert r["ok"] is False, f"forged identity {forged!r} was accepted!"
        assert "agent:" in r["error"]
    # Confirm the work order is still untouched
    assert store.get(wo.id).status == "pending_approval"


def test_propose_decision_requires_rationale(mcp_db) -> None:
    store, wo = _seed_pending_wo(mcp_db)
    r = mcp_server.propose_decision_for_work_order(
        work_order_id=wo.id,
        recommendation="reject",
        rationale="bad",  # < 10 chars
        proposed_by="agent:claude-1",
    )
    assert r["ok"] is False
    assert "rationale" in r["error"]


def test_propose_decision_rejects_unknown_wo(mcp_db) -> None:
    r = mcp_server.propose_decision_for_work_order(
        work_order_id="does-not-exist",
        recommendation="approve",
        rationale="anything reasonably long",
        proposed_by="agent:claude-1",
    )
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_propose_decision_only_on_pending(mcp_db) -> None:
    """Recommendations are not accepted once a work order is already decided."""
    from pdm_agent.workorder import WorkOrderStore
    store, wo = _seed_pending_wo(mcp_db)
    store.decide(wo.id, approve=True, decided_by="human:operator-via-ui")
    r = mcp_server.propose_decision_for_work_order(
        work_order_id=wo.id,
        recommendation="reject",
        rationale="too late, but trying anyway",
        proposed_by="agent:claude-1",
    )
    assert r["ok"] is False
    assert "pending_approval" in r["error"]


def test_list_work_orders_filters(mcp_db) -> None:
    _seed_pending_wo(mcp_db, sample_id="a")
    _seed_pending_wo(mcp_db, sample_id="b")
    listed = mcp_server.list_work_orders(status="pending_approval")
    assert listed["n"] == 2
    other = mcp_server.list_work_orders(status="approved")
    assert other["n"] == 0


def test_list_work_orders_clamps_limit(mcp_db) -> None:
    for i in range(5):
        _seed_pending_wo(mcp_db, sample_id=f"s-{i}")
    r = mcp_server.list_work_orders(limit=2)
    assert r["n"] == 2
    # Clamp upper bound
    r_big = mcp_server.list_work_orders(limit=10_000)
    assert r_big["n"] == 5


def test_audit_trail_via_mcp_tool(mcp_db) -> None:
    store, wo = _seed_pending_wo(mcp_db)
    store.decide(wo.id, approve=True, decided_by="human:operator-ui")
    trail = mcp_server.list_audit_trail(wo.id)
    events = [e["event"] for e in trail["events"]]
    assert events == ["created", "approval_requested", "approved"]


def test_config_resource_includes_scope_and_calibration_note() -> None:
    import json
    raw = mcp_server.config_resource()
    data = json.loads(raw)
    assert data["method"] == "envelope-spectrum-v2-family"
    assert "ball-defect" in " ".join(data["known_failure_modes"])
    assert "NOT validated" in data["scope"]


def test_security_resource_documents_advisory_only() -> None:
    text = mcp_server.security_resource()
    assert "Authentication: NONE" in text
    assert "advisory" in text.lower()
    assert "agent:" in text
    assert "human:" in text  # should explicitly forbid


def test_error_analysis_resource_exists_or_explains() -> None:
    text = mcp_server.error_analysis_resource()
    assert "Error" in text or "Not yet generated" in text


def test_registered_tools_match_documented_set() -> None:
    """Schema test: list_tools() returns exactly the documented surface."""
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "diagnose_vibration",
        "diagnose_synthetic",
        "poll_opc_asset",
        "list_work_orders",
        "propose_decision_for_work_order",
        "list_agent_recommendations",
        "list_audit_trail",
    }
    assert names == expected, f"tool drift: extra={names - expected}, missing={expected - names}"
    # And confirm none of the legacy approve/reject names slipped back in
    for legacy in ("approve_work_order", "reject_work_order"):
        assert legacy not in names, f"legacy write tool {legacy} re-appeared"


def test_registered_resources_match_documented_set() -> None:
    resources = asyncio.run(mcp_server.mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    assert uris == {"pdm://config", "pdm://security", "pdm://error-analysis"}
