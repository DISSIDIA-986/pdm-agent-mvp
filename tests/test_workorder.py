"""Tests for the work-order persistence + audit trail."""
from __future__ import annotations

import pytest

from pdm_agent.workorder import WorkOrderStore


@pytest.fixture
def store(tmp_path) -> WorkOrderStore:
    return WorkOrderStore(tmp_path / "test.db")


def _draft(store: WorkOrderStore, severity: str = "alert"):
    return store.create_draft(
        sample_id="s-1",
        asset_id="Microgrid/BESS_Site_A/CoolingPump_01",
        severity=severity,
        predicted_class="inner_race",
        confidence=0.78,
        summary="example",
        evidence={"top": "inner_race"},
    )


def test_create_draft_persists(store: WorkOrderStore) -> None:
    wo = _draft(store)
    fetched = store.get(wo.id)
    assert fetched is not None
    assert fetched.status == "draft"
    assert fetched.evidence == {"top": "inner_race"}


def test_lifecycle_approve(store: WorkOrderStore) -> None:
    wo = _draft(store)
    store.request_approval(wo.id)
    assert store.get(wo.id).status == "pending_approval"
    store.decide(wo.id, approve=True, decided_by="human:alice", note="confirmed by site")
    final = store.get(wo.id)
    assert final.status == "approved"
    assert final.decided_by == "human:alice"
    assert final.decision_note == "confirmed by site"


def test_lifecycle_reject(store: WorkOrderStore) -> None:
    wo = _draft(store, severity="alert")
    store.request_approval(wo.id)
    store.decide(wo.id, approve=False, decided_by="human:bob", note="false positive")
    assert store.get(wo.id).status == "rejected"


def test_audit_trail_records_every_event(store: WorkOrderStore) -> None:
    wo = _draft(store, severity="critical")
    store.request_approval(wo.id)
    store.decide(wo.id, approve=True, decided_by="human:alice")
    trail = store.audit_trail(wo.id)
    events = [t["event"] for t in trail]
    assert events == ["created", "approval_requested", "approved"]
    # Each event has a timestamp
    assert all(t["at"] for t in trail)


def test_cannot_double_decide(store: WorkOrderStore) -> None:
    wo = _draft(store)
    store.request_approval(wo.id)
    store.decide(wo.id, approve=True, decided_by="human:alice")
    with pytest.raises(ValueError, match="cannot decide"):
        store.decide(wo.id, approve=False, decided_by="human:alice")


def test_list_by_status_filters(store: WorkOrderStore) -> None:
    a = _draft(store)
    b = _draft(store)
    store.request_approval(a.id)
    drafts = store.list_by_status("draft")
    pending = store.list_by_status("pending_approval")
    assert b.id in {x.id for x in drafts}
    assert a.id in {x.id for x in pending}
