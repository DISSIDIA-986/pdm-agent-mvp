"""LangGraph state machine: anomaly detect → draft work order → human approval → persist.

State transitions:
  ingest --> diagnose --> route_severity
    if severity == normal     --> end (logged only)
    if severity in {watch}    --> draft_workorder --> persist_pending (auto, low-risk)
    if severity in {alert,
                     critical} --> draft_workorder --> human_approval (INTERRUPT)
                                                    --> persist_decision

Key design choices (matched to the "runtime, not copilot" thesis):
  - Severity gating is pure code, not LLM judgement (deterministic + auditable)
  - LLM (Claude) only writes the natural-language summary — optional, fallback
    to deterministic template if ANTHROPIC_API_KEY is absent
  - Every transition writes to audit_log via WorkOrderStore
  - Human approval uses LangGraph interrupt() — workflow halts until resume()
"""
from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver, MemorySaver  # type: ignore[attr-defined]
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from .data import VibrationSample
from .diagnostic import Diagnosis, diagnose
from .workorder import WorkOrder, WorkOrderStore

log = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    # Input
    asset_id: str
    sample: VibrationSample
    # Derived
    diagnosis: Diagnosis | None
    summary: str | None
    work_order: WorkOrder | None
    decision: dict | None  # {"approve": bool, "decided_by": str, "note": str}
    # Bookkeeping
    final_status: str | None  # "normal_no_action" | "auto_pending" | "human_decided"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_diagnose(state: WorkflowState) -> WorkflowState:
    sample = state["sample"]
    d = diagnose(sample)
    log.info(
        "diagnose: sample=%s pred=%s severity=%s confidence=%.2f",
        sample.sample_id, d.predicted_class, d.severity, d.confidence,
    )
    return {"diagnosis": d}


def _llm_summary(diagnosis: Diagnosis, asset_id: str) -> str:
    """Generate a natural-language summary. Falls back to template if no API key."""
    template = (
        f"Asset {asset_id} sample {diagnosis.sample_id}: "
        f"predicted {diagnosis.predicted_class} (severity={diagnosis.severity}, "
        f"confidence={diagnosis.confidence:.2f}). "
        f"Top evidence: "
        + ", ".join(
            f"{e.fault_class}@{e.peak_freq_hz:.1f}Hz score={e.score:.2f}"
            for e in sorted(diagnosis.evidence, key=lambda x: x.score, reverse=True)[:2]
        )
        + f". Time features: RMS={diagnosis.time_features.rms:.3f}, kurtosis={diagnosis.time_features.kurtosis:.2f}."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return template + " [LLM summary unavailable: no API key, deterministic template used.]"

    try:  # local import so missing anthropic doesn't break the workflow
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a maintenance ops summariser. Given this PdM diagnosis "
                        "for a microgrid BESS cooling pump bearing, write a 2-sentence "
                        "summary for an operator. Be specific about which fault frequency "
                        "and what evidence ratio. Do NOT invent fault types not in the data. "
                        "Do NOT recommend a specific action; you only describe.\n\n"
                        f"Diagnosis JSON: {diagnosis.to_dict()}\n"
                        f"Asset: {asset_id}"
                    ),
                }
            ],
        )
        return msg.content[0].text  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        log.warning("LLM summary failed (%s) — using template", e)
        return template + f" [LLM summary failed: {e}]"


def node_draft(state: WorkflowState, store: WorkOrderStore) -> WorkflowState:
    d = state["diagnosis"]
    asset_id = state["asset_id"]
    summary = _llm_summary(d, asset_id)
    wo = store.create_draft(
        sample_id=d.sample_id,
        asset_id=asset_id,
        severity=d.severity,
        predicted_class=d.predicted_class,
        confidence=d.confidence,
        summary=summary,
        evidence=d.to_dict(),
    )
    log.info("draft work order %s severity=%s", wo.id, wo.severity)
    return {"work_order": wo, "summary": summary}


def node_request_human(state: WorkflowState, store: WorkOrderStore) -> Command:
    """Halt the workflow until a human resumes with a decision payload."""
    wo = state["work_order"]
    store.request_approval(wo.id)
    log.info("waiting for human decision on %s", wo.id)
    # LangGraph interrupt freezes execution. Caller resumes via Command(resume=...)
    decision = interrupt(
        {
            "work_order_id": wo.id,
            "asset_id": wo.asset_id,
            "severity": wo.severity,
            "predicted_class": wo.predicted_class,
            "confidence": wo.confidence,
            "summary": wo.summary,
            "ask": "Approve maintenance work order? Reply with {approve: bool, decided_by, note}.",
        }
    )
    return Command(update={"decision": decision})


def node_persist_decision(state: WorkflowState, store: WorkOrderStore) -> WorkflowState:
    wo = state["work_order"]
    decision = state["decision"] or {"approve": False, "decided_by": "auto:no_decision", "note": "default reject"}
    store.decide(
        wo.id,
        approve=bool(decision.get("approve", False)),
        decided_by=str(decision.get("decided_by", "human:unknown")),
        note=decision.get("note"),
    )
    return {"final_status": "human_decided"}


def node_persist_auto(state: WorkflowState, store: WorkOrderStore) -> WorkflowState:
    """Low-severity branch: keep as pending_approval for human review later, no auto-approve."""
    wo = state["work_order"]
    store.request_approval(wo.id)
    return {"final_status": "auto_pending"}


def node_normal(state: WorkflowState) -> WorkflowState:
    return {"final_status": "normal_no_action"}


def route_severity(state: WorkflowState) -> str:
    """Route based on diagnosed severity. Deterministic — not LLM-decided."""
    d = state["diagnosis"]
    if d is None:
        return "normal"
    if d.severity == "normal":
        return "normal"
    if d.severity == "watch":
        return "auto_branch"
    return "human_branch"  # alert / critical


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_workflow(store: WorkOrderStore, *, checkpoint_db: str | None = None):
    """Build the LangGraph workflow.

    `checkpoint_db`: path to a SQLite file. If None, uses in-memory checkpoints
    (tests only). For real use pass a stable path so human-approval state
    survives a process restart.
    """
    g = StateGraph(WorkflowState)
    g.add_node("diagnose", node_diagnose)
    g.add_node("draft", lambda s: node_draft(s, store))
    g.add_node("auto_branch", lambda s: node_persist_auto(s, store))
    g.add_node("human_branch", lambda s: node_request_human(s, store))
    g.add_node("persist_decision", lambda s: node_persist_decision(s, store))
    g.add_node("normal_branch", node_normal)

    g.set_entry_point("diagnose")
    g.add_conditional_edges(
        "diagnose",
        route_severity,
        {
            "normal": "normal_branch",
            "auto_branch": "draft",
            "human_branch": "draft",
        },
    )

    def post_draft_router(state: WorkflowState) -> str:
        sev = state["diagnosis"].severity
        return "auto_branch" if sev == "watch" else "human_branch"

    g.add_conditional_edges(
        "draft",
        post_draft_router,
        {"auto_branch": "auto_branch", "human_branch": "human_branch"},
    )
    g.add_edge("human_branch", "persist_decision")
    g.add_edge("persist_decision", END)
    g.add_edge("auto_branch", END)
    g.add_edge("normal_branch", END)

    if checkpoint_db:
        import sqlite3
        # check_same_thread=False so the workflow can be invoked from any thread;
        # SqliteSaver itself serialises writes via its own lock.
        conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
    else:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def assert_thread_unused(graph, thread_id: str) -> None:
    """Raise if the given thread_id already has a checkpoint.

    Prevents the codex-flagged drift where re-using a thread_id with a new sample
    silently overwrites the prior incident's checkpoint and leaves the first
    work order orphaned. Callers should pass work-order-derived thread IDs
    (one thread per incident).
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    # A fresh thread has no values + no tasks. Existing thread has either a
    # state.values dict or pending tasks/interrupts.
    if state.values or state.tasks:
        raise ValueError(
            f"thread_id {thread_id!r} already has a checkpoint — refusing to "
            "overwrite an incident in-flight. Use a fresh incident-derived id."
        )


# ---------------------------------------------------------------------------
# Convenience: synchronous run helpers (for scripts and tests)
# ---------------------------------------------------------------------------

def run_until_decision(graph, *, asset_id: str, sample: VibrationSample, thread_id: str) -> dict[str, Any]:
    """Run the graph; if it interrupts, return the interrupt payload.

    Returns dict with one of: {"interrupt": payload}, {"final": state}.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"asset_id": asset_id, "sample": sample}, config=config)
    # If interrupted, the result contains an "__interrupt__" sentinel inside the
    # state snapshot. Check via get_state.
    state = graph.get_state(config)
    if state.tasks and any(t.interrupts for t in state.tasks):
        # Pull the first interrupt payload
        intr = state.tasks[0].interrupts[0]
        return {"interrupt": intr.value, "thread_id": thread_id}
    return {"final": result, "thread_id": thread_id}


def resume_with_decision(
    graph,
    *,
    thread_id: str,
    approve: bool,
    decided_by: str,
    note: str | None = None,
    expected_work_order_id: str | None = None,
) -> dict:
    """Resume a paused workflow with a human decision.

    `expected_work_order_id`: if provided, the resume is rejected unless the
    paused interrupt belongs to that work_order. This prevents accidentally
    approving the wrong incident (codex-flagged drift).
    """
    config = {"configurable": {"thread_id": thread_id}}
    if expected_work_order_id is not None:
        state = graph.get_state(config)
        active_wo = None
        for t in state.tasks:
            for intr in t.interrupts:
                payload = intr.value if hasattr(intr, "value") else intr
                if isinstance(payload, dict):
                    active_wo = payload.get("work_order_id")
        if active_wo != expected_work_order_id:
            raise ValueError(
                f"resume mismatch: thread_id {thread_id!r} is paused on "
                f"work_order {active_wo!r}, not the expected {expected_work_order_id!r}"
            )
    result = graph.invoke(
        Command(resume={"approve": approve, "decided_by": decided_by, "note": note}),
        config=config,
    )
    return {"final": result, "thread_id": thread_id}
