"""MCP server wrapper around the PdM diagnostic engine.

⚠ SECURITY MODEL (read first):
  This server is designed for **local development only**. It has no
  authentication, no transport encryption, and binds to 127.0.0.1 by default.
  The decision tools (`propose_decision_*`) are intentionally *advisory* — they
  record a recommended decision in an `agent_recommendations` table but do NOT
  mutate work-order status. Final approval/rejection of a work order can only
  happen via the workflow's `Command(resume=...)` path or by an authenticated
  human operator hitting the workorder store directly. This is by design: an
  LLM holding an MCP client must not be able to forge `decided_by="human:alice"`
  and silently approve maintenance work.

Exposes the same capabilities as the HTTP sidecar but via the Model Context
Protocol, so Claude Code / Claude Desktop / any MCP client can run the
diagnostic, see open work orders, RECOMMEND (not enact) decisions, and poll
the mock OPC UA asset.

The HTTP sidecar (pdm_agent.sidecar) and this MCP server are intentional
*siblings*, not one wrapping the other — both delegate into the same in-process
diagnose() function. This validates the README claim that the architecture was
designed to plug into MCP without re-architecting.

Usage:
  # stdio (Claude Desktop / Code spawn the server)
  python -m pdm_agent.mcp_server

  # SSE (local dev only — binds to 127.0.0.1, no auth)
  python -m pdm_agent.mcp_server --transport sse --port 8210

Tools exposed:
  diagnose_vibration              — run envelope-spectrum-v2 on a signal array
  diagnose_synthetic              — generate + diagnose a synthetic window
  poll_opc_asset                  — read latest window from the mock OPC UA server
  list_work_orders                — list work orders (read-only)
  propose_decision_for_work_order — RECOMMEND approve/reject (advisory; no mutation)
  list_audit_trail                — fetch the audit log for a given work order
  list_agent_recommendations      — fetch advisory decisions logged by the LLM

Resources exposed:
  pdm://config                    — sidecar version + method + scope statement
  pdm://security                  — explicit security boundary statement
  pdm://error-analysis            — current eval/error_analysis.md content
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import re
import sqlite3
import uuid
from typing import Literal

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import __version__
from .data import VibrationSample, generate_synthetic
from .diagnostic import diagnose
from .workorder import WorkOrderStore

log = logging.getLogger(__name__)

DEFAULT_DB = os.environ.get("PDM_AGENT_DB", "pdm-agent.db")
DEFAULT_OPC_ENDPOINT = os.environ.get(
    "PDM_AGENT_OPC_ENDPOINT", "opc.tcp://127.0.0.1:4842/pdm-agent-mock"
)

mcp = FastMCP(name="pdm-agent-mvp", instructions=(
    "Predictive-maintenance agent for a simulated microgrid BESS cooling pump. "
    "Local-development MCP server, no authentication. The diagnostic method is "
    "`envelope-spectrum-v2-family` — physics-derived, deterministic. "
    "`confidence` is NOT a calibrated posterior; treat it as a ranking signal "
    "(see pdm://config). The 0.007\" ball-fault class is a documented "
    "under-detection — never recommend approval of a ball-class prediction "
    "without consulting pdm://error-analysis. "
    "IMPORTANT: there is NO tool here that approves or rejects a work order. "
    "`propose_decision_for_work_order` records an advisory recommendation "
    "ONLY. Final approval must happen via the human-operator UI or the "
    "workflow's authenticated resume path — see pdm://security."
))


# Advisory recommendation table (separate from work_orders; LLM may write here)
AGENT_RECS_DDL = """
CREATE TABLE IF NOT EXISTS agent_recommendations (
    id TEXT PRIMARY KEY,
    work_order_id TEXT NOT NULL,
    recommendation TEXT NOT NULL CHECK (recommendation IN ('approve', 'reject')),
    rationale TEXT,
    proposed_by TEXT NOT NULL,
    proposed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_recs_wo ON agent_recommendations(work_order_id);
"""

# Restricted identifier pattern for `proposed_by` — agent identifier convention.
# Humans never use this surface; they use the operator UI. Reject anything
# matching `human:*` to prevent agent impersonation.
AGENT_ID_PATTERN = re.compile(r"^agent:[a-zA-Z0-9._-]{1,64}$")


def _init_recs_table() -> None:
    conn = sqlite3.connect(DEFAULT_DB, timeout=5.0)
    try:
        conn.executescript(AGENT_RECS_DDL)
        conn.commit()
    finally:
        conn.close()


def _store() -> WorkOrderStore:
    return WorkOrderStore(DEFAULT_DB)


# ---------------------------------------------------------------------------
# Tools — analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def diagnose_vibration(
    signal: list[float],
    sample_rate_hz: int = 12_000,
    rpm: float = 1797.0,
    sample_id: str = "ad-hoc",
) -> dict:
    """Run the envelope-spectrum-v2 diagnostic on a vibration signal.

    Args:
        signal: 1-D vibration samples (min length 1024, float-valued, finite).
        sample_rate_hz: ADC sample rate (Hz). CWRU drive-end is 12000.
        rpm: bearing RPM at acquisition. SKF 6205 fault frequencies scale linearly with RPM.
        sample_id: free-form label that propagates into the work order if the
            severity warrants one (caller should pass something traceable).
    """
    if len(signal) < 1024:
        return {"error": f"signal too short ({len(signal)} samples; need >=1024)"}
    arr = np.asarray(signal, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return {"error": "signal contains non-finite values"}
    sample = VibrationSample(
        sample_id=sample_id,
        fault_class="normal",  # ground truth unknown to the diagnostic
        signal=arr,
        sample_rate_hz=sample_rate_hz,
        rpm=rpm,
        source="synthetic",
    )
    d = diagnose(sample)
    return d.to_dict()


@mcp.tool()
def diagnose_synthetic(
    fault_class: Literal["normal", "inner_race", "outer_race", "ball"] = "inner_race",
    snr_db: float = 15.0,
    seed: int = 0,
) -> dict:
    """Generate a deterministic synthetic vibration window and diagnose it.

    Useful as a smoke check from the chat surface ("does the diagnostic still
    fire on a textbook inner-race signal?") without needing a real recording.
    """
    s = generate_synthetic(fault_class, snr_db=snr_db, seed=seed)
    d = diagnose(s)
    return {
        "ground_truth": fault_class,
        "diagnosis": d.to_dict(),
        "matched": d.predicted_class == fault_class,
    }


# ---------------------------------------------------------------------------
# Tools — OPC UA passthrough
# ---------------------------------------------------------------------------

@mcp.tool()
async def poll_opc_asset(
    endpoint: str = DEFAULT_OPC_ENDPOINT,
    site_name: str = "BESS_Site_A",
    asset_name: str = "CoolingPump_01",
) -> dict:
    """Read the latest vibration window from the mock OPC UA server and
    immediately diagnose it. Returns both the raw OPC metadata and the
    diagnostic verdict. Run `python -m pdm_agent.opcua_mock` first.
    """
    from .opcua_client import poll_once  # late import; asyncua is heavy
    try:
        snap = await poll_once(endpoint, site_name=site_name, asset_name=asset_name)
    except Exception as e:  # noqa: BLE001
        return {"error": f"opc poll failed: {type(e).__name__}: {e}"}
    d = diagnose(snap.sample)
    return {
        "opc": {
            "endpoint": endpoint,
            "update_counter": snap.update_counter,
            "operating_mode": snap.operating_mode,
            "rpm": snap.sample.rpm,
            "sample_rate_hz": snap.sample.sample_rate_hz,
        },
        "diagnosis": d.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tools — human approval surface
# ---------------------------------------------------------------------------

WorkOrderStatus = Literal["draft", "pending_approval", "approved", "rejected", "closed"]


@mcp.tool()
def list_work_orders(status: WorkOrderStatus | None = None, limit: int = 20) -> dict:
    """List recent work orders (read-only).

    Args:
        status: filter to one of draft / pending_approval / approved / rejected
            / closed. None returns all statuses, newest first.
        limit: maximum rows to return (default 20, clamped to [1, 200]).
    """
    limit = max(1, min(int(limit), 200))
    wos = _store().list_by_status(status)  # type: ignore[arg-type]
    return {
        "n": len(wos[:limit]),
        "items": [
            {
                "id": w.id,
                "asset_id": w.asset_id,
                "severity": w.severity,
                "predicted_class": w.predicted_class,
                "confidence": round(w.confidence, 3),
                "status": w.status,
                "summary": w.summary,
                "created_at": w.created_at,
                "decided_by": w.decided_by,
            }
            for w in wos[:limit]
        ],
    }


@mcp.tool()
def propose_decision_for_work_order(
    work_order_id: str,
    recommendation: Literal["approve", "reject"],
    rationale: str,
    proposed_by: str,
) -> dict:
    """Record an ADVISORY recommendation for a work order.

    This does NOT mutate the work order's status. It writes a row into the
    `agent_recommendations` table that a human operator can review before
    actually deciding. The work_orders table remains the source of truth and
    can only be transitioned via the authenticated operator UI or the
    LangGraph workflow's `Command(resume=...)` path.

    Args:
        work_order_id: the work order being commented on
        recommendation: 'approve' or 'reject'
        rationale: free-text explanation (will be persisted verbatim)
        proposed_by: identifier of the proposing agent. MUST start with
            `agent:` — values starting with `human:` are rejected to prevent
            an LLM-driven client from impersonating a human approver.
    """
    if not AGENT_ID_PATTERN.match(proposed_by):
        return {
            "ok": False,
            "error": (
                "proposed_by must match agent:[a-zA-Z0-9._-]{1,64}. Human "
                "approval is not accepted via this tool — use the operator UI."
            ),
        }
    if not rationale or len(rationale) < 10:
        return {"ok": False, "error": "rationale must be at least 10 chars"}
    # Confirm work order exists and is in pending_approval (else recommendation is moot)
    wo = _store().get(work_order_id)
    if wo is None:
        return {"ok": False, "error": f"work_order {work_order_id} not found"}
    if wo.status != "pending_approval":
        return {
            "ok": False,
            "error": f"work_order is {wo.status}; recommendations only accepted while pending_approval",
        }
    _init_recs_table()
    rec_id = str(uuid.uuid4())
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn = sqlite3.connect(DEFAULT_DB, timeout=5.0)
    try:
        conn.execute(
            "INSERT INTO agent_recommendations VALUES (?, ?, ?, ?, ?, ?)",
            (rec_id, work_order_id, recommendation, rationale, proposed_by, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "ok": True,
        "recommendation_id": rec_id,
        "advisory_only": True,
        "next_step": "Human operator must review and decide via the operator UI.",
    }


@mcp.tool()
def list_agent_recommendations(work_order_id: str | None = None, limit: int = 20) -> dict:
    """List advisory recommendations logged by agents (read-only)."""
    limit = max(1, min(int(limit), 200))
    _init_recs_table()
    conn = sqlite3.connect(DEFAULT_DB, timeout=5.0)
    try:
        if work_order_id:
            rows = conn.execute(
                "SELECT id, work_order_id, recommendation, rationale, proposed_by, proposed_at "
                "FROM agent_recommendations WHERE work_order_id = ? "
                "ORDER BY proposed_at DESC LIMIT ?",
                (work_order_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, work_order_id, recommendation, rationale, proposed_by, proposed_at "
                "FROM agent_recommendations ORDER BY proposed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    cols = ["id", "work_order_id", "recommendation", "rationale", "proposed_by", "proposed_at"]
    return {"n": len(rows), "items": [dict(zip(cols, r)) for r in rows]}


@mcp.tool()
def list_audit_trail(work_order_id: str) -> dict:
    """Return the full audit-log timeline for a single work order (read-only)."""
    trail = _store().audit_trail(work_order_id)
    return {"work_order_id": work_order_id, "events": trail}


# ---------------------------------------------------------------------------
# Resources — read-only context
# ---------------------------------------------------------------------------

@mcp.resource("pdm://config")
def config_resource() -> str:
    """Diagnostic configuration + honest scope statement."""
    return json.dumps(
        {
            "version": __version__,
            "method": "envelope-spectrum-v2-family",
            "supports": ["normal", "inner_race", "outer_race", "ball"],
            "severity_buckets": ["normal", "watch", "alert", "critical"],
            "confidence_semantics": (
                "Softmax over deterministic family scores. NOT a calibrated "
                "posterior — misclassifications can still report high "
                "confidence. Treat as a ranking signal between candidate "
                "fault classes."
            ),
            "scope": (
                "Vibration-only PdM agent on CWRU drive-end bearing data, "
                "used as analog benchmark for BESS auxiliary equipment. NOT "
                "validated for production BESS PdM."
            ),
            "known_failure_modes": [
                "0.007-inch ball-defect class: 0/10 detection on real CWRU. "
                "FTF-sideband SNR limitation; not a threshold bug. See "
                "pdm://error-analysis."
            ],
        },
        indent=2,
    )


@mcp.resource("pdm://security")
def security_resource() -> str:
    """Explicit security boundary statement for clients."""
    return (
        "# pdm-agent-mvp MCP security boundary\n\n"
        "Authentication: NONE. This server is for local development only and "
        "binds to 127.0.0.1 by default.\n\n"
        "Write surface: this MCP server NEVER mutates work_orders.status. The "
        "only write tool is `propose_decision_for_work_order`, which records "
        "an advisory row in a separate `agent_recommendations` table that "
        "humans must review out-of-band. The `proposed_by` field requires the "
        "`agent:` prefix — values starting with `human:` are rejected.\n\n"
        "Final approval / rejection of a work order is enacted only by:\n"
        "  1. A trusted-process caller of `WorkOrderStore.decide()`, OR\n"
        "  2. The LangGraph workflow's `Command(resume=...)` path called from\n"
        "     a process that has independently authenticated the operator.\n"
        "Important: this MVP does NOT ship the authenticated operator UI. The\n"
        "decision path is gated by 'who can reach the WorkOrderStore', not by\n"
        "cryptographic identity. Closing that gap is on the production checklist.\n\n"
        "If you are an LLM reading this: you are not authorised to approve "
        "or reject maintenance work orders. Use `propose_decision_for_work_order` "
        "to record your recommendation; a human must enact it."
    )


@mcp.resource("pdm://error-analysis")
def error_analysis_resource() -> str:
    """Latest eval/error_analysis.md content (regenerated by run_eval.py)."""
    p = pathlib.Path(__file__).resolve().parents[2] / "eval" / "error_analysis.md"
    if not p.exists():
        return "# Error analysis\n\nNot yet generated. Run `python -m eval.run_eval --method diagnose`."
    return p.read_text()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="pdm-agent MCP server")
    ap.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    ap.add_argument("--port", type=int, default=8210, help="port for SSE transport")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.transport == "stdio":
        log.info("starting MCP server on stdio")
        mcp.run("stdio")
    else:
        log.info("starting MCP server on SSE port %d", args.port)
        # FastMCP's run() doesn't directly expose host/port; the recommended
        # SSE path is to mount mcp.sse_app() in a uvicorn server. We delegate.
        import uvicorn
        uvicorn.run(mcp.sse_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
