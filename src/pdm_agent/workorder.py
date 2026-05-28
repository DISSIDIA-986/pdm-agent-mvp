"""Work-order persistence + audit log (SQLite, no external DB for MVP).

Schema (intentionally minimal):
  work_orders
    id (str PK)            — UUID
    sample_id (str)        — links back to diagnosis
    asset_id (str)         — e.g. "BESS_Site_A/CoolingPump_01"
    severity (str)         — normal|watch|alert|critical
    predicted_class (str)
    confidence (float)
    summary (str)          — LLM- or rule-generated natural language
    evidence_json (str)    — full diagnosis dict for traceability
    status (str)           — draft|pending_approval|approved|rejected|closed
    created_at (str ISO)
    decided_at (str ISO)
    decided_by (str)       — "human:<operator_id>" or "auto:<reason>"
    decision_note (str)

  audit_log
    id (int PK)
    work_order_id (str)
    event (str)            — created|approval_requested|approved|rejected|closed
    payload_json (str)
    at (str ISO)
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Iterator, Literal

WorkOrderStatus = Literal["draft", "pending_approval", "approved", "rejected", "closed"]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS work_orders (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT,
    decision_note TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_order_id TEXT NOT NULL,
    event TEXT NOT NULL,
    payload_json TEXT,
    at TEXT NOT NULL,
    FOREIGN KEY (work_order_id) REFERENCES work_orders(id)
);

CREATE INDEX IF NOT EXISTS ix_audit_workorder ON audit_log(work_order_id);
CREATE INDEX IF NOT EXISTS ix_workorders_status ON work_orders(status);
"""


@dataclasses.dataclass
class WorkOrder:
    id: str
    sample_id: str
    asset_id: str
    severity: str
    predicted_class: str
    confidence: float
    summary: str
    evidence: dict
    status: WorkOrderStatus
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    decision_note: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


class WorkOrderStore:
    """Thin SQLite wrapper. Synchronous on purpose — SQLite is fine for MVP."""

    def __init__(self, db_path: pathlib.Path | str = "pdm-agent.db") -> None:
        self.db_path = pathlib.Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL + busy timeout pragmas set."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _cursor(self, *, immediate: bool = False) -> Iterator[sqlite3.Cursor]:
        """Per-call connection with explicit transaction control.

        `immediate=True` upgrades to BEGIN IMMEDIATE to acquire the reserved
        write lock atomically (used for state transitions to avoid TOCTOU).
        Connection runs in autocommit (`isolation_level=None`) so we control
        BEGIN/COMMIT explicitly.
        """
        conn = self._connect()
        conn.isolation_level = None  # autocommit; we BEGIN manually
        in_tx = False
        try:
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("BEGIN")
            in_tx = True
            cur = conn.cursor()
            yield cur
            conn.execute("COMMIT")
            in_tx = False
        except Exception:
            if in_tx:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
            raise
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def create_draft(
        self,
        *,
        sample_id: str,
        asset_id: str,
        severity: str,
        predicted_class: str,
        confidence: float,
        summary: str,
        evidence: dict,
    ) -> WorkOrder:
        wo = WorkOrder(
            id=str(uuid.uuid4()),
            sample_id=sample_id,
            asset_id=asset_id,
            severity=severity,
            predicted_class=predicted_class,
            confidence=confidence,
            summary=summary,
            evidence=evidence,
            status="draft",
            created_at=self._now(),
        )
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO work_orders
                   (id, sample_id, asset_id, severity, predicted_class, confidence,
                    summary, evidence_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    wo.id, wo.sample_id, wo.asset_id, wo.severity, wo.predicted_class,
                    wo.confidence, wo.summary, json.dumps(wo.evidence), wo.status,
                    wo.created_at,
                ),
            )
            cur.execute(
                "INSERT INTO audit_log (work_order_id, event, payload_json, at) VALUES (?, ?, ?, ?)",
                (wo.id, "created", json.dumps({"severity": severity}), wo.created_at),
            )
        return wo

    def request_approval(self, wo_id: str) -> None:
        """Move draft -> pending_approval. Idempotent: no-op if already pending."""
        ts = self._now()
        with self._cursor() as cur:
            cur.execute("SELECT status FROM work_orders WHERE id = ?", (wo_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"work order {wo_id} not found")
            current = row[0]
            if current == "pending_approval":
                return  # already requested — workflow may have re-entered this node
            if current != "draft":
                raise ValueError(
                    f"cannot request approval for {wo_id} (status={current})"
                )
            cur.execute(
                "UPDATE work_orders SET status = ? WHERE id = ?",
                ("pending_approval", wo_id),
            )
            cur.execute(
                "INSERT INTO audit_log (work_order_id, event, payload_json, at) VALUES (?, ?, ?, ?)",
                (wo_id, "approval_requested", None, ts),
            )

    def decide(
        self,
        wo_id: str,
        *,
        approve: bool,
        decided_by: str,
        note: str | None = None,
    ) -> None:
        """Atomic decision transition. Only succeeds from draft/pending_approval."""
        ts = self._now()
        new_status = "approved" if approve else "rejected"
        with self._cursor(immediate=True) as cur:
            cur.execute(
                """UPDATE work_orders
                   SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?
                   WHERE id = ? AND status IN ('pending_approval', 'draft')""",
                (new_status, ts, decided_by, note, wo_id),
            )
            if cur.rowcount != 1:
                cur.execute("SELECT status FROM work_orders WHERE id = ?", (wo_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"work order {wo_id} not found")
                raise ValueError(f"cannot decide {wo_id} in status {row[0]}")
            cur.execute(
                "INSERT INTO audit_log (work_order_id, event, payload_json, at) VALUES (?, ?, ?, ?)",
                (
                    wo_id,
                    new_status,
                    json.dumps({"decided_by": decided_by, "note": note}),
                    ts,
                ),
            )

    def get(self, wo_id: str) -> WorkOrder | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM work_orders WHERE id = ?", (wo_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            data = dict(zip(cols, row))
        return WorkOrder(
            id=data["id"],
            sample_id=data["sample_id"],
            asset_id=data["asset_id"],
            severity=data["severity"],
            predicted_class=data["predicted_class"],
            confidence=data["confidence"],
            summary=data["summary"],
            evidence=json.loads(data["evidence_json"]),
            status=data["status"],
            created_at=data["created_at"],
            decided_at=data["decided_at"],
            decided_by=data["decided_by"],
            decision_note=data["decision_note"],
        )

    def list_by_status(self, status: WorkOrderStatus | None = None) -> list[WorkOrder]:
        with self._cursor() as cur:
            if status:
                cur.execute("SELECT * FROM work_orders WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cur.execute("SELECT * FROM work_orders ORDER BY created_at DESC")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        out: list[WorkOrder] = []
        for row in rows:
            data = dict(zip(cols, row))
            out.append(
                WorkOrder(
                    id=data["id"], sample_id=data["sample_id"], asset_id=data["asset_id"],
                    severity=data["severity"], predicted_class=data["predicted_class"],
                    confidence=data["confidence"], summary=data["summary"],
                    evidence=json.loads(data["evidence_json"]), status=data["status"],
                    created_at=data["created_at"], decided_at=data["decided_at"],
                    decided_by=data["decided_by"], decision_note=data["decision_note"],
                )
            )
        return out

    def audit_trail(self, wo_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT event, payload_json, at FROM audit_log WHERE work_order_id = ? ORDER BY id",
                (wo_id,),
            )
            rows = cur.fetchall()
        return [
            {"event": ev, "payload": json.loads(p) if p else None, "at": at}
            for ev, p, at in rows
        ]
