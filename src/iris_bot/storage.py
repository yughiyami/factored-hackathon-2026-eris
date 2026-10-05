"""Runtime SQLite store: disputes, handoffs, conversation state, case outcomes, audit log/traces."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS disputes (
    dispute_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    customer_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    amount REAL, currency TEXT, amount_usd REAL,
    status TEXT NOT NULL,
    conversation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    conversation_id TEXT, customer_id TEXT, trigger TEXT, priority TEXT,
    payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    conversation_id TEXT PRIMARY KEY,
    language TEXT, intent TEXT, outcome TEXT, handed_off INTEGER, handoff_trigger TEXT,
    resolved INTEGER, feedback TEXT, csat INTEGER, turns INTEGER, clarification_turns INTEGER,
    cost_usd REAL, started_at TEXT, ended_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, trace_id TEXT, conversation_id TEXT, customer_id TEXT,
    step TEXT NOT NULL, tool TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL, outcome TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_conv ON audit_log(conversation_id);
CREATE INDEX IF NOT EXISTS ix_disputes_customer ON disputes(customer_id);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._con.execute(sql, params)

    def _all(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, params).fetchall()]

    # --- audit ------------------------------------------------------------------------------
    def audit(self, *, step: str, trace_id: str | None = None, conversation_id: str | None = None,
              customer_id: str | None = None, tool: str | None = None, latency_ms: float | None = None,
              input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
              outcome: str | None = None, detail: dict[str, Any] | None = None) -> None:
        self._exec(
            "INSERT INTO audit_log(ts,trace_id,conversation_id,customer_id,step,tool,latency_ms,"
            "input_tokens,output_tokens,cost_usd,outcome,detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), trace_id, conversation_id, customer_id, step, tool, latency_ms, input_tokens,
             output_tokens, cost_usd, outcome, json.dumps(detail or {}, ensure_ascii=False, default=str)),
        )

    def audit_rows(self, conversation_id: str | None = None, step: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM audit_log WHERE 1=1", []
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if step:
            sql += " AND step = ?"
            params.append(step)
        return self._all(sql + " ORDER BY id", params)

    # --- disputes ---------------------------------------------------------------------------
    def insert_dispute(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Idempotent insert keyed by idempotency_key; returns the stored row."""
        existing = self.dispute_by_key(rec["idempotency_key"])
        if existing:
            return existing
        rec = {"dispute_id": "DSP-" + uuid.uuid4().hex[:12].upper(), "created_at": now_iso(), **rec}
        cols = ",".join(rec)
        self._exec(f"INSERT OR IGNORE INTO disputes({cols}) VALUES ({','.join('?' * len(rec))})",
                   list(rec.values()))
        return self.dispute_by_key(rec["idempotency_key"])  # type: ignore[return-value]

    def dispute_by_key(self, key: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM disputes WHERE idempotency_key = ?", (key,))
        return rows[0] if rows else None

    def dispute(self, dispute_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM disputes WHERE dispute_id = ?", (dispute_id,))
        return rows[0] if rows else None

    def disputes_for(self, customer_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM disputes WHERE customer_id = ? ORDER BY created_at DESC",
                         (customer_id,))

    # --- handoffs ---------------------------------------------------------------------------
    def insert_handoff(self, handoff_id: str, conversation_id: str, customer_id: str | None,
                       trigger: str, priority: str, payload: dict[str, Any]) -> None:
        self._exec("INSERT OR REPLACE INTO handoffs VALUES (?,?,?,?,?,?,?,?)",
                   (handoff_id, conversation_id, customer_id, trigger, priority,
                    json.dumps(payload, ensure_ascii=False, default=str), "queued", now_iso()))

    def handoff(self, handoff_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM handoffs WHERE handoff_id = ?", (handoff_id,))
        if not rows:
            return None
        row = rows[0]
        row["payload"] = json.loads(row["payload"])
        return row

    def handoffs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM handoffs ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # --- conversations ----------------------------------------------------------------------
    def load_conversation(self, conversation_id: str) -> str | None:
        rows = self._all("SELECT state FROM conversations WHERE conversation_id = ?", (conversation_id,))
        return rows[0]["state"] if rows else None

    def save_conversation(self, conversation_id: str, state_json: str) -> None:
        self._exec("INSERT OR REPLACE INTO conversations VALUES (?,?,?)",
                   (conversation_id, state_json, now_iso()))

    def delete_conversation(self, conversation_id: str) -> None:
        self._exec("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))

    # --- case outcomes ----------------------------------------------------------------------
    def upsert_case(self, rec: dict[str, Any]) -> None:
        cols = ",".join(rec)
        self._exec(f"INSERT OR REPLACE INTO cases({cols}) VALUES ({','.join('?' * len(rec))})",
                   list(rec.values()))

    def cases(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM cases")
