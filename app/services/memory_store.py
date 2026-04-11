"""SQLite-backed long-term memory store for chat sessions."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any


class MemoryStore:
    """Persist chat sessions, usage, and task events in SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_usage (
                    session_id TEXT PRIMARY KEY,
                    estimated_tokens REAL NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    token_budget REAL NOT NULL,
                    cost_budget_usd REAL NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model TEXT,
                    reason TEXT,
                    step_index INTEGER,
                    total_steps INTEGER,
                    retry INTEGER,
                    meta_json TEXT,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_index INTEGER,
                    step_task TEXT,
                    tool TEXT,
                    source TEXT,
                    status TEXT,
                    reason TEXT,
                    model TEXT,
                    output_excerpt TEXT,
                    created_at INTEGER NOT NULL
                )
                """
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if not session_id or not content:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO session_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, int(time.time())),
            )

    def load_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        if not session_id:
            return []
        with self._lock, self._connect() as conn:
            sql = "SELECT role, content FROM session_messages WHERE session_id = ? ORDER BY id ASC"
            params: tuple[Any, ...] = (session_id,)
            if isinstance(limit, int) and limit > 0:
                sql += " LIMIT ?"
                params = (session_id, limit)
            rows = conn.execute(sql, params).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def upsert_usage(self, session_id: str, usage: dict[str, float]) -> None:
        if not session_id:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_usage (session_id, estimated_tokens, estimated_cost_usd, token_budget, cost_budget_usd, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    estimated_tokens=excluded.estimated_tokens,
                    estimated_cost_usd=excluded.estimated_cost_usd,
                    token_budget=excluded.token_budget,
                    cost_budget_usd=excluded.cost_budget_usd,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    float(usage.get("estimated_tokens", 0.0)),
                    float(usage.get("estimated_cost_usd", 0.0)),
                    float(usage.get("token_budget", 0.0)),
                    float(usage.get("cost_budget_usd", 0.0)),
                    int(time.time()),
                ),
            )

    def load_usage(self, session_id: str) -> dict[str, float] | None:
        if not session_id:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT estimated_tokens, estimated_cost_usd, token_budget, cost_budget_usd
                FROM session_usage WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "estimated_tokens": float(row[0]),
            "estimated_cost_usd": float(row[1]),
            "token_budget": float(row[2]),
            "cost_budget_usd": float(row[3]),
        }

    def append_task_event(self, session_id: str, event: dict[str, Any]) -> None:
        if not session_id:
            return
        meta_json = json.dumps(event.get("meta", {}), ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_events (
                    session_id, stage, message, model, reason, step_index, total_steps, retry, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(event.get("stage", "")),
                    str(event.get("message", "")),
                    event.get("model"),
                    event.get("reason"),
                    event.get("step_index"),
                    event.get("total_steps"),
                    event.get("retry"),
                    meta_json,
                    int(time.time()),
                ),
            )

    def clear_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_usage WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM task_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM evidence_items WHERE session_id = ?", (session_id,))

    def append_evidence(self, session_id: str, item: dict[str, Any]) -> None:
        if not session_id:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_items (
                    session_id, step_index, step_task, tool, source, status, reason, model, output_excerpt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    item.get("step_index"),
                    item.get("step_task"),
                    item.get("tool"),
                    item.get("source"),
                    item.get("status"),
                    item.get("reason"),
                    item.get("model"),
                    item.get("output_excerpt"),
                    int(time.time()),
                ),
            )
