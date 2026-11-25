"""SQLite persistence layer for sessions and multi-turn chat messages."""

from datetime import datetime, timezone
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    """Return current UTC server timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe SQLite repository for chat sessions and message history."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        # Keep a persistent connection when using :memory: so schema/data survive across calls.
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_if_not_memory(self, conn: sqlite3.Connection) -> None:
        if self._memory_conn is None:
            conn.close()

    def init_schema(self) -> None:
        """Create required tables and indexes on startup."""
        with self._lock:
            conn = self._connect()
            try:
                if self._memory_conn is None:
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                        ON messages (session_id, id);
                    """
                )
                conn.commit()
            finally:
                self._close_if_not_memory(conn)

    def check_health(self) -> bool:
        """Verify SQLite connection is operational."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT 1").fetchone()
                return row is not None and row[0] == 1
            finally:
                self._close_if_not_memory(conn)

    def get_session_context(
        self, session_id: str
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Return (saved_system_prompt, history_messages) for a session."""
        with self._lock:
            conn = self._connect()
            try:
                session_row = conn.execute(
                    "SELECT system_prompt FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                saved_system_prompt = (
                    session_row["system_prompt"] if session_row is not None else None
                )

                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
                history = [
                    {"role": row["role"], "content": row["content"]} for row in rows
                ]
                return saved_system_prompt, history
            finally:
                self._close_if_not_memory(conn)

    def save_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        metadata: Dict[str, Any],
        system_prompt: Optional[str],
    ) -> None:
        """Persist a completed user + assistant turn atomically."""
        now = utc_now_iso()
        metadata_json = json.dumps(metadata)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt),
                        updated_at = excluded.updated_at
                    """,
                    (session_id, system_prompt, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'user', ?, NULL, ?)
                    """,
                    (session_id, user_content, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'assistant', ?, ?, ?)
                    """,
                    (session_id, assistant_content, metadata_json, now),
                )
                conn.commit()
            finally:
                self._close_if_not_memory(conn)

    def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return oldest-first stored messages for a session, or None if unknown."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT role, content, metadata
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
                if not rows:
                    return None

                result: List[Dict[str, Any]] = []
                for row in rows:
                    item: Dict[str, Any] = {
                        "role": row["role"],
                        "content": row["content"],
                    }
                    if row["role"] == "assistant":
                        parsed_meta: Dict[str, Any] = {}
                        if row["metadata"]:
                            parsed_meta = json.loads(row["metadata"])
                        item["metadata"] = parsed_meta
                    result.append(item)
                return result
            finally:
                self._close_if_not_memory(conn)
