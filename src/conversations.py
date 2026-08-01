from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .config import CONVERSATION_DB_PATH, CONVERSATION_HISTORY_MESSAGES


@dataclass(frozen=True)
# Represent one stored user or assistant message.
class ConversationMessage:
    role: str
    content: str


# Persist and retrieve channel conversation history with SQLite.
class ConversationStore:
    """Persist channel conversations in a small local SQLite database."""

    # Configure the database path and bounded history window.
    def __init__(
        self,
        path: Path = CONVERSATION_DB_PATH,
        history_limit: int = CONVERSATION_HISTORY_MESSAGES,
    ) -> None:
        self.path = path
        self.history_limit = history_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # Open a SQLite connection with named-row access.
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    # Create the message table and lookup index when absent.
    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_lookup
                ON conversation_messages (
                    channel, conversation_id, user_id, id
                )
                """
            )

    # Load the most recent messages for one isolated conversation.
    def history(
        self,
        *,
        channel: str,
        conversation_id: str,
        user_id: str | None,
    ) -> list[dict[str, str]]:
        effective_user_id = user_id or ""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM conversation_messages
                    WHERE channel = ?
                      AND conversation_id = ?
                      AND user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (
                    channel,
                    conversation_id,
                    effective_user_id,
                    self.history_limit,
                ),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    # Atomically append a completed user and assistant exchange.
    def append_exchange(
        self,
        *,
        channel: str,
        conversation_id: str,
        user_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        effective_user_id = user_id or ""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_messages (
                    channel, conversation_id, user_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        channel,
                        conversation_id,
                        effective_user_id,
                        "user",
                        question,
                        timestamp,
                    ),
                    (
                        channel,
                        conversation_id,
                        effective_user_id,
                        "assistant",
                        answer,
                        timestamp,
                    ),
                ],
            )
