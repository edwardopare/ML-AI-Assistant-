"""Conversation persistence service boundary.

Allowed callers: API layer (channels/api) only.
Do NOT import from ingest, retrieval, or generation services.
"""
from __future__ import annotations
from pathlib import Path
from ..conversations import ConversationStore
from ..memory_cleanup import cleanup_old_conversations
from ..config import CONVERSATION_DB_PATH, CONVERSATION_HISTORY_MESSAGES, CONVERSATION_MAX_AGE_DAYS


class ConversationService:
    """Facade over the SQLite conversation store and cleanup job."""

    def __init__(
        self,
        path: Path = CONVERSATION_DB_PATH,
        history_limit: int = CONVERSATION_HISTORY_MESSAGES,
    ) -> None:
        self._store = ConversationStore(path=path, history_limit=history_limit)

    def history(
        self,
        *,
        channel: str,
        conversation_id: str,
        user_id: str | None,
    ) -> list[dict[str, str]]:
        return self._store.history(
            channel=channel, conversation_id=conversation_id, user_id=user_id
        )

    def append_exchange(
        self,
        *,
        channel: str,
        conversation_id: str,
        user_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        self._store.append_exchange(
            channel=channel,
            conversation_id=conversation_id,
            user_id=user_id,
            question=question,
            answer=answer,
        )

    def cleanup(self, max_age_days: int = CONVERSATION_MAX_AGE_DAYS) -> int:
        """Delete conversation rows older than *max_age_days*. Returns row count."""
        return cleanup_old_conversations(max_age_days=max_age_days)
