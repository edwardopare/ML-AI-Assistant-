"""Conversation-memory housekeeping.

Provides ``cleanup_old_conversations`` which removes rows from the SQLite
conversation store that are older than ``CONVERSATION_MAX_AGE_DAYS``.

This is invoked:
- Automatically by the APScheduler job registered in ``api.create_app`` (once per day).
- From the CLI via ``python main.py cleanup-conversations``.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import CONVERSATION_DB_PATH, CONVERSATION_MAX_AGE_DAYS

logger = logging.getLogger(__name__)


def cleanup_old_conversations(
    max_age_days: int = CONVERSATION_MAX_AGE_DAYS,
    db_path: Path = CONVERSATION_DB_PATH,
) -> int:
    """Delete conversation rows older than *max_age_days*.

    Returns the number of rows deleted.  Safe to call when the database does
    not exist yet (returns 0 in that case).
    """
    if not db_path.exists():
        logger.debug("Conversation database does not exist; skipping cleanup")
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    try:
        connection = sqlite3.connect(db_path, timeout=10)
        try:
            cursor = connection.execute(
                "DELETE FROM conversation_messages WHERE created_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            connection.commit()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.error("Conversation cleanup failed", extra={"error": str(exc)})
        return 0

    if deleted:
        logger.info(
            "Cleaned up old conversation messages",
            extra={"deleted_rows": deleted, "cutoff_date": cutoff[:10]},
        )
    else:
        logger.debug("No old conversation messages to clean up")
    return deleted
