from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Protocol
from .agent import RAGAgent

_MENTION_RE = re.compile(r"<at>.*?</at>", re.IGNORECASE)


class AnsweringAgent(Protocol):
    def answer(self, question: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ChannelRequest:
    channel: str
    question: str
    conversation_id: str | None = None
    user_id: str | None = None


class ChannelService:
    """Route every client channel through the same RAG pipeline."""

    def __init__(self, agent: AnsweringAgent | None = None) -> None:
        self._agent = agent

    @property
    def agent(self) -> AnsweringAgent:
        # Delay heavyweight embedding/RAG initialization until the first query.
        if self._agent is None:
            self._agent = RAGAgent()
        return self._agent

    def ask(self, request: ChannelRequest) -> dict[str, Any]:
        result = self.agent.answer(request.question.strip())
        return {
            **result,
            "channel": request.channel,
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
        }


def teams_question(activity: dict[str, Any]) -> str:
    """Extract user text from a Bot Framework message activity."""
    text = str(activity.get("text") or "")
    return _MENTION_RE.sub("", text).strip()


def teams_citation_lines(citations: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for citation_id, citation in citations.items():
        source = citation.get("source") or "Unknown source"
        page = citation.get("page")
        suffix = f", page {page}" if page is not None else ""
        lines.append(f"[{citation_id}] {source}{suffix}")
    return lines


def teams_reply(activity: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Create a Bot Framework reply activity suitable for Microsoft Teams."""
    citations = teams_citation_lines(result.get("citations", {}))
    text = str(result["answer"])
    if citations:
        text = f"{text}\n\nSources:\n" + "\n".join(citations)
    return {
        "type": "message",
        "text": text,
        "textFormat": "markdown",
        "replyToId": activity.get("id"),
        "conversation": activity.get("conversation"),
        "from": activity.get("recipient"),
        "recipient": activity.get("from"),
    }
