from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Protocol
from .agent import RAGAgent
from .conversations import ConversationStore

_MENTION_RE = re.compile(r"<at>.*?</at>", re.IGNORECASE)


# Define the interface required from a RAG answering agent.
class AnsweringAgent(Protocol):
    # Answer a question with optional conversation history.
    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
# Carry a normalized request between channel adapters and the RAG service.
class ChannelRequest:
    channel: str
    question: str
    conversation_id: str | None = None
    user_id: str | None = None


# Coordinate conversation memory and the shared RAG pipeline.
class ChannelService:
    """Route every client channel through the same RAG pipeline."""

    # Configure lazily created or injected channel dependencies.
    def __init__(
        self,
        agent: AnsweringAgent | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._agent = agent
        self._conversation_store = conversation_store

    @property
    # Return the shared RAG agent, creating it on first use.
    def agent(self) -> AnsweringAgent:
        # Delay heavyweight embedding/RAG initialization until the first query.
        if self._agent is None:
            self._agent = RAGAgent()
        return self._agent

    @property
    # Return the conversation store, creating it on first use.
    def conversation_store(self) -> ConversationStore:
        if self._conversation_store is None:
            self._conversation_store = ConversationStore()
        return self._conversation_store

    # Answer a request and persist its conversational exchange when identified.
    def ask(self, request: ChannelRequest) -> dict[str, Any]:
        question = request.question.strip()
        history: list[dict[str, str]] = []
        if request.conversation_id:
            history = self.conversation_store.history(
                channel=request.channel,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
            )
        result = self.agent.answer(question, history=history)
        if request.conversation_id:
            self.conversation_store.append_exchange(
                channel=request.channel,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                question=question,
                answer=str(result["answer"]),
            )
        return {
            **result,
            "channel": request.channel,
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "history_messages_used": len(history),
        }


# Extract question text from a Teams message activity.
def teams_question(activity: dict[str, Any]) -> str:
    """Extract user text from a Bot Framework message activity."""
    text = str(activity.get("text") or "")
    return _MENTION_RE.sub("", text).strip()


# Format citation metadata as readable Teams source lines.
def teams_citation_lines(citations: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for citation_id, citation in citations.items():
        source = citation.get("source") or "Unknown source"
        page = citation.get("page")
        suffix = f", page {page}" if page is not None else ""
        lines.append(f"[{citation_id}] {source}{suffix}")
    return lines


# Convert a RAG result into a Teams reply activity.
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
