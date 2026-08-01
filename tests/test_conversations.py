from pathlib import Path

from src.channels import ChannelRequest, ChannelService
from src.conversations import ConversationStore


class HistoryRecordingAgent:
    def __init__(self):
        self.histories = []

    def answer(self, question, history=None):
        self.histories.append(history or [])
        return {
            "query": question,
            "answer": f"Answer to {question}",
            "citations": {},
            "sources": [],
            "metrics": {"total_time_s": 0.0, "cache_hit": False},
        }


def test_channel_service_persists_and_reuses_conversation_history(tmp_path: Path):
    agent = HistoryRecordingAgent()
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    service = ChannelService(agent, store)
    identity = {
        "channel": "web",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
    }

    first = service.ask(ChannelRequest(question="First question", **identity))
    second = service.ask(ChannelRequest(question="Follow up", **identity))

    assert first["history_messages_used"] == 0
    assert second["history_messages_used"] == 2
    assert agent.histories[1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Answer to First question"},
    ]


def test_conversations_are_isolated_by_user_and_channel(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.append_exchange(
        channel="web",
        conversation_id="shared",
        user_id="user-1",
        question="Private question",
        answer="Private answer",
    )

    assert store.history(
        channel="web", conversation_id="shared", user_id="user-2"
    ) == []
    assert store.history(
        channel="teams", conversation_id="shared", user_id="user-1"
    ) == []
