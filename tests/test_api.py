from fastapi.testclient import TestClient

from src.api import create_app
from src.channels import ChannelService


class FakeAgent:
    def answer(self, question, history=None):
        return {
            "query": question,
            "answer": "A grounded answer [S1].",
            "citations": {
                "S1": {
                    "source": "guide.pdf",
                    "relative_path": "guide.pdf",
                    "page": 2,
                    "chunk": 1,
                    "retrieval_score": 0.9,
                }
            },
            "sources": [],
            "metrics": {
                "retrieval_time_s": 0.01,
                "generation_time_s": 0.02,
                "total_time_s": 0.03,
                "cache_hit": False,
            },
        }


def client():
    from tempfile import TemporaryDirectory

    from src.conversations import ConversationStore

    temporary_directory = TemporaryDirectory()
    store = ConversationStore(
        __import__("pathlib").Path(temporary_directory.name) / "conversations.sqlite3"
    )
    api = TestClient(create_app(ChannelService(FakeAgent(), store)))
    api._conversation_test_directory = temporary_directory
    return api


def test_web_channel_returns_typed_rag_response():
    with client() as api:
        response = api.post(
            "/api/v1/channels/web/messages",
            json={
                "message": "What is covered?",
                "conversation_id": "web-1",
                "user_id": "user-1",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["channel"] == "web"
    assert payload["answer"] == "A grounded answer [S1]."
    assert payload["citations"]["S1"]["source"] == "guide.pdf"


def test_teams_channel_removes_bot_mention_and_builds_reply_activity():
    with client() as api:
        response = api.post(
            "/api/v1/channels/teams/messages",
            json={
                "type": "message",
                "id": "activity-1",
                "text": "<at>RAG bot</at> What is covered?",
                "conversation": {"id": "teams-1"},
                "from": {"id": "user-1", "name": "User"},
                "recipient": {"id": "bot-1", "name": "RAG bot"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["replyToId"] == "activity-1"
    assert "[S1] guide.pdf, page 2" in payload["text"]
    assert payload["recipient"]["id"] == "user-1"


def test_teams_ignores_non_message_activity():
    with client() as api:
        response = api.post(
            "/api/v1/channels/teams/messages",
            json={"type": "conversationUpdate"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
