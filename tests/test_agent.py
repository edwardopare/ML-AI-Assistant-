import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.agent import (
    NO_EVIDENCE_ANSWER,
    OpenRouterClient,
    OpenRouterError,
    ResponseCache,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = lines or []
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_no_evidence_skips_openrouter():
    client = OpenRouterClient(api_key="test", session=FakeSession([]))
    answer, citations = client.answer_question("Unknown?", [])

    assert answer == NO_EVIDENCE_ANSWER
    assert citations == {}


def test_non_streaming_answer_validates_citations():
    response = FakeResponse(
        payload={"choices": [{"message": {"content": "Grounded fact [S1]."}}]}
    )
    client = OpenRouterClient(api_key="test", session=FakeSession([response]))
    document = Document(
        page_content="Grounded fact.",
        metadata={"source": "doc.pdf", "page": 3, "chunk": 1},
    )

    answer, citations = client.answer_question("What is the fact?", [document])

    assert answer == "Grounded fact [S1]."
    assert citations["S1"]["page"] == 3
    assert response.closed


def test_stream_parser_rejects_error_event():
    line = "data: " + json.dumps({"error": {"message": "failure"}})
    response = FakeResponse(lines=[line])
    client = OpenRouterClient(api_key="test", session=FakeSession([response]))
    stream = client.generate([{"role": "user", "content": "test"}], stream=True)

    with pytest.raises(OpenRouterError, match="streaming"):
        list(stream)


def test_authentication_error_is_safe():
    client = OpenRouterClient(
        api_key="bad",
        max_retries=0,
        session=FakeSession([FakeResponse(status_code=401)]),
    )
    with pytest.raises(OpenRouterError, match="authentication"):
        client.generate([{"role": "user", "content": "test"}])


def test_cache_key_changes_with_corpus_and_cache_is_atomic(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    first = ResponseCache.make_key("Question", {"corpus": "one"})
    second = ResponseCache.make_key("Question", {"corpus": "two"})
    assert first != second

    cache.set(first, {"answer": "value"})
    assert ResponseCache(tmp_path).get(first) == {"answer": "value"}
    assert not list(tmp_path.glob("*.tmp"))
