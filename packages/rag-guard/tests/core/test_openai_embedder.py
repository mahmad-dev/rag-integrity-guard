from __future__ import annotations

from typing import Any

import httpx
import pytest

from rag_guard.core.embedder import OpenAIEmbedder


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict[str, Any]:
        return self._payload


def _embedding_response(vectors: list[list[float]], *, shuffled: bool = False) -> _FakeResponse:
    items = [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]
    if shuffled:
        items = list(reversed(items))
    return _FakeResponse({"data": items})


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires an API key"):
        OpenAIEmbedder()


def test_explicit_api_key_is_accepted() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    assert embedder.api_key == "sk-test"


def test_env_var_api_key_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    embedder = OpenAIEmbedder()
    assert embedder.api_key == "sk-from-env"


def test_embed_query_returns_the_mocked_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _embedding_response([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OpenAIEmbedder(api_key="sk-test")
    result = embedder.embed_query("hello world")

    assert result == [0.1, 0.2, 0.3]
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"] == {"model": "text-embedding-3-small", "input": ["hello world"]}


def test_embed_documents_reorders_by_index_defensively(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        return _embedding_response([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], shuffled=True)

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OpenAIEmbedder(api_key="sk-test")
    result = embedder.embed_documents(["a", "b", "c"])

    assert result == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        return _FakeResponse({}, status_code=401)

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OpenAIEmbedder(api_key="sk-bad")
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("hello")


def test_custom_model_and_base_url_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        captured["url"] = url
        captured["json"] = json
        return _embedding_response([[0.5]])

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OpenAIEmbedder(
        model="text-embedding-3-large", api_key="sk-test", base_url="https://example.com/v1/"
    )
    embedder.embed_query("hi")

    assert captured["url"] == "https://example.com/v1/embeddings"
    assert captured["json"]["model"] == "text-embedding-3-large"
