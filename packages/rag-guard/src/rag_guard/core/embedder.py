from __future__ import annotations

import hashlib
import os
from typing import Protocol, cast, runtime_checkable

import numpy as np

from rag_guard.core.hasher import normalize_text


@runtime_checkable
class Embedder(Protocol):
    """Structural interface matching LangChain's `Embeddings` ABC, so any
    LangChain embeddings instance (OpenAI, HuggingFace, etc.) satisfies it
    without a hard dependency on `langchain`."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder using hashed character n-gram
    features (the "hashing trick"). It is not a semantic model — it exists so
    the guard has a zero-cost, offline default that needs no API key or
    downloaded weights. Swap in a real `Embeddings` implementation (via
    LangChain) for production-grade semantic drift detection.
    """

    def __init__(self, dimensions: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> list[str]:
        normalized = normalize_text(text).lower()
        lo, hi = self.ngram_range
        grams = [
            normalized[i : i + n]
            for n in range(lo, hi + 1)
            for i in range(len(normalized) - n + 1)
        ]
        return grams or [normalized]

    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for gram in self._ngrams(text):
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class OpenAIEmbedder:
    """Real semantic embeddings via OpenAI's Embeddings API.

    Deliberately dependency-light -- a couple of plain HTTP calls (`httpx`,
    imported lazily so it's only required if this class is actually used),
    not the `openai` SDK and definitely not a locally-loaded model. A local
    sentence-transformers/ONNX model would be "free" per call but drags
    torch or onnxruntime into the deployment bundle; for a Vercel serverless
    function that's the difference between deploying and blowing the
    function size limit. An HTTP call costs a fraction of a cent and keeps
    the bundle tiny.

    Requires `OPENAI_API_KEY` (or pass `api_key=`).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAIEmbedder requires an API key: pass api_key=... or set "
                "the OPENAI_API_KEY environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        items = response.json()["data"]
        # The API returns embeddings in input order, but sort defensively on
        # the index it echoes back rather than trusting that.
        ordered = sorted(items, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


def normalize_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        return array.tolist()
    return cast(list[float], (array / norm).tolist())


def embed_and_normalize(embedder: Embedder, text: str) -> list[float]:
    """The chunk's "semantic vector signature": its embedding, L2-normalized
    so cosine similarity reduces to a dot product at comparison time."""
    return normalize_vector(embedder.embed_query(text))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
