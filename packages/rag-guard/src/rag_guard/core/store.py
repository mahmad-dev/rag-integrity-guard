from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from rag_guard.core.schema import ChunkFingerprint


@runtime_checkable
class FingerprintStore(Protocol):
    """Structural interface for fingerprint lookup, so a retriever can be
    backed by anything from an in-memory dict to a database table."""

    def get(self, chunk_id: str) -> ChunkFingerprint | None: ...


class InMemoryFingerprintStore:
    """Zero-cost default fingerprint store: a plain in-memory dict. Fine for
    the eval harness and single-instance runs; not shared across concurrent
    serverless invocations, so a production deployment should swap in a
    database-backed store behind the same `FingerprintStore` interface.
    """

    def __init__(self, fingerprints: Iterable[ChunkFingerprint] | None = None) -> None:
        self._by_chunk_id: dict[str, ChunkFingerprint] = {}
        if fingerprints:
            self.bulk_put(fingerprints)

    def put(self, fingerprint: ChunkFingerprint) -> None:
        self._by_chunk_id[fingerprint.chunk_id] = fingerprint

    def bulk_put(self, fingerprints: Iterable[ChunkFingerprint]) -> None:
        for fingerprint in fingerprints:
            self.put(fingerprint)

    def get(self, chunk_id: str) -> ChunkFingerprint | None:
        return self._by_chunk_id.get(chunk_id)

    def __len__(self) -> int:
        return len(self._by_chunk_id)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._by_chunk_id
