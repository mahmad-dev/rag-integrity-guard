"""Process-wide, in-memory application state.

Everything here lives in the memory of a single warm serverless instance: a
cold start (or a different concurrent instance) gets a fresh, empty store.
That's the explicit zero-cost tradeoff of the Vercel hobby-tier deployment
target -- swap in a persistent Chroma client and a database-backed
`FingerprintStore` for anything beyond a demo.

Assumes `sys.path` already has `packages/rag-guard/src` on it (index.py, the
real entrypoint, sets that up before importing this module).
"""

from __future__ import annotations

import uuid

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from rag_guard.core.embedder import Embedder, HashingEmbedder
from rag_guard.core.store import InMemoryFingerprintStore

COLLECTION_NAME_PREFIX = "rag_guard_documents"


def _new_collection(client: ClientAPI, name: str) -> Collection:
    collection: Collection = client.get_or_create_collection(
        name,
        embedding_function=None,  # we supply our own embeddings, computed via `embedder`
        metadata={"hnsw:space": "cosine"},
    )
    return collection


class AppState:
    def __init__(self) -> None:
        self.embedder: Embedder = HashingEmbedder()
        self.fingerprint_store = InMemoryFingerprintStore()
        self._client = chromadb.EphemeralClient()
        # chromadb's EphemeralClient caches its in-memory backend per process
        # keyed by settings, not per client instance -- two EphemeralClient()
        # calls with the same collection name silently share state. A unique
        # per-instance name keeps separate AppState instances (e.g. in tests)
        # genuinely isolated; the real app only ever has one live instance
        # (see get_state), so this is invisible in production.
        self._collection_name = f"{COLLECTION_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
        self.collection: Collection = _new_collection(self._client, self._collection_name)

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self.collection = _new_collection(self._client, self._collection_name)
        self.fingerprint_store = InMemoryFingerprintStore()


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
