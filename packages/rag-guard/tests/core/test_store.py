from rag_guard.core.schema import ChunkFingerprint
from rag_guard.core.store import InMemoryFingerprintStore


def _fingerprint(chunk_id: str) -> ChunkFingerprint:
    return ChunkFingerprint(
        chunk_id=chunk_id,
        document_id="doc-1",
        content_hash="deadbeef",
        embedding_signature=[0.1, 0.2, 0.3],
        embedding_model="hashing-embedder",
    )


def test_store_starts_empty() -> None:
    store = InMemoryFingerprintStore()
    assert len(store) == 0
    assert "c1" not in store
    assert store.get("c1") is None


def test_put_and_get_roundtrip() -> None:
    store = InMemoryFingerprintStore()
    store.put(_fingerprint("c1"))
    assert len(store) == 1
    assert "c1" in store
    assert store.get("c1") is not None
    assert store.get("c1").chunk_id == "c1"


def test_bulk_put_adds_all_fingerprints() -> None:
    store = InMemoryFingerprintStore()
    store.bulk_put([_fingerprint("c1"), _fingerprint("c2")])
    assert len(store) == 2
    assert "c1" in store and "c2" in store


def test_constructor_accepts_initial_fingerprints() -> None:
    store = InMemoryFingerprintStore([_fingerprint("c1"), _fingerprint("c2")])
    assert len(store) == 2


def test_put_overwrites_existing_chunk_id() -> None:
    store = InMemoryFingerprintStore()
    store.put(_fingerprint("c1"))
    replacement = _fingerprint("c1")
    store.put(replacement)
    assert len(store) == 1
    assert store.get("c1") is replacement
