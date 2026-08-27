from rag_guard.core.embedder import embed_and_normalize
from rag_guard.core.fingerprint import generate_fingerprint
from state import AppState


def test_reset_clears_the_collection_and_fingerprint_store() -> None:
    state = AppState()
    state.collection.add(
        ids=["c1"],
        documents=["hello world"],
        embeddings=[embed_and_normalize(state.embedder, "hello world")],
    )
    state.fingerprint_store.put(
        generate_fingerprint(
            chunk_id="c1",
            document_id="doc-1",
            content="hello world",
            embedder=state.embedder,
            embedding_model="hashing-embedder",
        )
    )
    assert state.collection.count() == 1
    assert len(state.fingerprint_store) == 1

    state.reset()

    assert state.collection.count() == 0
    assert len(state.fingerprint_store) == 0
    assert state.fingerprint_store.get("c1") is None
