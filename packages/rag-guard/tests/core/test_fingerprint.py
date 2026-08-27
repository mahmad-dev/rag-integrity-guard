from rag_guard.core.embedder import HashingEmbedder
from rag_guard.core.fingerprint import generate_fingerprint, verify_chunk
from rag_guard.core.schema import ValidationStatus

ORIGINAL = "The Eiffel Tower is located in Paris, France."


def _fingerprint(embedder: HashingEmbedder):
    return generate_fingerprint(
        chunk_id="chunk-1",
        document_id="doc-1",
        content=ORIGINAL,
        embedder=embedder,
        embedding_model="hashing-embedder",
    )


def test_verify_chunk_unmodified_content_is_valid() -> None:
    embedder = HashingEmbedder()
    fingerprint = _fingerprint(embedder)

    result = verify_chunk(content=ORIGINAL, fingerprint=fingerprint, embedder=embedder)

    assert result.status is ValidationStatus.VALID
    assert result.similarity_score == 1.0


def test_verify_chunk_flags_wholesale_replacement_as_hash_mismatch() -> None:
    embedder = HashingEmbedder()
    fingerprint = _fingerprint(embedder)
    unrelated = "Quantum entanglement enables correlated particle measurements at a distance."

    result = verify_chunk(content=unrelated, fingerprint=fingerprint, embedder=embedder)

    assert result.status is ValidationStatus.HASH_MISMATCH
    assert not result.is_valid
    assert result.similarity_score is not None and result.similarity_score < 0.90


def test_verify_chunk_flags_near_duplicate_tampering_as_semantic_drift() -> None:
    embedder = HashingEmbedder()
    fingerprint = _fingerprint(embedder)
    # A single-character swap: high n-gram overlap with the original, but the
    # hash still changes -- this is the "disguised" forgery case.
    near_duplicate = "The Eiffel Tower is located in Paris, Francs."

    result = verify_chunk(
        content=near_duplicate,
        fingerprint=fingerprint,
        embedder=embedder,
        similarity_threshold=0.5,
    )

    assert result.status is ValidationStatus.SEMANTIC_DRIFT
    assert not result.is_valid


def test_verify_chunk_never_marks_a_hash_mismatch_as_valid() -> None:
    embedder = HashingEmbedder()
    fingerprint = _fingerprint(embedder)
    near_duplicate = "The Eiffel Tower is located in Paris, Francs."

    result = verify_chunk(
        content=near_duplicate,
        fingerprint=fingerprint,
        embedder=embedder,
        similarity_threshold=0.0,
    )

    assert result.status is not ValidationStatus.VALID
