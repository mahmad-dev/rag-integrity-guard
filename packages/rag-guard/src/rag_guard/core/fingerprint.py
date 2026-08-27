from __future__ import annotations

from typing import Any

from rag_guard.core.embedder import Embedder, cosine_similarity, embed_and_normalize
from rag_guard.core.hasher import HashAlgorithm, compute_hash, verify_hash
from rag_guard.core.schema import ChunkFingerprint, ValidationResult, ValidationStatus

# Below this cosine similarity, a hash-mismatched chunk is treated as a blunt
# rewrite (HASH_MISMATCH) rather than a disguised, semantically-close forgery
# (SEMANTIC_DRIFT). Either status is a violation — see verify_chunk.
DEFAULT_SIMILARITY_THRESHOLD = 0.90


def generate_fingerprint(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    embedder: Embedder,
    embedding_model: str,
    hash_algorithm: HashAlgorithm = HashAlgorithm.BLAKE3,
    source: str | None = None,
    chunk_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChunkFingerprint:
    """Compute the dual signature (cryptographic hash + semantic vector) for
    a chunk at ingestion time."""
    return ChunkFingerprint(
        chunk_id=chunk_id,
        document_id=document_id,
        content_hash=compute_hash(content, hash_algorithm),
        hash_algorithm=hash_algorithm,
        embedding_signature=embed_and_normalize(embedder, content),
        embedding_model=embedding_model,
        source=source,
        chunk_index=chunk_index,
        metadata=metadata or {},
    )


def verify_chunk(
    *,
    content: str,
    fingerprint: ChunkFingerprint,
    embedder: Embedder,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> ValidationResult:
    """Re-verify a retrieved chunk's current content against its stored
    fingerprint. Only an exact hash match is VALID — a chunk that has drifted
    from its stored embedding signature just enough to stay plausible is
    exactly what a semantic-poisoning attack looks like, so similarity is
    used to classify the violation, never to wave one through.
    """
    if verify_hash(content, fingerprint.content_hash, fingerprint.hash_algorithm):
        return ValidationResult(
            chunk_id=fingerprint.chunk_id,
            status=ValidationStatus.VALID,
            similarity_score=1.0,
            detail="Content hash matches the stored fingerprint exactly.",
        )

    current_signature = embed_and_normalize(embedder, content)
    similarity = cosine_similarity(current_signature, fingerprint.embedding_signature)

    if similarity >= similarity_threshold:
        return ValidationResult(
            chunk_id=fingerprint.chunk_id,
            status=ValidationStatus.SEMANTIC_DRIFT,
            similarity_score=similarity,
            detail=(
                f"Content hash mismatch, but the embedding signature remains highly "
                f"similar (cosine={similarity:.4f} >= {similarity_threshold}) — "
                "consistent with a semantically-disguised forgery (paraphrase or "
                "poisoning) rather than an unrelated edit."
            ),
        )

    return ValidationResult(
        chunk_id=fingerprint.chunk_id,
        status=ValidationStatus.HASH_MISMATCH,
        similarity_score=similarity,
        detail=(
            f"Content hash mismatch and embedding signature diverges "
            f"(cosine={similarity:.4f} < {similarity_threshold})."
        ),
    )
