import pytest
from pydantic import ValidationError

from rag_guard.core.hasher import HashAlgorithm
from rag_guard.core.schema import (
    ChunkFingerprint,
    StrictnessMode,
    ValidationResult,
    ValidationStatus,
)


def test_chunk_fingerprint_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ChunkFingerprint()  # type: ignore[call-arg]


def test_chunk_fingerprint_defaults() -> None:
    fingerprint = ChunkFingerprint(
        chunk_id="c1",
        document_id="d1",
        content_hash="deadbeef",
        embedding_signature=[0.1, 0.2, 0.3],
        embedding_model="hashing-embedder",
    )
    assert fingerprint.hash_algorithm is HashAlgorithm.BLAKE3
    assert fingerprint.metadata == {}
    assert fingerprint.created_at.tzinfo is not None


def test_validation_result_is_valid_property() -> None:
    valid = ValidationResult(chunk_id="c1", status=ValidationStatus.VALID)
    tampered = ValidationResult(chunk_id="c1", status=ValidationStatus.HASH_MISMATCH)
    assert valid.is_valid
    assert not tampered.is_valid


def test_strictness_mode_values() -> None:
    assert {m.value for m in StrictnessMode} == {"strict", "filter", "log_only"}
