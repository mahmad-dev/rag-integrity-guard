from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from rag_guard.core.hasher import HashAlgorithm


class StrictnessMode(StrEnum):
    """How `IntegrityGuardRetriever` (Step 3) reacts to a non-VALID chunk."""

    STRICT = "strict"
    FILTER = "filter"
    LOG_ONLY = "log_only"


class ValidationStatus(StrEnum):
    VALID = "valid"
    HASH_MISMATCH = "hash_mismatch"
    SEMANTIC_DRIFT = "semantic_drift"
    MISSING_FINGERPRINT = "missing_fingerprint"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ChunkFingerprint(BaseModel):
    """Dual signature recorded for a chunk at ingestion time."""

    chunk_id: str
    document_id: str
    content_hash: str
    hash_algorithm: HashAlgorithm = HashAlgorithm.BLAKE3
    embedding_signature: list[float]
    embedding_model: str
    source: str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ValidationResult(BaseModel):
    """Outcome of re-verifying a retrieved chunk against its fingerprint."""

    chunk_id: str
    status: ValidationStatus
    similarity_score: float | None = None
    detail: str = ""
    checked_at: datetime = Field(default_factory=_utcnow)

    @property
    def is_valid(self) -> bool:
        return self.status is ValidationStatus.VALID
