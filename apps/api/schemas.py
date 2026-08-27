from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from rag_guard.core.schema import StrictnessMode
from rag_guard.eval.attacks import AttackType


class IngestChunk(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunks: list[IngestChunk]


class IngestedChunk(BaseModel):
    chunk_id: str
    content_hash: str


class IngestResponse(BaseModel):
    document_id: str
    ingested: list[IngestedChunk]


class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=4, ge=1, le=20)
    strictness: StrictnessMode = StrictnessMode.FILTER
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    integrity_status: str
    integrity_similarity: float | None


class QueryResponse(BaseModel):
    query: str
    strictness: StrictnessMode
    blocked: bool
    violations: list[str] = Field(default_factory=list)
    results: list[RetrievedChunk]


class AttackRequest(BaseModel):
    chunk_id: str
    attack_type: AttackType


class AttackResponse(BaseModel):
    chunk_id: str
    attack_type: AttackType
    original_excerpt: str
    tampered_excerpt: str
    resulting_status: str
    resulting_similarity: float | None
