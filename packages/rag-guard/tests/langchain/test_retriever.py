import asyncio
import logging

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_guard.core.embedder import HashingEmbedder
from rag_guard.core.fingerprint import generate_fingerprint
from rag_guard.core.schema import StrictnessMode, ValidationStatus
from rag_guard.core.store import InMemoryFingerprintStore
from rag_guard.langchain.retriever import (
    CHUNK_ID_METADATA_KEY,
    IntegrityGuardRetriever,
    IntegrityViolationError,
)

ORIGINAL = "Paris is the capital of France."


class StaticRetriever(BaseRetriever):
    """Minimal LangChain retriever that just returns a fixed document list."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


def _build_guard(
    documents: list[Document], strictness: StrictnessMode = StrictnessMode.STRICT, **kwargs
) -> tuple[IntegrityGuardRetriever, InMemoryFingerprintStore, HashingEmbedder]:
    embedder = HashingEmbedder()
    store = InMemoryFingerprintStore()
    for document in documents:
        fingerprint = generate_fingerprint(
            chunk_id=document.metadata[CHUNK_ID_METADATA_KEY],
            document_id="doc-1",
            content=document.page_content,
            embedder=embedder,
            embedding_model="hashing-embedder",
        )
        store.put(fingerprint)

    guard = IntegrityGuardRetriever(
        wrapped_retriever=StaticRetriever(documents=documents),
        fingerprint_store=store,
        embedder=embedder,
        strictness=strictness,
        **kwargs,
    )
    return guard, store, embedder


def test_unmodified_chunks_pass_through_and_are_annotated() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"})]
    guard, _, _ = _build_guard(docs)

    result = guard.invoke("capital of France")

    assert len(result) == 1
    assert result[0].metadata["integrity_status"] == ValidationStatus.VALID.value
    assert result[0].metadata["integrity_similarity"] == 1.0


def test_strict_mode_raises_on_tampered_chunk() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"})]
    guard, _, _ = _build_guard(docs, strictness=StrictnessMode.STRICT)

    # Simulate the vector store being tampered with after ingestion.
    docs[0].page_content = "Berlin is the capital of France."

    with pytest.raises(IntegrityViolationError) as exc_info:
        guard.invoke("capital of France")

    assert exc_info.value.violations[0].chunk_id == "c1"
    assert not exc_info.value.violations[0].is_valid


def test_filter_mode_drops_tampered_chunks_without_raising() -> None:
    docs = [
        Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"}),
        Document(
            page_content="Berlin is the capital of Germany.",
            metadata={CHUNK_ID_METADATA_KEY: "c2"},
        ),
    ]
    guard, _, _ = _build_guard(docs, strictness=StrictnessMode.FILTER)
    docs[0].page_content = "Berlin is the capital of France."  # tamper c1 only

    result = guard.invoke("capitals of Europe")

    assert [d.metadata[CHUNK_ID_METADATA_KEY] for d in result] == ["c2"]


def test_log_only_mode_returns_everything_but_logs_violations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"})]
    guard, _, _ = _build_guard(docs, strictness=StrictnessMode.LOG_ONLY)
    docs[0].page_content = "Berlin is the capital of France."

    with caplog.at_level(logging.WARNING, logger="rag_guard.langchain.retriever"):
        result = guard.invoke("capital of France")

    assert len(result) == 1
    assert result[0].metadata["integrity_status"] != ValidationStatus.VALID.value
    assert any("c1" in record.message for record in caplog.records)


def test_missing_fingerprint_is_treated_as_a_violation() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "unknown-chunk"})]
    embedder = HashingEmbedder()
    store = InMemoryFingerprintStore()  # deliberately empty
    guard = IntegrityGuardRetriever(
        wrapped_retriever=StaticRetriever(documents=docs),
        fingerprint_store=store,
        embedder=embedder,
        strictness=StrictnessMode.STRICT,
    )

    with pytest.raises(IntegrityViolationError) as exc_info:
        guard.invoke("capital of France")

    assert exc_info.value.violations[0].status is ValidationStatus.MISSING_FINGERPRINT


def test_document_missing_chunk_id_key_is_a_violation() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={})]
    embedder = HashingEmbedder()
    store = InMemoryFingerprintStore()
    guard = IntegrityGuardRetriever(
        wrapped_retriever=StaticRetriever(documents=docs),
        fingerprint_store=store,
        embedder=embedder,
        strictness=StrictnessMode.FILTER,
    )

    result = guard.invoke("capital of France")

    assert result == []


def test_async_invocation_matches_sync_behavior() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"})]
    guard, _, _ = _build_guard(docs, strictness=StrictnessMode.STRICT)

    result = asyncio.run(guard.ainvoke("capital of France"))

    assert len(result) == 1
    assert result[0].metadata["integrity_status"] == ValidationStatus.VALID.value


def test_custom_similarity_threshold_is_respected() -> None:
    docs = [Document(page_content=ORIGINAL, metadata={CHUNK_ID_METADATA_KEY: "c1"})]
    guard, _, _ = _build_guard(
        docs, strictness=StrictnessMode.LOG_ONLY, similarity_threshold=0.0
    )
    # Any hash mismatch scores >= 0.0 similarity, so with threshold 0.0 every
    # mismatch is classified as SEMANTIC_DRIFT rather than HASH_MISMATCH.
    docs[0].page_content = "Something entirely unrelated about quantum mechanics."

    result = guard.invoke("capital of France")

    assert result[0].metadata["integrity_status"] == ValidationStatus.SEMANTIC_DRIFT.value
