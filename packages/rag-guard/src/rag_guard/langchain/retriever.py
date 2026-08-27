from __future__ import annotations

import logging

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_guard.core.embedder import Embedder
from rag_guard.core.fingerprint import DEFAULT_SIMILARITY_THRESHOLD, verify_chunk
from rag_guard.core.schema import StrictnessMode, ValidationResult, ValidationStatus
from rag_guard.core.store import FingerprintStore

logger = logging.getLogger("rag_guard.langchain.retriever")

CHUNK_ID_METADATA_KEY = "chunk_id"


class IntegrityViolationError(RuntimeError):
    """Raised in STRICT mode when one or more retrieved chunks fail fingerprint verification."""

    def __init__(self, violations: list[ValidationResult]) -> None:
        self.violations = violations
        summary = "; ".join(
            f"{v.chunk_id} ({v.status.value}, similarity={v.similarity_score})"
            for v in violations
        )
        super().__init__(f"Integrity violation on {len(violations)} chunk(s): {summary}")


class IntegrityGuardRetriever(BaseRetriever):
    """Wraps a LangChain retriever and re-verifies every retrieved chunk
    against the fingerprint recorded for it at ingestion, before the chunk
    can reach a prompt template.

    `strictness` controls what happens when a chunk fails verification:
      - `STRICT`: raise `IntegrityViolationError` for the whole call.
      - `FILTER`: silently drop failing chunks, return only `VALID` ones.
      - `LOG_ONLY`: return every chunk (annotated + logged), block nothing.

    Every returned `Document` is annotated with `metadata["integrity_status"]`
    and `metadata["integrity_similarity"]` regardless of mode.
    """

    wrapped_retriever: BaseRetriever
    fingerprint_store: FingerprintStore
    embedder: Embedder
    strictness: StrictnessMode = StrictnessMode.STRICT
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    chunk_id_key: str = CHUNK_ID_METADATA_KEY

    def _verify(
        self, documents: list[Document]
    ) -> tuple[list[Document], list[ValidationResult]]:
        verified: list[Document] = []
        violations: list[ValidationResult] = []

        for document in documents:
            chunk_id = document.metadata.get(self.chunk_id_key)
            fingerprint = self.fingerprint_store.get(chunk_id) if chunk_id else None

            if fingerprint is None:
                result = ValidationResult(
                    chunk_id=chunk_id or "<unknown>",
                    status=ValidationStatus.MISSING_FINGERPRINT,
                    detail=(
                        f"Document metadata has no '{self.chunk_id_key}' key."
                        if not chunk_id
                        else "No fingerprint on record for this chunk_id."
                    ),
                )
            else:
                result = verify_chunk(
                    content=document.page_content,
                    fingerprint=fingerprint,
                    embedder=self.embedder,
                    similarity_threshold=self.similarity_threshold,
                )

            document.metadata["integrity_status"] = result.status.value
            document.metadata["integrity_similarity"] = result.similarity_score

            if result.is_valid:
                verified.append(document)
            else:
                violations.append(result)

        return verified, violations

    def _resolve(
        self,
        documents: list[Document],
        verified: list[Document],
        violations: list[ValidationResult],
    ) -> list[Document]:
        for violation in violations:
            logger.warning(
                "rag-integrity-guard: chunk_id=%s status=%s detail=%s",
                violation.chunk_id,
                violation.status.value,
                violation.detail,
            )

        if violations and self.strictness is StrictnessMode.STRICT:
            raise IntegrityViolationError(violations)

        if self.strictness is StrictnessMode.LOG_ONLY:
            return documents

        return verified

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = self.wrapped_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        verified, violations = self._verify(documents)
        return self._resolve(documents, verified, violations)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = await self.wrapped_retriever.ainvoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        verified, violations = self._verify(documents)
        return self._resolve(documents, verified, violations)
