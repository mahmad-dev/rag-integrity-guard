"""A thin LangChain BaseRetriever over a raw chromadb Collection.

Not part of `rag_guard` itself: the core package stays vector-store
agnostic, and this is Chroma-specific glue that belongs to the API app that
chose Chroma.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_guard.core.embedder import Embedder, embed_and_normalize
from rag_guard.langchain.retriever import CHUNK_ID_METADATA_KEY


class ChromaRetriever(BaseRetriever):
    """Embeds the query with the same `Embedder` used at ingestion (so the
    vector space is consistent end to end) and does a top-k similarity
    search against the collection."""

    collection: Collection
    embedder: Embedder
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = embed_and_normalize(self.embedder, query)
        # chromadb's stub is invariant on the outer list, which plain list[float]
        # can never satisfy structurally -- the runtime accepts this fine.
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=self.k,
        )

        ids = (results.get("ids") or [[]])[0]
        texts = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        documents: list[Document] = []
        for chunk_id, text, metadata in zip(ids, texts, metadatas, strict=True):
            merged_metadata = dict(metadata or {})
            merged_metadata.setdefault(CHUNK_ID_METADATA_KEY, chunk_id)
            documents.append(Document(page_content=text or "", metadata=merged_metadata))
        return documents
