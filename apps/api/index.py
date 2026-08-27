"""Vercel Python serverless entrypoint. Exposes a FastAPI ASGI `app`.

`packages/rag-guard/src` is bundled alongside this function (see vercel.json
`includeFiles`) rather than pip-installed, so it's added to `sys.path` here.
"""

import sys
from pathlib import Path

_RAG_GUARD_SRC = Path(__file__).resolve().parents[2] / "packages" / "rag-guard" / "src"
if _RAG_GUARD_SRC.exists() and str(_RAG_GUARD_SRC) not in sys.path:
    # Under pytest (see pyproject.toml [tool.pytest.ini_options]), this path
    # is already on sys.path before index.py is ever imported, so this
    # branch only fires in the actual Vercel deployment.
    sys.path.insert(0, str(_RAG_GUARD_SRC))  # pragma: no cover

import random  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from chroma_retriever import ChromaRetriever  # noqa: E402
from rag_guard import __version__ as rag_guard_version  # noqa: E402
from rag_guard.core.embedder import embed_and_normalize  # noqa: E402
from rag_guard.core.fingerprint import generate_fingerprint, verify_chunk  # noqa: E402
from rag_guard.core.schema import ValidationStatus  # noqa: E402
from rag_guard.eval.attacks import apply_attack  # noqa: E402
from rag_guard.eval.report import BenchmarkReport  # noqa: E402
from rag_guard.langchain.retriever import (  # noqa: E402
    CHUNK_ID_METADATA_KEY,
    IntegrityGuardRetriever,
    IntegrityViolationError,
)
from schemas import (  # noqa: E402
    AttackRequest,
    AttackResponse,
    IngestedChunk,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from state import get_state  # noqa: E402

_BENCHMARK_REPORT_PATH = Path(__file__).resolve().parent / "data" / "benchmark_report.json"

app = FastAPI(
    title="rag-integrity-guard API",
    description="Ingest, query, attack, and benchmark endpoints for the RAG integrity guard.",
    version=rag_guard_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "rag_guard_version": rag_guard_version}


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Fingerprint each chunk and populate the (in-memory) Chroma collection."""
    state = get_state()

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, object]] = []
    ingested: list[IngestedChunk] = []

    for i, chunk in enumerate(request.chunks):
        chunk_id = f"{request.document_id}:{i}"
        fingerprint = generate_fingerprint(
            chunk_id=chunk_id,
            document_id=request.document_id,
            content=chunk.text,
            embedder=state.embedder,
            embedding_model=state.embedder.__class__.__name__,
        )
        state.fingerprint_store.put(fingerprint)

        ids.append(chunk_id)
        documents.append(chunk.text)
        embeddings.append(embed_and_normalize(state.embedder, chunk.text))
        metadatas.append({**chunk.metadata, CHUNK_ID_METADATA_KEY: chunk_id})
        ingested.append(IngestedChunk(chunk_id=chunk_id, content_hash=fingerprint.content_hash))

    if ids:
        # chromadb's stub is invariant on these list types, which our plain
        # list[list[float]] / list[dict[str, object]] can't satisfy
        # structurally -- the runtime accepts this fine.
        state.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    return IngestResponse(document_id=request.document_id, ingested=ingested)


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a guarded RAG query: retrieve from Chroma, then re-verify every
    chunk against its stored fingerprint before it's returned."""
    state = get_state()

    guard = IntegrityGuardRetriever(
        wrapped_retriever=ChromaRetriever(
            collection=state.collection, embedder=state.embedder, k=request.k
        ),
        fingerprint_store=state.fingerprint_store,
        embedder=state.embedder,
        strictness=request.strictness,
        similarity_threshold=request.similarity_threshold,
    )

    try:
        documents = guard.invoke(request.query)
    except IntegrityViolationError as exc:
        return QueryResponse(
            query=request.query,
            strictness=request.strictness,
            blocked=True,
            violations=[f"{v.chunk_id}: {v.status.value}" for v in exc.violations],
            results=[],
        )

    results = [
        RetrievedChunk(
            chunk_id=str(document.metadata.get(CHUNK_ID_METADATA_KEY, "<unknown>")),
            text=document.page_content,
            integrity_status=str(
                document.metadata.get("integrity_status", ValidationStatus.VALID.value)
            ),
            integrity_similarity=document.metadata.get("integrity_similarity"),
        )
        for document in documents
    ]
    return QueryResponse(
        query=request.query, strictness=request.strictness, blocked=False, results=results
    )


@app.post("/api/attack", response_model=AttackResponse)
def attack(request: AttackRequest) -> AttackResponse:
    """Tamper a chunk that's already live in the vector store, simulating a
    poisoning attack that happened after ingestion -- the fingerprint on
    record is left untouched, exactly like a real attacker would leave it."""
    state = get_state()

    existing = state.collection.get(ids=[request.chunk_id], include=["documents"])
    existing_texts = existing["documents"] or []
    if not existing["ids"] or not existing_texts:
        raise HTTPException(
            status_code=404, detail=f"No chunk with id '{request.chunk_id}' in the collection."
        )
    original_text = existing_texts[0]

    replacement_pool: list[str] = []
    if request.attack_type.value == "semantic_drift":
        everything = state.collection.get(include=["documents"])
        everything_ids = everything["ids"] or []
        everything_texts = everything["documents"] or []
        replacement_pool = [
            text
            for cid, text in zip(everything_ids, everything_texts, strict=True)
            if cid != request.chunk_id
        ]

    tampered_text = apply_attack(
        request.attack_type,
        original_text,
        rng=random.Random(),
        replacement_pool=replacement_pool,
    )

    new_embedding = embed_and_normalize(state.embedder, tampered_text)
    state.collection.update(
        ids=[request.chunk_id],
        documents=[tampered_text],
        embeddings=[new_embedding],  # type: ignore[arg-type]
    )

    fingerprint = state.fingerprint_store.get(request.chunk_id)
    if fingerprint is not None:
        result = verify_chunk(
            content=tampered_text, fingerprint=fingerprint, embedder=state.embedder
        )
        resulting_status, resulting_similarity = result.status.value, result.similarity_score
    else:
        resulting_status, resulting_similarity = ValidationStatus.MISSING_FINGERPRINT.value, None

    return AttackResponse(
        chunk_id=request.chunk_id,
        attack_type=request.attack_type,
        original_excerpt=original_text[:200],
        tampered_excerpt=tampered_text[:200],
        resulting_status=resulting_status,
        resulting_similarity=resulting_similarity,
    )


@app.get("/api/benchmark", response_model=BenchmarkReport)
def benchmark() -> BenchmarkReport:
    """Return the precomputed FEVER benchmark report bundled with the API.

    Recomputing live against Hugging Face on every request would pull in the
    `datasets` dependency and add multi-second, network-dependent latency to
    a serverless request -- the opposite of "zero-cost". Regenerate the
    bundled report with:
    `python -m rag_guard.eval.run --output apps/api/data/benchmark_report.json`
    """
    if not _BENCHMARK_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No precomputed benchmark report bundled. Generate one with "
                "`python -m rag_guard.eval.run "
                "--output apps/api/data/benchmark_report.json`."
            ),
        )
    return BenchmarkReport.model_validate_json(_BENCHMARK_REPORT_PATH.read_text())
