"""Vercel Python serverless entrypoint. Exposes a FastAPI ASGI `app`.

`packages/rag-guard/src` is bundled alongside this function (see vercel.json
`includeFiles`) rather than pip-installed, so it's added to `sys.path` here.
"""

import sys
from pathlib import Path

_RAG_GUARD_SRC = Path(__file__).resolve().parents[2] / "packages" / "rag-guard" / "src"
if _RAG_GUARD_SRC.exists() and str(_RAG_GUARD_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_GUARD_SRC))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from rag_guard import __version__ as rag_guard_version  # noqa: E402

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


# /api/ingest, /api/query, /api/attack, /api/benchmark land in Step 5.
