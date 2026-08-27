"""Vercel serverless function entrypoint.

Vercel's Python function auto-discovery only scans a literal top-level
`/api` directory -- it can't be pointed at an arbitrary path via
`vercel.json`. The real FastAPI app lives at `apps/api/index.py` (so local
dev via `uvicorn index:app` and the test suite are unaffected); this file
just re-exports it for Vercel.
"""

import sys
from pathlib import Path

_APPS_API = Path(__file__).resolve().parent.parent / "apps" / "api"
if str(_APPS_API) not in sys.path:
    sys.path.insert(0, str(_APPS_API))

# mypy resolves this against apps/api/index.py fine at the type-checker
# level in isolation, but errors when checked alongside it in one run
# ("Duplicate module named index") since both files map to the same
# module name -- a static-analysis quirk, not a runtime issue (only one
# is ever actually on sys.path at a time; verified with a live TestClient
# call through this exact shim).
from index import app  # noqa: E402  # type: ignore[attr-defined]

__all__ = ["app"]
