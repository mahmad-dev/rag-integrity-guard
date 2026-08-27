# rag-guard

Core Python package for `rag-integrity-guard`.

- `rag_guard.core` — dual fingerprinting (cryptographic hash + semantic embedding signature) and chunk schema.
- `rag_guard.langchain` — `IntegrityGuardRetriever`, a `BaseRetriever` subclass that re-verifies chunks at query time.
- `rag_guard.eval` — FEVER-backed benchmark harness and attack simulators (exact mutation, payload injection, semantic drift).

Implementation lands incrementally; see the repo root README for the build plan.
