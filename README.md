# rag-integrity-guard
A from-scratch, MIT-licensed system that fingerprints documents at ingestion and re-verifies retrieved context against that fingerprint before it reaches an LLM — catching retrieval poisoning and tampering. Benchmarked against a public dataset, with real numbers.

## Monorepo layout

- [`packages/rag-guard`](packages/rag-guard) — core Python package: dual fingerprinting (BLAKE3/SHA-256 + semantic embedding signature), the `IntegrityGuardRetriever` LangChain interceptor, and the FEVER-backed benchmark/attack-simulation harness.
- [`apps/api`](apps/api) — FastAPI backend (`/api/ingest`, `/api/query`, `/api/attack`, `/api/benchmark`) deployed as a Vercel Python serverless function.
- [`apps/web`](apps/web) — Next.js 14 dashboard: live playground, attack simulator, and benchmark visualizer.

Status: scaffold in place; fingerprinting engine, retriever, eval suite, API, and dashboard land in subsequent commits.

### Local development

```bash
pnpm install               # apps/web
pip install -e packages/rag-guard
pip install -r apps/api/requirements-dev.txt
pytest                     # packages/rag-guard + apps/api tests
pnpm dev                   # Next.js dashboard on :3000
```
