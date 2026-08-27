# rag-integrity-guard

[![CI](https://github.com/mahmad-dev/rag-integrity-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/mahmad-dev/rag-integrity-guard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A from-scratch, MIT-licensed system that fingerprints documents at ingestion and
re-verifies retrieved context against that fingerprint before it reaches an LLM —
catching retrieval poisoning and tampering. Benchmarked against a public dataset,
with real numbers.

**Live demo:** not yet deployed. See [Deployment](#deployment) to put your own up on
Vercel's free tier.

## Why

Standard RAG trusts whatever the vector store hands back at query time. If that
store is tampered with — a compromised ingestion pipeline, a poisoned document, a
direct database edit — the LLM has no way to tell the difference between real
context and an attacker's substitute. `rag-integrity-guard` sits directly between
retrieval and prompt construction and closes that gap: every chunk is fingerprinted
when it's ingested, and re-verified against that fingerprint every time it's
retrieved.

## How it works

Each chunk gets a **dual signature** at ingestion ([`core/fingerprint.py`](packages/rag-guard/src/rag_guard/core/fingerprint.py)):

1. **Cryptographic hash** (BLAKE3 or SHA-256) of the normalized text — exact-match
   tamper detection.
2. **Semantic embedding signature** — an L2-normalized vector, cosine-comparable at
   verification time.

At query time, [`IntegrityGuardRetriever`](packages/rag-guard/src/rag_guard/langchain/retriever.py)
wraps any LangChain `BaseRetriever` and re-verifies every returned chunk:

- **Hash matches exactly** → `VALID`. Nothing else matters.
- **Hash doesn't match, but the embedding is still highly similar** → `SEMANTIC_DRIFT`.
  This is the interesting case: content changed just enough to stay plausible —
  exactly what a disguised poisoning attack looks like.
- **Hash doesn't match and the embedding has diverged too** → `HASH_MISMATCH`, a
  blunter rewrite.
- **No fingerprint on record for this chunk** → `MISSING_FINGERPRINT`.

Only `VALID` chunks are ever treated as trustworthy — a mismatch is never waved
through just because it "looks similar." A configurable `StrictnessMode` controls
what happens to a non-`VALID` chunk:

| Mode | Behavior |
|---|---|
| `STRICT` | Raises `IntegrityViolationError`; the whole query is blocked. |
| `FILTER` | Silently drops the offending chunk; the rest of the query proceeds. |
| `LOG_ONLY` | Returns everything, annotated with its status, for auditing. |

## Monorepo layout

- [`packages/rag-guard`](packages/rag-guard) — the core Python package:
  - `core/` — `hasher.py` (BLAKE3/SHA-256), `embedder.py` (an `Embedder` protocol
    matching LangChain's `Embeddings`, plus a dependency-free `HashingEmbedder`
    default), `schema.py` (Pydantic models), `fingerprint.py`, `store.py`.
  - `langchain/` — `IntegrityGuardRetriever`.
  - `eval/` — the FEVER benchmark harness and the three attack simulators.
- [`apps/api`](apps/api) — FastAPI backend (`/api/ingest`, `/api/query`,
  `/api/attack`, `/api/benchmark`), deployed as a Vercel Python serverless
  function, backed by an in-memory Chroma collection.
- [`apps/web`](apps/web) — Next.js 14 dashboard: live playground, attack
  simulator, and benchmark visualizer.

## Quick start

Requires Node 20+, pnpm 9+, and Python 3.11+.

```bash
git clone https://github.com/mahmad-dev/rag-integrity-guard.git
cd rag-integrity-guard

# Python
pip install -e "packages/rag-guard[eval,dev]"
pip install -r apps/api/requirements-dev.txt
pytest                       # 72 tests: core + langchain + eval + api

# Node
pnpm install
```

Run the API and the dashboard in two terminals:

```bash
# Terminal 1 -- API on :8000
cd apps/api && uvicorn index:app --reload --port 8000

# Terminal 2 -- dashboard on :3000, pointed at the local API
cd apps/web
cp .env.local.example .env.local   # uncomment NEXT_PUBLIC_API_BASE_URL
pnpm dev
```

Open `http://localhost:3000`: ingest a few sample facts, run a query, then launch
an attack on one of the ingested chunks and watch the guard react.

<img src="docs/screenshots/attack-simulator.png" alt="Attack simulator: before/after tampering, with the resulting integrity status" width="720">

Run the benchmark CLI directly:

```bash
python -m rag_guard.eval.run --sample-size 500 --output apps/api/data/benchmark_report.json
```

## Benchmark

Run against 500 real (claim, gold-evidence) pairs from
[FEVER](https://fever.ai/) (via the [`pietrolesci/nli_fever`](https://huggingface.co/datasets/pietrolesci/nli_fever)
reformatting, dev split), split into a benign baseline and the three attack types
below:

| Attack type | Description | Detection rate (TPR) |
|---|---|---|
| Benign (no attack) | Untouched chunk | — (0.0% false-positive rate) |
| Exact mutation | A handful of characters flipped — a direct database/vector-store edit | 100% |
| Payload injection | An indirect prompt-injection payload appended to the chunk | 100% |
| Semantic drift | Wholesale replacement with a different, topically-plausible passage | 100% |

**Overall: 100% true-positive rate, 0% false-positive rate.** Verification
overhead per call (using the default `HashingEmbedder`, no external API):
**2.73ms mean / 7.07ms p95**.

<img src="docs/screenshots/benchmark.png" alt="Benchmark visualizer: TPR/FPR stat tiles and a per-attack-type detection-rate chart" width="720">

**On those numbers:** TPR = 1.0 and FPR = 0.0 aren't estimates — they're a
*designed invariant* of hash-based verification, not a probabilistic classifier's
accuracy. Any byte-level change fails the hash check by construction; identical
content always passes. The actual signal in this benchmark is the
`HASH_MISMATCH` vs `SEMANTIC_DRIFT` classification split (which attacks look
"blunt" vs. "disguised") and the latency numbers, not a detection rate that could
plausibly be less than perfect. The full report — including the per-attack-type
breakdown — is bundled at [`apps/api/data/benchmark_report.json`](apps/api/data/benchmark_report.json)
and served live from `/api/benchmark`.

## API reference

All endpoints are under `/api`. See [`apps/api/schemas.py`](apps/api/schemas.py)
for full request/response shapes.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness check. |
| `/api/ingest` | POST | Fingerprint a list of text chunks, populate the vector store. |
| `/api/query` | POST | Guarded retrieval: `{ query, k, strictness, similarity_threshold }`. |
| `/api/attack` | POST | Tamper a chunk already in the store: `{ chunk_id, attack_type }`. |
| `/api/benchmark` | GET | The precomputed FEVER benchmark report. |

State (the vector store and fingerprints) lives in the memory of a single warm
API instance — it's the explicit zero-cost tradeoff of serverless hobby-tier
deployment. A cold start gets a clean slate; there's no per-browser-session
isolation. Swap in a persistent Chroma client and a database-backed
`FingerprintStore` (both are defined behind small interfaces — see
[`core/store.py`](packages/rag-guard/src/rag_guard/core/store.py)) for anything
beyond a demo.

## Testing & CI

```bash
pytest                                    # everything, including one live
                                           # network test against Hugging Face
pytest -m "not integration"               # what CI runs: fast, deterministic
pytest --cov=rag_guard --cov=index --cov=state --cov=chroma_retriever --cov=schemas \
       --cov-report=term-missing          # coverage report
ruff check packages/rag-guard apps/api    # lint
mypy packages/rag-guard/src apps/api/*.py # types (strict)
```

72 tests, 97% coverage excluding the one network-gated code path (the actual
Hugging Face download, covered separately by the `integration`-marked test).
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs ruff, mypy, and pytest
against Python 3.11 and 3.12, plus `pnpm lint`/`pnpm build` for the dashboard, on
every push and pull request.

## Deployment

The root [`vercel.json`](vercel.json) is configured for a single Vercel project
covering both apps: `apps/web` builds as the Next.js site, `apps/api/index.py`
deploys as a Python serverless function, and `/api/*` is rewritten straight to it
— no separate backend deployment needed.

```bash
npm i -g vercel
vercel link      # connect this repo to a Vercel project
vercel --prod
```

Note the [state caveat](#api-reference) above: this deploys a working, zero-cost
demo, not a production-durable store.

## License

[MIT](LICENSE)
