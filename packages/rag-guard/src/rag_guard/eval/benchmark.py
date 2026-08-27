"""Core benchmark logic, kept separate from `run.py`'s CLI wrapper.

(If `run_benchmark` lived in `run.py` and `eval/__init__.py` imported it
eagerly, `python -m rag_guard.eval.run` would import that same module twice
under two different identities -- once via the package import, once as
`__main__` -- which Python warns about. Keeping the CLI file import-only lets
`-m rag_guard.eval.run` work cleanly.)
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Iterable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_guard.core.embedder import HashingEmbedder
from rag_guard.core.fingerprint import DEFAULT_SIMILARITY_THRESHOLD, generate_fingerprint
from rag_guard.core.schema import StrictnessMode, ValidationStatus
from rag_guard.core.store import InMemoryFingerprintStore
from rag_guard.eval.attacks import AttackType, apply_attack
from rag_guard.eval.dataset import FEVER_DATASET_ID, FeverExample
from rag_guard.eval.report import AttackTypeMetrics, BenchmarkReport, ConfusionCounts
from rag_guard.langchain.retriever import CHUNK_ID_METADATA_KEY, IntegrityGuardRetriever

_ATTACK_CYCLE = (
    AttackType.NONE,
    AttackType.EXACT_MUTATION,
    AttackType.PAYLOAD_INJECTION,
    AttackType.SEMANTIC_DRIFT,
)


class _LiveCorpusRetriever(BaseRetriever):
    """Stand-in for a vector store retriever: returns whatever is currently
    on record for the requested chunk_id, which may have been tampered with
    after ingestion. Retrieval *accuracy* isn't under test here -- the guard
    is -- so this always "finds" the right chunk_id and lets the benchmark
    control exactly what content comes back for it.
    """

    live_corpus: dict[str, str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunk_id = query
        return [
            Document(
                page_content=self.live_corpus[chunk_id],
                metadata={CHUNK_ID_METADATA_KEY: chunk_id},
            )
        ]


def run_benchmark(
    examples: Iterable[FeverExample],
    *,
    split: str = "dev",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    seed: int = 42,
) -> BenchmarkReport:
    """Fingerprint each example's evidence as if freshly ingested, tamper a
    live copy of the corpus with a cycling attack assignment (including a
    NONE/benign bucket), then re-verify every chunk through
    `IntegrityGuardRetriever` and score the result.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("run_benchmark received no examples.")

    rng = random.Random(seed)
    embedder = HashingEmbedder()
    store = InMemoryFingerprintStore()

    original_corpus: dict[str, str] = {}
    for i, example in enumerate(examples):
        chunk_id = f"chunk-{i}"
        original_corpus[chunk_id] = example.evidence
        store.put(
            generate_fingerprint(
                chunk_id=chunk_id,
                document_id=example.claim_id,
                content=example.evidence,
                embedder=embedder,
                embedding_model="hashing-embedder",
            )
        )

    replacement_pool = list(original_corpus.values())
    assigned_attacks: dict[str, AttackType] = {}
    live_corpus: dict[str, str] = {}
    for i, chunk_id in enumerate(original_corpus):
        attack_type = _ATTACK_CYCLE[i % len(_ATTACK_CYCLE)]
        assigned_attacks[chunk_id] = attack_type
        live_corpus[chunk_id] = apply_attack(
            attack_type,
            original_corpus[chunk_id],
            rng=rng,
            replacement_pool=replacement_pool,
        )

    guard = IntegrityGuardRetriever(
        wrapped_retriever=_LiveCorpusRetriever(live_corpus=live_corpus),
        fingerprint_store=store,
        embedder=embedder,
        strictness=StrictnessMode.LOG_ONLY,
        similarity_threshold=similarity_threshold,
    )

    counts: dict[AttackType, ConfusionCounts] = {t: ConfusionCounts() for t in AttackType}
    latencies_ms: list[float] = []

    for chunk_id, attack_type in assigned_attacks.items():
        start = time.perf_counter()
        result = guard.invoke(chunk_id)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        flagged = result[0].metadata["integrity_status"] != ValidationStatus.VALID.value
        is_attacked = attack_type is not AttackType.NONE
        bucket = counts[attack_type]

        if is_attacked and flagged:
            bucket.true_positive += 1
        elif is_attacked and not flagged:
            # Structurally unreachable given verify_chunk's guarantee (any
            # byte change fails the hash check) -- kept for a complete
            # confusion matrix, not because this fires in practice.
            bucket.false_negative += 1  # pragma: no cover
        elif not is_attacked and flagged:
            # Same guarantee, opposite direction: unmodified content always
            # hash-matches, so this never fires either.
            bucket.false_positive += 1  # pragma: no cover
        else:
            bucket.true_negative += 1

    attacked_tp = sum(counts[t].true_positive for t in AttackType if t is not AttackType.NONE)
    attacked_fn = sum(counts[t].false_negative for t in AttackType if t is not AttackType.NONE)
    overall_tpr = attacked_tp / (attacked_tp + attacked_fn) if (attacked_tp + attacked_fn) else 0.0

    benign = counts[AttackType.NONE]
    overall_fpr = (
        benign.false_positive / (benign.false_positive + benign.true_negative)
        if (benign.false_positive + benign.true_negative)
        else 0.0
    )

    per_attack_type: list[AttackTypeMetrics] = []
    for attack_type in _ATTACK_CYCLE:
        bucket = counts[attack_type]
        if attack_type is AttackType.NONE:
            sample_count = bucket.false_positive + bucket.true_negative
            tpr = None
        else:
            sample_count = bucket.true_positive + bucket.false_negative
            tpr = bucket.true_positive / sample_count if sample_count else None
        per_attack_type.append(
            AttackTypeMetrics(
                attack_type=attack_type.value,
                sample_count=sample_count,
                true_positive_rate=tpr,
                counts=bucket,
            )
        )

    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))

    return BenchmarkReport(
        dataset=FEVER_DATASET_ID,
        split=split,
        sample_size=len(examples),
        similarity_threshold=similarity_threshold,
        overall_true_positive_rate=overall_tpr,
        overall_false_positive_rate=overall_fpr,
        per_attack_type=per_attack_type,
        verification_overhead_ms_mean=statistics.mean(latencies_ms) if latencies_ms else 0.0,
        verification_overhead_ms_p95=sorted_latencies[p95_index] if sorted_latencies else 0.0,
    )
