"""Similarity-threshold sensitivity analysis.

`run_benchmark` (see `benchmark.py`) already proves TPR=1.0 / FPR=0.0 are
structural guarantees of hash-based verification, not something a threshold
sweep can move -- any byte change fails the hash check regardless of
`similarity_threshold`. The threshold only affects *classification*: whether
a tampered chunk gets labeled the blunter `HASH_MISMATCH` or the more
concerning `SEMANTIC_DRIFT` (a disguised, still-plausible forgery). This
module isolates and measures exactly that effect, holding a single tampered
corpus fixed and only varying the threshold -- so the sweep reflects the
threshold's effect alone, not fresh randomness in the attacks each time.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterable

from pydantic import BaseModel

from rag_guard.core.embedder import (
    Embedder,
    HashingEmbedder,
    cosine_similarity,
    embed_and_normalize,
)
from rag_guard.eval.attacks import AttackType, apply_attack
from rag_guard.eval.dataset import FeverExample

_ATTACK_TYPES = (AttackType.EXACT_MUTATION, AttackType.PAYLOAD_INJECTION, AttackType.SEMANTIC_DRIFT)


class ThresholdRow(BaseModel):
    similarity_threshold: float
    attack_type: str
    sample_count: int
    classified_semantic_drift: int
    classified_hash_mismatch: int
    mean_similarity: float
    median_similarity: float


def _build_tampered_corpus(
    examples: list[FeverExample], seed: int
) -> tuple[dict[str, AttackType], dict[str, float]]:
    """Attack each example's evidence once (fixed by `seed`) and return, per
    chunk_id: which attack it got, and the resulting cosine similarity to
    its original embedding. Every attacked chunk fails the hash check by
    construction, so similarity is the only threshold-sensitive quantity.
    """
    rng = random.Random(seed)
    embedder: Embedder = HashingEmbedder()

    originals = {f"chunk-{i}": example.evidence for i, example in enumerate(examples)}
    pool = list(originals.values())

    assigned: dict[str, AttackType] = {}
    similarities: dict[str, float] = {}
    for i, (chunk_id, original_text) in enumerate(originals.items()):
        attack_type = _ATTACK_TYPES[i % len(_ATTACK_TYPES)]
        assigned[chunk_id] = attack_type

        tampered_text = apply_attack(attack_type, original_text, rng=rng, replacement_pool=pool)
        original_signature = embed_and_normalize(embedder, original_text)
        tampered_signature = embed_and_normalize(embedder, tampered_text)
        similarities[chunk_id] = cosine_similarity(original_signature, tampered_signature)

    return assigned, similarities


def run_threshold_sensitivity(
    examples: Iterable[FeverExample],
    thresholds: list[float],
    *,
    seed: int = 42,
) -> list[ThresholdRow]:
    examples = list(examples)
    if not examples:
        raise ValueError("run_threshold_sensitivity received no examples.")
    if not thresholds:
        raise ValueError("run_threshold_sensitivity received no thresholds.")

    assigned, similarities = _build_tampered_corpus(examples, seed)

    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        for attack_type in _ATTACK_TYPES:
            sims = [
                similarities[chunk_id]
                for chunk_id, chunk_attack in assigned.items()
                if chunk_attack is attack_type
            ]
            semantic_drift = sum(1 for s in sims if s >= threshold)
            rows.append(
                ThresholdRow(
                    similarity_threshold=threshold,
                    attack_type=attack_type.value,
                    sample_count=len(sims),
                    classified_semantic_drift=semantic_drift,
                    classified_hash_mismatch=len(sims) - semantic_drift,
                    mean_similarity=statistics.mean(sims) if sims else 0.0,
                    median_similarity=statistics.median(sims) if sims else 0.0,
                )
            )
    return rows
