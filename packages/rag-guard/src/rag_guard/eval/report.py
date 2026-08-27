from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ConfusionCounts(BaseModel):
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0


class AttackTypeMetrics(BaseModel):
    attack_type: str
    sample_count: int
    true_positive_rate: float | None
    """Fraction of attacked chunks the guard flagged as non-VALID. None for
    the unattacked (NONE) bucket, which instead contributes to the overall
    false-positive rate."""
    counts: ConfusionCounts


class BenchmarkReport(BaseModel):
    dataset: str
    split: str
    sample_size: int
    similarity_threshold: float
    overall_true_positive_rate: float
    overall_false_positive_rate: float
    per_attack_type: list[AttackTypeMetrics]
    verification_overhead_ms_mean: float
    verification_overhead_ms_p95: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
