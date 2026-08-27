import pytest

from rag_guard.eval.attacks import AttackType
from rag_guard.eval.benchmark import run_benchmark
from rag_guard.eval.dataset import FeverExample

EXAMPLES = [
    FeverExample(
        claim_id=f"c{i}",
        claim=f"Claim number {i} about some fact.",
        evidence=evidence,
        fever_label="SUPPORTS",
    )
    for i, evidence in enumerate(
        [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas.",
            "The Great Wall of China is a series of fortifications built across northern China.",
            "The Amazon rainforest is a moist broadleaf tropical rainforest in South America.",
            "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy.",
            "The Sahara is a desert on the African continent, the largest hot desert in the world.",
            "The Great Barrier Reef is the world's largest coral reef system off Australia.",
            "Niagara Falls is a group of three waterfalls at the border of Canada and the US.",
        ]
    )
]


def test_run_benchmark_reports_full_sample_size() -> None:
    report = run_benchmark(EXAMPLES, seed=7)
    assert report.sample_size == len(EXAMPLES)


def test_run_benchmark_never_false_positives_on_unmodified_content() -> None:
    # A byte-identical chunk always hash-matches -> VALID, by construction of
    # verify_chunk. This should hold regardless of seed.
    report = run_benchmark(EXAMPLES, seed=7)
    assert report.overall_false_positive_rate == 0.0


def test_run_benchmark_detects_every_tampered_chunk() -> None:
    # Any byte-level change fails the hash check -> non-VALID, by
    # construction of verify_chunk -- detection is deterministic here, not
    # probabilistic, so TPR should be exactly 1.0.
    report = run_benchmark(EXAMPLES, seed=7)
    assert report.overall_true_positive_rate == 1.0


def test_run_benchmark_covers_every_attack_type() -> None:
    report = run_benchmark(EXAMPLES, seed=7)
    reported_types = {m.attack_type for m in report.per_attack_type}
    assert reported_types == {t.value for t in AttackType}


def test_run_benchmark_records_verification_overhead() -> None:
    report = run_benchmark(EXAMPLES, seed=7)
    assert report.verification_overhead_ms_mean >= 0.0
    assert report.verification_overhead_ms_p95 >= 0.0


def test_run_benchmark_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        run_benchmark([])
