import pytest

from rag_guard.eval.attacks import AttackType
from rag_guard.eval.dataset import FeverExample
from rag_guard.eval.sensitivity import run_threshold_sensitivity

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
        ]
    )
]


def test_run_threshold_sensitivity_covers_every_attack_type_at_every_threshold() -> None:
    rows = run_threshold_sensitivity(EXAMPLES, [0.5, 0.9], seed=7)

    seen = {(row.similarity_threshold, row.attack_type) for row in rows}
    expected_types = {t.value for t in AttackType if t is not AttackType.NONE}
    assert seen == {(t, at) for t in (0.5, 0.9) for at in expected_types}


def test_lower_threshold_never_classifies_fewer_as_semantic_drift() -> None:
    # Lowering the bar for "similar enough" can only pull more borderline
    # cases into SEMANTIC_DRIFT, never fewer, for a fixed corpus.
    rows = run_threshold_sensitivity(EXAMPLES, [0.99, -2.0], seed=7)
    by_threshold = {(row.similarity_threshold, row.attack_type): row for row in rows}

    for attack_type in ("exact_mutation", "payload_injection", "semantic_drift"):
        high = by_threshold[(0.99, attack_type)]
        low = by_threshold[(-2.0, attack_type)]
        assert low.classified_semantic_drift >= high.classified_semantic_drift


def test_threshold_below_minus_one_classifies_everything_as_semantic_drift() -> None:
    # Cosine similarity is mathematically bounded to [-1, 1], so a threshold
    # below -1 is guaranteed to clear every real similarity value.
    rows = run_threshold_sensitivity(EXAMPLES, [-2.0], seed=7)
    for row in rows:
        assert row.classified_hash_mismatch == 0
        assert row.classified_semantic_drift == row.sample_count


def test_threshold_above_one_classifies_everything_as_hash_mismatch() -> None:
    rows = run_threshold_sensitivity(EXAMPLES, [1.01], seed=7)
    for row in rows:
        assert row.classified_semantic_drift == 0
        assert row.classified_hash_mismatch == row.sample_count


def test_sample_counts_split_evenly_across_three_attack_types() -> None:
    rows = run_threshold_sensitivity(EXAMPLES, [0.5], seed=7)
    counts = {row.attack_type: row.sample_count for row in rows}
    assert sum(counts.values()) == len(EXAMPLES)
    assert len(set(counts.values())) <= 2  # even split, off by at most one


def test_raises_on_empty_examples() -> None:
    with pytest.raises(ValueError):
        run_threshold_sensitivity([], [0.5])


def test_raises_on_empty_thresholds() -> None:
    with pytest.raises(ValueError):
        run_threshold_sensitivity(EXAMPLES, [])
