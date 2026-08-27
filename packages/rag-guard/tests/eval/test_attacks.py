import random

import pytest

from rag_guard.eval.attacks import (
    AttackType,
    apply_attack,
    exact_mutation,
    payload_injection,
    semantic_drift,
)

LONG_TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
    "in Paris, France, named after the engineer Gustave Eiffel."
)


def test_exact_mutation_changes_text_but_keeps_length() -> None:
    mutated = exact_mutation(LONG_TEXT, rng=random.Random(1))
    assert mutated != LONG_TEXT
    assert len(mutated) == len(LONG_TEXT)


def test_exact_mutation_is_a_no_op_on_text_without_letters() -> None:
    digits_only = "1234 5678 !!"
    assert exact_mutation(digits_only, rng=random.Random(1)) == digits_only


def test_payload_injection_appends_to_original_text() -> None:
    injected = payload_injection(LONG_TEXT, rng=random.Random(2))
    assert injected.startswith(LONG_TEXT)
    assert len(injected) > len(LONG_TEXT)


def test_semantic_drift_picks_a_different_pool_member() -> None:
    pool = [LONG_TEXT, "Berlin is the capital of Germany.", "Water boils at 100 degrees Celsius."]
    drifted = semantic_drift(LONG_TEXT, pool, rng=random.Random(3))
    assert drifted != LONG_TEXT
    assert drifted in pool


def test_semantic_drift_falls_back_to_original_without_alternatives() -> None:
    assert semantic_drift(LONG_TEXT, [LONG_TEXT], rng=random.Random(3)) == LONG_TEXT
    assert semantic_drift(LONG_TEXT, [], rng=random.Random(3)) == LONG_TEXT


def test_apply_attack_none_is_identity() -> None:
    assert apply_attack(AttackType.NONE, LONG_TEXT, rng=random.Random(4)) == LONG_TEXT


def test_apply_attack_rejects_unknown_attack_type() -> None:
    with pytest.raises(ValueError, match="Unknown attack type"):
        apply_attack("not_a_real_attack", LONG_TEXT, rng=random.Random(6))  # type: ignore[arg-type]


def test_apply_attack_dispatches_to_each_attack_type() -> None:
    pool = [LONG_TEXT, "Some other unrelated passage of comparable length here."]
    rng = random.Random(5)
    assert apply_attack(AttackType.EXACT_MUTATION, LONG_TEXT, rng=rng) != LONG_TEXT
    assert apply_attack(AttackType.PAYLOAD_INJECTION, LONG_TEXT, rng=rng).startswith(LONG_TEXT)
    drifted = apply_attack(AttackType.SEMANTIC_DRIFT, LONG_TEXT, rng=rng, replacement_pool=pool)
    assert drifted in pool
