import pytest

from rag_guard.eval.dataset import load_fever_examples


@pytest.mark.integration
def test_load_fever_examples_from_real_dataset() -> None:
    examples = load_fever_examples(split="dev", limit=5, seed=42)

    assert 0 < len(examples) <= 5
    seen_evidence = set()
    for example in examples:
        assert example.fever_label in ("SUPPORTS", "REFUTES")
        assert example.claim
        assert example.evidence
        assert example.evidence not in seen_evidence
        seen_evidence.add(example.evidence)
