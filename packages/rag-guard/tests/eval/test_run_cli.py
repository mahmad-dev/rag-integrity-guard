import json

import pytest

from rag_guard.eval import run as run_module
from rag_guard.eval.dataset import FeverExample

FAKE_EXAMPLES = [
    FeverExample(
        claim_id=f"c{i}",
        claim=f"Claim {i}",
        evidence=evidence,
        fever_label="SUPPORTS",
    )
    for i, evidence in enumerate(
        [
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "Mount Everest is Earth's highest mountain above sea level.",
            "The Great Wall of China stretches across northern China.",
            "The Amazon rainforest is a moist broadleaf forest in South America.",
        ]
    )
]


@pytest.fixture(autouse=True)
def _fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_module, "load_fever_examples", lambda split, limit, seed: FAKE_EXAMPLES[:limit]
    )


def test_main_returns_a_report_and_prints_a_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_module.main(["--sample-size", "4", "--seed", "7"])

    assert report.sample_size == 4
    captured = capsys.readouterr()
    assert "rag-integrity-guard benchmark" in captured.out
    assert "overall TPR" in captured.out
    assert "per attack type" in captured.out


def test_main_writes_json_report_to_output_path(tmp_path) -> None:
    output_path = tmp_path / "report.json"

    report = run_module.main(["--sample-size", "4", "--output", str(output_path)])

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["sample_size"] == report.sample_size
    assert written["dataset"] == report.dataset


def test_main_respects_similarity_threshold_argument() -> None:
    report = run_module.main(["--sample-size", "4", "--similarity-threshold", "0.5"])
    assert report.similarity_threshold == 0.5
