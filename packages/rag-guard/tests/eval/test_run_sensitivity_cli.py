import json

import pytest

from rag_guard.eval import run_sensitivity as run_sensitivity_module
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
        ]
    )
]


@pytest.fixture(autouse=True)
def _fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_sensitivity_module,
        "load_fever_examples",
        lambda split, limit, seed: FAKE_EXAMPLES[:limit],
    )


def test_main_prints_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    rows = run_sensitivity_module.main(["--sample-size", "3", "--thresholds", "0.5", "0.9"])

    assert len(rows) == 3 * 2  # 3 attack types x 2 thresholds
    captured = capsys.readouterr()
    assert "threshold" in captured.out
    assert "exact_mutation" in captured.out


def test_main_writes_json_output(tmp_path) -> None:
    output_path = tmp_path / "sensitivity.json"

    rows = run_sensitivity_module.main(
        ["--sample-size", "3", "--thresholds", "0.5", "--output", str(output_path)]
    )

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert len(written) == len(rows)
    assert written[0]["similarity_threshold"] == 0.5


def test_default_thresholds_are_used_when_not_specified() -> None:
    rows = run_sensitivity_module.main(["--sample-size", "3"])
    thresholds = {row.similarity_threshold for row in rows}
    assert thresholds == set(run_sensitivity_module.DEFAULT_THRESHOLDS)
