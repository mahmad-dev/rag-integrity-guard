from __future__ import annotations

from dataclasses import dataclass

# FEVER reformatted for NLI: each row already carries the claim ("premise")
# paired with its gold Wikipedia evidence text ("hypothesis"), so no separate
# multi-GB Wikipedia dump has to be downloaded and joined. A single split is
# one parquet file (~4-5MB for dev/test), cached under the user's
# `~/.cache/huggingface`, never written into the repo.
FEVER_DATASET_ID = "pietrolesci/nli_fever"
_VERIFIABLE_LABELS = ("SUPPORTS", "REFUTES")


@dataclass(frozen=True)
class FeverExample:
    claim_id: str
    claim: str
    evidence: str
    fever_label: str  # "SUPPORTS" | "REFUTES"


def load_fever_examples(
    *,
    split: str = "dev",
    limit: int | None = None,
    seed: int = 42,
) -> list[FeverExample]:
    """Load (claim, gold evidence) pairs from FEVER, keeping only claims that
    are actually verifiable against a specific passage (SUPPORTS/REFUTES) --
    a NOT ENOUGH INFO claim has no single gold chunk to attack and re-verify.
    Deduplicated to one example per unique evidence passage.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "rag_guard.eval requires the 'eval' extra: pip install 'rag-guard[eval]'"
        ) from exc

    raw = load_dataset(FEVER_DATASET_ID, split=split).shuffle(seed=seed)

    examples: list[FeverExample] = []
    seen_evidence: set[str] = set()
    for row in raw:
        label = row["fever_gold_label"]
        if label not in _VERIFIABLE_LABELS:
            continue

        evidence = (row["hypothesis"] or "").strip()
        claim = (row["premise"] or "").strip()
        if not evidence or not claim or evidence in seen_evidence:
            continue

        seen_evidence.add(evidence)
        examples.append(
            FeverExample(
                claim_id=str(row["cid"]),
                claim=claim,
                evidence=evidence,
                fever_label=label,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples
