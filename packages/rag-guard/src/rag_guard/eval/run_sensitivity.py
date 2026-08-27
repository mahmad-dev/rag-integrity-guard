"""Executable threshold-sensitivity sweep: `python -m rag_guard.eval.run_sensitivity`.

Holds a single tampered FEVER corpus fixed and sweeps `similarity_threshold`
across it, showing how the SEMANTIC_DRIFT vs HASH_MISMATCH classification
split shifts per attack type. See `sensitivity.py` for why TPR/FPR aren't
part of this sweep -- they're a hash-check guarantee the threshold can't move.
"""

from __future__ import annotations

import argparse

from rag_guard.eval.dataset import load_fever_examples
from rag_guard.eval.sensitivity import ThresholdRow, run_threshold_sensitivity

DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=f"Similarity thresholds to sweep (default: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser


def _print_table(rows: list[ThresholdRow]) -> None:
    print("\nthreshold  attack_type          n     drift  mismatch  mean_sim  median_sim")
    for row in rows:
        print(
            f"{row.similarity_threshold:<10.2f}"
            f"{row.attack_type:<21}"
            f"{row.sample_count:<6}"
            f"{row.classified_semantic_drift:<7}"
            f"{row.classified_hash_mismatch:<10}"
            f"{row.mean_similarity:<10.4f}"
            f"{row.median_similarity:.4f}"
        )


def main(argv: list[str] | None = None) -> list[ThresholdRow]:
    args = _build_arg_parser().parse_args(argv)

    examples = load_fever_examples(split=args.split, limit=args.sample_size, seed=args.seed)
    rows = run_threshold_sensitivity(examples, args.thresholds, seed=args.seed)

    _print_table(rows)

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump([row.model_dump() for row in rows], f, indent=2)
        print(f"\nwrote {args.output}")

    return rows


if __name__ == "__main__":
    main()
