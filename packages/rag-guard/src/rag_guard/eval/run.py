"""Executable benchmark: `python -m rag_guard.eval.run`.

Loads a sample of FEVER (claim, gold evidence) pairs, fingerprints the
evidence at "ingestion", then tampers a live copy of the corpus with each
attack type before re-verifying it through `IntegrityGuardRetriever`.
Reports True Positive Rate / False Positive Rate per attack type and the
per-call verification overhead. Core logic lives in `benchmark.py`; this
module is CLI plumbing only.
"""

from __future__ import annotations

import argparse

from rag_guard.core.fingerprint import DEFAULT_SIMILARITY_THRESHOLD
from rag_guard.eval.benchmark import run_benchmark
from rag_guard.eval.dataset import load_fever_examples
from rag_guard.eval.report import BenchmarkReport


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="FEVER split to load (default: dev)")
    parser.add_argument(
        "--sample-size", type=int, default=200, help="Number of unique claims to evaluate"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity above which a hash mismatch is classified SEMANTIC_DRIFT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to write the JSON report"
    )
    return parser


def _print_report(report: BenchmarkReport) -> None:
    print(f"\nrag-integrity-guard benchmark -- {report.dataset} ({report.split})")
    print(
        f"sample size: {report.sample_size}  similarity threshold: {report.similarity_threshold}"
    )
    print(f"overall TPR: {report.overall_true_positive_rate:.3f}")
    print(f"overall FPR: {report.overall_false_positive_rate:.3f}")
    print(
        f"verification overhead: mean={report.verification_overhead_ms_mean:.3f}ms  "
        f"p95={report.verification_overhead_ms_p95:.3f}ms"
    )
    print("\nper attack type:")
    for metric in report.per_attack_type:
        tpr = (
            f"{metric.true_positive_rate:.3f}" if metric.true_positive_rate is not None else "n/a"
        )
        print(f"  {metric.attack_type:<20} n={metric.sample_count:<5} TPR={tpr}")


def main(argv: list[str] | None = None) -> BenchmarkReport:
    args = _build_arg_parser().parse_args(argv)

    examples = load_fever_examples(split=args.split, limit=args.sample_size, seed=args.seed)
    report = run_benchmark(
        examples,
        split=args.split,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed,
    )

    _print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nwrote {args.output}")

    return report


if __name__ == "__main__":
    main()
