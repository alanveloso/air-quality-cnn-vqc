#!/usr/bin/env python3
"""CLI for the Farooq-style fair benchmark (selection + final evaluation)."""

from __future__ import annotations

import argparse
import logging
import sys

from qml_air_quality.experiments.fair_benchmark import run_fair_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Farooq-style fair classical vs QSVM benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/farooq_fair_benchmark.yaml",
        help="Path to experiment YAML",
    )
    parser.add_argument(
        "--stage",
        choices=["selection", "final", "all"],
        default="all",
        help="selection=val-only HP search; final=frozen test; all=both",
    )
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Run classical arms only (faster local smoke)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = run_fair_benchmark(
        args.config,
        stage=args.stage,
        skip_quantum=args.skip_quantum,
    )
    if "selection" in result and result["selection"] is not None:
        print(f"selection rows: {len(result['selection'])}")
    if "metrics" in result and result["metrics"] is not None:
        print(f"final metric rows: {len(result['metrics'])}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
