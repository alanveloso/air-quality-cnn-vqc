#!/usr/bin/env python
"""Run quantum angular ablation (Q01–Q10) and write report/plots."""

from __future__ import annotations

import argparse
import logging

from qml_air_quality.experiments.quantum_ablation import run_ablation
from qml_air_quality.reporting.ablation_report import plot_ablation_figures, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-kernels", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ablation(force_kernels=args.force_kernels, skip_test=args.skip_test)
    report = write_ablation_report()
    plots = plot_ablation_figures()
    print("Report:", report)
    print("Plots:", len(plots))


if __name__ == "__main__":
    main()
