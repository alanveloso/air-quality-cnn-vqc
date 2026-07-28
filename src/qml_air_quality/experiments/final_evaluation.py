"""Final evaluation stage wrapper for the fair benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qml_air_quality.experiments.fair_benchmark import run_fair_benchmark


def run_final_evaluation(
    config_path: str | Path,
    skip_quantum: bool = False,
) -> dict[str, Any]:
    """Run only the final (frozen) evaluation stage."""
    return run_fair_benchmark(config_path, stage="final", skip_quantum=skip_quantum)
