"""Classify pairwise QSVM vs classical results without overclaiming."""

from __future__ import annotations

from typing import Any


def classify_fair_result(
    delta_auprc_mean: float,
    delta_auprc_ci_low: float,
    delta_f2_mean: float,
    wins_majority_seeds: bool,
    beats_full_without_threshold_fix: bool = False,
    min_delta_auprc: float = 0.02,
) -> dict[str, str]:
    """Assign interpretation labels for the fair benchmark.

    Never use 'quantum advantage' wording when the CI includes zero.
    """
    clear_gain = (
        delta_auprc_mean >= min_delta_auprc
        and delta_auprc_ci_low > 0
        and delta_f2_mean >= 0
        and wins_majority_seeds
    )
    if clear_gain:
        return {
            "label": "CANDIDATE_QUANTUM_GAIN",
            "conclusion": (
                "QSVM superou o clássico pareado em AUPRC com IC da diferença acima de zero, "
                "F2 não inferior e maioria das sementes positivas. Resultado preliminar — "
                "não interpretar como vantagem computacional quântica geral."
            ),
        }
    if beats_full_without_threshold_fix and not clear_gain:
        return {
            "label": "THRESHOLD_OR_CAPACITY_EFFECT",
            "conclusion": (
                "O ganho aparente frente ao modelo completo sem correção de limiar "
                "provavelmente reflete capacidade ou limiar, não o kernel quântico."
            ),
        }
    return {
        "label": "NO_CLEAR_QUANTUM_GAIN",
        "conclusion": (
            "Empate ou diferença não estável frente ao melhor modelo pareado "
            "(intervalo de confiança da ΔAUPRC inclui zero ou critérios de ganho não atendidos)."
        ),
    }


def majority_seed_wins(deltas_by_seed: list[float]) -> bool:
    if not deltas_by_seed:
        return False
    return sum(1 for d in deltas_by_seed if d > 0) > len(deltas_by_seed) / 2


def summarize_pairwise(
    comparison_name: str,
    boot: dict[str, Any],
    seed_deltas: list[float],
    beats_full_without_threshold_fix: bool = False,
) -> dict[str, Any]:
    wins = majority_seed_wins(seed_deltas)
    classification = classify_fair_result(
        delta_auprc_mean=float(boot.get("delta_auprc_mean", float("nan"))),
        delta_auprc_ci_low=float(boot.get("delta_auprc_ci_low", float("nan"))),
        delta_f2_mean=float(boot.get("delta_f2_mean", float("nan"))),
        wins_majority_seeds=wins,
        beats_full_without_threshold_fix=beats_full_without_threshold_fix,
    )
    return {
        "comparison": comparison_name,
        **boot,
        "seed_delta_auprc_mean": float(sum(seed_deltas) / len(seed_deltas)) if seed_deltas else float("nan"),
        "wins_majority_seeds": wins,
        **classification,
    }
