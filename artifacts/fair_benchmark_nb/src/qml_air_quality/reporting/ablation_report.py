"""Ablation report and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qml_air_quality.config import project_root


def write_ablation_report(
    ablation_dir: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    root = project_root()
    abl = Path(ablation_dir) if ablation_dir else root / "artifacts" / "ablation"
    out = Path(out_path) if out_path else abl / "reports" / "ablation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(abl / "validation_ranking.csv") if (abl / "validation_ranking.csv").exists() else None
    test = pd.read_csv(abl / "test_final.csv") if (abl / "test_final.csv").exists() else None
    selection = json.loads((abl / "selection.json").read_text()) if (abl / "selection.json").exists() else {}
    summary = json.loads((abl / "final_summary.json").read_text()) if (abl / "final_summary.json").exists() else {}

    lines = [
        "# Relatório de ablação — escala angular e kernels quânticos",
        "",
        "## Contexto",
        "",
        "Tarefa temporal sem vazamento: prever se PM2.5 em t+24h excede o P90 do **treino**.",
        "Esta iteração testa se escala angular, dimensionalidade e profundidade do feature map",
        "tornam o kernel quântico mais informativo — **sem** reproduzir a classificação circular",
        "de AQI de Farooq et al. (2024).",
        "",
        "## Seleção (somente validação)",
        "",
        "```json",
        json.dumps(selection, indent=2),
        "```",
        "",
    ]
    if val is not None:
        cols = [
            c
            for c in [
                "id",
                "pca_components",
                "angular_scaler",
                "feature_map",
                "reps",
                "average_precision",
                "recall_extreme",
                "alignment",
                "effective_rank",
                "delta_k",
                "kernel_seconds",
            ]
            if c in val.columns
        ]
        lines += [
            "## Ranking de validação",
            "",
            "```",
            val[cols].to_string(index=False),
            "```",
            "",
        ]
    if test is not None:
        lines += [
            "## Teste final (uma vez)",
            "",
            "```",
            test.to_string(index=False),
            "```",
            "",
        ]
    if summary:
        lines += [
            "## Classificação automática",
            "",
            f"- Rótulo: `{summary.get('classification', {}).get('label')}`",
            f"- Conclusão: {summary.get('classification', {}).get('conclusion')}",
            "",
            "### Bootstrap da diferença (QSVM − clássico pareado)",
            "",
            "```json",
            json.dumps(summary.get("bootstrap", {}), indent=2),
            "```",
            "",
        ]
    lines += [
        "## Comparação metodológica com Farooq et al.",
        "",
        "| Aspecto | Farooq et al. | Esta PoC |",
        "|---|---|---|",
        "| Alvo | AQI contemporâneo (faixas) | Extremo PM2.5 em t+24h |",
        "| Features | PM2.5 + temperatura (~2D) | lags/rolling/meteo → PCA |",
        "| Escala | MinMax [0,1] | ablação none/[0,1]/[0,π]/[-π,π]/quantile |",
        "| Métrica | accuracy | **AUPRC** |",
        "| Split | 80/20 (não temporal explícito) | temporal 60/20/20 |",
        "",
        "## Limitações",
        "",
        "- Uma seed principal na execução padrão.",
        "- Simulador ideal (sem ruído de hardware).",
        "- Subamostra finita para viabilidade do kernel.",
        "- Melhoria estrutural do kernel ≠ vantagem computacional quântica.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def plot_ablation_figures(ablation_dir: str | Path | None = None) -> list[Path]:
    root = project_root()
    abl = Path(ablation_dir) if ablation_dir else root / "artifacts" / "ablation"
    plots = abl / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    val_path = abl / "validation_ranking.csv"
    if not val_path.exists():
        return written
    val = pd.read_csv(val_path)

    fig, ax = plt.subplots(figsize=(9, 4))
    order = val.sort_values("average_precision")
    ax.barh(order["id"], order["average_precision"])
    ax.set_xlabel("AUPRC (validação)")
    ax.set_title("AUPRC de validação por configuração")
    fig.tight_layout()
    p = plots / "validation_auprc_by_configuration.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    if "alignment" in val.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        order = val.sort_values("alignment")
        ax.barh(order["id"], order["alignment"])
        ax.set_xlabel("Kernel-target alignment (treino)")
        ax.set_title("Alinhamento kernel–rótulo")
        fig.tight_layout()
        p = plots / "kernel_alignment_comparison.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if "effective_rank" in val.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        order = val.sort_values("effective_rank")
        ax.barh(order["id"], order["effective_rank"])
        ax.set_xlabel("Effective rank")
        ax.set_title("Posto efetivo dos kernels")
        fig.tight_layout()
        p = plots / "kernel_effective_rank.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if "delta_k" in val.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        order = val.sort_values("delta_k")
        ax.barh(order["id"], order["delta_k"])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("ΔK (same − different)")
        ax.set_title("Contraste entre classes no kernel")
        fig.tight_layout()
        p = plots / "kernel_contrast_same_vs_different.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    # Kernel heatmaps for all cached configs
    kernels = abl / "kernels"
    mats = sorted(kernels.glob("*_seed*_K_train.npy"))
    if mats:
        n = len(mats)
        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes_arr = np.atleast_1d(axes).ravel()
        for ax, path in zip(axes_arr, mats):
            K = np.load(path)
            ax.imshow(K, cmap="viridis", aspect="auto")
            ax.set_title(path.name.split("_seed")[0], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes_arr[len(mats) :]:
            ax.axis("off")
        fig.suptitle("Matrizes de kernel (treino)")
        fig.tight_layout()
        p = plots / "kernel_matrices_by_configuration.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

        fig, ax = plt.subplots(figsize=(8, 4))
        for path in mats:
            K = np.load(path)
            off = K[~np.eye(K.shape[0], dtype=bool)]
            ax.hist(off, bins=40, alpha=0.35, label=path.name.split("_seed")[0], density=True)
        ax.set_xlabel("K_ij (i≠j)")
        ax.set_title("Histogramas dos valores fora da diagonal")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = plots / "kernel_value_histograms.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    test_path = abl / "test_final.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(test["id"].astype(str), test["average_precision"])
        ax.set_xlabel("AUPRC (teste)")
        ax.set_title("QSVM vs clássicos pareados (teste)")
        fig.tight_layout()
        p = plots / "qsvm_vs_classical_paired.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    return written
