"""Plots for the Farooq-style fair benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

FAMILY_COLORS = {
    "classical_full": "#4C78A8",
    "classical_paired": "#F58518",
    "qsvm": "#54A24B",
}


def _plots_dir(artifacts_dir: Path) -> Path:
    out = artifacts_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_summary(artifacts_dir: Path) -> pd.DataFrame | None:
    path = artifacts_dir / "final_metrics_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _prediction_csvs(artifacts_dir: Path) -> list[Path]:
    pred_dir = artifacts_dir / "predictions"
    if not pred_dir.exists():
        return []
    return sorted(pred_dir.glob("*.csv"))


def _parse_pred_stem(stem: str) -> dict[str, Any] | None:
    """Parse '{model}_{seed}_{train_size}' with model possibly containing underscores."""
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    model, seed_s, n_s = parts
    try:
        return {"model": model, "seed": int(seed_s), "train_size": int(n_s)}
    except ValueError:
        return None


def _mean_scores_by_model(
    artifacts_dir: Path, train_size: int | None = None
) -> dict[str, pd.DataFrame]:
    """Average scores across seeds on the fixed test set (same row order)."""
    buckets: dict[str, list[pd.DataFrame]] = {}
    for path in _prediction_csvs(artifacts_dir):
        meta = _parse_pred_stem(path.stem)
        if meta is None:
            continue
        if train_size is not None and meta["train_size"] != train_size:
            continue
        df = pd.read_csv(path)
        key = f"{meta['model']}|{meta['train_size']}"
        buckets.setdefault(key, []).append(df)

    out: dict[str, pd.DataFrame] = {}
    for key, frames in buckets.items():
        base = frames[0][["timestamp", "y_true", "original_index"]].copy() if "original_index" in frames[0] else frames[0][["timestamp", "y_true"]].copy()
        scores = np.mean([f["score"].to_numpy(dtype=float) for f in frames], axis=0)
        # majority / mean-threshold pred from mean score vs first threshold
        thr = float(frames[0]["threshold"].iloc[0])
        base["score"] = scores
        base["y_pred"] = (scores >= thr).astype(int)
        base["threshold"] = thr
        base["model"] = key.split("|")[0]
        base["train_size"] = int(key.split("|")[1])
        out[key] = base
    return out


def plot_auprc_bars(artifacts_dir: Path) -> Path | None:
    summary = _load_summary(artifacts_dir)
    if summary is None or summary.empty:
        return None
    plots = _plots_dir(artifacts_dir)
    train_sizes = sorted(summary["train_size"].unique())
    n = len(train_sizes)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ts in zip(axes, train_sizes):
        sub = summary[summary["train_size"] == ts].sort_values("average_precision_mean")
        colors = [FAMILY_COLORS.get(f, "#999999") for f in sub["family"]]
        y = np.arange(len(sub))
        err = sub["average_precision_std"].fillna(0).to_numpy()
        ax.barh(y, sub["average_precision_mean"], xerr=err, color=colors, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"])
        ax.set_xlabel("AUPRC (teste)")
        ax.set_title(f"train_size={ts}")
        ax.set_xlim(0, max(0.05, float(sub["average_precision_mean"].max()) * 1.15))

    fig.suptitle("Fair benchmark — AUPRC médio (± std entre sementes)", y=1.02)
    fig.tight_layout()
    out = plots / "auprc_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pr_roc(artifacts_dir: Path) -> Path | None:
    summary = _load_summary(artifacts_dir)
    if summary is None:
        return None
    train_size = int(summary["train_size"].max())
    scored = _mean_scores_by_model(artifacts_dir, train_size=train_size)
    if not scored:
        return None

    plots = _plots_dir(artifacts_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, df in scored.items():
        name = key.split("|")[0]
        y = df["y_true"].to_numpy()
        s = df["score"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        PrecisionRecallDisplay.from_predictions(y, s, name=name, ax=axes[0])
        RocCurveDisplay.from_predictions(y, s, name=name, ax=axes[1])
    axes[0].set_title(f"Precision–Recall (n={train_size}, scores médios)")
    axes[1].set_title(f"ROC (n={train_size}, scores médios)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = plots / "pr_roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrices(artifacts_dir: Path, max_models: int = 6) -> Path | None:
    summary = _load_summary(artifacts_dir)
    if summary is None:
        return None
    train_size = int(summary["train_size"].max())
    scored = _mean_scores_by_model(artifacts_dir, train_size=train_size)
    if not scored:
        return None

    # Prefer qsvm + paired + best full by AUPRC
    ranking = summary[summary["train_size"] == train_size].sort_values(
        "average_precision_mean", ascending=False
    )
    wanted = []
    for _, row in ranking.iterrows():
        key = f"{row['model']}|{train_size}"
        if key in scored and key not in wanted:
            wanted.append(key)
        if len(wanted) >= max_models:
            break

    n = len(wanted)
    if n == 0:
        return None
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes_list, wanted):
        df = scored[key]
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        ax.set_title(key.split("|")[0], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes_list[n:]:
        ax.axis("off")
    fig.suptitle(f"Matrizes de confusão (limiar F2 congelado, n={train_size})", y=1.02)
    fig.tight_layout()
    out = _plots_dir(artifacts_dir) / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bootstrap_deltas(artifacts_dir: Path) -> Path | None:
    path = artifacts_dir / "pairwise_bootstrap.csv"
    if not path.exists() or path.stat().st_size < 2:
        return None
    boot = pd.read_csv(path)
    if boot.empty or "delta_auprc_mean" not in boot.columns:
        return None

    plots = _plots_dir(artifacts_dir)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * len(boot) + 1)))
    y = np.arange(len(boot))
    means = boot["delta_auprc_mean"].to_numpy()
    lo = boot["delta_auprc_ci_low"].to_numpy()
    hi = boot["delta_auprc_ci_high"].to_numpy()
    for yi, mean, lo_i, hi_i in zip(y, means, lo, hi):
        if mean > 0 and lo_i > 0:
            color = "#54A24B"
        elif mean < 0 and hi_i < 0:
            color = "#E45756"
        else:
            color = "#999999"
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo_i], [hi_i - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
        )
    ax.axvline(0, color="gray", ls="--", lw=1)
    labels = [f"{r.get('comparison', '')}"[:70] for _, r in boot.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ΔAUPRC (QSVM − baseline) com IC 95%")
    ax.set_title("Block bootstrap — diferenças pareadas")
    fig.tight_layout()
    out = plots / "bootstrap_deltas.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_learning_curves(artifacts_dir: Path) -> Path | None:
    summary = _load_summary(artifacts_dir)
    if summary is None:
        return None
    sizes = sorted(summary["train_size"].unique())
    if len(sizes) < 2:
        return None
    plots = _plots_dir(artifacts_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, g in summary.groupby("model"):
        g = g.sort_values("train_size")
        fam = g["family"].iloc[0]
        ax.errorbar(
            g["train_size"],
            g["average_precision_mean"],
            yerr=g["average_precision_std"].fillna(0),
            marker="o",
            label=model,
            color=FAMILY_COLORS.get(fam, None),
            capsize=3,
        )
    ax.set_xlabel("Tamanho do treino")
    ax.set_ylabel("AUPRC (teste)")
    ax.set_title("Curva amostral — eficiência com n=200 vs 500")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = plots / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_selection_quantum(artifacts_dir: Path) -> Path | None:
    path = artifacts_dir / "model_selection.csv"
    if not path.exists():
        return None
    sel = pd.read_csv(path)
    q = sel[sel["model_id"].astype(str).str.startswith("qsvm")].copy()
    if q.empty:
        return None
    agg = (
        q.groupby("model_id", as_index=False)
        .agg(val_auprc=("val_auprc", "mean"), val_f2=("val_f2", "mean"))
        .sort_values("val_auprc")
    )
    plots = _plots_dir(artifacts_dir)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(agg))
    ax.barh(y, agg["val_auprc"], color="#54A24B")
    ax.set_yticks(y)
    ax.set_yticklabels(agg["model_id"])
    ax.set_xlabel("AUPRC (validação, média das sementes)")
    ax.set_title("Seleção quântica Q01–Q04 (só validação)")
    fig.tight_layout()
    out = plots / "quantum_selection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_kernel_heatmap(artifacts_dir: Path) -> Path | None:
    kernels = sorted((artifacts_dir / "kernels").glob("final_*_K_train.npy"))
    if not kernels:
        kernels = sorted((artifacts_dir / "kernels").glob("Q*_K_train.npy"))
    if not kernels:
        return None
    K = np.load(kernels[0])
    plots = _plots_dir(artifacts_dir)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(K, cmap="viridis", aspect="auto")
    ax.set_title(kernels[0].stem)
    ax.set_xlabel("amostra")
    ax.set_ylabel("amostra")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = plots / "kernel_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_fair_benchmark_plots(artifacts_dir: str | Path) -> list[Path]:
    """Build the standard plot panel from saved fair-benchmark artifacts."""
    artifacts_dir = Path(artifacts_dir)
    generators = [
        plot_auprc_bars,
        plot_pr_roc,
        plot_confusion_matrices,
        plot_bootstrap_deltas,
        plot_learning_curves,
        plot_selection_quantum,
        plot_kernel_heatmap,
    ]
    written: list[Path] = []
    for fn in generators:
        path = fn(artifacts_dir)
        if path is not None:
            written.append(path)
    return written
