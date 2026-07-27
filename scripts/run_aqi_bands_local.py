#!/usr/bin/env python3
"""Smoke local: Farooq stats + alvos por faixa AQI (além do extremo P90).

Alvos:
  1) extreme_p90     — futuro PM2.5 >= P90 treino (tarefa original)
  2) aqi_bad         — t+24h fora de Good/Moderate (EPA PM2.5)
  3) aqi_good_vs_mod — só Good vs Moderate (mais perto do Farooq binário)

Uso:
  .venv/bin/python scripts/run_aqi_bands_local.py
  .venv/bin/python scripts/run_aqi_bands_local.py --train-size 80 --skip-quantum
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qml_air_quality.config import project_root
from qml_air_quality.data import download_dataset, load_station_csv
from qml_air_quality.features import (
    add_aqi_targets,
    add_farooq_style_stats,
    add_future_target,
    apply_causal_ffill,
    impute_with_train_medians,
    make_binary_target,
)
from qml_air_quality.metrics import binary_metrics
from qml_air_quality.models import (
    make_dummy,
    make_fidelity_kernel,
    make_linear_svm,
    make_logistic,
    make_rbf_svm,
    predict_proba_positive,
)
from qml_air_quality.split import stratified_subsample, temporal_split


def _prepare(root: Path):
    csv_path = download_dataset(station="Aotizhongxin", raw_dir=root / "data" / "raw")
    df = load_station_csv(csv_path)
    df = apply_causal_ffill(df, ["PM2.5", "TEMP"], limit_hours=3)
    df, feat_cols = add_farooq_style_stats(df, window=24, min_periods=12)
    df = add_future_target(df, target_column="PM2.5", horizon_hours=24)
    df = df.dropna(subset=["future_pm25"] + feat_cols).reset_index(drop=True)

    train, val, test = temporal_split(df, 0.60, 0.20, 0.20)
    train, val, test = impute_with_train_medians(train, val, test, columns=feat_cols)
    train, val, test, tmeta = make_binary_target(train, val, test, percentile=0.90)
    train, val, test, ameta = add_aqi_targets(train, val, test, value_col="future_pm25")
    return train, val, test, feat_cols, tmeta, ameta


def _eval_task(
    train, val, test, feat_cols, target_col: str, *,
    train_size: int, test_size: int, seed: int, skip_quantum: bool, out: Path, task_name: str,
):
    # evita colisão com a coluna 'target' do extremo P90
    tr0 = train.dropna(subset=[target_col]).copy()
    te0 = test.dropna(subset=[target_col]).copy()
    va0 = val.dropna(subset=[target_col]).copy()
    for part in (tr0, te0, va0):
        part["target"] = part[target_col].astype(int)
        # remove colunas auxiliares que não entram no subsample
        drop_cols = [c for c in ("aqi_bad", "aqi_good_vs_moderate") if c in part.columns and c != target_col]
        # se o alvo não era 'target', remove o target extremo antigo após copiar
        if target_col != "target" and "target" in part.columns:
            # já sobrescrevemos 'target' acima — ok
            pass

    print(f"\n===== TASK {task_name} (target={target_col}) =====")
    print("aqi_band train:\n", train["aqi_band"].value_counts().to_string())
    print("pos rate train/test:", float(tr0["target"].mean()), float(te0["target"].mean()))
    print("sizes before subsample:", len(tr0), len(va0), len(te0))

    if int(tr0["target"].nunique()) < 2 or int(te0["target"].nunique()) < 2:
        print("SKIP: precisa das duas classes no treino e teste")
        return None
    if int(va0["target"].nunique()) < 2:
        va0 = pd.concat([va0, tr0.groupby("target", as_index=False).head(1)], ignore_index=True)

    tr_s, _, te_s, _ = stratified_subsample(
        tr0, va0, te0,
        train_size=train_size,
        validation_size=min(40, len(va0)),
        test_size=test_size,
        seed=seed,
    )
    print("subsample:", len(tr_s), len(te_s), "pos:", int(tr_s["target"].sum()), int(te_s["target"].sum()))

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_tr = pipe.fit_transform(tr_s[feat_cols])
    X_te = pipe.transform(te_s[feat_cols])
    y_tr = tr_s["target"].to_numpy()
    y_te = te_s["target"].to_numpy()

    models = {
        "dummy": make_dummy(seed=seed),
        "logistic": make_logistic(seed=seed),
        "svm_linear": make_linear_svm(seed=seed),
        "svm_rbf": make_rbf_svm(seed=seed),
    }
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        y_hat = model.predict(X_te)
        y_prob = predict_proba_positive(model, X_te)
        row = binary_metrics(y_te, y_hat, y_prob, model_name=name, training_seconds=train_s)
        row["task"] = task_name
        row["family"] = "classical"
        rows.append(row)
        print(f"  {name}: Acc≈{(y_hat==y_te).mean():.3f} AUPRC={row['average_precision']:.3f} F1={row['f1_extreme']:.3f}")

    if not skip_quantum:
        pipe_q = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=seed)),
        ])
        X_tr_q = pipe_q.fit_transform(tr_s[feat_cols])
        X_te_q = pipe_q.transform(te_s[feat_cols])
        ang = MinMaxScaler(feature_range=(0.0, 1.0))
        X_tr_q = ang.fit_transform(X_tr_q)
        X_te_q = ang.transform(X_te_q)
        kernel = make_fidelity_kernel(n_qubits=2, reps=1, feature_map_name="ZZFeatureMap")
        print("  [Q_farooq] building kernel…")
        t0 = time.perf_counter()
        K_tr = kernel.evaluate(x_vec=X_tr_q)
        K_te = kernel.evaluate(x_vec=X_te_q, y_vec=X_tr_q)
        ks = time.perf_counter() - t0
        clf = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
        clf.fit(K_tr, y_tr)
        y_hat = clf.predict(K_te)
        y_prob = predict_proba_positive(clf, K_te)
        row = binary_metrics(y_te, y_hat, y_prob, model_name="Q_farooq_pipeline")
        row["task"] = task_name
        row["family"] = "qsvm"
        row["kernel_seconds"] = ks
        rows.append(row)
        print(f"  Q_farooq: AUPRC={row['average_precision']:.3f} F1={row['f1_extreme']:.3f} ({ks:.1f}s)")

    df = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    df.to_csv(out / f"comparison_{task_name}.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--test-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-quantum", action="store_true")
    args = parser.parse_args()

    root = project_root()
    out = root / "artifacts" / "aqi_bands_local"
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    print("== prepare ==")
    train, val, test, feat_cols, tmeta, ameta = _prepare(root)
    print("features:", feat_cols)
    print("extreme threshold:", tmeta["threshold"])

    # distribuição das faixas (futuro t+24)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts = train["aqi_band"].value_counts()
    order = ["Good", "Moderate", "Unhealthy_Sensitive", "Unhealthy", "Very_Unhealthy", "Hazardous"]
    counts = counts.reindex([c for c in order if c in counts.index])
    ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    ax.set_title("Faixas AQI (EPA) no treino — future PM2.5 t+24h")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots / "aqi_band_distribution_train.png", dpi=140)
    plt.close(fig)

    # export features + bands
    export_cols = ["timestamp"] + feat_cols + [
        "PM2.5", "TEMP", "future_pm25", "target", "aqi_band", "aqi_bad", "aqi_good_vs_moderate"
    ]
    parts = []
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        tmp = part[[c for c in export_cols if c in part.columns]].copy()
        tmp["partition"] = name
        parts.append(tmp)
    pd.concat(parts, ignore_index=True).to_csv(out / "features_with_aqi_bands.csv", index=False)

    tasks = [
        ("extreme_p90", "target"),
        ("aqi_bad", "aqi_bad"),
        ("aqi_good_vs_moderate", "aqi_good_vs_moderate"),
    ]
    summaries = []
    for task_name, col in tasks:
        df = _eval_task(
            train, val, test, feat_cols, col,
            train_size=args.train_size, test_size=args.test_size,
            seed=args.seed, skip_quantum=args.skip_quantum, out=out, task_name=task_name,
        )
        if df is not None:
            summaries.append(df)

    if summaries:
        all_df = pd.concat(summaries, ignore_index=True)
        all_df.to_csv(out / "comparison_all_tasks.csv", index=False)
        # ranking plot por task
        fig, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 4), sharey=False)
        if len(summaries) == 1:
            axes = [axes]
        for ax, df in zip(axes, summaries):
            d = df.sort_values("average_precision")
            ax.barh(d["model"], d["average_precision"])
            ax.set_title(df["task"].iloc[0])
            ax.set_xlabel("AUPRC")
        fig.tight_layout()
        fig.savefig(plots / "auprc_by_task.png", dpi=140)
        plt.close(fig)
        print("\n=== ALL TASKS ===")
        print(all_df[["task", "model", "average_precision", "f1_extreme", "family"]].to_string(index=False))

    meta = {
        "extreme_threshold": tmeta,
        "aqi": ameta,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "skip_quantum": args.skip_quantum,
        "note": "Local smoke: Farooq features + AQI bands (EPA) on future PM2.5 t+24h",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nArtifacts:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
