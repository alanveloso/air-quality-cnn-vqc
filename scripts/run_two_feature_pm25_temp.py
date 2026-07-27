#!/usr/bin/env python3
"""Experimento com poucas features brutas ou stats estilo Farooq vs QSVM.

Protocolo temporal da PoC:
  - alvo = extremo de PM2.5 em t+24h (P90 do treino)
  - split temporal 60/20/20
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_air_quality.config import load_config, project_root
from qml_air_quality.data import download_dataset, load_station_csv
from qml_air_quality.features import (
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
from qml_air_quality.preprocessing.angular_scaling import make_angular_scaler
from qml_air_quality.split import stratified_subsample, temporal_split


def _build_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    root = project_root()
    csv_path = download_dataset(
        dataset_id=cfg["data"]["dataset_id"],
        station=cfg["data"]["station"],
        raw_dir=root / cfg["data"]["raw_dir"],
    )
    df = load_station_csv(csv_path)
    mode = cfg["features"].get("mode", "raw")

    if mode == "farooq_stats":
        source_map = dict(cfg["features"]["source_columns"])
        base_cols = list(source_map.keys())
        df = apply_causal_ffill(df, base_cols, limit_hours=cfg["features"]["ffill_limit_hours"])
        df, feat_cols = add_farooq_style_stats(
            df,
            source_map=source_map,
            window=int(cfg["features"].get("rolling_window_hours", 24)),
            min_periods=int(cfg["features"].get("rolling_min_periods", 12)),
        )
    else:
        feat_cols = list(cfg["features"]["raw_columns"])
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}. Disponíveis: {list(df.columns)}")
        df = apply_causal_ffill(df, feat_cols, limit_hours=cfg["features"]["ffill_limit_hours"])

    df = add_future_target(
        df,
        target_column=cfg["data"]["target_column"],
        horizon_hours=cfg["forecast"]["horizon_hours"],
    )
    df = df.dropna(subset=["future_pm25"] + feat_cols).reset_index(drop=True)

    train, val, test = temporal_split(
        df,
        train_fraction=cfg["split"]["train_fraction"],
        validation_fraction=cfg["split"]["validation_fraction"],
        test_fraction=cfg["split"]["test_fraction"],
    )
    train, val, test = impute_with_train_medians(train, val, test, columns=feat_cols)
    train, val, test, tmeta = make_binary_target(
        train, val, test, percentile=cfg["forecast"]["extreme_percentile"]
    )
    return train, val, test, feat_cols, tmeta


def _scale_partitions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: list[str],
    n_pca: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if n_pca is not None:
        steps.append(("pca", PCA(n_components=n_pca, random_state=42)))
    pipe = Pipeline(steps=steps)
    X_tr = pipe.fit_transform(train[feat_cols])
    X_va = pipe.transform(val[feat_cols])
    X_te = pipe.transform(test[feat_cols])
    return X_tr, X_va, X_te, pipe


def _attach_scaled(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, X_tr, X_va, X_te
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    n = X_tr.shape[1]
    cols = [f"f{i}" for i in range(n)]
    tr, va, te = train.copy(), val.copy(), test.copy()
    for i, c in enumerate(cols):
        tr[c] = X_tr[:, i]
        va[c] = X_va[:, i]
        te[c] = X_te[:, i]
    return tr, va, te, cols


def _eval_classical(X_tr, y_tr, X_te, y_te, seed: int) -> pd.DataFrame:
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
        t1 = time.perf_counter()
        y_hat = model.predict(X_te)
        y_prob = predict_proba_positive(model, X_te)
        inf_s = time.perf_counter() - t1
        row = binary_metrics(
            y_te, y_hat, y_prob, model_name=name, training_seconds=train_s, inference_seconds=inf_s
        )
        row["family"] = "classical"
        row["n_features"] = X_tr.shape[1]
        rows.append(row)
        print(f"  {name}: AUPRC={row['average_precision']:.4f} F1={row['f1_extreme']:.4f}")
    return pd.DataFrame(rows)


def _eval_qsvm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    cfg_id: str,
    angular_name: str,
    feature_map: str,
    reps: int,
    entanglement: str,
    seed: int,
    kernel_dir: Path,
) -> dict:
    ang = make_angular_scaler(angular_name, seed=seed)
    X_tr_q = ang.fit_transform(X_tr)
    X_te_q = ang.transform(X_te)

    n_qubits = X_tr_q.shape[1]
    kernel = make_fidelity_kernel(
        n_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        feature_map_name=feature_map,
        enforce_psd=True,
    )
    print(f"  [{cfg_id}] building kernel ({X_tr_q.shape[0]}x{X_tr_q.shape[0]}, {n_qubits}q) …")
    t0 = time.perf_counter()
    K_tr = kernel.evaluate(x_vec=X_tr_q)
    K_te = kernel.evaluate(x_vec=X_te_q, y_vec=X_tr_q)
    kernel_s = time.perf_counter() - t0
    print(f"  [{cfg_id}] kernel done in {kernel_s:.1f}s")

    np.save(kernel_dir / f"{cfg_id}_K_train.npy", K_tr)
    np.save(kernel_dir / f"{cfg_id}_K_test.npy", K_te)

    clf = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
    t0 = time.perf_counter()
    clf.fit(K_tr, y_tr)
    train_s = time.perf_counter() - t0
    y_hat = clf.predict(K_te)
    y_prob = predict_proba_positive(clf, K_te)
    row = binary_metrics(y_te, y_hat, y_prob, model_name=cfg_id, training_seconds=train_s)
    row["family"] = "qsvm"
    row["n_features"] = n_qubits
    row["angular_scaler"] = angular_name
    row["feature_map"] = feature_map
    row["reps"] = reps
    row["kernel_seconds"] = kernel_s
    print(
        f"  [{cfg_id}] AUPRC={row['average_precision']:.4f} "
        f"F1={row['f1_extreme']:.4f} recall={row['recall_extreme']:.4f}"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Few features / Farooq-style stats: classical vs QSVM")
    parser.add_argument("--config", type=str, default="config/two_feature_pm25_temp.yaml")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--quantum-ids", type=str, default="")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    seed = int(cfg["project"]["random_seed"])
    np.random.seed(seed)

    out = root / cfg["paths"]["artifacts_dir"]
    plots = out / "plots"
    kernels = out / "kernels"
    for d in (out, plots, kernels):
        d.mkdir(parents=True, exist_ok=True)

    train, val, test, feat_cols, tmeta = _build_frames(cfg)
    print(f"== prepare ({len(feat_cols)} features) ==")
    print("features:", feat_cols)
    print("threshold:", tmeta["threshold"])
    print(
        "sizes full:",
        len(train),
        len(val),
        len(test),
        "pos rate train:",
        float(train["target"].mean()),
    )

    # Classical: all engineered features (StandardScaler only)
    X_tr_c, X_va_c, X_te_c, _ = _scale_partitions(train, val, test, feat_cols, n_pca=None)
    tr_c, va_c, te_c, fcols_c = _attach_scaled(train, val, test, X_tr_c, X_va_c, X_te_c)

    n_tr = args.train_size or int(cfg["sampling"]["train_size"])
    n_va = args.val_size or int(cfg["sampling"]["validation_size"])
    n_te = args.test_size or int(cfg["sampling"]["test_size"])
    tr_s, va_s, te_s, idx = stratified_subsample(
        tr_c, va_c, te_c, train_size=n_tr, validation_size=n_va, test_size=n_te, seed=seed
    )
    idx.to_csv(out / f"sample_indices_seed_{seed}.csv", index=False)

    X_tr = tr_s[fcols_c].to_numpy()
    X_te = te_s[fcols_c].to_numpy()
    y_tr = tr_s["target"].to_numpy()
    y_te = te_s["target"].to_numpy()
    print(f"subsample: train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
    print(f"positives: train={int(y_tr.sum())} test={int(y_te.sum())}")

    print(f"== classical ({X_tr.shape[1]} features) ==")
    classical = _eval_classical(X_tr, y_tr, X_te, y_te, seed=seed)

    # Quantum path: optional PCA (Farooq: 8 stats → PCA → 2 dims → MinMax)
    n_pca_q = cfg.get("pca_for_quantum")
    q_rows: list[dict] = []
    if not args.skip_quantum:
        print(f"== quantum (pca_for_quantum={n_pca_q}) ==")
        if n_pca_q is not None:
            X_tr_q_full, _, X_te_q_full, pipe_q = _scale_partitions(
                train, val, test, feat_cols, n_pca=int(n_pca_q)
            )
            explained = pipe_q.named_steps["pca"].explained_variance_ratio_.tolist()
            print("PCA explained variance:", explained)
            # Reuse same subsample indices via original_index
            tr_q = train.copy()
            te_q = test.copy()
            for i in range(int(n_pca_q)):
                tr_q[f"pc{i}"] = X_tr_q_full[:, i]
                te_q[f"pc{i}"] = X_te_q_full[:, i]
            tr_q = tr_q.reset_index(names="original_index")
            te_q = te_q.reset_index(names="original_index")
            # Map from classical subsample original indices
            tr_idx = tr_s["original_index"].to_numpy()
            te_idx = te_s["original_index"].to_numpy()
            pcs = [f"pc{i}" for i in range(int(n_pca_q))]
            X_tr_q = tr_q.set_index("original_index").loc[tr_idx, pcs].to_numpy()
            X_te_q = te_q.set_index("original_index").loc[te_idx, pcs].to_numpy()
            y_tr_q, y_te_q = y_tr, y_te
        else:
            X_tr_q, X_te_q, y_tr_q, y_te_q = X_tr, X_te, y_tr, y_te

        wanted = {x.strip() for x in args.quantum_ids.split(",") if x.strip()}
        for qcfg in cfg["quantum_variants"]:
            if wanted and qcfg["id"] not in wanted:
                continue
            q_rows.append(
                _eval_qsvm(
                    X_tr_q,
                    y_tr_q,
                    X_te_q,
                    y_te_q,
                    cfg_id=qcfg["id"],
                    angular_name=qcfg["angular_scaler"],
                    feature_map=qcfg["feature_map"],
                    reps=int(qcfg["reps"]),
                    entanglement=qcfg.get("entanglement", "linear"),
                    seed=seed,
                    kernel_dir=kernels,
                )
            )

    comparison = pd.concat(
        [classical, pd.DataFrame(q_rows)] if q_rows else [classical],
        ignore_index=True,
    )
    comparison = comparison.sort_values("average_precision", ascending=False).reset_index(drop=True)
    comparison.to_csv(out / "comparison_metrics.csv", index=False)

    meta = {
        "seed": seed,
        "features": feat_cols,
        "n_features": len(feat_cols),
        "mode": cfg["features"].get("mode", "raw"),
        "pca_for_quantum": n_pca_q,
        "threshold": tmeta["threshold"],
        "train_size": len(tr_s),
        "val_size": len(va_s),
        "test_size": len(te_s),
        "positives_train": int(y_tr.sum()),
        "positives_test": int(y_te.sum()),
        "feature_notes": cfg["features"].get("notes", ""),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")

    title = f"Farooq-style stats ({len(feat_cols)}f) — classical vs QSVM"
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(comparison["model"], comparison["average_precision"])
    ax.set_xlabel("AUPRC (test)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(plots / "comparison_auprc.png", dpi=140)
    plt.close(fig)

    print("\n== ranking (test AUPRC) ==")
    cols = [
        c
        for c in [
            "model",
            "average_precision",
            "precision_extreme",
            "recall_extreme",
            "f1_extreme",
            "family",
            "n_features",
            "angular_scaler",
            "reps",
            "kernel_seconds",
        ]
        if c in comparison.columns
    ]
    print(comparison[cols].to_string(index=False))
    print(f"\nArtifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
