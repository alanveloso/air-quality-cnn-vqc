"""Smoke run of the notebook pipeline (non-interactive)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.svm import SVC

from qml_air_quality.config import load_config, project_root
from qml_air_quality.data import download_dataset, load_station_csv, missing_report
from qml_air_quality.features import (
    add_calendar_features,
    add_future_target,
    add_lag_and_rolling,
    apply_causal_ffill,
    feature_columns,
    impute_with_train_medians,
    make_binary_target,
    save_target_metadata,
)
from qml_air_quality.metrics import binary_metrics, metrics_table
from qml_air_quality.models import (
    make_dummy,
    make_linear_svm,
    make_logistic,
    make_qsvc,
    make_rbf_svm,
    predict_proba_positive,
)
from qml_air_quality.preprocess import fit_transform_pca, save_pipeline
from qml_air_quality.split import save_sample_indices, stratified_subsample, temporal_split

# Smaller quantum subset for a timely smoke run; notebooks keep full YAML sizes.
SMOKE_TRAIN, SMOKE_VAL, SMOKE_TEST = 60, 30, 30


def main() -> None:
    ROOT = project_root()
    cfg = load_config()
    SEED = cfg["project"]["random_seed"]
    np.random.seed(SEED)

    datasets = ROOT / cfg["paths"]["datasets_dir"]
    kernels = ROOT / cfg["paths"]["kernels_dir"]
    plots = ROOT / cfg["paths"]["plots_dir"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    for d in (datasets, kernels, plots, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("== download ==")
    csv_path = download_dataset(
        dataset_id=cfg["data"]["dataset_id"],
        station=cfg["data"]["station"],
        raw_dir=ROOT / cfg["data"]["raw_dir"],
    )
    df = load_station_csv(csv_path)
    raw_cols = cfg["features"]["raw_columns"]
    miss = missing_report(df, raw_cols)
    print(miss.to_string(index=False))

    print("== features ==")
    df = apply_causal_ffill(df, raw_cols, limit_hours=cfg["features"]["ffill_limit_hours"])
    df = add_lag_and_rolling(
        df,
        columns=raw_cols,
        lag_hours=cfg["features"]["lag_hours"],
        rolling_windows=cfg["features"]["rolling_windows"],
    )
    if cfg["features"]["add_calendar_features"]:
        df = add_calendar_features(df)
    df = add_future_target(
        df,
        target_column=cfg["data"]["target_column"],
        horizon_hours=cfg["forecast"]["horizon_hours"],
    )

    train, val, test = temporal_split(
        df,
        train_fraction=cfg["split"]["train_fraction"],
        validation_fraction=cfg["split"]["validation_fraction"],
        test_fraction=cfg["split"]["test_fraction"],
    )
    feat_cols = feature_columns(train)
    train, val, test = impute_with_train_medians(train, val, test, columns=feat_cols)
    train, val, test, tmeta = make_binary_target(
        train, val, test, percentile=cfg["forecast"]["extreme_percentile"]
    )
    save_target_metadata(tmeta, datasets / "target_metadata.json")
    print("threshold", tmeta)

    n_pca = cfg["features"]["pca_components"]
    X_tr, X_va, X_te, pipe, explained = fit_transform_pca(
        train, val, test, feat_cols, n_components=n_pca
    )
    save_pipeline(pipe, models_dir / "pca_pipeline.joblib")
    (datasets / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")

    train = train.copy()
    val = val.copy()
    test = test.copy()
    for part, X in [(train, X_tr), (val, X_va), (test, X_te)]:
        for i in range(n_pca):
            part[f"pc{i + 1}"] = X[:, i]

    # Full-size sample as configured (for notebooks continuity)
    tr_full, va_full, te_full, idx_full = stratified_subsample(
        train,
        val,
        test,
        train_size=cfg["sampling"]["quantum_train_size"],
        validation_size=cfg["sampling"]["quantum_validation_size"],
        test_size=cfg["sampling"]["quantum_test_size"],
        seed=SEED,
    )
    save_sample_indices(idx_full, datasets / f"quantum_sample_seed_{SEED}.csv")

    # Smoke subsample for classical+quantum comparison in this run
    tr_s, va_s, te_s, idx_smoke = stratified_subsample(
        train,
        val,
        test,
        train_size=SMOKE_TRAIN,
        validation_size=SMOKE_VAL,
        test_size=SMOKE_TEST,
        seed=SEED,
    )
    save_sample_indices(idx_smoke, datasets / f"smoke_sample_seed_{SEED}.csv")

    pc_cols = [f"pc{i + 1}" for i in range(n_pca)]
    # Persist full-size arrays for notebooks
    np.save(datasets / "X_train.npy", tr_full[pc_cols].to_numpy())
    np.save(datasets / "X_val.npy", va_full[pc_cols].to_numpy())
    np.save(datasets / "X_test.npy", te_full[pc_cols].to_numpy())
    np.save(datasets / "y_train.npy", tr_full["target"].to_numpy())
    np.save(datasets / "y_val.npy", va_full["target"].to_numpy())
    np.save(datasets / "y_test.npy", te_full["target"].to_numpy())

    # Smoke arrays
    X_train = tr_s[pc_cols].to_numpy()
    X_test = te_s[pc_cols].to_numpy()
    y_train = tr_s["target"].to_numpy()
    y_test = te_s["target"].to_numpy()
    np.save(datasets / "X_train_smoke.npy", X_train)
    np.save(datasets / "X_test_smoke.npy", X_test)
    np.save(datasets / "y_train_smoke.npy", y_train)
    np.save(datasets / "y_test_smoke.npy", y_test)

    meta = {
        "seed": SEED,
        "n_features_before_pca": len(feat_cols),
        "pca_components": n_pca,
        "explained_variance_ratio": explained.tolist(),
        "train_size": len(tr_full),
        "val_size": len(va_full),
        "test_size": len(te_full),
        "smoke_train_size": len(tr_s),
        "smoke_test_size": len(te_s),
        "threshold": tmeta["threshold"],
    }
    (datasets / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("prepare meta", meta)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, len(explained) + 1), explained)
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(plots / "pca_explained_variance.png", dpi=120)
    plt.close(fig)

    print("== classical ==")
    models = {
        "dummy": make_dummy(seed=SEED),
        "logistic": make_logistic(seed=SEED),
        "svm_linear": make_linear_svm(seed=SEED),
        "svm_rbf": make_rbf_svm(seed=SEED),
    }
    rows = []
    proba_test = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_hat = model.predict(X_test)
        y_prob = predict_proba_positive(model, X_test)
        inf_s = time.perf_counter() - t1
        proba_test[name] = y_prob
        rows.append(
            binary_metrics(
                y_test, y_hat, y_prob, model_name=name, training_seconds=train_s, inference_seconds=inf_s
            )
        )
        print(name, rows[-1]["average_precision"])

    table = metrics_table(rows)
    table.to_csv(datasets / "classical_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, y_prob in proba_test.items():
        if y_prob is None:
            continue
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=axes[0])
    axes[0].set_title("PR classical (smoke)")
    axes[1].barh(table["model"], table["average_precision"])
    axes[1].set_title("AUPRC")
    fig.tight_layout()
    fig.savefig(plots / "classical_pr_roc.png", dpi=120)
    plt.close(fig)

    print("== quantum kernel ==")
    qsvc, qkernel = make_qsvc(
        n_qubits=n_pca,
        reps=cfg["quantum"]["repetitions"],
        entanglement=cfg["quantum"]["entanglement"],
        enforce_psd=cfg["quantum"]["enforce_psd"],
    )
    t0 = time.perf_counter()
    K_train = qkernel.evaluate(x_vec=X_train)
    K_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
    kernel_seconds = time.perf_counter() - t0
    print("kernel seconds", kernel_seconds, "shapes", K_train.shape, K_test.shape)
    assert np.allclose(K_train, K_train.T, atol=1e-5)
    assert np.allclose(np.diag(K_train), 1.0, atol=1e-3)
    np.save(kernels / f"train_kernel_seed_{SEED}_smoke.npy", K_train)
    np.save(kernels / f"test_kernel_seed_{SEED}_smoke.npy", K_test)

    clf = SVC(kernel="precomputed", class_weight="balanced", random_state=SEED)
    t0 = time.perf_counter()
    clf.fit(K_train, y_train)
    train_s = time.perf_counter() - t0
    y_hat = clf.predict(K_test)
    y_prob = predict_proba_positive(clf, K_test)
    q_row = binary_metrics(
        y_test, y_hat, y_prob, model_name="qsvm_fidelity", training_seconds=train_s
    )
    q_row["kernel_seconds"] = kernel_seconds
    print("qsvm", q_row)

    comparison = pd.concat([table, pd.DataFrame([q_row])], ignore_index=True)
    comparison = comparison.sort_values("average_precision", ascending=False).reset_index(drop=True)
    comparison.to_csv(datasets / "comparison_metrics.csv", index=False)
    print(comparison.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(comparison["model"], comparison["average_precision"])
    axes[0].set_title("Classical vs QSVM (smoke)")
    im = axes[1].imshow(K_train, cmap="viridis", aspect="auto")
    axes[1].set_title("Quantum kernel")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots / "comparison_auprc.png", dpi=120)
    fig.savefig(plots / "quantum_kernel_matrix.png", dpi=120)
    plt.close(fig)

    print("SMOKE OK")


if __name__ == "__main__":
    main()
