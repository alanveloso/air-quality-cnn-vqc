"""Farooq-style fair benchmark: purge, fixed eval sets, paired PCA-2, two stages."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from qml_air_quality.config import load_config, project_root
from qml_air_quality.data import download_dataset, load_station_csv
from qml_air_quality.evaluation.block_bootstrap import block_bootstrap_deltas
from qml_air_quality.evaluation.statistical_comparison import summarize_pairwise
from qml_air_quality.evaluation.threshold_selection import apply_threshold
from qml_air_quality.experiments.model_selection import (
    freeze_selection,
    ranking_scores,
    select_logistic,
    select_qsvm,
    select_svm,
)
from qml_air_quality.features import (
    add_farooq_style_stats,
    add_future_target,
    apply_causal_ffill,
    impute_with_train_medians,
    make_binary_target,
    save_target_metadata,
)
from qml_air_quality.metrics import binary_metrics
from qml_air_quality.models import (
    make_fidelity_kernel,
    make_linear_svm,
    make_logistic,
    make_precomputed_svm,
    make_rbf_svm,
)
from qml_air_quality.paired_representation import (
    fit_paired_representation,
    save_preprocessors,
)
from qml_air_quality.reporting.fair_benchmark_plots import generate_fair_benchmark_plots
from qml_air_quality.sample_registry import (
    SampleRegistry,
    build_fixed_evaluation_sets,
    sample_train_for_seed,
)
from qml_air_quality.split import apply_temporal_purge, temporal_split

logger = logging.getLogger(__name__)

QUANTUM_CONFIGS = [
    {"id": "Q01", "angular_scaler": "minmax_0_1", "reps": 1, "farooq_style": False},
    {"id": "Q02", "angular_scaler": "minmax_0_1", "reps": 2, "farooq_style": True},
    {"id": "Q03", "angular_scaler": "minmax_0_pi", "reps": 1, "farooq_style": False},
    {"id": "Q04", "angular_scaler": "minmax_0_pi", "reps": 2, "farooq_style": False},
]

ANGULAR_ALIASES = {
    "minmax_0_1": "0_1",
    "minmax_0_pi": "0_pi",
}


def _artifact_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    base = root / cfg["paths"]["artifacts_dir"]
    dirs = {
        "base": base,
        "samples": base / "sample_indices",
        "preprocessors": base / "preprocessors",
        "kernels": base / "kernels",
        "predictions": base / "predictions",
        "embeddings": base / "embeddings",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _persist_prediction(
    directory: Path,
    key: str,
    *,
    scores: np.ndarray,
    preds: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    original_index: np.ndarray | None = None,
    threshold: float,
    model_id: str,
    seed: int,
    train_size: int,
) -> None:
    """Save raw scores/preds as npz + csv for every model×seed×n."""
    safe = key.replace("|", "_")
    payload: dict[str, Any] = {
        "scores": np.asarray(scores, dtype=float),
        "preds": np.asarray(preds, dtype=int),
        "y": np.asarray(y, dtype=int),
        "timestamps": np.asarray(timestamps),
        "threshold": np.asarray(threshold),
    }
    if original_index is not None:
        payload["original_index"] = np.asarray(original_index)
    np.savez(directory / f"{safe}.npz", **payload)

    rows = {
        "timestamp": pd.to_datetime(timestamps),
        "y_true": y,
        "score": scores,
        "y_pred": preds,
        "threshold": threshold,
        "model": model_id,
        "seed": seed,
        "train_size": train_size,
    }
    if original_index is not None:
        rows["original_index"] = original_index
    pd.DataFrame(rows).to_csv(directory / f"{safe}.csv", index=False)


def prepare_fair_frames(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict, dict]:
    """Download → Farooq stats → split → purge → P90 on purged train."""
    root = project_root()
    csv_path = download_dataset(
        dataset_id=cfg["data"]["dataset_id"],
        station=cfg["data"]["station"],
        raw_dir=root / cfg["data"]["raw_dir"],
    )
    df = load_station_csv(csv_path)
    source_map = dict(cfg["features"]["source_columns"])
    base_cols = list(source_map.keys())
    df = apply_causal_ffill(df, base_cols, limit_hours=cfg["features"]["ffill_limit_hours"])
    df, feat_cols = add_farooq_style_stats(
        df,
        source_map=source_map,
        window=int(cfg["features"]["rolling_window_hours"]),
        min_periods=int(cfg["features"]["rolling_min_periods"]),
    )
    df = add_future_target(
        df,
        target_column=cfg["data"]["target_column"],
        horizon_hours=cfg["experiment"]["horizon_hours"],
    )
    df = df.dropna(subset=["future_pm25"] + feat_cols).reset_index(drop=True)

    train, val, test = temporal_split(
        df,
        train_fraction=cfg["split"]["train_fraction"],
        validation_fraction=cfg["split"]["validation_fraction"],
        test_fraction=cfg["split"]["test_fraction"],
    )
    purge_hours = int(cfg["data"].get("purge_hours", cfg["experiment"]["horizon_hours"]))
    train, val, test, purge_meta = apply_temporal_purge(train, val, test, purge_hours=purge_hours)

    train, val, test = impute_with_train_medians(train, val, test, columns=feat_cols)
    train, val, test, tmeta = make_binary_target(
        train,
        val,
        test,
        percentile=cfg["experiment"]["extreme_percentile"],
        source="purged_training_partition",
        horizon_hours=cfg["experiment"]["horizon_hours"],
    )
    audit = {
        "station": cfg["data"]["station"],
        "n_rows_after_dropna": len(df),
        "feature_columns": feat_cols,
        "train_size_full": len(train),
        "validation_size_full": len(val),
        "test_size_full": len(test),
        "train_positive_rate": float(train["target"].mean()),
    }
    return train, val, test, feat_cols, tmeta, {**purge_meta, **audit}


def _cw(value: Any) -> str | None:
    if value is None or value == "null":
        return None
    return value


def _classical_grids(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "logistic_C": cfg["classical"]["logistic"]["C"],
        "logistic_cw": cfg["classical"]["logistic"]["class_weight"],
        "svm_linear_C": cfg["classical"]["svm_linear"]["C"],
        "svm_linear_cw": cfg["classical"]["svm_linear"]["class_weight"],
        "svm_rbf_C": cfg["classical"]["svm_rbf"]["C"],
        "svm_rbf_cw": cfg["classical"]["svm_rbf"]["class_weight"],
        "svm_rbf_gamma": cfg["classical"]["svm_rbf"]["gamma"],
        "qsvm_C": cfg["quantum"]["C"],
        "qsvm_cw": cfg["quantum"]["class_weight"],
    }


def _eval_row(
    name: str,
    family: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    seed: int,
    train_size: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preds = apply_threshold(scores, threshold)
    row = binary_metrics(y_true, preds, scores, model_name=name)
    row.update(
        {
            "family": family,
            "seed": seed,
            "train_size": train_size,
            "threshold": threshold,
            "split": "test",
        }
    )
    if extra:
        row.update(extra)
    return row


def _fit_frozen_classical(
    family: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if family == "logistic":
        return make_logistic(
            seed=seed,
            C=float(params["C"]),
            class_weight=_cw(params.get("class_weight")),
        ).fit(X_train, y_train)
    if family == "svm_linear":
        return make_linear_svm(
            seed=seed,
            C=float(params["C"]),
            class_weight=_cw(params.get("class_weight")),
            calibrated=False,
        ).fit(X_train, y_train)
    if family == "svm_rbf":
        gamma = params.get("gamma", "scale")
        return make_rbf_svm(
            seed=seed,
            C=float(params["C"]),
            gamma=gamma,
            class_weight=_cw(params.get("class_weight")),
            calibrated=False,
        ).fit(X_train, y_train)
    raise ValueError(family)


def run_selection_stage(
    cfg: dict[str, Any],
    train_full: pd.DataFrame,
    val_fixed: pd.DataFrame,
    feat_cols: list[str],
    paths: dict[str, Path],
    *,
    skip_quantum: bool = False,
) -> pd.DataFrame:
    """Stage A: hyperparameter + quantum config selection on validation; test blocked."""
    grids = _classical_grids(cfg)
    seeds = list(cfg["experiment"]["selection_seeds"])
    train_size = int(cfg["sampling"]["train_sizes"][0])
    registry = SampleRegistry(paths["samples"])
    rows: list[dict[str, Any]] = []
    kernel_cache: dict[str, np.ndarray] = {}

    for seed in seeds:
        logger.info("selection seed=%s train_size=%s", seed, train_size)
        tr = sample_train_for_seed(train_full, train_size=train_size, seed=seed)
        registry.save_train(tr, seed=seed, train_size=train_size)
        y_tr = tr["target"].to_numpy()
        y_va = val_fixed["target"].to_numpy()

        repr_ = fit_paired_representation(tr, val_fixed, val_fixed, feat_cols, pca_components=2, seed=seed)
        save_preprocessors(repr_, paths["preprocessors"] / f"seed_{seed}_n{train_size}")

        # Group A — full 8D
        full_selections: list[tuple[str, dict[str, Any]]] = [
            (
                "logistic_full_8d",
                select_logistic(
                    repr_.X_train_full,
                    y_tr,
                    repr_.X_val_full,
                    y_va,
                    grids["logistic_C"],
                    grids["logistic_cw"],
                    seed=seed,
                ),
            ),
            (
                "svm_linear_full_8d",
                select_svm(
                    repr_.X_train_full,
                    y_tr,
                    repr_.X_val_full,
                    y_va,
                    kernel="linear",
                    C_grid=grids["svm_linear_C"],
                    class_weight_grid=grids["svm_linear_cw"],
                    seed=seed,
                ),
            ),
            (
                "svm_rbf_full_8d",
                select_svm(
                    repr_.X_train_full,
                    y_tr,
                    repr_.X_val_full,
                    y_va,
                    kernel="rbf",
                    C_grid=grids["svm_rbf_C"],
                    class_weight_grid=grids["svm_rbf_cw"],
                    gamma_grid=grids["svm_rbf_gamma"],
                    seed=seed,
                ),
            ),
        ]
        for name, sel in full_selections:
            frozen = freeze_selection(sel)
            frozen.update(
                {
                    "model_id": name,
                    "group": "full",
                    "seed": seed,
                    "train_size": train_size,
                    "angular_scaler": None,
                    "reps": None,
                }
            )
            rows.append(frozen)

        # Group B — PCA2 + angular scales
        for ang_name, ang_tag in ANGULAR_ALIASES.items():
            X_tr_a, X_va_a, _, _, _ = repr_.angular(ang_name, seed=seed)
            paired_selections: list[tuple[str, dict[str, Any]]] = [
                (
                    f"logistic_pca2_{ang_tag}",
                    select_logistic(
                        X_tr_a,
                        y_tr,
                        X_va_a,
                        y_va,
                        grids["logistic_C"],
                        grids["logistic_cw"],
                        seed=seed,
                    ),
                ),
                (
                    f"svm_linear_pca2_{ang_tag}",
                    select_svm(
                        X_tr_a,
                        y_tr,
                        X_va_a,
                        y_va,
                        kernel="linear",
                        C_grid=grids["svm_linear_C"],
                        class_weight_grid=grids["svm_linear_cw"],
                        seed=seed,
                    ),
                ),
                (
                    f"svm_rbf_pca2_{ang_tag}",
                    select_svm(
                        X_tr_a,
                        y_tr,
                        X_va_a,
                        y_va,
                        kernel="rbf",
                        C_grid=grids["svm_rbf_C"],
                        class_weight_grid=grids["svm_rbf_cw"],
                        gamma_grid=grids["svm_rbf_gamma"],
                        seed=seed,
                    ),
                ),
            ]
            for name, sel in paired_selections:
                frozen = freeze_selection(sel)
                frozen.update(
                    {
                        "model_id": name,
                        "group": "paired",
                        "seed": seed,
                        "train_size": train_size,
                        "angular_scaler": ang_name,
                        "reps": None,
                    }
                )
                rows.append(frozen)

            if skip_quantum:
                continue

            for qcfg in QUANTUM_CONFIGS:
                if qcfg["angular_scaler"] != ang_name:
                    continue
                qid = qcfg["id"]
                cache_key = f"{qid}_seed{seed}_n{train_size}"
                t0 = time.perf_counter()
                sel = select_qsvm(
                    X_tr_a,
                    y_tr,
                    X_va_a,
                    y_va,
                    C_grid=grids["qsvm_C"],
                    class_weight_grid=grids["qsvm_cw"],
                    reps=int(qcfg["reps"]),
                    feature_map=cfg["quantum"]["feature_map"],
                    entanglement=cfg["quantum"]["entanglement"],
                    seed=seed,
                    kernel_cache=kernel_cache,
                    cache_key=cache_key,
                )
                kernel_s = time.perf_counter() - t0
                np.save(paths["kernels"] / f"{cache_key}_K_train.npy", sel["K_train"])
                np.save(paths["kernels"] / f"{cache_key}_K_val.npy", sel["K_val"])
                frozen = freeze_selection(sel)
                frozen.update(
                    {
                        "model_id": f"qsvm_pca2_{ang_tag}_reps{qcfg['reps']}",
                        "quantum_id": qid,
                        "farooq_style": qcfg["farooq_style"],
                        "group": "paired",
                        "seed": seed,
                        "train_size": train_size,
                        "angular_scaler": ang_name,
                        "reps": qcfg["reps"],
                        "kernel_seconds": kernel_s,
                    }
                )
                rows.append(frozen)

    selection_df = pd.DataFrame(rows)
    selection_df.to_csv(paths["base"] / "model_selection.csv", index=False)

    # Aggregate: pick best hyperparams per model_id by mean val_auprc across seeds
    def _mode_or_first(s: pd.Series) -> Any:
        s = s.dropna()
        if s.empty:
            return None
        modes = s.mode()
        return modes.iloc[0] if len(modes) else s.iloc[0]

    agg_spec: dict[str, Any] = {
        "val_auprc": ("val_auprc", "mean"),
        "val_recall_extreme": ("val_recall_extreme", "mean"),
        "C": ("C", _mode_or_first),
        "class_weight": ("class_weight", _mode_or_first),
        "gamma": ("gamma", _mode_or_first),
        "angular_scaler": ("angular_scaler", "first"),
        "reps": ("reps", "first"),
        "group": ("group", "first"),
        "threshold": ("threshold", "median"),
    }
    if "quantum_id" in selection_df.columns:
        agg_spec["quantum_id"] = ("quantum_id", "first")
    if "farooq_style" in selection_df.columns:
        agg_spec["farooq_style"] = ("farooq_style", "first")
    agg = selection_df.groupby("model_id", as_index=False).agg(**agg_spec)
    # Prefer quantum config by mean val AUPRC among QSVM models
    q_mask = agg["model_id"].astype(str).str.startswith("qsvm_")
    models_records = json.loads(agg.to_json(orient="records"))
    if q_mask.any():
        best_q = agg.loc[q_mask].sort_values(
            ["val_auprc", "val_recall_extreme"], ascending=False
        ).iloc[0]
        frozen_cfg = {
            "selection_split": "validation",
            "best_quantum_model_id": best_q["model_id"],
            "best_quantum_id": None if pd.isna(best_q.get("quantum_id")) else best_q.get("quantum_id"),
            "best_quantum_angular_scaler": best_q["angular_scaler"],
            "best_quantum_reps": int(best_q["reps"]) if pd.notna(best_q["reps"]) else None,
            "models": models_records,
        }
    else:
        frozen_cfg = {"selection_split": "validation", "models": models_records}

    (paths["base"] / "selection_frozen.json").write_text(
        json.dumps(frozen_cfg, indent=2, allow_nan=False), encoding="utf-8"
    )
    return selection_df


def _params_for_model(frozen: dict[str, Any], model_id: str) -> dict[str, Any]:
    for m in frozen["models"]:
        if m["model_id"] == model_id:
            return m
    raise KeyError(model_id)


def run_final_stage(
    cfg: dict[str, Any],
    train_full: pd.DataFrame,
    val_fixed: pd.DataFrame,
    test_fixed: pd.DataFrame,
    feat_cols: list[str],
    paths: dict[str, Path],
    *,
    skip_quantum: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stage B: frozen configs, 10 seeds, train sizes 200/500, test unlocked."""
    frozen_path = paths["base"] / "selection_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    seeds = list(cfg["experiment"]["final_seeds"])
    train_sizes = list(cfg["sampling"]["train_sizes"])
    registry = SampleRegistry(paths["samples"])
    metric_rows: list[dict[str, Any]] = []
    pred_store: dict[str, dict[str, Any]] = {}

    # Models to evaluate in final: full refs + paired for selected angular + best Q
    best_ang = frozen.get("best_quantum_angular_scaler") or "minmax_0_1"
    best_reps = frozen.get("best_quantum_reps") or 2
    ang_tag = ANGULAR_ALIASES[best_ang]
    final_model_ids = [
        "logistic_full_8d",
        "svm_linear_full_8d",
        "svm_rbf_full_8d",
        f"logistic_pca2_{ang_tag}",
        f"svm_linear_pca2_{ang_tag}",
        f"svm_rbf_pca2_{ang_tag}",
    ]
    q_model_id = f"qsvm_pca2_{ang_tag}_reps{best_reps}"
    if not skip_quantum:
        final_model_ids.append(q_model_id)

    for train_size in train_sizes:
        for seed in seeds:
            logger.info("final seed=%s train_size=%s", seed, train_size)
            tr = sample_train_for_seed(train_full, train_size=train_size, seed=seed)
            registry.save_train(tr, seed=seed, train_size=train_size)
            y_tr = tr["target"].to_numpy()
            y_va = val_fixed["target"].to_numpy()
            y_te = test_fixed["target"].to_numpy()
            ts_te = test_fixed["timestamp"].to_numpy()
            idx_te = (
                test_fixed["original_index"].to_numpy()
                if "original_index" in test_fixed.columns
                else None
            )

            repr_ = fit_paired_representation(
                tr, val_fixed, test_fixed, feat_cols, pca_components=2, seed=seed
            )
            X_tr_a, X_va_a, X_te_a, _, _ = repr_.angular(best_ang, seed=seed)

            # Raw embeddings for this seed / train size (paired space = quantum inputs)
            np.savez(
                paths["embeddings"] / f"seed{seed}_n{train_size}.npz",
                X_train_full=repr_.X_train_full,
                X_val_full=repr_.X_val_full,
                X_test_full=repr_.X_test_full,
                X_train_pca=repr_.X_train_pca,
                X_val_pca=repr_.X_val_pca,
                X_test_pca=repr_.X_test_pca,
                X_train_angular=X_tr_a,
                X_val_angular=X_va_a,
                X_test_angular=X_te_a,
                y_train=y_tr,
                y_val=y_va,
                y_test=y_te,
                angular_scaler=np.asarray(best_ang),
                feature_cols=np.asarray(feat_cols),
            )

            for mid in final_model_ids:
                params = _params_for_model(frozen, mid)
                thr = float(params["threshold"])

                if mid.endswith("_full_8d"):
                    family = mid.replace("_full_8d", "")
                    model = _fit_frozen_classical(family, params, repr_.X_train_full, y_tr, seed)
                    scores = ranking_scores(model, repr_.X_test_full)
                    preds = apply_threshold(scores, thr)
                    row = _eval_row(
                        mid, "classical_full", y_te, scores, thr,
                        seed=seed, train_size=train_size,
                        extra={"angular_scaler": None, "reps": None},
                    )
                    metric_rows.append(row)
                    key = f"{mid}|{seed}|{train_size}"
                    pred_store[key] = {
                        "scores": scores,
                        "preds": preds,
                        "y": y_te,
                        "timestamps": ts_te,
                    }
                    _persist_prediction(
                        paths["predictions"],
                        key,
                        scores=scores,
                        preds=preds,
                        y=y_te,
                        timestamps=ts_te,
                        original_index=idx_te,
                        threshold=thr,
                        model_id=mid,
                        seed=seed,
                        train_size=train_size,
                    )
                    continue

                if mid.startswith("qsvm_"):
                    qk = make_fidelity_kernel(
                        n_qubits=X_tr_a.shape[1],
                        reps=int(best_reps),
                        entanglement=cfg["quantum"]["entanglement"],
                        feature_map_name=cfg["quantum"]["feature_map"],
                    )
                    t0 = time.perf_counter()
                    K_tr = qk.evaluate(x_vec=X_tr_a)
                    K_te = qk.evaluate(x_vec=X_te_a, y_vec=X_tr_a)
                    kernel_s = time.perf_counter() - t0
                    np.save(
                        paths["kernels"] / f"final_{mid}_seed{seed}_n{train_size}_K_train.npy",
                        K_tr,
                    )
                    np.save(
                        paths["kernels"] / f"final_{mid}_seed{seed}_n{train_size}_K_test.npy",
                        K_te,
                    )
                    model = make_precomputed_svm(
                        seed=seed,
                        C=float(params["C"]),
                        class_weight=_cw(params.get("class_weight")),
                    )
                    model.fit(K_tr, y_tr)
                    scores = ranking_scores(model, K_te)
                    preds = apply_threshold(scores, thr)
                    row = _eval_row(
                        mid, "qsvm", y_te, scores, thr,
                        seed=seed, train_size=train_size,
                        extra={
                            "angular_scaler": best_ang,
                            "reps": best_reps,
                            "kernel_seconds": kernel_s,
                            "farooq_style": best_ang == "minmax_0_1" and best_reps == 2,
                        },
                    )
                    metric_rows.append(row)
                    key = f"{mid}|{seed}|{train_size}"
                    pred_store[key] = {
                        "scores": scores,
                        "preds": preds,
                        "y": y_te,
                        "timestamps": ts_te,
                    }
                    _persist_prediction(
                        paths["predictions"],
                        key,
                        scores=scores,
                        preds=preds,
                        y=y_te,
                        timestamps=ts_te,
                        original_index=idx_te,
                        threshold=thr,
                        model_id=mid,
                        seed=seed,
                        train_size=train_size,
                    )
                    continue

                # paired classical on angular PCA space
                family = mid.split("_pca2_")[0]
                model = _fit_frozen_classical(family, params, X_tr_a, y_tr, seed)
                scores = ranking_scores(model, X_te_a)
                preds = apply_threshold(scores, thr)
                row = _eval_row(
                    mid, "classical_paired", y_te, scores, thr,
                    seed=seed, train_size=train_size,
                    extra={"angular_scaler": best_ang, "reps": None},
                )
                metric_rows.append(row)
                key = f"{mid}|{seed}|{train_size}"
                pred_store[key] = {
                    "scores": scores,
                    "preds": preds,
                    "y": y_te,
                    "timestamps": ts_te,
                }
                _persist_prediction(
                    paths["predictions"],
                    key,
                    scores=scores,
                    preds=preds,
                    y=y_te,
                    timestamps=ts_te,
                    original_index=idx_te,
                    threshold=thr,
                    model_id=mid,
                    seed=seed,
                    train_size=train_size,
                )

            del y_va, X_va_a

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(paths["base"] / "final_metrics_by_seed.csv", index=False)

    summary = (
        metrics_df.groupby(["model", "train_size", "family"], as_index=False)
        .agg(
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            auroc_mean=("auroc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            precision_mean=("precision_extreme", "mean"),
            recall_mean=("recall_extreme", "mean"),
            f1_mean=("f1_extreme", "mean"),
            f2_mean=("f2_extreme", "mean"),
            mcc_mean=("mcc", "mean"),
            false_alert_rate_mean=("false_alert_rate", "mean"),
            missed_extreme_rate_mean=("missed_extreme_rate", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(paths["base"] / "final_metrics_summary.csv", index=False)

    # Block bootstrap pairwise on pooled last train_size or both
    boot_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"selection": frozen, "classifications": []}

    for train_size in train_sizes:
        # Pool predictions across seeds for a fixed train_size
        def _pool(
            model_id: str, n_train: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ys, scores, preds, tss = [], [], [], []
            for seed in seeds:
                key = f"{model_id}|{seed}|{n_train}"
                if key not in pred_store:
                    continue
                p = pred_store[key]
                ys.append(p["y"])
                scores.append(p["scores"])
                preds.append(p["preds"])
                tss.append(p["timestamps"])
            if not ys:
                raise KeyError(model_id)
            return (
                np.concatenate(ys),
                np.concatenate(scores),
                np.concatenate(preds),
                np.concatenate(tss),
            )

        q_id = q_model_id if not skip_quantum else None
        if q_id and any(k.startswith(f"{q_id}|") and k.endswith(f"|{train_size}") for k in pred_store):
            y_q, s_q, p_q, ts = _pool(q_id, train_size)
            comparisons = [
                (f"svm_rbf_pca2_{ang_tag}", "QSVM − SVM-RBF paired"),
                (f"logistic_pca2_{ang_tag}", "QSVM − logistic paired"),
                ("svm_rbf_full_8d", "QSVM − best classic full proxy (svm_rbf_full)"),
            ]
            for other_id, label in comparisons:
                if not any(k.startswith(f"{other_id}|") and k.endswith(f"|{train_size}") for k in pred_store):
                    continue
                _, s_o, p_o, _ = _pool(other_id, train_size)
                # Align lengths (same test set per seed so concat order matches)
                boot = block_bootstrap_deltas(
                    y_q,
                    s_q,
                    s_o,
                    pred_a=p_q,
                    pred_b=p_o,
                    timestamps=ts,
                    n_boot=int(cfg["evaluation"]["bootstrap_iterations"]),
                    block_size_hours=int(cfg["evaluation"]["bootstrap_block_hours"]),
                    seed=42,
                )
                seed_deltas = []
                for seed in seeds:
                    kq = f"{q_id}|{seed}|{train_size}"
                    ko = f"{other_id}|{seed}|{train_size}"
                    if kq in pred_store and ko in pred_store:
                        seed_deltas.append(
                            float(
                                average_precision_score(pred_store[kq]["y"], pred_store[kq]["scores"])
                                - average_precision_score(pred_store[ko]["y"], pred_store[ko]["scores"])
                            )
                        )
                summary_row = summarize_pairwise(
                    f"{label} [n={train_size}]",
                    boot,
                    seed_deltas,
                    beats_full_without_threshold_fix=False,
                )
                summary_row["train_size"] = train_size
                summary_row["model_a"] = q_id
                summary_row["model_b"] = other_id
                boot_rows.append(summary_row)
                report["classifications"].append(summary_row)

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(paths["base"] / "pairwise_bootstrap.csv", index=False)

    return metrics_df, boot_df, report


def write_final_report(
    paths: dict[str, Path],
    tmeta: dict[str, Any],
    split_meta: dict[str, Any],
    report: dict[str, Any],
    summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Farooq-style fair benchmark — relatório final",
        "",
        "## Alvo",
        f"- percentil: {tmeta.get('percentile')}",
        f"- limiar: {tmeta.get('threshold')}",
        f"- fonte: {tmeta.get('source')}",
        "",
        "## Split e purga",
        f"- train rows após purga: {split_meta.get('train_rows_after')}",
        f"- validation rows após purga: {split_meta.get('validation_rows_after')}",
        f"- test rows: {split_meta.get('test_rows')}",
        "",
        "## Resumo de métricas (teste)",
        "",
        summary_df.to_string(index=False),
        "",
        "## Comparações pareadas (block bootstrap)",
        "",
    ]
    for c in report.get("classifications", []):
        lines.append(
            f"- **{c.get('comparison')}**: label=`{c.get('label')}` "
            f"ΔAUPRC={c.get('delta_auprc_mean'):.4f} "
            f"IC=[{c.get('delta_auprc_ci_low'):.4f}, {c.get('delta_auprc_ci_high'):.4f}]"
        )
        lines.append(f"  - {c.get('conclusion')}")
    lines.append("")
    lines.append(
        "Nota: não usar o termo 'vantagem quântica' quando o intervalo de confiança inclui zero."
    )
    (paths["base"] / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def run_fair_benchmark(
    config_path: str | Path,
    stage: str = "all",
    skip_quantum: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Entry point for selection / final / all stages."""
    root = project_root()
    cfg = load_config(root / config_path if not Path(config_path).is_absolute() else config_path)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    paths = _artifact_dirs(cfg)
    (paths["base"] / "run_config.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "stage": stage,
                "skip_quantum": skip_quantum,
                "overrides": overrides or {},
                "sampling": cfg.get("sampling"),
                "experiment_seeds": {
                    "selection": cfg["experiment"]["selection_seeds"],
                    "final": cfg["experiment"]["final_seeds"],
                },
                "evaluation": cfg.get("evaluation"),
                "artifacts_dir": cfg["paths"]["artifacts_dir"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    train, val, test, feat_cols, tmeta, split_meta = prepare_fair_frames(cfg)
    save_target_metadata(tmeta, paths["base"] / "target_metadata.json")
    (paths["base"] / "split_metadata.json").write_text(
        json.dumps(split_meta, indent=2, default=str), encoding="utf-8"
    )
    (paths["base"] / "data_audit.json").write_text(
        json.dumps(
            {
                "station": cfg["data"]["station"],
                "features": feat_cols,
                "n_train": len(train),
                "n_validation": len(val),
                "n_test": len(test),
                "threshold": tmeta["threshold"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    va_fixed, te_fixed, sample_meta = build_fixed_evaluation_sets(
        val,
        test,
        validation_size=int(cfg["sampling"]["validation_size"]),
        test_size=int(cfg["sampling"]["test_size"]),
        evaluation_seed=int(cfg["sampling"]["evaluation_seed"]),
    )
    registry = SampleRegistry(paths["samples"])
    registry.save_fixed(va_fixed, te_fixed)
    (paths["base"] / "sample_meta.json").write_text(
        json.dumps({k: v for k, v in sample_meta.items() if k not in {"validation_indices", "test_indices"}}, indent=2),
        encoding="utf-8",
    )

    result: dict[str, Any] = {"target_metadata": tmeta, "split_metadata": split_meta}

    if stage in {"selection", "all"}:
        sel = run_selection_stage(
            cfg, train, va_fixed, feat_cols, paths, skip_quantum=skip_quantum
        )
        result["selection"] = sel

    if stage in {"final", "all"}:
        if not (paths["base"] / "selection_frozen.json").exists():
            raise FileNotFoundError("selection_frozen.json missing; run selection stage first")
        metrics_df, boot_df, report = run_final_stage(
            cfg, train, va_fixed, te_fixed, feat_cols, paths, skip_quantum=skip_quantum
        )
        summary = pd.read_csv(paths["base"] / "final_metrics_summary.csv")
        write_final_report(paths, tmeta, split_meta, report, summary)
        plot_paths = generate_fair_benchmark_plots(paths["base"])
        result["metrics"] = metrics_df
        result["bootstrap"] = boot_df
        result["report"] = report
        result["plots"] = [str(p) for p in plot_paths]

    return result
