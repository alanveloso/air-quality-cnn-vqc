"""Leakage-free quantum angular ablation experiment (Q01–Q10)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from qml_air_quality.config import load_config, project_root
from qml_air_quality.data import download_dataset, load_station_csv
from qml_air_quality.evaluation.kernel_diagnostics import diagnose_kernel
from qml_air_quality.features import (
    add_calendar_features,
    add_future_target,
    add_lag_and_rolling,
    apply_causal_ffill,
    feature_columns,
    impute_with_train_medians,
    make_binary_target,
)
from qml_air_quality.metrics import binary_metrics
from qml_air_quality.models import (
    make_dummy,
    make_fidelity_kernel,
    make_knn,
    make_linear_svm,
    make_logistic,
    make_poly_svm,
    make_rbf_svm,
    predict_proba_positive,
)
from qml_air_quality.preprocess import fit_transform_pca
from qml_air_quality.preprocessing.angular_scaling import (
    make_angular_scaler,
    save_angular_scaler,
)
from qml_air_quality.split import stratified_subsample, temporal_split

logger = logging.getLogger(__name__)


@dataclass
class PartitionData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    paths = {
        "ablation": root / cfg["paths"]["ablation_dir"],
        "kernels": root / cfg["paths"]["kernels_dir"],
        "plots": root / cfg["paths"]["plots_dir"],
        "reports": root / cfg["paths"]["reports_dir"],
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def prepare_temporal_frames(poc_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Reproduce the leakage-free prepare pipeline; return train/val/test + features + target meta."""
    root = project_root()
    csv_path = download_dataset(
        dataset_id=poc_cfg["data"]["dataset_id"],
        station=poc_cfg["data"]["station"],
        raw_dir=root / poc_cfg["data"]["raw_dir"],
    )
    df = load_station_csv(csv_path)
    raw_cols = poc_cfg["features"]["raw_columns"]
    df = apply_causal_ffill(df, raw_cols, limit_hours=poc_cfg["features"]["ffill_limit_hours"])
    df = add_lag_and_rolling(
        df,
        columns=raw_cols,
        lag_hours=poc_cfg["features"]["lag_hours"],
        rolling_windows=poc_cfg["features"]["rolling_windows"],
    )
    if poc_cfg["features"]["add_calendar_features"]:
        df = add_calendar_features(df)
    df = add_future_target(
        df,
        target_column=poc_cfg["data"]["target_column"],
        horizon_hours=poc_cfg["forecast"]["horizon_hours"],
    )
    train, val, test = temporal_split(
        df,
        train_fraction=poc_cfg["split"]["train_fraction"],
        validation_fraction=poc_cfg["split"]["validation_fraction"],
        test_fraction=poc_cfg["split"]["test_fraction"],
    )
    feat_cols = feature_columns(train)
    train, val, test = impute_with_train_medians(train, val, test, columns=feat_cols)
    train, val, test, tmeta = make_binary_target(
        train, val, test, percentile=poc_cfg["forecast"]["extreme_percentile"]
    )
    return train, val, test, feat_cols, tmeta


def build_pca_partitions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: list[str],
    n_components: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> PartitionData:
    X_tr, X_va, X_te, _, _ = fit_transform_pca(
        train, val, test, feat_cols, n_components=n_components
    )
    tr = train.copy()
    va = val.copy()
    te = test.copy()
    for i in range(n_components):
        tr[f"pc{i + 1}"] = X_tr[:, i]
        va[f"pc{i + 1}"] = X_va[:, i]
        te[f"pc{i + 1}"] = X_te[:, i]
    tr_s, va_s, te_s, _ = stratified_subsample(
        tr, va, te, train_size=train_size, validation_size=val_size, test_size=test_size, seed=seed
    )
    pcs = [f"pc{i + 1}" for i in range(n_components)]
    return PartitionData(
        X_train=tr_s[pcs].to_numpy(),
        X_val=va_s[pcs].to_numpy(),
        X_test=te_s[pcs].to_numpy(),
        y_train=tr_s["target"].to_numpy(),
        y_val=va_s["target"].to_numpy(),
        y_test=te_s["target"].to_numpy(),
    )


def apply_angular(part: PartitionData, scaler_name: str, seed: int) -> tuple[PartitionData, Any]:
    scaler = make_angular_scaler(scaler_name, seed=seed)
    X_train = scaler.fit_transform(part.X_train)
    X_val = scaler.transform(part.X_val)
    X_test = scaler.transform(part.X_test)
    return (
        PartitionData(X_train, X_val, X_test, part.y_train, part.y_val, part.y_test),
        scaler,
    )


def evaluate_precomputed_svm(
    K_train: np.ndarray,
    K_eval: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    model_name: str,
) -> dict[str, Any]:
    clf = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
    t0 = time.perf_counter()
    clf.fit(K_train, y_train)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_hat = clf.predict(K_eval)
    scores = clf.decision_function(K_eval)
    y_prob = 1.0 / (1.0 + np.exp(-scores))
    inf_s = time.perf_counter() - t1
    return binary_metrics(
        y_eval,
        y_hat,
        y_prob,
        model_name=model_name,
        training_seconds=train_s,
        inference_seconds=inf_s,
    )


def evaluate_classical_on_embedding(
    part: PartitionData,
    seed: int,
    split: str = "val",
) -> list[dict[str, Any]]:
    X_tr, y_tr = part.X_train, part.y_train
    if split == "val":
        X_ev, y_ev = part.X_val, part.y_val
    else:
        X_ev, y_ev = part.X_test, part.y_test

    models = {
        "dummy": make_dummy(seed=seed),
        "logistic": make_logistic(seed=seed),
        "svm_linear": make_linear_svm(seed=seed),
        "svm_rbf": make_rbf_svm(seed=seed),
        "svm_poly": make_poly_svm(seed=seed),
        "knn": make_knn(),
    }
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_hat = model.predict(X_ev)
        y_prob = predict_proba_positive(model, X_ev)
        inf_s = time.perf_counter() - t1
        rows.append(
            binary_metrics(
                y_ev,
                y_hat,
                y_prob,
                model_name=name,
                training_seconds=train_s,
                inference_seconds=inf_s,
            )
        )
    return rows


def _kernel_paths(kernels_dir: Path, cfg_id: str, seed: int) -> dict[str, Path]:
    return {
        "train": kernels_dir / f"{cfg_id}_seed{seed}_K_train.npy",
        "val": kernels_dir / f"{cfg_id}_seed{seed}_K_val.npy",
        "test": kernels_dir / f"{cfg_id}_seed{seed}_K_test.npy",
        "meta": kernels_dir / f"{cfg_id}_seed{seed}_meta.json",
    }


def compute_or_load_kernels(
    qkernel: Any,
    part: PartitionData,
    paths: dict[str, Path],
    cfg_id: str,
    seed: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if (
        not force
        and paths["train"].exists()
        and paths["val"].exists()
        and paths["test"].exists()
    ):
        logger.info("Loading cached kernels for %s", cfg_id)
        return (
            np.load(paths["train"]),
            np.load(paths["val"]),
            np.load(paths["test"]),
            float(json.loads(paths["meta"].read_text()).get("kernel_seconds", float("nan"))),
        )

    t0 = time.perf_counter()
    logger.info("Evaluating quantum kernel for %s …", cfg_id)
    K_train = qkernel.evaluate(x_vec=part.X_train)
    K_val = qkernel.evaluate(x_vec=part.X_val, y_vec=part.X_train)
    K_test = qkernel.evaluate(x_vec=part.X_test, y_vec=part.X_train)
    kernel_seconds = time.perf_counter() - t0
    np.save(paths["train"], K_train)
    np.save(paths["val"], K_val)
    np.save(paths["test"], K_test)
    paths["meta"].write_text(
        json.dumps({"id": cfg_id, "seed": seed, "kernel_seconds": kernel_seconds}, indent=2),
        encoding="utf-8",
    )
    return K_train, K_val, K_test, kernel_seconds


def rank_key(row: dict[str, Any]) -> tuple:
    """Selection order: val AUPRC, recall, alignment, lower kernel time."""
    return (
        -float(row.get("average_precision", 0.0)),
        -float(row.get("recall_extreme", 0.0)),
        -float(row.get("alignment", 0.0) or 0.0),
        float(row.get("kernel_seconds", 1e9) or 1e9),
    )


def run_ablation(
    ablation_config_path: str | Path | None = None,
    force_kernels: bool = False,
    skip_test: bool = False,
) -> dict[str, Any]:
    """Run full Q01–Q10 validation ranking; optionally final test of selected models."""
    root = project_root()
    abl_cfg = load_config(ablation_config_path or (root / "config" / "quantum_ablation.yaml"))
    poc_cfg = load_config(root / abl_cfg["data"].get("config_ref", "config/poc.yaml"))
    paths = _ensure_dirs(abl_cfg)
    seed = int(abl_cfg["project"]["seeds"][0])
    np.random.seed(seed)

    logger.info("Preparing temporal dataset…")
    train, val, test, feat_cols, tmeta = prepare_temporal_frames(poc_cfg)
    (paths["ablation"] / "target_metadata.json").write_text(json.dumps(tmeta, indent=2), encoding="utf-8")

    sampling = abl_cfg["sampling"]
    pca_cache: dict[int, PartitionData] = {}
    for n in abl_cfg["dimensionality"]["pca_components"]:
        pca_cache[int(n)] = build_pca_partitions(
            train,
            val,
            test,
            feat_cols,
            n_components=int(n),
            train_size=sampling["train_size"],
            val_size=sampling["validation_size"],
            test_size=sampling["test_size"],
            seed=seed,
        )

    val_rows: list[dict[str, Any]] = []
    classical_val_rows: list[dict[str, Any]] = []

    # Paired classical on each PCA dim (no angular) — for selection baseline
    for n, part in pca_cache.items():
        for row in evaluate_classical_on_embedding(part, seed=seed, split="val"):
            row = {**row, "pca_components": n, "angular_scaler": "none", "split": "val", "family": "classical_paired"}
            classical_val_rows.append(row)

    for qcfg in abl_cfg["quantum_configs"]:
        cfg_id = qcfg["id"]
        n = int(qcfg["pca_components"])
        part0 = pca_cache[n]
        part, scaler = apply_angular(part0, qcfg["angular_scaler"], seed=seed)
        save_angular_scaler(scaler, paths["ablation"] / f"{cfg_id}_angular_scaler.joblib")

        qkernel = make_fidelity_kernel(
            n_qubits=n,
            reps=int(qcfg["reps"]),
            entanglement=qcfg.get("entanglement", "linear"),
            feature_map_name=qcfg["feature_map"],
        )
        kpaths = _kernel_paths(paths["kernels"], cfg_id, seed)
        K_train, K_val, K_test, kernel_seconds = compute_or_load_kernels(
            qkernel, part, kpaths, cfg_id, seed, force=force_kernels
        )
        diag = diagnose_kernel(K_train, part.y_train)
        metrics = evaluate_precomputed_svm(
            K_train, K_val, part.y_train, part.y_val, seed=seed, model_name=f"qsvm_{cfg_id}"
        )
        row = {
            **metrics,
            "id": cfg_id,
            "pca_components": n,
            "angular_scaler": qcfg["angular_scaler"],
            "feature_map": qcfg["feature_map"],
            "reps": qcfg["reps"],
            "entanglement": qcfg.get("entanglement"),
            "kernel_seconds": kernel_seconds,
            "alignment": diag.get("alignment"),
            "effective_rank": diag.get("effective_rank"),
            "delta_k": diag.get("delta_k"),
            "cv_k": diag.get("cv_k"),
            "off_mean": diag.get("off_mean"),
            "off_std": diag.get("off_std"),
            "split": "val",
            "family": "qsvm",
            "diagnostics": diag,
        }
        val_rows.append(row)
        logger.info(
            "%s val AUPRC=%.4f align=%.4f ΔK=%.4f",
            cfg_id,
            row["average_precision"],
            row.get("alignment") or 0.0,
            row.get("delta_k") or 0.0,
        )

    val_df = pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in val_rows])
    val_df = val_df.sort_values(by=["average_precision", "recall_extreme", "alignment"], ascending=False)
    val_df.to_csv(paths["ablation"] / "validation_ranking.csv", index=False)
    pd.DataFrame(classical_val_rows).to_csv(paths["ablation"] / "classical_paired_validation.csv", index=False)

    # Persist full diagnostics
    diag_path = paths["ablation"] / "kernel_diagnostics_val.json"
    diag_path.write_text(
        json.dumps({r["id"]: r["diagnostics"] for r in val_rows}, indent=2, default=float),
        encoding="utf-8",
    )

    best_q = min(val_rows, key=rank_key)
    best_classical_paired = max(classical_val_rows, key=lambda r: r["average_precision"])
    selection = {
        "best_qsvm_id": best_q["id"],
        "best_qsvm_val_auprc": best_q["average_precision"],
        "best_classical_paired": best_classical_paired["model"],
        "best_classical_paired_pca": best_classical_paired["pca_components"],
        "best_classical_paired_val_auprc": best_classical_paired["average_precision"],
        "control_id": "Q10",
        "seed": seed,
        "note": "Selection used validation only; test not consulted.",
    }
    (paths["ablation"] / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "selection": selection,
        "validation": val_df,
        "paths": {k: str(v) for k, v in paths.items()},
    }

    if skip_test:
        return result

    # ---- Final test: original control, best QSVM, best paired classical, dummy ----
    test_rows: list[dict[str, Any]] = []
    final_ids = list(dict.fromkeys([best_q["id"], "Q10"]))  # unique order

    for cfg_id in final_ids:
        qcfg = next(c for c in abl_cfg["quantum_configs"] if c["id"] == cfg_id)
        n = int(qcfg["pca_components"])
        part, _ = apply_angular(pca_cache[n], qcfg["angular_scaler"], seed=seed)
        kpaths = _kernel_paths(paths["kernels"], cfg_id, seed)
        K_train = np.load(kpaths["train"])
        K_test = np.load(kpaths["test"])
        m = evaluate_precomputed_svm(
            K_train, K_test, part.y_train, part.y_test, seed=seed, model_name=f"qsvm_{cfg_id}"
        )
        diag = diagnose_kernel(K_train, part.y_train)
        test_rows.append(
            {
                **m,
                "id": cfg_id,
                "pca_components": n,
                "angular_scaler": qcfg["angular_scaler"],
                "feature_map": qcfg["feature_map"],
                "reps": qcfg["reps"],
                "alignment": diag.get("alignment"),
                "effective_rank": diag.get("effective_rank"),
                "delta_k": diag.get("delta_k"),
                "split": "test",
                "family": "qsvm",
            }
        )

    # Best paired classical on test
    n_best = int(best_classical_paired["pca_components"])
    part_c = pca_cache[n_best]
    for row in evaluate_classical_on_embedding(part_c, seed=seed, split="test"):
        if row["model"] in {best_classical_paired["model"], "dummy"}:
            test_rows.append(
                {
                    **row,
                    "id": f"classical_{row['model']}_pca{n_best}",
                    "pca_components": n_best,
                    "angular_scaler": "none",
                    "split": "test",
                    "family": "classical_paired",
                }
            )

    # Bootstrap delta: best QSVM vs best classical paired on test
    best_q_test = next(r for r in test_rows if r.get("id") == best_q["id"])
    best_c_test = next(
        r
        for r in test_rows
        if r.get("family") == "classical_paired" and r["model"] == best_classical_paired["model"]
    )
    # Need scores for bootstrap — recompute quickly
    qcfg_b = next(c for c in abl_cfg["quantum_configs"] if c["id"] == best_q["id"])
    n_b = int(qcfg_b["pca_components"])
    part_b, _ = apply_angular(pca_cache[n_b], qcfg_b["angular_scaler"], seed=seed)
    K_train_b = np.load(_kernel_paths(paths["kernels"], best_q["id"], seed)["train"])
    K_test_b = np.load(_kernel_paths(paths["kernels"], best_q["id"], seed)["test"])
    clf_q = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
    clf_q.fit(K_train_b, part_b.y_train)
    q_scores = 1.0 / (1.0 + np.exp(-clf_q.decision_function(K_test_b)))

    model_c = {
        "dummy": make_dummy(seed=seed),
        "logistic": make_logistic(seed=seed),
        "svm_linear": make_linear_svm(seed=seed),
        "svm_rbf": make_rbf_svm(seed=seed),
        "svm_poly": make_poly_svm(seed=seed),
        "knn": make_knn(),
    }[best_classical_paired["model"]]
    part_c = pca_cache[n_best]
    model_c.fit(part_c.X_train, part_c.y_train)
    c_scores = predict_proba_positive(model_c, part_c.X_test)
    assert c_scores is not None

    boot = bootstrap_auprc_delta(
        part_c.y_test,
        q_scores,
        c_scores,
        n_boot=int(abl_cfg["evaluation"]["bootstrap_iterations"]),
        seed=seed,
    )
    classification = classify_result(
        delta_mean=boot["delta_mean"],
        ci_low=boot["ci_low"],
        ci_high=boot["ci_high"],
        min_delta=float(abl_cfg["evaluation"]["min_auprc_delta"]),
        q_auprc=best_q_test["average_precision"],
        prevalence=float(np.mean(part_c.y_test)),
        alignment=float(best_q.get("alignment") or 0.0),
        delta_k=float(best_q.get("delta_k") or 0.0),
    )

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(paths["ablation"] / "test_final.csv", index=False)
    summary = {
        "selection": selection,
        "bootstrap": boot,
        "classification": classification,
        "best_qsvm_test": best_q_test,
        "best_classical_test": best_c_test,
    }
    (paths["ablation"] / "final_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    result["test"] = test_df
    result["summary"] = summary
    return result


def bootstrap_auprc_delta(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = average_precision_score(y_true[idx], scores_a[idx])
        b = average_precision_score(y_true[idx], scores_b[idx])
        deltas.append(a - b)
    deltas_arr = np.asarray(deltas)
    alpha = (1 - confidence) / 2
    return {
        "delta_mean": float(np.mean(deltas_arr)),
        "delta_std": float(np.std(deltas_arr)),
        "ci_low": float(np.quantile(deltas_arr, alpha)),
        "ci_high": float(np.quantile(deltas_arr, 1 - alpha)),
        "n_boot_effective": len(deltas_arr),
    }


def classify_result(
    delta_mean: float,
    ci_low: float,
    ci_high: float,
    min_delta: float,
    q_auprc: float,
    prevalence: float,
    alignment: float,
    delta_k: float,
) -> dict[str, str]:
    """Automatic interpretation labels from the revised plan (no 'quantum advantage' wording)."""
    near_dummy = abs(q_auprc - prevalence) < 0.03 and abs(alignment) < 0.05 and abs(delta_k) < 0.02
    if near_dummy:
        label = "QUANTUM_KERNEL_NOT_INFORMATIVE"
        conclusion = (
            "O baixo desempenho não decorreu apenas da ausência de normalização angular. "
            "O feature map testado não representa adequadamente a estrutura preditiva da tarefa."
        )
    elif ci_low > 0 and delta_mean >= min_delta:
        label = "CANDIDATE_PREDICTIVE_QUANTUM_GAIN"
        conclusion = (
            "QSVM superou o baseline clássico pareado no teste com intervalo bootstrap da "
            "diferença acima de zero. Resultado preliminar — não interpretar como vantagem "
            "computacional quântica geral."
        )
    elif delta_mean > 0 and ci_low <= 0:
        label = "NON_GENERALIZING_QUANTUM_GAIN"
        conclusion = (
            "Há sinal positivo pontual, mas o intervalo bootstrap da diferença inclui zero "
            "(ganho não generaliza de forma estável)."
        )
    elif q_auprc > prevalence + 0.03 and delta_mean <= 0:
        label = "BETTER_QUANTUM_REPRESENTATION_WITHOUT_ADVANTAGE"
        conclusion = (
            "A adaptação angular tornou o kernel mais informativo que o controle/Dummy, "
            "mas o ganho não foi suficiente para superar kernels clássicos equivalentes."
        )
    else:
        label = "QUANTUM_KERNEL_NOT_INFORMATIVE"
        conclusion = (
            "Nenhuma configuração selecionada produziu evidência preditiva estável "
            "frente aos baselines clássicos pareados."
        )
    return {"label": label, "conclusion": conclusion}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_ablation()
    print(json.dumps(out.get("summary", out.get("selection")), indent=2, default=float))
