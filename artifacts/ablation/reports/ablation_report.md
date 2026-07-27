# Relatório de ablação — escala angular e kernels quânticos

## Contexto

Tarefa temporal sem vazamento: prever se PM2.5 em t+24h excede o P90 do **treino**.
Esta iteração testa se escala angular, dimensionalidade e profundidade do feature map
tornam o kernel quântico mais informativo — **sem** reproduzir a classificação circular
de AQI de Farooq et al. (2024).

## Seleção (somente validação)

```json
{
  "best_qsvm_id": "Q06",
  "best_qsvm_val_auprc": 0.5326708842046494,
  "best_classical_paired": "svm_rbf",
  "best_classical_paired_pca": 4,
  "best_classical_paired_val_auprc": 0.45108421844262653,
  "control_id": "Q10",
  "seed": 42,
  "note": "Selection used validation only; test not consulted."
}
```

## Ranking de validação

```
 id  pca_components     angular_scaler  feature_map  reps  average_precision  recall_extreme  alignment  effective_rank   delta_k  kernel_seconds
Q06               2        minmax_0_pi ZZFeatureMap     2           0.532671        0.684211   0.547529       15.086844  0.056667     1995.731945
Q03               2        minmax_0_pi ZZFeatureMap     1           0.469809        0.684211   0.518967       14.001063  0.063823     1706.650990
Q07               2        minmax_0_pi  ZFeatureMap     1           0.427743        0.789474   0.555524        9.357083  0.095428     1732.650453
Q02               2         minmax_0_1 ZZFeatureMap     1           0.374816        0.789474   0.543587        6.481034  0.020621     1746.425854
Q08               4        minmax_0_pi ZZFeatureMap     1           0.363672        0.315789   0.372944       91.977609  0.013573     2261.450754
Q04               2 minmax_minus_pi_pi ZZFeatureMap     1           0.190020        0.368421   0.479672       19.440668  0.001626     1721.771936
Q09               4        minmax_0_pi ZZFeatureMap     2           0.172648        0.315789   0.416657      109.725561  0.009131     2642.232287
Q10               4               none ZZFeatureMap     2           0.158226        0.210526   0.407549      149.867411 -0.002949     2622.379708
Q05               2      quantile_0_pi ZZFeatureMap     1           0.157603        0.631579   0.478417       17.893012  0.002615     1780.990110
Q01               2               none ZZFeatureMap     1           0.149000        0.368421   0.480507       20.142198  0.001680     1731.360397
```

## Teste final (uma vez)

```
   model  average_precision  balanced_accuracy  precision_extreme  recall_extreme  f1_extreme       mcc    auroc  training_seconds  inference_seconds                     id  pca_components angular_scaler  feature_map  reps  alignment  effective_rank   delta_k split           family
qsvm_Q06           0.331725           0.755121           0.333333        0.666667    0.444444  0.384025 0.758446          0.007200           0.002789                    Q06               2    minmax_0_pi ZZFeatureMap   2.0   0.547529       15.086844  0.056667  test             qsvm
qsvm_Q10           0.097081           0.462490           0.071429        0.142857    0.095238 -0.056462 0.474062          0.007694           0.002906                    Q10               4           none ZZFeatureMap   2.0   0.407549      149.867411 -0.002949  test             qsvm
   dummy           0.105000           0.500000           0.000000        0.000000    0.000000  0.000000 0.500000          0.000565           0.001274   classical_dummy_pca4               4           none          NaN   NaN        NaN             NaN       NaN  test classical_paired
 svm_rbf           0.378001           0.500000           0.000000        0.000000    0.000000  0.000000 0.807396          0.044267           0.027887 classical_svm_rbf_pca4               4           none          NaN   NaN        NaN             NaN       NaN  test classical_paired
```

## Classificação automática

- Rótulo: `BETTER_QUANTUM_REPRESENTATION_WITHOUT_ADVANTAGE`
- Conclusão: A adaptação angular tornou o kernel mais informativo que o controle/Dummy, mas o ganho não foi suficiente para superar kernels clássicos equivalentes.

### Bootstrap da diferença (QSVM − clássico pareado)

```json
{
  "delta_mean": -0.04571140705530314,
  "delta_std": 0.08428816064241093,
  "ci_low": -0.20985578645760927,
  "ci_high": 0.11860546635085381,
  "n_boot_effective": 1000
}
```

## Comparação metodológica com Farooq et al.

| Aspecto | Farooq et al. | Esta PoC |
|---|---|---|
| Alvo | AQI contemporâneo (faixas) | Extremo PM2.5 em t+24h |
| Features | PM2.5 + temperatura (~2D) | lags/rolling/meteo → PCA |
| Escala | MinMax [0,1] | ablação none/[0,1]/[0,π]/[-π,π]/quantile |
| Métrica | accuracy | **AUPRC** |
| Split | 80/20 (não temporal explícito) | temporal 60/20/20 |

## Limitações

- Uma seed principal na execução padrão.
- Simulador ideal (sem ruído de hardware).
- Subamostra finita para viabilidade do kernel.
- Melhoria estrutural do kernel ≠ vantagem computacional quântica.
