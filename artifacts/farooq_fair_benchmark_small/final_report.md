# Farooq-style fair benchmark — relatório final

## Alvo
- percentil: 0.9
- limiar: 185.0
- fonte: purged_training_partition

## Split e purga
- train rows após purga: 20523
- validation rows após purga: 6825
- test rows: 6849

## Resumo de métricas (teste)

              model  train_size           family  average_precision_mean  average_precision_std  auroc_mean  balanced_accuracy_mean  precision_mean  recall_mean  f1_mean  f2_mean  mcc_mean  false_alert_rate_mean  missed_extreme_rate_mean  n_seeds
   logistic_full_8d          80   classical_full                0.554861                    NaN    0.909722                0.708333        0.400000         0.50 0.444444 0.476190  0.377964                  0.075                      0.50        1
  logistic_pca2_0_1          80 classical_paired                0.353409                    NaN    0.854167                0.777778        0.300000         0.75 0.428571 0.576923  0.384900                  0.175                      0.25        1
qsvm_pca2_0_1_reps1          80             qsvm                0.568254                    NaN    0.916667                0.930556        0.444444         1.00 0.615385 0.800000  0.618640                  0.125                      0.00        1
 svm_linear_full_8d          80   classical_full                0.499242                    NaN    0.875000                0.583333        0.250000         0.25 0.250000 0.250000  0.166667                  0.075                      0.75        1
svm_linear_pca2_0_1          80 classical_paired                0.304016                    NaN    0.791667                0.555556        0.166667         0.25 0.200000 0.227273  0.093352                  0.125                      0.75        1
    svm_rbf_full_8d          80   classical_full                0.328526                    NaN    0.819444                0.680556        0.285714         0.50 0.363636 0.434783  0.285112                  0.125                      0.50        1
   svm_rbf_pca2_0_1          80 classical_paired                0.377083                    NaN    0.888889                0.486111        0.000000         0.00 0.000000 0.000000 -0.053376                  0.025                      1.00        1

## Comparações pareadas (block bootstrap)

- **QSVM − SVM-RBF paired [n=80]**: label=`CANDIDATE_QUANTUM_GAIN` ΔAUPRC=0.0940 IC=[0.0095, 0.1917]
  - QSVM superou o clássico pareado em AUPRC com IC da diferença acima de zero, F2 não inferior e maioria das sementes positivas. Resultado preliminar — não interpretar como vantagem computacional quântica geral.
- **QSVM − logistic paired [n=80]**: label=`NO_CLEAR_QUANTUM_GAIN` ΔAUPRC=0.0831 IC=[0.0000, 0.2083]
  - Empate ou diferença não estável frente ao melhor modelo pareado (intervalo de confiança da ΔAUPRC inclui zero ou critérios de ganho não atendidos).
- **QSVM − best classic full proxy (svm_rbf_full) [n=80]**: label=`CANDIDATE_QUANTUM_GAIN` ΔAUPRC=0.1166 IC=[0.0027, 0.2399]
  - QSVM superou o clássico pareado em AUPRC com IC da diferença acima de zero, F2 não inferior e maioria das sementes positivas. Resultado preliminar — não interpretar como vantagem computacional quântica geral.

Nota: não usar o termo 'vantagem quântica' quando o intervalo de confiança inclui zero.