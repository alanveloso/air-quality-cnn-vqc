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

              model  train_size           family  average_precision_mean  average_precision_std  f2_mean  recall_mean  false_alert_rate_mean  missed_extreme_rate_mean
   logistic_full_8d         200   classical_full                0.451919               0.074851 0.525298     0.825806               0.315000                  0.174194
   logistic_full_8d         500   classical_full                0.485713               0.035982 0.542992     0.935484               0.381000                  0.064516
  logistic_pca2_0_1         200 classical_paired                0.368648               0.026118 0.365566     1.000000               0.896667                  0.000000
  logistic_pca2_0_1         500 classical_paired                0.384838               0.004913 0.365566     1.000000               0.896667                  0.000000
 svm_linear_full_8d         200   classical_full                0.451117               0.057042 0.548339     0.900000               0.343333                  0.100000
 svm_linear_full_8d         500   classical_full                0.484139               0.040970 0.552862     0.932258               0.361333                  0.067742
svm_linear_pca2_0_1         200 classical_paired                0.368930               0.024239 0.555165     0.851613               0.292000                  0.148387
svm_linear_pca2_0_1         500 classical_paired                0.380726               0.006156 0.557682     0.938710               0.361333                  0.061290
    svm_rbf_full_8d         200   classical_full                0.368332               0.080249 0.522196     0.767742               0.271667                  0.232258
    svm_rbf_full_8d         500   classical_full                0.381896               0.070151 0.589422     0.835484               0.232667                  0.164516
   svm_rbf_pca2_0_1         200 classical_paired                0.389212               0.063819 0.559571     0.845161               0.284000                  0.154839
   svm_rbf_pca2_0_1         500 classical_paired                0.408028               0.054191 0.582276     0.867742               0.268333                  0.132258

## Comparações pareadas (block bootstrap)


Nota: não usar o termo 'vantagem quântica' quando o intervalo de confiança inclui zero.