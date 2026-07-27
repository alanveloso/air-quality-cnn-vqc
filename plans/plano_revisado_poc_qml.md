# Plano revisado de desenvolvimento da PoC QML

## 1. Objetivo da nova iteração

Avaliar se diferentes estratégias de codificação angular e configurações de circuitos quânticos conseguem produzir um kernel informativo para a previsão de episódios extremos de PM2.5 em (t+24), mantendo:

* divisão estritamente temporal;
* limiar calculado somente no treinamento;
* mesma amostra para modelos clássicos e quânticos;
* AUPRC como métrica principal;
* ausência de PM2.5 futuro nas entradas;
* comparação com baselines clássicos fortes.

A pergunta experimental passa a ser:

> A adaptação da escala, dimensionalidade e profundidade do feature map melhora o QSVM na tarefa temporal sem vazamento, ou seu desempenho permanece equivalente ao classificador aleatório?

## 2. Hipóteses

### H1 — Escala angular

A aplicação direta de componentes produzidos por `StandardScaler` e PCA pode gerar ângulos inadequados para o `ZZFeatureMap`, fazendo com que o kernel apresente concentração ou pouca discriminação.

Espera-se que o mapeamento para intervalos angulares produza uma matriz mais informativa.

### H2 — Dimensionalidade

Quatro componentes principais e quatro qubits podem introduzir complexidade desnecessária para a quantidade de amostras disponível.

Espera-se que duas dimensões produzam um kernel mais estável.

### H3 — Profundidade

O uso de `reps=2` pode tornar o circuito excessivamente expressivo e aumentar a concentração dos valores do kernel.

Espera-se que `reps=1` produza melhor separação entre amostras.

### H4 — Emaranhamento

O emaranhamento pode não ser responsável por qualquer eventual melhoria.

A comparação com circuitos sem emaranhamento determinará se o ganho decorre efetivamente das interações introduzidas pelo `ZZFeatureMap`.

## 3. Escopo mantido

### Tarefa

Prever:

$$
y_t =
\mathbb{1}
\left[
PM2.5_{t+24} \geq P90_{\text{treino}}
\right]
$$

### Dataset

* Beijing Multi-Site Air Quality;
* estação Aotizhongxin;
* registros horários;
* horizonte de 24 horas;
* percentil 90 calculado exclusivamente no treinamento.

### Métrica principal

```text
Average Precision — AUPRC
```

### Métricas secundárias

* recall da classe extrema;
* precision da classe extrema;
* F1 da classe extrema;
* balanced accuracy;
* MCC;
* AUROC;
* Brier Score;
* tempo de construção do kernel;
* tempo de treinamento;
* tempo de inferência.

## 4. Correções no pipeline quântico

### 4.1 Separar o pipeline clássico do pipeline quântico

O pipeline clássico continuará usando:

```text
imputação
→ StandardScaler
→ PCA
→ classificador clássico
```

O pipeline quântico deverá usar:

```text
imputação
→ StandardScaler
→ PCA
→ escalonamento angular
→ kernel quântico
→ SVM
```

O escalonamento angular deverá ser ajustado somente no treinamento.

### 4.2 Estratégias de escalonamento angular

Implementar quatro estratégias.

#### Estratégia A — MinMax para ([0,1])

```python
MinMaxScaler(feature_range=(0.0, 1.0))
```

Essa configuração aproxima o pré-processamento utilizado no artigo.

#### Estratégia B — MinMax para ([0,π])

```python
MinMaxScaler(feature_range=(0.0, np.pi))
```

Essa será a configuração principal da nova PoC.

#### Estratégia C — MinMax para ([-π,π])

```python
MinMaxScaler(feature_range=(-np.pi, np.pi))
```

Permite verificar se a centralização angular melhora a separação.

#### Estratégia D — Quantile Transformer para ângulos

Aplicar:

```python
QuantileTransformer(
    output_distribution="uniform",
    random_state=seed
)
```

e converter o resultado para:

$$
x_{\text{angular}} = \pi \, x_{\text{uniforme}}
$$

Essa estratégia reduz a influência de valores extremos sobre o intervalo angular.

## 5. Matriz experimental mínima

Executar as seguintes combinações:

| ID  | PCs/qubits | Escala             | Feature map | Reps | Emaranhamento |
| --- | ---------: | ------------------ | ----------- | ---: | ------------- |
| Q01 |          2 | sem escala angular | ZZ          |    1 | linear        |
| Q02 |          2 | ([0,1])            | ZZ          |    1 | linear        |
| Q03 |          2 | ([0,π])            | ZZ          |    1 | linear        |
| Q04 |          2 | ([-π,π])           | ZZ          |    1 | linear        |
| Q05 |          2 | quantile ([0,π])   | ZZ          |    1 | linear        |
| Q06 |          2 | ([0,π])            | ZZ          |    2 | linear        |
| Q07 |          2 | ([0,π])            | ZFeatureMap |    1 | nenhum        |
| Q08 |          4 | ([0,π])            | ZZ          |    1 | linear        |
| Q09 |          4 | ([0,π])            | ZZ          |    2 | linear        |
| Q10 |          4 | sem escala angular | ZZ          |    2 | linear        |

O experimento Q10 representa a configuração original da PoC e deverá ser usado como controle.

A configuração Q02 aproxima o pré-processamento do artigo, sem reproduzir sua tarefa circular.

### Detalhamento das configurações

* **Q01**: controle de dimensionalidade baixa sem escala angular; isola o efeito de reduzir de 4 para 2 PCs mantendo o pipeline original após PCA.
* **Q02**: aproximação do pré-processamento de Farooq et al. (intervalo [0,1]), sem alterar a tarefa temporal.
* **Q03**: configuração principal — escala angular natural para rotações em [0,π] com circuito raso.
* **Q04**: testa se a simetria em torno de zero ([-π,π]) melhora a discriminação.
* **Q05**: robustez a outliers via transformação quantílica uniforme seguida de mapeamento para [0,π].
* **Q06**: mesmo embedding de Q03 com maior profundidade (`reps=2`); isola H3.
* **Q07**: feature map sem emaranhamento; isola H4.
* **Q08**: quatro qubits com escala [0,π] e `reps=1`; compara dimensionalidade sob a melhor escala candidata.
* **Q09**: quatro qubits com escala [0,π] e `reps=2`; combina dimensionalidade e profundidade.
* **Q10**: controle histórico — configuração original da PoC (4 PCs, sem escala angular, ZZ, `reps=2`).

## 6. Comparações clássicas equivalentes

Cada representação reduzida deverá ser avaliada também por modelos clássicos.

Para embeddings com dois componentes:

* Logistic Regression;
* SVM linear;
* SVM RBF;
* SVM polinomial;
* KNN;
* DummyClassifier.

Para embeddings com quatro componentes:

* os mesmos modelos;
* mesma subamostra;
* mesmas partições;
* mesmas sementes.

Também deverão existir dois cenários clássicos:

### Cenário clássico completo

Utiliza todas as características processadas, sem PCA obrigatório.

### Cenário clássico pareado

Utiliza exatamente os mesmos dois ou quatro componentes apresentados ao circuito quântico.

A comparação cientificamente direta deverá ser feita com o cenário pareado.

## 7. Diagnóstico da matriz do kernel

Não avaliar apenas a AUPRC final. Para cada kernel, gerar métricas estruturais.

### 7.1 Estatísticas básicas

Calcular sobre os elementos fora da diagonal:

```text
média dos elementos fora da diagonal
desvio-padrão
mínimo
máximo
percentis 5, 25, 50, 75 e 95
```

### 7.2 Contraste entre classes

Calcular:

$$
\mu_{\text{same}}
=
\mathbb{E}[K(x_i,x_j)\mid y_i=y_j]
$$

$$
\mu_{\text{different}}
=
\mathbb{E}[K(x_i,x_j)\mid y_i\neq y_j]
$$

e:

$$
\Delta_K
=
\mu_{\text{same}}
-
\mu_{\text{different}}
$$

Um kernel potencialmente informativo deve apresentar contraste positivo entre pares da mesma classe e pares de classes diferentes.

### 7.3 Concentração do kernel

Registrar:

$$
CV_K =
\frac{\sigma(K_{i\neq j})}
{\mu(K_{i\neq j})}
$$

Valores muito baixos podem indicar que quase todos os pares possuem similaridades próximas.

### 7.4 Effective rank

Calcular o posto efetivo da matriz:

$$
r_{\text{eff}}
=
\exp
\left(
-\sum_i p_i \log p_i
\right)
$$

onde:

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

Isso ajudará a identificar kernels quase constantes ou dominados por poucos autovalores.

### 7.5 Kernel-target alignment

Calcular o alinhamento entre o kernel e os rótulos:

$$
A(K,Y)
=
\frac{\langle K,Y\rangle_F}
{\|K\|_F \|Y\|_F}
$$

com:

$$
Y = yy^{\top}
$$

adaptando os rótulos para \(\{-1,+1\}\).

O alinhamento deverá ser calculado somente no treinamento.

## 8. Novos arquivos

Adicionar:

```text
src/qml_air_quality/preprocessing/angular_scaling.py
src/qml_air_quality/evaluation/kernel_diagnostics.py
src/qml_air_quality/experiments/quantum_ablation.py
src/qml_air_quality/reporting/ablation_report.py

tests/test_angular_scaling.py
tests/test_kernel_diagnostics.py
tests/test_quantum_ablation.py
```

Criar configuração:

```text
config/quantum_ablation.yaml
```

### Responsabilidades esperadas

* `angular_scaling.py`: interface comum dos escaladores angulares, fit somente no treino, transform em val/teste, serialização.
* `kernel_diagnostics.py`: estatísticas off-diagonal, \(\Delta_K\), \(CV_K\), effective rank, alinhamento, espectro.
* `quantum_ablation.py`: orquestração da matriz Q01–Q10, ranking por validação, bloqueio de acesso ao teste na seleção.
* `ablation_report.py`: tabelas, gráficos, classificação automática dos casos A–D.

## 9. Configuração proposta

```yaml
experiment:
  name: quantum-angular-ablation
  seeds: [42, 123, 2026]

data:
  station: Aotizhongxin
  forecast_horizon_hours: 24
  extreme_percentile: 0.90

sampling:
  train_size: 500
  validation_size: 150
  test_size: 200

dimensionality:
  pca_components:
    - 2
    - 4

angular_scalers:
  - name: none

  - name: minmax_0_1
    minimum: 0.0
    maximum: 1.0

  - name: minmax_0_pi
    minimum: 0.0
    maximum: 3.141592653589793

  - name: minmax_minus_pi_pi
    minimum: -3.141592653589793
    maximum: 3.141592653589793

  - name: quantile_0_pi
    n_quantiles: 100

quantum_models:
  - id: zz_reps_1
    feature_map: ZZFeatureMap
    reps: 1
    entanglement: linear

  - id: zz_reps_2
    feature_map: ZZFeatureMap
    reps: 2
    entanglement: linear

  - id: z_reps_1
    feature_map: ZFeatureMap
    reps: 1
    entanglement: none

evaluation:
  primary_metric: average_precision
  bootstrap_iterations: 1000
  confidence_level: 0.95

kernel_diagnostics:
  calculate_alignment: true
  calculate_effective_rank: true
  calculate_class_contrast: true
  calculate_eigenvalues: true
```

A matriz Q01–Q10 deve ser materializada a partir desta configuração, sem hardcoding paralelo de hiperparâmetros conflitantes.

## 10. Regras de prevenção de vazamento

O Cursor deverá garantir que:

1. o PCA seja ajustado somente no treinamento;
2. o MinMaxScaler angular seja ajustado somente no treinamento;
3. o QuantileTransformer seja ajustado somente no treinamento;
4. o P90 seja calculado somente no treinamento;
5. a seleção da melhor configuração use somente a validação;
6. o teste seja executado uma única vez para a configuração selecionada;
7. nenhuma configuração seja escolhida com base na AUPRC do teste.

Além disso:

* índices das amostras de treino, validação e teste devem ser persistidos;
* kernels calculados devem ser armazenados com identificador da configuração e da semente;
* qualquer transformação fitada deve ser serializada junto aos artefatos do experimento.

## 11. Protocolo de seleção

### Etapa 1 — Treinamento e validação

Executar todas as configurações no treinamento e na validação.

Ordenar as configurações por:

1. AUPRC de validação;
2. recall da classe extrema;
3. kernel-target alignment;
4. menor tempo de construção do kernel.

### Etapa 2 — Seleção

Selecionar:

* melhor QSVM;
* melhor SVM clássico pareado;
* melhor modelo clássico completo.

### Etapa 3 — Teste final

Executar no teste apenas:

* configuração original da PoC;
* melhor QSVM;
* melhor SVM pareado;
* melhor clássico completo;
* DummyClassifier.

O conjunto de teste não participa de nenhuma decisão intermediária. Em caso de empate na validação, usar os critérios de desempate na ordem listada na Etapa 1.

## 12. Resultados necessários

Gerar a tabela:

| Modelo | PCs | Escala | Feature map | Reps | AUPRC val. | AUPRC teste | Recall | Alignment | Effective rank |
| ------ | --: | ------ | ----------- | ---: | ---------: | ----------: | -----: | --------: | -------------: |

Gerar os gráficos:

```text
angular_scaling_distributions.png
kernel_matrices_by_configuration.png
kernel_value_histograms.png
kernel_alignment_comparison.png
kernel_effective_rank.png
validation_auprc_by_configuration.png
test_precision_recall_curves.png
qsvm_vs_classical_paired.png
kernel_contrast_same_vs_different.png
```

### Conteúdo mínimo por gráfico

* `angular_scaling_distributions.png`: histogramas/KDE dos embeddings após cada estratégia angular no treino.
* `kernel_matrices_by_configuration.png`: heatmaps das matrizes de kernel (amostra de treino) por configuração.
* `kernel_value_histograms.png`: distribuição dos valores fora da diagonal.
* `kernel_alignment_comparison.png`: barras de \(A(K,Y)\) por configuração.
* `kernel_effective_rank.png`: posto efetivo normalizado por configuração.
* `validation_auprc_by_configuration.png`: AUPRC de validação (média ± IC quando houver múltiplas sementes).
* `test_precision_recall_curves.png`: curvas PR apenas dos modelos liberados para o teste.
* `qsvm_vs_classical_paired.png`: comparação direta QSVM vs SVM pareado.
* `kernel_contrast_same_vs_different.png`: \(\mu_{\text{same}}\), \(\mu_{\text{different}}\) e \(\Delta_K\).

## 13. Interpretação automática

### Caso A — Kernel continua semelhante ao Dummy

Classificação:

```text
QUANTUM_KERNEL_NOT_INFORMATIVE
```

Condições indicativas:

* AUPRC próxima à prevalência;
* alinhamento próximo de zero;
* contraste entre classes próximo de zero;
* matriz quase constante;
* baixa variação fora da diagonal.

Conclusão:

> O baixo desempenho não decorreu apenas da ausência de normalização angular. O feature map testado não representa adequadamente a estrutura preditiva da tarefa.

### Caso B — Escala melhora o kernel, mas não supera o clássico

Classificação:

```text
BETTER_QUANTUM_REPRESENTATION_WITHOUT_ADVANTAGE
```

Condições indicativas:

* melhoria estrutural frente a Q10 (maior \(\Delta_K\), maior alinhamento ou maior \(CV_K\));
* AUPRC de validação/teste ainda inferior ou equivalente ao melhor SVM pareado.

Conclusão:

> A adaptação angular tornou o kernel mais informativo, mas o ganho não foi suficiente para superar kernels clássicos equivalentes.

### Caso C — QSVM supera o clássico apenas na validação

Classificação:

```text
NON_GENERALIZING_QUANTUM_GAIN
```

Condições indicativas:

* melhor QSVM acima do melhor SVM pareado na validação;
* diferença some ou inverte no teste temporal.

Conclusão:

> O ganho observado durante a seleção não se manteve no período temporal futuro.

### Caso D — QSVM supera o clássico pareado no teste

Classificação:

```text
CANDIDATE_PREDICTIVE_QUANTUM_GAIN
```

Requisitos:

* AUPRC superior;
* intervalo bootstrap da diferença acima de zero;
* diferença mínima de 0,02;
* melhor desempenho em pelo menos duas das três sementes;
* recall da classe extrema não inferior;
* resultado reproduzível.

Mesmo nesse caso, não usar o termo “vantagem quântica computacional”.

## 14. Critérios de aceite

A nova iteração será considerada concluída quando:

* todas as estratégias angulares forem ajustadas somente no treinamento;
* as configurações com dois e quatro qubits forem executadas;
* `reps=1` e `reps=2` forem comparados;
* circuitos com e sem emaranhamento forem comparados;
* diagnósticos estruturais dos kernels forem produzidos;
* a seleção ocorrer somente pela validação;
* o conjunto de teste não for usado na busca de configuração;
* QSVM e SVM usarem exatamente os mesmos embeddings;
* o relatório distinguir desempenho preditivo de vantagem quântica;
* os resultados da PoC original permanecerem disponíveis como controle.

## 15. Backlog para o Cursor

### Fase 1 — Escalonamento angular

Implementar:

* interface comum para escaladores angulares;
* MinMax ([0,1]);
* MinMax ([0,π]);
* MinMax ([-π,π]);
* Quantile ([0,π]);
* persistência dos escaladores;
* testes contra vazamento.

Commit:

```text
feat: add train-only angular scaling for quantum embeddings
```

### Fase 2 — Configurações de dois qubits

Implementar:

* PCA com dois componentes;
* ZZFeatureMap com `reps=1`;
* ZZFeatureMap com `reps=2`;
* ZFeatureMap sem emaranhamento;
* cache independente dos kernels.

Commit:

```text
feat: add two-qubit quantum kernel ablations
```

### Fase 3 — Diagnóstico dos kernels

Implementar:

* distribuição dos valores;
* contraste entre classes;
* effective rank;
* alinhamento kernel-target;
* espectro de autovalores.

Commit:

```text
feat: add quantum kernel informativeness diagnostics
```

### Fase 4 — Experimento de ablação

Implementar:

* execução automática da matriz experimental;
* validação por configuração;
* ranking;
* persistência dos resultados;
* prevenção do acesso ao teste durante a seleção.

Commit:

```text
feat: implement leakage-free quantum ablation experiment
```

### Fase 5 — Comparação final

Implementar:

* teste da melhor configuração;
* comparação pareada com SVM;
* bootstrap da diferença;
* classificação automática do resultado.

Commit:

```text
feat: compare selected qsvm against paired classical baselines
```

### Fase 6 — Relatório científico

Implementar:

* relatório de ablação;
* gráficos;
* tabelas;
* interpretação das matrizes;
* limitações;
* comparação metodológica com Farooq et al.

Commit:

```text
docs: report angular scaling and quantum kernel ablations
```

## 16. Prompt para o Cursor

```text
Implemente a nova iteração da PoC descrita no arquivo
PLAN_QUANTUM_ABLATION.md.

Contexto científico:

A PoC original utiliza uma tarefa temporal difícil e sem vazamento:
prever se PM2.5 em t+24 horas excederá o percentil 90 do treinamento.

O QSVM original obteve AUPRC próxima ao Dummy. Uma possível causa é que os
quatro componentes PCA foram enviados diretamente ao ZZFeatureMap após
StandardScaler, sem transformação para um intervalo angular.

O objetivo desta iteração não é facilitar a tarefa nem reproduzir a
classificação circular de AQI do artigo de Farooq et al. O objetivo é verificar
se a escala angular, a dimensionalidade e a profundidade do circuito tornam o
kernel quântico mais informativo na tarefa temporal existente.

Implemente:

1. escalonamento angular ajustado somente no treinamento:
   - sem escala adicional;
   - MinMax [0,1];
   - MinMax [0,pi];
   - MinMax [-pi,pi];
   - QuantileTransformer seguido de [0,pi];

2. PCA com 2 e 4 componentes;

3. feature maps:
   - ZZFeatureMap reps=1, linear;
   - ZZFeatureMap reps=2, linear;
   - ZFeatureMap reps=1, sem emaranhamento;

4. diagnósticos dos kernels:
   - estatísticas fora da diagonal;
   - contraste entre pares da mesma classe e de classes diferentes;
   - kernel-target alignment;
   - effective rank;
   - autovalores;
   - histogramas e heatmaps;

5. seleção usando somente validação;

6. teste final apenas da melhor configuração e dos controles;

7. comparação pareada com SVM linear e RBF usando exatamente os mesmos
   componentes e amostras;

8. AUPRC como métrica principal;

9. bootstrap temporal da diferença entre QSVM e melhor baseline pareado;

10. relatório automático sem usar o termo vantagem quântica computacional.

Requisitos:

- não alterar a definição do alvo;
- não alterar o split temporal;
- não calcular P90 fora do treinamento;
- não usar o teste para escolher configurações;
- não omitir resultados negativos;
- manter a configuração quântica anterior como controle;
- armazenar os índices das amostras;
- armazenar os kernels calculados;
- escrever testes para todos os escaladores e diagnósticos;
- registrar versões das bibliotecas e sementes.

Implemente uma fase por vez.
Ao final de cada fase, execute pytest e ruff.
Não avance para a fase seguinte até que os critérios de aceite da fase atual
estejam atendidos.
```

## 17. Próxima decisão após essa iteração

A evolução dependerá do resultado.

### Se nenhuma configuração produzir kernel informativo

Não aumentar imediatamente o número de qubits ou a profundidade.

A próxima abordagem deverá ser:

* feature map treinável;
* kernel-target alignment como função objetivo;
* codificador clássico supervisionado;
* comparação com projected quantum kernels.

### Se houver melhora estrutural, mas não preditiva

Testar:

* seleção supervisionada de duas características;
* Partial Least Squares;
* Linear Discriminant Analysis ajustada apenas no treinamento;
* autoencoder temporal;
* embeddings produzidos por uma TCN.

### Se houver ganho preditivo consistente

Repetir o protocolo com:

* cinco janelas walk-forward;
* múltiplas estações;
* horizontes de 6, 12, 24 e 48 horas;
* percentis 90 e 95;
* simulador com ruído;
* subconjunto em hardware quântico real.
