# Plano de desenvolvimento da PoC no Cursor

## 1. Objetivo da PoC

Desenvolver uma aplicação reproduzível para comparar modelos clássicos e um modelo híbrido quântico-clássico na seguinte tarefa:

> Prever, com 24 horas de antecedência, se ocorrerá um episódio extremo de PM2.5 em uma estação de monitoramento de Beijing.

A PoC não deverá tentar reproduzir toda a complexidade do artigo científico futuro. Ela deverá validar:

1. aquisição e preparação correta dos dados;
2. criação de um alvo futuro sem vazamento;
3. redução clássica das características;
4. treinamento de modelos clássicos;
5. treinamento de um QSVM com kernel quântico;
6. comparação justa entre os modelos;
7. geração automática de métricas e gráficos.

## 2. Escopo fechado da primeira versão

### Dataset

Usar o **Beijing Multi-Site Air Quality**, disponível no UCI Machine Learning Repository.

Para a primeira versão, utilizar apenas a estação:

```text
Aotizhongxin
```

O dataset contém registros horários de poluentes e variáveis meteorológicas em diferentes estações de Beijing.

### Tarefa

Classificação binária:

```text
0 = PM2.5 futuro abaixo do limite extremo
1 = PM2.5 futuro igual ou acima do limite extremo
```

Horizonte:

```text
24 horas
```

Definição do evento extremo:

```text
PM2.5(t + 24h) >= percentil 90 do PM2.5 do conjunto de treinamento
```

O percentil deverá ser calculado exclusivamente com os dados de treinamento.

### Modelos obrigatórios

Implementar:

* DummyClassifier;
* LogisticRegression;
* SVM com kernel linear;
* SVM com kernel RBF;
* QSVM com kernel quântico de fidelidade.

### Execução quântica

A primeira versão deverá utilizar somente simulador local ideal.

Não incluir inicialmente:

* hardware quântico real;
* simulação de ruído;
* kernel quântico treinável;
* múltiplas estações;
* Graph Neural Networks;
* previsão por regressão;
* interface web.

Esses itens deverão ficar registrados como extensões futuras.

## 3. Stack tecnológica

### Linguagem

```text
Python 3.11
```

O pacote atual `qiskit-machine-learning` requer Python 3.10 ou superior.

### Bibliotecas

```text
pandas
numpy
scikit-learn
matplotlib
qiskit
qiskit-machine-learning
qiskit-aer
ucimlrepo
pyyaml
joblib
pytest
ruff
```

O Qiskit Machine Learning 0.9 é compatível com Qiskit 2.x.

O Qiskit Machine Learning fornece oficialmente:

* `FidelityQuantumKernel`;
* `TrainableFidelityQuantumKernel`;
* `QuantumKernelTrainer`;
* `QSVC`.

O `QSVC` estende o `SVC` do scikit-learn e permite o fornecimento de um kernel quântico.

### Gerenciamento do projeto

Usar:

```text
pyproject.toml
```

O projeto deve funcionar com:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No Windows:

```bash
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 4. Estrutura de diretórios

O Cursor deverá criar a seguinte estrutura:

```text
qml-air-quality-poc/
├── .cursor/
│   └── rules/
│       └── project-guidelines.mdc
├── config/
│   └── poc.yaml
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── artifacts/
│   ├── datasets/
│   ├── kernels/
│   ├── models/
│   ├── plots/
│   └── reports/
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   └── qml_air_quality/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py
│       │   ├── load.py
│       │   ├── preprocess.py
│       │   ├── features.py
│       │   └── split.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── classical.py
│       │   ├── quantum.py
│       │   └── factory.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── bootstrap.py
│       │   └── plots.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── classical_experiment.py
│       │   ├── quantum_experiment.py
│       │   └── comparison.py
│       └── reporting/
│           ├── __init__.py
│           └── report.py
├── tests/
│   ├── test_download.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_temporal_split.py
│   ├── test_no_leakage.py
│   ├── test_classical_models.py
│   ├── test_quantum_kernel.py
│   └── test_end_to_end.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
└── LICENSE
```

## 5. Arquivo de configuração

Criar `config/poc.yaml`:

```yaml
project:
  name: qml-air-quality-poc
  random_seeds: [42, 123, 2026]

data:
  dataset_id: 501
  station: Aotizhongxin
  target_column: PM2.5
  timestamp_column: timestamp

forecast:
  horizon_hours: 24
  extreme_percentile: 0.90

features:
  raw_columns:
    - PM2.5
    - PM10
    - SO2
    - NO2
    - CO
    - O3
    - TEMP
    - PRES
    - DEWP
    - RAIN
    - WSPM

  rolling_windows:
    - 6
    - 24

  add_calendar_features: true
  pca_components: 4

split:
  train_fraction: 0.60
  validation_fraction: 0.20
  test_fraction: 0.20

sampling:
  quantum_train_size: 500
  quantum_validation_size: 150
  quantum_test_size: 200

classical_models:
  dummy:
    strategy: prior

  logistic_regression:
    class_weight: balanced
    max_iter: 2000

  linear_svm:
    kernel: linear
    class_weight: balanced
    probability: true

  rbf_svm:
    kernel: rbf
    class_weight: balanced
    probability: true
    C: 1.0
    gamma: scale

quantum:
  feature_map: ZZFeatureMap
  repetitions: 2
  entanglement: linear
  enforce_psd: true
  cache_kernel: true

evaluation:
  primary_metric: average_precision
  bootstrap_iterations: 500
  confidence_level: 0.95
```

## 6. Pipeline de dados

### 6.1 Download

Implementar um comando:

```bash
python -m qml_air_quality.cli download
```

Responsabilidades:

1. verificar se os dados já existem;
2. baixar o dataset do UCI;
3. salvar os arquivos em `data/raw`;
4. registrar data, origem e checksum;
5. não baixar novamente quando o checksum estiver correto.

O pacote `ucimlrepo` pode ser utilizado para importar datasets do UCI diretamente em scripts Python.

### 6.2 Construção do timestamp

Unir:

```text
year
month
day
hour
```

em:

```text
timestamp
```

Garantir que:

```python
df["timestamp"].is_monotonic_increasing
```

seja verdadeiro após a ordenação.

Remover:

* registros duplicados;
* linhas sem timestamp válido;
* colunas de identificação que não sejam usadas.

### 6.3 Tratamento de valores ausentes

Aplicar uma estratégia causal:

1. ordenar temporalmente;
2. usar `forward fill` limitado a três horas;
3. dividir os dados temporalmente;
4. calcular medianas apenas no treinamento;
5. usar as medianas do treinamento para preencher os valores restantes da validação e do teste.

É proibido:

* interpolar utilizando valores futuros;
* calcular média ou mediana com o dataset completo;
* remover todas as linhas com valores ausentes sem registrar o impacto.

### 6.4 Criação do alvo

Criar:

```python
future_pm25 = df["PM2.5"].shift(-24)
```

Depois da divisão temporal, calcular:

```python
threshold = train["future_pm25"].quantile(0.90)
```

Criar:

```python
target = (future_pm25 >= threshold).astype(int)
```

Salvar o limiar em:

```text
artifacts/datasets/target_metadata.json
```

Exemplo:

```json
{
  "target": "PM2.5",
  "horizon_hours": 24,
  "percentile": 0.9,
  "threshold": 187.4,
  "threshold_source": "training_partition"
}
```

O valor acima é apenas ilustrativo. O sistema deverá calcular o valor real.

## 7. Engenharia de características

Para cada variável numérica, criar:

```text
valor atual
lag de 1 hora
lag de 6 horas
lag de 12 horas
lag de 24 horas
média móvel de 6 horas
desvio-padrão móvel de 6 horas
média móvel de 24 horas
desvio-padrão móvel de 24 horas
```

Adicionar características cíclicas:

```python
hour_sin
hour_cos
month_sin
month_cos
```

Equações:

```python
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)

month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)
```

Todas as características devem usar somente informações disponíveis até o instante `t`.

## 8. Divisão temporal

Usar divisão cronológica:

```text
60% treinamento
20% validação
20% teste
```

Não usar:

```python
train_test_split(..., shuffle=True)
```

Os métodos comuns de validação aleatória são inadequados para dados ordenados no tempo porque podem treinar com observações futuras e avaliar observações passadas. O `TimeSeriesSplit` do scikit-learn foi desenvolvido para preservar essa ordem.

A PoC poderá implementar uma divisão temporal fixa, mas a função deverá ser compatível com futura avaliação `TimeSeriesSplit`.

## 9. Pré-processamento clássico

Criar um pipeline scikit-learn com:

```text
SimpleImputer
StandardScaler
PCA
```

Regras:

* ajustar o imputer somente no treinamento;
* ajustar o scaler somente no treinamento;
* ajustar o PCA somente no treinamento;
* usar quatro componentes principais;
* salvar todos os objetos ajustados;
* aplicar os mesmos objetos na validação e no teste.

O resultado esperado é:

```text
X_train_quantum.shape = (n_train, 4)
X_validation_quantum.shape = (n_validation, 4)
X_test_quantum.shape = (n_test, 4)
```

Como serão usados quatro componentes, o circuito deverá ter quatro qubits.

## 10. Subamostragem para o experimento quântico

O cálculo de uma matriz de kernel exige comparações entre pares de amostras. Para manter a PoC executável localmente, limitar o experimento a:

```text
500 amostras de treinamento
150 amostras de validação
200 amostras de teste
```

A subamostragem deverá:

1. ocorrer depois da divisão temporal;
2. ser estratificada pela classe;
3. utilizar uma semente registrada;
4. fornecer exatamente os mesmos dados para SVM e QSVM;
5. preservar os índices originais;
6. salvar os índices utilizados.

Arquivo esperado:

```text
artifacts/datasets/quantum_sample_seed_42.csv
```

Os modelos clássicos deverão ser avaliados em dois cenários:

```text
classical-full
classical-quantum-subset
```

A comparação direta com o QSVM deverá utilizar `classical-quantum-subset`.

## 11. Modelos clássicos

Criar uma interface comum:

```python
class ModelResult:
    model_name: str
    seed: int
    predictions: np.ndarray
    probabilities: np.ndarray | None
    training_seconds: float
    inference_seconds: float
    metadata: dict
```

Implementar:

### DummyClassifier

Objetivo:

```text
estabelecer o desempenho mínimo
```

### Regressão logística

Configuração:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=seed
)
```

### SVM linear

```python
SVC(
    kernel="linear",
    class_weight="balanced",
    probability=True,
    random_state=seed
)
```

### SVM RBF

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=seed
)
```

Na primeira versão, não realizar uma busca extensa de hiperparâmetros.

Realizar apenas uma busca pequena usando a validação:

```yaml
C: [0.1, 1.0, 10.0]
gamma: [scale, auto]
```

O conjunto de teste só poderá ser usado uma vez, após a escolha da configuração.

## 12. Modelo quântico

### 12.1 Feature map

Utilizar:

```python
ZZFeatureMap(
    feature_dimension=4,
    reps=2,
    entanglement="linear"
)
```

### 12.2 Kernel

Utilizar:

```python
FidelityQuantumKernel
```

O kernel de fidelidade calcula a similaridade pelo módulo quadrado da sobreposição entre estados produzidos pelo feature map.

Conceitualmente:

```text
K(x, y) = |<phi(x)|phi(y)>|²
```

### 12.3 Classificador

Utilizar:

```python
QSVC(
    quantum_kernel=quantum_kernel,
    class_weight="balanced",
    probability=True
)
```

Caso a API instalada não aceite diretamente algum dos parâmetros herdados do `SVC`, o Cursor deverá verificar a assinatura da versão instalada e documentar a adaptação.

### 12.4 Cache da matriz

Calcular e salvar:

```text
K_train_train
K_validation_train
K_test_train
```

Arquivos:

```text
artifacts/kernels/train_kernel_seed_42.npy
artifacts/kernels/validation_kernel_seed_42.npy
artifacts/kernels/test_kernel_seed_42.npy
artifacts/kernels/kernel_metadata_seed_42.json
```

O metadata deverá incluir:

```json
{
  "feature_map": "ZZFeatureMap",
  "qubits": 4,
  "repetitions": 2,
  "entanglement": "linear",
  "train_samples": 500,
  "validation_samples": 150,
  "test_samples": 200,
  "seed": 42
}
```

### 12.5 Validação do kernel

Testar:

```python
K.shape == (n_samples, n_samples)
np.allclose(K, K.T, atol=1e-8)
np.all(np.isfinite(K))
np.allclose(np.diag(K), 1.0, atol=1e-5)
```

Quando `enforce_psd=True`, verificar numericamente se os autovalores negativos, caso existam por imprecisão numérica, estão próximos de zero.

## 13. Métricas

### Métrica principal

```text
Average Precision — AUPRC
```

### Métricas secundárias

```text
balanced accuracy
precision da classe extrema
recall da classe extrema
F1 da classe extrema
AUROC
MCC
Brier Score, quando houver probabilidades
tempo de treinamento
tempo de inferência
```

Não utilizar acurácia simples como métrica principal, pois a classe extrema será minoritária.

## 14. Avaliação estatística

Executar cada experimento com:

```text
seed 42
seed 123
seed 2026
```

A divisão temporal deve permanecer igual. As sementes deverão controlar:

* subamostragem;
* modelos estocásticos;
* bootstrap;
* qualquer inicialização aleatória.

Calcular:

```text
média
desvio-padrão
mediana
intervalo de confiança bootstrap de 95%
```

Para cada modelo, gerar:

```text
metrics_by_seed.csv
metrics_summary.csv
predictions.csv
```

Para a diferença entre QSVM e o melhor SVM clássico, calcular:

```text
delta de AUPRC
intervalo bootstrap da diferença
```

## 15. Critério de resultado da PoC

A PoC não deverá declarar “vantagem quântica”.

Ela poderá registrar um **sinal preliminar de vantagem preditiva** quando:

```text
AUPRC média do QSVM > AUPRC média do melhor baseline clássico
```

e:

```text
limite inferior do intervalo bootstrap da diferença > 0
```

Para a PoC, utilizar adicionalmente um limiar de relevância:

```text
diferença absoluta de AUPRC >= 0,02
```

Classificação automática do resultado:

```text
NO_EVIDENCE
INCONCLUSIVE
CANDIDATE_QUANTUM_GAIN
CLASSICAL_ADVANTAGE
```

Regras:

```python
if qs_delta_ci_high < 0:
    result = "CLASSICAL_ADVANTAGE"
elif qs_delta_ci_low > 0 and qs_delta_mean >= 0.02:
    result = "CANDIDATE_QUANTUM_GAIN"
elif qs_delta_ci_low <= 0 <= qs_delta_ci_high:
    result = "INCONCLUSIVE"
else:
    result = "NO_EVIDENCE"
```

O relatório deverá deixar claro que um resultado da PoC não demonstra vantagem computacional ou superioridade quântica geral.

## 16. Gráficos obrigatórios

Gerar:

```text
class_distribution.png
pm25_over_time.png
missing_values.png
pca_explained_variance.png
precision_recall_curves.png
roc_curves.png
confusion_matrices.png
metrics_comparison.png
training_time_comparison.png
bootstrap_delta_qsvm_vs_rbf.png
quantum_kernel_matrix.png
```

Os gráficos devem ser salvos em:

```text
artifacts/plots/
```

## 17. Relatório automático

Gerar:

```text
artifacts/reports/poc_report.md
```

Estrutura:

```markdown
# QML Air Quality PoC Report

## 1. Configuração
## 2. Dataset
## 3. Divisão temporal
## 4. Definição do alvo
## 5. Pré-processamento
## 6. Modelos
## 7. Resultados
## 8. Comparação estatística
## 9. Custos computacionais
## 10. Limitações
## 11. Classificação do resultado
## 12. Próximos experimentos
```

O relatório deverá mostrar explicitamente:

* período de treinamento;
* período de validação;
* período de teste;
* limiar de PM2.5;
* distribuição das classes;
* número de amostras;
* número de características antes do PCA;
* variância explicada pelo PCA;
* configuração do circuito;
* quantidade de qubits;
* matriz completa de métricas;
* tempo de execução;
* classificação final do resultado.

## 18. Interface de linha de comando

Implementar:

```bash
python -m qml_air_quality.cli download
python -m qml_air_quality.cli prepare
python -m qml_air_quality.cli train-classical
python -m qml_air_quality.cli train-quantum
python -m qml_air_quality.cli compare
python -m qml_air_quality.cli report
python -m qml_air_quality.cli run-all
```

Todos os comandos deverão aceitar:

```bash
--config config/poc.yaml
```

O comando completo deverá ser:

```bash
python -m qml_air_quality.cli run-all --config config/poc.yaml
```

## 19. Makefile

Criar:

```makefile
setup:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest -q

download:
	python -m qml_air_quality.cli download --config config/poc.yaml

prepare:
	python -m qml_air_quality.cli prepare --config config/poc.yaml

classical:
	python -m qml_air_quality.cli train-classical --config config/poc.yaml

quantum:
	python -m qml_air_quality.cli train-quantum --config config/poc.yaml

compare:
	python -m qml_air_quality.cli compare --config config/poc.yaml

report:
	python -m qml_air_quality.cli report --config config/poc.yaml

all:
	python -m qml_air_quality.cli run-all --config config/poc.yaml
```

## 20. Testes obrigatórios

### Teste de vazamento temporal

Para cada amostra:

```python
feature_max_timestamp < target_timestamp
```

Mais especificamente:

```python
target_timestamp == feature_timestamp + timedelta(hours=24)
```

### Teste do limiar

Confirmar que o percentil foi calculado somente com o treinamento:

```python
assert metadata["threshold_source"] == "training_partition"
```

### Teste dos transformadores

Confirmar que:

* imputer foi ajustado no treinamento;
* scaler foi ajustado no treinamento;
* PCA foi ajustado no treinamento;
* validação e teste não foram usados no `fit`.

### Teste da divisão

Confirmar:

```python
train.timestamp.max() < validation.timestamp.min()
validation.timestamp.max() < test.timestamp.min()
```

### Teste das características

Confirmar que nenhum nome de coluna contém:

```text
future
target
t_plus_24
```

no conjunto de entrada do modelo.

### Teste do kernel

Confirmar:

* simetria;
* diagonal aproximadamente unitária;
* ausência de NaN;
* dimensões corretas;
* reprodutibilidade.

### Teste end-to-end

Executar uma configuração reduzida:

```text
50 amostras de treinamento
20 de validação
20 de teste
2 componentes PCA
2 qubits
```

O teste deverá verificar se o pipeline completo produz:

```text
metrics_summary.csv
poc_report.md
```

## 21. Critérios de aceite

A fase será considerada concluída quando:

* `make lint` executar sem erros;
* `make test` executar sem falhas;
* `make all` concluir o pipeline;
* o dataset puder ser recriado do zero;
* os splits forem estritamente temporais;
* o alvo estiver 24 horas à frente;
* o percentil for calculado somente no treinamento;
* SVM e QSVM usarem exatamente o mesmo subconjunto;
* as matrizes do kernel forem armazenadas;
* as métricas forem reproduzíveis;
* o relatório final for criado automaticamente;
* nenhuma conclusão automática use o termo “vantagem quântica”;
* limitações e custos computacionais sejam reportados.

## 22. Backlog para o Cursor

### Fase 1 — Bootstrap do projeto

Entregas:

* estrutura de diretórios;
* `pyproject.toml`;
* `.gitignore`;
* `Makefile`;
* configuração YAML;
* CLI inicial;
* testes mínimos.

Commit sugerido:

```text
chore: initialize reproducible qml air quality project
```

### Fase 2 — Aquisição e auditoria dos dados

Entregas:

* download;
* carregamento;
* construção do timestamp;
* relatório de valores ausentes;
* notebook exploratório.

Commit sugerido:

```text
feat: add UCI air quality data acquisition and audit
```

### Fase 3 — Pipeline temporal sem vazamento

Entregas:

* tratamento causal;
* lags;
* estatísticas móveis;
* target futuro;
* threshold de treinamento;
* divisão temporal;
* testes de vazamento.

Commit sugerido:

```text
feat: implement leakage-free temporal dataset pipeline
```

### Fase 4 — Baselines clássicos

Entregas:

* Dummy;
* regressão logística;
* SVM linear;
* SVM RBF;
* métricas;
* persistência dos modelos.

Commit sugerido:

```text
feat: add classical air quality classification baselines
```

### Fase 5 — Redução dimensional

Entregas:

* scaler;
* PCA;
* persistência;
* gráfico de variância explicada;
* dataset de quatro dimensões.

Commit sugerido:

```text
feat: add train-only dimensionality reduction pipeline
```

### Fase 6 — Kernel quântico e QSVM

Entregas:

* ZZFeatureMap;
* FidelityQuantumKernel;
* QSVC;
* cache das matrizes;
* testes do kernel;
* medição de tempo.

Commit sugerido:

```text
feat: implement fidelity quantum kernel classifier
```

### Fase 7 — Comparação e estatística

Entregas:

* três sementes;
* bootstrap;
* comparação pareada;
* gráficos;
* classificação automática do resultado.

Commit sugerido:

```text
feat: add statistical comparison of classical and quantum models
```

### Fase 8 — Relatório e documentação

Entregas:

* relatório Markdown;
* README;
* diagrama do pipeline;
* instruções de reprodução;
* limitações.

Commit sugerido:

```text
docs: add reproducible poc report and execution guide
```

## 23. Prompt mestre para colar no Cursor

```text
Você deverá desenvolver uma PoC científica e reproduzível para comparar modelos
clássicos e um QSVM na previsão antecipada de episódios extremos de PM2.5.

Leia integralmente o arquivo PLAN.md antes de implementar qualquer código.

Objetivo:
Prever se PM2.5 em t+24 horas será igual ou superior ao percentil 90 calculado
exclusivamente no conjunto de treinamento.

Dataset:
Beijing Multi-Site Air Quality, estação Aotizhongxin.

Modelos obrigatórios:
1. DummyClassifier
2. LogisticRegression
3. SVM linear
4. SVM RBF
5. QSVC com FidelityQuantumKernel e ZZFeatureMap

Requisitos fundamentais:
- preservar rigorosamente a ordem temporal;
- não utilizar valores futuros nas características;
- calcular imputação, normalização, PCA e threshold somente no treinamento;
- reduzir os dados para quatro componentes PCA e quatro qubits;
- utilizar exatamente o mesmo subconjunto para SVM e QSVM;
- executar inicialmente em simulador local ideal;
- salvar configurações, índices, modelos, kernels, métricas e gráficos;
- escrever testes para detectar vazamento;
- produzir um relatório Markdown automaticamente;
- não declarar vantagem quântica com base apenas na PoC.

Trabalhe uma fase por vez.
Antes de iniciar cada fase:
1. descreva os arquivos que serão criados ou alterados;
2. descreva as decisões de projeto;
3. confirme os critérios de aceite daquela fase.

Depois de implementar cada fase:
1. execute os testes;
2. execute o linter;
3. apresente um resumo das alterações;
4. registre problemas ou limitações;
5. aguarde a solicitação para avançar para a fase seguinte.

Não implemente todas as fases em uma única alteração.
Não modifique testes apenas para ocultar erros.
Não use dados de validação ou teste no ajuste de transformações.
Não use valores fixos silenciosos: parâmetros experimentais devem estar no YAML.
Use type hints, docstrings, logging e tratamento explícito de erros.
```

## 24. Prompt da primeira fase

```text
Implemente somente a Fase 1 do PLAN.md.

Crie:
- estrutura do projeto;
- pyproject.toml;
- Makefile;
- config/poc.yaml;
- módulo de configuração;
- CLI com comandos ainda vazios ou com mensagens informativas;
- configuração do Ruff;
- configuração do Pytest;
- README inicial;
- testes de importação e carregamento da configuração.

Não implemente download, processamento de dados ou modelos nesta fase.

Ao concluir:
1. execute pytest;
2. execute ruff check;
3. mostre a árvore de diretórios;
4. liste os comandos disponíveis;
5. informe qualquer decisão que tenha divergido do PLAN.md.
```

## 25. Extensões após a PoC

Depois que a versão inicial estiver estável, evoluir nesta ordem:

1. incluir as 12 estações;
2. adicionar horizontes de 6, 12 e 48 horas;
3. comparar percentis 90 e 95;
4. implementar validação walk-forward;
5. incluir kernel quântico treinável;
6. comparar circuito com e sem emaranhamento;
7. implementar simulação de ruído;
8. executar um subconjunto em QPU;
9. adicionar regressão para concentração de PM2.5;
10. substituir PCA por um autoencoder temporal;
11. adicionar TCN ou GRU como codificador clássico;
12. validar no KDD Cup 2018.
