# QML Air Quality PoC (notebooks)

PoC enxuta para comparar baselines clássicos e um QSVM na previsão (t+24h) de episódios extremos de PM2.5 na estação **Aotizhongxin** (Beijing Multi-Site Air Quality, UCI).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

O dataset UCI 501 não está disponível via `ucimlrepo` import; o download usa o ZIP oficial do archive.ics.uci.edu (com cache em `data/raw/`).

Smoke rápido (subset menor que o YAML):

```bash
python scripts/smoke_run.py
```

## Notebooks (nessa ordem)

1. [`notebooks/01_prepare_data.ipynb`](notebooks/01_prepare_data.ipynb) — download, pipeline temporal, PCA, subset
2. [`notebooks/02_classical_baselines.ipynb`](notebooks/02_classical_baselines.ipynb) — Dummy, LogReg, SVM linear/RBF
3. [`notebooks/03_quantum_vs_classical.ipynb`](notebooks/03_quantum_vs_classical.ipynb) — QSVM + comparação (controle original)
4. [`notebooks/04_quantum_ablation.ipynb`](notebooks/04_quantum_ablation.ipynb) — ablação angular Q01–Q10

Ou via script:

```bash
python scripts/run_ablation.py
```

Configuração base: [`config/poc.yaml`](config/poc.yaml).  
Ablação: [`config/quantum_ablation.yaml`](config/quantum_ablation.yaml).  
Planos: [`PLAN.md`](PLAN.md), [`PLAN_QUANTUM_ABLATION.md`](PLAN_QUANTUM_ABLATION.md).

## Regras importantes

- Split estritamente temporal (60/20/20)
- Threshold do extremo (P90) só no treino
- Imputer / scaler / PCA / escala angular ajustados só no treino
- SVM e QSVM usam exatamente o mesmo subset
- Seleção de configuração quântica **somente na validação**
- Simulador quântico local ideal (sem hardware)

## Artefatos

- PoC original: `artifacts/`
- Ablação: `artifacts/ablation/` (kernels, ranking, relatório)

## Kaggle

A ablação completa demora horas em CPU. Pacote e instruções em [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md).
