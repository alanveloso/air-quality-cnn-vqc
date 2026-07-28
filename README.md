# qml-air-quality

PoC de **machine learning quântico** para previsão (t+24h) de episódios extremos de PM2.5 na estação **Aotizhongxin** (Beijing Multi-Site, UCI 501), com comparação clássico vs QSVM, ablação angular, notebook Kaggle e frontend **Névoa**.

Repositório: [`github.com/alanveloso/qml-air-quality`](https://github.com/alanveloso/qml-air-quality) · Demo (GitHub Pages): [`alanveloso.github.io/qml-air-quality`](https://alanveloso.github.io/qml-air-quality/)

## Estrutura

```text
src/qml_air_quality/   # biblioteca (dados, features, modelos, ablação)
config/                # poc.yaml, farooq_style_stats.yaml, …
scripts/               # smoke_run, farooq experiments
notebooks/             # PoC + Kaggle Farooq-style
frontend/              # Névoa — demo UI (representa QSVM; sem inferência live)
artifacts/farooq_style/  # resultados (kaggle_medium/ + local_small/)
kaggle/                # instruções Kaggle
tests/
```

Resultados: [`artifacts/farooq_style/`](artifacts/farooq_style/README.md).

## Setup Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dataset: download automático do ZIP UCI (cache em `data/raw/`).

```bash
python scripts/smoke_run.py
python scripts/run_ablation.py
python scripts/run_aqi_bands_local.py --train-size 80 --test-size 40
```

### Notebooks

1. `notebooks/01_prepare_data.ipynb`
2. `notebooks/02_classical_baselines.ipynb`
3. `notebooks/03_quantum_vs_classical.ipynb`
4. `notebooks/04_quantum_ablation.ipynb`
5. `notebooks/kaggle_farooq_style_qsvm.ipynb` — independente para Kaggle (`MODE = small|medium|large`)

Guia Kaggle: [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md).

## Frontend (Névoa)

```bash
cd frontend
npm install
npm run dev
```

Mapa em tela cheia com as 12 estações de Beijing; recomendações por faixa AQI. A interface **apresenta o QSVM Farooq como modelo da PoC** (demo) — não executa predição. Artefatos de referência: `artifacts/farooq_style/kaggle_medium/`.

Deploy automático via GitHub Actions (`.github/workflows/pages.yml`) ao push em `main`/`master` com mudanças em `frontend/`. Em **Settings → Pages**, use source **GitHub Actions**.

## Regras do experimento

- Split temporal 60/20/20
- Limiar do extremo (P90) só no treino
- Imputer / scaler / PCA / escala angular só no treino
- Métrica principal: **AUPRC**
- Simulador quântico local (sem hardware)

## Planos

- [`PLAN.md`](PLAN.md)
- [`PLAN_QUANTUM_ABLATION.md`](PLAN_QUANTUM_ABLATION.md)
