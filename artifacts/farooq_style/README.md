# Farooq-style — artefatos

Protocolo da PoC: features 8D (min/max/mediana/variância de PM2.5 e TEMP), limiar P90 no treino, QSVM com PCA→2 qubits. Notebook: `notebooks/kaggle_farooq_style_qsvm.ipynb`.

## Pastas

| Pasta | Origem | Conteúdo |
|---|---|---|
| `kaggle_medium/` | Kaggle, modo `medium` (2026-07-27) | **Resultado principal** da PoC |
| `local_small/` | Smoke local | Run pequeno (seed 42) para debug |
| `kaggle_medium_results.zip` | Arquivo do export | Cópia compactada (opcional; regenerável a partir de `kaggle_medium/`) |

## Arquivos-chave em `kaggle_medium/`

- `meta.json` — setup (seeds, sizes, limiar, Wilcoxon)
- `summary_mean_std.csv` — métricas médias ± std
- `runs_raw.csv` — uma linha por modelo × seed
- `wilcoxon.json` — QSVM pipeline vs logistic (AUPRC)
- `predictions_*.csv` — scores/labels no teste
- `plots/` — curvas PR/ROC, matrizes, etc.
- `features_processed.csv` — tabela de features do experimento (pesado)

## Snapshot (`kaggle_medium`)

- Estação Aotizhongxin · limiar **184** · train/val/test **200/60/80** · **5 seeds**
- Melhor AUPRC médio: **logistic** (0,359) · QSVM pipeline 0,347 · Wilcoxon **não significativo**
- Frontend **Névoa** usa esses números só como narrativa demo (sem inferência live)
