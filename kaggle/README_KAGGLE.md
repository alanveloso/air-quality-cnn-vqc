# Rodar no Kaggle

## Opção A — Notebook independente (recomendado)

[`notebooks/kaggle_farooq_style_qsvm.ipynb`](../notebooks/kaggle_farooq_style_qsvm.ipynb)

- **Não precisa** anexar zip do repositório
- Baixa Beijing (UCI) sozinho
- Features estilo Farooq (min/max/median/variance de PM2.5 e temperatura)
- Clássicos + QSVM (PCA→2 + MinMax + ZZFeatureMap)
- Settings: Internet **ON** · Accelerator **None**
- Amostra padrão pequena (`80/40/40`); aumente na célula de config

1. New Notebook → Upload `kaggle_farooq_style_qsvm.ipynb`
2. Run All
3. Baixe `farooq_style_kaggle_results.zip` em Output
4. Extraia para `artifacts/farooq_style/kaggle_medium/` (ver [`artifacts/farooq_style/README.md`](../artifacts/farooq_style/README.md))

## Opção B — Ablação Q01–Q10 (pacote do repo)

Cada configuração de 2 qubits levou ~13 min aqui (CPU). As 10 configs + 4 qubits devem passar de **2–3 h**. No Kaggle (sessão longa / CPU) dá para deixar rodando sem travar a máquina.

### O que já está pronto

| Arquivo | Uso |
|---|---|
| [`kaggle/qml_air_quality_poc.zip`](qml_air_quality_poc.zip) | Código + configs (~35 KB) |
| [`kaggle/ablation_kernel_cache_q01_q03.zip`](ablation_kernel_cache_q01_q03.zip) | Kernels Q01–Q03 já calculados (~6 MB) — opcional, economiza ~40 min |
| [`notebooks/kaggle_quantum_ablation.ipynb`](../notebooks/kaggle_quantum_ablation.ipynb) | Notebook para colar/upload no Kaggle |

Resultados parciais locais (validação):

| ID | Escala | AUPRC val | Alignment | ΔK |
|---|---|---:|---:|---:|
| Q01 | none | 0.135 | 0.481 | 0.002 |
| Q02 | [0,1] | 0.317 | 0.544 | 0.021 |
| Q03 | [0,π] | **0.337** | 0.519 | 0.064 |

A escala angular já está ajudando vs o controle sem escala.

### Passo a passo (ablacão)

1. **New Dataset** → upload `qml_air_quality_poc.zip` (nome sugerido: `qml-air-quality-poc`).
   - O Kaggle **extrai o zip automaticamente**. No notebook você verá pastas `src/` e `config/`, não o `.zip`. Isso é esperado.
2. (Opcional) Outro dataset com `ablation_kernel_cache_q01_q03.zip`.
3. **New Notebook** → **Add Data** → anexe o(s) dataset(s).
4. Use o notebook `notebooks/kaggle_quantum_ablation.ipynb`.
5. Settings: **Internet ON** · Accelerator **None**.
6. Rode a 1ª célula e confira `Datasets em /kaggle/input:` + `OK código em ...`.
7. Run All. No fim baixe `ablation_results.zip`.

### Se der erro "não achei o código"

Na 1ª célula, o print lista o que há em `/kaggle/input`. Confirme que o dataset está em **Input** (barra direita).

## Depois do Kaggle

```bash
unzip ~/Downloads/farooq_style_kaggle_results.zip -d artifacts/farooq_style/
# ou, para a ablação:
unzip ~/Downloads/ablation_kaggle_output.zip -d artifacts/ablation/
```
