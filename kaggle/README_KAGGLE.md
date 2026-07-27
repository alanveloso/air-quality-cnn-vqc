# Rodar a ablação no Kaggle

Cada configuração de 2 qubits levou ~13 min aqui (CPU). As 10 configs + 4 qubits devem passar de **2–3 h**. No Kaggle (sessão longa / CPU) dá para deixar rodando sem travar a máquina.

## O que já está pronto

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

## Passo a passo no Kaggle

1. **New Dataset** → upload `qml_air_quality_poc.zip` (nome sugerido: `qml-air-quality-poc`).
   - O Kaggle **extrai o zip automaticamente**. No notebook você verá pastas `src/` e `config/`, não o `.zip`. Isso é esperado.
2. (Opcional) Outro dataset com `ablation_kernel_cache_q01_q03.zip`.
3. **New Notebook** → **Add Data** → anexe o(s) dataset(s).
4. Use o notebook atualizado `notebooks/kaggle_quantum_ablation.ipynb` (detecta zip **ou** pastas já extraídas).
5. Settings: **Internet ON** · Accelerator **None**.
6. Rode a 1ª célula e confira `Datasets em /kaggle/input:` + `OK código em ...`.
7. Run All. No fim baixe `ablation_results.zip`.

### Se der erro "não achei o código"

Na 1ª célula, o print lista o que há em `/kaggle/input`. Confirme que o dataset está em **Input** (barra direita). Se o dataset tiver outro layout, diga o que apareceu no print.

## Depois do Kaggle

Copie de volta para o repo:

```bash
# exemplo
unzip ~/Downloads/ablation_kaggle_output.zip -d artifacts/ablation/
```

Ou rode localmente só o relatório se os CSVs/kernels já estiverem em `artifacts/ablation/`:

```bash
python -c "from qml_air_quality.reporting.ablation_report import write_ablation_report, plot_ablation_figures as p; print(write_ablation_report()); print(p())"
```
