# Névoa — frontend MVP

Dashboard de **Recife (PE)**: mapa, faixas AQI e recomendações.

A UI **representa** o uso do **QSVM Farooq** em um cenário **fictício** — não há medição local nem inferência em tempo real. O treino veio de Beijing (UCI); Recife é só contexto visual da PoC. Snapshot: `artifacts/farooq_style/kaggle_medium/` (`src/data/benchmark.ts`).

```bash
cd frontend
npm install
npm run dev
```

Abra a URL do Vite (geralmente `http://localhost:5173`).

## GitHub Pages

Build de produção (base `/qml-air-quality/`):

```bash
VITE_BASE_PATH=/qml-air-quality/ npm run build
```

Publicado em: https://alanveloso.github.io/qml-air-quality/
