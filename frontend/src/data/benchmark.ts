/**
 * Snapshot do experimento Farooq-style (Kaggle medium).
 * Fonte: artifacts/farooq_style/kaggle_medium/
 * O frontend NÃO executa o modelo — só representa a PoC.
 */
export const DEMO_CITY = {
  name: 'Recife',
  state: 'PE',
  country: 'Brasil',
  lat: -8.0476,
  lon: -34.877,
} as const;

export const POC = {
  protocol: 'farooq_style',
  /** Estação real usada no treino (UCI Beijing). */
  trainingStation: 'Aotizhongxin',
  trainingDataset: 'Beijing Multi-Site (UCI 501)',
  /** Ponto fictício na UI que representa esse treino. */
  demoStation: 'Recife Antigo',
  demoCity: 'Recife, PE',
  artifactPath: 'artifacts/farooq_style/kaggle_medium',
  mode: 'medium',
  seeds: [42, 7, 11, 13, 21],
  thresholdPm25: 184,
  trainSize: 200,
  valSize: 60,
  testSize: 80,
  /** Modelo que a UI apresenta como “em uso” (demo). */
  demoModelId: 'qsvm' as const,
  demoModelLabel: 'QSVM Farooq',
  demoNote:
    'Demonstração em Recife: cenários fictícios inspirados no experimento Beijing. Sem medição nem inferência em tempo real.',
  correlationNote:
    'Correlação ilustrativa — o QSVM foi treinado em Beijing; Recife aparece só como contexto visual da PoC.',
  headline: {
    bestMeanAuprc: 'logistic',
    qsvmVariant: 'Q_farooq_pipeline',
    qsvmAuprcMean: 0.347,
    logisticAuprcMean: 0.359,
    wilcoxonSignificant: false,
  },
} as const;
