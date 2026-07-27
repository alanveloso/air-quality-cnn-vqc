export type AqiBand =
  | 'Good'
  | 'Moderate'
  | 'Unhealthy_Sensitive'
  | 'Unhealthy'
  | 'Very_Unhealthy'
  | 'Hazardous';

export type ModelId = 'qsvm' | 'logreg' | 'svm_rbf';

export interface BandInfo {
  id: AqiBand;
  label: string;
  pm25: string;
  color: string;
  tone: string;
  summary: string;
  actions: string[];
  avoid: string[];
  /** Modelo sugerido nesta PoC (honesto, com base nos smokes locais) */
  preferredModel: ModelId;
  modelWhy: string;
}

export const STATION = {
  name: 'Aotizhongxin',
  city: 'Beijing',
  lat: 39.982,
  lon: 116.397,
};

export const MODEL_LABEL: Record<ModelId, string> = {
  qsvm: 'QSVM Farooq',
  logreg: 'Regressão logística',
  svm_rbf: 'SVM RBF',
};

/** Faixas EPA (µg/m³) + recomendações práticas */
export const BANDS: BandInfo[] = [
  {
    id: 'Good',
    label: 'Bom',
    pm25: '≤ 12',
    color: '#2f6f4e',
    tone: 'linear-gradient(145deg, #d8efe3 0%, #8fbfa3 45%, #3d7a5c 100%)',
    summary: 'Ar adequado para a maioria das pessoas.',
    actions: [
      'Atividades ao ar livre liberadas',
      'Janelas podem permanecer abertas',
      'Bom momento para exercício externo',
    ],
    avoid: ['Não há restrições relevantes'],
    preferredModel: 'svm_rbf',
    modelWhy: 'No smoke Good vs Moderate, SVM RBF teve melhor AUPRC.',
  },
  {
    id: 'Moderate',
    label: 'Moderado',
    pm25: '12,1 – 35,4',
    color: '#b8860b',
    tone: 'linear-gradient(145deg, #f3e6c0 0%, #d4b45a 50%, #9a7b20 100%)',
    summary: 'Aceitável; grupos sensíveis devem se atentar.',
    actions: [
      'Rotina normal para a maioria',
      'Grupos sensíveis: reduzir esforço prolongado ao ar livre',
      'Acompanhar a previsão das próximas 24 h',
    ],
    avoid: ['Exercício intenso prolongado se houver sintomas respiratórios'],
    preferredModel: 'svm_rbf',
    modelWhy: 'Mesma faixa binária Good/Moderate — SVM RBF foi o melhor clássico no smoke.',
  },
  {
    id: 'Unhealthy_Sensitive',
    label: 'Ruim p/ sensíveis',
    pm25: '35,5 – 55,4',
    color: '#c45c26',
    tone: 'linear-gradient(145deg, #f0d0bc 0%, #d9844f 50%, #a34a1c 100%)',
    summary: 'Crianças, idosos e asmáticos devem limitar exposição.',
    actions: [
      'Reduzir tempo ao ar livre para grupos sensíveis',
      'Preferir atividades em ambientes fechados filtrados',
      'Usar máscara PFF2/N95 em trajetos longos',
    ],
    avoid: ['Corrida/treino intenso ao ar livre para sensíveis'],
    preferredModel: 'qsvm',
    modelWhy: 'No alvo aqi_bad (pior que Moderado), QSVM liderou AUPRC no smoke.',
  },
  {
    id: 'Unhealthy',
    label: 'Não saudável',
    pm25: '55,5 – 150,4',
    color: '#b33a3a',
    tone: 'linear-gradient(145deg, #efc4c4 0%, #d16666 45%, #8f2a2a 100%)',
    summary: 'Toda a população deve reduzir esforço ao ar livre.',
    actions: [
      'Ficar mais tempo em ambientes fechados',
      'Máscara PFF2/N95 ao sair',
      'Manter janelas fechadas; usar purificador se houver',
    ],
    avoid: ['Exercício ao ar livre', 'Exposição prolongada em vias congestionadas'],
    preferredModel: 'qsvm',
    modelWhy: 'QSVM teve melhor AUPRC no alvo aqi_bad (faixas acima de Moderado).',
  },
  {
    id: 'Very_Unhealthy',
    label: 'Muito não saudável',
    pm25: '150,5 – 250,4',
    color: '#7a2e3a',
    tone: 'linear-gradient(145deg, #e8c8ce 0%, #b35464 45%, #5c1c28 100%)',
    summary: 'Alerta sanitário: minimize saída e proteja vias respiratórias.',
    actions: [
      'Evitar sair sem necessidade',
      'Máscara PFF2/N95 obrigatória ao ar livre',
      'Ambientes fechados com filtragem; hidratação',
    ],
    avoid: ['Qualquer exercício externo', 'Abrir janelas por longos períodos'],
    preferredModel: 'qsvm',
    modelWhy: 'Sobreposição com extremos de PM2.5 — QSVM liderou AUPRC no extremo P90.',
  },
  {
    id: 'Hazardous',
    label: 'Perigoso',
    pm25: '> 250,4',
    color: '#5c1a1a',
    tone: 'linear-gradient(145deg, #d4a8a8 0%, #8b3a3a 40%, #3a1010 100%)',
    summary: 'Emergência de qualidade do ar. Permaneça protegido.',
    actions: [
      'Permaneça em casa com janelas fechadas',
      'Só sair com máscara PFF2/N95 e pelo menor tempo',
      'Procure orientação de saúde se houver falta de ar',
    ],
    avoid: ['Toda atividade externa', 'Ventilação natural sem filtragem'],
    preferredModel: 'qsvm',
    modelWhy: 'Alinha com extremos (≥ P90 ~184 µg/m³) — QSVM melhor AUPRC no smoke.',
  },
];

export function bandById(id: AqiBand): BandInfo {
  return BANDS.find((b) => b.id === id) ?? BANDS[0];
}
