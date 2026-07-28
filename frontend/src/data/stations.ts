import type { AqiBand } from './aqi';
import { DEMO_CITY } from './benchmark';

export interface Station {
  id: string;
  name: string;
  district: string;
  lat: number;
  lon: number;
  /** Cenário demo inicial (não é predição ao vivo) */
  demoBand: AqiBand;
  /** PoC QSVM treinada no proxy deste ponto (correlação fictícia) */
  hasTrainedModel: boolean;
}

export { DEMO_CITY };

/**
 * 12 pontos de monitoramento na região metropolitana de Recife (PE).
 * Cenários demo: faixas AQI simuladas, sem medição real.
 * Recife Antigo usa proxy do experimento Beijing/Aotizhongxin (UCI).
 */
export const STATIONS: Station[] = [
  {
    id: 'recife_antigo',
    name: 'Recife Antigo',
    district: 'Centro',
    lat: -8.063,
    lon: -34.871,
    demoBand: 'Unhealthy',
    hasTrainedModel: true,
  },
  {
    id: 'boa_viagem',
    name: 'Boa Viagem',
    district: 'Zona Sul',
    lat: -8.126,
    lon: -34.894,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
  {
    id: 'casa_forte',
    name: 'Casa Forte',
    district: 'Zona Norte',
    lat: -8.038,
    lon: -34.918,
    demoBand: 'Good',
    hasTrainedModel: false,
  },
  {
    id: 'madalena',
    name: 'Madalena',
    district: 'Zona Norte',
    lat: -8.058,
    lon: -34.905,
    demoBand: 'Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'afogados',
    name: 'Afogados',
    district: 'Zona Oeste',
    lat: -8.078,
    lon: -34.918,
    demoBand: 'Unhealthy_Sensitive',
    hasTrainedModel: false,
  },
  {
    id: 'torre',
    name: 'Torre',
    district: 'Zona Norte',
    lat: -8.045,
    lon: -34.895,
    demoBand: 'Very_Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'olinda',
    name: 'Olinda',
    district: 'Região metropolitana',
    lat: -8.009,
    lon: -34.855,
    demoBand: 'Good',
    hasTrainedModel: false,
  },
  {
    id: 'jaboatao',
    name: 'Jaboatão dos Guararapes',
    district: 'Região metropolitana',
    lat: -8.113,
    lon: -34.918,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
  {
    id: 'cidade_universitaria',
    name: 'Cidade Universitária',
    district: 'Zona Oeste',
    lat: -8.051,
    lon: -34.945,
    demoBand: 'Hazardous',
    hasTrainedModel: false,
  },
  {
    id: 'ilha_do_leite',
    name: 'Ilha do Leite',
    district: 'Centro',
    lat: -8.07,
    lon: -34.892,
    demoBand: 'Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'santo_amaro',
    name: 'Santo Amaro',
    district: 'Centro',
    lat: -8.04,
    lon: -34.878,
    demoBand: 'Unhealthy_Sensitive',
    hasTrainedModel: false,
  },
  {
    id: 'pina',
    name: 'Pina',
    district: 'Zona Sul',
    lat: -8.088,
    lon: -34.882,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
];

export const DEFAULT_STATION_ID = 'recife_antigo';

export function stationById(id: string): Station {
  return STATIONS.find((s) => s.id === id) ?? STATIONS[0];
}
