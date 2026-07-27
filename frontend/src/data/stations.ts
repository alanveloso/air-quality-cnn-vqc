import type { AqiBand } from './aqi';

export interface Station {
  id: string;
  name: string;
  district: string;
  lat: number;
  lon: number;
  /** Cenário demo inicial (não é predição ao vivo) */
  demoBand: AqiBand;
  /** PoC QML treinada nesta estação */
  hasTrainedModel: boolean;
}

/**
 * 12 estações do Beijing Multi-Site (UCI 501).
 * Coordenadas: Zhang et al. / literatura do dataset.
 */
export const STATIONS: Station[] = [
  {
    id: 'aotizhongxin',
    name: 'Aotizhongxin',
    district: 'Chaoyang',
    lat: 39.982,
    lon: 116.397,
    demoBand: 'Unhealthy',
    hasTrainedModel: true,
  },
  {
    id: 'changping',
    name: 'Changping',
    district: 'Changping',
    lat: 40.217,
    lon: 116.23,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
  {
    id: 'dingling',
    name: 'Dingling',
    district: 'Changping',
    lat: 40.292,
    lon: 116.22,
    demoBand: 'Good',
    hasTrainedModel: false,
  },
  {
    id: 'dongsi',
    name: 'Dongsi',
    district: 'Dongcheng',
    lat: 39.929,
    lon: 116.417,
    demoBand: 'Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'guanyuan',
    name: 'Guanyuan',
    district: 'Xicheng',
    lat: 39.929,
    lon: 116.339,
    demoBand: 'Unhealthy_Sensitive',
    hasTrainedModel: false,
  },
  {
    id: 'gucheng',
    name: 'Gucheng',
    district: 'Shijingshan',
    lat: 39.914,
    lon: 116.184,
    demoBand: 'Very_Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'huairou',
    name: 'Huairou',
    district: 'Huairou',
    lat: 40.328,
    lon: 116.628,
    demoBand: 'Good',
    hasTrainedModel: false,
  },
  {
    id: 'nongzhanguan',
    name: 'Nongzhanguan',
    district: 'Chaoyang',
    lat: 39.937,
    lon: 116.461,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
  {
    id: 'shunyi',
    name: 'Shunyi',
    district: 'Shunyi',
    lat: 40.127,
    lon: 116.655,
    demoBand: 'Moderate',
    hasTrainedModel: false,
  },
  {
    id: 'tiantan',
    name: 'Tiantan',
    district: 'Dongcheng',
    lat: 39.886,
    lon: 116.407,
    demoBand: 'Unhealthy',
    hasTrainedModel: false,
  },
  {
    id: 'wanliu',
    name: 'Wanliu',
    district: 'Haidian',
    lat: 39.987,
    lon: 116.287,
    demoBand: 'Unhealthy_Sensitive',
    hasTrainedModel: false,
  },
  {
    id: 'wanshouxigong',
    name: 'Wanshouxigong',
    district: 'Xicheng',
    lat: 39.878,
    lon: 116.352,
    demoBand: 'Hazardous',
    hasTrainedModel: false,
  },
];

export const DEFAULT_STATION_ID = 'aotizhongxin';

export function stationById(id: string): Station {
  return STATIONS.find((s) => s.id === id) ?? STATIONS[0];
}
