import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { BANDS, MODEL_LABEL, bandById, type AqiBand } from './data/aqi';
import { DEFAULT_STATION_ID, STATIONS, stationById, type Station } from './data/stations';
import './App.css';

function FlyTo({ station }: { station: Station }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo([station.lat, station.lon], 11, { duration: 0.8 });
  }, [map, station.lat, station.lon]);
  return null;
}

export default function App() {
  const [stationId, setStationId] = useState(DEFAULT_STATION_ID);
  const station = stationById(stationId);
  const [bandId, setBandId] = useState<AqiBand>(station.demoBand);
  const [panelOpen, setPanelOpen] = useState(true);
  const band = bandById(bandId);

  useEffect(() => {
    setBandId(station.demoBand);
  }, [station.id, station.demoBand]);

  useEffect(() => {
    document.documentElement.style.setProperty('--accent', band.color);
  }, [band]);

  return (
    <div className="map-app">
      <MapContainer
        center={[39.95, 116.4]}
        zoom={10}
        className="map-full"
        scrollWheelZoom
        zoomControl={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution="&copy; OSM &copy; CARTO"
        />
        <FlyTo station={station} />
        {STATIONS.map((s) => {
          const b = bandById(s.id === station.id ? bandId : s.demoBand);
          const active = s.id === station.id;
          return (
            <CircleMarker
              key={s.id}
              center={[s.lat, s.lon]}
              radius={active ? 16 : 10}
              eventHandlers={{
                click: () => {
                  setStationId(s.id);
                  setPanelOpen(true);
                },
              }}
              pathOptions={{
                color: active ? '#f4efe6' : 'rgba(244,239,230,0.35)',
                fillColor: b.color,
                fillOpacity: active ? 0.95 : 0.72,
                weight: active ? 3 : 1.25,
              }}
            >
              <Popup>
                <strong>{s.name}</strong>
                <br />
                {s.district} · {b.label}
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Brand overlay — leve, não compete com o mapa */}
      <header className="overlay-brand">
        <p className="brand">Névoa</p>
        <p className="brand-sub">Beijing · 12 estações</p>
      </header>

      {/* Status flutuante */}
      <div className="overlay-status" style={{ borderColor: band.color }}>
        <p className="overlay-status__place">
          {station.name}
          <span>{station.district}</span>
        </p>
        <p className="overlay-status__band" style={{ color: band.color }}>
          {band.label}
        </p>
        <p className="overlay-status__meta">
          PM2.5 {band.pm25} · t+24h
          {station.hasTrainedModel ? ' · modelo PoC' : ' · demo'}
        </p>
      </div>

      {/* Seletor de estação */}
      <div className="overlay-stations">
        <label>
          <span>Estação</span>
          <select
            value={stationId}
            onChange={(e) => {
              setStationId(e.target.value);
              setPanelOpen(true);
            }}
          >
            {STATIONS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.district}){s.hasTrainedModel ? ' ★' : ''}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Painel de recomendações — drawer sobre o mapa */}
      <aside className={`overlay-panel ${panelOpen ? 'is-open' : ''}`} aria-hidden={!panelOpen}>
        <button
          type="button"
          className="panel-toggle"
          onClick={() => setPanelOpen((v) => !v)}
          aria-expanded={panelOpen}
        >
          {panelOpen ? 'Fechar' : 'Recomendações'}
        </button>

        {panelOpen && (
          <div className="panel-body">
            <p className="eyebrow">O que fazer · 24 h</p>
            <h2>{band.label}</h2>
            <p className="lede">{band.summary}</p>

            <div className="model-chip">
              <span>Modelo sugerido</span>
              <strong>{MODEL_LABEL[band.preferredModel]}</strong>
              <small>{band.modelWhy}</small>
            </div>

            <div className="advice-grid">
              <div>
                <h3>Faça</h3>
                <ul>
                  {band.actions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h3>Evite</h3>
                <ul>
                  {band.avoid.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="band-switch" role="tablist" aria-label="Simular faixa">
              {BANDS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  role="tab"
                  aria-selected={b.id === band.id}
                  className={b.id === band.id ? 'is-active' : undefined}
                  style={{ ['--band' as string]: b.color }}
                  onClick={() => setBandId(b.id)}
                >
                  {b.label}
                </button>
              ))}
            </div>
            <p className="fineprint">
              Clique nos pontos do mapa. Cenários demo — só Aotizhongxin tem modelo treinado.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
