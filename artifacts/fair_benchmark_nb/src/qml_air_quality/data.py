"""Download and load Beijing Multi-Site Air Quality (UCI)."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from qml_air_quality.config import project_root

logger = logging.getLogger(__name__)

UCI_ZIP_URLS = [
    # Prefer the historical PRSA archive (full per-station CSVs).
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip",
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
]


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_bytes(urls: list[str]) -> tuple[bytes, str]:
    last_err: Exception | None = None
    for url in urls:
        try:
            logger.info("Downloading %s", url)
            with urlopen(url, timeout=120) as resp:
                data = resp.read()
            if data:
                return data, url
        except Exception as exc:  # noqa: BLE001 — try next mirror
            last_err = exc
            logger.warning("Download failed for %s: %s", url, exc)
    raise RuntimeError(f"Could not download UCI dataset: {last_err}")


def _extract_station_csv(zip_bytes: bytes, station: str) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        match = [n for n in members if station.lower() in Path(n).name.lower()]
        if not match:
            raise FileNotFoundError(
                f"No CSV for station={station!r} in zip. Available: {members}"
            )
        with zf.open(match[0]) as f:
            return pd.read_csv(f)


def download_dataset(
    dataset_id: int = 501,
    station: str = "Aotizhongxin",
    raw_dir: str | Path | None = None,
) -> Path:
    """Download UCI zip (if needed) and save station CSV to data/raw.

    Note: ucimlrepo does not expose dataset 501 for Python import, so we
    fetch the official zip mirrors directly.
    """
    _ = dataset_id  # kept for config compatibility
    root = project_root()
    raw = Path(raw_dir) if raw_dir else root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    out_csv = raw / f"{station.lower()}_air_quality.csv"
    meta_path = raw / f"{station.lower()}_download_meta.json"
    zip_path = raw / "beijing_multi_site_air_quality.zip"

    if out_csv.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") == _checksum(out_csv):
            logger.info("Raw data already present and checksum OK: %s", out_csv)
            return out_csv

    def _zip_has_station(data: bytes) -> bool:
        try:
            with zipfile.ZipFile(BytesIO(data)) as zf:
                return any(
                    station.lower() in Path(n).name.lower() and n.lower().endswith(".csv")
                    for n in zf.namelist()
                )
        except zipfile.BadZipFile:
            return False

    zip_bytes: bytes | None = None
    source = ""
    if zip_path.exists():
        candidate = zip_path.read_bytes()
        if _zip_has_station(candidate):
            zip_bytes = candidate
            source = str(zip_path)
        else:
            logger.warning("Cached zip missing station CSV; re-downloading")

    if zip_bytes is None:
        zip_bytes, source = _download_bytes(UCI_ZIP_URLS)
        if not _zip_has_station(zip_bytes):
            # Try remaining mirrors explicitly if first zip is the wrong bundle
            for url in UCI_ZIP_URLS:
                if url == source:
                    continue
                try:
                    with urlopen(url, timeout=120) as resp:
                        alt = resp.read()
                    if _zip_has_station(alt):
                        zip_bytes, source = alt, url
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Mirror failed %s: %s", url, exc)
        zip_path.write_bytes(zip_bytes)

    station_df = _extract_station_csv(zip_bytes, station)
    if station_df.empty:
        raise ValueError(f"No rows for station={station!r}")

    station_df.to_csv(out_csv, index=False)
    meta: dict[str, Any] = {
        "dataset_id": dataset_id,
        "station": station,
        "source": source,
        "n_rows": len(station_df),
        "sha256": _checksum(out_csv),
        "path": str(out_csv.relative_to(root)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved %s rows to %s", len(station_df), out_csv)
    return out_csv


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Build monotonic timestamp from year/month/day/hour."""
    out = df.copy()
    required = ["year", "month", "day", "hour"]
    rename = {}
    for c in required:
        for col in out.columns:
            if col.lower() == c and col != c:
                rename[col] = c
    if rename:
        out = out.rename(columns=rename)
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing time columns: {missing}")

    out["timestamp"] = pd.to_datetime(
        {
            "year": out["year"].astype(int),
            "month": out["month"].astype(int),
            "day": out["day"].astype(int),
            "hour": out["hour"].astype(int),
        }
    )
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    out = out.reset_index(drop=True)
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp is not monotonic increasing after sort")
    return out


def load_station_csv(path: str | Path) -> pd.DataFrame:
    """Load station CSV and attach timestamp."""
    df = pd.read_csv(path)
    return build_timestamp(df)


def missing_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column missing counts and rates."""
    rows = []
    n = len(df)
    for c in columns:
        if c not in df.columns:
            continue
        miss = int(df[c].isna().sum())
        rows.append({"column": c, "missing": miss, "rate": miss / n if n else 0.0})
    return pd.DataFrame(rows)
