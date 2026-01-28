from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .utils import CACHE_DIR, FIGURES_DIR


# =========================
# Geocoding (LocationIQ via requests + cache; optional Nominatim fallback)
# =========================
@dataclass(frozen=True)
class GeocodeConfig:
    cache_csv: Path = CACHE_DIR / "geocode_cache.csv"

    # Provider selection:
    # - "locationiq" (default, via requests)
    # - "nominatim" (via geopy, optional fallback)
    provider: str = os.getenv("IAEA_GEOCODER_PROVIDER", "locationiq").lower()

    # LocationIQ (requests)
    locationiq_api_key: str = os.getenv("LOCATIONIQ_API_KEY", "")
    locationiq_domain: str = os.getenv("LOCATIONIQ_DOMAIN", "api.locationiq.com")

    # Nominatim (only used if provider="nominatim")
    # Needs descriptive UA with contact
    user_agent: str = os.getenv(
        "IAEA_GEOCODER_UA",
        "CNM_CycNucMed/1.0 (contact: your_email@example.com)"
    )

    # pacing / networking
    min_delay_seconds: float = float(os.getenv("IAEA_GEOCODER_MIN_DELAY", "0.2"))
    timeout_seconds: int = int(os.getenv("IAEA_GEOCODER_TIMEOUT", "15"))
    max_retries: int = int(os.getenv("IAEA_GEOCODER_RETRIES", "2"))
    backoff_base_seconds: float = float(os.getenv("IAEA_GEOCODER_BACKOFF_BASE", "1.0"))

    # LocationIQ-specific: when true, store the raw "display_name" if present
    keep_display_name: bool = os.getenv("IAEA_GEOCODER_KEEP_DISPLAY_NAME", "1").strip() not in {"0", "false", "no"}


def geocode_country_cities(df: pd.DataFrame, country: str, config: GeocodeConfig = GeocodeConfig()):
    """Return list of (city, lat, lon) for all cities in a country, using a persistent CSV cache.

    - cache-first: never hits network if we already have lat/lon
    - avoids FutureWarning: no pd.concat for appends
    - robust: retries + exponential backoff + polite min delay
    - LocationIQ uses pure requests (no geopy dependency)
    """
    import requests

    cache_path = Path(config.cache_csv)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["Country_iso3", "Country", "City", "lat", "lon", "display_name"]
    if cache_path.exists():
        geo_cache = pd.read_csv(cache_path)
    else:
        geo_cache = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    # Ensure required columns exist
    for c in cols:
        if c not in geo_cache.columns:
            geo_cache[c] = pd.Series(dtype="object")

    # Normalize types
    for c in ["Country_iso3", "Country", "City", "display_name"]:
        geo_cache[c] = geo_cache[c].astype("string")

    def _norm(x) -> str:
        return str(x).strip().lower()

    def _append_cache_row(row: dict):
        """Append one row without pandas concat (avoids FutureWarning)."""
        nonlocal geo_cache
        geo_cache.loc[len(geo_cache)] = {
            "Country_iso3": row.get("Country_iso3", "") or "",
            "Country": row.get("Country", "") or "",
            "City": row.get("City", "") or "",
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "display_name": row.get("display_name"),
        }
        geo_cache.to_csv(cache_path, index=False)

    def _lookup_cache(country_: str, city_: str, country_iso3: Optional[str]):
        """Return (lat,lon) if present, else (None,None)."""
        if country_iso3:
            hit = geo_cache[
                (geo_cache["Country_iso3"].apply(_norm) == _norm(country_iso3))
                & (geo_cache["City"].apply(_norm) == _norm(city_))
            ]
            if not hit.empty and pd.notna(hit.iloc[0]["lat"]) and pd.notna(hit.iloc[0]["lon"]):
                return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"])

        hit = geo_cache[
            (geo_cache["Country"].apply(_norm) == _norm(country_))
            & (geo_cache["City"].apply(_norm) == _norm(city_))
        ]
        if not hit.empty and pd.notna(hit.iloc[0]["lat"]) and pd.notna(hit.iloc[0]["lon"]):
            return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"])

        return None, None

    # ----------------------------
    # Provider: LocationIQ via requests
    # ----------------------------
    last_call_ts = 0.0

    def _rate_limit_sleep():
        nonlocal last_call_ts
        now = time.time()
        elapsed = now - last_call_ts
        if elapsed < config.min_delay_seconds:
            time.sleep(config.min_delay_seconds - elapsed)
        last_call_ts = time.time()

    def _locationiq_geocode(query: str) -> Optional[Tuple[float, float, Optional[str]]]:
        """Return (lat, lon, display_name) or None."""
        if not config.locationiq_api_key:
            raise RuntimeError("LOCATIONIQ_API_KEY is not set but provider=locationiq")

        base = f"https://{config.locationiq_domain}/v1/search"
        params = {
            "key": config.locationiq_api_key,
            "q": query,
            "format": "json",
            "limit": 1,
        }

        # Retries w/ backoff for transient issues + 429
        for attempt in range(config.max_retries + 1):
            _rate_limit_sleep()
            try:
                r = requests.get(base, params=params, timeout=config.timeout_seconds)
                # Handle rate limit explicitly
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(float(retry_after))
                        except Exception:
                            pass
                    # then backoff
                    if attempt < config.max_retries:
                        time.sleep(config.backoff_base_seconds * (2**attempt))
                        continue
                    return None

                if r.status_code >= 400:
                    # Other errors: backoff and retry
                    if attempt < config.max_retries:
                        time.sleep(config.backoff_base_seconds * (2**attempt))
                        continue
                    return None

                data = r.json()
                if not data:
                    return None

                item = data[0]
                lat = float(item["lat"])
                lon = float(item["lon"])
                display_name = item.get("display_name") if config.keep_display_name else None
                return lat, lon, display_name

            except Exception:
                if attempt < config.max_retries:
                    time.sleep(config.backoff_base_seconds * (2**attempt))
                    continue
                return None

        return None

    # ----------------------------
    # Optional provider: Nominatim via geopy (only if you explicitly choose it)
    # ----------------------------
    def _nominatim_geocode(query: str):
        # Only import geopy if you really use it
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent=config.user_agent, timeout=config.timeout_seconds)
        for attempt in range(config.max_retries + 1):
            try:
                _rate_limit_sleep()
                loc = geolocator.geocode(query)
                if loc is not None:
                    return float(loc.latitude), float(loc.longitude), getattr(loc, "address", None)
            except Exception:
                pass
            if attempt < config.max_retries:
                time.sleep(config.backoff_base_seconds * (2**attempt))
        return None

    # ----------------------------
    # Unified geocode wrapper
    # ----------------------------
    provider = (config.provider or "locationiq").lower()

    def _geocode(query: str):
        if provider == "locationiq":
            return _locationiq_geocode(query)
        if provider == "nominatim":
            return _nominatim_geocode(query)
        raise ValueError(f"Unknown geocoder provider: {provider}")

    def get_latlon(country_: str, city_: str, country_iso3: Optional[str] = None):
        country_ = str(country_).strip()
        city_ = str(city_).strip()
        country_iso3 = None if country_iso3 is None else str(country_iso3).strip()

        # 1) Cache-first
        lat, lon = _lookup_cache(country_, city_, country_iso3)
        if lat is not None and lon is not None:
            return lat, lon

        # 2) Queries: structured, reduce ambiguity and calls
        queries = [f"{city_}, {country_}"]
        if country_iso3:
            queries.append(f"{city_}, {country_iso3}")

        res = None
        for q in queries:
            res = _geocode(q)
            if res is not None:
                break

        # 3) Cache miss -> store miss too (avoid hammering)
        if res is None:
            _append_cache_row(
                {
                    "Country_iso3": country_iso3 or "",
                    "Country": country_,
                    "City": city_,
                    "lat": None,
                    "lon": None,
                    "display_name": None,
                }
            )
            return None, None

        # 4) Cache hit -> store coordinates
        lat, lon, display_name = res
        _append_cache_row(
            {
                "Country_iso3": country_iso3 or "",
                "Country": country_,
                "City": city_,
                "lat": lat,
                "lon": lon,
                "display_name": display_name,
            }
        )
        return lat, lon

    # --- Build the list of unique cities in the given country
    sub = df[df["Country"].str.lower() == str(country).strip().lower()].copy()
    cities = sorted([c for c in sub["City"].dropna().unique() if str(c).strip()])

    iso3 = None
    if "Country_iso3" in sub.columns:
        iso3_vals = sub["Country_iso3"].dropna().unique()
        iso3 = iso3_vals[0] if len(iso3_vals) else None

    pts = []
    for city in cities:
        lat, lon = get_latlon(country, city, country_iso3=iso3)
        if lat is not None and lon is not None:
            pts.append((city, lat, lon))
    return pts


# =========================
# Plotting map (unchanged)
# =========================
def save_country_map(df: pd.DataFrame, country: str, out_path: Optional[Path] = None) -> Optional[Path]:
    """Create a country map with city points and (optional) population density raster.

    Requires: geopandas, shapely, rasterio, matplotlib.
    If dependencies are missing or no points can be geocoded, returns None.
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        from rasterio.mask import mask
        from matplotlib.colors import LogNorm
        from shapely.geometry import Point
    except Exception:
        return None

    pts = geocode_country_cities(df, country)
    if not pts:
        return None

    sub = df[df["Country"].str.lower() == str(country).strip().lower()].copy()
    iso3 = None
    if "Country_iso3" in sub.columns:
        v = sub["Country_iso3"].dropna().unique()
        iso3 = v[0] if len(v) else None

    # Natural Earth polygons (remote zip)
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    )

    country_poly = None
    if iso3 and "ISO_A3" in world.columns:
        cp = world[world["ISO_A3"] == iso3].copy()
        if not cp.empty:
            country_poly = cp

    if country_poly is None:
        cp = world[world["NAME"].str.lower() == str(country).strip().lower()].copy()
        if not cp.empty:
            country_poly = cp

    out_path = out_path or (FIGURES_DIR / f"{country}_map.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_pts = gpd.GeoDataFrame(
        [{"City": c, "geometry": Point(lon, lat)} for c, lat, lon in pts], crs="EPSG:4326"
    )

    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()

    # Optional raster background (public mirror). If it fails, we still plot polygons + points.
    gpw_url = "https://pacific-data.sprep.org/system/files/Global_2020_PopulationDensity30sec_GPWv4.tiff"

    raster_ok = False
    if country_poly is not None:
        try:
            with rasterio.open("/vsicurl/" + gpw_url) as src:
                geom = [country_poly.geometry.iloc[0]]
                out_img, out_transform = mask(src, geom, crop=True)
                arr = out_img[0].astype(float)
                arr[arr <= 0] = np.nan

                ax.imshow(
                    arr,
                    origin="upper",
                    norm=LogNorm(vmin=np.nanpercentile(arr, 10), vmax=np.nanpercentile(arr, 99)),
                    extent=(
                        out_transform[2],
                        out_transform[2] + out_transform[0] * arr.shape[1],
                        out_transform[5] + out_transform[4] * arr.shape[0],
                        out_transform[5],
                    ),
                )
                raster_ok = True
        except Exception:
            raster_ok = False

    # Country boundary
    if country_poly is not None:
        country_poly.boundary.plot(ax=ax, linewidth=1)

    # Points
    gdf_pts.plot(
        ax=ax,
        markersize=30,
        color="red",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    title = f"{country} – cyclotron cities"
    if raster_ok:
        title += " (pop density background)"

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path
