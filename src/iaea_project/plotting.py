from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import CACHE_DIR, FIGURES_DIR


@dataclass(frozen=True)
class GeocodeConfig:
    cache_csv: Path = CACHE_DIR / "geocode_cache.csv"
    user_agent: str = "iaea_cyclotron_geocoder"
    min_delay_seconds: float = 1.0


def geocode_country_cities(df: pd.DataFrame, country: str, config: GeocodeConfig = GeocodeConfig()):
    """Return list of (city, lat, lon) for all cities in a country, using a persistent CSV cache."""
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    cache_path = Path(config.cache_csv)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        geo_cache = pd.read_csv(cache_path)
    else:
        geo_cache = pd.DataFrame(columns=["Country_iso3", "Country", "City", "lat", "lon", "display_name"])

    for col in ["Country_iso3", "Country", "City"]:
        if col not in geo_cache.columns:
            geo_cache[col] = ""
        geo_cache[col] = geo_cache[col].astype(str)

    geolocator = Nominatim(user_agent=config.user_agent)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=config.min_delay_seconds, swallow_exceptions=True)

    def _norm(x):
        return str(x).strip().lower()

    def get_latlon(country_: str, city_: str, country_iso3: Optional[str] = None):
        nonlocal geo_cache
        country_ = str(country_).strip()
        city_ = str(city_).strip()
        country_iso3 = None if country_iso3 is None else str(country_iso3).strip()

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

        queries = [f"{city_}, {country_}"]
        if country_iso3:
            queries.append(f"{city_}, {country_iso3}")
        queries.append(city_)

        loc = None
        for q in queries:
            loc = geocode(q)
            if loc is not None:
                break

        if loc is None:
            new_row = {
                "Country_iso3": country_iso3 or "",
                "Country": country_,
                "City": city_,
                "lat": None,
                "lon": None,
                "display_name": None,
            }
            geo_cache = pd.concat([geo_cache, pd.DataFrame([new_row])], ignore_index=True)
            geo_cache.to_csv(cache_path, index=False)
            return None, None

        lat, lon = loc.latitude, loc.longitude
        display_name = getattr(loc, "address", None)

        new_row = {
            "Country_iso3": country_iso3 or "",
            "Country": country_,
            "City": city_,
            "lat": lat,
            "lon": lon,
            "display_name": display_name,
        }

        geo_cache = pd.concat([geo_cache, pd.DataFrame([new_row])], ignore_index=True)
        geo_cache.to_csv(cache_path, index=False)
        return lat, lon

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
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
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
    gdf_pts.plot(ax=ax, markersize=30)

    title = f"{country} – cyclotron cities"
    if raster_ok:
        title += " (pop density background)"

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path
