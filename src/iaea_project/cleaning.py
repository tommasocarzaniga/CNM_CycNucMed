from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from rapidfuzz import fuzz

from .utils import CACHE_DIR

# -----------------------------
# Country canonicalization (country_converter)
# -----------------------------


@dataclass(frozen=True)
class CountryCanonConfig:
    cache_path: Path = CACHE_DIR / "country_fix_cache.json"


def canonicalize_countries(
    df: pd.DataFrame,
    col: str = "Country",
    out_col: Optional[str] = None,
    config: CountryCanonConfig = CountryCanonConfig(),
    llm_fix: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Canonicalize country names to country_converter's `name_short`.

    - Deterministic conversion first.
    - Optional LLM fallback only for values that still fail.
    - Adds Country_iso3 column (ISO3) for mapping.

    Parameters
    ----------
    llm_fix:
        Optional function raw_country -> fixed_country_name (short form).
        If omitted, unresolved values stay as-is.
    """
    import country_converter as coco

    df = df.copy()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(config.cache_path)
    if cache_path.exists():
        fix_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        fix_cache = {}

    allowed = set(coco.CountryConverter().data["name_short"].dropna().astype(str))

    def cache_or_fix(raw: str) -> str:
        raw = str(raw).strip()
        if raw in fix_cache:
            return fix_cache[raw]
        if llm_fix is None:
            fix_cache[raw] = raw
            cache_path.write_text(json.dumps(fix_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            return raw

        candidate = str(llm_fix(raw)).strip().strip('"').strip("'")
        if candidate in allowed:
            fixed = candidate
        else:
            fixed = coco.convert(names=candidate, to="name_short", not_found=raw)

        fix_cache[raw] = fixed
        cache_path.write_text(json.dumps(fix_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return fixed

    # Step A: deterministic conversion
    unique = df[col].dropna().unique()
    short_names = coco.convert(names=list(unique), to="name_short", not_found=None)
    mapping = dict(zip(unique, short_names))

    tmp_col = out_col or f"{col}__canon"
    df[tmp_col] = df[col].map(mapping)

    # Step B: optional fallback for missing
    missing = df.loc[df[tmp_col].isna(), col].dropna().unique()
    for val in missing:
        fixed = cache_or_fix(val)
        df.loc[df[col] == val, tmp_col] = fixed

    # Step C: replace original column if requested
    df[col] = df[tmp_col]
    if tmp_col != col and (out_col is None):
        df.drop(columns=[tmp_col], inplace=True)

    # Add ISO3 for mapping
    df["Country_iso3"] = coco.convert(names=df[col].fillna(""), to="ISO3", not_found=None)

    return df


# -----------------------------
# Manufacturer canonicalization (LLM-optional)
# -----------------------------


@dataclass(frozen=True)
class ManufacturerCanonConfig:
    canon_path: Path = CACHE_DIR / "manufacturer_canon_set.json"
    cache_path: Path = CACHE_DIR / "manufacturer_llm_cache.json"


SEED_CANON = {
    "Siemens Healthineers",                            # acquired CTI
    "Avelion (Alcen)",                                 # PMB -> Avelion (Alcen)
    "Best Cyclotron Systems (BCS)",                    # ABT -> BCS
    "Advanced Cyclotron Systems, Inc. (ACSI)",         # ACSI/ASCI
    "Rosatom",                                         # Rosatom
    "Sichuan Longevous Beamtech Co., Ltd (LBT)",       # LBT
}


def _basic_cleanup(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\.;:\-]+$", "", s).strip()
    return s


def _norm_key(s: str) -> str:
    s = _basic_cleanup(s).lower()
    s = re.sub(r"\(([^)]{1,20})\)", " ", s)   # remove short bracket chunks
    s = re.sub(r"[^\w\s]", " ", s)            # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _manual_map(raw: str) -> Optional[str]:
    k = _norm_key(raw)
    if not k:
        return None

    if k in {"rossatom", "ross atom", "rosatom"} or "rossatom" in k:
        return "Rosatom"

    if (
        k in {"acsi", "asci"}
        or k.startswith("acsi ")
        or k.startswith("asci ")
        or "advanced cyclotron systems" in k
    ):
        return "Advanced Cyclotron Systems"

    if "siemens" in k or k == "cti" or k.startswith("cti "):
        return "Siemens Healthineers"

    if "pmb" in k:
        return "Avelion"

    if (
        k == "abt"
        or k.startswith("abt ")
        or "advanced beam technologies" in k
        or "bcs" in k
        or "best cyclotron systems" in k
    ):
        return "Best Cyclotron Systems"

    return None


def _looks_like_acsi(raw: str) -> bool:
    k = _norm_key(raw)
    return (
        k in {"acsi", "asci"}
        or k.startswith("acsi ")
        or k.startswith("asci ")
        or "advanced cyclotron systems" in k
    )


def _load_json_set(path: Path, seed: set[str]) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set(seed)


def _load_json_dict(path: Path) -> dict[str, str]:
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    return {}


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def canonicalize_manufacturers(
    df: pd.DataFrame,
    col: str = "Manufacturer",
    out_col: str = "Manufacturer_clean",
    overwrite: bool = False,
    keep_backup: bool = True,
    grow_canon: bool = True,
    llm_choose: Optional[Callable[[str, list[str]], str]] = None,
    config: ManufacturerCanonConfig = ManufacturerCanonConfig(),
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Canonicalize manufacturer names.

    If `llm_choose` is provided, it must take (raw, canon_list) and return either:
    - an EXACT canonical string from canon_list, OR
    - "NEW:<name>" to introduce a new canonical.

    Without `llm_choose`, we use manual rules + "NEW" as the cleaned raw.
    """
    df = df.copy()

    canon_set = _load_json_set(Path(config.canon_path), seed=SEED_CANON)
    llm_cache = _load_json_dict(Path(config.cache_path))

    canon_set.update(SEED_CANON)

    if keep_backup and overwrite:
        raw_col = f"{col}_raw"
        if raw_col not in df.columns:
            df[raw_col] = df[col]

    uniq = sorted(set(_basic_cleanup(x) for x in df[col].dropna().astype(str).unique()))
    uniq = [u for u in uniq if u]

    mapping: dict[str, str] = {}

    def choose_or_new(raw: str, canon_list: list[str]) -> str:
        raw0 = _basic_cleanup(raw)
        if not raw0:
            return raw0

        if raw0 in llm_cache:
            return llm_cache[raw0]

        # Manual layer first
        m = _manual_map(raw0)
        if m:
            llm_cache[raw0] = m
            _save_json(Path(config.cache_path), llm_cache)
            return m

        if llm_choose is None:
            llm_cache[raw0] = raw0
            _save_json(Path(config.cache_path), llm_cache)
            return raw0

        ans = str(llm_choose(raw0, canon_list)).strip().strip('"').strip("'")

        if ans.startswith("NEW:"):
            canon = _basic_cleanup(ans[4:].strip())
            llm_cache[raw0] = canon
            _save_json(Path(config.cache_path), llm_cache)
            return canon

        chosen = _basic_cleanup(ans)

        # If it chose an existing canonical, validate similarity (guardrail)
        if chosen in canon_set:
            if chosen == "Advanced Cyclotron Systems" and not _looks_like_acsi(raw0):
                # prevent collapse to ACS
                llm_cache[raw0] = raw0
                _save_json(Path(config.cache_path), llm_cache)
                return raw0

            score = fuzz.token_sort_ratio(_norm_key(raw0), _norm_key(chosen))
            if score >= 90:
                llm_cache[raw0] = chosen
                _save_json(Path(config.cache_path), llm_cache)
                return chosen

            # too dissimilar -> NEW
            llm_cache[raw0] = raw0
            _save_json(Path(config.cache_path), llm_cache)
            return raw0

        # If it returned something not in canon without NEW:, accept as new
        llm_cache[raw0] = chosen
        _save_json(Path(config.cache_path), llm_cache)
        return chosen

    if verbose:
        print(f"Unique manufacturers: {len(uniq)}")
        print(f"Canon set size (start): {len(canon_set)}")

    for i, raw in enumerate(uniq, start=1):
        canon_list = sorted(canon_set)
        chosen = choose_or_new(raw, canon_list)
        mapping[raw] = chosen

        if grow_canon and chosen:
            canon_set.add(chosen)

        if verbose and i % 50 == 0:
            print(f"  resolved {i}/{len(uniq)} (canon now {len(canon_set)})")

    _save_json(Path(config.canon_path), sorted(canon_set))

    df[out_col] = df[col].map(
        lambda x: mapping.get(_basic_cleanup(x), _basic_cleanup(x)) if pd.notna(x) else None
    )

    if overwrite:
        df[col] = df[out_col]
        df.drop(columns=[out_col], inplace=True)

    if verbose:
        print(f"Canon set size (end): {len(canon_set)}")

    return df, mapping


# -----------------------------
# Generic cleaning helpers
# -----------------------------


def normalize_whitespace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def to_nan_if_unspecified(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "unspecified", "n/a", "na"):
        return None
    return s


def parse_energy_to_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    vals = [float(n) for n in nums]
    return max(vals)  # choose max if ranges exist


def add_energy_num(df: pd.DataFrame, energy_col: str = "Proton energy (MeV)") -> pd.DataFrame:
    df = df.copy()
    if energy_col in df.columns:
        df["Energy_num"] = df[energy_col].apply(parse_energy_to_float)
    else:
        df["Energy_num"] = None
    return df


def clean_cyclotron_df(
    df: pd.DataFrame,
    *,
    canonicalize_country: bool = True,
    canonicalize_manufacturer: bool = True,
    llm_fix_country: Optional[Callable[[str], str]] = None,
    llm_choose_manufacturer: Optional[Callable[[str, list[str]], str]] = None,
) -> pd.DataFrame:
    """End-to-end cleaner to match the notebook pipeline."""
    df = df.copy()

    # Whitespace
    df = normalize_whitespace(df, ["Country", "City", "Facility", "Manufacturer", "Model", "Proton energy (MeV)"])

    # Unspecified -> NaN
    for col in ["Facility", "Manufacturer", "Model", "City", "Country", "Proton energy (MeV)"]:
        if col in df.columns:
            df[col] = df[col].apply(to_nan_if_unspecified)

    # Countries + ISO3
    if canonicalize_country and "Country" in df.columns:
        df = canonicalize_countries(df, col="Country", llm_fix=llm_fix_country)

    # Manufacturers
    if canonicalize_manufacturer and "Manufacturer" in df.columns:
        df, _ = canonicalize_manufacturers(
            df,
            col="Manufacturer",
            overwrite=True,
            keep_backup=True,
            llm_choose=llm_choose_manufacturer,
            verbose=False,
        )

    # Energy numeric
    df = add_energy_num(df, energy_col="Proton energy (MeV)")

    return df
