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


SEED_CANON: set[str] = {
    "Siemens Healthineers",                            # acquired CTI
    "Avelion (Alcen)",                                 # PMB -> Avelion (Alcen)
    "Best Cyclotron Systems (BCS)",                    # ABT -> BCS
    "Advanced Cyclotron Systems, Inc. (ACSI)",         # ACSI/ASCI
    "Rosatom",                                         # Rosatom
    "Sichuan Longevous Beamtech Co., Ltd (LBT)",       # LBT
}

# --- Stronger normalization for manufacturer matching ---
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "plc",
    "gmbh", "ag", "sa", "sarl", "srl", "spa", "bv", "nv", "kg", "oy", "ab", "as",
    "pte", "pty", "kk", "ltda", "sas",
}

GENERIC_TOKENS = {
    "the", "and", "&",
    "group", "international", "global", "holding", "holdings",
    "systems", "system", "technology", "technologies",
    "medical", "health", "healthcare",
    "negative", "ion", "ions",
}


def _basic_cleanup(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\.;:\-]+$", "", s).strip()
    return s


def _norm_key(s: str) -> str:
    """Light normalization (kept for compatibility)."""
    s = _basic_cleanup(s).lower()
    s = re.sub(r"\(([^)]{1,20})\)", " ", s)   # remove short bracket chunks
    s = re.sub(r"[^\w\s]", " ", s)            # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_key_strong(s: str) -> str:
    """Stronger normalization for dedup/collapse."""
    s = _basic_cleanup(s).lower()
    s = re.sub(r"\(([^)]{1,60})\)", " ", s)   # remove bracket chunks (acronyms/legal suffix often)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks = s.split()
    toks2: list[str] = []
    for t in toks:
        if t in LEGAL_SUFFIXES:
            continue
        if t in GENERIC_TOKENS:
            continue
        toks2.append(t)

    return " ".join(toks2).strip()


def _is_acronym(s: str) -> bool:
    s = _basic_cleanup(s)
    return bool(re.fullmatch(r"[A-Z0-9]{2,6}", s))


def _choose_rep_label(labels: list[str]) -> str:
    """Pick the 'best' representative label among duplicates."""
    def score(x: str) -> tuple[int, int]:
        x0 = _basic_cleanup(x)
        has_paren = 1 if ("(" in x0 and ")" in x0) else 0
        return (has_paren, len(x0))

    return max(labels, key=score)


def _best_fuzzy_to_canon(raw: str, canon_list: list[str], threshold: int = 93) -> Optional[str]:
    """Conservative fuzzy match raw -> existing canonical using strong keys."""
    rk = _norm_key_strong(raw)
    if not rk:
        return None

    best = None
    best_score = -1
    for c in canon_list:
        sc = fuzz.token_sort_ratio(rk, _norm_key_strong(c))
        if sc > best_score:
            best_score = sc
            best = c

    if best is not None and best_score >= threshold:
        return best
    return None


def _manual_map(raw: str) -> Optional[str]:
    """
    Deterministic vendor alias rules (high precision).
    Uses both light and strong keys to catch typos / hyphenated descriptors robustly.
    """
    k = _norm_key(raw)          # light key (keeps more tokens)
    ks = _norm_key_strong(raw)  # strong key (drops legal suffixes & boilerplate)

    if not k and not ks:
        return None

    # --- Rosatom ---
    if k in {"rossatom", "ross atom", "rosatom"} or "rossatom" in k or "rosatom" in k:
        return "Rosatom"

    # --- ACSI / ASCI ---
    if (
        k in {"acsi", "asci"}
        or k.startswith("acsi ")
        or k.startswith("asci ")
        or "advanced cyclotron systems" in k
        or ks in {"acsi", "asci"}
        or "advanced cyclotron" in ks
    ):
        return "Advanced Cyclotron Systems, Inc. (ACSI)"

    # --- Siemens / CTI ---
    if "siemens" in k or k == "cti" or k.startswith("cti ") or "siemens" in ks or ks == "cti":
        return "Siemens Healthineers"

    # --- PMB / Avelion-Alcen ---
    if "pmb" in k or "pmb" in ks or "alcen" in ks or "avelion" in ks:
        return "Avelion (Alcen)"

    # --- ABT / BCS ---
    if (
        k == "abt"
        or k.startswith("abt ")
        or "advanced beam technologies" in k
        or "bcs" in k
        or "best cyclotron systems" in k
        or ks == "abt"
        or "best cyclotron" in ks
        or "bcs" in ks
    ):
        return "Best Cyclotron Systems (BCS)"

    # --- Sichuan Longevous Beamtech / LBT ---
    if "longevous" in ks or re.search(r"\blbt\b", ks):
        return "Sichuan Longevous Beamtech Co., Ltd (LBT)"

    # --- TCC variants ---
    # catches: "TCC", "TCC The Cyclotron Corporation", punctuation variants
    if ks == "tcc" or ("cyclotron" in ks and "corporation" in ks and "tcc" in (k or ks)):
        # pick ONE canonical label (choose whichever you prefer)
        return "TCC (The Cyclotron Corporation)"  # <— recommended

    # --- Scanditronix variants ---
    # catches: "Scanditronix-Negative-Ion", "Scanditronix Negative Ion", etc.
    if "scanditronix" in ks:
        return "Scanditronix"

    # --- Sumitomo typo variants ---
    if ks in {"sumotomo", "sumitomo"} or "sumitomo" in ks or "sumotomo" in ks:
        return "Sumitomo"

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
    *,
    fuzzy_threshold: int = 93,
    dedup_canon_threshold: int = 95,
    dedup_existing_canon: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Canonicalize manufacturer names.

    Improvements vs original:
    - Strong normalization + non-LLM fuzzy collapse to existing canon
    - Optional one-time dedup of the canon set itself
    - Safer canon growth to avoid runaway duplicates

    If `llm_choose` is provided, it must take (raw, canon_list) and return either:
    - an EXACT canonical string from canon_list, OR
    - "NEW:<name>" to introduce a new canonical.
    """
    df = df.copy()

    canon_set = _load_json_set(Path(config.canon_path), seed=SEED_CANON)
    llm_cache = _load_json_dict(Path(config.cache_path))
    canon_set.update(SEED_CANON)

    # Optional: dedup the canon set itself (helps if you already persisted duplicates)
    canon_rep_map: dict[str, str] = {}

    def _dedup_canon(cset: set[str], threshold: int) -> tuple[set[str], dict[str, str]]:
        lst = sorted(cset)
        groups: list[list[str]] = []
        for name in lst:
            placed = False
            for g in groups:
                if fuzz.token_sort_ratio(_norm_key_strong(name), _norm_key_strong(g[0])) >= threshold:
                    g.append(name)
                    placed = True
                    break
            if not placed:
                groups.append([name])

        rep_map: dict[str, str] = {}
        new_set: set[str] = set()
        for g in groups:
            rep = _choose_rep_label(g)
            new_set.add(rep)
            for x in g:
                rep_map[x] = rep
        return new_set, rep_map

    if dedup_existing_canon and canon_set:
        canon_set, canon_rep_map = _dedup_canon(canon_set, threshold=dedup_canon_threshold)

    if keep_backup and overwrite:
        raw_col = f"{col}_raw"
        if raw_col not in df.columns:
            df[raw_col] = df[col]

    uniq = sorted(set(_basic_cleanup(x) for x in df[col].dropna().astype(str).unique()))
    uniq = [u for u in uniq if u]

    mapping: dict[str, str] = {}

    def _safe_to_add_canon(label: str) -> bool:
        if not label:
            return False
        lab = _basic_cleanup(label)
        if len(lab) < 3:
            return False
        if _is_acronym(lab):
            return True
        if len(_norm_key_strong(lab)) < 3:
            return False
        return True

def choose_or_new(raw: str, canon_list: list[str]) -> str:
    raw0 = _basic_cleanup(raw)
    if not raw0:
        return raw0

    # 1) Always try deterministic/manual rules first (they should override old cache)
    m = _manual_map(raw0)
    if m:
        llm_cache[raw0] = m
        _save_json(Path(config.cache_path), llm_cache)
        return m

    # 2) Always try fuzzy collapse to existing canon (also overrides old cache)
    hit = _best_fuzzy_to_canon(raw0, canon_list, threshold=fuzzy_threshold)
    if hit:
        llm_cache[raw0] = hit
        _save_json(Path(config.cache_path), llm_cache)
        return hit

    # 3) If cached, use it (only after manual+fuzzy had a chance)
    if raw0 in llm_cache:
        return llm_cache[raw0]

    # 4) No LLM: accept raw as canonical
    if llm_choose is None:
        llm_cache[raw0] = raw0
        _save_json(Path(config.cache_path), llm_cache)
        return raw0

    # 5) LLM path (unchanged, but keep the fuzzy collapse on NEW:)
    ans = str(llm_choose(raw0, canon_list)).strip().strip('"').strip("'")

    if ans.startswith("NEW:"):
        canon = _basic_cleanup(ans[4:].strip())
        hit2 = _best_fuzzy_to_canon(canon, canon_list, threshold=fuzzy_threshold)
        chosen_final = hit2 or canon
        llm_cache[raw0] = chosen_final
        _save_json(Path(config.cache_path), llm_cache)
        return chosen_final

    chosen = _basic_cleanup(ans)

    if chosen in canon_set:
        score = fuzz.token_sort_ratio(_norm_key_strong(raw0), _norm_key_strong(chosen))
        if score >= 90:
            llm_cache[raw0] = chosen
            _save_json(Path(config.cache_path), llm_cache)
            return chosen
        llm_cache[raw0] = raw0
        _save_json(Path(config.cache_path), llm_cache)
        return raw0

    hit3 = _best_fuzzy_to_canon(chosen, canon_list, threshold=fuzzy_threshold)
    chosen_final = hit3 or chosen
    llm_cache[raw0] = chosen_final
    _save_json(Path(config.cache_path), llm_cache)
    return chosen_final

    if verbose:
        print(f"Unique manufacturers: {len(uniq)}")
        print(f"Canon set size (start): {len(canon_set)}")

    for i, raw in enumerate(uniq, start=1):
        canon_list = sorted(canon_set)
        chosen = choose_or_new(raw, canon_list)
        chosen = canon_rep_map.get(chosen, chosen)

        mapping[_basic_cleanup(raw)] = chosen

        if grow_canon and _safe_to_add_canon(chosen):
            canon_set.add(chosen)

        if verbose and i % 50 == 0:
            print(f"  resolved {i}/{len(uniq)} (canon now {len(canon_set)})")

    _save_json(Path(config.canon_path), sorted(canon_set))
    _save_json(Path(config.cache_path), llm_cache)

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
            # Practical default while iterating: avoid canon explosion.
            grow_canon=False,
            verbose=False,
        )

    # Energy numeric
    df = add_energy_num(df, energy_col="Proton energy (MeV)")

    return df
