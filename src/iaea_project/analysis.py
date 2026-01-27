from __future__ import annotations

import pandas as pd


def country_summary(df: pd.DataFrame, country: str, top_n: int = 15) -> dict:
    sub = df[df["Country"].str.lower() == str(country).strip().lower()].copy()
    if sub.empty:
        candidates = (
            df[df["Country"].str.lower().str.contains(str(country).strip().lower(), na=False)]["Country"]
            .dropna()
            .unique()
        )
        return {"country": country, "found": False, "did_you_mean": sorted(candidates)[:20]}

    total = len(sub)
    cities = sub.groupby("City").size().sort_values(ascending=False)
    facilities = sub.groupby("Facility").size().sort_values(ascending=False)
    manufacturers = sub.groupby("Manufacturer").size().sort_values(ascending=False)
    models = sub.groupby("Model").size().sort_values(ascending=False)
    manu_model = sub.groupby(["Manufacturer", "Model"]).size().sort_values(ascending=False)

    energy_stats = {
        "count_numeric": int(sub["Energy_num"].notna().sum()) if "Energy_num" in sub.columns else 0,
        "min": float(sub["Energy_num"].min()) if "Energy_num" in sub.columns and sub["Energy_num"].notna().any() else None,
        "median": float(sub["Energy_num"].median()) if "Energy_num" in sub.columns and sub["Energy_num"].notna().any() else None,
        "max": float(sub["Energy_num"].max()) if "Energy_num" in sub.columns and sub["Energy_num"].notna().any() else None,
    }

    return {
        "country": country,
        "found": True,
        "total_cyclotrons": total,
        "cities_top": cities.head(top_n),
        "facilities_top": facilities.head(top_n),
        "manufacturers": manufacturers,
        "models_top": models.head(top_n),
        "manufacturer_model_top": manu_model.head(top_n),
        "energy_stats": energy_stats,
        "all_cities_count": int(cities.shape[0]),
        "all_facilities_count": int(facilities.shape[0]),
    }


def country_report_for_llm(df: pd.DataFrame, country: str, top_n: int = 10) -> str:
    out = country_summary(df, country, top_n=top_n)
    if not out.get("found"):
        hints = out.get("did_you_mean", [])
        hint_txt = ", ".join(hints) if hints else "(no close matches)"
        return f"No exact match for '{country}'. Did you mean: {hint_txt}?"

    lines = []
    lines.append(f"Country: {out['country']}")
    lines.append(f"Total cyclotrons (rows): {out['total_cyclotrons']}")
    lines.append(f"Unique cities: {out['all_cities_count']}")
    lines.append(f"Unique facilities: {out['all_facilities_count']}")

    es = out.get("energy_stats", {})
    lines.append(
        f"Energy (MeV) numeric count={es.get('count_numeric')}, min={es.get('min')}, median={es.get('median')}, max={es.get('max')}"
    )

    def fmt_series(s: pd.Series, title: str) -> None:
        lines.append("")
        lines.append(title)
        for k, v in s.items():
            lines.append(f"- {k}: {int(v)}")

    fmt_series(out["cities_top"], f"Top {top_n} cities")
    fmt_series(out["facilities_top"], f"Top {top_n} facilities")
    fmt_series(out["manufacturers"].head(top_n), f"Top {top_n} manufacturers")
    fmt_series(out["models_top"], f"Top {top_n} models")

    return "\n".join(lines)


def global_comparison_tables(df: pd.DataFrame):
    top_countries = df["Country"].dropna().value_counts().head(25)
    top_manu = df["Manufacturer"].dropna().value_counts().head(15)

    energy_country = (
        df.dropna(subset=["Country"])
        .groupby("Country")["Energy_num"]
        .agg(["count", "min", "median", "max"])
        .sort_values("count", ascending=False)
        .head(25)
    )

    return top_countries, top_manu, energy_country


def data_quality_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    missing = {}
    for col in ["City", "Facility", "Manufacturer", "Model", "Proton energy (MeV)", "Energy_num"]:
        if col in df.columns:
            missing[col] = int(df[col].isna().sum())
        else:
            missing[col] = total

    return {
        "total_rows": total,
        "missing_counts": missing,
        "missing_pct": {k: (v / total * 100 if total else 0) for k, v in missing.items()},
    }

def top_facility_context(df: pd.DataFrame, country: str, top_rows: int = 8) -> dict:
    sub = df[df["Country"].str.lower() == str(country).strip().lower()].copy()
    if sub.empty:
        return {"found": False, "country": country}

    # top facility by number of rows
    fac_counts = sub["Facility"].dropna().astype(str).value_counts()
    if fac_counts.empty:
        return {"found": False, "country": country}

    top_fac = fac_counts.index[0]
    top_fac_n = int(fac_counts.iloc[0])

    sub_fac = sub[sub["Facility"].astype(str) == top_fac].copy()

    city_counts = sub_fac["City"].dropna().astype(str).value_counts().head(3).to_dict()
    manu_counts = sub_fac["Manufacturer"].dropna().astype(str).value_counts().head(3).to_dict()
    model_counts = sub_fac["Model"].dropna().astype(str).value_counts().head(3).to_dict()

    cols = [c for c in ["Facility", "City", "Manufacturer", "Model", "Proton energy (MeV)", "Energy_num"] if c in sub_fac.columns]
    examples = sub_fac[cols].head(top_rows).to_dict(orient="records")

    return {
        "found": True,
        "country": country,
        "top_facility": top_fac,
        "top_facility_rows": top_fac_n,
        "top_cities": city_counts,
        "top_manufacturers": manu_counts,
        "top_models": model_counts,
        "examples": examples,
    }
