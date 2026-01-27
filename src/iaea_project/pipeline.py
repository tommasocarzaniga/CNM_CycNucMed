from __future__ import annotations

from pathlib import Path


def run_pipeline(
    countries: list[str] | None = None,
    *,
    max_country_sections: int | None = None,
    disable_maps: bool = False,
    enable_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
    out_pdf: str | Path | None = None,
    skip_scrape: bool = False,
) -> Path:
    """
    Run the full pipeline (scrape -> clean -> analyze -> plot -> PDF) and return the output PDF path.

    This is the *library* entrypoint. It must not depend on scripts/ or argparse.
    """

    # Local imports keep import time low and avoid side-effects.
    from iaea_project.utils import (
        ensure_dirs,
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    )
    from iaea_project.scraper import scrape_iaea_cyclotrons, save_raw
    from iaea_project.cleaning import clean_cyclotron_df
    from iaea_project.analysis import (
    global_comparison_tables,
    country_summary,
    top_facility_context,  # NEW
    )
    
    from iaea_project.plotting import save_country_map
    from iaea_project.pdf_report import build_pdf_report

    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    clean_csv = PROCESSED_DIR / "iaea_cyclotrons_clean.csv"

    out_pdf_path = (REPORTS_DIR / "IAEA_Cyclotron_Report.pdf") if out_pdf is None else Path(out_pdf)

    # -------------------
    # 1) Scrape or read
    # -------------------
    if skip_scrape:
        import pandas as pd

        if not raw_csv.exists():
            raise FileNotFoundError(f"skip_scrape=True but raw CSV not found at: {raw_csv}")
        df_raw = pd.read_csv(raw_csv)
    else:
        df_raw = scrape_iaea_cyclotrons()
        save_raw(df_raw, raw_csv)

    # -------------------
    # 2) Clean (LLM optional)
    # -------------------
    llm_fix_country = None
    llm_choose_manu = None
    if enable_llm:
        from iaea_project.llm_adapter import (
            llm_fix_country_openai,
            llm_choose_manufacturer_openai,
        )

        llm_fix_country = lambda s: llm_fix_country_openai(s, model=llm_model)
        llm_choose_manu = lambda raw, canon: llm_choose_manufacturer_openai(raw, canon, model=llm_model)

    df_clean = clean_cyclotron_df(
        df_raw,
        llm_fix_country=llm_fix_country,
        llm_choose_manufacturer=llm_choose_manu,
    )
    df_clean.to_csv(clean_csv, index=False)

    # -------------------
    # 3) Global tables
    # -------------------
    top_countries, top_manu, energy_country = global_comparison_tables(df_clean)

    # -------------------
    # 4) Decide which countries to include
    # -------------------
    if countries is None:
        all_countries = df_clean.get("Country").dropna().astype(str).map(str.strip)
        countries_list = sorted({c for c in all_countries if c})
    else:
        seen: set[str] = set()
        countries_list: list[str] = []
        for c in countries:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                countries_list.append(c)

    if max_country_sections is not None and len(countries_list) > max_country_sections:
        countries_list = countries_list[:max_country_sections]

    # -------------------
    # 5) Per-country sections
    # -------------------
    import json
    from iaea_project.utils import CACHE_DIR

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    top_site_cache_path = CACHE_DIR / "top_site_blurb_cache.json"

    if top_site_cache_path.exists():
        try:
            top_site_cache = json.loads(top_site_cache_path.read_text(encoding="utf-8"))
        except Exception:
            top_site_cache = {}
    else:
        top_site_cache = {}

    def save_top_site_cache():
        top_site_cache_path.write_text(
            json.dumps(top_site_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Optional LLM function for the top-site blurb (only if enable_llm=True)
    llm_top_site_blurb = None
    if enable_llm:
        from iaea_project.llm_adapter import llm_top_site_blurb_openai  # you create this adapter

        llm_top_site_blurb = lambda ctx: llm_top_site_blurb_openai(ctx, model=llm_model)

    sections = []
    for c in countries_list:
        cs = country_summary(df_clean, c, top_n=10)
        if not cs.get("found"):
            continue

        images: list[Path] = []
        if not disable_maps:
            map_path = save_country_map(
                df_clean,
                c,
                out_path=FIGURES_DIR / f"{c}_map.png",
            )
            if map_path:
                images.append(map_path)

        summary_md = (
            f"Total cyclotrons (rows): {cs['total_cyclotrons']}\n"
            f"Energy stats (MeV): {cs['energy_stats']}\n"
            f"Unique cities: {cs['all_cities_count']}, Unique facilities: {cs['all_facilities_count']}"
        )

        # -------- NEW: top site context + (optional) LLM blurb + cache --------
        ctx = top_facility_context(df_clean, c, top_rows=8)

        print("[DEBUG]", c, "ctx found:", ctx.get("found"), "facility:", ctx.get("top_facility"))
    
        top_site_blurb = None

        if ctx.get("found"):
            cache_key = f"{ctx.get('country')}|{ctx.get('top_facility')}|{ctx.get('top_city')}"
        
            if cache_key in top_site_cache:
                top_site_blurb = top_site_cache[cache_key]
        
            elif llm_top_site_blurb is not None:
                try:
                    top_site_blurb = str(llm_top_site_blurb(ctx))
                except Exception:
                    top_site_blurb = None
        
                # ✅ CLEANUP BLOCK GOES HERE
                if top_site_blurb:
                    top_site_blurb = top_site_blurb.strip()
                    if len(top_site_blurb) > 600:
                        top_site_blurb = top_site_blurb[:600].rsplit(" ", 1)[0] + "…"
        
                    # Now cache the cleaned version
                    top_site_cache[cache_key] = top_site_blurb
                    save_top_site_cache()
        
        sections.append(
            {
                "country": c,
                "summary_md": summary_md,
                "top_site_blurb": top_site_blurb,  # <--- NEW
                "top_site_ctx": ctx,               # <--- optional: keep context for debugging
                "tables": {
                    "Top cities": cs["cities_top"],
                    "Top facilities": cs["facilities_top"],
                    "Top manufacturers": cs["manufacturers"].head(10),
                    "Top models": cs["models_top"],
                },
                "images": images,
            }
        )

    # -------------------
    # 6) Build PDF
    # -------------------
    build_pdf_report(
        out_pdf_path,
        title="Coding & KI - Exam",
        subtitle="Cyclotron Atlas: A Country-Selectable AI Guide for Medical Cyclotrons",
        top_countries=top_countries,
        top_manufacturers=top_manu,
        energy_country=energy_country,
        country_sections=sections,
    )

    print(f"Saved raw:   {raw_csv}")
    print(f"Saved clean: {clean_csv}")
    print(f"Saved PDF:   {out_pdf_path}")
    return out_pdf_path


__all__ = ["run_pipeline"]
