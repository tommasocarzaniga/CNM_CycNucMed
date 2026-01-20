from __future__ import annotations

import argparse
from pathlib import Path

from iaea_project.utils import (
    ensure_dirs,
    RAW_DIR,
    PROCESSED_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)
from iaea_project.scraper import scrape_iaea_cyclotrons, save_raw
from iaea_project.cleaning import clean_cyclotron_df
from iaea_project.analysis import global_comparison_tables, country_summary
from iaea_project.plotting import save_country_map
from iaea_project.pdf_report import build_pdf_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full IAEA cyclotron pipeline: scrape -> clean -> analyze -> map -> PDF",
    )

    p.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help=(
            "Countries to include as executive-summary sections. "
            "If omitted, the script uses the top N countries by row count."
        ),
    )
    p.add_argument(
        "--top-n-countries",
        type=int,
        default=10,
        help="If --countries is omitted, include the top N countries (default: 10).",
    )
    p.add_argument(
        "--max-country-sections",
        type=int,
        default=25,
        help="Safety cap on number of country sections in the PDF (default: 25).",
    )
    p.add_argument(
        "--disable-maps",
        action="store_true",
        help="Skip generating country map figures.",
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help=(
            "Enable optional LLM assistance for rare country fixes and manufacturer canonicalization. "
            "Requires: pip install openai and env var OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model name passed to the OpenAI SDK (default: gpt-4.1-mini).",
    )

    p.add_argument(
        "--out-pdf",
        default=str(REPORTS_DIR / "IAEA_Cyclotron_Report.pdf"),
        help="Output PDF path (default: outputs/reports/IAEA_Cyclotron_Report.pdf).",
    )
    p.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Do not scrape; instead read data/raw/iaea_cyclotrons_raw.csv.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    clean_csv = PROCESSED_DIR / "iaea_cyclotrons_clean.csv"
    out_pdf = Path(args.out_pdf)

    # -------------------
    # 1) Scrape or read
    # -------------------
    if args.skip_scrape:
        import pandas as pd

        if not raw_csv.exists():
            raise FileNotFoundError(
                f"--skip-scrape was set but raw CSV not found at: {raw_csv}"
            )
        df_raw = pd.read_csv(raw_csv)
    else:
        df_raw = scrape_iaea_cyclotrons()
        save_raw(df_raw, raw_csv)

    # -------------------
    # 2) Clean (LLM optional)
    # -------------------
    llm_fix_country = None
    llm_choose_manu = None
    if args.enable_llm:
        from iaea_project.llm_adapters import (
            llm_fix_country_openai,
            llm_choose_manufacturer_openai,
        )

        llm_fix_country = lambda s: llm_fix_country_openai(s, model=args.llm_model)
        llm_choose_manu = lambda raw, canon: llm_choose_manufacturer_openai(
            raw, canon, model=args.llm_model
        )

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

    # Decide which countries to include
    if args.countries:
        countries = list(dict.fromkeys(args.countries))  # de-dup, preserve order
    else:
        countries = list(top_countries.head(args.top_n_countries).index)

    if len(countries) > args.max_country_sections:
        countries = countries[: args.max_country_sections]

    # -------------------
    # 4) Per-country executive summary sections
    # -------------------
    sections = []
    for c in countries:
        cs = country_summary(df_clean, c, top_n=10)
        if not cs.get("found"):
            continue

        images = []
        if not args.disable_maps:
            map_path = save_country_map(
                df_clean,
                c,
                out_path=FIGURES_DIR / f"{c}_map.png",
            )
            if map_path:
                images.append(map_path)

        tables = {
            "Top cities": cs["cities_top"],
            "Top facilities": cs["facilities_top"],
            "Top manufacturers": cs["manufacturers"].head(10),
            "Top models": cs["models_top"],
        }

        summary_md = (
            f"Total cyclotrons (rows): {cs['total_cyclotrons']}\n"
            f"Energy stats (MeV): {cs['energy_stats']}\n"
            f"Unique cities: {cs['all_cities_count']}, Unique facilities: {cs['all_facilities_count']}"
        )

        sections.append(
            {
                "country": c,
                "summary_md": summary_md,
                "tables": tables,
                "images": images,
            }
        )

    # -------------------
    # 5) Build PDF
    # -------------------
    build_pdf_report(
        out_pdf,
        title="IAEA Cyclotron Report",
        subtitle="University project: modular Python package + reproducible pipeline",
        top_countries=top_countries,
        top_manufacturers=top_manu,
        energy_country=energy_country,
        country_sections=sections,
    )

    print(f"Saved raw:   {raw_csv}")
    print(f"Saved clean: {clean_csv}")
    print(f"Saved PDF:  {out_pdf}")


if __name__ == "__main__":
    main()
