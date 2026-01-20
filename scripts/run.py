
"""Single entry-point for the university project.

This module is designed to be used from **both**:

1) Terminal / CLI:
   python run.py --countries Switzerland Germany

2) Jupyter / Colab:
   from run import run
   run()                      # all countries
   run(["Switzerland"])       # only Switzerland

We avoid argparse crashes in notebooks by ignoring Jupyter-injected args.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def run(
    countries: list[str] | None = None,
    *,
    max_country_sections: int | None = None,
    disable_maps: bool = False,
    enable_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
    out_pdf: str | Path | None = None,
    skip_scrape: bool = False,
) -> Path:
    """Run the full pipeline and return the path to the generated PDF.

    Args:
        countries:
            - None  -> include *all* countries found in the cleaned dataset.
            - list  -> include only these countries (order preserved, de-duplicated).
        max_country_sections: Optional safety cap (None = no cap).
        disable_maps: Skip generating map figures.
        enable_llm: Enable optional OpenAI-based helper for edge-case normalization.
        llm_model: OpenAI model name (only used if enable_llm=True).
        out_pdf: Output path for the report PDF. If None, uses outputs/reports/IAEA_Cyclotron_Report.pdf
        skip_scrape: If True, read data/raw/iaea_cyclotrons_raw.csv instead of scraping.
    """

    # Local imports keep module import fast, and make it clearer what belongs to the package.
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

    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    clean_csv = PROCESSED_DIR / "iaea_cyclotrons_clean.csv"

    if out_pdf is None:
        out_pdf_path = REPORTS_DIR / "IAEA_Cyclotron_Report.pdf"
    else:
        out_pdf_path = Path(out_pdf)

    # -------------------
    # 1) Scrape or read
    # -------------------
    if skip_scrape:
        import pandas as pd

        if not raw_csv.exists():
            raise FileNotFoundError(
                f"skip_scrape=True but raw CSV not found at: {raw_csv}"
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
    if enable_llm:
        from iaea_project.llm_adapters import (
            llm_fix_country_openai,
            llm_choose_manufacturer_openai,
        )

        llm_fix_country = lambda s: llm_fix_country_openai(s, model=llm_model)
        llm_choose_manu = lambda raw, canon: llm_choose_manufacturer_openai(
            raw, canon, model=llm_model
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

    # -------------------
    # 4) Decide which countries to include
    # -------------------
    if countries is None:
        # Include *all* countries found in the data (stable order: alphabetical).
        all_countries = (
            df_clean.get("Country")
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        countries_list = sorted({c for c in all_countries if c})
    else:
        # De-dup while preserving order
        seen = set()
        countries_list = []
        for c in countries:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                countries_list.append(c)

    if max_country_sections is not None and len(countries_list) > max_country_sections:
        countries_list = countries_list[:max_country_sections]

    # -------------------
    # 5) Per-country executive summary sections
    # -------------------
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
    # 6) Build PDF
    # -------------------
    build_pdf_report(
        out_pdf_path,
        title="IAEA Cyclotron Report",
        subtitle="University project: modular Python package + reproducible pipeline",
        top_countries=top_countries,
        top_manufacturers=top_manu,
        energy_country=energy_country,
        country_sections=sections,
    )

    print(f"Saved raw:   {raw_csv}")
    print(f"Saved clean: {clean_csv}")
    print(f"Saved PDF:  {out_pdf_path}")
    return out_pdf_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the IAEA cyclotron pipeline and generate a PDF report.",
    )
    p.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Countries to include. If omitted, include ALL countries found in the dataset.",
    )
    p.add_argument(
        "--max-country-sections",
        type=int,
        default=0,
        help="Safety cap on number of country sections (0 = no cap).",
    )
    p.add_argument(
        "--disable-maps",
        action="store_true",
        help="Skip generating country map figures.",
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable optional OpenAI helpers (needs OPENAI_API_KEY).",
    )
    p.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model name (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--out-pdf",
        default=None,
        help="Output PDF path (default: outputs/reports/IAEA_Cyclotron_Report.pdf).",
    )
    p.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Do not scrape; read data/raw/iaea_cyclotrons_raw.csv.",
    )
    return p.parse_args(args=argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point.

    - Terminal: python run.py --countries Switzerland Germany
    - Notebook: main([]) or (better) from run import run; run(...)
    """

    # If called from a notebook without explicit argv, ignore injected kernel args.
    if argv is None:
        try:
            import sys

            if "ipykernel" in sys.modules:
                argv = []
        except Exception:
            argv = []

    args = _parse_args(argv=argv)
    cap = None if (args.max_country_sections or 0) <= 0 else int(args.max_country_sections)
    run(
        countries=args.countries,
        max_country_sections=cap,
        disable_maps=bool(args.disable_maps),
        enable_llm=bool(args.enable_llm),
        llm_model=str(args.llm_model),
        out_pdf=args.out_pdf,
        skip_scrape=bool(args.skip_scrape),
    )


if __name__ == "__main__":
    main()
