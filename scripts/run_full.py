from __future__ import annotations

from iaea_project.utils import ensure_dirs, RAW_DIR, PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR
from iaea_project.scraper import scrape_iaea_cyclotrons, save_raw
from iaea_project.cleaning import clean_cyclotron_df
from iaea_project.analysis import global_comparison_tables, country_summary
from iaea_project.plotting import save_country_map
from iaea_project.pdf_report import build_pdf_report


def main():
    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    clean_csv = PROCESSED_DIR / "iaea_cyclotrons_clean.csv"
    out_pdf = REPORTS_DIR / "IAEA_Cyclotron_Report.pdf"

    # 1) Scrape
    df_raw = scrape_iaea_cyclotrons()
    save_raw(df_raw, raw_csv)

    # 2) Clean
    df_clean = clean_cyclotron_df(df_raw)
    df_clean.to_csv(clean_csv, index=False)

    # 3) Global tables
    top_countries, top_manu, energy_country = global_comparison_tables(df_clean)

    # 4) One illustrative country section (edit this list as you wish)
    countries = ["Italy"]
    sections = []
    for c in countries:
        cs = country_summary(df_clean, c, top_n=10)
        if not cs.get("found"):
            continue

        map_path = save_country_map(df_clean, c, out_path=FIGURES_DIR / f"{c}_map.png")
        tables = {
            "Top cities": cs["cities_top"],
            "Top facilities": cs["facilities_top"],
            "Top manufacturers": cs["manufacturers"].head(10),
            "Top models": cs["models_top"],
        }
        sections.append(
            {
                "country": c,
                "summary_md": f"Total cyclotrons: {cs['total_cyclotrons']}\nEnergy stats: {cs['energy_stats']}",
                "tables": tables,
                "images": [map_path] if map_path else [],
            }
        )

    # 5) PDF
    build_pdf_report(
        out_pdf,
        title="IAEA Cyclotron Report",
        subtitle="Scrape → clean → analyze → map → PDF",
        top_countries=top_countries,
        top_manufacturers=top_manu,
        energy_country=energy_country,
        country_sections=sections,
    )

    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
