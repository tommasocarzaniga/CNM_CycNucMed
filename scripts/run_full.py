from pathlib import Path
import pandas as pd

from iaea_project.utils import ensure_dirs, RAW_DIR, PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR
from iaea_project.scraper import scrape_iaea_cyclotrons, save_raw
from iaea_project.cleaning import clean_cyclotron_df
from iaea_project.analysis import top_countries, top_manufacturers
from iaea_project.plotting import barplot_top
from iaea_project.pdf_report import build_pdf_report

def main():
    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    clean_csv = PROCESSED_DIR / "iaea_cyclotrons_clean.csv"
    out_pdf = REPORTS_DIR / "IAEA_Cyclotron_Report.pdf"

    df_raw = scrape_iaea_cyclotrons()
    save_raw(df_raw, raw_csv)

    df_clean = clean_cyclotron_df(df_raw)
    df_clean.to_csv(clean_csv, index=False)

    tc = top_countries(df_clean, n=15)
    tm = top_manufacturers(df_clean, n=15)

    fig1 = barplot_top(tc, "Country", "count", "Top Countries by Cyclotron Count", FIGURES_DIR / "top_countries.png")
    fig2 = barplot_top(tm, "Manufacturer", "count", "Top Manufacturers by Cyclotron Count", FIGURES_DIR / "top_manufacturers.png")

    build_pdf_report(out_pdf, tc, tm, figures=[fig1, fig2])
    print(f"Saved report to: {out_pdf}")

if __name__ == "__main__":
    main()
