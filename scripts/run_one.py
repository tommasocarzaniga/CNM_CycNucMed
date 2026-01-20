from __future__ import annotations

import pandas as pd

from iaea_project.utils import RAW_DIR, PROCESSED_DIR, ensure_dirs
from iaea_project.cleaning import clean_cyclotron_df
from iaea_project.analysis import country_report_for_llm


def main(country: str = "Italy"):
    ensure_dirs()

    raw_csv = RAW_DIR / "iaea_cyclotrons_raw.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Missing {raw_csv}. Run scripts/run_full.py once (or put a raw CSV there)."
        )

    df_raw = pd.read_csv(raw_csv)
    df_clean = clean_cyclotron_df(df_raw)
    (PROCESSED_DIR / "iaea_cyclotrons_clean.csv").write_text(df_clean.to_csv(index=False), encoding="utf-8")

    print(country_report_for_llm(df_clean, country, top_n=10))


if __name__ == "__main__":
    main()
