from __future__ import annotations
import pandas as pd
from pathlib import Path

def scrape_iaea_cyclotrons() -> pd.DataFrame:
    """
    Return a DataFrame with the raw scraped cyclotron table.
    Implement your SharePoint/IAEA pagination logic here.
    """
    raise NotImplementedError("Move your scraping logic here")

def save_raw(df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
