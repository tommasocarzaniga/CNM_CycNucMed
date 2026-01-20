from __future__ import annotations
import pandas as pd

def top_countries(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("Country", dropna=False)
          .size()
          .sort_values(ascending=False)
          .head(n)
          .reset_index(name="count")
    )

def top_manufacturers(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("Manufacturer", dropna=False)
          .size()
          .sort_values(ascending=False)
          .head(n)
          .reset_index(name="count")
    )
