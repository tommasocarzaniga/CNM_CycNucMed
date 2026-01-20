from __future__ import annotations
import pandas as pd

def clean_cyclotron_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, fix country names, remove duplicates,
    coerce numeric columns, etc.
    """
    df = df.copy()

    # Example: normalize column names
    df.columns = [c.strip() for c in df.columns]

    # TODO: paste your real cleaning rules here

    return df
