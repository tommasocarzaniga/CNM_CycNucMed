from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def barplot_top(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path) -> Path:
    fig, ax = plt.subplots()
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png
