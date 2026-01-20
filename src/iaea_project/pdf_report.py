from __future__ import annotations
from pathlib import Path
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def build_pdf_report(
    out_pdf: Path,
    top_countries_df: pd.DataFrame,
    top_mfr_df: pd.DataFrame,
    figures: list[Path] | None = None,
) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4)
    story = []

    story.append(Paragraph("IAEA Cyclotron Summary Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Tables
    story.append(Paragraph("Top Countries", styles["Heading2"]))
    story.append(_df_to_table(top_countries_df))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Manufacturers", styles["Heading2"]))
    story.append(_df_to_table(top_mfr_df))
    story.append(Spacer(1, 12))

    # Figures
    if figures:
        story.append(Paragraph("Figures", styles["Heading2"]))
        for p in figures:
            story.append(Image(str(p), width=450, height=280))
            story.append(Spacer(1, 12))

    doc.build(story)
    return out_pdf

def _df_to_table(df: pd.DataFrame) -> Table:
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t
