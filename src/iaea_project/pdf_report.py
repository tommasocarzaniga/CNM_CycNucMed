from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape as xml_escape

styles = getSampleStyleSheet()

# Centered Heading2
if "Heading2Center" not in styles.byName:
    styles.add(
        ParagraphStyle(
            name="Heading2Center",
            parent=styles["Heading2"],
            alignment=TA_CENTER,
        )
    )

# Centered subtitle style
if "SubtitleCenter" not in styles.byName:
    styles.add(
        ParagraphStyle(
            name="SubtitleCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=11,
            spaceAfter=8,
        )
    )


def escape_paragraph_text(s: str) -> str:
    """Escape &, <, > for ReportLab Paragraph mini-HTML + preserve newlines."""
    if s is None:
        return ""
    s = xml_escape(str(s))
    return s.replace("\n", "<br/>")


def df_to_table(
    df: pd.DataFrame,
    max_rows: int = 30,
    font_size: int = 8,
    repeat_header: bool = True,
) -> Table:
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    data = [list(df2.reset_index().columns)] + df2.reset_index().values.tolist()
    t = Table(data, repeatRows=1 if repeat_header else 0)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def two_column_toplists(
    left_title: str,
    left: pd.Series,
    right_title: str,
    right: pd.Series,
    max_rows: int = 20,
) -> Table:
    """Build a 2-column layout: left toplist and right toplist on the same page."""

    def series_to_df(s: pd.Series, name: str) -> pd.DataFrame:
        return s.head(max_rows).rename(name).to_frame()

    lt = df_to_table(series_to_df(left, "count"), max_rows=max_rows)
    rt = df_to_table(series_to_df(right, "count"), max_rows=max_rows)

    outer = Table(
        [
            [
                Paragraph(escape_paragraph_text(left_title), styles["Heading3"]),
                Paragraph(escape_paragraph_text(right_title), styles["Heading3"]),
            ],
            [lt, rt],
        ],
        colWidths=[270, 270],
    )
    outer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return outer


def two_column_tables(
    left_title: str,
    left_obj,
    right_title: str,
    right_obj,
    *,
    max_rows: int = 20,
    col_widths=(270, 270),
) -> Table:
    """Two arbitrary tables (Series/DataFrame) side-by-side with titles."""

    def obj_to_table(obj) -> Table:
        if isinstance(obj, pd.Series):
            tdf = obj.rename("value").to_frame()
            return df_to_table(tdf, max_rows=max_rows)
        if isinstance(obj, pd.DataFrame):
            return df_to_table(obj, max_rows=max_rows)
        # fallback: show as text
        return Table([[Paragraph(escape_paragraph_text(str(obj)), styles["Normal"])]])

    lt = obj_to_table(left_obj)
    rt = obj_to_table(right_obj)

    outer = Table(
        [
            [
                Paragraph(escape_paragraph_text(left_title), styles["Heading3"]),
                Paragraph(escape_paragraph_text(right_title), styles["Heading3"]),
            ],
            [lt, rt],
        ],
        colWidths=list(col_widths),
    )
    outer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return outer

def build_pdf_report(
    out_pdf: Path,
    *,
    title: str,
    subtitle: Optional[str] = None,
    top_countries: Optional[pd.Series] = None,
    top_manufacturers: Optional[pd.Series] = None,
    energy_country: Optional[pd.DataFrame] = None,
    country_sections: Optional[Iterable[dict]] = None,
    figures: Optional[Iterable[Path]] = None,
) -> Path:
    """Build a PDF report."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4)
    story = []

    story.append(Paragraph(escape_paragraph_text(title), styles["Title"]))

    # Subtitle centered
    if subtitle:
        story.append(Paragraph(escape_paragraph_text(subtitle), styles["SubtitleCenter"]))

    story.append(Spacer(1, 12))

    # Split Top countries into two columns
    if top_countries is not None:
        story.append(Paragraph("Top countries", styles["Heading2Center"]))
        story.append(Spacer(1, 8))
    
        s = top_countries.head(10)
        n = len(s)
        left_s = s.iloc[: (n + 1) // 2]
        right_s = s.iloc[(n + 1) // 2 :]
    
        story.append(
            two_column_tables(
                "Top countries (1/2)",
                left_s,
                "Top countries (2/2)",
                right_s,
                max_rows=5,   # how many per column
            )
        )
        story.append(Spacer(1, 12))

    # Split Top manufacturers into two columns
    if top_manufacturers is not None:
        story.append(Paragraph("Number of Cyclotrons per Manufacturer (Global)", styles["Heading2Center"]))
        story.append(Spacer(1, 8))

        s = top_manufacturers.head(10)
        n = len(s)
        left_s = s.iloc[: (n + 1) // 2]
        right_s = s.iloc[(n + 1) // 2 :]

        story.append(
            two_column_tables(
                "Top Manufacturers (1/2)",
                left_s,
                "Top Manufacturers (2/2)",
                right_s,
                max_rows=5,
            )
        )
        story.append(Spacer(1, 12))

    # Energy table split into two columns
    if energy_country is not None:
        story.append(Paragraph("Energy Distribution (Global)", styles["Heading2Center"]))
        story.append(Spacer(1, 8))

        s = energy_country.head(10)
        n = len(s)
        left_df = s.iloc[: (n + 1) // 2].copy()
        right_df = s.iloc[(n + 1) // 2 :].copy()

        story.append(
            two_column_tables(
                "Top Countries (1/2)",
                left_df,
                "Top Countries (2/2)",
                right_df,
                max_rows=5,
            )
        )
        story.append(Spacer(1, 12))

    # Figures
    if figures:
        story.append(Paragraph("Figures", styles["Heading2"]))
        story.append(Spacer(1, 8))
        for p in figures:
            p = Path(p)
            if p.exists():
                story.append(Image(str(p), width=520, height=320))
                story.append(Spacer(1, 10))
        story.append(PageBreak())

    
    # Country sections
    if country_sections:
        story.append(PageBreak())
        for sec in country_sections:
            ctry = sec.get("country", "")
            story.append(Paragraph(escape_paragraph_text(ctry), styles["Heading1"]))
            story.append(Spacer(1, 8))

            txt = sec.get("summary_md") or sec.get("summary")
            if txt:
                story.append(Paragraph(escape_paragraph_text(str(txt)), styles["BodyText"]))
                story.append(Spacer(1, 8))

            # NEW BLOCK HERE
            llm_report = sec.get("llm_report")
            if llm_report:
                story.append(Paragraph("Narrative summary", styles["Heading3"]))
                story.append(Paragraph(escape_paragraph_text(str(llm_report)), styles["BodyText"]))
                story.append(Spacer(1, 10))
            
            tables = sec.get("tables") or {}
            for name, obj in tables.items():
                story.append(Paragraph(escape_paragraph_text(str(name)), styles["Heading3"]))
                if isinstance(obj, pd.Series):
                    tdf = obj.rename("count").to_frame()
                    story.append(df_to_table(tdf, max_rows=20))
                elif isinstance(obj, pd.DataFrame):
                    story.append(df_to_table(obj, max_rows=20))
                else:
                    story.append(Paragraph(escape_paragraph_text(str(obj)), styles["Normal"]))
                story.append(Spacer(1, 10))

            imgs = sec.get("images") or []
            for p in imgs:
                p = Path(p)
                if p.exists():
                    story.append(Image(str(p), width=520, height=320))
                    story.append(Spacer(1, 10))

            story.append(PageBreak())

    doc.build(story)
    return out_pdf
