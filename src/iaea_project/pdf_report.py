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

    # Global comparison page (countries vs manufacturers)
    if top_countries is not None and top_manufacturers is not None:
        story.append(Paragraph("Global snapshot", styles["Heading2Center"]))
        story.append(Spacer(1, 8))
        story.append(
            two_column_toplists(
                "Top countries",
                top_countries,
                "Top manufacturers (global)",
                top_manufacturers,
                max_rows=10,
            )
        )
        story.append(Spacer(1, 12))

    # Split Top manufacturers into two columns
    if top_manufacturers is not None:
        story.append(Paragraph("Top manufacturers (global)", styles["Heading2Center"]))
        story.append(Spacer(1, 8))

        s = top_manufacturers.copy()
        n = len(s)
        left_s = s.iloc[: (n + 1) // 2]
        right_s = s.iloc[(n + 1) // 2 :]

        story.append(
            two_column_tables(
                "Top manufacturers (1/2)",
                left_s,
                "Top manufacturers (2/2)",
                right_s,
                max_rows=20,
            )
        )
        story.append(Spacer(1, 12))

    # Energy table split into two columns
    if energy_country is not None:
        story.append(Paragraph("Global Snapshot", styles["Heading2Center"]))
        story.append(Spacer(1, 8))

        n = len(energy_country)
        left_df = energy_country.iloc[: (n + 1) // 2].copy()
        right_df = energy_country.iloc[(n + 1) // 2 :].copy()

        story.append(
            two_column_tables(
                "Number of cyclotrons and energy distribution (1/2)",
                left_df,
                "Number of cyclotrons and energy distribution (2/2)",
                right_df,
                max_rows=20,
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
        for sec in country_sections:
            ctry = sec.get("country", "")
            story.append(Paragraph(escape_paragraph_text(ctry), styles["Heading1"]))
            story.append(Spacer(1, 8))

            txt = sec.get("summary_md") or sec.get("summary")
            if txt:
                story.append(Paragraph(escape_paragraph_text(str(txt)), styles["BodyText"]))
                story.append(Spacer(1, 8))

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
