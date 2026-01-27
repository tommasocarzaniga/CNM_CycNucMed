# CNM_CycNucMed --- IAEA Cyclotron Atlas

A full **end-to-end data pipeline** for scraping, cleaning, analyzing,
visualizing, and reporting on the **IAEA Cyclotron Master List**.\
The project outputs a professional **PDF report**, structured datasets,
and optional geographic visualizations.

> Designed as an academic-quality data engineering + analytics project,
> suitable for portfolios, coursework, and applied research.

------------------------------------------------------------------------

## ✨ Features

-   🔎 **Web scraping** of the IAEA Cyclotron Master List\
-   🧹 **Data cleaning & canonicalization**
    -   Country normalization (ISO3)
    -   Manufacturer normalization
    -   Energy parsing (MeV → numeric)
-   📊 **Statistical analysis**
    -   Top countries, manufacturers, facilities
    -   Per-country summaries
    -   Energy distributions
-   🗺️ **Geographic visualization**
    -   City geocoding (cached)
    -   Per-country maps with cyclotron locations
-   📄 **Automated PDF report generation**
    -   Tables, summaries, figures
    -   Country-by-country sections
-   🤖 Optional **LLM-assisted normalization** (OpenAI adapters)

------------------------------------------------------------------------

## 📁 Project Structure

    CNM_CycNucMed/
    │
    ├── data/
    │   ├── raw/                 # Raw scraped CSV
    │   └── processed/           # Cleaned dataset
    │
    ├── outputs/
    │   ├── figures/             # Generated maps
    │   └── reports/             # Final PDF report
    │
    ├── src/iaea_project/
    │   ├── pipeline.py          # Orchestrates the full pipeline
    │   ├── scraper.py           # Data collection
    │   ├── cleaning.py          # Data cleaning & normalization
    │   ├── analysis.py          # Aggregations & statistics
    │   ├── plotting.py          # Maps & visualizations
    │   ├── pdf_report.py        # PDF layout and generation
    │   └── utils.py             # Paths & helpers
    │
    └── README.md

------------------------------------------------------------------------

## 🚀 Quick Start

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Run the pipeline

``` python
from iaea_project.pipeline import run_pipeline

# Run full pipeline for all countries
pdf_path = run_pipeline()

# Or restrict to selected countries
pdf_path = run_pipeline(["Switzerland", "Italy"])

print("Report generated at:", pdf_path)
```

Output will be created under:

    outputs/reports/IAEA_Cyclotron_Report.pdf

------------------------------------------------------------------------

## 🧠 Pipeline Architecture

The project explicitly follows a **multi-step pipeline architecture**
(as required by many coursework guidelines):

1.  **Scraping step**
    -   Downloads and parses structured data from IAEA sources.
2.  **Cleaning step**
    -   Normalizes text fields (countries, manufacturers)
    -   Adds derived columns (numeric energy)
    -   Produces a reproducible clean dataset
3.  **Analysis step**
    -   Computes statistics, rankings, and summaries
    -   Produces structured tables for reporting
4.  **Visualization step (optional)**
    -   Geocodes cities
    -   Generates map figures
5.  **Reporting step**
    -   Assembles everything into a formatted PDF report

Each step consumes the output of the previous one, forming a **true data
pipeline**.

------------------------------------------------------------------------

## 📊 Example Outputs

The pipeline automatically generates:

-   `data/processed/iaea_cyclotrons_clean.csv`\
-   Country maps like:
    -   `outputs/figures/Italy_map.png`
    -   `outputs/figures/Switzerland_map.png`
-   Final report:
    -   `outputs/reports/IAEA_Cyclotron_Report.pdf`

The report includes: - Global top countries and manufacturers - Energy
statistics - Country-by-country summaries - Tables and figures

------------------------------------------------------------------------

## 🧩 Core Functions

### `run_pipeline(...)`

Main orchestration entrypoint:

``` python
run_pipeline(
    countries=None,
    max_country_sections=None,
    disable_maps=False,
    enable_llm=False,
    llm_model="gpt-4.1-mini"
)
```

### `country_summary(df, country)`

Returns detailed per-country statistics: - Total cyclotrons - Top
cities, facilities, manufacturers - Energy statistics

### `global_comparison_tables(df)`

Returns: - Top countries worldwide - Top manufacturers worldwide -
Energy distribution by country

------------------------------------------------------------------------

## 🤖 Optional LLM Integration

If enabled:

``` python
run_pipeline(enable_llm=True)
```

The pipeline can use LLMs to: - Fix ambiguous country names - Improve
manufacturer normalization

Adapters live in:

    src/iaea_project/llm_adapters.py

This is fully optional --- the project works without any API keys.

------------------------------------------------------------------------

## 🎯 Use Cases

-   Academic projects (data pipelines, applied analytics)
-   Portfolio demonstration (Python, Pandas, modular architecture)
-   Research on nuclear medicine infrastructure
-   Visualization and reporting automation

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**.\
Data originates from publicly available IAEA sources and may contain
inconsistencies.

------------------------------------------------------------------------

## 👤 Author

**Tommaso Carzaniga**\
EMBA Candidate -- University of St. Gallen (HSG)\
Radiopharmaceutical industry professional

------------------------------------------------------------------------

If you'd like, I can also provide: - A version optimized for **academic
grading rubrics** - A more **technical engineering README** - A shorter
**portfolio-style README**
