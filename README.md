# IAEA Cyclotron Report (project skeleton)

This repo is the "split into files" version of your `EMBA_Exam.ipynb` pipeline.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Install browser for playwright
playwright install chromium

# Run the end-to-end pipeline (scrape -> clean -> analyze -> map -> pdf)
python scripts/run_full.py

# Quick test: create clean CSV and print a country report to stdout
python scripts/run_test.py

# Unit tests
pytest -q
```

## Folder layout

- `data/raw/` : raw scrape output CSV
- `data/processed/` : cleaned CSV + caches
- `outputs/figures/` : plots / maps
- `outputs/reports/` : PDFs
- `src/iaea_project/` : all Python logic
- `scripts/` : runnable entry points

## Notes

- `cleaning.canonicalize_countries()` optionally supports an LLM fallback via `llm_fix_country(raw)->str`.
- `cleaning.canonicalize_manufacturers()` optionally supports LLM-driven canonical matching via `llm_choose(raw, canon_list)->str`.
- `plotting.save_country_map()` depends on optional GIS deps and remote datasets; it will silently return `None` if unavailable.
