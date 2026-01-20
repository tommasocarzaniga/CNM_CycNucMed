"""
CLI entrypoint for the cyclotron pipeline.

Examples:
  python scripts/cyclotrons_run.py --countries Switzerland Germany
  python scripts/cyclotrons_run.py --skip-scrape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iaea_project.pipeline import run_pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the IAEA cyclotron pipeline and generate a PDF report.",
    )
    p.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Countries to include. If omitted, include ALL countries found in the dataset.",
    )
    p.add_argument(
        "--max-country-sections",
        type=int,
        default=0,
        help="Safety cap on number of country sections (0 = no cap).",
    )
    p.add_argument(
        "--disable-maps",
        action="store_true",
        help="Skip generating country map figures.",
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable optional OpenAI helpers (needs OPENAI_API_KEY).",
    )
    p.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model name (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--out-pdf",
        default=None,
        help="Output PDF path (default: outputs/reports/IAEA_Cyclotron_Report.pdf).",
    )
    p.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Do not scrape; read data/raw/iaea_cyclotrons_raw.csv.",
    )
    return p.parse_args(args=argv)


def main(argv: list[str] | None = None) -> Path:
    """
    CLI-safe main.

    In notebooks, call main([]) to avoid ipykernel-injected arguments.
    """
    if argv is None and "ipykernel" in sys.modules:
        argv = []

    args = _parse_args(argv=argv)

    cap = None if (args.max_country_sections or 0) <= 0 else int(args.max_country_sections)

    return run_pipeline(
        countries=args.countries,
        max_country_sections=cap,
        disable_maps=bool(args.disable_maps),
        enable_llm=bool(args.enable_llm),
        llm_model=str(args.llm_model),
        out_pdf=args.out_pdf,
        skip_scrape=bool(args.skip_scrape),
    )


if __name__ == "__main__":
    main()
