from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from playwright.async_api import async_playwright


@dataclass(frozen=True)
class ScraperConfig:
    base_url: str = "https://nucleus.iaea.org/sites/accelerators/Pages/Cyclotron.aspx"
    # This is the SharePoint list-view hash that was used in your notebook.
    hash_prefix: str = "#InplviewHashd5afe566-18ad-4ac0-8aeb-ccf833dbc282="
    expected_cols: int = 6
    table_row_selector: str = "table.ms-listviewtable tbody tr"


HEADER_LABELS = [
    "country",
    "city",
    "facility",
    "manufacturer",
    "model",
    "proton energy (mev)",
    "proton energy",
]
HEADER_SET = set(HEADER_LABELS)


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def _row_fingerprint(cells: Iterable[str]) -> str:
    norm = [" ".join((c or "").split()) for c in cells]
    return hashlib.md5(" | ".join(norm).encode("utf-8")).hexdigest()


def _strip_header_prefix(cell: str) -> str:
    """Convert 'City: Vienna' -> 'Vienna' (for known header labels)."""
    c = (cell or "").strip()
    c_norm = _norm_text(c)
    for h in HEADER_LABELS:
        if re.match(rf"^{re.escape(h)}(\s*[:\-]?\s+)", c_norm):
            return re.sub(rf"(?i)^{re.escape(h)}\s*[:\-]?\s+", "", c).strip()
    return c


def _flatten_multiline_cells(raw_cells: Iterable[Any]) -> list[str]:
    tokens: list[str] = []
    for c in raw_cells:
        if not c:
            continue
        parts = re.split(r"[\t\r\n]+", str(c))
        for part in parts:
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


def _clean_and_align_tokens(raw_cells: Iterable[Any], expected_cols: int) -> list[str] | None:
    """Convert raw table cells -> exactly expected_cols cleaned fields."""
    tokens = _flatten_multiline_cells(raw_cells)

    processed: list[str] = []
    for t in tokens:
        t_norm = _norm_text(t)
        if t_norm in HEADER_SET:
            continue

        t2 = _strip_header_prefix(t)
        t2_norm = _norm_text(t2)
        if not t2 or t2_norm in HEADER_SET:
            continue

        processed.append(t2.strip())

    if len(processed) < expected_cols:
        return None

    if len(processed) == expected_cols:
        if any(_norm_text(x) in HEADER_SET for x in processed):
            return None
        return processed

    # More tokens than expected: choose best window of length expected_cols
    def badness(x: str) -> int:
        xn = _norm_text(x)
        if xn in HEADER_SET:
            return 100
        if xn.isdigit():
            return 10
        if any(xn.startswith(h + " ") for h in HEADER_LABELS):
            return 10
        return 0

    best_window: list[str] | None = None
    best_score: int | None = None

    for start in range(0, len(processed) - expected_cols + 1):
        window = processed[start : start + expected_cols]
        if any(_norm_text(w) in HEADER_SET for w in window):
            continue
        score = sum(badness(w) for w in window)
        if best_score is None or score < best_score:
            best_score = score
            best_window = window

    return best_window


async def _wait_for_table_refresh(page, prev_first_row_text: str, timeout_ms: int = 20000) -> None:
    """After clicking Next, wait until the first row content changes."""
    try:
        await page.wait_for_function(
            """(prev) => {
                const r = document.querySelector('table.ms-listviewtable tbody tr');
                return r && r.innerText && r.innerText !== prev;
            }""",
            arg=prev_first_row_text,
            timeout=timeout_ms,
        )
    except Exception:
        await page.wait_for_timeout(1200)


async def _robust_click(el) -> bool:
    """Scroll into view then force-click; fallback to JS click."""
    try:
        await el.scroll_into_view_if_needed()
    except Exception:
        pass

    try:
        await el.click(force=True, timeout=20000)
        return True
    except Exception:
        try:
            await el.evaluate("node => node.click()")
            return True
        except Exception:
            return False


async def scrape_all_pages_async(config: ScraperConfig = ScraperConfig()) -> pd.DataFrame:
    """Scrape the full IAEA SharePoint list view into a DataFrame."""
    all_rows: list[dict[str, str]] = []
    seen_rows: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Block heavy resources for speed.
        await context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "font", "media", "stylesheet")
            else route.continue_(),
        )

        page = await context.new_page()
        page.set_default_timeout(20000)

        url = config.base_url + config.hash_prefix
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")

        while True:
            # Ensure rows exist
            try:
                await page.wait_for_selector(config.table_row_selector, timeout=20000)
            except Exception:
                break

            raw_rows = await page.eval_on_selector_all(
                config.table_row_selector,
                """trs => trs.map(tr =>
                    Array.from(tr.querySelectorAll('td')).map(td => td.innerText)
                )""",
            )

            for raw_cells in raw_rows:
                cells = _clean_and_align_tokens(raw_cells, expected_cols=config.expected_cols)
                if cells is None or len(cells) != config.expected_cols:
                    continue
                if any(_norm_text(x) in HEADER_SET for x in cells):
                    continue

                fp = _row_fingerprint(cells)
                if fp in seen_rows:
                    continue
                seen_rows.add(fp)

                all_rows.append(
                    {
                        "Country": cells[0],
                        "City": cells[1],
                        "Facility": cells[2],
                        "Manufacturer": cells[3],
                        "Model": cells[4],
                        "Proton energy (MeV)": cells[5],
                    }
                )

            # Capture first row text to detect refresh
            try:
                prev_first = await page.locator(config.table_row_selector).first.inner_text()
            except Exception:
                prev_first = ""

            next_el = page.locator(
                'a[title="Next"], a[aria-label="Next"], a[title="Next page"], '
                'a[aria-label="Next page"], a:has-text("Next")'
            ).first

            if await next_el.count() == 0:
                break
            if not await next_el.is_visible() or not await next_el.is_enabled():
                break

            ok = await _robust_click(next_el)
            if not ok:
                break

            await _wait_for_table_refresh(page, prev_first)

        await context.close()
        await browser.close()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["Country", "City"], kind="mergesort").reset_index(drop=True)
    return df


def scrape_iaea_cyclotrons(config: ScraperConfig = ScraperConfig()) -> pd.DataFrame:
    """Synchronous wrapper around the async scraper (CLI + Jupyter safe)."""
    coro = scrape_all_pages_async(config=config)

    try:
        # If this works, we're inside an existing event loop (Jupyter/Colab)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → normal script execution
        return asyncio.run(coro)

    # Running loop → Jupyter/Colab
    # We need nest_asyncio to allow nested loop.run_until_complete
    try:
        import nest_asyncio  # pip install nest_asyncio
        nest_asyncio.apply()
    except Exception as e:
        raise RuntimeError(
            "You're running inside a Jupyter/Colab event loop. "
            "Install nest_asyncio: pip install nest_asyncio"
        ) from e

    return loop.run_until_complete(coro)

def save_raw(df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
