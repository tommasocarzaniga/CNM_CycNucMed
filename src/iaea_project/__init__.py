"""IAEA cyclotron list project (scrape -> clean -> analyze -> plot -> PDF)."""

__all__ = [
    "scraper",
    "cleaning",
    "analysis",
    "plotting",
    "pdf_report",
    "utils",
    "run",   # NEW
]

def _resolve_run():
    """
    Resolve a stable run() entrypoint from common module layouts.
    Keeps package imports clean while supporting notebooks and CLI usage.
    """
    candidates = [
        ("run", "run"),
        ("runner", "run"),
        ("main", "main"),
        ("cli", "main"),
        ("__main__", "main"),
    ]

    for module_name, func_name in candidates:
        try:
            module = __import__(f"{__name__}.{module_name}", fromlist=[func_name])
            func = getattr(module, func_name, None)
            if callable(func):
                return func
        except Exception:
            continue

    raise ImportError(
        "No runnable entrypoint found. "
        "Expected one of: run.run(), runner.run(), main.main(), cli.main(), __main__.main()"
    )

# Expose package-level run()
run = _resolve_run()
