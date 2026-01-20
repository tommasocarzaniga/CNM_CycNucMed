from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

def run_coro(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine in both:
    - normal Python (no running loop) -> asyncio.run
    - Jupyter/Colab (loop already running) -> nest_asyncio + run_until_complete
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: normal CLI usage
        return asyncio.run(coro)

    # Running loop: Jupyter/Colab
    try:
        import nest_asyncio  # pip dependency
        nest_asyncio.apply()
    except Exception as e:
        raise RuntimeError(
            "Running inside an existing event loop (Jupyter/Colab). "
            "Install nest_asyncio: pip install nest_asyncio"
        ) from e

    return loop.run_until_complete(coro)
