"""Optional LLM adapters.

This project runs without any LLM.

If you want to demonstrate LLM integration, pass the callables defined here into:
`iaea_project.cleaning.clean_cyclotron_df(..., llm_fix_country=..., llm_choose_manufacturer=...)`.

These adapters use the OpenAI Python SDK (`pip install openai`) and expect an API key
in the environment variable `OPENAI_API_KEY`.
"""

from __future__ import annotations

import os
from typing import List


def _require_openai_client():
    """Return an OpenAI client class instance.

    We keep provider-specific imports inside this function so the project can run
    without OpenAI installed.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK not installed. Install with: pip install openai"
        ) from e

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set in the environment."
        )

    return OpenAI()


def llm_fix_country_openai(raw_country: str, model: str = "gpt-4.1-mini") -> str:
    """Fix messy country strings.

    Returns a short, human country name (e.g., 'United States', 'Cote d'Ivoire').
    The cleaning layer will still validate and map it via `country_converter`.
    """
    client = _require_openai_client()

    prompt = (
        "You are normalizing country names. "
        "Given an input string, return ONLY the correct country name (short form). "
        "Do not add punctuation, quotes, or extra words.\n\n"
        f"Input: {raw_country}\n"
        "Output:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return (resp.choices[0].message.content or "").strip()


def llm_choose_manufacturer_openai(
    raw_manufacturer: str,
    canon_list: List[str],
    model: str = "gpt-4.1-mini",
) -> str:
    """Choose a manufacturer canonical form from an evolving canon list.

    Contract (must match cleaning.canonicalize_manufacturers):
    - Return EXACTLY one string from `canon_list`, OR
    - Return "NEW:<name>" to propose a new canonical manufacturer.
    """
    client = _require_openai_client()

    canon_preview = "\n".join(f"- {c}" for c in canon_list)
    prompt = (
        "You are deduplicating cyclotron manufacturer names.\n"
        "Rules:\n"
        "1) If the raw manufacturer matches an existing canonical, output that EXACT canonical string.\n"
        "2) If it does not match, output NEW:<canonical name>.\n"
        "3) Output ONLY the answer, nothing else.\n\n"
        f"Raw: {raw_manufacturer}\n\n"
        "Canonical options:\n"
        f"{canon_preview}\n\n"
        "Answer:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return (resp.choices[0].message.content or "").strip()
