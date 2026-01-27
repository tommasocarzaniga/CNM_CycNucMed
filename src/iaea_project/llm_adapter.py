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
import json


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

def llm_describe_top_facility_openai(context: dict, model: str = "gpt-4.1-mini") -> str:
    # You already have OpenAI wiring in this file; reuse your client style.
    prompt = f"""
Write a 2–3 sentence description of the top cyclotron site in the country.
Use ONLY the facts provided below. Do NOT add any external knowledge.
If a detail is missing, say "unknown".

Facts (JSON):
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()

# return openai_chat(prompt, model=model)  # use your existing helper

def llm_top_site_blurb_openai(context: dict, model: str = "gpt-4.1-mini") -> str:
    """
    Produce a short (2–3 sentences) description of the top cyclotron site in a country,
    using ONLY the provided context dict (no external facts).

    Intended to be called once per country from pipeline.py and cached.
    """
    client = _require_openai_client()

    prompt = (
        "You are writing a short factual description for a PDF report.\n"
        "Write 2–3 sentences about the top cyclotron site in the country.\n"
        "Use ONLY the facts provided in the JSON below. Do NOT add external knowledge.\n"
        "If a detail is missing, say 'unknown'.\n\n"
        f"Facts (JSON):\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "Answer:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return (resp.choices[0].message.content or "").strip()
