# src/iaea_project/web_enrichment.py
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .utils import CACHE_DIR


@dataclass(frozen=True)
class SerperConfig:
    api_key: str = os.getenv("SERPER_API_KEY", "")
    country_gl: str = os.getenv("SERPER_GL", "ch")      # country bias (gl)
    language_hl: str = os.getenv("SERPER_HL", "en")     # UI language (hl)
    k: int = int(os.getenv("SERPER_K", "5"))            # top results
    timeout: int = int(os.getenv("SERPER_TIMEOUT", "20"))
    cache_json: Path = CACHE_DIR / "serper_facility_cache.json"
    max_chars_per_page: int = int(os.getenv("SERPER_MAX_CHARS_PER_PAGE", "6000"))
    max_pages_to_fetch: int = int(os.getenv("SERPER_MAX_PAGES", "2"))  # fetch top N pages for text


def _stable_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _strip_html_to_text(html: str) -> str:
    # very lightweight “good enough” stripper (no bs4 dependency)
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def serper_search(query: str, cfg: SerperConfig) -> List[Dict[str, Any]]:
    if not cfg.api_key:
        raise RuntimeError("SERPER_API_KEY is not set")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": cfg.api_key, "Content-Type": "application/json"}
    payload = {"q": query, "gl": cfg.country_gl, "hl": cfg.language_hl, "num": cfg.k}

    r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
    r.raise_for_status()
    data = r.json()

    out: List[Dict[str, Any]] = []
    for item in (data.get("organic") or [])[: cfg.k]:
        out.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "position": item.get("position"),
                "source": "serper_organic",
            }
        )
    return out


def fetch_pages_text(urls: List[str], cfg: SerperConfig) -> List[Dict[str, Any]]:
    texts: List[Dict[str, Any]] = []
    for u in urls[: cfg.max_pages_to_fetch]:
        try:
            resp = requests.get(u, timeout=cfg.timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                continue
            text = _strip_html_to_text(resp.text)
            if len(text) > cfg.max_chars_per_page:
                text = text[: cfg.max_chars_per_page]
            texts.append({"url": u, "text": text})
        except Exception:
            continue
    return texts


def enrich_facility(ctx: Dict[str, Any], cfg: Optional[SerperConfig] = None) -> Dict[str, Any]:
    """
    Enrich a top-facility context dict with web evidence:
    - serper_results: [{title, link, snippet, ...}]
    - page_texts: [{url, text}]
    Cached by (country|facility|city).
    """
    cfg = cfg or SerperConfig()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cfg.cache_json)
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    else:
        cache = {}

    country = str(ctx.get("country") or "").strip()
    facility = str(ctx.get("top_facility") or "").strip()
    city = str(ctx.get("top_city") or "").strip()

    base_key = f"{country}|{facility}|{city}"
    key = _stable_key(base_key)

    if key in cache:
        ctx["web_evidence"] = cache[key]
        return ctx

    # Build queries (few, precise)
    queries = [
        f'"{facility}" radiopharmacy {city} {country}',
        f'"{facility}" cyclotron {city} {country}',
        f'"{facility}" PET radiopharmaceuticals {city} {country}',
    ]

    all_results: List[Dict[str, Any]] = []
    for q in queries:
        try:
            all_results.extend(serper_search(q, cfg))
        except Exception:
            continue

    # Deduplicate links
    seen = set()
    uniq = []
    for r in all_results:
        link = r.get("link")
        if not link or link in seen:
            continue
        seen.add(link)
        uniq.append(r)

    top_links = [r["link"] for r in uniq if r.get("link")]
    page_texts = fetch_pages_text(top_links, cfg)

    evidence = {
        "query_key": base_key,
        "queries": queries,
        "serper_results": uniq[: cfg.k],
        "page_texts": page_texts,
    }

    cache[key] = evidence
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["web_evidence"] = evidence
    return ctx
