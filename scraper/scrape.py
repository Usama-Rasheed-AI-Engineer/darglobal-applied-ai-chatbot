"""Collect public text with provenance and fail safely when a site is unavailable."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public_sources.json"
SOURCES = [
    ("DarGlobal", "https://darglobal.co.uk/", "public",),
    ("DarGlobal", "https://darglobal.co.uk/", "luxury real estate developer investor"),
    ("Wasalt", "https://wasalt.sa/en", "properties for sale rent Saudi Arabia"),
]


def robots_allow(url: str) -> bool:
    parsed = httpx.URL(url)
    robots_url = f"{parsed.scheme}://{parsed.host}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        return parser.can_fetch("DarGlobal-AI-Assignment-Bot", url)
    except Exception:
        return True


def scrape() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for source, url, heading in SOURCES:
        if not robots_allow(url):
            continue
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": "DarGlobal-AI-Assignment-Bot/1.0"},
                timeout=20,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        if text:
            documents.append(
                {
                    "title": f"{source} public page - {heading}",
                    "text": text[:6000],
                    "url": str(response.url),
                    "source": source,
                    "collected_at": date.today().isoformat(),
                }
            )
    return documents


def main() -> None:
    scraped = scrape()
    if scraped:
        OUTPUT.write_text(json.dumps(scraped, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print("No pages were reachable; existing verified seed data was preserved.")


if __name__ == "__main__":
    main()
