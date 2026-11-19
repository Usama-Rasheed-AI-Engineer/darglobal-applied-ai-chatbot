"""Collect public text with provenance and fail safely when a site is unavailable."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval import chunk_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public_sources.json"
URL_MANIFEST = ROOT / "data" / "source_manifest.json"
SOURCES = [
    ("DarGlobal", "https://darglobal.co.uk/", ("project", "aida", "tierra-viva", "davinci", "d-villas", "properties", "about", "news")),
    ("Wasalt", "https://wasalt.sa/", ("properties", "sale", "rent", "project", "riyadh", "jeddah")),
]
SEED_URLS = {
    "https://darglobal.co.uk/": [
        "https://darglobal.co.uk/landing-page/aida",
        "https://darglobal.co.uk/tierra-viva",
        "https://darglobal.co.uk/davinci-tower-by-pagani",
        "https://darglobal.co.uk/d-villas-at-jge?slug=dg1",
        "https://darglobal.co.uk/projects/villas",
        "https://darglobal.co.uk/projects/apartments",
        "https://darglobal.co.uk/projects/hotel-residences",
        "https://darglobal.co.uk/projects/collections",
        "https://darglobal.co.uk/why-invest",
        "https://darglobal.co.uk/faq",
        "https://darglobal.co.uk/news",
        "https://darglobal.co.uk/about",
    ],
    "https://wasalt.sa/": [
        "https://wasalt.sa/en",
        "https://wasalt.sa/en/properties-for-sale-in-riyadh",
        "https://wasalt.sa/en/properties-for-rent-in-riyadh",
    ],
}


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


def extract_links(soup: BeautifulSoup, page_url: str, allowed_host: str) -> list[str]:
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(page_url, anchor["href"]).split("#")[0]
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.netloc == allowed_host:
            links.append(url)
    return list(dict.fromkeys(links))


def fetch_html(url: str) -> tuple[str, httpx.Response] | None:
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": "DarGlobal-AI-Assignment-Bot/1.0"},
            timeout=25,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text, response
    except httpx.HTTPError:
        return None


def discover_sitemap_urls(base_url: str) -> list[str]:
    parsed = httpx.URL(base_url)
    roots = [
        f"{parsed.scheme}://{parsed.host}/sitemap.xml",
        f"{parsed.scheme}://{parsed.host}/sitemap_index.xml",
    ]
    discovered: list[str] = []
    visited_sitemaps: set[str] = set()
    sitemap_queue = roots
    while sitemap_queue:
        root = sitemap_queue.pop(0)
        if root in visited_sitemaps:
            continue
        visited_sitemaps.add(root)
        fetched = fetch_html(root)
        if not fetched:
            continue
        xml, _ = fetched
        soup = BeautifulSoup(xml, "xml")
        for location in soup.find_all("loc"):
            url = location.get_text(strip=True)
            if not url:
                continue
            if url.endswith(".xml") or "/sitemap" in url.lower():
                sitemap_queue.append(url)
            elif urlparse(url).netloc == parsed.host:
                discovered.append(url)
    return list(dict.fromkeys(discovered))


def scrape() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    today = date.today().isoformat()

    for source, base_url, preferred_terms in SOURCES:
        if not robots_allow(base_url):
            continue
        base = urlparse(base_url)
        queue = [base_url]
        queue.extend(SEED_URLS.get(base_url, []))
        sitemap_urls = discover_sitemap_urls(base_url)
        def url_priority(url: str) -> tuple[int, str]:
            low = url.lower()
            if "/landing-page/" in low or "/tierra-viva" in low or "/aida" in low:
                return (0, url)
            if "/projects/" in low or low.rstrip("/").endswith("d-villas-at-jge"):
                return (1, url)
            if "/insights/" in low or "/press/" in low or "/blog/" in low:
                return (2, url)
            return (3, url)

        queue.extend(sorted(dict.fromkeys(sitemap_urls), key=url_priority))
        visited: set[str] = set()
        crawled = 0
        max_pages = 300 if source == "DarGlobal" else 200

        while queue and crawled < max_pages:
            url = queue.pop(0)
            normalized = url.rstrip("/")
            if normalized in visited:
                continue
            visited.add(normalized)
            fetched = fetch_html(url)
            if not fetched:
                manifest.append({
                    "url": str(url),
                    "source": source,
                    "status": "unavailable",
                    "collected_at": today,
                })
                continue
            html, response = fetched
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
                tag.decompose()

            title_tag = soup.find("title")
            heading_tag = soup.find(["h1", "h2"])
            title = title_tag.get_text(" ", strip=True) if title_tag else heading_tag.get_text(" ", strip=True) if heading_tag else f"{source} public page"

            content_root = soup.find("main") or soup.find("article") or soup.find("body")
            text = " ".join(content_root.stripped_strings) if content_root else ""
            if len(text) >= 200:
                documents.extend(chunk_text(text, str(response.url), source, title, today))
                crawled += 1
                manifest.append({
                    "url": str(response.url),
                    "source": source,
                    "title": title,
                    "status": "scraped",
                    "chunks": str(len(chunk_text(text, str(response.url), source, title, today))),
                    "collected_at": today,
                })

            if crawled < max_pages:
                for link in extract_links(soup, str(response.url), base.netloc):
                    if urlparse(link).scheme in ("http", "https") and link.rstrip("/") not in visited:
                        queue.append(link)

    URL_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return documents


def merge_sources(scraped: list[dict[str, str]]) -> list[dict[str, str]]:
    existing: list[dict[str, str]] = []
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    combined = existing + scraped
    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for document in combined:
        key = (
            document.get("url", "").split("#")[0],
            document.get("title", ""),
            document.get("text", "")[:300],
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(document)
    return deduplicated


def main() -> None:
    scraped = scrape()
    if scraped:
        combined = merge_sources(scraped)
        OUTPUT.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(combined)} source chunks.")
    else:
        print("No pages were reachable; existing verified seed data was preserved.")


if __name__ == "__main__":
    main()
