"""Collect public text with provenance and fail safely when a site is unavailable."""

from __future__ import annotations

import json
import re
import os
import time
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from curl_cffi.requests import Session
from lxml import etree
from selectolax.parser import HTMLParser

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
        "https://wasalt.sa/sitemap.xml",
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


def extract_page_text(html: str) -> tuple[str, str]:
    parser = HTMLParser(html)
    for tag in ("script", "style", "nav", "footer", "noscript", "svg"):
        for node in parser.css(tag):
            node.decompose()

    main = parser.css_first("main")
    article = parser.css_first("article")
    body = parser.css_first("body")
    root = main or article or body
    text = root.text(separator=" ", strip=True) if root else ""

    title_node = parser.css_first("title")
    heading_node = parser.css_first("h1")
    title = title_node.text(strip=True) if title_node else heading_node.text(strip=True) if heading_node else ""
    return re.sub(r"\s+", " ", text).strip(), re.sub(r"\s+", " ", title).strip()


def fetch_html(url: str) -> tuple[str, httpx.Response] | None:
    jina_key = os.getenv("JINA_API_KEY", "").strip()
    scraperapi_key = os.getenv("SCRAPERAPI_KEY", "").strip()
    try:
        session = Session(impersonate="chrome")
        response = session.get(url, timeout=25, allow_redirects=True)
        if response.status_code >= 400:
            raise httpx.HTTPError(f"{response.status_code}")
        final_url = str(response.url)
        return response.text, SimpleNamespace(url=final_url)
    except Exception:
        pass
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
        pass

    if jina_key:
        try:
            response = httpx.get(
                f"https://r.jina.ai/{url}",
                headers={"Authorization": f"Bearer {jina_key}", "Accept": "text/plain"},
                timeout=45,
            )
            response.raise_for_status()
            return response.text, SimpleNamespace(url=url)
        except httpx.HTTPError:
            pass

    if scraperapi_key:
        try:
            response = httpx.get(
                "https://api.scraperapi.com/",
                params={"api_key": scraperapi_key, "url": url},
                timeout=45,
            )
            response.raise_for_status()
            return response.text, response
        except httpx.HTTPError:
            pass

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
        try:
            root = etree.fromstring(xml.encode("utf-8"))
            locations = root.xpath(
                "//*[local-name()='url']/*[local-name()='loc']/text()"
                " | "
                "//*[local-name()='sitemap']/*[local-name()='loc']/text()"
            )
        except etree.XMLSyntaxError:
            locations = []
        for url in locations:
            url = url.strip()
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
            if source == "Wasalt":
                if "/project/" in low or "/property/" in low:
                    return (0, url)
                if "/en/villas-" in low or "/en/apartments-" in low or "/en/properties-" in low:
                    return (1, url)
                if "/corporate" in low or "/privacy" in low or "/terms" in low:
                    return (4, url)
            return (3, url)

        queue.extend(sorted(dict.fromkeys(sitemap_urls), key=url_priority))
        visited: set[str] = set()
        crawled = 0
        max_pages = int(os.getenv(
            "SCRAPE_MAX_DARGLOBAL" if source == "DarGlobal" else "SCRAPE_MAX_WASALT",
            "500" if source == "DarGlobal" else "1000",
        ))

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
                time.sleep(float(os.getenv("SCRAPE_REQUEST_DELAY", "0.15")))
                continue
            html, response = fetched
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
                tag.decompose()

            extracted_text, extracted_title = extract_page_text(html)
            title = extracted_title or f"{source} public page"
            text = extracted_text
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
                time.sleep(float(os.getenv("SCRAPE_REQUEST_DELAY", "0.15")))

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
