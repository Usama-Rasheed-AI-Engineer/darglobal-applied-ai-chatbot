"""Collect public text with provenance and fail safely when a site is unavailable."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from curl_cffi.requests import Session
from lxml import etree
from selectolax.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval import chunk_text

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
        "https://wasalt.sa/en/properties-for-sale-in-jeddah",
        "https://wasalt.sa/en/projects",
    ],
}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def robots_allow(url: str) -> bool:
    parsed = httpx.URL(url)
    robots_url = f"{parsed.scheme}://{parsed.host}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        resp = httpx.get(robots_url, headers=BROWSER_HEADERS, timeout=8, follow_redirects=True)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
            return parser.can_fetch("DarGlobal-AI-Assignment-Bot", url)
        return True
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


def extract_page_text(raw_html: str) -> tuple[str, str]:
    parser = HTMLParser(raw_html)
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


def fetch_html(url: str) -> tuple[str, object] | None:
    jina_key = os.getenv("JINA_API_KEY", "").strip()
    scraperapi_key = os.getenv("SCRAPERAPI_KEY", "").strip()

    try:
        response = httpx.get(
            url,
            headers=BROWSER_HEADERS,
            timeout=10,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text, response
    except Exception:
        pass

    try:
        session = Session(impersonate="chrome")
        response = session.get(url, timeout=10, allow_redirects=True)
        if response.status_code < 400:
            return response.text, SimpleNamespace(url=str(response.url))
    except Exception:
        pass

    if jina_key:
        try:
            response = httpx.get(
                f"https://r.jina.ai/{url}",
                headers={"Authorization": f"Bearer {jina_key}", "Accept": "text/plain"},
                timeout=20,
            )
            response.raise_for_status()
            return response.text, SimpleNamespace(url=url)
        except Exception:
            pass

    if scraperapi_key:
        try:
            response = httpx.get(
                "https://api.scraperapi.com/",
                params={"api_key": scraperapi_key, "url": url},
                timeout=25,
            )
            response.raise_for_status()
            return response.text, response
        except Exception:
            pass

    return None


def discover_sitemap_urls(base_url: str, max_sitemaps: int = 5) -> list[str]:
    parsed = httpx.URL(base_url)
    sitemap_queue = [
        f"{parsed.scheme}://{parsed.host}/sitemap.xml",
        f"{parsed.scheme}://{parsed.host}/sitemap_index.xml",
    ]
    discovered: list[str] = []
    visited_sitemaps: set[str] = set()

    while sitemap_queue and len(visited_sitemaps) < max_sitemaps:
        sm_url = sitemap_queue.pop(0)
        if sm_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sm_url)
        print(f"[sitemap] Checking {sm_url} ...", flush=True)
        fetched = fetch_html(sm_url)
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
        if not locations:
            locations = re.findall(r"<loc(?:\s[^>]*)?>(.*?)</loc>", xml, flags=re.IGNORECASE | re.DOTALL)
        for url in locations:
            url = html.unescape(url.strip())
            if not url:
                continue
            if url.endswith(".xml") or "/sitemap" in url.lower():
                sitemap_queue.append(url)
            elif urlparse(url).netloc == parsed.host:
                discovered.append(url)
    return list(dict.fromkeys(discovered))


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


def scrape() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    today = date.today().isoformat()
    only_source = os.getenv("SCRAPE_SOURCE", "").strip()

    for source, base_url, _preferred_terms in SOURCES:
        if only_source and source.lower() != only_source.lower():
            continue
        print(f"\n=== Starting {source} ({base_url}) ===", flush=True)
        if not robots_allow(base_url):
            print(f"[{source}] Disallowed by robots.txt", flush=True)
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
        attempts = 0
        max_pages = int(os.getenv(
            "SCRAPE_MAX_DARGLOBAL" if source == "DarGlobal" else "SCRAPE_MAX_WASALT",
            "25" if source == "DarGlobal" else "25",
        ))
        max_attempts = max_pages * 3
        source_docs: list[dict[str, str]] = []

        while queue and crawled < max_pages and attempts < max_attempts:
            url = queue.pop(0)
            normalized = url.rstrip("/")
            if normalized in visited:
                continue
            visited.add(normalized)
            attempts += 1
            print(f"[{source}] ({crawled}/{max_pages}, try {attempts}/{max_attempts}) {url}", flush=True)

            fetched = fetch_html(url)
            if not fetched:
                print(f"  -> unavailable", flush=True)
                manifest.append({
                    "url": str(url),
                    "source": source,
                    "status": "unavailable",
                    "collected_at": today,
                })
                continue

            raw_html, response = fetched
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
                tag.decompose()

            extracted_text, extracted_title = extract_page_text(raw_html)
            title = extracted_title or f"{source} public page"
            if len(extracted_text) >= 200:
                page_chunks = chunk_text(extracted_text, str(response.url), source, title, today)
                source_docs.extend(page_chunks)
                documents.extend(page_chunks)
                crawled += 1
                print(f"  -> scraped ({len(page_chunks)} chunks): {title[:60]}", flush=True)
                manifest.append({
                    "url": str(response.url),
                    "source": source,
                    "title": title,
                    "status": "scraped",
                    "chunks": str(len(page_chunks)),
                    "collected_at": today,
                })
                time.sleep(float(os.getenv("SCRAPE_REQUEST_DELAY", "0.15")))
            else:
                print(f"  -> skipped (text too short: {len(extracted_text)} chars)", flush=True)

            if crawled < max_pages:
                for link in extract_links(soup, str(response.url), base.netloc):
                    if urlparse(link).scheme in ("http", "https") and link.rstrip("/") not in visited:
                        queue.append(link)

        # Save after each source finishes so progress is never lost
        URL_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if source_docs:
            combined = merge_sources(source_docs)
            OUTPUT.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"=== {source} complete: saved {len(source_docs)} new chunks (total DB: {len(combined)}) ===", flush=True)

    return documents


def main() -> None:
    scraped = scrape()
    if scraped:
        combined = merge_sources(scraped)
        OUTPUT.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDone! Total chunks in database: {len(combined)}.", flush=True)
    else:
        print("\nNo new pages were reachable; existing verified seed data was preserved.", flush=True)


if __name__ == "__main__":
    main()

