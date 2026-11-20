import json
import re
from datetime import date
from pathlib import Path

OUTPUT_FILE = Path("enriched_rag_data.jsonl")
SCRAPE_DATE = date.today().isoformat()

LEGACY_PATTERNS = [
    r"golden\s*visa.*spain",
    r"spain.*golden\s*visa",
    r"500,?000\s*(eur|€).*residen",
]

COUNTRY_KEYWORDS = {
    "UAE": ["dubai", "uae", "united arab emirates", "ras al khaimah"],
    "Oman": ["oman", "muscat", "aida"],
    "Saudi Arabia": [
        "saudi", "riyadh", "jeddah", "makkah", "madinah", "wasalt",
        "الرياض", "جدة", "مكة", "المدينة", "وصلت",
    ],
    "Qatar": ["qatar", "doha", "les vagues"],
    "Spain": ["spain", "marbella", "benahavis", "costa del sol", "tierra viva"],
    "UK": ["uk", "united kingdom", "london", "old park lane"],
}

SKIP_NAMES = {
    "source_manifest.json",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "enriched_rag_data.jsonl",
}

def find_dataset() -> Path:
    for candidate in [Path("public_sources.json"), Path("../public_sources.json")]:
        if candidate.exists():
            return candidate.resolve()

    candidates = [
        p for p in list(Path(".").rglob("*.json*")) + list(Path("..").rglob("*.json*"))
        if p.name not in SKIP_NAMES and "node_modules" not in p.parts and ".venv" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError("Could not find public_sources.json or any scraped JSON dataset.")
    return max(candidates, key=lambda p: p.stat().st_size).resolve()

def extract_url_and_text(record: dict) -> tuple[str, str]:
    existing_meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    url = (
        record.get("url")
        or record.get("source_url")
        or record.get("link")
        or existing_meta.get("sourceURL")
        or existing_meta.get("url")
        or ""
    )
    text = (
        record.get("markdown")
        or record.get("content")
        or record.get("text")
        or record.get("description")
        or ""
    )
    return str(url), str(text)

def classify_record(record: dict) -> dict | None:
    url, text = extract_url_and_text(record)
    if not text.strip():
        return None

    url_lower = url.lower()
    combined = f"{url_lower} {text.lower()}"

    platform = "wasalt" if "wasalt" in url_lower else "darglobal"

    if any(seg in url_lower for seg in ["/blog/", "/press/", "/news/", "/insights/", "/articles/"]):
        source_type = "blog_or_press"
    elif any(seg in url_lower for seg in ["/projects/", "/project/", "/developments/", "/properties/"]):
        source_type = "project_page"
    elif platform == "wasalt":
        source_type = "platform_listing"
    else:
        source_type = "general_page"

    is_legacy = source_type == "blog_or_press" and any(
        re.search(pat, combined) for pat in LEGACY_PATTERNS
    )

    matched_countries = [
        country for country, kws in COUNTRY_KEYWORDS.items()
        if any(kw in combined for kw in kws)
    ]

    return {
        "text": text,
        "metadata": {
            "source_url": url,
            "platform": platform,
            "source_type": source_type,
            "policy_status": "legacy_verify" if is_legacy else "active",
            "countries": ", ".join(matched_countries) if matched_countries else "Unspecified",
            "last_scraped": SCRAPE_DATE,
        },
    }

def load_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

if __name__ == "__main__":
    input_path = find_dataset()
    print(f"Found dataset: {input_path}")

    records = load_records(input_path)
    if records and isinstance(records[0], dict):
        print(f"Record keys: {list(records[0].keys())}")

    written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for rec in records:
            if isinstance(rec, dict):
                enriched = classify_record(rec)
                if enriched:
                    out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Done: {written} enriched records saved to {OUTPUT_FILE.resolve()}")