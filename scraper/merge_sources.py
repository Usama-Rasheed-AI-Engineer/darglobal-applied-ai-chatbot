import json
from pathlib import Path


def main() -> None:
    artifact_root = Path("artifacts")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, str]] = []
    if (data_dir / "public_sources.json").exists():
        combined.extend(json.loads((data_dir / "public_sources.json").read_text(encoding="utf-8")))

    manifest: list[dict[str, str]] = []
    for file in artifact_root.glob("*/public_sources.json"):
        combined.extend(json.loads(file.read_text(encoding="utf-8")))
    for file in artifact_root.glob("*/source_manifest.json"):
        manifest.extend(json.loads(file.read_text(encoding="utf-8")))

    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in combined:
        key = (row.get("url", ""), row.get("text", "")[:300])
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)

    (data_dir / "public_sources.json").write_text(
        json.dumps(deduplicated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Merged {len(deduplicated)} source rows")


if __name__ == "__main__":
    main()
