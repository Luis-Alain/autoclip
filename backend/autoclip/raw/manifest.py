"""``manifest.json`` and ``manifest.csv``: one row per clip, for Luis and for Premiere work."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import TypeAdapter

from .models import ManifestRow

#: Excerpts longer than this are cut — the CSV is for scanning, not reading.
EXCERPT_CHARS = 400

_ROWS = TypeAdapter(list[ManifestRow])


def write_manifest(rows: list[ManifestRow], directory: Path) -> tuple[Path, Path]:
    """Write both manifest files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "manifest.json"
    csv_path = directory / "manifest.csv"

    json_path.write_bytes(_ROWS.dump_json(rows, indent=2))

    # utf-8-sig adds a BOM so Excel and Numbers show accents and ñ correctly.
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())

    return json_path, csv_path


def read_manifest(path: Path) -> list[ManifestRow]:
    return _ROWS.validate_json(path.read_bytes())


def excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1].rstrip() + "…"
