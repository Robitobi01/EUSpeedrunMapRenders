from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "path"


def scene_name_for_identifier(identifier: str) -> str:
    return slugify(identifier)


def display_name_pair_for_identifier(identifier: str) -> tuple[str, str]:
    cleaned = re.sub(r"^\d+[_-]+", "", identifier.strip())
    start_raw, separator, end_raw = cleaned.partition("_to_")
    if not separator or not start_raw or not end_raw:
        raise ValueError(f'Identifier "{identifier}" must use "_to_" to separate start and end station names')
    return humanize_station_name(start_raw), humanize_station_name(end_raw)


def humanize_station_name(value: str) -> str:
    text = re.sub(r"[_-]+", " ", value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        raise ValueError("Station name cannot be empty")
    return text
