from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manimpango import register_font

FONT_PATH = Path("data/OpenSans-Bold.ttf")
TILEMAP_SETTINGS_PATH = Path("tilemap_settings.json")


def register_default_font(font_path: Path = FONT_PATH) -> None:
    if not font_path.exists():
        return
    try:
        register_font(str(font_path))
    except Exception:
        return


def load_tilemap_settings(path: Path = TILEMAP_SETTINGS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {}


register_default_font()
tilemap_settings = load_tilemap_settings()
