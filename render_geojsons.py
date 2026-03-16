from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GEOJSON_SPECS: list[object] = []
GEOJSON_SCENE_NAMES: list[str] = []
GEOJSON_RENDER_MANIFEST: list[object] = []
SCENE_NAME_BY_GEOJSON: dict[str, str] = {}

if __name__ != "__main__":
    from rendering.routes import scenes as _impl

    GeoJSONScene = _impl.GeoJSONScene
    GEOJSON_SPECS = list(_impl.GEOJSON_SPECS)
    GEOJSON_SCENE_NAMES = list(_impl.GEOJSON_SCENE_NAMES)
    GEOJSON_RENDER_MANIFEST = list(_impl.GEOJSON_RENDER_MANIFEST)
    SCENE_NAME_BY_GEOJSON = dict(_impl.SCENE_NAME_BY_GEOJSON)
    get_scene_name_for_geojson = _impl.get_scene_name_for_geojson
    load_geojson_specs = _impl.load_geojson_specs

    for _scene_name in GEOJSON_SCENE_NAMES:
        _base = getattr(_impl, _scene_name)
        globals()[_scene_name] = type(_scene_name, (_base,), {"__module__": __name__})
else:
    from rendering.routes.manual import load_geojson_specs, scene_name_for_identifier


    def get_scene_name_for_geojson(identifier: str) -> str:
        specs = load_geojson_specs()
        for spec in specs:
            if spec.identifier == identifier:
                return scene_name_for_identifier(spec.identifier)
        raise KeyError(f'Unknown GeoJSON "{identifier}"')


def _scene_name_for_input(scene_input: str) -> str:
    candidate = Path(scene_input)
    identifier = candidate.stem if candidate.suffix else scene_input
    try:
        return get_scene_name_for_geojson(identifier)
    except KeyError:
        known = ", ".join(spec.path.name for spec in load_geojson_specs())
        raise SystemExit(f'Unknown GeoJSON "{identifier}". Known scenes: {known}') from None


def _available_scenes_text() -> str:
    specs = load_geojson_specs()
    if not specs:
        return "Available scenes:\n(none found in data/geojson/manual)"
    return "Available scenes:\n" + "\n".join(spec.path.name for spec in specs)


def _run_manim(scene_name: str, quality: str) -> int:
    command = [sys.executable, "-m", "manim", "-q", quality, __file__, scene_name]
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render_geojsons.py")
    parser.add_argument("scene", nargs="?")
    parser.add_argument("-q", "--quality", default="h")
    parser.add_argument("--list-scenes", action="store_true")
    args = parser.parse_args(argv)
    if args.list_scenes or not args.scene:
        print(_available_scenes_text())
        return 0
    scene_name = _scene_name_for_input(args.scene)
    return _run_manim(scene_name, args.quality)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GeoJSONScene",
    "GEOJSON_SPECS",
    "GEOJSON_SCENE_NAMES",
    "GEOJSON_RENDER_MANIFEST",
    "SCENE_NAME_BY_GEOJSON",
    "get_scene_name_for_geojson",
    "load_geojson_specs",
    *GEOJSON_SCENE_NAMES,
]
