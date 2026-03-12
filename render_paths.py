from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PATH_SPECS: list[object] = []
PATH_SCENE_NAMES: list[str] = []
PATH_RENDER_MANIFEST: list[object] = []
SCENE_NAME_BY_PATH: dict[str, str] = {}

if __name__ != "__main__":
    from rendering.paths import scenes as _impl

    PathScene = _impl.PathScene
    PATH_SPECS = list(_impl.PATH_SPECS)
    PATH_SCENE_NAMES = list(_impl.PATH_SCENE_NAMES)
    PATH_RENDER_MANIFEST = list(_impl.PATH_RENDER_MANIFEST)
    SCENE_NAME_BY_PATH = dict(_impl.SCENE_NAME_BY_PATH)
    get_scene_name_for_path = _impl.get_scene_name_for_path
    load_path_specs = _impl.load_path_specs

    for _scene_name in PATH_SCENE_NAMES:
        _base = getattr(_impl, _scene_name)
        globals()[_scene_name] = type(_scene_name, (_base,), {"__module__": __name__})
else:
    from rendering.paths.models import load_path_specs, scene_name_for_identifier

    def get_scene_name_for_path(identifier: str) -> str:
        specs = load_path_specs()
        for spec in specs:
            if spec.identifier == identifier:
                return scene_name_for_identifier(spec.identifier)
        raise KeyError(f'Unknown path "{identifier}"')


def _scene_name_for_input(path_input: str) -> str:
    candidate = Path(path_input)
    identifier = candidate.stem if candidate.suffix else path_input
    try:
        return get_scene_name_for_path(identifier)
    except KeyError:
        known = ", ".join(spec.identifier for spec in load_path_specs())
        raise SystemExit(f'Unknown path "{identifier}". Known paths: {known}') from None


def _run_manim(scene_name: str, quality: str) -> int:
    command = [sys.executable, "-m", "manim", "-q", quality, __file__, scene_name]
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render_paths.py")
    parser.add_argument("path_file", help="Path yaml file or path identifier")
    parser.add_argument("-q", "--quality", default="h")
    args = parser.parse_args(argv)
    scene_name = _scene_name_for_input(args.path_file)
    return _run_manim(scene_name, args.quality)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PathScene",
    "PATH_SPECS",
    "PATH_SCENE_NAMES",
    "PATH_RENDER_MANIFEST",
    "SCENE_NAME_BY_PATH",
    "get_scene_name_for_path",
    "load_path_specs",
    *PATH_SCENE_NAMES,
]
