from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from manim import BLACK, Circle, FullScreenRectangle, ImageMobject, RoundedRectangle, Scene, VGroup, WHITE, config
from manim.typing import Point3DLike
from tqdm import tqdm

from .geometry import as_point3, simplify_polyline, smooth_polyline
from .map_markers import MapMarker
from .route_visuals import TripRoute
from .runtime import tilemap_settings
from .tile_map import TileMap

PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"


class TileMapScene(Scene):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tm = self.create_tile_map(**tilemap_settings)
        self.tm_image: ImageMobject | None = None

    def create_tile_map(self, **kwargs: Any) -> TileMap:
        raise NotImplementedError

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = "#D9E7EB"
        self.tm_image = self.tm.make_image_mobject(self, set_width_to_frame=True)
        self.add(self.tm_image)

    def load_geojson(self, filename: str | Path) -> TripRoute:
        path = Path(filename)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        features = data.get("features")
        if not isinstance(features, list) or len(features) != 1:
            raise ValueError(f'GeoJSON "{path.as_posix()}" must contain exactly one feature')
        geometry = features[0].get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f'GeoJSON "{path.as_posix()}" has invalid geometry payload')
        geometry_type = geometry.get("type")
        if geometry_type != "LineString":
            raise ValueError(f'GeoJSON "{path.as_posix()}" must be a LineString, got {geometry_type!r}')
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list):
            raise ValueError(f'GeoJSON "{path.as_posix()}" has invalid coordinates')
        points = []
        with tqdm(
            total=len(coordinates),
            desc="Loading route geometry",
            unit="pt",
            bar_format=PROGRESS_BAR_FORMAT,
        ) as progress:
            for coordinate in coordinates:
                if not isinstance(coordinate, list) or len(coordinate) < 2:
                    progress.update(1)
                    continue
                lon = float(coordinate[0])
                lat = float(coordinate[1])
                points.append(self.tm.latlon_to_scene_coords(lat, lon, self))
                progress.update(1)
        shaped = [as_point3(point) for point in points]
        shaped = simplify_polyline(shaped)
        shaped = smooth_polyline(shaped)
        cache_path = self._route_lookup_cache_path(path)
        cache_signature = self._route_lookup_signature(path)
        cached_lookup = self._load_route_lookup(cache_path, cache_signature)
        route = TripRoute(shaped, distance_lookup=cached_lookup)
        if cached_lookup is None:
            self._save_route_lookup(cache_path, cache_signature, route)
        return route

    @staticmethod
    def _route_lookup_cache_path(path: Path) -> Path:
        cache_dir = Path("data/route_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{path.stem}.npz"

    @staticmethod
    def _route_lookup_signature(path: Path) -> str:
        stat = path.stat()
        return f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"

    @staticmethod
    def _load_route_lookup(path: Path, signature: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not path.exists():
            return None
        try:
            payload = np.load(path, allow_pickle=False)
            stored_signature = str(payload["signature"].item())
            if stored_signature != signature:
                return None
            fractions = np.asarray(payload["fractions"], dtype=float)
            proportions = np.asarray(payload["proportions"], dtype=float)
            if fractions.ndim != 1 or proportions.ndim != 1:
                return None
            if len(fractions) < 2 or len(proportions) < 2:
                return None
            return fractions, proportions
        except Exception:
            return None

    @staticmethod
    def _save_route_lookup(path: Path, signature: str, route: TripRoute) -> None:
        try:
            np.savez_compressed(
                path,
                signature=np.asarray(signature, dtype=np.str_),
                fractions=route._distance_fractions,
                proportions=route._distance_proportions,
            )
        except Exception:
            return

    def create_marker(self, label: str, lat: float, lon: float) -> MapMarker:
        scene_x, scene_y = self.tm.latlon_to_scene_coords(lat, lon, self)
        return MapMarker(label, np.array([scene_x, scene_y, 0.0], dtype=float))

    def create_soft_glow(
            self,
            center: Point3DLike,
            color: str,
            radii: list[float],
            opacities: list[float],
    ) -> VGroup:
        layers = VGroup()
        for radius, opacity in zip(radii, opacities):
            layers.add(Circle(radius=radius).set_fill(color, opacity=opacity).set_stroke(width=0).move_to(center))
        return layers

    def create_map_atmosphere(self, start: Point3DLike, end: Point3DLike) -> VGroup:
        frame_width = getattr(self.camera, "frame_width", float(config["frame_width"]))
        frame_height = getattr(self.camera, "frame_height", float(config["frame_height"]))
        veil = FullScreenRectangle(fill_color=BLACK, fill_opacity=0.06, stroke_width=0)
        border = RoundedRectangle(
            corner_radius=0.18,
            width=frame_width - 0.32,
            height=frame_height - 0.32,
        ).set_fill(opacity=0).set_stroke(WHITE, width=0.8, opacity=0.06)
        return VGroup(veil, border)
