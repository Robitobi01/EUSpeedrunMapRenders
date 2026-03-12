from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from manim import FadeIn, config
from tqdm import tqdm

from rendering.core.map_markers import MapMarker
from rendering.core.map_scene import TileMapScene
from rendering.core.route_visuals import SpeedProfile, SpeedSegment, TripRoute
from rendering.core.tile_map import TileMap

from .models import PathPoint, PathSpec, load_path_specs, scene_name_for_identifier
from .routing import ensure_geojson_for_spec_with_status, find_cached_geojson

config.renderer = "cairo"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 60
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"


@dataclass(frozen=True, slots=True)
class RenderPath:
    spec: PathSpec
    scene_name: str


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6371.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


def _route_geo_points(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return []
    geometry = features[0].get("geometry", {}) if isinstance(features[0], dict) else {}
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list):
        return []
    points: list[tuple[float, float]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            continue
        lon = float(coordinate[0])
        lat = float(coordinate[1])
        points.append((lat, lon))
    return points


def _station_progresses_on_route(
        station_coords: list[tuple[float, float]],
        route_points: list[tuple[float, float]],
) -> list[float]:
    if len(station_coords) < 2 or len(route_points) < 2:
        return []
    cumulative = _cumulative_distances(route_points)
    total_distance = cumulative[-1]
    if total_distance <= 1e-6:
        return []
    anchors = [0]
    cursor = 0
    for lat, lon in station_coords[1:-1]:
        nearest = _nearest_route_index(route_points, lat, lon, cursor)
        if nearest < cursor:
            nearest = cursor
        anchors.append(nearest)
        cursor = nearest
    anchors.append(len(route_points) - 1)
    return [float(np.clip(cumulative[index] / total_distance, 0.0, 1.0)) for index in anchors]


def _cumulative_distances(points: list[tuple[float, float]]) -> list[float]:
    if not points:
        return []
    cumulative = [0.0]
    running = 0.0
    for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:]):
        running += _haversine_km(lat_a, lon_a, lat_b, lon_b)
        cumulative.append(running)
    return cumulative


def _nearest_route_index(points: list[tuple[float, float]], target_lat: float, target_lon: float,
                         min_index: int) -> int:
    best_index = min_index
    best_distance = float("inf")
    for index in range(min_index, len(points)):
        lat, lon = points[index]
        distance = _haversine_km(lat, lon, target_lat, target_lon)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _fallback_speed_profile() -> SpeedProfile:
    return SpeedProfile([SpeedSegment(0.0, 1.0, 90.0)])


def _normalize_hour_series(values: list[float]) -> list[float]:
    if not values:
        return []
    output = [float(value) for value in values]
    for index in range(1, len(output)):
        while output[index] < output[index - 1] - 1e-6:
            output[index] += 24.0
    for index in range(1, len(output)):
        if output[index] <= output[index - 1] + 1e-4:
            output[index] = output[index - 1] + 1 / 120
    return output


def _node_timeline_hour(node: PathPoint) -> float:
    if node.offset_minutes is None:
        raise ValueError("Every YAML path point must include offset_minutes")
    return float(node.offset_minutes) / 60.0


class PathScene(TileMapScene):
    path_spec: PathSpec

    def _speed_label_enabled(self) -> bool:
        start = self.path_spec.start
        end = self.path_spec.end
        return start.offset_minutes is not None and end.offset_minutes is not None

    def create_tile_map(self, **kwargs: Any) -> TileMap:
        center_lat, center_lon, zoom = self.path_spec.map_view()
        return TileMap(
            center_lat,
            center_lon,
            max(0.0, zoom - 0.35),
            **kwargs,
        )

    def _frame_size(self) -> tuple[float, float]:
        frame_width = getattr(self.camera, "frame_width", float(config["frame_width"]))
        frame_height = getattr(self.camera, "frame_height", float(config["frame_height"]))
        return float(frame_width), float(frame_height)

    @staticmethod
    def _downsample_points(points: Sequence[np.ndarray], max_points: int = 700) -> list[np.ndarray]:
        total = len(points)
        if total <= max_points:
            return [np.array(point, dtype=float) for point in points]
        step = max(1, total // max_points)
        sampled = [np.array(point, dtype=float) for point in points[::step]]
        if not np.array_equal(sampled[-1], np.array(points[-1], dtype=float)):
            sampled.append(np.array(points[-1], dtype=float))
        return sampled

    def _make_markers(self, route: TripRoute) -> tuple[MapMarker, MapMarker]:
        start_label = self.path_spec.start.name or self.path_spec.identifier
        end_label = self.path_spec.end.name or self.path_spec.identifier
        start_marker = MapMarker(
            start_label,
            route.start,
            label_direction=route.start_label_direction(),
        )
        end_marker = MapMarker(
            end_label,
            route.end,
            label_direction=route.end_label_direction(),
        )
        frame_width, frame_height = self._frame_size()
        marker_route_points = self._downsample_points(route.points)
        start_marker.choose_label_direction_with_route(frame_width, frame_height, marker_route_points, 0)
        end_marker.choose_label_direction_with_route(frame_width, frame_height, marker_route_points, -1)
        start_marker.clamp_label_within_frame(frame_width, frame_height)
        end_marker.clamp_label_within_frame(frame_width, frame_height)
        return start_marker, end_marker

    def _build_speed_profile(self, geojson_path: Path) -> tuple[SpeedProfile, list[float]]:
        points = self.path_spec.points()
        if len(points) < 2:
            return _fallback_speed_profile(), []
        station_coords = [(float(point.lat), float(point.lon)) for point in points]
        route_points = _route_geo_points(geojson_path)
        if len(route_points) < 2:
            return _fallback_speed_profile(), []
        route_total_distance = _cumulative_distances(route_points)[-1]
        station_progresses = _station_progresses_on_route(station_coords, route_points)
        if len(station_progresses) != len(points):
            return _fallback_speed_profile(), station_progresses
        timed_indices = [index for index, point in enumerate(points) if point.offset_minutes is not None]
        if len(timed_indices) < 2:
            return _fallback_speed_profile(), station_progresses
        timed_hours = [_node_timeline_hour(points[index]) for index in timed_indices]
        timeline = _normalize_hour_series(timed_hours)
        if len(timeline) != len(timed_indices):
            return _fallback_speed_profile(), station_progresses
        profile_segments: list[SpeedSegment] = []
        previous_speed: float | None = None
        for segment_index, ((start_point_index, end_point_index), (start_hour, end_hour)) in enumerate(
                zip(zip(timed_indices, timed_indices[1:]), zip(timeline, timeline[1:]))
        ):
            if end_point_index <= start_point_index:
                continue
            raw_duration = max(1 / 120, end_hour - start_hour)
            start_point = points[start_point_index]
            dwell_hours = max(0.0, float(start_point.stopped_minutes or 0) / 60.0)
            moving_duration = raw_duration - min(raw_duration * 0.85, dwell_hours)
            duration = max(1 / 120, moving_duration)
            start_progress = station_progresses[start_point_index]
            end_progress = station_progresses[end_point_index]
            if end_progress <= start_progress:
                continue
            distance = max(0.01, (end_progress - start_progress) * max(0.01, route_total_distance))
            base_speed = distance / duration
            shaped_speed = base_speed * (1.0 + 0.04 * math.sin(segment_index * 1.3 + 0.5))
            if previous_speed is not None:
                shaped_speed = previous_speed * 0.62 + shaped_speed * 0.38
            shaped_speed = float(np.clip(shaped_speed, 34.0, 220.0))
            start = start_progress
            end = end_progress
            if end - start <= 1e-5:
                continue
            profile_segments.append(SpeedSegment(float(start), float(end), shaped_speed))
            previous_speed = shaped_speed
        if not profile_segments:
            return _fallback_speed_profile(), station_progresses
        return SpeedProfile(profile_segments), station_progresses

    def construct(self) -> None:
        geojson_path = self._ensure_geojson_with_progress()
        route = self.load_geojson(geojson_path)
        station_coords = [(float(point.lat), float(point.lon)) for point in self.path_spec.points()]
        station_progresses = _station_progresses_on_route(station_coords, _route_geo_points(geojson_path))
        if self._speed_label_enabled():
            speed_profile, profiled_progresses = self._build_speed_profile(geojson_path)
            if len(profiled_progresses) == len(station_coords):
                station_progresses = profiled_progresses
        else:
            speed_profile = None
        start_marker, end_marker = self._make_markers(route)
        start_marker.add_to_scene(self, foreground=True)
        end_marker.add_to_scene(self, foreground=True)
        atmosphere = self.create_map_atmosphere(start_marker.dot.get_center(), end_marker.dot.get_center())
        self.play(FadeIn(atmosphere, run_time=0.8))
        self.play(*start_marker.animate_creation())
        start_marker.show_final_state()
        self.play(*end_marker.animate_creation())
        end_marker.show_final_state()
        marker_front = [*start_marker.foreground_mobjects(), *end_marker.foreground_mobjects()]
        route.create_and_animate(
            self,
            keep_on_top=marker_front,
            speed_profile=speed_profile,
            station_progresses=station_progresses,
        )
        self.wait(0.4)

    def _ensure_geojson_with_progress(self) -> Path:
        cached = find_cached_geojson(self.path_spec)
        if cached is not None:
            with tqdm(
                total=1,
                desc="Using geojson cache",
                unit="seg",
                bar_format=PROGRESS_BAR_FORMAT,
            ) as progress:
                progress.update(1)
            return cached
        fallback_total = max(1, len(self.path_spec.route_points()) - 1)
        with tqdm(
            total=1,
            desc="Downloading geojson route",
            unit="seg",
            bar_format=PROGRESS_BAR_FORMAT,
        ) as progress:
            def update(event: dict[str, int | str]) -> None:
                total = max(1, int(event.get("total_segments", fallback_total)))
                done = max(0, min(total, int(event.get("done_segments", 0))))
                progress.total = total
                progress.n = done
                progress.refresh()

            path, _ = ensure_geojson_for_spec_with_status(self.path_spec, progress_callback=update)
        return path


PATH_SPECS = load_path_specs()
PATH_SCENE_NAMES: list[str] = []
SCENE_NAME_BY_PATH: dict[str, str] = {}
PATH_RENDER_MANIFEST: list[RenderPath] = []


def _register_path_scenes() -> None:
    for spec in PATH_SPECS:
        scene_name = scene_name_for_identifier(spec.identifier)
        scene_class = type(scene_name, (PathScene,), {"path_spec": spec})
        globals()[scene_name] = scene_class
        PATH_SCENE_NAMES.append(scene_name)
        SCENE_NAME_BY_PATH[spec.identifier] = scene_name
        PATH_RENDER_MANIFEST.append(RenderPath(spec=spec, scene_name=scene_name))


def get_scene_name_for_path(identifier: str) -> str:
    try:
        return SCENE_NAME_BY_PATH[identifier]
    except KeyError as exc:
        raise KeyError(f'Unknown path "{identifier}"') from exc


_register_path_scenes()
