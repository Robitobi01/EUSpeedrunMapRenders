from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_geojson_feature(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'GeoJSON "{path.as_posix()}" is not an object')
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != 1:
        raise ValueError(f'GeoJSON "{path.as_posix()}" must contain exactly one feature')
    feature = features[0]
    if not isinstance(feature, dict):
        raise ValueError(f'GeoJSON "{path.as_posix()}" has invalid feature data')
    return feature


def load_geojson_line_coordinates(path: Path) -> tuple[tuple[float, float], ...]:
    feature = load_geojson_feature(path)
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f'GeoJSON "{path.as_posix()}" has invalid geometry payload')
    if geometry.get("type") != "LineString":
        raise ValueError(f'GeoJSON "{path.as_posix()}" must be a LineString')
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError(f'GeoJSON "{path.as_posix()}" must contain at least two coordinates')
    points: list[tuple[float, float]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            continue
        points.append((float(coordinate[0]), float(coordinate[1])))
    if len(points) < 2:
        raise ValueError(f'GeoJSON "{path.as_posix()}" must contain at least two valid coordinates')
    return tuple(points)


def load_route_geo_points(path: Path) -> list[tuple[float, float]]:
    return [(lat, lon) for lon, lat in load_geojson_line_coordinates(path)]


def build_feature_collection(
    *,
    identifier: str,
    transport: str,
    profile: str,
    point_count: int,
    fallback_segments: int,
    coordinates: list[list[float]],
) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "path": identifier,
                    "transport": transport,
                    "profile": profile,
                    "pointCount": point_count,
                    "fallbackSegments": fallback_segments,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
            }
        ],
    }
