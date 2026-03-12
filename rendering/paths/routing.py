from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import requests

from .models import PathPoint, PathSpec

DEFAULT_BROUTER_URL = "https://brouter.de/brouter"
BROUTER_TIMEOUT_SECONDS = 10
DEFAULT_GEOJSON_ROOT = Path("data/geojson")

TRANSPORT_BROUTER_PROFILE = {
    "train": "rail",
    "walking": "trekking",
    "bus": "car-fast",
}


class RouteProgressEvent(TypedDict, total=False):
    type: str
    total_segments: int
    done_segments: int
    routed_segments: int
    fallback_segments: int


def geojson_path_for_spec(spec: PathSpec, root: Path = DEFAULT_GEOJSON_ROOT) -> Path:
    return root / spec.transport / f"{spec.file_stem}.geojson"


def _geojson_candidates_for_spec(spec: PathSpec, root: Path) -> list[Path]:
    return [root / spec.transport / f"{spec.file_stem}.geojson"]


def ensure_geojson_for_spec(
        spec: PathSpec,
        root: Path = DEFAULT_GEOJSON_ROOT,
        endpoint: str = DEFAULT_BROUTER_URL,
        timeout: int = BROUTER_TIMEOUT_SECONDS,
) -> Path:
    path, _ = ensure_geojson_for_spec_with_status(
        spec=spec,
        root=root,
        endpoint=endpoint,
        timeout=timeout,
    )
    return path


def find_cached_geojson(spec: PathSpec, root: Path = DEFAULT_GEOJSON_ROOT) -> Path | None:
    for candidate in _geojson_candidates_for_spec(spec, root=root):
        if not candidate.exists():
            continue
        try:
            load_existing_route(candidate)
            return candidate
        except Exception:
            continue
    return None


def ensure_geojson_for_spec_with_status(
        spec: PathSpec,
        root: Path = DEFAULT_GEOJSON_ROOT,
        endpoint: str = DEFAULT_BROUTER_URL,
        timeout: int = BROUTER_TIMEOUT_SECONDS,
        progress_callback: Callable[[RouteProgressEvent], None] | None = None,
) -> tuple[Path, bool]:
    target = geojson_path_for_spec(spec, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = find_cached_geojson(spec, root=root)
    if cached is not None:
        return cached, True
    profile = TRANSPORT_BROUTER_PROFILE.get(spec.transport)
    if profile is None:
        raise ValueError(f"Unsupported transport type: {spec.transport!r}")
    points = spec.route_points()
    if len(points) < 2:
        raise ValueError(f'Path "{spec.identifier}" must have at least two distinct coordinates')
    with requests.Session() as session:
        coordinates, fallback_segments = route_path(
            points=points,
            profile=profile,
            timeout=timeout,
            endpoint=endpoint,
            session=session,
            progress_callback=progress_callback,
        )
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "path": spec.identifier,
                    "transport": spec.transport,
                    "profile": profile,
                    "pointCount": len(points),
                    "fallbackSegments": fallback_segments,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
            }
        ],
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return target, False


def load_existing_route(path: Path) -> tuple[list[list[float]], int]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f'GeoJSON "{path.as_posix()}" has no features')
    feature = features[0] if isinstance(features[0], dict) else {}
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError(f'GeoJSON "{path.as_posix()}" has invalid coordinates')
    clean: list[list[float]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            continue
        clean.append([float(coordinate[0]), float(coordinate[1])])
    if len(clean) < 2:
        raise ValueError(f'GeoJSON "{path.as_posix()}" has fewer than two valid coordinates')
    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    fallback_segments = int(properties.get("fallbackSegments", 0) or 0)
    return clean, fallback_segments


def route_path(
        points: list[PathPoint],
        profile: str,
        timeout: int,
        endpoint: str,
        session: requests.Session,
        progress_callback: Callable[[RouteProgressEvent], None] | None = None,
) -> tuple[list[list[float]], int]:
    total_segments = max(1, len(points) - 1)
    if progress_callback is not None:
        progress_callback(
            {
                "type": "start",
                "total_segments": total_segments,
                "done_segments": 0,
                "routed_segments": 0,
                "fallback_segments": 0,
            }
        )
    segments: list[list[list[float]]] = []
    fallback_segments = 0
    routed_segments = 0
    for index in range(len(points) - 1):
        pair = [points[index], points[index + 1]]
        try:
            payload = request_brouter_route(
                points=pair,
                profile=profile,
                timeout=timeout,
                endpoint=endpoint,
                session=session,
            )
            coordinates = extract_route_coordinates(payload)
            routed_segments += 1
        except Exception:
            fallback_segments += 1
            coordinates = fallback_segment_coords(pair[0], pair[1])
        segments.append(coordinates)
        if progress_callback is not None:
            progress_callback(
                {
                    "type": "segment",
                    "total_segments": total_segments,
                    "done_segments": index + 1,
                    "routed_segments": routed_segments,
                    "fallback_segments": fallback_segments,
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "type": "done",
                "total_segments": total_segments,
                "done_segments": total_segments,
                "routed_segments": routed_segments,
                "fallback_segments": fallback_segments,
            }
        )
    return concat_coordinates(segments), fallback_segments


def request_brouter_route(
        points: list[PathPoint],
        profile: str,
        timeout: int,
        endpoint: str,
        session: requests.Session,
) -> dict[str, Any]:
    lonlats = "|".join(f"{point.lon},{point.lat}" for point in points)
    response = session.get(
        endpoint,
        params={
            "lonlats": lonlats,
            "profile": profile,
            "alternativeidx": 0,
            "format": "geojson",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        snippet = response.text[:400].replace("\n", " ")
        raise RuntimeError(f"BRouter request failed ({response.status_code}): {snippet}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("BRouter response is not a JSON object")
    return payload


def extract_route_coordinates(route_payload: dict[str, Any]) -> list[list[float]]:
    features = route_payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("BRouter response has no features")
    geometry = features[0].get("geometry") if isinstance(features[0], dict) else {}
    if not isinstance(geometry, dict):
        raise ValueError("BRouter response geometry is invalid")
    geometry_type = geometry.get("type")
    raw_coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        coordinates = raw_coordinates
    elif geometry_type == "MultiLineString":
        coordinates = []
        for segment in raw_coordinates or []:
            if not isinstance(segment, list):
                continue
            if not coordinates:
                coordinates.extend(segment)
                continue
            if segment and isinstance(segment[0], list) and coordinates[-1][:2] == segment[0][:2]:
                coordinates.extend(segment[1:])
            else:
                coordinates.extend(segment)
    else:
        raise ValueError(f"Unsupported route geometry type: {geometry_type}")
    clean: list[list[float]] = []
    for coordinate in coordinates or []:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            continue
        lon = float(coordinate[0])
        lat = float(coordinate[1])
        if clean and math.isclose(clean[-1][0], lon) and math.isclose(clean[-1][1], lat):
            continue
        clean.append([lon, lat])
    if len(clean) < 2:
        raise ValueError("Route has fewer than two coordinates")
    return clean


def concat_coordinates(parts: list[list[list[float]]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for part in parts:
        for coordinate in part:
            if merged and math.isclose(merged[-1][0], coordinate[0]) and math.isclose(merged[-1][1], coordinate[1]):
                continue
            merged.append(coordinate)
    if len(merged) < 2:
        raise ValueError("Merged route has fewer than two coordinates")
    return merged


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


def fallback_segment_coords(start: PathPoint, end: PathPoint) -> list[list[float]]:
    lat1, lon1 = float(start.lat), float(start.lon)
    lat2, lon2 = float(end.lat), float(end.lon)
    distance_km = haversine_km(lat1, lon1, lat2, lon2)
    steps = max(6, min(72, int(distance_km / 12)))
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)
    p1 = (
        math.cos(phi1) * math.cos(lambda1),
        math.cos(phi1) * math.sin(lambda1),
        math.sin(phi1),
    )
    p2 = (
        math.cos(phi2) * math.cos(lambda2),
        math.cos(phi2) * math.sin(lambda2),
        math.sin(phi2),
    )
    dot = max(-1.0, min(1.0, p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]))
    omega = math.acos(dot)
    if omega < 1e-9:
        return [[lon1, lat1], [lon2, lat2]]
    sin_omega = math.sin(omega)
    coordinates: list[list[float]] = []
    for index in range(steps + 1):
        t = index / steps
        a = math.sin((1 - t) * omega) / sin_omega
        b = math.sin(t * omega) / sin_omega
        x = a * p1[0] + b * p2[0]
        y = a * p1[1] + b * p2[1]
        z = a * p1[2] + b * p2[2]
        norm = math.sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        coordinates.append([lon, lat])
    return coordinates
