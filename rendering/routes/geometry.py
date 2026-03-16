from __future__ import annotations

import math

import numpy as np

MAX_MERCATOR_LAT = 85.0511287798066


def clamp_lat(lat: float) -> float:
    return max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)


def latlon_to_global_pixel(lat: float, lon: float, zoom: float, tile_size: int = 256) -> tuple[float, float]:
    lat = clamp_lat(lat)
    sin_lat = math.sin(math.radians(lat))
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * tile_size
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * tile_size
    return x, y


def compute_center_zoom(
    coordinates: list[list[float]],
    width_px: int = 1920,
    height_px: int = 1080,
    padding: float = 1.2,
) -> tuple[float, float, float]:
    if not coordinates:
        return 50.0, 10.0, 4.0
    lons = [float(coord[0]) for coord in coordinates]
    lats = [float(coord[1]) for coord in coordinates]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    if math.isclose(min_lon, max_lon) and math.isclose(min_lat, max_lat):
        return center_lat, center_lon, 14.0
    for step in range(64, 7, -1):
        zoom = step / 4.0
        left, top = latlon_to_global_pixel(max_lat, min_lon, zoom)
        right, bottom = latlon_to_global_pixel(min_lat, max_lon, zoom)
        span_x = abs(right - left) * padding
        span_y = abs(bottom - top) * padding
        if span_x <= width_px and span_y <= height_px:
            return center_lat, center_lon, zoom
    return center_lat, center_lon, 2.0


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6371.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


def cumulative_distances(points: list[tuple[float, float]]) -> list[float]:
    if not points:
        return []
    cumulative = [0.0]
    running = 0.0
    for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:]):
        running += haversine_km(lat_a, lon_a, lat_b, lon_b)
        cumulative.append(running)
    return cumulative


def nearest_route_index(
    points: list[tuple[float, float]],
    target_lat: float,
    target_lon: float,
    min_index: int,
) -> int:
    best_index = min_index
    best_distance = float("inf")
    for index in range(min_index, len(points)):
        lat, lon = points[index]
        distance = haversine_km(lat, lon, target_lat, target_lon)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def station_progresses_on_route(
    station_coords: list[tuple[float, float]],
    route_points: list[tuple[float, float]],
) -> list[float]:
    if len(station_coords) < 2 or len(route_points) < 2:
        return []
    cumulative = cumulative_distances(route_points)
    total_distance = cumulative[-1]
    if total_distance <= 1e-6:
        return []
    anchors = [0]
    cursor = 0
    for lat, lon in station_coords[1:-1]:
        nearest = nearest_route_index(route_points, lat, lon, cursor)
        if nearest < cursor:
            nearest = cursor
        anchors.append(nearest)
        cursor = nearest
    anchors.append(len(route_points) - 1)
    return [float(np.clip(cumulative[index] / total_distance, 0.0, 1.0)) for index in anchors]
