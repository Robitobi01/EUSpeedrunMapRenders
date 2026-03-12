from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

MAX_MERCATOR_LAT = 85.0511287798066
DEFAULT_PATHS_DIR = Path("data/paths")
TransportType = Literal["train", "walking", "bus"]
ALLOWED_TRANSPORTS: tuple[TransportType, ...] = ("train", "walking", "bus")


def slugify(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "path"


def scene_name_for_identifier(identifier: str) -> str:
    return normalize_path_identifier(identifier)


def normalize_path_identifier(identifier: str) -> str:
    return slugify(identifier)


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


def load_yaml_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if payload is not None else {}


def dump_yaml_data(value: Any) -> str:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def write_yaml_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            value,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


@dataclass(frozen=True, slots=True)
class PathPoint:
    lat: float
    lon: float
    name: str | None = None
    offset_minutes: int | None = None
    stopped_minutes: int | None = None

    def as_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lat": float(self.lat),
            "lon": float(self.lon),
        }
        if self.name:
            payload["name"] = self.name
        if self.offset_minutes is not None:
            payload["offset_minutes"] = int(self.offset_minutes)
        if self.stopped_minutes is not None:
            payload["stopped_minutes"] = int(self.stopped_minutes)
        return payload


@dataclass(frozen=True, slots=True)
class PathSpec:
    identifier: str
    transport: TransportType
    start: PathPoint
    end: PathPoint
    waypoints: tuple[PathPoint, ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if self.transport not in ALLOWED_TRANSPORTS:
            raise ValueError(f"Unsupported transport type: {self.transport!r}")
        if not self.identifier.strip():
            raise ValueError("Path identifier cannot be empty")
        if not self.start.name or not self.start.name.strip():
            raise ValueError(f'Path "{self.identifier}" start.name is required')
        if not self.end.name or not self.end.name.strip():
            raise ValueError(f'Path "{self.identifier}" end.name is required')

    @property
    def file_stem(self) -> str:
        return normalize_path_identifier(self.identifier)

    def points(self) -> tuple[PathPoint, ...]:
        return (self.start, *self.waypoints, self.end)

    def route_points(self) -> list[PathPoint]:
        deduped: list[PathPoint] = []
        for point in self.points():
            if deduped and math.isclose(deduped[-1].lat, point.lat) and math.isclose(deduped[-1].lon, point.lon):
                continue
            deduped.append(point)
        return deduped

    def map_view(self) -> tuple[float, float, float]:
        coordinates = [[point.lon, point.lat] for point in self.route_points()]
        return compute_center_zoom(coordinates)

    def as_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.name:
            payload["name"] = self.name
        payload["transport"] = self.transport
        payload["start"] = self.start.as_mapping()
        payload["end"] = self.end.as_mapping()
        if self.waypoints:
            payload["waypoints"] = [point.as_mapping() for point in self.waypoints]
        return payload


def load_path_spec(path: Path) -> PathSpec:
    payload = load_yaml_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Path config "{path.as_posix()}" must be a mapping')
    transport_raw = payload.get("transport")
    if not isinstance(transport_raw, str):
        raise ValueError(f'Path config "{path.as_posix()}" is missing transport')
    transport = transport_raw.strip().lower()
    if transport not in ALLOWED_TRANSPORTS:
        raise ValueError(
            f'Path config "{path.as_posix()}" transport must be one of {", ".join(ALLOWED_TRANSPORTS)}'
        )
    start = _parse_point(payload.get("start"), "start", path)
    end = _parse_point(payload.get("end"), "end", path)
    waypoints_raw = payload.get("waypoints", [])
    if waypoints_raw is None:
        waypoints_raw = []
    if not isinstance(waypoints_raw, list):
        raise ValueError(f'Path config "{path.as_posix()}" waypoints must be a list')
    waypoints = tuple(_parse_point(item, f"waypoints[{index}]", path) for index, item in enumerate(waypoints_raw))
    name_raw = payload.get("name")
    name = str(name_raw).strip() if isinstance(name_raw, str) and name_raw.strip() else None
    return PathSpec(
        identifier=path.stem,
        transport=transport,
        start=start,
        end=end,
        waypoints=waypoints,
        name=name,
    )


def load_path_specs(paths_dir: Path = DEFAULT_PATHS_DIR) -> list[PathSpec]:
    if not paths_dir.exists():
        return []
    files = sorted([*paths_dir.glob("*.yaml"), *paths_dir.glob("*.yml")], key=lambda path: path.name)
    return [load_path_spec(path) for path in files]


def write_path_spec(path: Path, spec: PathSpec) -> None:
    write_yaml_file(path, spec.as_mapping())


def _parse_point(value: Any, label: str, path: Path) -> PathPoint:
    if not isinstance(value, dict):
        raise ValueError(f'Path config "{path.as_posix()}" field {label} must be a mapping')
    lat_raw = value.get("lat")
    lon_raw = value.get("lon")
    if lat_raw is None or lon_raw is None:
        raise ValueError(f'Path config "{path.as_posix()}" field {label} requires lat/lon')
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except Exception as exc:
        raise ValueError(f'Path config "{path.as_posix()}" field {label} has invalid lat/lon') from exc
    name_raw = value.get("name")
    name = str(name_raw).strip() if isinstance(name_raw, str) and name_raw.strip() else None
    minutes_raw = value.get("offset_minutes")
    offset_minutes: int | None
    if minutes_raw is None:
        offset_minutes = None
    else:
        try:
            offset_minutes = int(minutes_raw)
        except Exception as exc:
            raise ValueError(f'Path config "{path.as_posix()}" field {label} has invalid offset_minutes') from exc
    stopped_raw = value.get("stopped_minutes")
    stopped_minutes: int | None
    if stopped_raw is None:
        stopped_minutes = None
    else:
        try:
            stopped_minutes = int(stopped_raw)
        except Exception as exc:
            raise ValueError(f'Path config "{path.as_posix()}" field {label} has invalid stopped_minutes') from exc
    return PathPoint(
        lat=lat,
        lon=lon,
        name=name,
        offset_minutes=offset_minutes,
        stopped_minutes=stopped_minutes,
    )
