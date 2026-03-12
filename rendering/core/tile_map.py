from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from manim import ImageMobject, Scene, config
from tqdm import tqdm

MAX_MERCATOR_LAT = 85.0511287798066
DEFAULT_TILE_SIZE = 256
DEFAULT_SUBDOMAINS = ("a", "b", "c", "d")
DEFAULT_TEMPLATE_URL = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
DEFAULT_TILE_CACHE_DIR = Path("data/manim_tile_cache")
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"


def clamp_lat(lat: float) -> float:
    return max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)


def latlon_to_global_pixel(lat: float, lon: float, zoom: float, tile_size: int = DEFAULT_TILE_SIZE) -> tuple[
    float, float]:
    clamped = clamp_lat(lat)
    sin_lat = math.sin(math.radians(clamped))
    n = 2.0 ** zoom
    pixel_x = (lon + 180.0) / 360.0 * n * tile_size
    pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * tile_size
    return pixel_x, pixel_y


def global_pixel_to_tile(px: float, py: float, tile_size: int = DEFAULT_TILE_SIZE) -> tuple[int, int]:
    return int(px // tile_size), int(py // tile_size)


class TileFetcher:
    def __init__(
            self,
            template_url: str = DEFAULT_TEMPLATE_URL,
            cache_dir: Path | str | None = None,
            subdomains: tuple[str, ...] = DEFAULT_SUBDOMAINS,
            timeout: int = 10,
            max_retries: int = 5,
            retry_backoff_seconds: float = 0.35,
    ) -> None:
        self.template_url = template_url
        self.subdomains = subdomains
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_dir = Path(cache_dir or DEFAULT_TILE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / f"tile_z{z}_x{x}_y{y}.png"

    def fetch(self, z: int, x: int, y: int) -> Image.Image:
        image, _ = self.fetch_with_source(z, x, y)
        return image

    def fetch_with_source(self, z: int, x: int, y: int) -> tuple[Image.Image, str]:
        path = self._cache_path(z, x, y)
        if path.exists():
            try:
                return Image.open(path).convert("RGBA"), "cache"
            except Exception:
                path.unlink(missing_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            subdomain = self.subdomains[(x + y + attempt - 1) % len(self.subdomains)]
            url = self.template_url.format(s=subdomain, z=z, x=x, y=y)
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGBA")
                image.save(path)
                return image, "download"
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                break
        raise RuntimeError(f"Failed to fetch tile z={z} x={x} y={y}: {last_error}")


def stitch_tiles(
        fetcher: TileFetcher,
        z: int,
        min_tx: int,
        min_ty: int,
        max_tx: int,
        max_ty: int,
        tile_size: int = DEFAULT_TILE_SIZE,
) -> Image.Image:
    cols = max_tx - min_tx + 1
    rows = max_ty - min_ty + 1
    total = cols * rows
    image = Image.new("RGBA", (cols * tile_size, rows * tile_size), (0, 0, 0, 0))
    world_size = 2 ** z
    with tqdm(
        total=total,
        desc="Downloading tiles",
        unit="tile",
        bar_format=PROGRESS_BAR_FORMAT,
    ) as progress:
        for ix, tx in enumerate(range(min_tx, max_tx + 1)):
            for iy, ty in enumerate(range(min_ty, max_ty + 1)):
                tile, _ = fetcher.fetch_with_source(z, tx % world_size, ty % world_size)
                if tile.width != tile_size or tile.height != tile_size:
                    raise ValueError(f"Tile size mismatch at z={z} x={tx} y={ty}: {tile.size}")
                image.paste(tile, (ix * tile_size, iy * tile_size))
                progress.update(1)
    return image


class TileMap:
    def __init__(
            self,
            center_lat: float,
            center_lon: float,
            zoom: float,
            output_width_px: int | None = None,
            output_height_px: int | None = None,
            template_url: str = DEFAULT_TEMPLATE_URL,
            tile_size: int = DEFAULT_TILE_SIZE,
            cache_dir: Path | str | None = None,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = float(zoom)
        self.output_width_px = output_width_px
        self.output_height_px = output_height_px
        self.tile_size = tile_size
        self.fetcher = TileFetcher(template_url=template_url, cache_dir=cache_dir)
        self.image: Image.Image | None = None
        self.min_tx: int | None = None
        self.min_ty: int | None = None
        self.max_tx: int | None = None
        self.max_ty: int | None = None
        self.global_top_left_px: tuple[float, float] | None = None
        self.width_px: int | None = None
        self.height_px: int | None = None

    def _resolve_output_size(self, scene: Scene | None) -> tuple[int, int]:
        if self.output_width_px is not None and self.output_height_px is not None:
            return int(self.output_width_px), int(self.output_height_px)
        if scene is None:
            raise ValueError("Scene is required when output dimensions are not specified")
        camera = getattr(scene, "camera", None)
        pixel_width = getattr(camera, "pixel_width", None)
        pixel_height = getattr(camera, "pixel_height", None)
        if pixel_width is None or pixel_height is None:
            pixel_width = int(config["pixel_width"])
            pixel_height = int(config["pixel_height"])
        self.output_width_px = int(pixel_width)
        self.output_height_px = int(pixel_height)
        return int(pixel_width), int(pixel_height)

    def build(self, scene: Scene | None = None) -> Image.Image:
        output_width, output_height = self._resolve_output_size(scene)
        tile_zoom = int(math.ceil(self.zoom))
        scale_fraction = 2 ** (self.zoom - tile_zoom)

        required_px_x = output_width / max(scale_fraction, 1e-9)
        required_px_y = output_height / max(scale_fraction, 1e-9)
        tiles_x = int(math.ceil(required_px_x / self.tile_size))
        tiles_y = int(math.ceil(required_px_y / self.tile_size))
        if tiles_x % 2 == 0:
            tiles_x += 1
        if tiles_y % 2 == 0:
            tiles_y += 1
        tiles_x += 2
        tiles_y += 2

        center_px, center_py = latlon_to_global_pixel(self.center_lat, self.center_lon, tile_zoom, self.tile_size)
        center_tile_x, center_tile_y = global_pixel_to_tile(center_px, center_py, self.tile_size)

        min_tx = center_tile_x - tiles_x // 2
        min_ty = center_tile_y - tiles_y // 2
        max_tx = min_tx + tiles_x - 1
        max_ty = min_ty + tiles_y - 1

        stitched = stitch_tiles(self.fetcher, tile_zoom, min_tx, min_ty, max_tx, max_ty, self.tile_size)
        top_left_tile_pixels = (min_tx * self.tile_size, min_ty * self.tile_size)

        if scale_fraction != 1.0:
            resized = stitched.resize(
                (
                    max(1, int(round(stitched.width * scale_fraction))),
                    max(1, int(round(stitched.height * scale_fraction))),
                ),
                resample=Image.LANCZOS,
            )
        else:
            resized = stitched

        target_w = output_width
        target_h = output_height
        current_w, current_h = resized.size

        rel_center_x = (center_px - top_left_tile_pixels[0]) * scale_fraction
        rel_center_y = (center_py - top_left_tile_pixels[1]) * scale_fraction

        desired_left = rel_center_x - target_w / 2
        desired_top = rel_center_y - target_h / 2

        if (0 <= desired_left <= current_w - target_w) and (0 <= desired_top <= current_h - target_h):
            left = int(round(desired_left))
            top = int(round(desired_top))
            final_image = resized.crop((left, top, left + target_w, top + target_h))
            crop_left = desired_left
            crop_top = desired_top
        else:
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            paste_x = int(round(-desired_left))
            paste_y = int(round(-desired_top))
            canvas.paste(resized, (paste_x, paste_y), resized)
            final_image = canvas
            crop_left = desired_left
            crop_top = desired_top

        top_left_frac_x = top_left_tile_pixels[0] * scale_fraction + crop_left
        top_left_frac_y = top_left_tile_pixels[1] * scale_fraction + crop_top

        self.image = final_image
        self.min_tx = min_tx
        self.min_ty = min_ty
        self.max_tx = max_tx
        self.max_ty = max_ty
        self.global_top_left_px = (top_left_frac_x, top_left_frac_y)
        self.width_px = final_image.width
        self.height_px = final_image.height
        return final_image

    def _frame_metrics(self, scene: Scene) -> tuple[float, float, np.ndarray]:
        camera = getattr(scene, "camera", None)
        frame_width = getattr(camera, "frame_width", None)
        frame_height = getattr(camera, "frame_height", None)
        if frame_width is None or frame_height is None:
            frame_width = float(config["frame_width"])
            frame_height = float(config["frame_height"])
        frame_center = getattr(camera, "frame_center", np.array([0.0, 0.0, 0.0], dtype=float))
        return float(frame_width), float(frame_height), np.array(frame_center, dtype=float)

    def get_numpy_image(self, scene: Scene | None = None) -> np.ndarray:
        if self.image is None:
            self.build(scene)
        if self.image is None:
            raise RuntimeError("TileMap image build failed")
        array = np.asarray(self.image)
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return array

    def make_image_mobject(self, scene: Scene, set_width_to_frame: bool = True) -> ImageMobject:
        image_mobject = ImageMobject(self.get_numpy_image(scene))
        if set_width_to_frame:
            frame_width, _, frame_center = self._frame_metrics(scene)
            image_mobject.set_width(frame_width)
            image_mobject.move_to(frame_center)
        return image_mobject

    def latlon_to_scene_coords(self, lat: float, lon: float, scene: Scene) -> tuple[float, float]:
        if self.image is None or self.global_top_left_px is None or self.width_px is None or self.height_px is None:
            raise RuntimeError("TileMap.build() must run before coordinate conversion")
        px, py = latlon_to_global_pixel(lat, lon, self.zoom, self.tile_size)
        rel_x = px - self.global_top_left_px[0]
        rel_y = py - self.global_top_left_px[1]
        frame_w, frame_h, _ = self._frame_metrics(scene)
        scene_x = (rel_x - self.width_px / 2) * (frame_w / self.width_px)
        scene_y = -(rel_y - self.height_px / 2) * (frame_h / self.height_px)
        return float(scene_x), float(scene_y)
