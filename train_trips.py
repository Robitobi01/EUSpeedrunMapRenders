from __future__ import annotations

import json
from pathlib import Path

from maputils import *

MANIFEST_PATH = Path("data/train_trip_manifest.json")


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    trips = payload.get("trips")
    if not isinstance(trips, list):
        raise ValueError(f'Invalid manifest at "{MANIFEST_PATH.as_posix()}": missing "trips" list')
    return trips


class TrainTripScene(TileMapScene):
    trip: dict

    def create_tile_map(self, **kwargs):
        map_info = self.trip["map"]
        return TileMap(
            map_info["center_lat"],
            map_info["center_lon"],
            map_info["zoom"],
            **kwargs,
        )

    def construct(self):
        route = self.load_geojson(self.trip["geojson_path"])
        start_marker = MapMarker(
            self.trip["from_label"],
            route.start,
            label_direction=route.start_label_direction(),
        )
        end_marker = MapMarker(
            self.trip["to_label"],
            route.end,
            label_direction=route.end_label_direction(),
        )
        frame_width = getattr(self.camera, "frame_width", float(config["frame_width"]))
        frame_height = getattr(self.camera, "frame_height", float(config["frame_height"]))
        start_marker.choose_label_direction_with_route(frame_width, frame_height, route.points, 0)
        end_marker.choose_label_direction_with_route(frame_width, frame_height, route.points, -1)
        start_marker.clamp_label_within_frame(frame_width, frame_height)
        end_marker.clamp_label_within_frame(frame_width, frame_height)
        start_marker.add_to_scene(self, foreground=True)
        end_marker.add_to_scene(self, foreground=True)
        atmosphere = self.create_map_atmosphere(start_marker.dot.get_center(), end_marker.dot.get_center())
        self.play(FadeIn(atmosphere, run_time=0.8))
        self.play(*start_marker.animate_creation())
        start_marker.show_final_state()
        Geojson.force_foreground(self, start_marker.foreground_mobjects())
        self.play(*end_marker.animate_creation())
        end_marker.show_final_state()
        Geojson.force_foreground(self, start_marker.foreground_mobjects() + end_marker.foreground_mobjects())
        marker_front = start_marker.foreground_mobjects() + end_marker.foreground_mobjects()
        route.create_and_animate(
            self,
            dash_animate_time=6,
            keep_on_top=marker_front,
        )
        self.play(end_marker.animate_arrival())
        self.wait(0.4)


TRIP_MANIFEST = load_manifest()
TRIP_SCENE_NAMES: list[str] = []

for trip in TRIP_MANIFEST:
    scene_name = trip["scene_name"]
    if not isinstance(scene_name, str) or not scene_name:
        continue
    scene_class = type(scene_name, (TrainTripScene,), {"trip": trip})
    globals()[scene_name] = scene_class
    TRIP_SCENE_NAMES.append(scene_name)
