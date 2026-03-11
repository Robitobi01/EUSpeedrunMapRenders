import json
from manim import *
from maputils import *

class EUCountry:
    def __init__(self, points: List[List[Point3D]], route_lines: List[Polygon]):
        self.points = points
        self.route_lines = route_lines
    
    def animate_creation(self, run_time=2) -> Animation:
        animations = [DrawBorderThenFill(line, stroke_width=1.3, run_time=run_time, rate_func=lambda t: route_rate(line)(t * 2) * 0.5 if t < 0.5 else double_smooth(t)) for line in self.route_lines]
        if len(animations) == 1:
            return animations[0]
        else:
            return LaggedStart(*[DrawBorderThenFill(line, stroke_width=1.3, run_time=run_time/1.2, rate_func=lambda t: route_rate(line)(t * 2) * 0.5 if t < 0.5 else double_smooth(t)) for line in self.route_lines], lag_ratio = 0.2 / (len(animations) - 1))

EU_COUNTRIES = [
    "Malta",
    "Italy",
    "France",
    "Spain",
    "Portugal",
    "Ireland",
    "Belgium",
    "Luxembourg",
    "Netherlands",
    "Germany",
    "Denmark",
    "Sweden",
    "Finland",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Poland",
    "Czech Republic",
    "Austria",
    "Slovenia",
    "Croatia",
    "Hungary",
    "Slovakia",
    "Romania",
    "Bulgaria",
    "Greece",
    "Cyprus"
]

class WholeEUScene(TileMapScene):
    def create_tile_map(self, **kwargs):
        return TileMap(57.51465184611381, 13.463387421839668, 3.45, **kwargs)

    def load_eu_country(self, name: str) -> EUCountry:
        with open('geojson/europe_clean.geojson') as f:
            data = json.load(f)
        for feat in data["features"]:
            props = feat.get("properties", {})
            if props.get("name", "") == name:
                country = feat
                break
        else:
            raise Exception(f"Couldn't find country with name {name}")
        geometry = country["geometry"]
        gtype = geometry["type"]
        coords = geometry["coordinates"]
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            raise Exception(f"Unknown geometry type {gtype}")
        
        points = []
        route_lines = []

        for polygon in polygons:
            outer_ring = polygon[0] # polygon is a list of rings: [outer_ring, hole1, hole2, ...]
            pts = []
            for lon, lat, *_ in outer_ring:
                sx, sy = self.tm.latlon_to_scene_coords(lat, lon, self)
                pts.append(np.array([sx, sy, 0.0], dtype=float))
            points.append(pts)
            route_line = Polygon(*pts)
            route_lines.append(route_line)
            route_line.set_stroke(color=DARK_BLUE, width=1.3)
            route_line.set_opacity(0.65)
        
        return EUCountry(points, route_lines)

class WholeEUIntroScene(WholeEUScene):
    def construct(self):
        intro_countries_order = [
            "Sweden",
            "Finland",
            "Estonia",
            "Latvia",
            "Lithuania",
            "Denmark",
            "Ireland",
            "Belgium",
            "Netherlands",
            "Luxembourg",
            "Germany",
            "Poland",
            "Slovakia",
            "Czech Republic",
            "France",
            "Austria",
            "Hungary",
            "Romania",
            "Bulgaria",
            "Croatia",
            "Slovenia",
            "Italy",
            "Spain",
            "Portugal",
            "Malta",
            "Greece",
            "Cyprus"
        ]
        self.play(LaggedStart(*[self.load_eu_country(name).animate_creation(run_time=0.3) for name in intro_countries_order], lag_ratio=0.2))
        self.wait(2)
