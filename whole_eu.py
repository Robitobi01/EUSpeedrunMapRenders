import json
from manim import *
from maputils import *

class EUCountry:
    def __init__(self, points: List[List[Point3D]], route_lines: List[Polygon]):
        self.points = points
        self.route_lines = route_lines
    
    def animate_creation(self) -> List[Animation]:
        return [DrawBorderThenFill(line, stroke_width=1.3, rate_func=lambda t: route_rate(line)(t * 2) * 0.5 if t < 0.5 else double_smooth(t)) for line in self.route_lines]

def load_eu_country(name: str, scene: TileMapScene) -> EUCountry:
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
            sx, sy = scene.tm.latlon_to_scene_coords(lat, lon, scene)
            pts.append(np.array([sx, sy, 0.0], dtype=float))
        points.append(pts)
        route_line = Polygon(*pts)
        route_lines.append(route_line)
        route_line.set_stroke(color=DARK_BLUE, width=1.3)
        route_line.set_opacity(0.65)
    
    return EUCountry(points, route_lines)

EU_COUNTRIES = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
]

class WholeEUIntroScene(TileMapScene):
    def create_tile_map(self, **kwargs):
        return TileMap(57.51465184611381, 13.463387421839668, 3.45, **kwargs)

    def construct(self):
        self.play(*[animation for name in EU_COUNTRIES for animation in load_eu_country(name, self).animate_creation()], lag_ratio=0.2)
