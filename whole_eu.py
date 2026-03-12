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

class VisitedScene(WholeEUScene):
    index: int
    
    def animate_creation(self, country: EUCountry) -> List[Animation]:
        title_top = any((pt[1] < -0.75 for pts in country.points for pt in pts))
        country_name = EU_COUNTRIES[self.index]
        if country_name == "Czech Republic":
            country_name = "Czechia"
        title = Text(country_name + " Visited!", font="Open Sans").set_color(DARK_BLUE)
        if title_top:
            title.to_edge(UP)
        else:
            title.to_edge(DOWN)
        return [country.animate_creation(), Write(title)]
    
    def construct(self):
        for country in EU_COUNTRIES[:self.index]:
            for line in self.load_eu_country(country).route_lines:
                self.add(line)
        
        self.play(*self.animate_creation(self.load_eu_country(EU_COUNTRIES[self.index])))
        self.wait(2)

class VisitedMaltaScene(VisitedScene):
    def __init__(self, *args, **kwargs):
        self.index = 0
        super().__init__(*args, **kwargs)
    
    def animate_creation(self, country):
        # Malta is so small we need to point to it
        malta_object = self.load_eu_country(EU_COUNTRIES[self.index]).route_lines[0]
        arrow = Arrow(malta_object.get_right() + 1 * RIGHT, malta_object.get_right() + 0.1 * LEFT).set_color(DARK_BLUE)
        label = Text("Malta", font="Open Sans").scale(0.4).next_to(arrow, buff=0.1).set_color(DARK_BLUE)
        return super().animate_creation(country) + [Create(arrow), Write(label)]

for index in range(1, len(EU_COUNTRIES)):
    scene_name = f"Visited{EU_COUNTRIES[index].replace(" ", "")}Scene"
    scene_class = type(scene_name, (VisitedScene,), {"index": index})
    globals()[scene_name] = scene_class

class IntroTitle(Scene):
    def construct(self):
        title = Text("EU Speedrun", font="Open Sans").set_color(WHITE)

        base_star = Star().set_color(0xffcc00).set_opacity(1.0).scale(0.3)
        base_star.reverse_points()
        # Use next_to once so we get the usual spacing/buffer
        base_star.next_to(title, RIGHT)
        # Euclidean distance between centres (includes the buffer)
        radius = np.linalg.norm(base_star.get_center() - title.get_center())
        # Put the star straight UP at that same centre-to-centre distance
        base_star.move_to(title.get_center() + radius * UP)

        self.play(LaggedStart(Write(title, run_time=1.5), DrawBorderThenFill(base_star, run_time=1), lag_ratio=0.75))

        # --- create copies stacked on the base star (total 12) ---
        stars = [base_star] + [base_star.copy() for _ in range(11)]
        for s in stars[1:]:
            s.move_to(base_star.get_center())
            self.add(s)

        center = title.get_center()
        n = len(stars)
        step = TAU / n

        # angles (start and target) for each star
        start_angles = [
            np.arctan2(s.get_center()[1] - center[1], s.get_center()[0] - center[0])
            for s in stars
        ]
        # choose a consistent set of target angles so they are evenly spaced
        angle0 = start_angles[0]                     # keep the base_star as the "first" star
        target_angles = [angle0 + i * step for i in range(n)]

        # build arcs and MoveAlongPath animations
        animations = []
        for s, a_start, a_target in zip(stars, start_angles, target_angles):
            # raw delta (positive => CCW). We want clockwise movement => negative angle.
            delta = a_target - a_start
            if delta > 0:
                delta -= TAU   # force the movement to go clockwise (negative angle)

            # create an arc centered on `center` with the computed (negative) angle
            arc = Arc(start_angle=a_start, angle=delta, radius=radius).shift(center)

            # MoveAlongPath translates the star along the arc but DOES NOT rotate the mobject
            animations.append(MoveAlongPath(s, arc))

        # play them together (or use LaggedStart for staggered fan-out)
        bg = FullScreenRectangle(fill_color=0x003399, fill_opacity=0, stroke_width=0)
        bg.set_z_index(-10)  # keep it behind everything
        self.add(bg)

        self.play(*(animations + [bg.animate.set_fill(opacity=1)]), run_time=2)

        self.wait(0.5)

        self.play(FadeOut(title))
        self.wait(1)
