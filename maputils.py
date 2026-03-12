from __future__ import annotations
import json
from bisect import bisect_right
from manim import *
from manim.typing import *
from manimpango import register_font
from tilemap_manim import *
from typing import Callable, List, Sequence, Tuple

try:
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
except Exception:
    OpenGLVMobject = None

register_font("OpenSans-Bold.ttf")

def load_tilemap_settings():
    try:
        with open("tilemap_settings.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

tilemap_settings = load_tilemap_settings()

def normalize_xy(vector: Point3DLike, fallback: Point3DLike = RIGHT) -> np.ndarray:
    value = np.array(vector, dtype=float)
    if value.shape[0] == 2:
        value = np.array([value[0], value[1], 0.0], dtype=float)
    value[2] = 0.0
    norm = np.linalg.norm(value[:2])
    if norm < 1e-6:
        return normalize_xy(fallback, RIGHT) if np.linalg.norm(np.array(fallback, dtype=float)[:2]) > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=float)
    return value / norm

def simplify_scene_polyline(points: Sequence[Point3D], min_distance: float = 0.035, collinear_dot: float = 0.9994) -> list[np.ndarray]:
    if len(points) <= 2:
        return [np.array(point, dtype=float) for point in points]
    simplified = [np.array(points[0], dtype=float)]
    for index in range(1, len(points) - 1):
        prev_kept = simplified[-1]
        current = np.array(points[index], dtype=float)
        nxt = np.array(points[index + 1], dtype=float)
        if np.linalg.norm((current - prev_kept)[:2]) < min_distance:
            continue
        incoming = current - prev_kept
        outgoing = nxt - current
        incoming_norm = np.linalg.norm(incoming[:2])
        outgoing_norm = np.linalg.norm(outgoing[:2])
        if incoming_norm < 1e-6 or outgoing_norm < 1e-6:
            continue
        incoming_unit = incoming[:2] / incoming_norm
        outgoing_unit = outgoing[:2] / outgoing_norm
        if incoming_norm < min_distance * 2.2 and outgoing_norm < min_distance * 2.2 and np.dot(incoming_unit, outgoing_unit) > collinear_dot:
            continue
        simplified.append(current)
    simplified.append(np.array(points[-1], dtype=float))
    return simplified

def soften_scene_polyline(
    points: Sequence[Point3D],
    passes: int = 2,
    keep_turn_dot: float = 0.9,
    max_shift: float = 0.022,
) -> list[np.ndarray]:
    shaped = [np.array(point, dtype=float) for point in points]
    if len(shaped) <= 2:
        return shaped
    for _ in range(max(0, passes)):
        updated = [shaped[0].copy()]
        for index in range(1, len(shaped) - 1):
            prev_point = shaped[index - 1]
            current_point = shaped[index]
            next_point = shaped[index + 1]
            incoming = normalize_xy(current_point - prev_point)
            outgoing = normalize_xy(next_point - current_point)
            if float(np.dot(incoming[:2], outgoing[:2])) < keep_turn_dot:
                updated.append(current_point.copy())
                continue
            target = 0.22 * prev_point + 0.56 * current_point + 0.22 * next_point
            shift = np.array(target - current_point, dtype=float)
            shift[2] = 0.0
            shift_norm = np.linalg.norm(shift[:2])
            if shift_norm > max_shift:
                shift *= max_shift / shift_norm
            updated.append(current_point + shift)
        updated.append(shaped[-1].copy())
        shaped = updated
    return shaped

def sample_scene_polyline(points: Sequence[Point3D], step: float = 0.18) -> list[np.ndarray]:
    if not points:
        return []
    if len(points) == 1:
        return [np.array(points[0], dtype=float)]
    samples = [np.array(points[0], dtype=float)]
    for start, end in zip(points, points[1:]):
        start_point = np.array(start, dtype=float)
        end_point = np.array(end, dtype=float)
        length = float(np.linalg.norm((end_point - start_point)[:2]))
        parts = max(1, int(np.ceil(length / max(1e-6, step))))
        for part in range(1, parts + 1):
            samples.append(interpolate(start_point, end_point, part / parts))
    return samples

def route_rate(route: VMobject, rate_func: Callable[[float], float] = smooth) -> Callable[[float], float]:
    curve_lengths = [c[1] for c in route.get_curve_functions_with_lengths()]
    t_per_interval = 1 / len(curve_lengths)
    relative_time = [(l / sum(curve_lengths)) for l in curve_lengths]
    v = 0
    cumulative_t = [0] + [v := v+t for t in relative_time]

    def curve_rf(t):
        t = rate_func(t)
        if t <= 0: return 0
        if t >= cumulative_t[-1]: return 1

        # find index of current curve
        i = bisect_right(cumulative_t, t) - 1

        a, b = cumulative_t[i], cumulative_t[i+1]
        t1 = i * t_per_interval
        t2 = (i + 1) * t_per_interval

        return interpolate(t1, t2, inverse_interpolate(a, b, t))

    return curve_rf

class DashedLine(VMobject):
    def __init__(self, base_obj: VMobject, dash_len: float = 0.2, dash_offset: float = 0.0, colors: List[ParsableManimColor] = [BLUE, WHITE], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_obj = base_obj
        self._dash_len = dash_len
        self._dash_offset = dash_offset
        self.colors = colors
        self.refresh_dashes()
    
    def set_dash_len(self, dash_len: float) -> DashedLine:
        self._dash_len = dash_len
        self.refresh_dashes()
        return self

    def set_dash_offset(self, dash_offset: float) -> DashedLine:
        self._dash_offset = dash_offset
        self.refresh_dashes()
        return self

    def refresh_dashes(self):
        self.submobjects.clear()

        # calculate the entire length by adding up short line-pieces
        norms = np.array(0)
        for k in range(self.base_obj.get_num_curves()):
            norms = np.append(norms, self.base_obj.get_nth_curve_length_pieces(k))
        # add up length-pieces in array form
        length_vals = np.cumsum(norms)
        ref_points = np.linspace(0, 1, length_vals.size)
        curve_length = length_vals[-1]
        
        if curve_length > 0 and self._dash_len > 0:
            dash_len = min(self._dash_len, curve_length)
            dash_proportion = dash_len / curve_length
            dash_offset = self._dash_offset % 1
            color_offset = -int(self._dash_offset) % len(self.colors)
            offset = dash_offset * dash_proportion

            if offset > 0:
                self.add(self.base_obj.get_subcurve(
                    0,
                    np.interp(offset * curve_length, length_vals, ref_points)
                ).match_style(self).set_color(self.colors[(color_offset - 1) % len(self.colors)]))
            i = 0
            while offset < curve_length:
                self.add(self.base_obj.get_subcurve(
                    np.interp(offset * curve_length, length_vals, ref_points), 
                    min(1, np.interp((offset + dash_proportion) * curve_length, length_vals, ref_points))
                ).match_style(self).set_color(self.colors[(i + color_offset) % len(self.colors)]))
                offset += dash_proportion
                i += 1

class DashOffsetAnimation(Animation):
    def __init__(self, mobject: DashedLine, dash_move_speed: float = 2.0, fade_in_time: float = 1.0, rate_func: Callable[[float], float] = linear, *args, **kwargs):
        super().__init__(mobject, *args, rate_func=rate_func, **kwargs)
        self.initial_offset = mobject._dash_offset
        self.dash_move_speed = dash_move_speed
        self.fade_in_time = fade_in_time

    def interpolate_mobject(self, alpha: float):
        time = alpha * self.run_time
        opacity = 1 if time >= self.fade_in_time or self.fade_in_time <= 0 else time / self.fade_in_time
        offset = self.initial_offset + time * self.dash_move_speed
        self.mobject.become(self.mobject.copy().set_dash_offset(offset))
        self.mobject.set_opacity(opacity)

class PolylineReveal(Animation):
    def __init__(self, mobject: VGroup, *args, **kwargs):
        super().__init__(mobject, *args, **kwargs)
        self.segment_refs = [segment for segment in mobject]
        self.segment_data = [
            (
                np.array(segment.get_start(), dtype=float),
                np.array(segment.get_end(), dtype=float),
                float(np.linalg.norm((np.array(segment.get_end(), dtype=float) - np.array(segment.get_start(), dtype=float))[:2])),
                float(getattr(segment, "target_stroke_opacity", segment.get_stroke_opacity())),
            )
            for segment in self.segment_refs
        ]
        self.total_length = max(1e-6, sum(length for _, _, length, _ in self.segment_data))

    def begin(self) -> None:
        super().begin()
        self.interpolate_mobject(0.0)

    def interpolate_mobject(self, alpha: float) -> None:
        travelled = alpha * self.total_length
        remaining = travelled
        for segment, (start, end, length, opacity) in zip(self.segment_refs, self.segment_data):
            direction = normalize_xy(end - start)
            epsilon = min(max(length * 0.001, 1e-4), max(length, 1e-4))
            if remaining <= 0:
                segment.put_start_and_end_on(start, start + direction * epsilon)
                segment.set_stroke(opacity=0)
                continue
            if remaining >= length:
                segment.put_start_and_end_on(start, end)
                segment.set_stroke(opacity=opacity)
                remaining -= length
                continue
            partial_alpha = max(remaining / max(length, 1e-6), epsilon / max(length, 1e-6))
            partial_end = interpolate(start, end, partial_alpha)
            segment.put_start_and_end_on(start, partial_end)
            segment.set_stroke(opacity=opacity)
            remaining = 0.0


class Geojson:
    def __init__(self, points: List[Point3D], route_line: VMobject, dashed_route_line: DashedLine | None):
        self.points = points
        self.route_line = route_line
        self.dashed_route_line = dashed_route_line
    
    def animate_line_creation(self) -> Animation:
        return Create(self.route_line, rate_func=route_rate(self.route_line, linear))
    
    def animate_dashes(self, run_time: float = 10) -> Animation:
        if self.dashed_route_line is None:
            raise ValueError("Dashed route line is not available")
        return DashOffsetAnimation(self.dashed_route_line, run_time=run_time)

    def _endpoint_label_vector(self, anchor: Point3D, neighbor: Point3D | None) -> np.ndarray:
        if neighbor is None:
            base = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        else:
            base = np.array(anchor, dtype=float) - np.array(neighbor, dtype=float)
        base[2] = 0.0
        if np.linalg.norm(base[:2]) < 1e-6:
            base = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        if abs(base[0]) < 0.22:
            base[0] += 0.35 if anchor[0] <= 0 else -0.35
        if abs(base[1]) < 0.18:
            base[1] += 0.28 if anchor[1] <= 0 else -0.28
        return base

    def start_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.start, None)
        window = self.points[1:min(len(self.points), 8)]
        weights = np.linspace(1.0, 0.45, len(window))
        weighted_center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.start, weighted_center)

    def end_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.end, None)
        window = self.points[max(0, len(self.points) - 8):-1]
        weights = np.linspace(0.45, 1.0, len(window))
        weighted_center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.end, weighted_center)

    def front_mobjects(self, markers: Sequence["MapMarker"] | None = None) -> list[Mobject]:
        pieces: list[Mobject] = []
        if markers is not None:
            for marker in markers:
                pieces.extend(marker.foreground_mobjects())
        return pieces

    def segment_lengths(self) -> list[float]:
        return [float(np.linalg.norm((end - start)[:2])) for start, end in zip(self.points, self.points[1:])]

    def build_segment_group(self, color: ParsableManimColor, width: float) -> VGroup:
        segments = []
        route_pairs = [(np.array(start, dtype=float), np.array(end, dtype=float)) for start, end in zip(self.points, self.points[1:])]
        if route_pairs:
            start_a, start_b = route_pairs[0]
            start_length = float(np.linalg.norm((start_b - start_a)[:2]))
            if start_length > 1e-6:
                inset = min(0.15, start_length * 0.35)
                route_pairs[0] = (interpolate(start_a, start_b, inset / start_length), start_b)
            end_a, end_b = route_pairs[-1]
            end_length = float(np.linalg.norm((end_b - end_a)[:2]))
            if end_length > 1e-6:
                inset = min(0.15, end_length * 0.35)
                route_pairs[-1] = (end_a, interpolate(end_b, end_a, inset / end_length))
        for start, end in route_pairs:
            segment = Line(start, end).set_fill(opacity=0).set_stroke(color=color, width=width, opacity=1.0)
            segment.target_stroke_opacity = 1.0
            segments.append(segment)
        return VGroup(*segments)

    def segment_draw_animation(self, group: VGroup, run_time: float) -> Animation:
        return PolylineReveal(group, run_time=run_time, rate_func=smooth)

    def endpoint_pulse(self, center: Point3DLike, color: ParsableManimColor, base_radius: float = 0.16) -> Animation:
        ring_a = Circle(radius=base_radius).move_to(center).set_fill(opacity=0).set_stroke(color, width=2.6, opacity=0.9)
        ring_b = Circle(radius=base_radius * 0.92).move_to(center).set_fill(opacity=0).set_stroke(WHITE, width=1.4, opacity=0.72)
        return AnimationGroup(
            FadeIn(ring_a, scale=0.45, run_time=0.12),
            FadeOut(ring_a, scale=2.7, run_time=0.72),
            FadeIn(ring_b, scale=0.6, run_time=0.14),
            FadeOut(ring_b, scale=2.1, run_time=0.58),
            lag_ratio=0.0,
        )

    @staticmethod
    def force_foreground(scene: Scene, mobjects: Sequence[Mobject] | None) -> None:
        if not mobjects:
            return
        scene.add_foreground_mobjects(*mobjects)

    def create_and_animate(
        self,
        scene: Scene,
        dash_animate_time: float = 10,
        keep_on_top: Sequence[Mobject] | None = None,
    ) -> Geojson:
        renderer = str(config["renderer"]).lower()
        if "opengl" in renderer:
            route_casing = self.build_segment_group("#20384E", 8.6)
            route_fill = self.build_segment_group("#6D86A3", 6.0)
            route_main = self.build_segment_group("#F8FBFF", 4.05)
            route_accent = self.build_segment_group("#FFD76A", 1.2)
            route_group = VGroup(route_casing, route_fill, route_main, route_accent)
            for layer in route_group:
                for segment in layer:
                    segment.set_stroke(opacity=0)
            scene.add(route_group)
            scene.play(
                self.endpoint_pulse(self.start, "#63B6FF"),
                run_time=0.72,
            )
            self.force_foreground(scene, keep_on_top)
            scene.play(
                AnimationGroup(
                    self.segment_draw_animation(route_casing, 1.46),
                    self.segment_draw_animation(route_fill, 1.52),
                    self.segment_draw_animation(route_main, 1.58),
                    self.segment_draw_animation(route_accent, 1.66),
                    lag_ratio=0.06,
                ),
            )
            self.force_foreground(scene, keep_on_top)
            scene.play(
                AnimationGroup(
                    self.endpoint_pulse(self.end, "#FFD76A"),
                    route_fill.animate.set_stroke(width=6.35),
                    route_main.animate.set_stroke(width=4.3),
                    route_accent.animate.set_stroke(width=1.4),
                    lag_ratio=0.0,
                ),
                run_time=0.78,
                rate_func=there_and_back_with_pause,
            )
            self.force_foreground(scene, keep_on_top)
            return self
        scene.play(self.animate_line_creation())
        if self.dashed_route_line is not None:
            scene.play(self.animate_dashes(run_time=dash_animate_time))
        if keep_on_top:
            scene.bring_to_front(*keep_on_top)
        return self

    @property
    def start(self) -> Point3D:
        return self.points[0]
    
    @property
    def end(self) -> Point3D:
        return self.points[-1]

class MapMarker(VGroup):
    def __init__(self, label: str, location: Point3DLike, label_direction: Point3DLike | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.halo = VGroup(
            Circle(radius=0.34).set_fill("#68B7FF", opacity=0.08).set_stroke(width=0),
            Circle(radius=0.24).set_fill("#FFD45C", opacity=0.08).set_stroke(width=0),
        ).move_to(location)
        self.dot = VGroup(
            Circle(radius=0.18).set_fill(WHITE, opacity=0.06).set_stroke(width=0),
            Circle(radius=0.11).set_fill(BLACK, opacity=0.62).set_stroke(WHITE, width=2.2, opacity=0.92),
            Circle(radius=0.052).set_fill("#FFD45C", opacity=1.0).set_stroke(WHITE, width=0.5, opacity=0.75),
        ).move_to(location)
        fallback = np.array([1.0 if location[0] <= 0 else -1.0, 0.35 if location[1] <= 0 else -0.35, 0.0], dtype=float)
        self.preferred_label_vector = normalize_xy(label_direction if label_direction is not None else fallback, fallback)
        self.label_direction = self.snap_label_direction(self.preferred_label_vector, location)
        self.text = Text(label, font="Open Sans", font_size=24).set_color(WHITE)
        self.label_shadow = RoundedRectangle(
            corner_radius=0.12,
            width=self.text.width + 0.34,
            height=self.text.height + 0.18,
        ).set_fill(BLACK, opacity=0.22).set_stroke(width=0)
        self.label_bg = RoundedRectangle(
            corner_radius=0.12,
            width=self.text.width + 0.34,
            height=self.text.height + 0.18,
        ).set_fill("#151B21", opacity=0.78).set_stroke(WHITE, width=1.0, opacity=0.18)
        self.label_shadow_fill_opacity = 0.22
        self.label_bg_fill_opacity = 0.78
        self.label_bg_stroke_opacity = 0.18
        self.text_opacity = 1.0
        self.label_shadow.shift(0.045 * RIGHT + 0.04 * DOWN)
        self.label = VGroup(self.label_shadow, self.label_bg, self.text)
        self.label.next_to(self.dot, self.label_direction, buff=0.24)
        self.label_target = self.label.copy()
        self.halo.save_state()
        self.dot.save_state()
        self.label_shadow.save_state()
        self.label_bg.save_state()
        self.text.save_state()
        self.label.save_state()
        self.prepare_intro_state()
        self.add(self.halo, self.dot, self.label)
        if hasattr(self.halo, "set_z_index"):
            self.halo.set_z_index(8)
            for piece in self.halo:
                piece.set_z_index(8)
        if hasattr(self.dot, "set_z_index"):
            self.dot.set_z_index(10)
            for piece in self.dot:
                piece.set_z_index(10)
        if hasattr(self.label, "set_z_index"):
            self.label.set_z_index(12)
            self.label_shadow.set_z_index(12)
            self.label_bg.set_z_index(12)
            self.text.set_z_index(13)
        if hasattr(self, "set_z_index"):
            self.set_z_index(12)

    @staticmethod
    def snap_label_direction(direction: Point3DLike | None, location: Point3DLike) -> np.ndarray:
        candidates = [LEFT, RIGHT, UP, DOWN, UL, UR, DL, DR]
        if direction is None:
            fallback = np.array([1.0 if location[0] <= 0 else -1.0, 0.35 if location[1] <= 0 else -0.35, 0.0], dtype=float)
            direction = fallback
        vector = np.array(direction, dtype=float)
        if vector.shape[0] == 2:
            vector = np.array([vector[0], vector[1], 0.0], dtype=float)
        vector[2] = 0.0
        norm = np.linalg.norm(vector[:2])
        if norm < 1e-6:
            vector = np.array([1.0 if location[0] <= 0 else -1.0, 0.35 if location[1] <= 0 else -0.35, 0.0], dtype=float)
            norm = np.linalg.norm(vector[:2])
        vector = vector / norm
        best = max(
            candidates,
            key=lambda candidate: np.dot(
                vector[:2],
                np.array(candidate[:2], dtype=float) / np.linalg.norm(np.array(candidate[:2], dtype=float)),
            ),
        )
        return np.array(best, dtype=float)

    def choose_label_direction(self, frame_width: float, frame_height: float, margin: float = 0.28) -> None:
        self.choose_label_direction_with_route(frame_width, frame_height, None, 0, margin=margin)

    @staticmethod
    def _box_distance(label_box: Mobject, point: Point3DLike) -> float:
        left = label_box.get_left()[0]
        right = label_box.get_right()[0]
        bottom = label_box.get_bottom()[1]
        top = label_box.get_top()[1]
        x = float(point[0])
        y = float(point[1])
        dx = 0.0 if left <= x <= right else min(abs(x - left), abs(x - right))
        dy = 0.0 if bottom <= y <= top else min(abs(y - bottom), abs(y - top))
        return float(np.hypot(dx, dy))

    def route_overlap_penalty(self, label_box: Mobject, route_samples: Sequence[Point3DLike]) -> float:
        if not route_samples:
            return 0.0
        penalty_sum = 0.0
        max_penalty = 0.0
        for index, sample in enumerate(route_samples[:80]):
            distance = self._box_distance(label_box, sample)
            weight = 1.65 / (1.0 + 0.15 * index)
            penalty = weight * np.exp(-((distance / 0.24) ** 2))
            penalty_sum += penalty
            max_penalty = max(max_penalty, penalty)
        return penalty_sum * 0.42 + max_penalty * 2.1

    def choose_label_direction_with_route(
        self,
        frame_width: float,
        frame_height: float,
        route_points: Sequence[Point3D] | None,
        endpoint_index: int,
        margin: float = 0.28,
    ) -> None:
        angles = np.linspace(0.0, TAU, 16, endpoint=False)
        candidates = [np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float) for angle in angles]
        min_x = -frame_width / 2 + margin
        max_x = frame_width / 2 - margin
        min_y = -frame_height / 2 + margin
        max_y = frame_height / 2 - margin
        desired = normalize_xy(self.preferred_label_vector)
        route_samples = sample_scene_polyline(route_points, step=0.16) if route_points else []
        if route_samples:
            route_samples = route_samples[1:] if endpoint_index == 0 else list(reversed(route_samples[:-1]))
        best_direction = self.label_direction
        best_score = -1e9
        for candidate in candidates:
            trial = self.label_target.copy()
            trial.next_to(self.dot, candidate, buff=0.28)
            overflow = 0.0
            overflow += max(0.0, min_x - trial.get_left()[0])
            overflow += max(0.0, trial.get_right()[0] - max_x)
            overflow += max(0.0, min_y - trial.get_bottom()[1])
            overflow += max(0.0, trial.get_top()[1] - max_y)
            alignment = float(np.dot(desired[:2], normalize_xy(candidate)[:2]))
            route_penalty = self.route_overlap_penalty(trial, route_samples)
            score = alignment * 4.4 - route_penalty * 1.6 - overflow * 18.0
            if score > best_score:
                best_score = score
                best_direction = np.array(candidate, dtype=float)
        self.label_direction = best_direction
        self.label_target.next_to(self.dot, self.label_direction, buff=0.28)
        self.label.restore()
        self.label.move_to(self.label_target)
        self.label_shadow.save_state()
        self.label_bg.save_state()
        self.text.save_state()
        self.label.save_state()
        self.prepare_intro_state()

    def clamp_label_within_frame(self, frame_width: float, frame_height: float, margin: float = 0.28) -> None:
        min_x = -frame_width / 2 + margin
        max_x = frame_width / 2 - margin
        min_y = -frame_height / 2 + margin
        max_y = frame_height / 2 - margin
        shift = np.array([0.0, 0.0, 0.0], dtype=float)
        if self.label_target.get_left()[0] < min_x:
            shift[0] += min_x - self.label_target.get_left()[0]
        if self.label_target.get_right()[0] > max_x:
            shift[0] -= self.label_target.get_right()[0] - max_x
        if self.label_target.get_bottom()[1] < min_y:
            shift[1] += min_y - self.label_target.get_bottom()[1]
        if self.label_target.get_top()[1] > max_y:
            shift[1] -= self.label_target.get_top()[1] - max_y
        if np.linalg.norm(shift[:2]) > 0:
            self.label_target.shift(shift)
            self.label.restore()
            self.label.move_to(self.label_target)
            self.label_shadow.save_state()
            self.label_bg.save_state()
            self.text.save_state()
            self.label.save_state()
            self.prepare_intro_state()

    def prepare_intro_state(self) -> None:
        self.halo.restore()
        self.dot.restore()
        self.label.restore()
        self.halo.scale(0.72).set_opacity(0)
        self.dot.scale(0.78).set_opacity(0)
        self.label.move_to(self.label_target).shift(-0.12 * self.label_direction)
        self.label_shadow.set_fill(opacity=0).set_stroke(opacity=0)
        self.label_bg.set_fill(opacity=0).set_stroke(opacity=0)
        self.text.set_fill(opacity=0).set_stroke(opacity=0)

    def foreground_mobjects(self) -> list[Mobject]:
        return [*self.halo, *self.dot, self.label_shadow, self.label_bg, self.text]

    def add_to_scene(self, scene: Scene, foreground: bool = True) -> None:
        if foreground:
            scene.add_foreground_mobjects(*self.foreground_mobjects())
        else:
            scene.add(*self.foreground_mobjects())

    def show_final_state(self) -> None:
        self.halo.restore()
        self.dot.restore()
        self.label_shadow.restore()
        self.label_bg.restore()
        self.text.restore()

    def animate_creation(self) -> Sequence[Animation]:
        return [
            Succession(
                AnimationGroup(
                    Restore(self.halo, run_time=0.46, rate_func=rate_functions.ease_out_sine),
                    Restore(self.dot, run_time=0.46, rate_func=rate_functions.ease_out_back),
                    Restore(self.label_shadow, run_time=0.34, rate_func=rate_functions.ease_out_cubic),
                    Restore(self.label_bg, run_time=0.34, rate_func=rate_functions.ease_out_cubic),
                    Restore(self.text, run_time=0.34, rate_func=rate_functions.ease_out_cubic),
                    lag_ratio=0.08,
                ),
                AnimationGroup(
                    self.halo.animate.scale(1.12).set_opacity(0.24),
                    self.dot.animate.scale(1.05),
                    run_time=0.28,
                    rate_func=there_and_back,
                ),
                lag_ratio=1.0,
            )
        ]

    def animate_arrival(self) -> Animation:
        pulse_hot = Circle(radius=0.18).move_to(self.dot.get_center()).set_fill(opacity=0).set_stroke("#FFD45C", width=2.6, opacity=0.95)
        pulse_cool = Circle(radius=0.18).move_to(self.dot.get_center()).set_fill(opacity=0).set_stroke(WHITE, width=1.8, opacity=0.85)
        return AnimationGroup(
            Indicate(self.dot, color="#FFD45C", scale_factor=1.14, run_time=0.7),
            FadeIn(pulse_hot, scale=0.5, run_time=0.12),
            FadeOut(pulse_hot, scale=2.8, run_time=0.62),
            FadeIn(pulse_cool, scale=0.75, run_time=0.12),
            FadeOut(pulse_cool, scale=2.2, run_time=0.55),
            lag_ratio=0.0,
        )

class TileMapScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tm = self.create_tile_map(**tilemap_settings)
        self.tm_image: ImageMobject = None
    
    def create_tile_map(self, **kwargs) -> TileMap:
        raise NotImplementedError("Subclasses must implement create_tile_map")
    
    def setup(self):
        super().setup()
        self.camera.background_color = "#D9E7EB"
        self.tm_image = self.tm.make_image_mobject(self, set_width_to_frame=True)
        self.add(self.tm_image)
    
    def load_geojson(self, filename: str) -> Geojson:
        with open(filename) as f:
            data = json.load(f)
        features = data["features"]
        if len(features) != 1:
            raise Exception(f"Geojson {filename} should have exactly one feature, found {len(features)}")
        geometry = features[0]["geometry"]
        if geometry["type"] != "LineString":
            raise Exception(f"Geojson {filename} should have LineString geometry, found {geometry["type"]}")
        points = [self.tm.latlon_to_scene_coords(lat, lon, self) for [lon, lat, *_] in geometry["coordinates"]]
        points = [np.array([sx, sy, 0.0], dtype=float) for sx, sy in points]
        points = simplify_scene_polyline(points)
        points = soften_scene_polyline(points)

        renderer = str(config["renderer"]).lower()
        if "opengl" in renderer and OpenGLVMobject is not None:
            route_line = OpenGLVMobject()
        else:
            route_line = VMobject()
        route_line.set_points_as_corners(points)
        route_line.set_fill(opacity=0).set_stroke(color=DARK_BLUE, width=6.0)

        if "opengl" in renderer:
            dashed_route_line = None
        else:
            dashed_route_line = DashedLine(route_line)

        return Geojson(points, route_line, dashed_route_line)
    
    def create_marker(self, label: str, lat: float, lon: float) -> MapMarker:
        sx, sy = self.tm.latlon_to_scene_coords(lat, lon, self)
        return MapMarker(label, np.array([sx, sy, 0.0]))

    def create_soft_glow(
        self,
        center: Point3DLike,
        color: ParsableManimColor,
        radii: Sequence[float],
        opacities: Sequence[float],
    ) -> VGroup:
        layers = VGroup()
        for radius, opacity in zip(radii, opacities):
            layers.add(Circle(radius=radius).set_fill(color, opacity=opacity).set_stroke(width=0).move_to(center))
        return layers

    def create_map_atmosphere(self, start: Point3DLike, end: Point3DLike) -> VGroup:
        frame_width = getattr(self.camera, "frame_width", None)
        frame_height = getattr(self.camera, "frame_height", None)
        if frame_width is None or frame_height is None:
            frame_width = float(config["frame_width"])
            frame_height = float(config["frame_height"])
        veil = FullScreenRectangle(fill_color=BLACK, fill_opacity=0.06, stroke_width=0)
        border = RoundedRectangle(
            corner_radius=0.18,
            width=frame_width - 0.32,
            height=frame_height - 0.32,
        ).set_fill(opacity=0).set_stroke(WHITE, width=0.8, opacity=0.06)
        atmosphere = VGroup(veil, border)
        if hasattr(atmosphere, "set_z_index"):
            atmosphere.set_z_index(0)
        return atmosphere
