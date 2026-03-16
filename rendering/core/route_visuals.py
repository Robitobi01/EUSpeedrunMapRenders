from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from manim import (
    Animation,
    AnimationGroup,
    CapStyleType,
    Circle,
    Create,
    FadeIn,
    LineJointType,
    Scene,
    Text,
    ValueTracker,
    VMobject,
    VGroup,
    interpolate,
    linear,
)
from manim.typing import Point3DLike

from .geometry import as_point3, route_distance_lookup, route_proportion_for_distance, route_rate


@dataclass(frozen=True, slots=True)
class RouteLayerStyle:
    color: str
    width: float
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class SpeedSegment:
    start: float
    end: float
    kmh: float


class SpeedProfile:
    def __init__(self, segments: Sequence[SpeedSegment]) -> None:
        cleaned = [segment for segment in segments if segment.end > segment.start]
        if not cleaned:
            cleaned = [SpeedSegment(0.0, 1.0, 80.0)]
        self.segments = sorted(cleaned, key=lambda segment: segment.start)

    def speed_at(self, progress: float) -> float:
        clamped = min(1.0, max(0.0, float(progress)))
        current_index = 0
        for index, segment in enumerate(self.segments):
            if segment.start <= clamped <= segment.end:
                current_index = index
                break
            if clamped >= segment.end:
                current_index = index
        segment = self.segments[current_index]
        speed = float(segment.kmh)
        blend_width = max(0.015, min(0.06, (segment.end - segment.start) * 0.38))
        if current_index > 0:
            previous_speed = float(self.segments[current_index - 1].kmh)
            start_blend_end = segment.start + blend_width
            if clamped < start_blend_end:
                t = float(np.clip((clamped - segment.start) / max(1e-6, blend_width), 0.0, 1.0))
                t = t * t * (3.0 - 2.0 * t)
                speed = float(interpolate(previous_speed, speed, t))
        last_index = len(self.segments) - 1
        if current_index < last_index:
            next_speed = float(self.segments[current_index + 1].kmh)
            end_blend_start = segment.end - blend_width
            if clamped > end_blend_start:
                t = float(np.clip((clamped - end_blend_start) / max(1e-6, blend_width), 0.0, 1.0))
                t = t * t * (3.0 - 2.0 * t)
                speed = float(interpolate(speed, next_speed, t))
        return float(np.clip(speed, 12.0, 380.0))


@dataclass(frozen=True, slots=True)
class RouteVisualStyle:
    layers: tuple[RouteLayerStyle, ...]
    draw_time: float = 3.1
    base_draw_time: float = 0.9


DEFAULT_ROUTE_STYLE = RouteVisualStyle(
    layers=(
        RouteLayerStyle("#1E3346", 9.4, 0.96),
        RouteLayerStyle("#6F89A6", 6.9, 1.0),
        RouteLayerStyle("#F7FBFF", 3.55, 1.0),
        RouteLayerStyle("#FFD76A", 1.15, 0.95),
    ),
)


class TripRoute:
    def __init__(
            self,
            points: Sequence[Point3DLike],
            style: RouteVisualStyle = DEFAULT_ROUTE_STYLE,
            distance_lookup: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        source = [as_point3(point) for point in points]
        if len(source) < 2:
            raise ValueError("Route requires at least two points")
        self.points = source
        self.style = style
        self.render_points = self._inset_endpoints(source)
        self.route_line = VMobject(joint_type=LineJointType.ROUND, cap_style=CapStyleType.ROUND)
        self.route_line.set_points_as_corners(self.render_points)
        self.route_line.set_fill(opacity=0)
        self._distance_fractions, self._distance_proportions = route_distance_lookup(
            self.route_line,
            lookup=distance_lookup,
        )
        self._length_rate = route_rate(
            self.route_line,
            linear,
            lookup=(self._distance_fractions, self._distance_proportions),
        )
        self.path_length = float(
            sum(
                np.linalg.norm((end - start)[:2])
                for start, end in zip(self.render_points, self.render_points[1:])
            )
        )

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def _distance_to_proportion(self, distance_fraction: float) -> float:
        return route_proportion_for_distance(
            self.route_line,
            distance_fraction,
            lookup=(self._distance_fractions, self._distance_proportions),
        )

    @staticmethod
    def _inset_endpoints(points: Sequence[np.ndarray], amount: float = 0.15) -> list[np.ndarray]:
        inset = [point.copy() for point in points]
        if len(inset) < 2:
            return inset
        start_length = float(np.linalg.norm((inset[1] - inset[0])[:2]))
        if start_length > 1e-6:
            start_ratio = min(1.0, amount / start_length)
            inset[0] = interpolate(inset[0], inset[1], start_ratio)
        end_length = float(np.linalg.norm((inset[-1] - inset[-2])[:2]))
        if end_length > 1e-6:
            end_ratio = min(1.0, amount / end_length)
            inset[-1] = interpolate(inset[-1], inset[-2], end_ratio)
        return inset

    def _endpoint_label_vector(self, anchor: np.ndarray, neighbor: np.ndarray | None) -> np.ndarray:
        if neighbor is None:
            vector = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        else:
            vector = as_point3(anchor - neighbor)
        if np.linalg.norm(vector[:2]) < 1e-6:
            vector = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        if abs(vector[0]) < 0.22:
            vector[0] += 0.35 if anchor[0] <= 0 else -0.35
        if abs(vector[1]) < 0.18:
            vector[1] += 0.28 if anchor[1] <= 0 else -0.28
        return vector

    def start_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.start, None)
        window = self.points[1: min(len(self.points), 8)]
        weights = np.linspace(1.0, 0.45, len(window))
        center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.start, center)

    def end_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.end, None)
        window = self.points[max(0, len(self.points) - 8): -1]
        weights = np.linspace(0.45, 1.0, len(window))
        center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.end, center)

    def _build_layer(self, layer_style: RouteLayerStyle) -> VMobject:
        layer = self.route_line.copy()
        layer.set_fill(opacity=0)
        layer.set_stroke(color=layer_style.color, width=layer_style.width, opacity=layer_style.opacity)
        return layer

    def _build_base_track(self) -> VGroup:
        base = VGroup(
            self.route_line.copy().set_fill(opacity=0).set_stroke(color="#1E3346", width=8.2, opacity=0.62),
            self.route_line.copy().set_fill(opacity=0).set_stroke(color="#F7FBFF", width=2.5, opacity=0.28),
        )
        return base

    def _route_normal(self, proportion: float) -> np.ndarray:
        p0 = self.route_line.point_from_proportion(max(0.0, proportion - 0.005))
        p1 = self.route_line.point_from_proportion(min(1.0, proportion + 0.005))
        tangent = p1 - p0
        tangent_norm = float(np.linalg.norm(tangent[:2]))
        if tangent_norm < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        tangent /= tangent_norm
        return np.array([-tangent[1], tangent[0], 0.0], dtype=float)

    def _build_speed_label(self, speed_profile: SpeedProfile, progress: ValueTracker) -> Text:
        label = Text("0 km/h", font="Open Sans", font_size=12).set_color("#1E2D3C")
        label.set_fill(opacity=0.98)
        label.set_stroke("#F2F7FB", width=1.15, opacity=0.82)
        initial_point = self.route_line.point_from_proportion(0.0)
        state = {
            "speed": 0.0,
            "display": -1,
            "normal": self._route_normal(0.0),
            "position": initial_point,
        }

        def update_label(mob: Text, dt: float) -> Text:
            raw = float(np.clip(progress.get_value(), 0.0, 1.0))
            target_speed = speed_profile.speed_at(raw)
            blend = min(1.0, dt * 8.0) if dt > 0 else 1.0
            state["speed"] = float(interpolate(state["speed"], target_speed, blend))
            display_speed = int(round(state["speed"]))
            if display_speed != state["display"]:
                state["display"] = display_speed
                mob.become(
                    Text(f"{display_speed} km/h", font="Open Sans", font_size=12)
                    .set_color("#1E2D3C")
                    .set_fill(opacity=0.98)
                    .set_stroke("#F2F7FB", width=1.15, opacity=0.82)
                )
            curve_progress = self._distance_to_proportion(raw)
            point = self.route_line.point_from_proportion(curve_progress)
            normal = self._route_normal(curve_progress)
            if float(np.dot(normal[:2], state["normal"][:2])) < 0.0:
                normal = -normal
            stable_normal = 0.9 * state["normal"] + 0.1 * normal
            norm = float(np.linalg.norm(stable_normal[:2]))
            if norm > 1e-6:
                stable_normal /= norm
            state["normal"] = stable_normal
            target_position = point + stable_normal * 0.24
            position_blend = min(1.0, dt * 14.0) if dt > 0 else 1.0
            state["position"] = interpolate(state["position"], target_position, position_blend)
            mob.move_to(state["position"])
            return mob

        label.add_updater(update_label)
        return label

    def _build_station_marks(self, station_progresses: Sequence[float]) -> VGroup:
        marks = VGroup()
        if len(station_progresses) < 3:
            return marks
        for progress in station_progresses[1:-1]:
            clamped = float(np.clip(progress, 0.0, 1.0))
            point = self.route_line.point_from_proportion(self._distance_to_proportion(clamped))
            outer = Circle(radius=0.038).move_to(point).set_fill("#DFE9F3", opacity=0.74).set_stroke("#1A2A3A",
                                                                                                     width=0.75,
                                                                                                     opacity=0.52)
            inner = Circle(radius=0.015).move_to(point).set_fill("#4B6784", opacity=0.86).set_stroke(width=0)
            marks.add(VGroup(outer, inner))
        return marks

    @staticmethod
    def force_foreground(scene: Scene, mobjects: Sequence | None) -> None:
        if mobjects:
            scene.add_foreground_mobjects(*mobjects)

    def create_and_animate(
            self,
            scene: Scene,
            draw_time: float | None = None,
            keep_on_top: Sequence | None = None,
            speed_profile: SpeedProfile | None = None,
            station_progresses: Sequence[float] | None = None,
    ) -> TripRoute:
        recommended = float(np.clip(2.5 + self.path_length * 0.32, 2.8, 7.2))
        duration = draw_time if draw_time is not None else max(self.style.draw_time, recommended)
        layer_templates = [self._build_layer(layer_style) for layer_style in self.style.layers]
        layers = [template.copy() for template in layer_templates]
        base_track = self._build_base_track()
        progress = ValueTracker(0.0)
        speed_label = self._build_speed_label(speed_profile, progress) if speed_profile is not None else None
        station_marks = self._build_station_marks(station_progresses or [])

        intro_animations: list[Animation] = [
            Create(base_track[0], run_time=self.style.base_draw_time, rate_func=self._length_rate),
            Create(base_track[1], run_time=self.style.base_draw_time * 1.05, rate_func=self._length_rate),
        ]
        if len(station_marks) > 0:
            intro_animations.append(FadeIn(station_marks, run_time=self.style.base_draw_time * 0.8))
        scene.play(AnimationGroup(*intro_animations, lag_ratio=0.08))
        trails = [0.0, 0.03, 0.06, 0.09]
        max_trail = max(trails)
        for index, (layer, template) in enumerate(zip(layers, layer_templates)):
            trail = trails[min(index, len(trails) - 1)]
            layer.pointwise_become_partial(template, 0.0, 0.0)
            layer.add_updater(
                lambda mob, dt, source=template, tail=trail: mob.pointwise_become_partial(
                    source,
                    0.0,
                    self._distance_to_proportion(
                        float(np.clip(progress.get_value() - tail, 0.0, 1.0))
                    ),
                )
            )
        scene.add(*layers)
        if speed_label is not None:
            scene.add_foreground_mobjects(speed_label)
        self.force_foreground(scene, keep_on_top)
        scene.play(
            progress.animate(rate_func=linear).set_value(1.0 + max_trail),
            run_time=duration,
        )
        for layer, template in zip(layers, layer_templates):
            layer.clear_updaters()
            layer.become(template)
        if speed_label is not None:
            speed_label.clear_updaters()
            scene.remove_foreground_mobjects(speed_label)
            scene.remove(speed_label)
        self.force_foreground(scene, keep_on_top)
        return self
