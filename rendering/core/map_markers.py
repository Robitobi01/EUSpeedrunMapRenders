from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import (
    Animation,
    AnimationGroup,
    BLACK,
    Circle,
    DOWN,
    DR,
    DL,
    FadeIn,
    FadeOut,
    Indicate,
    LEFT,
    RIGHT,
    Restore,
    RoundedRectangle,
    Scene,
    Succession,
    TAU,
    Text,
    UP,
    UR,
    UL,
    VGroup,
    WHITE,
    rate_functions,
    there_and_back,
)
from manim.typing import Point3DLike

from .geometry import as_point3, normalize_xy, sample_polyline


class MapMarker(VGroup):
    def __init__(
            self,
            label: str,
            location: Point3DLike,
            label_direction: Point3DLike | None = None,
    ) -> None:
        super().__init__()
        location_point = as_point3(location)
        fallback = np.array(
            [
                1.0 if location_point[0] <= 0 else -1.0,
                0.35 if location_point[1] <= 0 else -0.35,
                0.0,
            ],
            dtype=float,
        )
        self.preferred_label_vector = normalize_xy(label_direction if label_direction is not None else fallback,
                                                   fallback)
        self.label_direction = self.snap_label_direction(self.preferred_label_vector, location_point)

        self.halo = VGroup(
            Circle(radius=0.34).set_fill("#68B7FF", opacity=0.09).set_stroke(width=0),
            Circle(radius=0.24).set_fill("#FFD45C", opacity=0.10).set_stroke(width=0),
        ).move_to(location_point)
        self.dot = VGroup(
            Circle(radius=0.17).set_fill(WHITE, opacity=0.07).set_stroke(width=0),
            Circle(radius=0.105).set_fill(BLACK, opacity=0.64).set_stroke(WHITE, width=2.25, opacity=0.94),
            Circle(radius=0.05).set_fill("#FFD45C", opacity=1.0).set_stroke(WHITE, width=0.6, opacity=0.78),
        ).move_to(location_point)

        self.text = Text(label, font="Open Sans", font_size=24).set_color(WHITE)
        width = self.text.width + 0.34
        height = self.text.height + 0.18
        self.label_shadow = RoundedRectangle(corner_radius=0.12, width=width, height=height).set_fill(BLACK,
                                                                                                      opacity=0.22).set_stroke(
            width=0)
        self.label_bg = RoundedRectangle(corner_radius=0.12, width=width, height=height).set_fill("#151B21",
                                                                                                  opacity=0.8).set_stroke(
            WHITE, width=1.0, opacity=0.18)
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

    @staticmethod
    def snap_label_direction(direction: Point3DLike | None, location: Point3DLike) -> np.ndarray:
        candidates = [LEFT, RIGHT, UP, DOWN, UL, UR, DL, DR]
        location_point = as_point3(location)
        fallback = np.array(
            [
                1.0 if location_point[0] <= 0 else -1.0,
                0.35 if location_point[1] <= 0 else -0.35,
                0.0,
            ],
            dtype=float,
        )
        target = normalize_xy(direction if direction is not None else fallback, fallback)
        best = max(candidates, key=lambda candidate: float(np.dot(target[:2], normalize_xy(candidate)[:2])))
        return as_point3(best)

    @staticmethod
    def _box_distance(label_box: VGroup, point: Point3DLike) -> float:
        left = label_box.get_left()[0]
        right = label_box.get_right()[0]
        bottom = label_box.get_bottom()[1]
        top = label_box.get_top()[1]
        x = float(point[0])
        y = float(point[1])
        dx = 0.0 if left <= x <= right else min(abs(x - left), abs(x - right))
        dy = 0.0 if bottom <= y <= top else min(abs(y - bottom), abs(y - top))
        return float(np.hypot(dx, dy))

    def _route_overlap_penalty(self, label_box: VGroup, route_samples: Sequence[Point3DLike]) -> float:
        penalty_sum = 0.0
        max_penalty = 0.0
        min_distance = float("inf")
        for index, sample in enumerate(route_samples[:90]):
            distance = self._box_distance(label_box, sample)
            min_distance = min(min_distance, distance)
            weight = 1.7 / (1.0 + 0.14 * index)
            penalty = weight * np.exp(-((distance / 0.24) ** 2))
            penalty_sum += penalty
            max_penalty = max(max_penalty, penalty)
        hard_penalty = max(0.0, 0.19 - min_distance) * 120.0 if route_samples else 0.0
        return float(penalty_sum * 0.5 + max_penalty * 2.6 + hard_penalty)

    def choose_label_direction(self, frame_width: float, frame_height: float, margin: float = 0.28) -> None:
        self.choose_label_direction_with_route(frame_width, frame_height, None, 0, margin=margin)

    def choose_label_direction_with_route(
            self,
            frame_width: float,
            frame_height: float,
            route_points: Sequence[Point3DLike] | None,
            endpoint_index: int,
            margin: float = 0.28,
    ) -> None:
        angles = np.linspace(0.0, TAU, 24, endpoint=False)
        candidates = [np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float) for angle in angles]
        min_x = -frame_width / 2 + margin
        max_x = frame_width / 2 - margin
        min_y = -frame_height / 2 + margin
        max_y = frame_height / 2 - margin
        desired = normalize_xy(self.preferred_label_vector)
        route_samples = sample_polyline(route_points or [], step=0.16)
        if route_samples:
            route_samples = route_samples[1:] if endpoint_index == 0 else list(reversed(route_samples[:-1]))

        best_direction = self.label_direction
        best_score = -1e12
        for candidate in candidates:
            trial = self.label_target.copy()
            trial.next_to(self.dot, candidate, buff=0.34)
            overflow = 0.0
            overflow += max(0.0, min_x - trial.get_left()[0])
            overflow += max(0.0, trial.get_right()[0] - max_x)
            overflow += max(0.0, min_y - trial.get_bottom()[1])
            overflow += max(0.0, trial.get_top()[1] - max_y)
            alignment = float(np.dot(desired[:2], normalize_xy(candidate)[:2]))
            penalty = self._route_overlap_penalty(trial, route_samples)
            score = alignment * 4.2 - penalty * 2.2 - overflow * 22.0
            if score > best_score:
                best_score = score
                best_direction = as_point3(candidate)
        self.label_direction = best_direction
        self.label_target.next_to(self.dot, self.label_direction, buff=0.34)
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
        if np.linalg.norm(shift[:2]) <= 0:
            return
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

    def foreground_mobjects(self) -> list:
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
        pulse_hot = Circle(radius=0.18).move_to(self.dot.get_center()).set_fill(opacity=0).set_stroke("#FFD45C",
                                                                                                      width=2.6,
                                                                                                      opacity=0.95)
        pulse_cool = Circle(radius=0.18).move_to(self.dot.get_center()).set_fill(opacity=0).set_stroke(WHITE, width=1.8,
                                                                                                       opacity=0.85)
        return AnimationGroup(
            Indicate(self.dot, color="#FFD45C", scale_factor=1.14, run_time=0.7),
            FadeIn(pulse_hot, scale=0.5, run_time=0.12),
            FadeOut(pulse_hot, scale=2.8, run_time=0.62),
            FadeIn(pulse_cool, scale=0.75, run_time=0.12),
            FadeOut(pulse_cool, scale=2.2, run_time=0.55),
            lag_ratio=0.0,
        )
