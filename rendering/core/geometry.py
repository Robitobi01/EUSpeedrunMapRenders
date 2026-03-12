from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence

import numpy as np
from manim import RIGHT, VMobject, interpolate, inverse_interpolate, smooth
from manim.typing import Point3DLike


def as_point3(vector: Point3DLike) -> np.ndarray:
    value = np.array(vector, dtype=float)
    if value.shape[0] == 2:
        value = np.array([value[0], value[1], 0.0], dtype=float)
    value[2] = 0.0
    return value


def normalize_xy(vector: Point3DLike, fallback: Point3DLike = RIGHT) -> np.ndarray:
    value = as_point3(vector)
    norm = float(np.linalg.norm(value[:2]))
    if norm >= 1e-6:
        return value / norm
    fallback_value = as_point3(fallback)
    fallback_norm = float(np.linalg.norm(fallback_value[:2]))
    if fallback_norm >= 1e-6:
        return fallback_value / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=float)


def simplify_polyline(
        points: Sequence[Point3DLike],
        min_distance: float = 0.035,
        collinear_dot: float = 0.9994,
) -> list[np.ndarray]:
    if len(points) <= 2:
        return [as_point3(point) for point in points]
    simplified = [as_point3(points[0])]
    for index in range(1, len(points) - 1):
        previous = simplified[-1]
        current = as_point3(points[index])
        following = as_point3(points[index + 1])
        if np.linalg.norm((current - previous)[:2]) < min_distance:
            continue
        incoming = current - previous
        outgoing = following - current
        incoming_norm = float(np.linalg.norm(incoming[:2]))
        outgoing_norm = float(np.linalg.norm(outgoing[:2]))
        if incoming_norm < 1e-6 or outgoing_norm < 1e-6:
            continue
        incoming_unit = incoming[:2] / incoming_norm
        outgoing_unit = outgoing[:2] / outgoing_norm
        if incoming_norm < min_distance * 2.2 and outgoing_norm < min_distance * 2.2:
            if float(np.dot(incoming_unit, outgoing_unit)) > collinear_dot:
                continue
        simplified.append(current)
    simplified.append(as_point3(points[-1]))
    return simplified


def smooth_polyline(
        points: Sequence[Point3DLike],
        passes: int = 2,
        keep_turn_dot: float = 0.9,
        max_shift: float = 0.022,
) -> list[np.ndarray]:
    shaped = [as_point3(point) for point in points]
    if len(shaped) <= 2:
        return shaped
    for _ in range(max(0, passes)):
        updated = [shaped[0].copy()]
        for index in range(1, len(shaped) - 1):
            previous = shaped[index - 1]
            current = shaped[index]
            following = shaped[index + 1]
            incoming = normalize_xy(current - previous)
            outgoing = normalize_xy(following - current)
            if float(np.dot(incoming[:2], outgoing[:2])) < keep_turn_dot:
                updated.append(current.copy())
                continue
            target = 0.22 * previous + 0.56 * current + 0.22 * following
            shift = as_point3(target - current)
            shift_norm = float(np.linalg.norm(shift[:2]))
            if shift_norm > max_shift:
                shift *= max_shift / shift_norm
            updated.append(current + shift)
        updated.append(shaped[-1].copy())
        shaped = updated
    return shaped


def sample_polyline(points: Sequence[Point3DLike], step: float = 0.18) -> list[np.ndarray]:
    if not points:
        return []
    source = [as_point3(point) for point in points]
    if len(source) == 1:
        return [source[0]]
    samples = [source[0]]
    for start, end in zip(source, source[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment[:2]))
        parts = max(1, int(np.ceil(segment_length / max(step, 1e-6))))
        for part in range(1, parts + 1):
            samples.append(interpolate(start, end, part / parts))
    return samples


def route_rate(route: VMobject, rate_func: Callable[[float], float] = smooth) -> Callable[[float], float]:
    curves = route.get_curve_functions_with_lengths()
    if not curves:
        return rate_func
    lengths = [float(length) for _, length in curves]
    total = sum(lengths)
    if total <= 1e-9:
        return rate_func
    curve_fraction = 1 / len(lengths)
    cumulative = [0.0]
    running = 0.0
    for length in lengths:
        running += length / total
        cumulative.append(running)

    def mapped(t: float) -> float:
        eased = rate_func(t)
        if eased <= 0:
            return 0.0
        if eased >= cumulative[-1]:
            return 1.0
        index = bisect_right(cumulative, eased) - 1
        start = cumulative[index]
        end = cumulative[index + 1]
        interval_start = index * curve_fraction
        interval_end = (index + 1) * curve_fraction
        return float(interpolate(interval_start, interval_end, inverse_interpolate(start, end, eased)))

    return mapped
