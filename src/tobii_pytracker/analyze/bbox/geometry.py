from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.path import Path as MplPath
import pandas as pd

def bbox_edges_centered(bbox: Dict[str, float]) -> Dict[str, float]:
    cx = float(bbox["cx"])
    cy = float(bbox["cy"])
    w = float(bbox["w"])
    h = float(bbox["h"])

    return {
        "x_min": cx - w / 2.0,
        "x_max": cx + w / 2.0,
        "y_min": cy - h / 2.0,
        "y_max": cy + h / 2.0,
    }


def polygon_vertices(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None

    vertices: List[List[float]] = []

    try:
        for point in value:
            if isinstance(point, dict):
                vertices.append([float(point["x"]), float(point["y"])])
            else:
                x, y = point
                vertices.append([float(x), float(y)])
    except (TypeError, ValueError, KeyError, IndexError):
        return None

    if len(vertices) < 3:
        return None

    return np.asarray(vertices, dtype=float)


def point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    return bool(MplPath(polygon, closed=True).contains_point((x, y)))


def polygon_to_plot_coords(
    polygon: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    return np.column_stack([
        width / 2.0 + polygon[:, 0],
        height / 2.0 - polygon[:, 1],
    ])


def point_inside_bbox(
    x: float,
    y: float,
    bbox: Dict[str, float],
    margin: float = 2.0,
) -> bool:
    edges = bbox_edges_centered(bbox)
    return (
        edges["x_min"] - margin <= x <= edges["x_max"] + margin
        and edges["y_min"] - margin <= y <= edges["y_max"] + margin
    )


def get_plot_bounds(
        area_x: float,
        area_y: float,
) -> tuple[float, float, float, float]:
    margin_left = 60
    margin_bottom = 40

    plot_x_min = float(margin_left)
    plot_y_min = 0.0
    plot_x_max = float(area_x)
    plot_y_max = float(area_y - margin_bottom)

    return plot_x_min, plot_y_min, plot_x_max, plot_y_max

def calculate_points(
        input_data: np.ndarray,
        area_x: float,
        area_y: float,
        plot_x_min: float,
        plot_y_min: float,
        plot_x_max: float,
        plot_y_max: float,
        g_min: float,
        g_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(input_data)

    plot_width = plot_x_max - plot_x_min
    plot_height = plot_y_max - plot_y_min
    band_height = plot_height

    x_tl = (
            plot_x_min
            + (np.arange(n_points) + 0.5)
            * (plot_width / n_points)
    )
    points_x = x_tl - area_x / 2.0

    t = (input_data - g_min) / (g_max - g_min)

    y_tl = (
            (plot_y_min + band_height)
            - t * band_height
    )
    points_y = area_y / 2.0 - y_tl

    return points_x, points_y


def get_valid_gaze(
        row: Any, background_data: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    gaze_x = (
        row["avg_gaze_x"]
        if "avg_gaze_x" in row
        else background_data["avg_gaze_x"].to_numpy()
    )
    gaze_y = (
        row["avg_gaze_y"]
        if "avg_gaze_y" in row
        else background_data["avg_gaze_y"].to_numpy()
    )

    if not isinstance(gaze_x, np.ndarray):
        gaze_x = background_data["avg_gaze_x"].to_numpy(dtype=float)

    if not isinstance(gaze_y, np.ndarray):
        gaze_y = background_data["avg_gaze_y"].to_numpy(dtype=float)

    valid_gaze = np.isfinite(gaze_x) & np.isfinite(gaze_y)

    gaze_x = gaze_x[valid_gaze]
    gaze_y = gaze_y[valid_gaze]

    return gaze_x, gaze_y

def _get_bbox_bounds(
        bbox: dict[str, Any],
) -> tuple[float, float, float, float]:
    x_min = bbox["cx"] - bbox["w"] / 2.0
    x_max = bbox["cx"] + bbox["w"] / 2.0
    y_min = bbox["cy"] - bbox["h"] / 2.0
    y_max = bbox["cy"] + bbox["h"] / 2.0

    return x_min, x_max, y_min, y_max

def get_visited_bboxes(
        timeseries_bboxes: list[dict[str, Any]],
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
) -> list[dict[str, Any]]:
    visited_bboxes = []

    for bbox_info in timeseries_bboxes:
        bbox = bbox_info["bbox"]

        x_min, x_max, y_min, y_max = _get_bbox_bounds(bbox)

        inside = (
                (gaze_x >= x_min)
                & (gaze_x <= x_max)
                & (gaze_y >= y_min)
                & (gaze_y <= y_max)
        )

        if np.any(inside):
            visited_bboxes.append(bbox_info)

    return visited_bboxes


def get_series_range(
        input_data: np.ndarray,
) -> tuple[float, float]:
    series_2d = input_data[:, np.newaxis]

    g_min = float(np.nanmin(series_2d))
    g_max = float(np.nanmax(series_2d))

    if np.isclose(g_min, g_max):
        g_min -= 0.5
        g_max += 0.5

    return g_min, g_max