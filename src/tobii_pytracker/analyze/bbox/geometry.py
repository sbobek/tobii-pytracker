from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.path import Path as MplPath


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
