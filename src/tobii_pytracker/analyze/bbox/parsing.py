from __future__ import annotations

import json
from typing import Any, Dict
import ast
from pathlib import Path

import numpy as np

def parse_objects_bboxes(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return {"image_bboxes": []}

    return {"image_bboxes": []}


def parse_input_data(raw_input_data):
    if isinstance(raw_input_data, str):
        return np.fromstring(raw_input_data.strip("[]"), sep=" ")
    return np.asarray(raw_input_data, dtype=float)

def extract_timeseries_bboxes(raw_bboxes):
    boxes = []
    if isinstance(raw_bboxes, dict):
        if "timeseries_bboxes" in raw_bboxes:
            boxes.extend(raw_bboxes["timeseries_bboxes"])
        return boxes
    if isinstance(raw_bboxes, list):
        for item in raw_bboxes:
            if isinstance(item, dict) and "timeseries_bboxes" in item:
                boxes.extend(item["timeseries_bboxes"])
    return boxes


def parse_serialized(value):
    """
    Convert a string representation of a Python object into the object itself.

    Values already represented as dictionaries or lists are returned unchanged.
    """
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


def extract_text_bboxes(raw_bboxes, level="words"):
    """
    Extract word-level or line-level text bounding boxes.

    Supported structures:
        {"words": [...], "lines": [...]}

    and:

        [
            {"words": [...], "lines": [...]},
            ...
        ]
    """
    raw_bboxes = parse_serialized(raw_bboxes)
    boxes = []

    if isinstance(raw_bboxes, dict):
        boxes.extend(raw_bboxes.get(level, []))

    elif isinstance(raw_bboxes, list):
        for item in raw_bboxes:
            item = parse_serialized(item)

            if isinstance(item, dict):
                boxes.extend(item.get(level, []))

    return boxes


def extract_gaze_points(row, slide_data):
    """
    Extract gaze coordinates in the center-origin coordinate system.

    First try flattened avg_gaze_x/avg_gaze_y columns. If unavailable,
    extract coordinates from the gaze_data list.
    """
    if (
            "avg_gaze_x" in slide_data.columns
            and "avg_gaze_y" in slide_data.columns
    ):
        gaze_x = slide_data["avg_gaze_x"].to_numpy(dtype=float)
        gaze_y = slide_data["avg_gaze_y"].to_numpy(dtype=float)

    elif "gaze_data" in row:
        gaze_data = parse_serialized(row["gaze_data"])

        if not isinstance(gaze_data, list):
            gaze_data = []

        gaze_x = np.asarray(
            [
                sample.get("avg_gaze_x", np.nan)
                for sample in gaze_data
                if isinstance(sample, dict)
            ],
            dtype=float,
        )

        gaze_y = np.asarray(
            [
                sample.get("avg_gaze_y", np.nan)
                for sample in gaze_data
                if isinstance(sample, dict)
            ],
            dtype=float,
        )

    else:
        gaze_x = np.asarray([], dtype=float)
        gaze_y = np.asarray([], dtype=float)

    valid = np.isfinite(gaze_x) & np.isfinite(gaze_y)

    return gaze_x[valid], gaze_y[valid]


def gaze_inside_bbox(gaze_x, gaze_y, bbox, padding=0.0):
    """
    Return a Boolean mask indicating which gaze samples are inside a bbox.

    padding expands the bounding box by the given number of pixels.
    """
    x_min = bbox["cx"] - bbox["w"] / 2.0 - padding
    x_max = bbox["cx"] + bbox["w"] / 2.0 + padding
    y_min = bbox["cy"] - bbox["h"] / 2.0 - padding
    y_max = bbox["cy"] + bbox["h"] / 2.0 + padding

    return (
            (gaze_x >= x_min)
            & (gaze_x <= x_max)
            & (gaze_y >= y_min)
            & (gaze_y <= y_max)
    )


def parse_literal(value):
    """
    Parse Python-like lists/dictionaries stored as CSV strings.

    If value is already parsed, return it unchanged.
    """
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "Could not parse a logged Python literal."
        ) from exc

def extract_image_bboxes(raw_bboxes):
    """
    Extract image bounding boxes from the possible logged structures.

    Supported examples:
        {"image_bboxes": [...]}

        [
            {"image_bboxes": [...]},
            ...
        ]

        A string representation of either structure.
    """
    raw_bboxes = parse_literal(raw_bboxes)
    boxes = []

    if isinstance(raw_bboxes, dict):
        boxes.extend(raw_bboxes.get("image_bboxes", []))

    elif isinstance(raw_bboxes, list):
        for item in raw_bboxes:
            if isinstance(item, dict):
                if "image_bboxes" in item:
                    boxes.extend(item["image_bboxes"])

                # Also support a directly logged list of bbox entries.
                elif "bbox" in item:
                    boxes.append(item)

    return boxes

def extract_gaze(row, slide_data):
    """
    Extract gaze samples in center-origin stimulus coordinates.

    It first tries the flattened avg_gaze_x/avg_gaze_y columns. If they
    are unavailable, it reads the gaze_data list stored in the row.
    """
    if (
            "avg_gaze_x" in slide_data.columns
            and "avg_gaze_y" in slide_data.columns
    ):
        gaze_x = slide_data["avg_gaze_x"].to_numpy(dtype=float)
        gaze_y = slide_data["avg_gaze_y"].to_numpy(dtype=float)

    elif "gaze_data" in row:
        gaze_data = parse_literal(row["gaze_data"])

        if not isinstance(gaze_data, list):
            raise ValueError(
                "gaze_data must contain a list of gaze-event dictionaries."
            )

        gaze_x = np.asarray(
            [event.get("avg_gaze_x", np.nan) for event in gaze_data],
            dtype=float,
        )
        gaze_y = np.asarray(
            [event.get("avg_gaze_y", np.nan) for event in gaze_data],
            dtype=float,
        )

    else:
        raise KeyError(
            "No avg_gaze_x/avg_gaze_y columns or gaze_data field found."
        )

    valid = np.isfinite(gaze_x) & np.isfinite(gaze_y)

    return gaze_x[valid], gaze_y[valid]

def resolve_image_path(path_value, root="../"):
    """
    Resolve image paths produced on Windows or Linux.

    Windows backslashes are converted to platform-independent separators.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("The image path is missing.")

    normalized = path_value.replace("\\", "/")
    path = Path(normalized)

    candidates = [
        path,
        Path(root) / path,
        Path.cwd() / path,
        Path.cwd() / root / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find image '{path_value}'. Attempted:\n{attempted}"
    )

def bbox_contains_gaze(bbox, gaze_x, gaze_y):
    """
    Return a mask indicating which gaze samples are inside the bbox.
    """
    x_min = float(bbox["cx"]) - float(bbox["w"]) / 2.0
    x_max = float(bbox["cx"]) + float(bbox["w"]) / 2.0
    y_min = float(bbox["cy"]) - float(bbox["h"]) / 2.0
    y_max = float(bbox["cy"]) + float(bbox["h"]) / 2.0

    return (
            (gaze_x >= x_min)
            & (gaze_x <= x_max)
            & (gaze_y >= y_min)
            & (gaze_y <= y_max)
    )