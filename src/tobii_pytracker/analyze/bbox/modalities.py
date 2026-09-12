from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .parsing import parse_objects_bboxes
from .plotting import plot_bbox_attention
from .scoring import analyze_bbox_attention


def parse_input_data(raw_input_data: Any) -> np.ndarray:
    if isinstance(raw_input_data, np.ndarray):
        return raw_input_data.astype(float).ravel()
    if isinstance(raw_input_data, (list, tuple)):
        return np.asarray(raw_input_data, dtype=float).ravel()
    if isinstance(raw_input_data, str):
        value = raw_input_data.strip()
        if value == "":
            return np.array([], dtype=float)
        try:
            parsed = json.loads(value)
            return np.asarray(parsed, dtype=float).ravel()
        except Exception:
            try:
                parsed = ast.literal_eval(value)
                return np.asarray(parsed, dtype=float).ravel()
            except Exception:
                return np.fromstring(value.strip("[]"), sep=" ", dtype=float)
    return np.asarray(raw_input_data, dtype=float).ravel()


def extract_timeseries_bboxes(raw_bboxes: Any) -> List[Dict[str, Any]]:
    parsed = parse_objects_bboxes(raw_bboxes)
    boxes: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        values = parsed.get("timeseries_bboxes", [])
        if isinstance(values, list):
            boxes.extend(values)
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            values = item.get("timeseries_bboxes", [])
            if isinstance(values, list):
                boxes.extend(values)
    return [
        bbox_info
        for bbox_info in boxes
        if isinstance(bbox_info, dict) and isinstance(bbox_info.get("bbox"), dict)
    ]


def extract_text_bboxes(raw_bboxes: Any, level: str = "words") -> List[Dict[str, Any]]:
    parsed = parse_objects_bboxes(raw_bboxes)
    boxes: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        values = parsed.get(level, [])
        if isinstance(values, list):
            boxes.extend(values)
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            values = item.get(level, [])
            if isinstance(values, list):
                boxes.extend(values)
    return [
        bbox_info
        for bbox_info in boxes
        if isinstance(bbox_info, dict) and isinstance(bbox_info.get("bbox"), dict)
    ]


def _filter_slide_data(
    slide_data: pd.DataFrame,
    set_name: Optional[Any],
    slide_index: Optional[Any],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    data = slide_data.copy()
    if "slide_index" in data.columns:
        data = normalize_slide_index_column(data)
    return filter_set_and_slide(data, set_name=set_name, slide_index=slide_index)


def _extract_valid_gaze(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gaze_x = pd.to_numeric(data["avg_gaze_x"], errors="coerce").to_numpy(dtype=float)
    gaze_y = pd.to_numeric(data["avg_gaze_y"], errors="coerce").to_numpy(dtype=float)
    valid_gaze = np.isfinite(gaze_x) & np.isfinite(gaze_y)
    return gaze_x[valid_gaze], gaze_y[valid_gaze]


def analyze_bbox_timeseries(
    slide_data: pd.DataFrame,
    set_name: Optional[Any],
    slide_index: Optional[Any],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    if slide_data is None or slide_data.empty:
        return pd.DataFrame()

    data = _filter_slide_data(
        slide_data=slide_data,
        set_name=set_name,
        slide_index=slide_index,
        normalize_slide_index_column=normalize_slide_index_column,
        filter_set_and_slide=filter_set_and_slide,
    )
    if data.empty:
        return pd.DataFrame()
    if "input_data" not in data.columns:
        raise ValueError("slide_data must contain 'input_data'.")
    if "objects_bboxes" not in data.columns:
        raise ValueError("slide_data must contain 'objects_bboxes'.")
    if "avg_gaze_x" not in data.columns or "avg_gaze_y" not in data.columns:
        raise ValueError("slide_data must contain 'avg_gaze_x' and 'avg_gaze_y'.")

    row = data.iloc[0]
    input_data = parse_input_data(row["input_data"])
    if input_data.size == 0:
        raise ValueError("input_data is empty for the selected slide.")
    timeseries_bboxes = extract_timeseries_bboxes(row["objects_bboxes"])
    gaze_x, gaze_y = _extract_valid_gaze(data)

    records: List[Dict[str, Any]] = []
    for bbox_index, bbox_info in enumerate(timeseries_bboxes):
        bbox = bbox_info["bbox"]
        cx = float(bbox.get("cx", np.nan))
        cy = float(bbox.get("cy", np.nan))
        w = float(bbox.get("w", np.nan))
        h = float(bbox.get("h", np.nan))
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(w) or not np.isfinite(h):
            continue
        x_min = cx - w / 2.0
        x_max = cx + w / 2.0
        y_min = cy - h / 2.0
        y_max = cy + h / 2.0
        inside = (gaze_x >= x_min) & (gaze_x <= x_max) & (gaze_y >= y_min) & (gaze_y <= y_max)
        hit_count = int(np.count_nonzero(inside))
        records.append(
            {
                "set_name": row.get("set_name", None),
                "slide_index": row.get("slide_index", None),
                "bbox_index": bbox_index,
                "channel_idx": int(bbox_info.get("channel_idx", 0)),
                "channel_name": bbox_info.get("channel_name", None),
                "input_size": int(input_data.size),
                "gaze_sample_count": int(gaze_x.size),
                "hit_count": hit_count,
                "is_visited": bool(hit_count > 0),
                "bbox": bbox,
            }
        )
    return pd.DataFrame(records)


def plot_bbox_timeseries(
    slide_data: pd.DataFrame,
    scored_bboxes: Optional[pd.DataFrame],
    set_name: Optional[Any],
    slide_index: Optional[Any],
    area_x: Optional[float],
    area_y: Optional[float],
    title: Optional[str],
    show: bool,
    save_path: Optional[Path],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
):
    if slide_data is None or slide_data.empty:
        raise ValueError("slide_data is empty.")

    data = _filter_slide_data(
        slide_data=slide_data,
        set_name=set_name,
        slide_index=slide_index,
        normalize_slide_index_column=normalize_slide_index_column,
        filter_set_and_slide=filter_set_and_slide,
    )
    if data.empty:
        raise ValueError("No rows match the provided set_name/slide_index filters.")

    row = data.iloc[0]
    input_data = parse_input_data(row["input_data"])
    if input_data.size == 0:
        raise ValueError("input_data is empty for the selected slide.")
    timeseries_bboxes = extract_timeseries_bboxes(row["objects_bboxes"])
    gaze_x, gaze_y = _extract_valid_gaze(data)

    max_abs_x = max(1.0, float(np.nanmax(np.abs(gaze_x))) if gaze_x.size else 1.0)
    max_abs_y = max(1.0, float(np.nanmax(np.abs(gaze_y))) if gaze_y.size else 1.0)
    for bbox_info in timeseries_bboxes:
        bbox = bbox_info.get("bbox", {})
        cx = float(bbox.get("cx", 0.0))
        cy = float(bbox.get("cy", 0.0))
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))
        max_abs_x = max(max_abs_x, abs(cx) + w / 2.0)
        max_abs_y = max(max_abs_y, abs(cy) + h / 2.0)

    area_x = float(area_x) if area_x is not None else 2.0 * max_abs_x
    area_y = float(area_y) if area_y is not None else 2.0 * max_abs_y
    margin_left = 60.0
    margin_bottom = 40.0
    plot_x_min = margin_left
    plot_y_min = 0.0
    plot_x_max = float(area_x)
    plot_y_max = float(area_y - margin_bottom)

    n_points = int(input_data.size)
    g_min = float(np.nanmin(input_data))
    g_max = float(np.nanmax(input_data))
    if np.isclose(g_min, g_max):
        g_min -= 0.5
        g_max += 0.5
    plot_width = plot_x_max - plot_x_min
    plot_height = plot_y_max - plot_y_min
    x_tl = plot_x_min + (np.arange(n_points) + 0.5) * (plot_width / n_points)
    points_x = x_tl - area_x / 2.0
    t = (input_data - g_min) / (g_max - g_min)
    y_tl = (plot_y_min + plot_height) - t * plot_height
    points_y = area_y / 2.0 - y_tl

    visited_indexes = set()
    if scored_bboxes is not None and not scored_bboxes.empty:
        visited_rows = scored_bboxes[scored_bboxes["is_visited"] == True]
        visited_indexes = set(visited_rows["bbox_index"].astype(int).tolist())

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(title or "Input data with gaze-visited time-series bounding boxes", fontsize=13)
    ax.set_xlabel("Center-origin x (px)")
    ax.set_ylabel("Center-origin y (px)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-area_x / 2.0, area_x / 2.0)
    ax.set_ylim(-area_y / 2.0, area_y / 2.0)
    ax.plot(points_x, points_y, color="#9aa0a6", linewidth=1.0, alpha=0.35)
    ax.scatter(points_x, points_y, c=np.linspace(0.0, 1.0, n_points), cmap="viridis", s=18, edgecolors="none", alpha=0.9)
    ax.scatter(gaze_x, gaze_y, s=10, c="black", alpha=0.2, label="gaze samples")

    channel_palette = plt.get_cmap("tab10")
    for bbox_index, bbox_info in enumerate(timeseries_bboxes):
        bbox = bbox_info["bbox"]
        channel_idx = int(bbox_info.get("channel_idx", 0))
        is_visited = bbox_index in visited_indexes
        rect = plt.Rectangle(
            (bbox["cx"] - bbox["w"] / 2.0, bbox["cy"] - bbox["h"] / 2.0),
            bbox["w"],
            bbox["h"],
            fill=False,
            linewidth=2.4 if is_visited else 1.0,
            edgecolor="red" if is_visited else channel_palette(channel_idx % 10),
            alpha=1.0 if is_visited else 0.55,
        )
        ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def analyze_bbox_text(
    slide_data: pd.DataFrame,
    level: str,
    set_name: Optional[Any],
    slide_index: Optional[Any],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    if slide_data is None or slide_data.empty:
        return pd.DataFrame()
    data = _filter_slide_data(
        slide_data=slide_data,
        set_name=set_name,
        slide_index=slide_index,
        normalize_slide_index_column=normalize_slide_index_column,
        filter_set_and_slide=filter_set_and_slide,
    )
    if data.empty:
        return pd.DataFrame()
    if "objects_bboxes" not in data.columns:
        raise ValueError("slide_data must contain 'objects_bboxes'.")
    if "avg_gaze_x" not in data.columns or "avg_gaze_y" not in data.columns:
        raise ValueError("slide_data must contain 'avg_gaze_x' and 'avg_gaze_y'.")

    row = data.iloc[0]
    text_bboxes = extract_text_bboxes(row["objects_bboxes"], level=level)
    gaze_x, gaze_y = _extract_valid_gaze(data)

    records: List[Dict[str, Any]] = []
    for bbox_index, bbox_info in enumerate(text_bboxes):
        bbox = bbox_info["bbox"]
        x_min = float(bbox.get("cx", np.nan)) - float(bbox.get("w", np.nan)) / 2.0
        x_max = float(bbox.get("cx", np.nan)) + float(bbox.get("w", np.nan)) / 2.0
        y_min = float(bbox.get("cy", np.nan)) - float(bbox.get("h", np.nan)) / 2.0
        y_max = float(bbox.get("cy", np.nan)) + float(bbox.get("h", np.nan)) / 2.0
        if not all(np.isfinite([x_min, x_max, y_min, y_max])):
            continue
        inside = (gaze_x >= x_min) & (gaze_x <= x_max) & (gaze_y >= y_min) & (gaze_y <= y_max)
        hit_count = int(np.count_nonzero(inside))
        records.append(
            {
                "set_name": row.get("set_name", None),
                "slide_index": row.get("slide_index", None),
                "bbox_index": bbox_index,
                "text": bbox_info.get("text", None),
                "line_idx": bbox_info.get("line_idx", None),
                "word_idx": bbox_info.get("word_idx", None),
                "gaze_sample_count": int(gaze_x.size),
                "hit_count": hit_count,
                "is_visited": bool(hit_count > 0),
                "bbox": bbox,
            }
        )
    return pd.DataFrame(records)


def plot_bbox_text(
    slide_data: pd.DataFrame,
    level: str,
    scored_bboxes: Optional[pd.DataFrame],
    set_name: Optional[Any],
    slide_index: Optional[Any],
    area_x: Optional[float],
    area_y: Optional[float],
    title: Optional[str],
    show: bool,
    save_path: Optional[Path],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
):
    if slide_data is None or slide_data.empty:
        raise ValueError("slide_data is empty.")
    data = _filter_slide_data(
        slide_data=slide_data,
        set_name=set_name,
        slide_index=slide_index,
        normalize_slide_index_column=normalize_slide_index_column,
        filter_set_and_slide=filter_set_and_slide,
    )
    if data.empty:
        raise ValueError("No rows match the provided set_name/slide_index filters.")

    row = data.iloc[0]
    text_bboxes = extract_text_bboxes(row["objects_bboxes"], level=level)
    gaze_x, gaze_y = _extract_valid_gaze(data)
    visited_indexes = set()
    if scored_bboxes is not None and not scored_bboxes.empty:
        visited_indexes = set(
            scored_bboxes.loc[scored_bboxes["is_visited"] == True, "bbox_index"].astype(int).tolist()
        )

    max_abs_x = max(1.0, float(np.nanmax(np.abs(gaze_x))) if gaze_x.size else 1.0)
    max_abs_y = max(1.0, float(np.nanmax(np.abs(gaze_y))) if gaze_y.size else 1.0)
    for bbox_info in text_bboxes:
        bbox = bbox_info.get("bbox", {})
        cx = float(bbox.get("cx", 0.0))
        cy = float(bbox.get("cy", 0.0))
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))
        max_abs_x = max(max_abs_x, abs(cx) + w / 2.0)
        max_abs_y = max(max_abs_y, abs(cy) + h / 2.0)
    area_x = float(area_x) if area_x is not None else 2.0 * max_abs_x
    area_y = float(area_y) if area_y is not None else 2.0 * max_abs_y

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(title or "Text with gaze-visited bounding boxes", fontsize=13)
    ax.set_xlabel("Center-origin x (px)")
    ax.set_ylabel("Center-origin y (px)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-area_x / 2.0, area_x / 2.0)
    ax.set_ylim(-area_y / 2.0, area_y / 2.0)
    ax.scatter(gaze_x, gaze_y, s=10, c="black", alpha=0.2, label="gaze samples")

    for bbox_index, bbox_info in enumerate(text_bboxes):
        bbox = bbox_info["bbox"]
        is_visited = bbox_index in visited_indexes
        rect = plt.Rectangle(
            (bbox["cx"] - bbox["w"] / 2.0, bbox["cy"] - bbox["h"] / 2.0),
            bbox["w"],
            bbox["h"],
            fill=False,
            linewidth=2.2 if is_visited else 1.0,
            edgecolor="red" if is_visited else "#1f77b4",
            alpha=0.95 if is_visited else 0.6,
        )
        ax.add_patch(rect)
        label = bbox_info.get("text", "")
        if label:
            ax.text(
                bbox["cx"] - bbox["w"] / 2.0,
                bbox["cy"] + bbox["h"] / 2.0 + 6,
                str(label),
                fontsize=7,
                alpha=0.85 if is_visited else 0.6,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def analyze_bbox_image(
    slide_data: pd.DataFrame,
    set_name: Optional[Any],
    slide_index: Optional[Any],
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame],
    filter_set_and_slide: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    if slide_data is None or slide_data.empty:
        return pd.DataFrame()
    data = slide_data.copy()
    if "slide_index" in data.columns:
        data = normalize_slide_index_column(data)
    data = filter_set_and_slide(data, set_name=set_name, slide_index=slide_index)
    if data.empty:
        return pd.DataFrame()
    if "objects_bboxes" not in data.columns:
        raise ValueError("slide_data must contain 'objects_bboxes'.")
    if "avg_gaze_x" not in data.columns or "avg_gaze_y" not in data.columns:
        raise ValueError("slide_data must contain 'avg_gaze_x' and 'avg_gaze_y'.")

    if "set_name" in data.columns and "slide_index" in data.columns:
        raw_data = (
            data[["set_name", "slide_index", "objects_bboxes"]]
            .groupby(["set_name", "slide_index"], as_index=False)
            .agg({"objects_bboxes": "first"})
        )
    else:
        raw_data = data[["objects_bboxes"]].head(1).copy()
        raw_data["set_name"] = None
        raw_data["slide_index"] = None
        raw_data = raw_data[["set_name", "slide_index", "objects_bboxes"]]
    gaze_data = data
    return analyze_bbox_attention(
        raw_data=raw_data,
        gaze_data=gaze_data,
        use_fixations=False,
        normalize_slide_index_column=normalize_slide_index_column,
        filter_set_and_slide=filter_set_and_slide,
        resolve_gaze_columns=lambda use_fixations: ("avg_gaze_x", "avg_gaze_y", None),
    )


def plot_bbox_image(
    scored_bboxes: pd.DataFrame,
    gaze_data: pd.DataFrame,
    screenshot_path: Path,
    set_name: Optional[str],
    slide_index: Optional[int],
    title: Optional[str],
    top_k: Optional[int],
    min_hits: int,
    show_gaze: bool,
    show: bool,
    save_path: Optional[Path],
    filter_set_and_slide: Callable[..., pd.DataFrame],
):
    return plot_bbox_attention(
        scored_bboxes=scored_bboxes,
        gaze_data=gaze_data,
        screenshot_path=screenshot_path,
        set_name=set_name,
        slide_index=slide_index,
        title=title,
        top_k=top_k,
        min_hits=min_hits,
        show_gaze=show_gaze,
        show=show,
        save_path=save_path,
        filter_set_and_slide=filter_set_and_slide,
    )
