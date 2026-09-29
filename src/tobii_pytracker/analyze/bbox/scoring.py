from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .geometry import (
    point_inside_bbox,
    point_inside_polygon,
    polygon_vertices,
)
from .parsing import parse_objects_bboxes


def _default_normalize_slide_index_column(
    data: pd.DataFrame,
    column: str = "slide_index",
) -> pd.DataFrame:
    normalized = data.copy()
    normalized[column] = pd.to_numeric(
        normalized[column],
        errors="coerce",
    ).astype("Int64")
    return normalized


def _default_filter_set_and_slide(
    data: pd.DataFrame,
    set_name: Optional[Any] = None,
    slide_index: Optional[Any] = None,
) -> pd.DataFrame:
    filtered = data

    if set_name is not None and "set_name" in filtered.columns:
        filtered = filtered[
            filtered["set_name"].astype(str) == str(set_name)
        ]

    if slide_index is not None and "slide_index" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(
                filtered["slide_index"],
                errors="coerce",
            ) == int(slide_index)
        ]

    return filtered


def _default_resolve_gaze_columns(
    use_fixations: bool,
) -> tuple[str, str, Optional[str]]:
    if use_fixations:
        return "x_mean", "y_mean", "duration"
    return "avg_gaze_x", "avg_gaze_y", None


def analyze_bbox_attention(
    raw_data: pd.DataFrame,
    gaze_data: pd.DataFrame,
    use_fixations: bool = False,
    *,
    normalize_slide_index_column: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    filter_set_and_slide: Callable[[pd.DataFrame, Optional[Any], Optional[Any]], pd.DataFrame] | None = None,
    resolve_gaze_columns: Callable[[bool], tuple[str, str, Optional[str]]] | None = None,
) -> pd.DataFrame:
    records = []

    if "set_name" not in raw_data.columns:
        raise ValueError("raw_data must contain set_name column.")

    normalize_slide_index_column = (
        normalize_slide_index_column or _default_normalize_slide_index_column
    )
    filter_set_and_slide = filter_set_and_slide or _default_filter_set_and_slide
    resolve_gaze_columns = resolve_gaze_columns or _default_resolve_gaze_columns

    raw_data = raw_data.copy()
    gaze_data = gaze_data.copy()

    if "slide_index" not in raw_data.columns:
        raw_data["slide_index"] = raw_data.groupby("set_name").cumcount()

    raw_data = normalize_slide_index_column(raw_data)
    gaze_data = normalize_slide_index_column(gaze_data)
    x_col, y_col, duration_col = resolve_gaze_columns(use_fixations=use_fixations)

    for _, row in raw_data.iterrows():
        set_name = str(row["set_name"])
        slide_index = int(row["slide_index"])

        slide_gaze = filter_set_and_slide(
            gaze_data,
            set_name=set_name,
            slide_index=slide_index,
        ).copy()
        slide_gaze = slide_gaze.dropna(subset=[x_col, y_col])

        bbox_container = parse_objects_bboxes(row.get("objects_bboxes", {}))
        image_bboxes = bbox_container.get("image_bboxes", [])
        total_points = len(slide_gaze)

        for bbox_index, bbox_record in enumerate(image_bboxes):
            bbox_payload = bbox_record.get("bbox", {})
            polygon = polygon_vertices(bbox_payload)
            if polygon is None:
                polygon = polygon_vertices(bbox_record.get("polygon"))

            rect_bbox = bbox_record.get("rect_bbox", {})
            if not isinstance(rect_bbox, dict):
                rect_bbox = {}

            if polygon is not None:
                x_min = float(polygon[:, 0].min())
                x_max = float(polygon[:, 0].max())
                y_min = float(polygon[:, 1].min())
                y_max = float(polygon[:, 1].max())
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
            elif isinstance(bbox_payload, dict) and {
                "cx",
                "cy",
                "w",
                "h",
            }.issubset(bbox_payload.keys()):
                cx = float(bbox_payload["cx"])
                cy = float(bbox_payload["cy"])
                bbox_width = float(bbox_payload["w"])
                bbox_height = float(bbox_payload["h"])
            elif {"cx", "cy", "w", "h"}.issubset(rect_bbox.keys()):
                cx = float(rect_bbox["cx"])
                cy = float(rect_bbox["cy"])
                bbox_width = float(rect_bbox["w"])
                bbox_height = float(rect_bbox["h"])
            else:
                continue

            hits = []
            hit_gaze_indices = []

            for gaze_idx, gaze_row in slide_gaze.iterrows():
                x = float(gaze_row[x_col])
                y = float(gaze_row[y_col])

                inside = (
                    point_inside_polygon(x, y, polygon)
                    if polygon is not None
                    else point_inside_bbox(
                        x,
                        y,
                        {
                            "cx": cx,
                            "cy": cy,
                            "w": bbox_width,
                            "h": bbox_height,
                        },
                    )
                )

                if inside:
                    hits.append(gaze_row)
                    hit_gaze_indices.append(int(gaze_idx))

            hit_count = len(hits)
            coverage = hit_count / total_points if total_points else 0.0

            if hits and duration_col and duration_col in slide_gaze.columns:
                dwell_time = float(
                    pd.DataFrame(hits)[duration_col].fillna(0).sum()
                )
            else:
                dwell_time = float(hit_count)

            records.append({
                "set_name": set_name,
                "slide_index": slide_index,
                "screenshot_file": row.get("screenshot_file"),
                "input_data": row.get("input_data"),
                "bbox_index": bbox_index,
                "bbox_class": bbox_record.get("class"),
                "bbox_conf": bbox_record.get("conf"),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(bbox_width),
                "h": float(bbox_height),
                "polygon": polygon.tolist() if polygon is not None else None,
                "total_gaze_points": total_points,
                "hit_count": hit_count,
                "hit_gaze_indices": hit_gaze_indices,
                "coverage": coverage,
                "dwell_time": dwell_time,
                "attention_score": coverage,
            })

    result = pd.DataFrame(records)

    if not result.empty:
        result["attention_rank"] = (
            result.groupby(["set_name", "slide_index"])["attention_score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

    return result


def evaluate_bbox_attention(scored_bboxes: pd.DataFrame) -> pd.DataFrame:
    if scored_bboxes is None or scored_bboxes.empty:
        return pd.DataFrame()

    grouped = []

    for (set_name, slide_index), group in scored_bboxes.groupby(
        ["set_name", "slide_index"]
    ):
        total_points = int(group["total_gaze_points"].max())
        bbox_count = len(group)
        attended_bboxes = int((group["hit_count"] > 0).sum())
        total_bbox_memberships = int(group["hit_count"].sum())

        unique_hit_indices = set()
        if "hit_gaze_indices" in group.columns:
            for indices in group["hit_gaze_indices"]:
                if isinstance(indices, (list, tuple, set, np.ndarray)):
                    unique_hit_indices.update(int(i) for i in indices)

        unique_gaze_hit_count = len(unique_hit_indices)
        unique_coverage = (
            unique_gaze_hit_count / total_points if total_points else 0.0
        )
        overlap_factor = (
            total_bbox_memberships / unique_gaze_hit_count
            if unique_gaze_hit_count
            else 0.0
        )

        grouped.append({
            "set_name": set_name,
            "slide_index": slide_index,
            "bbox_count": bbox_count,
            "attended_bboxes": attended_bboxes,
            "attended_bbox_ratio": (
                attended_bboxes / bbox_count if bbox_count else 0.0
            ),
            "total_gaze_points": total_points,
            "bbox_hit_count": total_bbox_memberships,
            "unique_gaze_hit_count": unique_gaze_hit_count,
            "coverage_by_bboxes": unique_coverage,
            "overlap_factor": overlap_factor,
            "max_attention_score": float(group["attention_score"].max()),
            "mean_attention_score": float(group["attention_score"].mean()),
        })

    return pd.DataFrame(grouped)
