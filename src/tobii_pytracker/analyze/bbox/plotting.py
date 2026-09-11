from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

from .geometry import polygon_to_plot_coords, polygon_vertices


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


def plot_bbox_attention(
    scored_bboxes: pd.DataFrame,
    gaze_data: pd.DataFrame,
    screenshot_path: Path,
    *,
    set_name: Optional[str] = None,
    slide_index: Optional[int] = None,
    title: Optional[str] = None,
    top_k: Optional[int] = 20,
    min_hits: int = 1,
    show_gaze: bool = True,
    show: bool = True,
    save_path: Optional[Path] = None,
    filter_set_and_slide: Callable[[pd.DataFrame, Optional[Any], Optional[Any]], pd.DataFrame] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    screenshot_path = Path(screenshot_path)

    if not screenshot_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

    filter_set_and_slide = filter_set_and_slide or _default_filter_set_and_slide

    boxes = scored_bboxes.copy()
    slide_gaze = gaze_data.copy()

    boxes = filter_set_and_slide(
        boxes,
        set_name=set_name,
        slide_index=slide_index,
    )
    slide_gaze = filter_set_and_slide(
        slide_gaze,
        set_name=set_name,
        slide_index=slide_index,
    )

    if min_hits is not None:
        boxes = boxes[boxes["hit_count"] >= min_hits]

    if boxes.empty:
        raise ValueError("No attended bboxes to plot for the given filters.")

    if top_k is not None:
        boxes = boxes.sort_values("attention_score", ascending=False).head(top_k)

    img = mpimg.imread(screenshot_path)
    height, width = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, origin="upper")

    if show_gaze and not slide_gaze.empty:
        gaze_x = width / 2.0 + slide_gaze["avg_gaze_x"].astype(float)
        gaze_y = height / 2.0 - slide_gaze["avg_gaze_y"].astype(float)
        ax.scatter(
            gaze_x,
            gaze_y,
            s=40,
            alpha=0.8,
            c="#FF1493",
            edgecolors="white",
            linewidths=1.5,
            label="gaze samples",
            zorder=10,
        )

    max_score = float(boxes["attention_score"].max())
    border_color = "white"
    border_linewidth = 2.5
    border_halo_linewidth = 5.5
    fill_color = "#00E5FF"

    for _, row in boxes.iterrows():
        score = float(row["attention_score"])
        intensity = score / max_score if max_score > 0 else 0.0
        polygon = polygon_vertices(row.get("polygon"))
        if polygon is None:
            polygon = polygon_vertices(row.get("bbox"))

        if polygon is not None:
            fill_rgba = mcolors.to_rgba(fill_color, alpha=0.08 + 0.22 * intensity)
            polygon_xy = polygon_to_plot_coords(polygon, width, height)
            patch = patches.Polygon(
                polygon_xy,
                closed=True,
                linewidth=border_linewidth,
                edgecolor=border_color,
                facecolor=fill_rgba,
                zorder=8,
            )
            patch.set_path_effects([
                pe.Stroke(linewidth=border_halo_linewidth, foreground="black"),
                pe.Normal(),
            ])
            ax.add_patch(patch)
            label_x, label_y = polygon_xy.mean(axis=0)
        else:
            fill_rgba = mcolors.to_rgba(fill_color, alpha=0.08 + 0.22 * intensity)
            cx = float(row["cx"])
            cy = float(row["cy"])
            bbox_width = float(row["w"])
            bbox_height = float(row["h"])
            x_min = width / 2.0 + (cx - bbox_width / 2.0)
            y_min = height / 2.0 - (cy + bbox_height / 2.0)

            rectangle = patches.Rectangle(
                (x_min, y_min),
                bbox_width,
                bbox_height,
                linewidth=border_linewidth,
                edgecolor=border_color,
                facecolor=fill_rgba,
                zorder=8,
            )
            rectangle.set_path_effects([
                pe.Stroke(linewidth=border_halo_linewidth, foreground="black"),
                pe.Normal(),
            ])
            ax.add_patch(rectangle)
            label_x, label_y = x_min, y_min

        ax.text(
            label_x + 3,
            label_y + 14,
            f"{int(row['hit_count'])}",
            color="white",
            fontsize=10,
            fontweight="bold",
            zorder=11,
            bbox=dict(
                facecolor="black",
                edgecolor="white",
                alpha=1.0,
                pad=2,
            ),
        )

    set_label = set_name if set_name is not None else "All sets"
    slide_label = slide_index if slide_index is not None else "All slides"
    ax.set_title(title or f"BBox attention — {set_label}, slide {slide_label}")
    ax.axis("off")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
