# datasets_full.py
import os
import random
import math
import importlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from psychopy import visual
from PIL import Image

from tobii_pytracker.utils.custom_logger import CustomLogger


# -------------------------
# Helpers
# -------------------------
def _to_centered_bbox_from_tl(x_min_px: float, y_min_px: float, x_max_px: float, y_max_px: float,
                              area_x: int, area_y: int) -> Dict[str, float]:
    """
    Convert bbox in top-left pixel coords (image coordinates where (0,0) is top-left)
    into center-origin pixel coords where (0,0) is center of AOI and y positive is up.

    Returns dict: {"cx":..., "cy":..., "w":..., "h":...} in pixels (not normalized).
    """
    w_px = max(1.0, x_max_px - x_min_px)
    h_px = max(1.0, y_max_px - y_min_px)
    x_center_px = x_min_px + w_px / 2.0
    y_center_px = y_min_px + h_px / 2.0

    # center-origin
    cx = x_center_px - (area_x / 2.0)
    # convert y: top-left -> center-origin with positive-up
    cy = (area_y / 2.0) - y_center_px

    return {"cx": float(cx), "cy": float(cy), "w": float(w_px), "h": float(h_px)}


def _image_load_size(path: str) -> Tuple[int, int]:
    """Return (width, height) of image at path using PIL."""
    with Image.open(path) as im:
        return im.size  # width, height


# -------------------------
# Base dataset
# -------------------------
class CustomDataset:
    def __init__(self, config: Any, calculate_bboxes: bool = False):
        self.config = config
        self.dataset_path = config.get_dataset_config()["path"]
        self.logger = CustomLogger("debug", __name__).logger
        self.classes: List[str] = []
        self.data: List[Dict[str, Any]] = []
        self.calculate_bboxes = calculate_bboxes

    def prepare_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_classes(self) -> List[str]:
        return self.classes

    @property
    def is_text(self) -> bool:
        return isinstance(self, TextDataset)

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# -------------------------
# TextDataset
# -------------------------
class TextDataset(CustomDataset):
    """
    Text dataset that draws text into a PsychoPy window and returns exact per-line and per-word bboxes
    in center-origin pixel coordinates. The draw_stimulus method draws text and returns:
      {"words": [{"word":str,"conf":1.0,"bbox":{cx,cy,w,h}}...],
       "lines": [{"text":str,"conf":1.0,"bbox":{...}}...]}
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False):
        super().__init__(config, calculate_bboxes)
        self.text_cfg = self.config.get_dataset_text_config()
        self.font_height = int(self.text_cfg.get("font_height", 35))
        # fraction of AOI width used for wrap (leave small margins)
        self.wrap_frac: float = float(self.text_cfg.get("wrap_fraction", 0.95))
        self._load_data()

    def _load_data(self):
        if not self.dataset_path.endswith(".csv"):
            raise ValueError("TextDataset requires CSV file")
        df = pd.read_csv(self.dataset_path, header=0)
        label_col = self.text_cfg["label_column_name"]
        text_col = self.text_cfg["text_column_name"]
        self.classes = [str(c).lower() for c in df[label_col].unique()]
        self.classes.append("none")
        self.data = [{"class": str(r[label_col]).lower(), "data": str(r[text_col])}
                     for _, r in df.iterrows()]

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draw text into provided PsychoPy window and compute per-line & per-word bboxes.
        Returns dict with 'words' and 'lines' lists of bbox dicts.
        """
        text = str(sample["data"])
        area_x, area_y = self.config.get_area_of_interest_size()
        wrap_width = int(area_x * self.wrap_frac)

        # Draw full paragraph so participant sees exactly the rendering
        paragraph = visual.TextStim(
            win=window,
            text=text,
            pos=(0, 0),
            height=self.font_height,
            wrapWidth=wrap_width,
            alignText="left",   # we'll center the block ourselves by computing left offset
            color="white"
        )
        paragraph.draw()
        window.flip()

        # Prepare measurement: measure each word width/height using same window/font
        # We do NOT create a temporary window; we use the same 'window'.
        # Split into explicit lines first (respect \n in text)
        # We'll reflow words into lines using measured widths and wrap_width (PsychoPy's wrapWidth)
        raw_lines = text.split("\n")
        words_all = []
        for rl in raw_lines:
            for w in rl.split():
                words_all.append(w)

        # measure widths/heights
        word_dims: List[Tuple[str, float, float]] = []
        for w in words_all:
            ts = visual.TextStim(win=window, text=w, height=self.font_height, wrapWidth=None)
            w_w, w_h = ts.boundingBox
            word_dims.append((w, float(w_w), float(w_h)))

        # space width
        space_ts = visual.TextStim(win=window, text=" ", height=self.font_height)
        space_w = float(space_ts.boundingBox[0] or (self.font_height * 0.3))

        # Reflow words into lines using measured widths and wrap_width
        lines: List[List[Tuple[str, float, float]]] = []
        cur_line: List[Tuple[str, float, float]] = []
        cur_width = 0.0
        iter_words = iter(word_dims)
        for w, w_w, w_h in iter_words:
            add_w = w_w if cur_width == 0 else (space_w + w_w)
            if cur_width + add_w <= wrap_width or cur_width == 0:
                cur_line.append((w, w_w, w_h))
                cur_width = cur_width + add_w if cur_width != 0 else w_w
            else:
                lines.append(cur_line)
                cur_line = [(w, w_w, w_h)]
                cur_width = w_w
        if cur_line:
            lines.append(cur_line)

        # compute line heights and total height
        line_heights = [max((h for (_, _, h) in line), default=self.font_height) for line in lines]
        total_text_height = sum(line_heights)
        top_y = (area_y - total_text_height) / 2.0  # top coordinate (px, top-left origin)

        # compute line widths to center each line horizontally
        line_widths = []
        for line in lines:
            lw = 0.0
            for i, (_, w_w, _) in enumerate(line):
                lw += w_w
                if i < len(line) - 1:
                    lw += space_w
            line_widths.append(lw)

        # Build line and word bboxes (pixel coords top-left origin), then convert to centered coords
        words_out: List[Dict[str, Any]] = []
        lines_out: List[Dict[str, Any]] = []
        y_cursor = top_y

        for li, line in enumerate(lines):
            lw = line_widths[li]
            line_h = line_heights[li]
            left_x = (area_x - lw) / 2.0  # left x to center the line in AOI
            line_x_min = left_x
            line_x_max = left_x + lw
            line_y_min = y_cursor
            line_y_max = y_cursor + line_h

            line_bbox_centered = _to_centered_bbox_from_tl(line_x_min, line_y_min, line_x_max, line_y_max, area_x, area_y)
            lines_out.append({
                "text": " ".join([w for (w, _, _) in line]),
                "conf": 1.0,
                "bbox": line_bbox_centered
            })

            # words
            x_cursor = left_x
            for word, w_w, w_h in line:
                w_x_min = x_cursor
                w_x_max = x_cursor + w_w
                # center vertically in the line
                w_y_min = y_cursor + (line_h - w_h) / 2.0
                w_y_max = w_y_min + w_h

                word_bbox_centered = _to_centered_bbox_from_tl(w_x_min, w_y_min, w_x_max, w_y_max, area_x, area_y)
                words_out.append({
                    "word": word,
                    "conf": 1.0,
                    "bbox": word_bbox_centered
                })

                x_cursor += w_w + space_w

            y_cursor += line_h

        return {"words": words_out, "lines": lines_out}


# -------------------------
# ImageDataset
# -------------------------
class ImageDataset(CustomDataset):
    """
    Image dataset draws image to window scaled to (area_x, area_y).
    If calculate_bboxes and a model is provided (or specified in config), model.process(image_path)
    is called. Model.process must return a list of detections like:
      [(class_name, confidence, (x_min, y_min, x_max, y_max)), ...]
    where coordinates are in pixels on the input image.
    We rescale those coordinates to AOI size and convert them to center-origin pixel coords.
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False, model: Optional[Any] = None):
        super().__init__(config, calculate_bboxes)
        self.model = model
        if self.calculate_bboxes and self.model is None:
            # try load from config (same pattern as your code)
            try:
                cfg = self.config.get_model_config()
                ModelClass = getattr(importlib.import_module(f"{cfg['folder']}.{cfg['module']}"), cfg['class'])
                self.model = ModelClass(config, self)  # model must implement process(path)
            except Exception as e:
                self.logger.error(f"Could not load model from config: {e}")
                self.model = None
        self._load_data()

    def _load_data(self):
        self.classes = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        self.classes.append("none")

        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        sample = {"class": class_name, "data": os.path.join(root, f)}
                        if self.calculate_bboxes and self.model is not None:
                            sample["bboxes"] = self._compute_image_bboxes(sample["data"])
                        samples.append(sample)
        random.shuffle(samples)
        self.data = samples

    def _compute_image_bboxes(self, image_path: str) -> List[Dict[str, Any]]:
        area_x, area_y = self.config.get_area_of_interest_size()
        detections_out: List[Dict[str, Any]] = []
        try:
            preds = self.model.process(image_path)
            # get original image size
            img_w, img_h = _image_load_size(image_path)
            # scale image to AOI size: assume ImageStim uses size=(area_x,area_y)
            sx = area_x / float(img_w)
            sy = area_y / float(img_h)
            for det in preds:
                class_name, conf, box = det[0], float(det[1]), det[2]
                x_min, y_min, x_max, y_max = box
                # map model coords (image top-left origin) -> AOI top-left sized coords
                x_min_a = x_min * sx
                x_max_a = x_max * sx
                y_min_a = y_min * sy
                y_max_a = y_max * sy
                # convert to center-origin bbox
                bbox_centered = _to_centered_bbox_from_tl(x_min_a, y_min_a, x_max_a, y_max_a, area_x, area_y)
                detections_out.append({
                    "class": class_name,
                    "conf": conf,
                    "bbox": bbox_centered
                })
        except Exception as e:
            self.logger.error(f"Error computing image bboxes: {e}")
        return detections_out

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draw image to window (scaled to area_x x area_y) and return detections (if enabled).
        Returned dict contains 'image_bboxes' list (empty if not computed).
        """
        area_x, area_y = self.config.get_area_of_interest_size()
        img_path = sample["data"]
        stim = visual.ImageStim(win=window, image=img_path, size=(area_x, area_y), pos=(0, 0))
        stim.draw()
        window.flip()

        bboxes = sample.get("bboxes") if ("bboxes" in sample and sample["bboxes"] is not None) else []
        # If model exists but sample didn't precompute, compute now
        if self.calculate_bboxes and self.model is not None and not bboxes:
            bboxes = self._compute_image_bboxes(img_path)
        return {"image_bboxes": bboxes}


# -------------------------
# TimeSeriesDataset
# -------------------------
class TimeSeriesDataset(CustomDataset):
    """
    Time series dataset: CSV rows are: index, v1, v2, ..., vN, class
    draw_stimulus draws the time series as a polyline in the AOI and returns per-timestamp or per-window bboxes.
    Bboxes are center-origin pixel coords (cx,cy,w,h) where cx,cy are in pixels relative to AOI center.
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False, window_size: int = 1):
        super().__init__(config, calculate_bboxes)
        self.window_size = max(1, int(window_size))
        self._load_data()

    def _load_data(self):
        cfg = self.config.get_dataset_timeseries_config()
        file_path = cfg["path"]
        label_col = cfg["label_column_name"]
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()
        index_col = cols[0]
        ts_cols = [c for c in cols[1:] if c != label_col]
        self.classes = df[label_col].unique().tolist()
        self.classes.append("none")
        samples = []
        for _, row in df.iterrows():
            series = row[ts_cols].astype(float).to_numpy()
            samples.append({"class": row[label_col], "data": series, "index": row[index_col]})
        self.data = samples

    def _compute_timeseries_bboxes_from_series(self, series: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create bboxes per timestamp OR binned by window_size.
        Each bbox is a vertical slice centered at the x position of the timestamp(s) and spanning a fraction of AOI height.
        """
        area_x, area_y = self.config.get_area_of_interest_size()
        n = len(series)
        if n == 0:
            return []

        # map series y-values to AOI vertical pixel coordinates.
        # Normalize series to [0,1] by min/max (if constant, make it centered)
        min_v, max_v = float(np.min(series)), float(np.max(series))
        if math.isclose(min_v, max_v):
            # flat series — center it
            norm_vals = np.full_like(series, 0.5, dtype=float)
        else:
            norm_vals = (series - min_v) / (max_v - min_v)

        # x pixel positions across AOI (spread uniformly across width)
        # we choose n points from x = 0..(n-1) mapped across area_x
        xs = np.linspace(0, area_x, n, endpoint=False) + (area_x / (2 * n))  # center each bin
        bboxes_out: List[Dict[str, Any]] = []

        # produce per-window bboxes
        for start in range(0, n, self.window_size):
            end = min(start + self.window_size, n)
            xs_window = xs[start:end]
            ys_window = norm_vals[start:end]
            # compute bounding x-range in pixels for this window
            x_min_px = float(xs_window.min() - (area_x / (2 * n)))  # leftmost edge of first bin
            x_max_px = float(xs_window.max() + (area_x / (2 * n)))  # rightmost edge of last bin

            # compute a representative y range: use min and max of values in window (map to pixels)
            # map norm_vals where 0 => top? we want y positive up, but top-left y used below then converted
            # For consistency we compute pixel y (top-left origin): y_px = (1 - norm) * area_y
            y_pixels = (1.0 - ys_window) * area_y
            y_min_px = float(y_pixels.min())
            y_max_px = float(y_pixels.max())

            # Expand vertical span a bit to cover area near points (optional)
            pad_v = max(1.0, 0.02 * area_y)
            y_min_px = max(0.0, y_min_px - pad_v)
            y_max_px = min(area_y - 1.0, y_max_px + pad_v)

            # Ensure inside area
            x_min_px = max(0.0, x_min_px)
            x_max_px = min(area_x - 1.0, x_max_px)

            # convert to centered bbox
            bbox_centered = _to_centered_bbox_from_tl(x_min_px, y_min_px, x_max_px, y_max_px, area_x, area_y)
            bboxes_out.append({
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "bbox": bbox_centered
            })

        return bboxes_out

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draw time series as polyline into window. Return dict {'timeseries_bboxes': [...]}
        where each bbox has start_idx,end_idx,bbox center-origin.
        """
        series = np.asarray(sample["data"], dtype=float)
        area_x, area_y = self.config.get_area_of_interest_size()
        n = len(series)
        if n == 0:
            return {"timeseries_bboxes": []}

        # Normalize series for plotting vertically into AOI
        min_v, max_v = float(series.min()), float(series.max())
        if math.isclose(min_v, max_v):
            norm_vals = np.full(n, 0.5, dtype=float)
        else:
            norm_vals = (series - min_v) / (max_v - min_v)

        # build points in PsychoPy coordinates (center-origin)
        xs = []
        ys = []
        for i, v in enumerate(norm_vals):
            # x position: map i to [-area_x/2 .. area_x/2]
            x_px = (i + 0.5) * (area_x / n)  # top-left origin x (0..area_x)
            x_centered = x_px - (area_x / 2.0)
            # y px in top-left origin: y_px = (1 - v) * area_y
            y_px = (1.0 - v) * area_y
            y_centered = (area_y / 2.0) - y_px
            xs.append(x_centered)
            ys.append(y_centered)

        # Draw polyline using visual.ShapeStim (faster than many small Rects)
        # Build vertices as list of (x,y) pairs
        verts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        # Slight smoothing: join with line segments
        line = visual.ShapeStim(win=window, vertices=verts, closeShape=False, lineWidth=2.0, lineColor='black', fillColor=None)
        line.draw()
        window.flip()

        # compute bboxes (based on pixel positions with top-left origin)
        # we reuse the helper that expects top-left coords, so reconstruct top-left x,y in px
        # For each bin/window, compute x_min_px..x_max_px in top-left coords and y_min..y_max
        bboxes = self._compute_timeseries_bboxes_from_series(series)
        return {"timeseries_bboxes": bboxes}
