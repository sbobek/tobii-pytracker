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
        self.dataset_path = config.get_dataset_path()
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
    
    @property
    def is_time_series(self) -> bool:
        return isinstance(self, TimeSeriesDataset)
    
    @property
    def is_image(self) -> bool:
        return isinstance(self, ImageDataset)

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
        self.text_cfg = self.config.get_text_dataset_config()
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
        self.data = [{"class": str(r[label_col]).lower(), "data": str(r[text_col]), "id": str(r[text_col])[:20].replace(" ","_")}
                     for _, r in df.iterrows()]  # add short id

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

class ImageDataset(CustomDataset):
    """
    Image dataset that draws images and computes bounding boxes.

    - Does NOT accept a model directly.
    - If config contains a bbox model, it is loaded automatically.
    - If not, fallback AOI detection (grid/superpixel/saliency) is used.

    This keeps the dataset completely model-agnostic.
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False):
        super().__init__(config, calculate_bboxes)

        cfg = self.config.get_image_dataset_config()
        if cfg.get("bbox_model") is not None:
            self.calculate_bboxes=True

        self.model = None
        self.default_detector = None
        if self.calculate_bboxes:
            self.default_detector = cfg.get("bbox_model")  # grid | superpixel | saliency | custom model class
    
        # Attempt to load a custom model from config
        if self.calculate_bboxes:
            try:
                cfg = self.config.get_bbox_model_config()
                ModelClass = getattr(
                    importlib.import_module(f"{cfg['folder']}.{cfg['module']}"),
                    cfg['class']
                )
                self.model = ModelClass(config, self)
            except Exception as e:
                self.logger.warning(
                    f"No valid custom bbox model found in config; "
                    f"using fallback detector ({self.default_detector}). Error: {e}"
                )
                self.model = None

        self._load_data()

    # ------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------
    def _load_data(self):
        self.classes = [
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]
        self.classes.append("none")

        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        full_path = os.path.join(root, f)
                        sample = {"class": class_name, "data": full_path, "id": os.path.basename(full_path)}

                        if self.calculate_bboxes:
                            sample["bboxes"] = self._compute_image_bboxes(full_path)

                        samples.append(sample)

        random.shuffle(samples)
        self.data = samples

    # ------------------------------------------------------------------
    # Choose between model or fallback methods
    # ------------------------------------------------------------------
    def _compute_image_bboxes(self, image_path: str) -> List[Dict[str, Any]]:
        if self.model is not None:
            return self._compute_bboxes_with_model(image_path)
        else:
            return self._compute_bboxes_fallback(image_path)

    # ------------------------------------------------------------------
    def _compute_bboxes_with_model(self, image_path: str):
        """Handles model inference + AOI rescaling."""
        area_x, area_y = self.config.get_area_of_interest_size()
        detections_out = []

        try:
            preds = self.model.process(image_path)
            img_w, img_h = _image_load_size(image_path)
            sx, sy = area_x / img_w, area_y / img_h

            for class_name, conf, (x_min, y_min, x_max, y_max) in preds:
                detections_out.append({
                    "class": class_name,
                    "conf": float(conf),
                    "bbox": _to_centered_bbox_from_tl(
                        x_min * sx, y_min * sy,
                        x_max * sx, y_max * sy,
                        area_x, area_y
                    )
                })

        except Exception as e:
            self.logger.error(f"Error in model detection: {e}")

        return detections_out

    # ------------------------------------------------------------------
    # Fallback detection (grid, superpixel, saliency)
    # ------------------------------------------------------------------
    def _compute_bboxes_fallback(self, image_path: str):
        method = self.default_detector

        if method == "grid":
            return self._detect_grid(image_path)
        elif method == "superpixel":
            return self._detect_superpixels(image_path)
        elif method == "saliency":
            return self._detect_saliency(image_path)

        self.logger.error(f"Unknown fallback detection method '{method}'. Returning empty list.")
        return []

    # ------------------------------------------------------------------
    # GRID fallback
    # ------------------------------------------------------------------
    def _detect_grid(self, image_path: str, grid_x: int = 3, grid_y: int = 3):
        area_x, area_y = self.config.get_area_of_interest_size()
        cell_w = area_x / grid_x
        cell_h = area_y / grid_y

        detections = []
        for i in range(grid_x):
            for j in range(grid_y):
                x_min, y_min = i * cell_w, j * cell_h
                x_max, y_max = x_min + cell_w, y_min + cell_h

                detections.append({
                    "class": "grid",
                    "conf": 1.0,
                    "bbox": _to_centered_bbox_from_tl(
                        x_min, y_min, x_max, y_max,
                        area_x, area_y
                    )
                })
        return detections

    # ------------------------------------------------------------------
    # SUPERPIXEL fallback
    # ------------------------------------------------------------------
    def _detect_superpixels(self, image_path: str, n_segments: int = 50):
        from skimage.segmentation import slic
        from skimage.io import imread
        import numpy as np

        area_x, area_y = self.config.get_area_of_interest_size()
        img = imread(image_path)
        h, w = img.shape[:2]

        segments = slic(img, n_segments=n_segments, compactness=10)
        detections = []

        for seg_id in np.unique(segments):
            ys, xs = np.where(segments == seg_id)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            detections.append({
                "class": "superpixel",
                "conf": 1.0,
                "bbox": _to_centered_bbox_from_tl(
                    x_min * (area_x / w),
                    y_min * (area_y / h),
                    x_max * (area_x / w),
                    y_max * (area_y / h),
                    area_x, area_y
                )
            })

        return detections

    # ------------------------------------------------------------------
    # SALIENCY fallback
    # ------------------------------------------------------------------
    def _detect_saliency(self, image_path: str, threshold: float = 0.6):
        import cv2
        import numpy as np

        area_x, area_y = self.config.get_area_of_interest_size()

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        success, sal_map = sal.computeSaliency(img)
        sal_bin = (sal_map > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(sal_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            x_min, y_min, width, height = cv2.boundingRect(cnt)
            x_max = x_min + width
            y_max = y_min + height

            detections.append({
                "class": "saliency",
                "conf": 1.0,
                "bbox": _to_centered_bbox_from_tl(
                    x_min * (area_x / w),
                    y_min * (area_y / h),
                    x_max * (area_x / w),
                    y_max * (area_y / h),
                    area_x, area_y
                )
            })

        return detections

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        area_x, area_y = self.config.get_area_of_interest_size()
        img_path = sample["data"]

        stim = visual.ImageStim(win=window, image=img_path, size=(area_x, area_y), pos=(0, 0))
        stim.draw()
        window.flip()

        bboxes = sample.get("bboxes", [])
        if self.calculate_bboxes and not bboxes:
            bboxes = self._compute_image_bboxes(img_path)

        return {"image_bboxes": bboxes}




import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from psychopy import visual


# -------------------------
# TimeSeriesDataset
# -------------------------
class TimeSeriesDataset(CustomDataset):
    """
    Multi-channel time series dataset.

    Layout
    ------
    The AOI is divided into:
      - A left margin  (margin_left px)  : y-axis tick labels + channel name
      - A bottom margin (margin_bottom px): x-axis tick labels
      - The remaining rectangle is the **plot area** where data is drawn.

    All bounding boxes are computed exclusively within the plot area.
    Axis decorations (tick labels, channel names) are drawn outside it and
    never contribute to any bbox.

    Scaling
    -------
    All channels share the same y-axis scale so that relative magnitudes are
    preserved (important for ECG, EEG, etc.).  The global min/max across every
    channel is used to map values to pixels.  No per-channel normalisation is
    performed.

    Data format
    -----------
    CSV: first column = index, last column = label, columns in between = channels.

    Bounding boxes
    --------------
    For each channel c and each time window w:
        bbox covers the horizontal extent of the window and the vertical extent
        of the data values within that window (plus a small padding so that the
        line itself is inside the box).  The bbox is in AOI center-origin
        pixel coordinates.

    Parameters
    ----------
    config          : CustomConfig
    calculate_bboxes: bool   – compute bboxes during draw_stimulus
    window_size     : int    – number of time steps per bbox (1 = per-sample)
    margin_left     : int    – pixels reserved on the left for axis decoration
    margin_bottom   : int    – pixels reserved at the bottom for x-axis labels
    line_colors     : list   – PsychoPy color strings per channel (cycles)
    line_width      : float  – polyline width in pixels
    """

    # Default decoration margins (in pixels within the AOI)
    DEFAULT_MARGIN_LEFT   = 60
    DEFAULT_MARGIN_BOTTOM = 40

    def __init__(
        self,
        config: Any,
        calculate_bboxes: bool = False,
        window_size: int = 1,
        margin_left: int = DEFAULT_MARGIN_LEFT,
        margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
        line_colors: Optional[List[str]] = None,
        line_width: float = 2.0,
    ):
        super().__init__(config, calculate_bboxes)
        self.window_size    = max(1, int(window_size))
        self.margin_left    = int(margin_left)
        self.margin_bottom  = int(margin_bottom)
        self.line_colors    = line_colors or ["white", "blue", "red", "green", "orange", "purple"]
        self.line_width     = float(line_width)
        self.channel_names: List[str] = []
        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self):
        cfg = self.config.get_time_series_dataset_config()
        if cfg.get("bbox_model") is not None:
            self.calculate_bboxes=True
        label_col = cfg["label_column_name"]
        df = pd.read_csv(self.dataset_path)
        cols = df.columns.tolist()
        index_col = cols[0]
        ts_cols = [c for c in cols[1:] if c != label_col]
        self.channel_names = ts_cols  # preserve column names as channel labels
        self.classes = df[label_col].unique().tolist()
        self.classes.append("none")
        self.data = []
        for _, row in df.iterrows():
            # shape: (n_timepoints, n_channels) if multi-channel, else (n_timepoints,)
            series = row[ts_cols].astype(float).to_numpy()
            self.data.append({
                "class": row[label_col],
                "data": series,          # 1-D array: univariate; 2-D: multivariate
                "index": row[index_col],
                "id": f"{row[index_col]}"
            })

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------
    def _plot_area(self, area_x: int, area_y: int) -> Tuple[float, float, float, float]:
        """
        Return (plot_x_min, plot_y_min, plot_x_max, plot_y_max) in TOP-LEFT pixel
        coordinates within the AOI.

        The plot area is the rectangle where data lines are drawn.
        It excludes left and bottom decoration margins.
        """
        plot_x_min = float(self.margin_left)
        plot_y_min = 0.0                             # top of AOI  (top-left origin)
        plot_x_max = float(area_x)
        plot_y_max = float(area_y - self.margin_bottom)
        return plot_x_min, plot_y_min, plot_x_max, plot_y_max

    def _ensure_2d(self, series: np.ndarray) -> np.ndarray:
        """
        Guarantee series is 2-D: shape (n_timepoints, n_channels).
        """
        if series.ndim == 1:
            return series[:, np.newaxis]
        return series

    def _global_scale(self, series2d: np.ndarray) -> Tuple[float, float]:
        """
        Return (global_min, global_max) across ALL channels so that relative
        magnitudes are preserved when mapping to pixels.
        """
        g_min = float(np.nanmin(series2d))
        g_max = float(np.nanmax(series2d))
        if math.isclose(g_min, g_max):
            g_min -= 0.5
            g_max += 0.5
        return g_min, g_max

    def _value_to_y_tl(
        self,
        value: float,
        g_min: float,
        g_max: float,
        ch_idx: int,
        n_channels: int,
        plot_y_min: float,
        plot_y_max: float,
    ) -> float:
        """
        Map a data value to a top-left pixel y coordinate within the channel's
        horizontal band.

        Each channel gets an equal vertical band inside the plot area.
        Within its band the value is scaled linearly (g_min → bottom,
        g_max → top of band), preserving the global scale.
        """
        plot_h = plot_y_max - plot_y_min          # total plot height in px
        band_h = plot_h / n_channels              # height of each channel band
        # channel 0 occupies the top band, channel n-1 the bottom
        band_top    = plot_y_min + ch_idx * band_h
        band_bottom = band_top + band_h

        # map value: g_max → band_top (small y = top), g_min → band_bottom
        t = (value - g_min) / (g_max - g_min)     # 0..1, higher value = higher on screen
        y_tl = band_bottom - t * band_h            # invert: t=1 → band_top, t=0 → band_bottom
        return float(np.clip(y_tl, band_top, band_bottom))

    def _x_positions_tl(
        self,
        n: int,
        plot_x_min: float,
        plot_x_max: float,
    ) -> np.ndarray:
        """
        Return the centre x position (top-left pixel) for each of the n time
        steps, uniformly distributed across the plot area width.
        """
        plot_w = plot_x_max - plot_x_min
        return plot_x_min + (np.arange(n) + 0.5) * (plot_w / n)

    def _tl_to_psychopy(
        self,
        x_tl: float,
        y_tl: float,
        area_x: int,
        area_y: int,
    ) -> Tuple[float, float]:
        """Convert top-left pixel coords to PsychoPy center-origin coords."""
        return x_tl - area_x / 2.0, area_y / 2.0 - y_tl

    # ------------------------------------------------------------------
    # Bbox computation
    # ------------------------------------------------------------------
    def _compute_timeseries_bboxes_from_series(
        self,
        series2d: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Compute per-channel, per-window bounding boxes.

        Each bbox tightly wraps the data values in that window for that channel,
        restricted to the channel's vertical band within the plot area.
        Axis decoration margins are excluded.

        Returns a flat list of dicts:
          {
            "channel_idx": int,
            "channel_name": str,
            "start_idx": int,
            "end_idx": int,
            "bbox": {"cx", "cy", "w", "h"}   # AOI center-origin pixels
          }
        """
        area_x, area_y = self.config.get_area_of_interest_size()
        plot_x_min, plot_y_min, plot_x_max, plot_y_max = self._plot_area(area_x, area_y)
        series2d = self._ensure_2d(series2d)
        n, n_ch = series2d.shape
        if n == 0:
            return []

        g_min, g_max = self._global_scale(series2d)
        xs = self._x_positions_tl(n, plot_x_min, plot_x_max)
        bin_half_w = (plot_x_max - plot_x_min) / (2 * n)

        bboxes_out: List[Dict[str, Any]] = []

        for ch in range(n_ch):
            ch_vals = series2d[:, ch]
            band_h = (plot_y_max - plot_y_min) / n_ch
            band_top    = plot_y_min + ch * band_h
            band_bottom = band_top + band_h

            # small vertical padding so the drawn line sits inside the bbox
            pad_v = max(2.0, 0.01 * band_h)

            for start in range(0, n, self.window_size):
                end = min(start + self.window_size, n)

                # x extent: edges of the outermost bins in the window
                x_min_px = float(xs[start])   - bin_half_w
                x_max_px = float(xs[end - 1]) + bin_half_w
                x_min_px = max(plot_x_min, x_min_px)
                x_max_px = min(plot_x_max, x_max_px)

                # y extent: actual data range mapped to top-left pixels
                window_vals = ch_vals[start:end]
                y_tl_values = np.array([
                    self._value_to_y_tl(v, g_min, g_max, ch, n_ch, plot_y_min, plot_y_max)
                    for v in window_vals
                ])
                y_min_px = float(y_tl_values.min()) - pad_v  # smaller y = higher on screen
                y_max_px = float(y_tl_values.max()) + pad_v
                # clamp to the channel's band (never bleed into neighbour bands)
                y_min_px = max(band_top,    y_min_px)
                y_max_px = min(band_bottom, y_max_px)

                ch_name = (
                    self.channel_names[ch]
                    if ch < len(self.channel_names)
                    else f"ch{ch}"
                )
                bboxes_out.append({
                    "channel_idx":  ch,
                    "channel_name": ch_name,
                    "start_idx":    int(start),
                    "end_idx":      int(end - 1),
                    "bbox": _to_centered_bbox_from_tl(
                        x_min_px, y_min_px, x_max_px, y_max_px, area_x, area_y
                    ),
                })

        return bboxes_out

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_axes(
        self,
        window: "visual.Window",
        series2d: np.ndarray,
        area_x: int,
        area_y: int,
        n_channels: int,
        g_min: float,
        g_max: float,
        n_timepoints: int,
    ) -> None:
        """
        Draw axis decorations (tick marks, scale labels, channel names) entirely
        within the margin regions so they never overlap the plot area bboxes.
        """
        plot_x_min, plot_y_min, plot_x_max, plot_y_max = self._plot_area(area_x, area_y)
        font_h = max(10, min(16, self.margin_left // 5))   # auto font size

        # ---- y-axis: draw two tick labels per channel (min & max of global scale)
        for ch in range(n_channels):
            band_h      = (plot_y_max - plot_y_min) / n_channels
            band_top    = plot_y_min + ch * band_h
            band_bottom = band_top + band_h
            band_mid    = (band_top + band_bottom) / 2.0

            # tick marks at top and bottom of band
            for y_tl, label_val in [(band_top + font_h * 0.5, g_max),
                                    (band_bottom - font_h * 0.5, g_min)]:
                x_c, y_c = self._tl_to_psychopy(
                    self.margin_left / 2.0, y_tl, area_x, area_y
                )
                visual.TextStim(
                    win=window, text=f"{label_val:.2g}",
                    pos=(x_c, y_c), height=font_h,
                    color="white", anchorHoriz="center", anchorVert="center"
                ).draw()

            # channel name centred in the left margin, vertically centred in band
            ch_name = (
                self.channel_names[ch]
                if ch < len(self.channel_names) else f"ch{ch}"
            )
            x_c, y_c = self._tl_to_psychopy(
                self.margin_left / 2.0, band_mid, area_x, area_y
            )
            visual.TextStim(
                win=window, text=ch_name,
                pos=(x_c, y_c), height=font_h,
                color="yellow", anchorHoriz="center", anchorVert="center"
            ).draw()

        # ---- x-axis: a few evenly-spaced index labels at the bottom
        n_xticks = min(10, n_timepoints)
        tick_indices = np.linspace(0, n_timepoints - 1, n_xticks, dtype=int)
        xs_tl = self._x_positions_tl(n_timepoints, plot_x_min, plot_x_max)
        y_label_tl = area_y - self.margin_bottom / 2.0
        for ti in tick_indices:
            x_c, y_c = self._tl_to_psychopy(xs_tl[ti], y_label_tl, area_x, area_y)
            visual.TextStim(
                win=window, text=str(ti),
                pos=(x_c, y_c), height=font_h,
                color="white", anchorHoriz="center", anchorVert="center"
            ).draw()

        # ---- axis border lines (thin grey lines separating channels)
        for ch in range(n_channels):
            band_h   = (plot_y_max - plot_y_min) / n_channels
            sep_y_tl = plot_y_min + (ch + 1) * band_h   # bottom edge of band
            if sep_y_tl >= plot_y_max:
                continue
            x_left_c,  y_sep_c = self._tl_to_psychopy(plot_x_min, sep_y_tl, area_x, area_y)
            x_right_c, _       = self._tl_to_psychopy(plot_x_max, sep_y_tl, area_x, area_y)
            visual.Line(
                win=window,
                start=(x_left_c, y_sep_c), end=(x_right_c, y_sep_c),
                lineWidth=1.0, lineColor="grey"
            ).draw()

    def draw_stimulus(
        self,
        window: "visual.Window",
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Draw all channels as polylines and optional axis decorations.

        Returns
        -------
        dict with key ``"timeseries_bboxes"``: list of per-channel per-window
        bbox dicts (only populated when ``calculate_bboxes=True``).
        """
        raw = np.asarray(sample["data"], dtype=float)
        series2d = self._ensure_2d(raw)
        n, n_ch = series2d.shape

        area_x, area_y = self.config.get_area_of_interest_size()
        plot_x_min, plot_y_min, plot_x_max, plot_y_max = self._plot_area(area_x, area_y)

        if n == 0:
            window.flip()
            return {"timeseries_bboxes": []}

        g_min, g_max = self._global_scale(series2d)
        xs_tl = self._x_positions_tl(n, plot_x_min, plot_x_max)

        # ---- draw each channel as a polyline
        for ch in range(n_ch):
            color = self.line_colors[ch % len(self.line_colors)]
            verts = []
            for i in range(n):
                y_tl = self._value_to_y_tl(
                    series2d[i, ch], g_min, g_max,
                    ch, n_ch, plot_y_min, plot_y_max
                )
                x_c, y_c = self._tl_to_psychopy(xs_tl[i], y_tl, area_x, area_y)
                verts.append((x_c, y_c))

            visual.ShapeStim(
                win=window, vertices=verts,
                closeShape=False, lineWidth=self.line_width,
                lineColor=color, fillColor=None
            ).draw()

        # ---- axis decorations (outside the plot area — no bboxes touch these)
        self._draw_axes(window, series2d, area_x, area_y, n_ch, g_min, g_max, n)

        window.flip()

        # ---- bboxes (only when requested)
        bboxes: List[Dict[str, Any]] = []
        if self.calculate_bboxes:
            bboxes = self._compute_timeseries_bboxes_from_series(series2d)

        return {"timeseries_bboxes": bboxes}