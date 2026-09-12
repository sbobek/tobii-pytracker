import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any, Dict, List, Literal
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .data_loader import DataLoader
from scipy.ndimage import gaussian_filter


# ======================================================
# BASE ANALYZER
# ======================================================
class BaseAnalyzer:
    """
    Base class for all analyzers.

    Each analyzer implements:
      - analyze(data, per='slide'|'set'|'global', subset=None)
      - plot_analysis(...)

    They do not rely on DataLoader but expect clean, preformatted data:
      columns like avg_gaze_x, avg_gaze_y, input_data, slide_index, etc.
    """

    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results: Optional[pd.DataFrame] = None

    def analyze(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def plot_analysis(self, *args, **kwargs):
        raise NotImplementedError

    def save_results(self, filename: Optional[str] = None):
        if self.results is None:
            return
        filename = filename or f"{self.__class__.__name__}_results.json"
        filepath = self.output_folder / filename
        self.results.to_json(filepath, orient="records", indent=4, force_ascii=False)

    @staticmethod
    def _normalize_slide_index_column(
        data: pd.DataFrame,
        column: str = "slide_index",
    ) -> pd.DataFrame:
        normalized = data.copy()
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        ).astype("Int64")
        return normalized

    @staticmethod
    def _filter_set_and_slide(
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

    @staticmethod
    def _resolve_gaze_columns(
        use_fixations: bool,
    ) -> tuple[str, str, Optional[str]]:
        if use_fixations:
            return "x_mean", "y_mean", "duration"
        return "avg_gaze_x", "avg_gaze_y", None


# ---------------------------------------------------------------------
# ------------------------- ANALYZERS --------------------------------
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Optional


class HeatmapAnalyzer:
    """
    Generates gaze heatmaps and overlays them over screenshots.

    This analyzer is independent of DataLoader — it operates directly
    on pandas DataFrames and image paths.

    Parameters
    ----------
    output_folder : Path
        Directory where results or plots can be saved.

    Notes
    -----
    The DataFrame must contain:
        - 'avg_gaze_x' : x gaze coordinate (centered)
        - 'avg_gaze_y' : y gaze coordinate (centered)
        - 'set_name'   : subject or participant ID
        - 'slide_index': slide identifier (int)
        - 'input_data' : image identifier (str)
    """

    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(
            self,
            background_data: pd.DataFrame,
            per: str = "global",
    ) -> pd.DataFrame:
        """
        Perform gaze heatmap analysis by aggregating gaze data per group.

        Parameters
        ----------
        background_data : pd.DataFrame
            Combined gaze data from one or more subjects.
        per : str, optional
            Aggregation mode:
            - 'global' → compute statistics across all data (no grouping)
            - 'set'    → compute per subject
            - 'slide'  → compute per subject and slide

        Returns
        -------
        pd.DataFrame
            DataFrame with mean gaze positions and counts per group.
        """
        if per not in ["global", "set", "slide"]:
            raise ValueError("Parameter 'per' must be one of: ['global', 'set', 'slide'].")

        if per == "global":
            # Treat the entire dataset as a single group
            results = pd.DataFrame([{
                "avg_gaze_x": background_data["avg_gaze_x"].mean(),
                "avg_gaze_y": background_data["avg_gaze_y"].mean(),
                "gaze_count": background_data["avg_gaze_x"].count(),
            }])
        elif per == "set":
            results = (
                background_data.groupby("set_name")
                .agg(
                    avg_gaze_x=("avg_gaze_x", "mean"),
                    avg_gaze_y=("avg_gaze_y", "mean"),
                    gaze_count=("avg_gaze_x", "count"),
                )
                .reset_index()
            )
        else:  # per == "slide"
            results = (
                background_data.groupby(["set_name", "slide_index"])
                .agg(
                    avg_gaze_x=("avg_gaze_x", "mean"),
                    avg_gaze_y=("avg_gaze_y", "mean"),
                    gaze_count=("avg_gaze_x", "count"),
                )
                .reset_index()
            )

        self.results = results
        return results

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
            self,
            background_data: pd.DataFrame,
            screenshot_path: Path,
            title: Optional[str] = None,
            flip_y: bool = True,
            blur_sigma: float = 3.0,
            bins: int = 100,
            cmap: str = "hot",
            alpha: float = 0.6,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Plot gaze heatmap overlayed over the given screenshot.

        Parameters
        ----------
        background_data : pd.DataFrame
            The gaze data to visualize (usually a subset).
        screenshot_path : Path
            Path to the screenshot to overlay.
        title : str, optional
            Custom title for the plot.
        flip_y : bool, optional
            Whether to invert Y axis to align with screen coordinates.
        blur_sigma : float, optional
            Gaussian smoothing factor for the heatmap.
        bins : int, optional
            Number of bins for the 2D histogram.
        cmap : str, optional
            Colormap for the heatmap overlay.
        alpha : float, optional
            Transparency level for the heatmap overlay.
        show : bool, optional
            Whether to display the figure interactively.
        save_path : Path, optional
            If provided, saves the figure to this location.
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        # --- Load screenshot ---
        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        # --- Prepare gaze coordinates ---
        avg_x = W / 2 + background_data["avg_gaze_x"].dropna().values
        avg_y = H / 2 - background_data["avg_gaze_y"].dropna().values if flip_y else H / 2 + background_data[
            "avg_gaze_y"].dropna().values

        # --- Compute heatmap ---
        heatmap, _, _ = np.histogram2d(avg_x, avg_y, bins=bins, range=[[0, W], [0, H]])
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")
        ax.imshow(
            heatmap.T,
            cmap=cmap,
            alpha=alpha,
            origin="upper",
            extent=[0, W, H, 0],
        )

        ax.set_title(title or f"Heatmap ({len(background_data)} samples)")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


class FocusMapAnalyzer:
    """
    Focus Map Analyzer — shows areas NOT looked at (inverted heatmap).

    API mirrors HeatmapAnalyzer:
      - analyze(background_data: pd.DataFrame, per: str = "global") -> pd.DataFrame
      - plot_analysis(background_data: pd.DataFrame, screenshot_path: Path, title: Optional[str]=None, ...)
        <- signature matches HeatmapAnalyzer.plot_analysis exactly.

    Required columns in background_data:
      - 'avg_gaze_x', 'avg_gaze_y' (centered coords)
      - 'set_name', 'slide_index', 'input_data' depending on per mode
    """

    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(
            self,
            background_data: pd.DataFrame,
            per: str = "global",
    ) -> pd.DataFrame:
        """
        Compute summary stats similarly to HeatmapAnalyzer but for API consistency.

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze data.
        per : str
            'global' | 'set' | 'slide' (same semantics as HeatmapAnalyzer)

        Returns
        -------
        pd.DataFrame
            Summary with columns ['avg_gaze_x','avg_gaze_y','gaze_count'] and grouping keys when appropriate.
        """
        if per not in ["global", "set", "slide"]:
            raise ValueError("Parameter 'per' must be one of: ['global','set','slide'].")

        if per == "global":
            results = pd.DataFrame([{
                "avg_gaze_x": background_data["avg_gaze_x"].mean(),
                "avg_gaze_y": background_data["avg_gaze_y"].mean(),
                "gaze_count": background_data["avg_gaze_x"].count(),
            }])
        elif per == "set":
            results = (
                background_data.groupby("set_name")
                .agg(
                    avg_gaze_x=("avg_gaze_x", "mean"),
                    avg_gaze_y=("avg_gaze_y", "mean"),
                    gaze_count=("avg_gaze_x", "count"),
                )
                .reset_index()
            )
        else:  # per == "slide"
            results = (
                background_data.groupby(["set_name", "slide_index"])
                .agg(
                    avg_gaze_x=("avg_gaze_x", "mean"),
                    avg_gaze_y=("avg_gaze_y", "mean"),
                    gaze_count=("avg_gaze_x", "count"),
                )
                .reset_index()
            )

        self.results = results
        return results

    # ======================================================
    # VISUALIZATION (signature matches HeatmapAnalyzer.plot_analysis)
    # ======================================================
    def plot_analysis(
            self,
            background_data: pd.DataFrame,
            screenshot_path: Path,
            title: Optional[str] = None,
            flip_y: bool = True,
            blur_sigma: float = 3.0,
            bins: int = 100,
            cmap: str = "hot",
            alpha: float = 0.6,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Plot focus map (inverted heatmap) overlayed over the screenshot.
        Only the "hot" parts (unseen areas) are visible — transparent elsewhere.
        Signature matches HeatmapAnalyzer.plot_analysis.
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        df = background_data.copy()
        if df.empty:
            raise ValueError("background_data is empty — nothing to visualize.")

        # --- Load image ---
        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        # --- Convert gaze coordinates ---
        xs = W / 2 + df["avg_gaze_x"].dropna().values
        ys = df["avg_gaze_y"].dropna().values
        ys = (H / 2 - ys) if flip_y else (H / 2 + ys)

        # --- Compute gaze density ---
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0, W], [0, H]])
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        # --- Normalize and invert to get focus mask (1 = unseen, 0 = looked) ---
        max_val = heatmap.max() if heatmap.size and heatmap.max() > 0 else 1.0
        norm = heatmap / max_val
        focus_mask = 1.0 - norm

        # --- Build transparent colormap ---
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(focus_mask.T)  # shape (bins,bins,4)
        rgba[..., -1] = focus_mask.T * alpha  # scale transparency by intensity

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")
        ax.imshow(
            rgba,
            origin="upper",
            extent=[0, W, H, 0],
            interpolation="bilinear",
        )
        ax.set_title(title or f"Focus Map (transparent hot overlay) — {len(df)} samples")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


class SaccadeAnalyzer:
    """
    Calculates saccade metrics (dx, dy, dt, amplitude, velocity, acceleration) from gaze data.

    This version supports multiple parameterizations:
      - Velocity-based or acceleration-based saccade detection.
      - Optional micro-saccade filtering.

    Parameters
    ----------
    output_folder : Path
        Directory where results or plots can be saved.
    method : {'ivt', 'acceleration'}, optional
        Saccade detection algorithm:
            - 'ivt' → Velocity-threshold (I-VT)
            - 'acceleration' → Acceleration-threshold
        Default = 'ivt'.
    velocity_threshold : float, optional
        Minimum velocity (pixels/second) to classify as a saccade (I-VT only). Default = 100.
    acceleration_threshold : float, optional
        Minimum acceleration (pixels/second²) to classify as a saccade (acceleration only). Default = 5000.
    min_duration : float, optional
        Minimum duration (seconds) to consider a saccade valid. Default = 0.01.
    filter_micro_saccades : bool, optional
        Whether to remove micro-saccades (small-amplitude movements). Default = False.
    micro_saccade_threshold : float, optional
        Amplitude threshold (in pixels) below which a saccade is considered a micro-saccade and removed.
        Default = 30 pixels (roughly ~1° visual angle at 60 cm and 1080p resolution).
    """

    def __init__(
            self,
            output_folder: Path,
            method: Literal["ivt", "acceleration"] = "ivt",
            velocity_threshold: float = 100.0,
            acceleration_threshold: float = 5000.0,
            min_duration: float = 0.01,
            filter_micro_saccades: bool = False,
            micro_saccade_threshold: float = 30.0,
    ):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.method = method
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_duration = min_duration
        self.filter_micro_saccades = filter_micro_saccades
        self.micro_saccade_threshold = micro_saccade_threshold

        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(self, background_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute saccade events and metrics.

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze data with columns:
            ['avg_gaze_x', 'avg_gaze_y', 'system_time', 'set_name', 'slide_index']

        Returns
        -------
        pd.DataFrame
            DataFrame of saccade events with columns:
            ['set_name','slide_index','start_time','end_time','duration',
             'x_start','y_start','x_end','y_end','amplitude','peak_velocity',
             'mean_velocity','mean_acceleration']
        """
        df = background_data.copy()
        events_list = []

        for (set_name, slide_index), group in df.groupby(["set_name", "slide_index"]):
            g = group.sort_values("system_time").reset_index(drop=True)
            if len(g) < 2:
                continue

            g["x_prev"] = g["avg_gaze_x"].shift(1)
            g["y_prev"] = g["avg_gaze_y"].shift(1)
            g["t_prev"] = g["system_time"].shift(1)
            g["dx"] = g["avg_gaze_x"] - g["x_prev"]
            g["dy"] = g["avg_gaze_y"] - g["y_prev"]
            g["dt"] = g["system_time"] - g["t_prev"]
            g = g.dropna(subset=["x_prev", "y_prev", "t_prev", "dt"])
            g["dt"] = g["dt"].replace(0, np.nan)

            g["amplitude"] = np.sqrt(g["dx"] ** 2 + g["dy"] ** 2)
            g["velocity"] = g["amplitude"] / g["dt"]
            g["acceleration"] = g["velocity"].diff() / g["dt"]

            # Saccade detection rule
            if self.method == "ivt":
                mask = g["velocity"] > self.velocity_threshold
            elif self.method == "acceleration":
                mask = np.abs(g["acceleration"]) > self.acceleration_threshold
            else:
                raise ValueError(f"Unknown saccade detection method: {self.method}")

            g["is_saccade"] = mask

            # Detect contiguous saccade segments
            saccades = []
            start_idx = None
            for i, row in g.iterrows():
                if row["is_saccade"] and start_idx is None:
                    start_idx = i
                elif not row["is_saccade"] and start_idx is not None:
                    end_idx = i - 1
                    saccades.append((start_idx, end_idx))
                    start_idx = None
            if start_idx is not None:
                saccades.append((start_idx, len(g) - 1))

            # Aggregate per saccade
            for start_idx, end_idx in saccades:
                seg = g.iloc[start_idx:end_idx + 1]
                duration = seg["dt"].sum()
                if duration < self.min_duration:
                    continue

                amp = np.sqrt(
                    (seg["avg_gaze_x"].iloc[-1] - seg["x_prev"].iloc[0]) ** 2 +
                    (seg["avg_gaze_y"].iloc[-1] - seg["y_prev"].iloc[0]) ** 2
                )

                # Optional micro-saccade filtering
                if self.filter_micro_saccades and amp < self.micro_saccade_threshold:
                    continue

                events_list.append({
                    "set_name": set_name,
                    "slide_index": slide_index,
                    "start_time": seg["system_time"].iloc[0],
                    "end_time": seg["system_time"].iloc[-1],
                    "duration": duration,
                    "x_start": seg["x_prev"].iloc[0],
                    "y_start": seg["y_prev"].iloc[0],
                    "x_end": seg["avg_gaze_x"].iloc[-1],
                    "y_end": seg["avg_gaze_y"].iloc[-1],
                    "amplitude": amp,
                    "peak_velocity": seg["velocity"].max(),
                    "mean_velocity": seg["velocity"].mean(),
                    "mean_acceleration": seg["acceleration"].abs().mean(),
                })

        if events_list:
            events_df = pd.DataFrame(events_list)
        else:
            events_df = pd.DataFrame(columns=[
                "set_name", "slide_index", "start_time", "end_time", "duration",
                "x_start", "y_start", "x_end", "y_end", "amplitude",
                "peak_velocity", "mean_velocity", "mean_acceleration"
            ])

        self.results = events_df
        return events_df

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
            self,
            saccades: pd.DataFrame,
            screenshot_path: Path,
            set_name: Optional[str] = None,
            slide_index: Optional[int] = None,
            title: Optional[str] = None,
            flip_y: bool = True,
            color: str = "cyan",
            alpha: float = 0.8,
            linewidth: float = 2.0,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Overlay saccades on top of the screenshot.

        Parameters
        ----------
        saccades : pd.DataFrame
            Output of analyze(). Can contain all sets/slides.
        screenshot_path : Path
            Path to the corresponding screenshot.
        set_name : str, optional
            Filter by participant ID.
        slide_index : int, optional
            Filter by slide index.
        title : str, optional
            Custom title.
        flip_y : bool, optional
            Whether to invert Y axis.
        color : str, optional
            Line color for saccades.
        alpha : float, optional
            Transparency level.
        linewidth : float, optional
            Line thickness.
        show : bool, optional
        save_path : Path, optional
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        df = saccades.copy()
        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None:
            df = df[df["slide_index"] == slide_index]

        if df.empty:
            raise ValueError("No saccade data to plot for the given filters.")

        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")

        for _, row in df.iterrows():
            x0 = W / 2 + row["x_start"]
            x1 = W / 2 + row["x_end"]
            y0 = H / 2 - row["y_start"] if flip_y else H / 2 + row["y_start"]
            y1 = H / 2 - row["y_end"] if flip_y else H / 2 + row["y_end"]
            ax.arrow(
                x0, y0,
                x1 - x0, y1 - y0,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                head_width=10,
                length_includes_head=True
            )

        ax.set_title(title or f"Saccades — {set_name or 'All'}, Slide {slide_index or '?'}")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


class FixationAnalyzer:
    """
    Detects and visualizes gaze fixations from time-series gaze data.

    The analyzer supports simple dispersion-based fixation detection and
    outputs per-fixation metrics (centroid, duration, dispersion, etc.).
    Always processes data *per slide*.

    Parameters
    ----------
    output_folder : Path
        Directory where results or plots can be saved.
    method : {'dispersion', 'velocity'}, optional
        Fixation detection method. Default = 'dispersion'.
    dispersion_threshold : float, optional
        Maximum visual angle (in pixels or units of your coordinate system)
        for gaze points to be considered within a fixation (used in 'dispersion' method).
        Default = 50.
    min_duration : float, optional
        Minimum duration (in seconds) for a fixation to be valid. Default = 0.1.
    velocity_threshold : float, optional
        Maximum velocity (pixels/sec) to consider samples as belonging to a fixation
        when using the 'velocity' method. Default = 100.
    """

    def __init__(
            self,
            output_folder: Path,
            method: Literal["dispersion", "velocity"] = "dispersion",
            dispersion_threshold: float = 50.0,
            min_duration: float = 0.1,
            velocity_threshold: float = 100.0,
    ):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.method = method
        self.dispersion_threshold = dispersion_threshold
        self.min_duration = min_duration
        self.velocity_threshold = velocity_threshold

        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # FIXATION DETECTION
    # ======================================================
    def analyze(self, background_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect fixations per slide.

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze data with columns:
            ['avg_gaze_x', 'avg_gaze_y', 'system_time', 'set_name', 'slide_index']

        Returns
        -------
        pd.DataFrame
            Fixations with columns:
            ['set_name', 'slide_index', 'fix_start', 'fix_end', 'duration',
             'x_mean', 'y_mean', 'dispersion']
        """
        df = background_data.copy()
        fixations_list = []

        for (set_name, slide_index), group in df.groupby(["set_name", "slide_index"]):
            g = group.sort_values("system_time").reset_index(drop=True)

            if self.method == "dispersion":
                fixs = self._detect_fixations_dispersion(g)
            elif self.method == "velocity":
                fixs = self._detect_fixations_velocity(g)
            else:
                raise ValueError(f"Unknown fixation detection method: {self.method}")

            if fixs.empty:
                continue

            fixs["set_name"] = set_name
            fixs["slide_index"] = slide_index
            fixations_list.append(fixs)

        if fixations_list:
            fixations_df = pd.concat(fixations_list, ignore_index=True)
        else:
            fixations_df = pd.DataFrame(
                columns=[
                    "set_name", "slide_index", "fix_start", "fix_end",
                    "duration", "x_mean", "y_mean", "dispersion"
                ]
            )

        self.results = fixations_df
        return fixations_df

    # ======================================================
    # DISPERSION METHOD
    # ======================================================
    def _detect_fixations_dispersion(self, g: pd.DataFrame) -> pd.DataFrame:
        """Simple I-DT (dispersion-threshold) fixation detection."""
        fixations = []
        start_idx = 0
        while start_idx < len(g):
            end_idx = start_idx + 1
            while end_idx < len(g):
                window = g.iloc[start_idx:end_idx]
                dispersion = (window["avg_gaze_x"].max() - window["avg_gaze_x"].min()) + \
                             (window["avg_gaze_y"].max() - window["avg_gaze_y"].min())
                if dispersion > self.dispersion_threshold:
                    break
                end_idx += 1

            window = g.iloc[start_idx:end_idx]
            duration = window["system_time"].iloc[-1] - window["system_time"].iloc[0]
            if duration >= self.min_duration:
                fixations.append({
                    "fix_start": window["system_time"].iloc[0],
                    "fix_end": window["system_time"].iloc[-1],
                    "duration": duration,
                    "x_mean": window["avg_gaze_x"].mean(),
                    "y_mean": window["avg_gaze_y"].mean(),
                    "dispersion": (window["avg_gaze_x"].max() - window["avg_gaze_x"].min()) +
                                  (window["avg_gaze_y"].max() - window["avg_gaze_y"].min())
                })
            start_idx = end_idx

        return pd.DataFrame(fixations)

    # ======================================================
    # VELOCITY METHOD
    # ======================================================
    def _detect_fixations_velocity(self, g: pd.DataFrame) -> pd.DataFrame:
        """Velocity-threshold fixation detection."""
        g = g.sort_values("system_time").reset_index(drop=True)
        g["dx"] = g["avg_gaze_x"].diff()
        g["dy"] = g["avg_gaze_y"].diff()
        g["dt"] = g["system_time"].diff()
        g["velocity"] = np.sqrt(g["dx"] ** 2 + g["dy"] ** 2) / g["dt"]
        g["is_fix"] = g["velocity"] < self.velocity_threshold

        fixations = []
        current_fix = []
        for i, row in g.iterrows():
            if row["is_fix"]:
                current_fix.append(row)
            else:
                if current_fix:
                    fixations.append(current_fix)
                    current_fix = []
        if current_fix:
            fixations.append(current_fix)

        fix_list = []
        for f in fixations:
            f_df = pd.DataFrame(f)
            duration = f_df["system_time"].iloc[-1] - f_df["system_time"].iloc[0]
            if duration >= self.min_duration:
                fix_list.append({
                    "fix_start": f_df["system_time"].iloc[0],
                    "fix_end": f_df["system_time"].iloc[-1],
                    "duration": duration,
                    "x_mean": f_df["avg_gaze_x"].mean(),
                    "y_mean": f_df["avg_gaze_y"].mean(),
                    "dispersion": (f_df["avg_gaze_x"].max() - f_df["avg_gaze_x"].min()) +
                                  (f_df["avg_gaze_y"].max() - f_df["avg_gaze_y"].min())
                })

        return pd.DataFrame(fix_list)

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
            self,
            fixations: pd.DataFrame,
            screenshot_path: Path,
            set_name: Optional[str] = None,
            slide_index: Optional[int] = None,
            title: Optional[str] = None,
            flip_y: bool = True,
            color: str = "yellow",
            alpha: float = 0.7,
            size_scale: float = 2000.0,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Overlay fixations on a slide image.

        Parameters
        ----------
        fixations : pd.DataFrame
            Output of analyze(). Can contain all sets/slides.
        screenshot_path : Path
            Path to the slide screenshot.
        set_name : str, optional
            Filter by subject.
        slide_index : int, optional
            Filter by slide.
        title : str, optional
            Custom plot title.
        flip_y : bool, optional
            Whether to invert Y (origin top-left).
        color : str, optional
            Fixation circle color.
        alpha : float, optional
            Fixation transparency.
        size_scale : float, optional
            Scales fixation size according to duration.
        show : bool, optional
        save_path : Path, optional
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        df = fixations.copy()
        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None:
            df = df[df["slide_index"] == slide_index]

        if df.empty:
            raise ValueError("No fixations to plot for the given filters.")

        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        x = W / 2 + df["x_mean"].values
        y = H / 2 - df["y_mean"].values if flip_y else H / 2 + df["y_mean"].values
        sizes = size_scale * df["duration"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")
        ax.scatter(x, y, s=sizes, c=color, alpha=alpha, edgecolors="black", linewidths=1.2)

        for xi, yi, dur in zip(x, y, df["duration"]):
            ax.text(xi, yi, f"{dur:.2f}s", color="white", fontsize=8, ha="center", va="center")

        ax.set_title(title or f"Fixations — {set_name or 'All Sets'}, Slide {slide_index or '?'}")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Optional
from math import log2


class EntropyAnalyzer:
    """
    Computes gaze entropy and dispersion measures (Shannon entropy and Convex Hull area).

    Parameters
    ----------
    output_folder : Path
        Directory where results and plots are saved.
    """

    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(
            self,
            background_data: pd.DataFrame,
            per: str = "slide",
            bins: int = 100,
            use_convex_hull: bool = True,
    ) -> pd.DataFrame:
        """
        Compute spatial entropy of gaze distributions.

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze data containing ['avg_gaze_x', 'avg_gaze_y', 'set_name', 'slide_index'].
        per : {'global', 'set', 'slide'}, optional
            How to group data before computing entropy.
        bins : int, optional
            Number of bins for 2D histogram.
        use_convex_hull : bool, optional
            If True, computes convex hull area as an additional dispersion measure.

        Returns
        -------
        pd.DataFrame
            DataFrame with entropy and convex hull metrics per group.
        """
        if per not in ["global", "set", "slide"]:
            raise ValueError("`per` must be one of ['global', 'set', 'slide'].")

        # Prepare grouping
        if per == "global":
            groups = [("global", background_data)]
        elif per == "set":
            groups = background_data.groupby("set_name")
        else:  # per == "slide"
            groups = background_data.groupby(["set_name", "slide_index"])

        results = []
        for group_key, df in groups:
            coords = df[["avg_gaze_x", "avg_gaze_y"]].dropna().to_numpy()
            if len(coords) < 3:
                continue

            # Compute 2D histogram (spatial distribution)
            heatmap, _, _ = np.histogram2d(
                df["avg_gaze_x"], df["avg_gaze_y"],
                bins=bins
            )
            p = heatmap / np.sum(heatmap)
            p_nonzero = p[p > 0]

            # Shannon entropy (base 2)
            entropy = -np.sum(p_nonzero * np.log2(p_nonzero))

            # Convex hull area
            convex_area = np.nan
            if use_convex_hull:
                try:
                    hull = ConvexHull(coords)
                    convex_area = hull.volume  # 2D "volume" == area
                except Exception:
                    convex_area = np.nan

            # Store results
            result = {
                "entropy": entropy,
                "convex_hull_area": convex_area,
                "num_points": len(coords),
            }

            if per == "set":
                result["set_name"] = group_key
            elif per == "slide":
                result["set_name"], result["slide_index"] = group_key

            results.append(result)

        results_df = pd.DataFrame(results)
        self.results = results_df
        return results_df

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
            self,
            background_data: pd.DataFrame,
            screenshot_path: Path,
            title: Optional[str] = None,
            flip_y: bool = True,
            bins: int = 100,
            blur_sigma: float = 3.0,
            cmap: str = "hot",
            alpha: float = 0.6,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Visualize gaze entropy overlayed on an image (heatmap + convex hull).

        Parameters
        ----------
        background_data : pd.DataFrame
            The gaze data used to compute entropy.
        screenshot_path : Path
            Path to screenshot image.
        title : str, optional
            Plot title.
        flip_y : bool, optional
            Whether to flip Y-axis (for screen coordinates).
        bins : int, optional
            Histogram bins for the spatial heatmap.
        blur_sigma : float, optional
            Gaussian smoothing factor for heatmap.
        cmap : str, optional
            Colormap for heatmap.
        alpha : float, optional
            Transparency for heatmap overlay.
        show : bool, optional
            Whether to show the plot.
        save_path : Path, optional
            If provided, save the figure to this path.
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        # Load screenshot
        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        # Convert coordinates
        xs = W / 2 + background_data["avg_gaze_x"].dropna().values
        ys = H / 2 - background_data["avg_gaze_y"].dropna().values if flip_y else H / 2 + background_data[
            "avg_gaze_y"].dropna().values

        # Compute heatmap for visualization
        heatmap, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, W], [0, H]])
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")
        ax.imshow(
            heatmap.T,
            cmap=cmap,
            alpha=alpha,
            origin="upper",
            extent=[0, W, H, 0],
        )

        # Optional convex hull outline
        coords = np.column_stack([xs, ys])
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                for simplex in hull.simplices:
                    ax.plot(coords[simplex, 0], coords[simplex, 1], "c-", lw=2, alpha=0.7)
            except Exception:
                pass

        ax.set_title(title or "Gaze Entropy Visualization")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import Optional, List, Union
from sklearn.cluster import DBSCAN, KMeans


# ======================================================
# CLUSTER ANALYZER
# ======================================================
class ClusterAnalyzer:
    """
    Performs clustering on gaze coordinates.

    Supports flexible clustering backends (DBSCAN, KMeans, or custom).
    Results can be visualized by overlaying cluster-colored gaze points
    on the corresponding screenshot.

    Parameters
    ----------
    output_folder : Path
        Directory where results or plots can be saved.
    columns : list[str], optional
        Columns to use for clustering (default: ['avg_gaze_x', 'avg_gaze_y']).
    clustering_model : object, optional
        Custom scikit-learn–compatible clustering model.
        If None, DBSCAN(eps, min_samples) is used.
    eps : float, optional
        DBSCAN epsilon parameter (ignored if using custom model).
    min_samples : int, optional
        DBSCAN min_samples parameter (ignored if using custom model).
    n_clusters : int, optional
        KMeans number of clusters (only used if clustering_model='kmeans').
    """

    def __init__(
            self,
            output_folder: Path,
            columns: Optional[List[str]] = None,
            clustering_model: Optional[object] = None,
            eps: float = 0.05,
            min_samples: int = 5,
            n_clusters: Optional[int] = None,
    ):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.columns = columns or ["avg_gaze_x", "avg_gaze_y"]
        self.clustering_model = clustering_model
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(
            self,
            data: pd.DataFrame,
            clustering_model: Optional[object] = None,
            eps: Optional[float] = None,
            min_samples: Optional[int] = None,
            n_clusters: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform clustering on gaze coordinates.

        Parameters
        ----------
        data : pd.DataFrame
            Flattened gaze data with columns like ['avg_gaze_x', 'avg_gaze_y', 'set_name', 'slide_index'].
        clustering_model : object, optional
            Custom clustering model. Must implement .fit(X) and .labels_.
        eps : float, optional
            DBSCAN epsilon parameter.
        min_samples : int, optional
            DBSCAN min_samples parameter.
        n_clusters : int, optional
            KMeans n_clusters parameter.

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional 'cluster' column.
        """
        df = data.copy()

        # Determine which model to use
        model = clustering_model or self.clustering_model

        if model is None:
            # Default to DBSCAN
            eps = eps or self.eps
            min_samples = min_samples or self.min_samples
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif isinstance(model, str) and model.lower() == "kmeans":
            n_clusters = n_clusters or self.n_clusters or 5
            model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)

        # Extract feature columns
        X = df[self.columns].dropna().to_numpy()
        if len(X) == 0:
            raise ValueError("No valid gaze data available for clustering.")

        # Fit model
        model.fit(X)
        labels = model.labels_

        # Append labels
        df = df.loc[df[self.columns].dropna().index]
        df["cluster"] = labels

        # Store results
        self.results = df
        return df

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
            self,
            background_data: pd.DataFrame,
            screenshot_path: Path,
            title: Optional[str] = None,
            set_name: Optional[str] = None,
            slide_index: Optional[int] = None,
            flip_y: bool = True,
            alpha: float = 0.7,
            point_size: float = 30.0,
            show_noise: bool = True,
            show: bool = True,
            save_path: Optional[Path] = None,
    ):
        """
        Visualize gaze points colored by cluster assignment.

        Parameters
        ----------
        background_data : pd.DataFrame
            Data returned by analyze(), with a 'cluster' column.
        screenshot_path : Path
            Path to the screenshot for this slide.
        title : str, optional
            Plot title.
        set_name : str, optional
            Filter to a particular set.
        slide_index : int, optional
            Filter to a particular slide.
        flip_y : bool, optional
            Whether to flip Y-axis (consistent with other analyzers).
        alpha : float, optional
            Transparency of points.
        point_size : float, optional
            Marker size.
        show_noise : bool, optional
            Whether to show noise points (cluster = -1).
        show : bool, optional
            Whether to display the figure interactively.
        save_path : Path, optional
            If provided, saves the plot to this path.
        """
        if "cluster" not in background_data.columns:
            raise ValueError("Data must contain a 'cluster' column from analyze().")

        df = background_data.copy()
        if set_name is not None and "set_name" in df.columns:
            df = df[df["set_name"] == set_name]
        if slide_index is not None and "slide_index" in df.columns:
            df = df[df["slide_index"] == slide_index]

        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")

        clusters = np.unique(df["cluster"].dropna())
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        for cluster, color in zip(clusters, colors):
            if cluster == -1 and not show_noise:
                continue
            subset = df[df["cluster"] == cluster]
            xs = W / 2 + subset["avg_gaze_x"]
            ys = H / 2 - subset["avg_gaze_y"] if flip_y else H / 2 + subset["avg_gaze_y"]
            ax.scatter(xs, ys, s=point_size, c=[color], alpha=alpha, label=f"Cluster {cluster}")

        ax.legend(loc="best", fontsize=8)
        ax.set_title(title or f"Cluster visualization — {set_name or 'All'} (Slide {slide_index or '?'})")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)



class ScanpathsAnalyzer(BaseAnalyzer):
    """
    Analyzes sequential transitions between fixations.

    Raw gaze data is converted to fixations with ``FixationAnalyzer``. A
    DataFrame containing fixation results can also be supplied directly.
    """

    _result_columns = [
        "set_name", "slide_index", "from_fixation", "to_fixation",
        "start_time", "end_time", "transition_duration", "x_start",
        "y_start", "x_end", "y_end", "distance", "start_duration",
        "end_duration",
    ]

    def __init__(self, output_folder: Path):
        super().__init__(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        background_data: pd.DataFrame,
        per: str = "slide",
    ) -> pd.DataFrame:
        """Compute ordered transitions between consecutive fixations."""
        if per not in ["global", "set", "slide"]:
            raise ValueError("Parameter 'per' must be one of: ['global', 'set', 'slide'].")

        required_group_columns = {"set_name", "slide_index"}
        missing_groups = required_group_columns - set(background_data.columns)
        if missing_groups:
            raise ValueError(
                f"background_data missing required columns: {sorted(missing_groups)}"
            )

        fixation_columns = {"fix_start", "fix_end", "duration", "x_mean", "y_mean"}
        if fixation_columns.issubset(background_data.columns):
            fixations = background_data.copy()
        else:
            gaze_columns = {"avg_gaze_x", "avg_gaze_y", "system_time"}
            missing_gaze = gaze_columns - set(background_data.columns)
            if missing_gaze:
                raise ValueError(
                    "background_data must contain fixation columns or gaze columns: "
                    f"{sorted(missing_gaze)}"
                )
            fixations = FixationAnalyzer(self.output_folder).analyze(background_data)

        if per == "global":
            groups = [("global", fixations.sort_values("fix_start"))]
        elif per == "set":
            groups = fixations.sort_values("fix_start").groupby("set_name")
        else:
            groups = fixations.sort_values("fix_start").groupby(
                ["set_name", "slide_index"]
            )

        transitions = []
        for group_key, group in groups:
            group = group.sort_values("fix_start").reset_index(drop=True)
            if len(group) < 2:
                continue

            for index in range(len(group) - 1):
                start = group.iloc[index]
                end = group.iloc[index + 1]
                record = {
                    "from_fixation": index,
                    "to_fixation": index + 1,
                    "start_time": start["fix_start"],
                    "end_time": end["fix_start"],
                    "transition_duration": end["fix_start"] - start["fix_end"],
                    "x_start": start["x_mean"],
                    "y_start": start["y_mean"],
                    "x_end": end["x_mean"],
                    "y_end": end["y_mean"],
                    "distance": np.sqrt(
                        (end["x_mean"] - start["x_mean"]) ** 2
                        + (end["y_mean"] - start["y_mean"]) ** 2
                    ),
                    "start_duration": start["duration"],
                    "end_duration": end["duration"],
                }

                if per == "global":
                    record["set_name"] = start["set_name"]
                    record["slide_index"] = start["slide_index"]
                elif per == "set":
                    record["set_name"] = group_key
                    record["slide_index"] = start["slide_index"]
                else:
                    record["set_name"], record["slide_index"] = group_key

                transitions.append(record)

        self.results = pd.DataFrame(transitions, columns=self._result_columns)
        return self.results

    def plot_analysis(
        self,
        scanpaths: Optional[pd.DataFrame],
        screenshot_path: Path,
        set_name: Optional[str] = None,
        slide_index: Optional[int] = None,
        title: Optional[str] = None,
        flip_y: bool = True,
        color: str = "cyan",
        alpha: float = 0.8,
        linewidth: float = 2.0,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """Overlay scanpath transitions on a screenshot."""
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        df = scanpaths.copy() if scanpaths is not None else self.results
        if df is None or df.empty:
            raise ValueError("No scanpath data available. Run analyze() first.")
        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None:
            df = df[df["slide_index"] == slide_index]
        if df.empty:
            raise ValueError("No scanpath data matches the provided filters.")

        img = mpimg.imread(screenshot_path)
        height, width = img.shape[:2]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper")

        for _, row in df.iterrows():
            x_start = width / 2 + row["x_start"]
            x_end = width / 2 + row["x_end"]
            y_start = height / 2 - row["y_start"] if flip_y else height / 2 + row["y_start"]
            y_end = height / 2 - row["y_end"] if flip_y else height / 2 + row["y_end"]
            ax.arrow(
                x_start,
                y_start,
                x_end - x_start,
                y_end - y_start,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                head_width=10,
                length_includes_head=True,
            )

        ax.set_title(title or f"Scanpath — {set_name or 'All'}, Slide {slide_index or '?'}")
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


   




import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from pathlib import Path
from typing import Optional, Any, Dict, List


class VoiceTranscription(BaseAnalyzer):
    """
    Transcribes per-slide voice recordings using Whisper and aligns transcript
    segments with gaze samples based on shared experiment time.

    This analyzer operates directly on flattened background_data, similarly
    to the other analyzers in this module.

    Required columns in background_data
    -----------------------------------
    - 'set_name'
    - 'slide_index'
    - 'voice_file'
    - 'voice_start_timestamp'
    - 'system_time'
    - 'avg_gaze_x'
    - 'avg_gaze_y'

    Optional columns that are preserved in output if present
    --------------------------------------------------------
    - 'screenshot_file'
    - 'input_data'
    - 'classification'
    - 'user_classification'
    - 'model_prediction'
    - 'objects_bboxes'

    Notes
    -----
    Alignment is computed as:
        absolute_segment_time = voice_start_timestamp + audio_relative_time + audio_start_offset_sec

    This assumes that:
        - 'voice_start_timestamp' was recorded using the same PsychoPy clock
          as gaze 'system_time'.
    """

    def __init__(
        self,
        output_folder: Path,
        loader_root: Optional[Path] = None,
        model_name: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        sentence_gap_threshold: float = 0.6,
        max_sentence_duration: float = 12.0,
        audio_start_offset_sec: float = 0.0,
        save_transcripts_json: bool = True,
        store_gaze_points: bool = True,
    ):
        super().__init__(output_folder)

        self.model_name = model_name
        self.language = language
        self.device = device
        self.sentence_gap_threshold = sentence_gap_threshold
        self.max_sentence_duration = max_sentence_duration
        self.audio_start_offset_sec = audio_start_offset_sec
        self.save_transcripts_json = save_transcripts_json
        self.store_gaze_points = store_gaze_points
        self._loader_root = loader_root

        self.raw_transcripts: Dict[str, Any] = {}
        self._whisper_model = None

    # ======================================================
    # INTERNAL: MODEL LOADING
    # ======================================================
    def _load_model(self):
        """
        Lazily load Whisper model.
        """
        if self._whisper_model is not None:
            return self._whisper_model

        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "VoiceTranscription requires the 'whisper' package. "
                "Install it with: pip install -U openai-whisper"
            ) from e

        kwargs = {}
        if self.device is not None:
            kwargs["device"] = self.device

        self._whisper_model = whisper.load_model(self.model_name, **kwargs)
        return self._whisper_model

    # ======================================================
    # INTERNAL: TEXT HELPERS
    # ======================================================
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize whitespace in text.
        """
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _looks_like_sentence_end(token: str) -> bool:
        """
        Heuristic check whether a token likely ends a sentence.
        """
        if not isinstance(token, str):
            return False
        token = token.strip()
        return token.endswith((".", "!", "?", ";", ":"))
    
    @staticmethod
    def _truncate_text(text: str, max_chars: int = 140) -> str:
        text = "" if text is None else str(text)
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def _build_sentence_record(
        self,
        word_group: List[Dict[str, Any]],
        sentence_index: int,
    ) -> Dict[str, Any]:
        """
        Build one sentence record from Whisper word-level timestamps.
        """
        text = self._normalize_text(
            " ".join(self._normalize_text(w.get("word", "")) for w in word_group)
        )

        return {
            "sentence_index": sentence_index,
            "sentence_text": text,
            "rel_start": float(word_group[0]["start"]),
            "rel_end": float(word_group[-1]["end"]),
            "duration": float(word_group[-1]["end"]) - float(word_group[0]["start"]),
            "word_count": len(word_group),
            "words": word_group,
        }

    def _words_to_sentences(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Whisper word-level timestamps into sentence-like chunks.

        Splitting rules:
        - punctuation suggests end of sentence
        - large temporal gap between words
        - max duration exceeded
        """
        if not words:
            return []

        sentences = []
        current_words = []
        sentence_index = 0

        for w in words:
            token = self._normalize_text(w.get("word", ""))
            start = w.get("start", None)
            end = w.get("end", None)

            if start is None or end is None or token == "":
                continue

            if not current_words:
                current_words.append(w)
                continue

            prev = current_words[-1]
            gap = float(start) - float(prev.get("end", start))
            current_duration = float(end) - float(current_words[0].get("start", start))

            should_split = (
                gap >= self.sentence_gap_threshold
                or current_duration >= self.max_sentence_duration
                or self._looks_like_sentence_end(prev.get("word", ""))
            )

            if should_split:
                sentences.append(self._build_sentence_record(current_words, sentence_index))
                sentence_index += 1
                current_words = [w]
            else:
                current_words.append(w)

        if current_words:
            sentences.append(self._build_sentence_record(current_words, sentence_index))

        return sentences

    # ======================================================
    # INTERNAL: TRANSCRIPTION
    # ======================================================
    def _transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Transcribe audio with Whisper using word timestamps when available.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        result = model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
            verbose=False,
        )

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append({
                        "word": w.get("word", ""),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "confidence": w.get("probability", np.nan),
                    })

        # fallback to segment-level units if word-level timestamps unavailable
        if words:
            sentences = self._words_to_sentences(words)
        else:
            sentences = []
            for i, seg in enumerate(result.get("segments", [])):
                text = self._normalize_text(seg.get("text", ""))
                if text == "":
                    continue
                sentences.append({
                    "sentence_index": i,
                    "sentence_text": text,
                    "rel_start": float(seg.get("start", 0.0)),
                    "rel_end": float(seg.get("end", 0.0)),
                    "duration": float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)),
                    "word_count": len(text.split()),
                    "words": [],
                })

        return {
            "full_text": self._normalize_text(result.get("text", "")),
            "language": result.get("language", self.language),
            "segments": result.get("segments", []),
            "words": words,
            "sentences": sentences,
        }

    # ======================================================
    # INTERNAL: ALIGNMENT
    # ======================================================
    def _align_sentences_with_gaze(
        self,
        sentences: List[Dict[str, Any]],
        gaze_subset: pd.DataFrame,
        voice_start_timestamp: float,
    ) -> List[Dict[str, Any]]:
        """
        Align transcript sentence windows with gaze samples.
        """
        aligned = []

        for sent in sentences:
            abs_start = float(voice_start_timestamp) + self.audio_start_offset_sec + float(sent["rel_start"])
            abs_end = float(voice_start_timestamp) + self.audio_start_offset_sec + float(sent["rel_end"])

            gaze_in_sentence = gaze_subset[
                (gaze_subset["system_time"] >= abs_start) &
                (gaze_subset["system_time"] <= abs_end)
            ].copy()

            row = {
                "sentence_index": sent["sentence_index"],
                "sentence_text": sent["sentence_text"],
                "rel_start": sent["rel_start"],
                "rel_end": sent["rel_end"],
                "abs_start": abs_start,
                "abs_end": abs_end,
                "duration": sent["duration"],
                "word_count": sent["word_count"],
                "gaze_sample_count": int(len(gaze_in_sentence)),
                "avg_gaze_x_mean": float(gaze_in_sentence["avg_gaze_x"].mean()) if not gaze_in_sentence.empty else np.nan,
                "avg_gaze_y_mean": float(gaze_in_sentence["avg_gaze_y"].mean()) if not gaze_in_sentence.empty else np.nan,
                "avg_gaze_x_std": float(gaze_in_sentence["avg_gaze_x"].std()) if not gaze_in_sentence.empty else np.nan,
                "avg_gaze_y_std": float(gaze_in_sentence["avg_gaze_y"].std()) if not gaze_in_sentence.empty else np.nan,
                "gaze_time_min": float(gaze_in_sentence["system_time"].min()) if not gaze_in_sentence.empty else np.nan,
                "gaze_time_max": float(gaze_in_sentence["system_time"].max()) if not gaze_in_sentence.empty else np.nan,
            }

            if self.store_gaze_points:
                row["gaze_points"] = gaze_in_sentence[
                    ["system_time", "avg_gaze_x", "avg_gaze_y"]
                ].to_dict(orient="records")

            aligned.append(row)

        return aligned

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(
        self,
        background_data: pd.DataFrame,
        per: str = "slide",
    ) -> pd.DataFrame:
        """
        Transcribe available voice recordings and align transcript sentences
        with gaze samples from flattened background_data.

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze DataFrame, one row per gaze point, with repeated
            slide-level metadata.
        per : str, optional
            Included for API consistency with other analyzers.
            Supported values: ['global', 'set', 'slide'].

            Note:
            Voice transcription is always computed internally per unique
            recording, i.e. per (set_name, slide_index), because each slide
            has its own voice file and start timestamp.

        Returns
        -------
        pd.DataFrame
            One row per aligned sentence.
        """
        if per not in ["global", "set", "slide"]:
            raise ValueError("Parameter 'per' must be one of: ['global', 'set', 'slide'].")

        required_cols = {
            "set_name",
            "slide_index",
            "voice_file",
            "voice_start_timestamp",
            "system_time",
            "avg_gaze_x",
            "avg_gaze_y",
        }
        missing = required_cols - set(background_data.columns)
        if missing:
            raise ValueError(f"background_data missing required columns: {sorted(missing)}")

        df = background_data.copy()

        # Keep only rows with non-empty voice file paths
        df = df[df["voice_file"].notna()].copy()
        df = df[df["voice_file"].astype(str).str.strip() != ""].copy()

        if df.empty:
            self.results = pd.DataFrame()
            return self.results

        results_rows = []

        # process per recording (per subject/session and slide)
        for (set_name, slide_index), group in df.groupby(["set_name", "slide_index"]):
            group = group.sort_values("system_time").reset_index(drop=True)

            valid_voice_files = group["voice_file"].dropna().astype(str)
            if valid_voice_files.empty:
                print(f"Warning: No valid voice file for set '{set_name}', slide {slide_index}. Skipping transcription.")
                continue

            valid_voice_starts = group["voice_start_timestamp"].dropna()
            if valid_voice_starts.empty:
                print(f"Warning: No valid voice start timestamp for set '{set_name}', slide {slide_index}. Skipping transcription.")
                continue

            voice_file = valid_voice_files.iloc[0]
            voice_start_timestamp = float(valid_voice_starts.iloc[0])

            audio_path = self._loader_root / Path(voice_file) if self._loader_root else Path(voice_file)
            if not audio_path.exists():
                print(f"Warning: Audio file not found for set '{set_name}', slide {slide_index}: {audio_path}. Skipping transcription.")
                continue

            transcript = self._transcribe_audio(audio_path)
            self.raw_transcripts[f"{set_name}__{slide_index}"] = transcript

            aligned_rows = self._align_sentences_with_gaze(
                sentences=transcript["sentences"],
                gaze_subset=group,
                voice_start_timestamp=voice_start_timestamp,
            )

            # Preserve slide-level metadata if present
            screenshot_file = group["screenshot_file"].iloc[0] if "screenshot_file" in group.columns else None
            input_data = group["input_data"].iloc[0] if "input_data" in group.columns else None
            classification = group["classification"].iloc[0] if "classification" in group.columns else None
            user_classification = group["user_classification"].iloc[0] if "user_classification" in group.columns else None
            model_prediction = group["model_prediction"].iloc[0] if "model_prediction" in group.columns else None
            objects_bboxes = group["objects_bboxes"].iloc[0] if "objects_bboxes" in group.columns else None

            for row in aligned_rows:
                row.update({
                    "set_name": set_name,
                    "slide_index": slide_index,
                    "voice_file": voice_file,
                    "voice_start_timestamp": voice_start_timestamp,
                    "screenshot_file": screenshot_file,
                    "input_data": input_data,
                    "classification": classification,
                    "user_classification": user_classification,
                    "model_prediction": model_prediction,
                    "objects_bboxes": objects_bboxes,
                    "transcript_language": transcript.get("language", None),
                    "full_transcript": transcript.get("full_text", ""),
                })
                results_rows.append(row)

        results_df = pd.DataFrame(results_rows)

        if not results_df.empty:
            results_df = results_df.sort_values(
                ["set_name", "slide_index", "sentence_index"]
            ).reset_index(drop=True)

        self.results = results_df

        if self.save_transcripts_json:
            raw_path = self.output_folder / "VoiceTranscription_raw_transcripts.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(self.raw_transcripts, f, ensure_ascii=False, indent=4)

        return results_df

    # ======================================================
    # SAVE RESULTS
    # ======================================================
    def save_results(self, filename: Optional[str] = None):
        """
        Save analysis results to JSON and CSV.

        Overrides BaseAnalyzer.save_results because:
        - results may contain nested lists/dicts (e.g. gaze_points)
        - CSV export is useful for inspection
        """
        if self.results is None:
            return

        filename = filename or f"{self.__class__.__name__}_results.json"
        filepath = self.output_folder / filename

        self.results.to_json(filepath, orient="records", indent=4, force_ascii=False)

        # optionally save a flat CSV version without raw gaze point lists
        csv_path = self.output_folder / f"{self.__class__.__name__}_results.csv"
        self.results.drop(columns=["gaze_points"], errors="ignore").to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
        )

    
# ======================================================
    # VISUALIZATION: SCREENSHOT + COLORED GAZE + TEXT BOXES
    # ======================================================
    def plot_analysis(
        self,
        transcription_data: Optional[pd.DataFrame] = None,
        screenshot_path: Optional[Path] = None,
        title: Optional[str] = None,
        set_name: Optional[str] = None,
        slide_index: Optional[int] = None,
        flip_y: bool = True,
        alpha: float = 0.75,
        point_size: float = 28.0,
        colormap: str = "tab10",
        max_text_chars: int = 140,
        wrap_width: int = 32,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Plot gaze points overlayed on the screenshot, colored by aligned sentence.
        On the right, render text boxes with the same color as the corresponding
        sentence gaze points.

        Parameters
        ----------
        transcription_data : pd.DataFrame, optional
            Output of analyze(). If None, uses self.results.
        screenshot_path : Path, optional
            Path to screenshot. If None, tries to infer it from the filtered data.
        title : str, optional
            Plot title.
        set_name : str, optional
            Filter by set/session name.
        slide_index : int, optional
            Filter by slide index.
        flip_y : bool, optional
            Whether to invert y coordinate for centered screen coordinates.
        alpha : float, optional
            Point transparency.
        point_size : float, optional
            Scatter point size.
        colormap : str, optional
            Matplotlib colormap name for sentence colors.
        max_text_chars : int, optional
            Maximum number of characters shown in each text box.
        wrap_width : int, optional
            Approximate line width for textbox wrapping.
        show : bool, optional
            Whether to display the figure.
        save_path : Path, optional
            If provided, save the figure.
        """
        df = transcription_data.copy() if transcription_data is not None else self.results

        if df is None or df.empty:
            raise ValueError("No transcription results available. Run analyze() first.")

        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None:
            df = df[df["slide_index"] == slide_index]

        if df.empty:
            raise ValueError("No transcription rows match the provided filters.")

        df = df.sort_values("sentence_index").reset_index(drop=True)

        if screenshot_path is None:
            if "screenshot_file" not in df.columns or df["screenshot_file"].dropna().empty:
                raise ValueError("screenshot_path was not provided and could not be inferred from transcription data.")
            screenshot_path = self._loader_root / Path(str(df["screenshot_file"].dropna().iloc[0])) if self._loader_root else Path(str(df["screenshot_file"].dropna().iloc[0]))
        else:
            screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        n_sentences = max(len(df), 1)
        cmap = plt.get_cmap(colormap, n_sentences)
        colors = [cmap(i) for i in range(n_sentences)]

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.05)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        # --- Left panel: screenshot + gaze points ---
        ax_img.imshow(img, origin="upper")

        for i, (_, row) in enumerate(df.iterrows()):
            color = colors[i]
            gaze_points = row.get("gaze_points", [])

            if gaze_points is None or len(gaze_points) == 0:
                continue

            gaze_df = pd.DataFrame(gaze_points)
            if gaze_df.empty:
                continue

            xs = W / 2 + gaze_df["avg_gaze_x"].astype(float).values
            ys = H / 2 - gaze_df["avg_gaze_y"].astype(float).values if flip_y else H / 2 + gaze_df["avg_gaze_y"].astype(float).values

            ax_img.scatter(
                xs,
                ys,
                s=point_size,
                c=[color],
                alpha=alpha,
                label=f"Sentence {int(row['sentence_index'])}",
            )

            # mark sentence centroid if available
            if not np.isnan(row.get("avg_gaze_x_mean", np.nan)) and not np.isnan(row.get("avg_gaze_y_mean", np.nan)):
                cx = W / 2 + float(row["avg_gaze_x_mean"])
                cy = H / 2 - float(row["avg_gaze_y_mean"]) if flip_y else H / 2 + float(row["avg_gaze_y_mean"])
                ax_img.scatter(
                    [cx],
                    [cy],
                    s=point_size * 3.0,
                    c=[color],
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=1.0,
                    marker="o",
                )
                ax_img.text(
                    cx,
                    cy,
                    str(int(row["sentence_index"])),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                )

        ax_img.set_title(title or f"Voice–gaze alignment — {set_name}, slide {slide_index}")
        ax_img.axis("off")

        # --- Right panel: colored sentence boxes ---
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        top_margin = 0.97
        bottom_margin = 0.03
        available_h = top_margin - bottom_margin
        box_h = available_h / max(n_sentences, 1)

        for i, (_, row) in enumerate(df.iterrows()):
            color = colors[i]
            y_top = top_margin - i * box_h
            y_center = y_top - box_h / 2

            sent_idx = int(row["sentence_index"])
            sent_text = self._truncate_text(str(row["sentence_text"]), max_chars=max_text_chars)
            sent_text = textwrap.fill(sent_text, width=wrap_width)

            time_info = f"[{row['rel_start']:.2f}s – {row['rel_end']:.2f}s]"
            gaze_info = f"gaze n={int(row['gaze_sample_count'])}"

            full_text = f"S{sent_idx} {time_info}\n{sent_text}\n{gaze_info}"

            ax_txt.text(
                0.02,
                y_center,
                full_text,
                ha="left",
                va="center",
                fontsize=9,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.55,
                ),
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis_summary(
        self,
        transcription_data: Optional[pd.DataFrame] = None,
        set_name: Optional[str] = None,
        slide_index: Optional[int] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Plot a simple timeline of transcript sentences and aligned gaze counts.

        Parameters
        ----------
        transcription_data : pd.DataFrame, optional
            Output of analyze(). If None, uses self.results.
        set_name : str, optional
            Filter by set/session name.
        slide_index : int, optional
            Filter by slide index.
        title : str, optional
            Plot title.
        show : bool, optional
            Whether to display the figure interactively.
        save_path : Path, optional
            If provided, saves the figure to this location.
        """
        df = transcription_data.copy() if transcription_data is not None else self.results

        if df is None or df.empty:
            raise ValueError("No transcription results available. Run analyze() first.")

        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None:
            df = df[df["slide_index"] == slide_index]

        if df.empty:
            raise ValueError("No transcription rows match the provided filters.")

        df = df.sort_values("sentence_index").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5))

        for _, row in df.iterrows():
            ax.plot(
                [row["abs_start"], row["abs_end"]],
                [row["sentence_index"], row["sentence_index"]],
                linewidth=6,
                solid_capstyle="butt",
            )
            ax.text(
                row["abs_start"],
                row["sentence_index"] + 0.12,
                f'{int(row["sentence_index"])}: {str(row["sentence_text"])[:80]}',
                fontsize=8,
                ha="left",
                va="bottom",
            )

        ax2 = ax.twinx()
        widths = np.maximum(df["duration"].fillna(0.1).values, 0.05)
        ax2.bar(
            df["abs_start"].values,
            df["gaze_sample_count"].fillna(0).values,
            width=widths,
            alpha=0.25,
        )

        ax.set_xlabel("Experiment time (s)")
        ax.set_ylabel("Sentence index")
        ax2.set_ylabel("Gaze sample count")
        ax.set_title(title or f"Voice–gaze alignment — {set_name}, slide {slide_index}")
        ax.grid(alpha=0.2)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

from .bbox import (
    analyze_bbox_attention,
    analyze_bbox_image,
    analyze_bbox_text,
    analyze_bbox_timeseries,
    bbox_edges_centered,
    extract_text_bboxes,
    extract_timeseries_bboxes,
    evaluate_bbox_attention,
    parse_input_data,
    parse_objects_bboxes,
    plot_bbox_attention,
    plot_bbox_image,
    plot_bbox_text,
    plot_bbox_timeseries,
    point_inside_bbox,
    point_inside_polygon,
    polygon_to_plot_coords,
    polygon_vertices,
)


class BBoxAttentionAnalyzer(BaseAnalyzer):

    @staticmethod
    def _parse_objects_bboxes(value: Any) -> Dict[str, Any]:
        return parse_objects_bboxes(value)

    @staticmethod
    def _bbox_edges_centered(bbox: Dict[str, float]) -> Dict[str, float]:
        return bbox_edges_centered(bbox)

    @staticmethod
    def _polygon_vertices(value: Any) -> Optional[np.ndarray]:
        return polygon_vertices(value)

    @staticmethod
    def _point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        return point_inside_polygon(x, y, polygon)

    @staticmethod
    def _polygon_to_plot_coords(
        polygon: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:
        return polygon_to_plot_coords(polygon, width, height)

    @staticmethod
    def _point_inside_bbox(
        x: float,
        y: float,
        bbox: Dict[str, float],
        margin: float = 2.0,
    ) -> bool:
        return point_inside_bbox(x, y, bbox, margin=margin)

    def analyze(
        self,
        raw_data: pd.DataFrame,
        gaze_data: pd.DataFrame,
        use_fixations: bool = False,
    ) -> pd.DataFrame:
        result = analyze_bbox_attention(
            raw_data=raw_data,
            gaze_data=gaze_data,
            use_fixations=use_fixations,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
            resolve_gaze_columns=self._resolve_gaze_columns,
        )
        self.results = result
        return result

    def evaluate(
            self,
            scored_bboxes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = scored_bboxes if scored_bboxes is not None else self.results
        return evaluate_bbox_attention(df)

    def plot_analysis(
            self,
            scored_bboxes: pd.DataFrame,
            gaze_data: pd.DataFrame,
            screenshot_path: Path,
            set_name: Optional[str] = None,
            slide_index: Optional[int] = None,
            title: Optional[str] = None,
            top_k: Optional[int] = 20,
            min_hits: int = 1,
            show_gaze: bool = True,
            show: bool = True,
            save_path: Optional[Path] = None,
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
            filter_set_and_slide=self._filter_set_and_slide,
        )


class BBoxTimeSeriesAnalyzer(BaseAnalyzer):
    @staticmethod
    def _parse_input_data(raw_input_data: Any) -> np.ndarray:
        return parse_input_data(raw_input_data)

    @staticmethod
    def _extract_timeseries_bboxes(raw_bboxes: Any) -> List[Dict[str, Any]]:
        return extract_timeseries_bboxes(raw_bboxes)

    def analyze(
        self,
        slide_data: pd.DataFrame,
        set_name: Optional[Any] = None,
        slide_index: Optional[Any] = None,
    ) -> pd.DataFrame:
        self.results = analyze_bbox_timeseries(
            slide_data=slide_data,
            set_name=set_name,
            slide_index=slide_index,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
        )
        return self.results

    def plot_analysis(
        self,
        slide_data: pd.DataFrame,
        scored_bboxes: Optional[pd.DataFrame] = None,
        set_name: Optional[Any] = None,
        slide_index: Optional[Any] = None,
        area_x: Optional[float] = None,
        area_y: Optional[float] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        if scored_bboxes is None:
            scored_bboxes = self.results
        return plot_bbox_timeseries(
            slide_data=slide_data,
            scored_bboxes=scored_bboxes,
            set_name=set_name,
            slide_index=slide_index,
            area_x=area_x,
            area_y=area_y,
            title=title,
            show=show,
            save_path=save_path,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
        )


class BBoxImageAnalyzer(BaseAnalyzer):
    def analyze(
        self,
        slide_data: pd.DataFrame,
        set_name: Optional[Any] = None,
        slide_index: Optional[Any] = None,
    ) -> pd.DataFrame:
        self.results = analyze_bbox_image(
            slide_data=slide_data,
            set_name=set_name,
            slide_index=slide_index,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
        )
        return self.results

    def plot_analysis(
        self,
        scored_bboxes: pd.DataFrame,
        gaze_data: pd.DataFrame,
        screenshot_path: Path,
        set_name: Optional[str] = None,
        slide_index: Optional[int] = None,
        title: Optional[str] = None,
        top_k: Optional[int] = 20,
        min_hits: int = 1,
        show_gaze: bool = True,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        return plot_bbox_image(
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
            filter_set_and_slide=self._filter_set_and_slide,
        )


class BBoxTextAnalyzer(BaseAnalyzer):
    @staticmethod
    def _extract_text_bboxes(raw_bboxes: Any, level: str = "words") -> List[Dict[str, Any]]:
        return extract_text_bboxes(raw_bboxes, level=level)

    def analyze(
        self,
        slide_data: pd.DataFrame,
        level: str = "words",
        set_name: Optional[Any] = None,
        slide_index: Optional[Any] = None,
    ) -> pd.DataFrame:
        self.results = analyze_bbox_text(
            slide_data=slide_data,
            level=level,
            set_name=set_name,
            slide_index=slide_index,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
        )
        return self.results

    def plot_analysis(
        self,
        slide_data: pd.DataFrame,
        level: str = "words",
        scored_bboxes: Optional[pd.DataFrame] = None,
        set_name: Optional[Any] = None,
        slide_index: Optional[Any] = None,
        area_x: Optional[float] = None,
        area_y: Optional[float] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        if scored_bboxes is None:
            scored_bboxes = self.results
        return plot_bbox_text(
            slide_data=slide_data,
            level=level,
            scored_bboxes=scored_bboxes,
            set_name=set_name,
            slide_index=slide_index,
            area_x=area_x,
            area_y=area_y,
            title=title,
            show=show,
            save_path=save_path,
            normalize_slide_index_column=self._normalize_slide_index_column,
            filter_set_and_slide=self._filter_set_and_slide,
        )
