import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any, List, Literal
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
        avg_y = H / 2 - background_data["avg_gaze_y"].dropna().values if flip_y else H / 2 + background_data["avg_gaze_y"].dropna().values

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

            g["amplitude"] = np.sqrt(g["dx"]**2 + g["dy"]**2)
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
                    (seg["avg_gaze_x"].iloc[-1] - seg["x_prev"].iloc[0])**2 +
                    (seg["avg_gaze_y"].iloc[-1] - seg["y_prev"].iloc[0])**2
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
        g["velocity"] = np.sqrt(g["dx"]**2 + g["dy"]**2) / g["dt"]
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
        ys = H / 2 - background_data["avg_gaze_y"].dropna().values if flip_y else H / 2 + background_data["avg_gaze_y"].dropna().values

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



class ConceptAnalyzer(BaseAnalyzer):
    """
    Defines AOI-like (Areas of Interest) concepts from clusters
    and computes engagement statistics per concept.

    This class operates directly on flattened background data.

    Parameters
    ----------
    background_data : pd.DataFrame
        Combined dataset across subjects, slides, and events.
    """

    def __init__(self, background_data: pd.DataFrame):
        super().__init__(background_data)

    #TODO: Implement methods for concept definition and analysis.

class ScanpathsAnalyzer(BaseAnalyzer):
    """
    Analyzes sequential transitions between fixations
    to characterize scanpaths per slide, set, or globally.
    """

    def __init__(self, background_data: pd.DataFrame):
        super().__init__(background_data)
    #TODO: Implement methods for concept definition and analysis.
   

class VoiceTranscription(BaseAnalyzer):
    """
    Transcribes and aligns voice data with gaze events.

    This is a placeholder implementation — replace transcription
    logic with a real speech-to-text model as needed.
    """

    def __init__(self, background_data: pd.DataFrame):
        super().__init__(background_data)
    #TODO: Implement methods for concept definition and analysis.
    
