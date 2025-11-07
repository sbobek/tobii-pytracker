import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any, List
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import Optional, Literal


class SaccadeAnalyzer:
    """
    Calculates saccade metrics (dx, dy, dt, amplitude, velocity) from gaze data.

    This version supports multiple parameterizations for different
    saccade detection methods (velocity- and acceleration-based).

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
        Minimum velocity (pixels/second) to classify a movement as a saccade.
        Used only for I-VT. Default = 100.
    acceleration_threshold : float, optional
        Minimum acceleration (pixels/second²) to classify a movement as a saccade.
        Used only for acceleration-based detection. Default = 5000.
    min_duration : float, optional
        Minimum duration (seconds) to consider a saccade valid. Default = 0.01.
    """

    def __init__(
        self,
        output_folder: Path,
        method: Literal["ivt", "acceleration"] = "ivt",
        velocity_threshold: float = 100.0,
        acceleration_threshold: float = 5000.0,
        min_duration: float = 0.01,
    ):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.method = method
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_duration = min_duration

        self.results: Optional[pd.DataFrame] = None

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(self, background_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute saccade events and metrics.

        Always processed per (set_name, slide_index).

        Parameters
        ----------
        background_data : pd.DataFrame
            Flattened gaze data with columns:
            ['avg_gaze_x', 'avg_gaze_y', 'system_time', 'set_name', 'slide_index']

        Returns
        -------
        pd.DataFrame
            DataFrame of saccade events with columns:
            ['set_name','slide_index','system_time','t_prev','x_prev','y_prev',
             'avg_gaze_x','avg_gaze_y','dx','dy','dt','amplitude','velocity','acceleration']
        """
        df = background_data.copy()
        events_list = []

        for (set_name, slide_index), group in df.groupby(["set_name", "slide_index"]):
            g = group.sort_values("system_time").reset_index(drop=True)
            if len(g) < 2:
                continue

            # Compute velocity & acceleration
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

            # Aggregate per detected saccade
            for start_idx, end_idx in saccades:
                seg = g.iloc[start_idx:end_idx + 1]
                duration = seg["dt"].sum()
                if duration < self.min_duration:
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
                    "amplitude": np.sqrt(
                        (seg["avg_gaze_x"].iloc[-1] - seg["x_prev"].iloc[0])**2 +
                        (seg["avg_gaze_y"].iloc[-1] - seg["y_prev"].iloc[0])**2
                    ),
                    "peak_velocity": seg["velocity"].max(),
                    "mean_velocity": seg["velocity"].mean(),
                    "mean_acceleration": seg["acceleration"].abs().mean(),
                })

        if events_list:
            events_df = pd.DataFrame(events_list)
        else:
            events_df = pd.DataFrame(columns=[
                "set_name", "slide_index", "start_time", "end_time", "duration",
                "x_start", "y_start", "x_end", "y_end",
                "amplitude", "peak_velocity", "mean_velocity", "mean_acceleration"
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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import Optional, Literal


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


# ======================================================
# CLUSTER ANALYZER
# ======================================================
class ClusterAnalyzer(BaseAnalyzer):
    """Performs clustering on gaze coordinates."""

    def analyze(self, data: pd.DataFrame, clustering_model=None, eps=0.05, min_samples=5) -> pd.DataFrame:
        X = data[["avg_gaze_x", "avg_gaze_y"]].dropna().to_numpy()
        model = clustering_model or DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        valid_idx = data[["avg_gaze_x", "avg_gaze_y"]].dropna().index
        data.loc[valid_idx, "cluster_label"] = labels
        self.results = data[["system_time", "cluster_label"]]
        return self.results

    def plot_analysis(self, data: pd.DataFrame, color_map="tab10"):
        df = data.dropna(subset=["avg_gaze_x", "avg_gaze_y", "cluster_label"])
        labels = sorted(set(df["cluster_label"]))
        cmap = cm.get_cmap(color_map, len(labels))
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(labels):
            cluster = df[df["cluster_label"] == label]
            plt.scatter(cluster["avg_gaze_x"], cluster["avg_gaze_y"], color=cmap(i), s=20, alpha=0.7)
        plt.title("Clusters")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


# ======================================================
# ENTROPY ANALYZER
# ======================================================
class EntropyAnalyzer(BaseAnalyzer):
    """Computes entropy of gaze distributions."""

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        hist, _, _ = np.histogram2d(data["avg_gaze_x"], data["avg_gaze_y"], bins=20)
        p = hist.flatten() / hist.sum()
        ent = entropy(p[p > 0])
        results = pd.DataFrame({"entropy": [ent]})
        self.results = results
        return results

    def plot_analysis(self):
        plt.bar(["Entropy"], [self.results["entropy"].iloc[0]], color="orange")
        plt.title("Gaze Entropy")
        plt.ylabel("Entropy")
        plt.show()


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

    def analyze(self, mode: str = "set", subset=None) -> pd.DataFrame:
        """
        Identify and analyze concepts based on cluster labels.

        Parameters
        ----------
        mode : str, optional
            'slide' — compute per-slide per-subject
            'set' — compute per-subject across slides
            'global' — aggregate across all data
        subset : list[str], optional
            Optional list of subject identifiers to include in analysis.

        Returns
        -------
        pd.DataFrame
            Concept engagement statistics.
        """
        data = self.background_data.copy()

        if subset is not None:
            data = data[data["set_name"].isin(subset)]

        if "cluster_label" not in data.columns:
            raise ValueError("Cluster labels required. Run ClusterAnalyzer first.")

        group_keys = {
            "slide": ["set_name", "slide_index", "cluster_label"],
            "set": ["set_name", "cluster_label"],
            "global": ["cluster_label"],
        }[mode]

        concept_stats = (
            data.groupby(group_keys)
            .agg(
                mean_duration=("system_time", lambda x: x.max() - x.min()),
                fixation_count=("event_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"cluster_label": "concept_id"})
        )

        self.results = concept_stats
        return concept_stats

    def plot_analysis(self, set_name: str = None, slide_index: int = None):
        """
        Visualize mean engagement duration per concept.

        Parameters
        ----------
        set_name : str, optional
            Subject name for filtering (used if mode='set' or 'slide').
        slide_index : int, optional
            Slide index (used if mode='slide').
        """
        if self.results is None or self.results.empty:
            raise ValueError("Run analyze() before plotting.")

        df = self.results.copy()
        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None and "slide_index" in df.columns:
            df = df[df["slide_index"] == slide_index]

        plt.figure(figsize=(8, 4))
        plt.bar(df["concept_id"], df["mean_duration"], color="teal")
        plt.title(f"Concept Engagement — {set_name or 'Global'}")
        plt.xlabel("Concept ID")
        plt.ylabel("Mean Duration")
        plt.show()


class ScanpathsAnalyzer(BaseAnalyzer):
    """
    Analyzes sequential transitions between fixations
    to characterize scanpaths per slide, set, or globally.
    """

    def __init__(self, background_data: pd.DataFrame):
        super().__init__(background_data)

    def analyze(self, mode: str = "set", subset=None) -> pd.DataFrame:
        """
        Compute fixation-to-fixation transitions (scanpaths).

        Parameters
        ----------
        mode : str, optional
            'slide' — per slide per subject
            'set' — per subject across slides
            'global' — across all subjects
        subset : list[str], optional
            Restrict analysis to specific subjects.

        Returns
        -------
        pd.DataFrame
            Transition counts.
        """
        data = self.background_data.copy()
        if subset is not None:
            data = data[data["set_name"].isin(subset)]

        group_keys = {
            "slide": ["set_name", "slide_index"],
            "set": ["set_name"],
            "global": [],
        }[mode]

        results = []
        for _, group in data.groupby(group_keys or [lambda _: True]):
            group = group.sort_values("system_time")
            group["next_event"] = group["event_id"].shift(-1)
            trans = (
                group.groupby(["event_id", "next_event"])
                .size()
                .reset_index(name="count")
            )
            for key, val in zip(group_keys, group[group_keys].iloc[0].values) if group_keys else []:
                trans[key] = val
            results.append(trans)

        transitions = pd.concat(results, ignore_index=True)
        self.results = transitions
        return transitions

    def plot_analysis(self, set_name: str = None, slide_index: int = None):
        """
        Plot transition frequencies for scanpaths.

        Parameters
        ----------
        set_name : str, optional
        slide_index : int, optional
        """
        if self.results is None or self.results.empty:
            raise ValueError("Run analyze() before plotting.")

        df = self.results.copy()
        if set_name is not None and "set_name" in df.columns:
            df = df[df["set_name"] == set_name]
        if slide_index is not None and "slide_index" in df.columns:
            df = df[df["slide_index"] == slide_index]

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(df)), df["count"])
        plt.title(f"Scanpath Transitions — {set_name or 'Global'}")
        plt.xlabel("Transition Index")
        plt.ylabel("Count")
        plt.show()


class VoiceTranscription(BaseAnalyzer):
    """
    Transcribes and aligns voice data with gaze events.

    This is a placeholder implementation — replace transcription
    logic with a real speech-to-text model as needed.
    """

    def __init__(self, background_data: pd.DataFrame):
        super().__init__(background_data)

    def analyze(self, mode: str = "set", subset=None) -> pd.DataFrame:
        """
        Generate aligned voice transcripts.

        Parameters
        ----------
        mode : str, optional
            'slide' — per slide
            'set' — per subject
            'global' — aggregated across all
        subset : list[str], optional
            Optional subset of subjects.
        """
        data = self.background_data.copy()
        if subset is not None:
            data = data[data["set_name"].isin(subset)]

        transcripts = data[["set_name", "slide_index", "system_time"]].copy()
        transcripts["transcribed_text"] = [
            "dummy transcription" for _ in range(len(transcripts))
        ]

        self.results = transcripts
        return transcripts

    def plot_analysis(self, set_name: str = None, slide_index: int = None):
        """
        Show a sample of transcribed text.
        """
        if self.results is None or self.results.empty:
            raise ValueError("Run analyze() before plotting.")

        df = self.results.copy()
        if set_name is not None:
            df = df[df["set_name"] == set_name]
        if slide_index is not None and "slide_index" in df.columns:
            df = df[df["slide_index"] == slide_index]

        sample_text = "\n".join(df["transcribed_text"].head(5))
        plt.figure(figsize=(8, 4))
        plt.text(0.05, 0.5, sample_text, fontsize=12)
        plt.axis("off")
        plt.title(f"Voice Transcription Sample — {set_name or 'Global'}")
        plt.show()
