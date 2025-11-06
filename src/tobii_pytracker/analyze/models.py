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




class SaccadeAnalyzer(BaseAnalyzer):
    """
    Calculates saccade metrics (dx, dy, dt, amplitude, velocity) from gaze data.

    Usage:
        analyzer = SaccadeAnalyzer(output_folder=Path("./out"))
        events = analyzer.analyze(background_data, per="slide")
        analyzer.plot_analysis(events, kind="time", set_name="subject01", slide_index=2)
    """

    def analyze(
        self,
        data: pd.DataFrame,
        per: str = "slide",
        subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute saccade events and metrics.

        Parameters
        ----------
        data : pd.DataFrame
            Flattened gaze data with columns 'avg_gaze_x', 'avg_gaze_y', 'system_time',
            'set_name' and 'slide_index'.
        per : str, optional
            'slide'  -> compute saccades per (set_name, slide_index)
            'set'    -> compute saccades per set_name
            'global' -> compute saccades across all (no grouping)
        subset : list[str], optional
            If provided, restrict processing to these set_name values.

        Returns
        -------
        pd.DataFrame
            DataFrame of saccade events with columns:
            ['set_name','slide_index','system_time_prev','system_time',
             'x_prev','y_prev','x','y','dx','dy','dt','amplitude','velocity'].
        """
        if per not in ["global", "set", "slide"]:
            raise ValueError("`per` must be one of ['global','set','slide'].")

        df = data.copy()

        if subset is not None:
            df = df[df["set_name"].isin(subset)]

        # choose grouping keys for iteration (not aggregation)
        if per == "global":
            group_keys = []
        elif per == "set":
            group_keys = ["set_name"]
        else:  # slide
            group_keys = ["set_name", "slide_index"]

        events_list = []

        # If no grouping, treat whole df as single group
        if not group_keys:
            groups = [("", df.sort_values("system_time"))]
        else:
            groups = list(df.groupby(group_keys))

        for grp_key, group in groups:
            g = group.sort_values("system_time").reset_index(drop=True)

            # compute previous sample values by shifting
            g["x_prev"] = g["avg_gaze_x"].shift(1)
            g["y_prev"] = g["avg_gaze_y"].shift(1)
            g["t_prev"] = g["system_time"].shift(1)

            # drop first row (no previous)
            ev = g.dropna(subset=["x_prev", "y_prev", "t_prev"]).copy()
            if ev.empty:
                continue

            ev["dx"] = ev["avg_gaze_x"] - ev["x_prev"]
            ev["dy"] = ev["avg_gaze_y"] - ev["y_prev"]
            ev["dt"] = ev["system_time"] - ev["t_prev"]
            # avoid division by zero
            ev["dt"] = ev["dt"].replace(0, np.nan)

            ev["amplitude"] = np.sqrt(ev["dx"] ** 2 + ev["dy"] ** 2)
            ev["velocity"] = ev["amplitude"] / ev["dt"]

            # keep identifiers (if grouped, grp_key may be tuple)
            if group_keys:
                if isinstance(grp_key, tuple):
                    for k, v in zip(group_keys, grp_key):
                        ev[k] = v
                else:
                    ev[group_keys[0]] = grp_key
            else:
                # for global, if original df has set_name/slide_index keep them per-row
                pass

            events_list.append(ev[[
                *(group_keys if group_keys else []),
                "system_time", "t_prev", "x_prev", "y_prev", "avg_gaze_x", "avg_gaze_y",
                "dx", "dy", "dt", "amplitude", "velocity"
            ]])

        if events_list:
            events_df = pd.concat(events_list, ignore_index=True)
        else:
            events_df = pd.DataFrame(
                columns=[
                    *(group_keys if group_keys else []),
                    "system_time", "t_prev", "x_prev", "y_prev", "avg_gaze_x", "avg_gaze_y",
                    "dx", "dy", "dt", "amplitude", "velocity"
                ]
            )

        # store results (saccade events)
        self.results = events_df
        return events_df

    def plot_analysis(
        self,
        events: Optional[pd.DataFrame] = None,
        kind: str = "time",
        set_name: Optional[str] = None,
        slide_index: Optional[int] = None,
        metric: str = "velocity",
        bins: int = 40,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Plot saccade analysis results.

        Parameters
        ----------
        events : pd.DataFrame, optional
            DataFrame returned by analyze(). If None, uses self.results.
        kind : str, optional
            'time' -> plot metric (velocity/amplitude) over time (system_time)
            'hist' -> histogram of metric
        set_name : str, optional
            Filter to a particular subject (if present in events)
        slide_index : int, optional
            Filter to a particular slide (if present)
        metric : str, optional
            Which metric to plot: 'velocity' or 'amplitude' (or any numeric column)
        bins : int, optional
            For histogram.
        show : bool, optional
        save_path : Path, optional
        """
        if events is None:
            events = self.results
        if events is None or events.empty:
            raise ValueError("No saccade events to plot. Run analyze() first or provide events.")

        df = events.copy()
        if set_name is not None and "set_name" in df.columns:
            df = df[df["set_name"] == set_name]
        if slide_index is not None and "slide_index" in df.columns:
            df = df[df["slide_index"] == slide_index]

        if df.empty:
            raise ValueError("Filtered events are empty; nothing to plot.")

        if kind == "time":
            if "system_time" not in df.columns or metric not in df.columns:
                raise ValueError("Required columns missing for time plot.")
            plt.figure(figsize=(10, 4))
            plt.plot(df["system_time"], df[metric], marker="o", linestyle="-", alpha=0.7)
            plt.xlabel("System time")
            plt.ylabel(metric.capitalize())
            plt.title(f"Saccade {metric} over time" + (f" — {set_name}" if set_name else ""))
            plt.grid(True)
        elif kind == "hist":
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not present in events.")
            plt.figure(figsize=(8, 4))
            plt.hist(df[metric].dropna(), bins=bins, edgecolor="black", alpha=0.7)
            plt.xlabel(metric.capitalize())
            plt.title(f"Histogram of saccade {metric}")
        else:
            raise ValueError("kind must be 'time' or 'hist'.")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close()


# ======================================================
# FIXATION ANALYZER
# ======================================================
class FixationAnalyzer(BaseAnalyzer):
    """Computes fixation duration, count, and dispersion."""

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        results = (
            data.groupby("event_id")
            .agg(
                start_time=("system_time", "min"),
                end_time=("system_time", "max"),
                duration=("system_time", lambda x: x.max() - x.min()),
                x_mean=("avg_gaze_x", "mean"),
                y_mean=("avg_gaze_y", "mean"),
                dispersion=("avg_gaze_x", lambda x: np.std(x) + np.std(data.loc[x.index, "avg_gaze_y"])),
            )
            .reset_index()
        )
        self.results = results
        return results

    def plot_analysis(self, data: pd.DataFrame, img_path: Path):
        img = mpimg.imread(img_path)
        H, W = img.shape[:2]
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.scatter(W / 2 + data["avg_gaze_x"], H / 2 - data["avg_gaze_y"], s=30, c="red", alpha=0.6)
        plt.title("Fixations")
        plt.axis("off")
        plt.show()


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
