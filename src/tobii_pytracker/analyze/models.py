import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .data_loader import DataLoader
from scipy.ndimage import gaussian_filter


class BaseAnalyzer:
    def __init__(self, data_loader: DataLoader, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.data_loader = data_loader
        self.results: Optional[pd.DataFrame] = None

    def analyze(self, *args, **kwargs) -> pd.DataFrame:
        """Run full analysis (override in subclass). Should return a pandas DataFrame with results."""
        raise NotImplementedError

    def plot_analysis(self, *args, **kwargs):
        """Plot analysis results (override in subclass)."""
        raise NotImplementedError

    def save_results(self, filename: Optional[str] = None):
        """Explicitly save results to disk."""
        if self.results is None:
            return

        filename = filename or f"{self.__class__.__name__}_results.json"
        filepath = self.output_folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results.to_dict(orient="records"), f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# ------------------------- ANALYZERS --------------------------------
# ---------------------------------------------------------------------


class HeatmapAnalyzer(BaseAnalyzer):
    """
    Generates gaze heatmaps (either per-slide or aggregated across subjects)
    and overlays them over real screenshots.

    The heatmap visualizes areas of visual attention intensity, using Gaussian-blurred
    density maps of average gaze positions.

    Modes
    -----
    • **per='slide'** — Computes heatmaps per individual slide (for a single subject).
    • **per='global'** — Aggregates data from *all subjects* for each unique
      `input_data` (i.e., each unique image/slide identifier), producing global
      attention maps across viewers.
    """

    def __init__(self, data_loader, output_folder: Path):
        super().__init__(data_loader, output_folder)

    # ======================================================
    # ANALYSIS
    # ======================================================
    def analyze(self, set_name: str, per: str = "slide") -> pd.DataFrame:
        """
        Perform gaze heatmap analysis.

        Parameters
        ----------
        set_name : str
            Subject folder name (if per='slide') or one subject to anchor aggregation.
        per : str
            'slide' — compute per-slide for one subject.
            'global' — compute aggregated heatmaps per input_data across all subjects.

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean gaze position and point counts)
            per slide or per input_data.
        """
        if per not in ["slide", "global"]:
            raise ValueError("`per` must be either 'slide' or 'global'")

        if per == "slide":
            data = self.data_loader.get_subject_data(set_name, flatten=True)
            group_key = "slide_index"
        else:
            # Aggregate across all subjects by input_data
            all_data = self.data_loader.get_all_data(flatten=True)
            data = pd.concat(all_data.values(), ignore_index=True)
            group_key = "input_data"

        # Compute mean gaze and count per grouping
        results = (
            data.groupby(group_key)
            .agg(
                avg_gaze_x=("avg_gaze_x", "mean"),
                avg_gaze_y=("avg_gaze_y", "mean"),
                gaze_count=("avg_gaze_x", "count"),
            )
            .reset_index()
        )

        self.results = results

        # Optionally register columns in DataLoader
        for col in results.columns:
            if col not in data.columns:
                try:
                    self.data_loader.add_column(col, value=None)
                except Exception:
                    pass

        return results

    # ======================================================
    # VISUALIZATION
    # ======================================================
    def plot_analysis(
        self,
        set_name: str,
        slide_index: Optional[int] = None,
        input_data: Optional[str] = None,
        per: str = "slide",
        flip_y: bool = True,
        blur_sigma: float = 3.0,
        bins: int = 50,
        cmap: str = "hot",
        alpha: float = 0.6,
        save_path: Optional[Path] = None,
        show: bool = True,
    ):
        """
        Plot gaze heatmap overlayed over the actual slide screenshot.

        Parameters
        ----------
        set_name : str
            Subject name (used to locate screenshot files).
        slide_index : int, optional
            Slide index (required if per='slide').
        input_data : str, optional
            Image identifier (required if per='global').
        per : str
            'slide' or 'global' (same as in analyze()).
        flip_y : bool, optional
            Whether to invert Y axis (for screen coordinates where y grows downward).
        blur_sigma : float, optional
            Standard deviation for Gaussian smoothing of heatmap.
        bins : int
            Resolution of the 2D histogram grid.
        cmap : str
            Colormap for heatmap visualization.
        alpha : float
            Transparency of the heatmap overlay.
        save_path : Path, optional
            Path to save the visualization.
        show : bool
            Whether to display the visualization interactively.
        """
        if per == "slide" and slide_index is None:
            raise ValueError("For per='slide', you must specify slide_index.")
        if per == "global" and input_data is None:
            raise ValueError("For per='global', you must specify input_data.")

        # Select data and corresponding image
        if per == "slide":
            slide_data = self.data_loader.get_slide_data(set_name, slide_index, flatten=True)
            img_path = self.data_loader.root / Path(slide_data["screenshot_file"].iloc[0])
            data = slide_data
        else:
            all_data = self.data_loader.get_all_data(flatten=True)
            data = pd.concat(all_data.values(), ignore_index=True)
            data = data[data["input_data"] == input_data]

            # Use image path from any subject containing that input_data
            ref_data = self.data_loader.get_subject_data(set_name, flatten=True)
            ref_row = ref_data[ref_data["input_data"] == input_data].iloc[0]
            img_path = self.data_loader.root / Path(ref_row["screenshot_file"])

        if not img_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {img_path}")

        # Load image
        img = mpimg.imread(img_path)
        H, W = img.shape[:2]

        # --- Prepare gaze coordinates ---
        avg_x = data["avg_gaze_x"].dropna().values
        avg_y = data["avg_gaze_y"].dropna().values
        avg_x = W / 2 + avg_x
        avg_y = (H / 2 - avg_y) if flip_y else (H / 2 + avg_y)

        # --- Build heatmap ---
        heatmap, xedges, yedges = np.histogram2d(avg_x, avg_y, bins=bins, range=[[0, W], [0, H]])
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        # --- Plot overlay ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, origin="upper", extent=[0, W, H, 0])  # ensures proper orientation
        ax.imshow(
            heatmap.T,
            cmap=cmap,
            alpha=alpha,
            origin="upper",
            extent=[0, W, H, 0],
        )

        title = (
            f"Aggregated Heatmap \n{input_data}"
            if per == "global"
            else f"Heatmap \n{set_name}, Slide {slide_index}"
        )
        ax.set_title(title)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)



class FocusMapAnalyzer(HeatmapAnalyzer):
    """Highlights areas of high attention."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        data["attention_weight"] = data["avg_pupil_size"] / data["avg_pupil_size"].max()
        results = (
            data.groupby("slide_index")["attention_weight"]
            .mean()
            .reset_index()
            .rename(columns={"attention_weight": "focus_intensity"})
        )
        self.results = results
        self.data_loader.add_column("focus_intensity", results["focus_intensity"])
        return results

    def plot_analysis(self, set_name: str, slide_index: int, cmap: str = "plasma"):
        slide_data = self.data_loader.get_slide_data(set_name, slide_index, flatten=True)
        screenshot_path = Path(slide_data["screenshot_file"].iloc[0])
        img = mpimg.imread(self.data_loader.root / screenshot_path)
        H, W = img.shape[:2]

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.scatter(
            W / 2 + slide_data["avg_gaze_x"],
            H / 2 - slide_data["avg_gaze_y"],
            s=80 * slide_data["avg_pupil_size"].fillna(1),
            c=slide_data["avg_pupil_size"],
            cmap=cmap,
            alpha=0.6,
        )
        plt.title(f"Focus Map — {set_name}, Slide {slide_index}")
        plt.axis("off")
        plt.show()


class FixationAnalyzer(BaseAnalyzer):
    """Calculates fixation metrics: duration, count, dispersion."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True)

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
        for col in ["start_time", "end_time", "duration", "dispersion"]:
            self.data_loader.add_column(col, results[col])
        return results

    def plot_analysis(self, set_name: str, slide_index: int):
        slide_data = self.data_loader.get_slide_data(set_name, slide_index, flatten=True)
        screenshot_path = Path(slide_data["screenshot_file"].iloc[0])
        img = mpimg.imread(self.data_loader.root / screenshot_path)
        H, W = img.shape[:2]

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.scatter(
            W / 2 + slide_data["avg_gaze_x"],
            H / 2 - slide_data["avg_gaze_y"],
            s=slide_data["avg_pupil_size"] * 50,
            c="red",
            alpha=0.6,
        )
        plt.title(f"Fixations — {set_name}, Slide {slide_index}")
        plt.axis("off")
        plt.show()


class SaccadeAnalyzer(BaseAnalyzer):
    """Calculates saccade metrics: amplitude, velocity."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True).sort_values("system_time")
        data["dx"] = data["avg_gaze_x"].diff()
        data["dy"] = data["avg_gaze_y"].diff()
        data["dt"] = data["system_time"].diff().replace(0, np.nan)
        data["amplitude"] = np.sqrt(data["dx"] ** 2 + data["dy"] ** 2)
        data["velocity"] = data["amplitude"] / data["dt"]

        results = data[["system_time", "amplitude", "velocity"]].dropna().reset_index(drop=True)
        self.results = results
        self.data_loader.add_column("amplitude", results["amplitude"])
        self.data_loader.add_column("velocity", results["velocity"])
        return results

    def plot_analysis(self, set_name: str, slide_index: int):
        if self.results is None:
            raise ValueError("Run analyze() before plotting.")
        plt.figure(figsize=(10, 4))
        plt.plot(self.results["system_time"], self.results["velocity"])
        plt.title(f"Saccade Velocity over Time — {set_name}, Slide {slide_index}")
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.show()


class EntropyAnalyzer(BaseAnalyzer):
    """Calculates entropy of gaze distributions."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        hist, _, _ = np.histogram2d(data["avg_gaze_x"], data["avg_gaze_y"], bins=20)
        p = hist.flatten() / hist.sum()
        ent = entropy(p[p > 0])
        results = pd.DataFrame({"entropy": [ent]})
        self.results = results
        self.data_loader.add_column("entropy", results["entropy"])
        return results

    def plot_analysis(self, set_name: str, slide_index: int):
        plt.bar(["Entropy"], [self.results["entropy"].iloc[0]], color="orange")
        plt.title(f"Gaze Entropy — {set_name}, Slide {slide_index}")
        plt.ylabel("Entropy")
        plt.show()

class ClusterAnalyzer(BaseAnalyzer):
    """Cluster Analyzer
    Identifies gaze clusters using DBSCAN or a user-provided clustering model.

    Parameters
    ----------
    clustering_model : object, optional
        Any initialized clustering model implementing `fit_predict(X)` method.
        If None, defaults to DBSCAN with provided `eps` and `min_samples`.

    Example
    -------
    ```python
    from sklearn.cluster import DBSCAN, AgglomerativeClustering

    analyzer = ClusterAnalyzer(data_loader=loader, output_folder=Path("./out"))

    # Default DBSCAN
    results = analyzer.analyze(set_name="subject_01")

    # Custom clustering algorithm
    custom_model = AgglomerativeClustering(n_clusters=4)
    results = analyzer.analyze(set_name="subject_01", clustering_model=custom_model)
    ```
    """

    def analyze(
        self,
        set_name: str,
        clustering_model=None,
        eps: float = 0.05,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        # --- Load and prepare gaze data ---
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        if "avg_gaze_x" not in data or "avg_gaze_y" not in data:
            raise ValueError("Input data must contain 'avg_gaze_x' and 'avg_gaze_y' columns.")

        X = data[["avg_gaze_x", "avg_gaze_y"]].dropna().to_numpy()

        # --- Choose clustering model ---
        model = clustering_model or DBSCAN(eps=eps, min_samples=min_samples)

        # --- Fit model and assign cluster labels ---
        labels = model.fit_predict(X)
        valid_idx = data[["avg_gaze_x", "avg_gaze_y"]].dropna().index
        data.loc[valid_idx, "cluster_label"] = labels

        # --- Prepare and save results ---
        results = data[["system_time", "cluster_label"]].copy()
        self.results = results

        # Add results back to DataLoader
        self.data_loader.add_column("cluster_label", data["cluster_label"])

        return results

    def plot_analysis(
        self,
        set_name: str,
        color_map: str = "tab10",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """Visualize cluster assignments on gaze data."""
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        if "cluster_label" not in data:
            raise ValueError("No cluster_label column found. Run analyze() first.")

        df = data.dropna(subset=["avg_gaze_x", "avg_gaze_y", "cluster_label"])
        labels = df["cluster_label"].to_numpy()
        unique_labels = sorted(set(labels))

        cmap = cm.get_cmap(color_map, len(unique_labels))
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, label in enumerate(unique_labels):
            cluster_points = df[df["cluster_label"] == label]
            ax.scatter(
                cluster_points["avg_gaze_x"],
                cluster_points["avg_gaze_y"],
                s=30,
                alpha=0.7,
                color=cmap(i),
                label=f"Cluster {label}" if label != -1 else "Noise",
                edgecolors="none"
            )

        ax.set_title(f"Cluster Visualization: {set_name}")
        ax.set_xlabel("avg_gaze_x")
        ax.set_ylabel("avg_gaze_y")
        ax.legend(loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


class ConceptAnalyzer(BaseAnalyzer):
    """Defines AOI-like concepts from clusters."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        if "cluster_label" not in data.columns:
            raise ValueError("Cluster labels required. Run ClusterAnalyzer first.")

        concept_stats = (
            data.groupby("cluster_label")
            .agg(
                mean_duration=("system_time", lambda x: x.max() - x.min()),
                fixation_count=("event_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"cluster_label": "concept_id"})
        )
        self.results = concept_stats
        self.data_loader.add_column("concept_id", concept_stats["concept_id"])
        return concept_stats

    def plot_analysis(self, set_name: str, slide_index: int):
        plt.bar(self.results["concept_id"], self.results["mean_duration"], color="teal")
        plt.title(f"Concept Engagement — {set_name}, Slide {slide_index}")
        plt.xlabel("Concept ID")
        plt.ylabel("Mean Duration")
        plt.show()


class ScanpathsAnalyzer(BaseAnalyzer):
    """Analyzes sequential transitions between fixations."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True).sort_values("system_time")
        data["next_event"] = data["event_id"].shift(-1)
        transitions = data.groupby(["event_id", "next_event"]).size().reset_index(name="count")

        self.results = transitions
        self.data_loader.add_column("scanpath_transition_count", transitions["count"])
        return transitions

    def plot_analysis(self, set_name: str, slide_index: int):
        if self.results is None:
            raise ValueError("Run analyze() before plotting.")
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(self.results)), self.results["count"])
        plt.title(f"Scanpath Transition Counts — {set_name}, Slide {slide_index}")
        plt.xlabel("Transition Index")
        plt.ylabel("Count")
        plt.show()


class VoiceTranscription(BaseAnalyzer):
    """Transcribes and aligns voice data with gaze events."""

    def analyze(self, set_name: str) -> pd.DataFrame:
        data = self.data_loader.get_subject_data(set_name, flatten=True)
        transcripts = pd.DataFrame({
            "system_time": data["system_time"],
            "transcribed_text": ["dummy transcription"] * len(data)
        })
        self.results = transcripts
        self.data_loader.add_column("transcribed_text", transcripts["transcribed_text"])
        return transcripts

    def plot_analysis(self, set_name: str, slide_index: int):
        if self.results is None:
            raise ValueError("Run analyze() before plotting.")
        plt.figure(figsize=(8, 4))
        plt.text(0.1, 0.5, "\n".join(self.results["transcribed_text"].head(5)), fontsize=12)
        plt.axis("off")
        plt.title(f"Voice Transcription (sample) — {set_name}, Slide {slide_index}")
        plt.show()
