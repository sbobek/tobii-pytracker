from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass
from tobii_pytracker.configs.custom_config import CustomConfig
import json
from tobii_pytracker.utils.custom_logger import CustomLogger

LOGGER = CustomLogger("INFO", __name__).logger

@dataclass
class GazePoint:
    x: Optional[float]
    y: Optional[float]
    pupil_size: Optional[float]  # renamed from 'event'
    timestamp: float


class DataLoader:
    """
    Main analysis orchestrator.
    Handles loading data from each participant’s set folder (set_name),
    and delegates work to specialized analyzers.
    """

    def __init__(self, config: CustomConfig, root: Optional[Path] = None):
        self.config = config
        self.root = root if root else Path('./')
        self.output_root = (
            root / Path(config.get_output_config()["folder"])
            if root
            else Path(config.get_output_config()["folder"])
        )
        if not self.output_root.exists():
            raise FileNotFoundError(f"Output directory '{self.output_root}' not found.")
        self.subjects = self._discover_subjects()

    # ======================================================
    # DATA LOADING
    # ======================================================
    def _discover_subjects(self) -> List[str]:
        """Find all participant set directories."""
        return [p.name for p in self.output_root.iterdir() if p.is_dir()]

    def _load_data(self, set_name: str) -> pd.DataFrame:
        """Load data.csv for one participant."""
        data_csv = self.output_root / set_name / "data.csv"
        if not data_csv.exists():
            raise FileNotFoundError(f"Missing data.csv for '{set_name}'")
        return pd.read_csv(data_csv, sep=";")

    def _parse_gaze_data(self, gaze_field):
        """
        Parse gaze_data field (string or list) into a list of dicts.

        Handles cases where gaze_data is stored as a stringified list of dicts.
        """
        if isinstance(gaze_field, list):
            return gaze_field

        if isinstance(gaze_field, str):
            try:
                # Try JSON first (if it's JSON formatted)
                return json.loads(gaze_field)
            except json.JSONDecodeError:
                try:
                    # Fall back to Python literal (ast.literal_eval for safety)
                    return ast.literal_eval(gaze_field)
                except Exception:
                    LOGGER.warning("Failed to parse gaze_data field; returning empty list.")
                    return []

        return []

    
    def _flatten_gaze_data(self, df: pd.DataFrame, set_name: str) -> pd.DataFrame:
        """Flatten new gaze data (dict-based) from all slides into one long DataFrame, 
        including per-eye and averaged gaze/pupil values."""
        records = []

        for i, row in df.iterrows():
            gaze_points = self._parse_gaze_data(row.get("gaze_data", []))
            if not gaze_points:
                continue

            for g in gaze_points:
                gaze_x_left = g.get("gaze_x_left")
                gaze_y_left = g.get("gaze_y_left")
                gaze_x_right = g.get("gaze_x_right")
                gaze_y_right = g.get("gaze_y_right")
                pupil_left = g.get("pupil_left")
                pupil_right = g.get("pupil_right")

                # --- Compute averages only if both values are present ---
                if (gaze_x_left is not None) and (gaze_x_right is not None):
                    avg_gaze_x = (gaze_x_left + gaze_x_right) / 2
                elif gaze_x_left is not None:
                    avg_gaze_x = gaze_x_left
                elif gaze_x_right is not None:
                    avg_gaze_x = gaze_x_right
                else:
                    avg_gaze_x = None

                if (gaze_y_left is not None) and (gaze_y_right is not None):
                    avg_gaze_y = (gaze_y_left + gaze_y_right) / 2
                elif gaze_y_left is not None:
                    avg_gaze_y = gaze_y_left
                elif gaze_y_right is not None:
                    avg_gaze_y = gaze_y_right
                else:
                    avg_gaze_y = None

                if (pupil_left is not None) and (pupil_right is not None):
                    avg_pupil_size = (pupil_left + pupil_right) / 2
                elif pupil_left is not None:
                    avg_pupil_size = pupil_left
                elif pupil_right is not None:
                    avg_pupil_size = pupil_right
                else:
                    avg_pupil_size = None

                # --- Collect record ---
                records.append({
                    "set_name": set_name,
                    "slide_index": i,
                    "screenshot_file": row.get("screenshot_file"),
                    "input_data": row.get("input_data"),
                    "classification": row.get("classification"),
                    "user_classification": row.get("user_classification"),
                    "model_prediction": row.get("model_prediction"),
                    "voice_file": row.get("voice_file"),
                    "voice_start_timestamp": row.get("voice_start_timestamp"),

                    # Raw gaze data
                    "event_type": g.get("event_type"),
                    "event_id": g.get("event_id"),
                    "logged_time": g.get("logged_time"),
                    "system_time": g.get("system_time"),

                    # Eye-specific data
                    "gaze_x_left": gaze_x_left,
                    "gaze_y_left": gaze_y_left,
                    "gaze_x_right": gaze_x_right,
                    "gaze_y_right": gaze_y_right,
                    "pupil_left": pupil_left,
                    "pupil_right": pupil_right,

                    # Averaged data
                    "avg_gaze_x": avg_gaze_x,
                    "avg_gaze_y": avg_gaze_y,
                    "avg_pupil_size": avg_pupil_size,
                })

        return pd.DataFrame(records)



    # ======================================================
    # DATA ACCESS
    # ======================================================
    def get_slide_data(self, set_name: str, index: int, flatten: bool = False) -> Any:
        """
        Return all information for a single slide.
        If flatten=True, returns a DataFrame of gaze points (via _flatten_gaze_data()).
        """
        df = self._load_data(set_name)
        if index >= len(df):
            raise IndexError(f"Index {index} out of range for {set_name}")

        row = df.iloc[index]
        gaze_data = row.get("gaze_data")

        # Ensure gaze_data is parsed if not already a list/dict
        if isinstance(gaze_data, str):
            gaze_data = self._parse_gaze_data(gaze_data)

        if flatten:
            # Reuse flatten logic — wrap row into a single-row DataFrame
            single_df = pd.DataFrame([row])
            return self._flatten_gaze_data(single_df, set_name=set_name)

        # Non-flattened version
        return {
            "set_name": set_name,
            "screenshot_path": self.root / Path(row["screenshot_file"]),
            "voice_file": (
                self.root / Path(row["voice_file"])
                if pd.notna(row.get("voice_file"))
                else None
            ),
            "voice_start_timestamp": row.get("voice_start_timestamp"),
            "gaze_data": gaze_data,
            "metadata": row.to_dict(),
        }


    def get_subject_data(self, set_name: str, flatten: bool = False) -> pd.DataFrame:
        """
        Return the full DataFrame for one subject.
        If flatten=True, expands gaze data into individual rows.
        """
        df = self._load_data(set_name)
        if flatten:
            return self._flatten_gaze_data(df, set_name)
        return df

    def get_all_data(self, flatten: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Return all subjects’ data.
        If flatten=True, returns flattened gaze data for each subject.
        """
        data = {}
        for s in self.subjects:
            df = self._load_data(s)
            data[s] = self._flatten_gaze_data(df, s) if flatten else df
        return data

    def get_subjects(self) -> List[str]:
        """Return list of all discovered subjects."""
        return self.subjects
    
    # ======================================================
    # DATA MANIPULATION
    # ======================================================
    def add_column(
        self,
        column_name: str,
        value: Any = None,
        func: Optional[callable] = None,
        subjects: Optional[List[str]] = None,
        overwrite: bool = False,
        save: bool = True,
    ):
        """
        Add a new column to one or more subjects' data.csv files.

        Parameters
        ----------
        column_name : str
            Name of the new column to add.
        value : Any, optional
            Constant value to assign to all rows (ignored if func is provided).
        func : callable, optional
            A function that takes a DataFrame and returns a Series or list
            of values to populate the new column (e.g. lambda df: df['x'] + df['y']).
        subjects : list of str, optional
            List of subject names to modify. If None, applies to all subjects.
        overwrite : bool, optional
            If False (default), raises an error if the column already exists.
        save : bool, optional
            If True, overwrites the modified data.csv on disk.
        """
        target_subjects = subjects if subjects else self.subjects

        for s in target_subjects:
            data_path = self.output_root / s / "data.csv"
            if not data_path.exists():
                print(f"⚠️ Skipping {s}: data.csv not found.")
                continue

            df = pd.read_csv(data_path, sep=";")

            # Check for overwrite
            if column_name in df.columns and not overwrite:
                continue

            # Compute or assign column
            if func is not None:
                try:
                    df[column_name] = func(df)
                except Exception as e:
                    continue
            else:
                df[column_name] = value

            # Save or return
            if save:
                df.to_csv(data_path, sep=";", index=False)


    # ======================================================
    # PLOTTING
    # ======================================================
    def plot_gaze(
        self,
        set_name: str,
        index: int,
        size: int = 40,
        alpha: float = 0.6,
        color: str = "red",
        save_path: Optional[Path] = None,
        show: bool = True,
        flip_y: bool = True,
        gradient: bool = False,
        cmap: str = "viridis",
        draw_both_eyes: bool = False,
        show_legend: bool = False,
    ):
        """
        Plot gaze points for a specified slide on its screenshot image.

        By default, draws average gaze positions (avg_gaze_x, avg_gaze_y).
        If draw_both_eyes=True, also plots separate left (L) and right (R) gaze traces.

        Parameters
        ----------
        set_name : str
            Subject folder name.
        index : int
            Slide index in data.csv.
        size : int, optional
            Marker size for gaze points.
        alpha : float, optional
            Transparency of gaze markers.
        color : str, optional
            Uniform color (ignored if gradient=True).
        save_path : Path, optional
            If given, saves the plot to this path.
        show : bool, optional
            Whether to display the plot interactively.
        flip_y : bool, optional
            Whether to invert Y axis (default True).
        gradient : bool, optional
            If True, color points along a colormap based on time/order.
        cmap : str, optional
            Colormap to use when gradient=True.
        draw_both_eyes : bool, optional
            If True, also plot separate left/right gaze traces (default False).
        show_legend : bool, optional
            If True, shows a legend for the gaze traces (default False).
        """
        slide_data = self.get_slide_data(set_name, index, flatten=True)
        screenshot_path = self.root / Path(slide_data["screenshot_file"].iloc[0])

        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        # Load the screenshot
        img = mpimg.imread(screenshot_path)
        H, W = img.shape[:2]

        # Prepare figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)

        # --- Plot average gaze ---
        if "avg_gaze_x" in slide_data.columns:
            avg_x = slide_data["avg_gaze_x"].dropna().values
            avg_y = slide_data["avg_gaze_y"].dropna().values
            ts = slide_data["system_time"].values

            if flip_y:
                avg_y = H / 2 - avg_y
            else:
                avg_y = H / 2 + avg_y
            avg_x = W / 2 + avg_x

            if len(avg_x) > 0:
                if gradient:
                    ts_min, ts_max = min(ts), max(ts)
                    norm = [(t - ts_min) / (ts_max - ts_min) if ts_max > ts_min else 0.5 for t in ts]
                    scatter = ax.scatter(
                        avg_x, avg_y, s=size, c=norm, cmap=cmap, alpha=alpha, edgecolors="none", label="Avg gaze"
                    )
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Relative time", rotation=270, labelpad=15)
                else:
                    ax.scatter(avg_x, avg_y, s=size, c=color, alpha=alpha, edgecolors="none", label="Avg gaze")

        # --- Optionally draw both eyes separately ---
        if draw_both_eyes:
            for eye, label, c in [
                ("left", "L", "cyan"),
                ("right", "R", "magenta")
            ]:
                gx_col = f"gaze_x_{eye}"
                gy_col = f"gaze_y_{eye}"
                if gx_col not in slide_data or gy_col not in slide_data:
                    continue

                gx = slide_data[gx_col].dropna().values
                gy = slide_data[gy_col].dropna().values
                if len(gx) == 0:
                    continue

                if flip_y:
                    gy = H / 2 - gy
                else:
                    gy = H / 2 + gy
                gx = W / 2 + gx

                ax.scatter(gx, gy, s=size * 0.6, c=c, alpha=alpha * 0.7, edgecolors="none", label=f"{label}-eye")

                # Label first few samples
                for i in range(min(3, len(gx))):
                    ax.text(gx[i], gy[i], label, color="white", fontsize=10, weight="bold")

        ax.set_title(f"Gaze points: {set_name} (slide {index})", fontsize=12)
        ax.axis("off")
        if show_legend:
            ax.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

