from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import ast

from tobii_pytracker.configs.custom_config import CustomConfig
from .base_analyzer import GazePoint
from .classic_analyzers import FixationAnalyzer,HeatmapAnalyzer,FocusMapAnalyzer
from .cluster_analyzer import ClusterAnalyzer
from .concept_analyzer import ConceptAnalyzer
from .transcript_analyzer import TranscriptAnalyzer


class Analyzer:
    """
    Main analysis orchestrator.
    Handles loading data from each participant’s set folder (set_name),
    and delegates work to specialized analyzers.
    """

    def __init__(self, config: CustomConfig, root: Optional[Path] = None):
        self.config = config
        self.output_root = root / Path(config.get_output_config()["folder"]) if root else Path(config.get_output_config()["folder"])
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

    def _parse_gaze_data(self, gaze_str: str) -> List[GazePoint]:
        """Parse gaze data string safely into GazePoint list."""
        try:
            gaze_tuples = ast.literal_eval(gaze_str)
            return [
                GazePoint(x=g[0][0], y=g[0][1], pupil_size=g[1], timestamp=g[2])
                for g in gaze_tuples if g[0] != (None, None)
            ]
        except Exception:
            return []

    # ======================================================
    # DATA ACCESS
    # ======================================================
    def get_slide_data(self, set_name: str, index: int) -> Dict[str, Any]:
        """Return all information for a single slide."""
        df = self._load_data(set_name)
        if index >= len(df):
            raise IndexError(f"Index {index} out of range for {set_name}")

        row = df.iloc[index]
        gaze_data = self._parse_gaze_data(row["gaze_data"])

        return {
            "set_name": set_name,
            "screenshot_path": self.output_root / set_name / row["screenshot_file"],
            "voice_file": (
                self.output_root / set_name / row["voice_file"]
                if pd.notna(row.get("voice_file"))
                else None
            ),
            "voice_start_timestamp": row.get("voice_start_timestamp"),
            "gaze_data": gaze_data,
            "metadata": row.to_dict(),
        }

    def get_subject_data(self, set_name: str) -> pd.DataFrame:
        """Return the full DataFrame for one subject."""
        return self._load_data(set_name)

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Return all subjects’ data as a dict."""
        return {s: self._load_data(s) for s in self.subjects}
    
    def get_subjects(self) -> List[str]:
        """Return list of all discovered subjects."""
        return self.subjects

    # ======================================================
    # ANALYSIS WRAPPERS
    # ======================================================
    def analyze_one(self, set_name: str, slide_index: int) -> Dict[str, Any]:
        """Run all analyzers on a single slide."""
        slide_data = self.get_slide_data(set_name, slide_index)
        data_csv = self.output_root / set_name / "data.csv"

        fixation = FixationAnalyzer(self.config, self.output_root / set_name, data_csv)
        cluster = ClusterAnalyzer(self.config, self.output_root / set_name, data_csv)
        concept = ConceptAnalyzer(self.config, self.output_root / set_name, backbone="resnet", data_csv=data_csv)
        transcript = TranscriptAnalyzer(self.config, self.output_root / set_name, data_csv=data_csv)
        heatmap = HeatmapAnalyzer(self.config, self.output_root / set_name, data_csv=data_csv)
        focusmap = FocusMapAnalyzer(self.config, self.output_root / set_name, data_csv=data_csv)

        return {
            "fixations": fixation.analyze_one(slide_index),
            "clusters": cluster.analyze_one(slide_index),
            "concepts": concept.analyze_one(slide_index),
            "transcripts": transcript.analyze_one(slide_index),
            "heatmap": heatmap.analyze_one(slide_index),
            "focusmap": focusmap.analyze_one(slide_index),
        }

    def analyze_subject(self, set_name: str) -> Dict[str, Any]:
        """Run all analyzers for all slides in one subject folder."""
        data_csv = self.output_root / set_name / "data.csv"
        fixation = FixationAnalyzer(self.config, self.output_root / set_name, data_csv)
        cluster = ClusterAnalyzer(self.config, self.output_root / set_name, data_csv)
        concept = ConceptAnalyzer(self.config, self.output_root / set_name, backbone="resnet", data_csv=data_csv)
        transcript = TranscriptAnalyzer(self.config, self.output_root / set_name, data_csv)

        return {
            "fixations": fixation.analyze_subject(set_name),
            "clusters": cluster.analyze_subject(set_name),
            "concepts": concept.analyze_subject(set_name),
            "transcripts": transcript.analyze_subject(set_name),
        }

    def analyze_all(self) -> Dict[str, Any]:
        """Run analyzers across all participants."""
        results = {}
        for s in self.subjects:
            print(f"[INFO] Analyzing subject: {s}")
            results[s] = self.analyze_subject(s)
        return results
