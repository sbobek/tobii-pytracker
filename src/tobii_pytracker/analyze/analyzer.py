from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from .data_structures import GazePoint
from .fixation_analyzer import FixationAnalyzer
from .classic_analyzers import SaccadeAnalyzer
from .classic_analyzers import EntropyAnalyzer
from .cluster_analyzer import ClusterAnalyzer
from .concept_analyzer import ConceptAnalyzer
from .classic_analyzers import HeatmapAnalyzer
from .transcript_analyzer import TranscriptAnalyzer
from .process_mining_analyzer import ProcessMiningAnalyzer
from .config import CustomConfig

import ast


class Analyzer:
    def __init__(self, config: CustomConfig):
        self.config = config
        self.output_dir = Path(config.output_dir or "output")
        self.data_csv = self.output_dir / "data.csv"
        self.data = self._load_data()

        # Factory-like analyzers
        self.fixation = FixationAnalyzer(self)
        self.saccade = SaccadeAnalyzer(self)
        self.entropy = EntropyAnalyzer(self)
        self.cluster = ClusterAnalyzer(self)
        self.concept = ConceptAnalyzer(self)
        self.heatmap = HeatmapAnalyzer(self)
        self.transcript = TranscriptAnalyzer(self)
        self.process_mining = ProcessMiningAnalyzer(self)

    def _load_data(self) -> pd.DataFrame:
        if not self.data_csv.exists():
            raise FileNotFoundError(f"data.csv not found in {self.output_dir}")
        return pd.read_csv(self.data_csv, sep=';')

    def get_slide_data(self, index: int) -> Dict[str, Any]:
        """Returns all relevant data for a single slide."""
        row = self.data.iloc[index]
        gaze_data = self._parse_gaze_data(row['gaze_data'])
        return {
            'screenshot_path': Path(row['screenshot_file']),
            'gaze_data': gaze_data,
            'voice_file': Path(row['voice_file']) if pd.notna(row['voice_file']) else None,
            'voice_start_timestamp': row.get('voice_start_timestamp'),
            'metadata': row.to_dict()
        }

    def _parse_gaze_data(self, gaze_str: str) -> List[GazePoint]:
        try:
            gaze_tuples = ast.literal_eval(gaze_str)
            return [GazePoint(x=g[0][0], y=g[0][1], pupil_size=g[1], timestamp=g[2]) for g in gaze_tuples]
        except Exception as e:
            print(f"Failed to parse gaze data: {e}")
            return []

    def list_slides(self) -> List[str]:
        return [Path(f).stem for f in self.data['screenshot_file']]
