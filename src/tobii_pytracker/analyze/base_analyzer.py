from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import ast
from dataclasses import dataclass
from tobii_pytracker.configs.custom_config import CustomConfig


@dataclass
class GazePoint:
    x: Optional[float]
    y: Optional[float]
    pupil_size: Optional[float]
    timestamp: float


class BaseAnalyzer:
    def __init__(self, config, output_folder: Path, data_csv: Optional[Path] = None):
        self.config = config
        self.output_folder = Path(output_folder)
        self.data_csv = data_csv or (self.output_folder / "data.csv")
        self.results = None

    def analyze(self, *args, **kwargs) -> Any:
        """Run full analysis (override in subclass)."""
        raise NotImplementedError

    def analyze_one(self, index: int):
        """Analyze one slide (override in subclass)."""
        raise NotImplementedError

    def analyze_subject(self, set_name: str):
        """Run analysis for all slides in one subject."""
        raise NotImplementedError

    def save_results(self, filename: Optional[str] = None):
        """Explicitly save results to disk."""
        if self.results is None:
            print(f"No results to save for {self.__class__.__name__}.")
            return

        filename = filename or f"{self.__class__.__name__}_results.json"
        filepath = self.output_folder / filename
        print(f"[INFO] Saving {self.__class__.__name__} results to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

