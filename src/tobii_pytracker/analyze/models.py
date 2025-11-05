
from pathlib import Path
from typing import  Optional, Any
from .data_loader import DataLoader


class BaseAnalyzer:
    def __init__(self, data_loader: DataLoader, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.data_loader = data_loader
        self.results = None

    def analyze(self, *args, **kwargs) -> Any:
        """Run full analysis (override in subclass)."""
        raise NotImplementedError
    

    def save_results(self, filename: Optional[str] = None):
        """Explicitly save results to disk."""
        if self.results is None:
            return

        filename = filename or f"{self.__class__.__name__}_results.json"
        filepath = self.output_folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

class HeatmapAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent



class FocusMapAnalyzer(HeatmapAnalyzer):
    def __init__(self, parent):
        self.parent = parent



class FixationAnalyzer:
    def __init__(self, parent):
        self.parent = parent



class SaccadeAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent



class EntropyAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent


class ClusterAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent

class ConceptAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent

class ScanpathsAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent


class VoiceTranscription(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent



