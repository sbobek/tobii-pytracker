from .base_analyzer import BaseAnalyzer

from .base_analyzer import BaseAnalyzer
from typing import Any
import pandas as pd

class ClusterAnalyzer(BaseAnalyzer):
    def __init__(self, config, output_folder, clustering_model, data_csv=None):
        super().__init__(config, output_folder, data_csv)
        self.clustering_model = clustering_model

    def analyze(self):
        df = pd.read_csv(self.data_csv)
        self.results = self._cluster_points(df)
        return self.results

    def analyze_one(self, index: int):
        df = pd.read_csv(self.data_csv)
        return self._cluster_points(df.iloc[[index]])

    def analyze_subject(self, set_name: str):
        data_csv = self.output_folder / set_name / "data.csv"
        df = pd.read_csv(data_csv)
        self.results = self._cluster_points(df)
        return self.results

    def _cluster_points(self, df: pd.DataFrame) -> Any:
        points = df[["x", "y"]].dropna()
        return self.clustering_model.fit_predict(points)