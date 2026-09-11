import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.analyze.models import (
    FixationAnalyzer,
    SaccadeAnalyzer,
    EntropyAnalyzer,
    BBoxAttentionAnalyzer,
    ClusterAnalyzer,
)


class TestFixationAnalyzerCoverage(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = Path(self.temp_dir)
    
    def test_fixation_analyzer_dispersion_method(self):
        
        analyzer = FixationAnalyzer(
            self.output_folder,
            method="dispersion",
            dispersion_threshold=30.0,
            min_duration=0.05
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": [100, 100.1, 100.2, 100.3, 200, 200.1, 200.2],
            "avg_gaze_y": [150, 150.1, 150.2, 150.3, 250, 250.1, 250.2],
            "system_time": [0, 16.67, 33.33, 50, 66.67, 83.33, 100],
            "set_name": ["s1"] * 7,
            "slide_index": [0] * 7,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_fixation_analyzer_velocity_method(self):
        
        analyzer = FixationAnalyzer(
            self.output_folder,
            method="velocity",
            min_duration=0.05,
            velocity_threshold=10
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": [100, 110, 120, 130, 140, 150],
            "avg_gaze_y": [150, 160, 170, 180, 190, 200],
            "system_time": [0, 16.67, 33.33, 50, 66.67, 83.33],
            "set_name": ["s1"] * 6,
            "slide_index": [0] * 6,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_fixation_analyzer_velocity_with_different_threshold(self):
        
        analyzer = FixationAnalyzer(
            self.output_folder,
            method="velocity",
            min_duration=0.1,
            velocity_threshold=50
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": np.linspace(100, 200, 20),
            "avg_gaze_y": np.linspace(150, 250, 20),
            "system_time": np.linspace(0, 316.67, 20),
            "set_name": ["s1"] * 20,
            "slide_index": [0] * 20,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_fixation_analyzer_with_multiple_sets(self):
        
        analyzer = FixationAnalyzer(self.output_folder, method="dispersion")
        
        df = pd.DataFrame({
            "avg_gaze_x": [100, 100.1, 100.2, 200, 200.1, 200.2] * 2,
            "avg_gaze_y": [150, 150.1, 150.2, 250, 250.1, 250.2] * 2,
            "system_time": list(range(33, 200, 33)) * 2,
            "set_name": ["s1"] * 6 + ["s2"] * 6,
            "slide_index": [0] * 12,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_fixation_analyzer_empty_dataframe(self):
        
        analyzer = FixationAnalyzer(self.output_folder)
        df = pd.DataFrame({
            "avg_gaze_x": [],
            "avg_gaze_y": [],
            "timestamp": [],
            "set_name": [],
            "slide_index": [],
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_fixation_analyzer_single_point(self):
        
        analyzer = FixationAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": [100],
            "avg_gaze_y": [150],
            "system_time": [0],
            "set_name": ["s1"],
            "slide_index": [0],
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)


class TestSaccadeAnalyzerCoverage(unittest.TestCase):
    
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = Path(self.temp_dir)
    
    def test_saccade_analyzer_acceleration_method(self):
        
        analyzer = SaccadeAnalyzer(
            self.output_folder,
            method="acceleration",
            min_duration=0.01
        )
        
        # Create data with acceleration peaks
        x = np.array([100 + i for i in range(50)])
        y = np.array([150 + i*0.5 for i in range(50)])
        
        df = pd.DataFrame({
            "avg_gaze_x": x,
            "avg_gaze_y": y,
            "system_time": np.linspace(0, 830, 50),
            "set_name": ["s1"] * 50,
            "slide_index": [0] * 50,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_saccade_analyzer_ivt_method_with_threshold(self):
        
        analyzer = SaccadeAnalyzer(
            self.output_folder,
            method="ivt",
            velocity_threshold=50
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": [100, 105, 110, 200, 205, 210, 100, 105],
            "avg_gaze_y": [150, 155, 160, 250, 255, 260, 150, 155],
            "system_time": [0, 16.67, 33.33, 50, 66.67, 83.33, 100, 116.67],
            "set_name": ["s1"] * 8,
            "slide_index": [0] * 8,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_saccade_analyzer_multiple_slides(self):
        
        analyzer = SaccadeAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": [110, 1010, 120, 200, 2210, 220] * 2,
            "avg_gaze_y": [110, 1660, 170, 2560, 260, 2670] * 2,
            "system_time": list(range(4, 94, 15)) * 2,
            "set_name": ["s1"] * 6 + ["s1"] * 6,
            "slide_index": [0] * 6 + [1] * 6,
            "is_saccade": [True, True, False, True, True, True] * 2
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_saccade_analyzer_with_large_dataset(self):
        
        analyzer = SaccadeAnalyzer(self.output_folder)
        
        n = 200
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, n),
            "avg_gaze_y": np.random.uniform(150, 600, n),
            "system_time": np.linspace(0, 3330, n),
            "set_name": ["s1"] * n,
            "slide_index": [0] * n,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_saccade_analyzer_with_micro_saccade_filter(self):
        
        analyzer = SaccadeAnalyzer(
            self.output_folder,
            filter_micro_saccades=True,
            micro_saccade_threshold=25
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, 80),
            "avg_gaze_y": np.random.uniform(150, 600, 80),
            "system_time": np.linspace(0, 1330, 80),
            "set_name": ["s1"] * 80,
            "slide_index": [0] * 80,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)


class TestEntropyAnalyzerCoverage(unittest.TestCase):
    
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = Path(self.temp_dir)
    
    def test_entropy_analyzer_global_mode_custom_bins(self):
        
        analyzer = EntropyAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, 50),
            "avg_gaze_y": np.random.uniform(150, 600, 50),
            "timestamp": np.linspace(0, 830, 50),
            "set_name": ["s1"] * 50,
            "slide_index": [0] * 50,
        })
        
        result = analyzer.analyze(df, per="global", bins=50)
        self.assertIsNotNone(result)
    
    def test_entropy_analyzer_per_set_mode(self):
        
        analyzer = EntropyAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": list(np.random.uniform(100, 300, 20)) + list(np.random.uniform(300, 500, 20)),
            "avg_gaze_y": list(np.random.uniform(150, 350, 20)) + list(np.random.uniform(350, 600, 20)),
            "timestamp": list(np.linspace(0, 330, 20)) + list(np.linspace(330, 660, 20)),
            "set_name": ["s1"] * 20 + ["s2"] * 20,
            "slide_index": [0] * 40,
        })
        
        result = analyzer.analyze(df, per="set")
        self.assertIsNotNone(result)
    
    def test_entropy_analyzer_per_slide_mode(self):
        
        analyzer = EntropyAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": list(np.random.uniform(100, 300, 20)) + list(np.random.uniform(300, 500, 20)),
            "avg_gaze_y": list(np.random.uniform(150, 350, 20)) + list(np.random.uniform(350, 600, 20)),
            "timestamp": list(np.linspace(0, 330, 20)) + list(np.linspace(330, 660, 20)),
            "set_name": ["s1"] * 40,
            "slide_index": [0] * 20 + [1] * 20,
        })
        
        result = analyzer.analyze(df, per="slide")
        self.assertIsNotNone(result)
    
    def test_entropy_analyzer_without_convex_hull(self):
        
        analyzer = EntropyAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, 50),
            "avg_gaze_y": np.random.uniform(150, 600, 50),
            "timestamp": np.linspace(0, 830, 50),
            "set_name": ["s1"] * 50,
            "slide_index": [0] * 50,
        })
        
        result = analyzer.analyze(df, use_convex_hull=False)
        self.assertIsNotNone(result)
    
    def test_entropy_analyzer_small_data(self):
        
        analyzer = EntropyAnalyzer(self.output_folder)
        
        df = pd.DataFrame({
            "avg_gaze_x": [100, 101, 102],
            "avg_gaze_y": [150, 151, 152],
            "timestamp": [0, 16.67, 33.33],
            "set_name": ["s1"] * 3,
            "slide_index": [0] * 3,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)


class TestBBoxAttentionAnalyzerCoverage(unittest.TestCase):
    
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = Path(self.temp_dir)
    
    def test_bbox_analyzer_initialization(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        self.assertIsNotNone(analyzer)
    
    def test_bbox_analyzer_analyze_basic(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        # Create minimal raw data
        raw_data = pd.DataFrame({
            "set_name": ["test_set"],
            "slide_index": [0],
            "objects_bboxes": [{}],
        })
        
        # Create gaze data
        gaze_data = pd.DataFrame({
            "avg_gaze_x": [100, 110, 120],
            "avg_gaze_y": [150, 160, 170],
            "timestamp": [0, 16.67, 33.33],
            "set_name": ["test_set"] * 3,
            "slide_index": [0] * 3,
        })
        
        result = analyzer.analyze(raw_data, gaze_data)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_with_polygon_bboxes(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        # BBox with polygon coordinates
        polygon = [[100, 150], [200, 150], [200, 250], [100, 250]]
        bbox_record = {
            "bbox": {"x": 100, "y": 150, "w": 100, "h": 100},
            "polygon": polygon,
        }
        
        raw_data = pd.DataFrame({
            "set_name": ["test_set"],
            "slide_index": [0],
            "objects_bboxes": [{"image_bboxes": [bbox_record]}],
        })
        
        gaze_data = pd.DataFrame({
            "avg_gaze_x": [110, 120, 130],
            "avg_gaze_y": [160, 170, 180],
            "timestamp": [0, 16.67, 33.33],
            "set_name": ["test_set"] * 3,
            "slide_index": [0] * 3,
        })
        
        result = analyzer.analyze(raw_data, gaze_data)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_with_rect_bbox(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        # BBox with centered format
        bbox_record = {
            "bbox": {"cx": 150, "cy": 200, "w": 100, "h": 100},
            "rect_bbox": {"x": 100, "y": 150, "width": 100, "height": 100},
        }
        
        raw_data = pd.DataFrame({
            "set_name": ["test_set"],
            "slide_index": [0],
            "objects_bboxes": [{"image_bboxes": [bbox_record]}],
        })
        
        gaze_data = pd.DataFrame({
            "avg_gaze_x": [110, 120, 130, 140, 150],
            "avg_gaze_y": [160, 170, 180, 190, 200],
            "timestamp": [0, 16.67, 33.33, 50, 66.67],
            "set_name": ["test_set"] * 5,
            "slide_index": [0] * 5,
        })
        
        result = analyzer.analyze(raw_data, gaze_data)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_with_numeric_slide_index(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        raw_data = pd.DataFrame({
            "set_name": ["test_set", "test_set"],
            "slide_index": ["0", "1"],
            "objects_bboxes": [{}, {}],
        })
        
        gaze_data = pd.DataFrame({
            "avg_gaze_x": [100, 110, 120, 200, 210],
            "avg_gaze_y": [150, 160, 170, 250, 260],
            "timestamp": [0, 16.67, 33.33, 50, 66.67],
            "set_name": ["test_set"] * 5,
            "slide_index": ["0", "0", "0", "1", "1"],
        })
        
        result = analyzer.analyze(raw_data, gaze_data)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_with_fixations(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        raw_data = pd.DataFrame({
            "set_name": ["test_set"],
            "slide_index": [0],
            "objects_bboxes": [{}],
        })
        
        # Fixation format data
        gaze_data = pd.DataFrame({
            "x_mean": [110, 120, 130],
            "y_mean": [160, 170, 180],
            "duration": [0.1, 0.15, 0.2],
            "timestamp": [0, 16.67, 33.33],
            "set_name": ["test_set"] * 3,
            "slide_index": [0] * 3,
        })
        
        result = analyzer.analyze(raw_data, gaze_data, use_fixations=True)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_with_empty_gaze_data(self):
        
        analyzer = BBoxAttentionAnalyzer(self.output_folder)
        
        raw_data = pd.DataFrame({
            "set_name": ["test_set"],
            "slide_index": [0],
            "objects_bboxes": [{}],
        })
        
        gaze_data = pd.DataFrame({
            "avg_gaze_x": [],
            "avg_gaze_y": [],
            "timestamp": [],
            "set_name": [],
            "slide_index": [],
        })
        
        result = analyzer.analyze(raw_data, gaze_data)
        self.assertIsNotNone(result)
    
    def test_bbox_analyzer_normalize_slide_index(self):
        
        df = pd.DataFrame({
            "slide_index": ["0", "1", "2"],
            "value": [1, 2, 3],
        })
        
        normalized = BBoxAttentionAnalyzer._normalize_slide_index_column(df)
        self.assertIn("slide_index", normalized.columns)
    
    def test_bbox_analyzer_filter_set_and_slide(self):
        
        df = pd.DataFrame({
            "set_name": ["s1", "s1", "s2", "s2"],
            "slide_index": [0, 1, 0, 1],
            "value": [1, 2, 3, 4],
        })
        
        filtered = BBoxAttentionAnalyzer._filter_set_and_slide(
            df, set_name="s1", slide_index=0
        )
        self.assertEqual(len(filtered), 1)
    
    def test_bbox_analyzer_resolve_gaze_columns(self):
        
        x, y, dur = BBoxAttentionAnalyzer._resolve_gaze_columns(use_fixations=True)
        self.assertEqual((x, y, dur), ("x_mean", "y_mean", "duration"))
        
        x, y, dur = BBoxAttentionAnalyzer._resolve_gaze_columns(use_fixations=False)
        self.assertEqual((x, y, dur), ("avg_gaze_x", "avg_gaze_y", None))


class TestClusterAnalyzerCoverage(unittest.TestCase):
    
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = Path(self.temp_dir)
    
    def test_cluster_analyzer_initialization(self):
        
        analyzer = ClusterAnalyzer(self.output_folder)
        self.assertIsNotNone(analyzer)
    
    def test_cluster_analyzer_with_kmeans(self):
        
        analyzer = ClusterAnalyzer(
            self.output_folder,
            clustering_model="kmeans",
            n_clusters=3
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, 50),
            "avg_gaze_y": np.random.uniform(150, 600, 50),
            "timestamp": np.linspace(0, 830, 50),
            "set_name": ["s1"] * 50,
            "slide_index": [0] * 50,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)
    
    def test_cluster_analyzer_with_dbscan(self):
        
        analyzer = ClusterAnalyzer(
            self.output_folder,
            eps=50,
            min_samples=5
        )
        
        df = pd.DataFrame({
            "avg_gaze_x": np.random.uniform(100, 500, 50),
            "avg_gaze_y": np.random.uniform(150, 600, 50),
            "system_time": np.linspace(0, 830, 50),
            "set_name": ["s1"] * 50,
            "slide_index": [0] * 50,
        })
        
        result = analyzer.analyze(df)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
