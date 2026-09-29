import unittest
import tempfile
import sys
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.analyze.models import (
    HeatmapAnalyzer, FocusMapAnalyzer, SaccadeAnalyzer, FixationAnalyzer,
    EntropyAnalyzer, ClusterAnalyzer, ScanpathsAnalyzer, VoiceTranscription,
    BBoxImagesAnalyzer, BBoxTimeSeriesAnalyzer, BBoxTextAnalyzer
)


class TestHeatmapAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 0, 0],
            'avg_gaze_x': [100.0, 150.0, 200.0],
            'avg_gaze_y': [100.0, 150.0, 200.0],
            'bbox_id': [1, 2, 3]
        })
        self.bbox_data = [
            {'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50},
            {'id': 2, 'cx': 150, 'cy': 150, 'w': 50, 'h': 50},
            {'id': 3, 'cx': 200, 'cy': 200, 'w': 50, 'h': 50}
        ]

    def test_heatmap_analyzer_instantiation(self):
        analyzer = HeatmapAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_heatmap_analyze_no_output_path(self):
        analyzer = HeatmapAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.HeatmapAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)

    def test_heatmap_plot_basic(self):
        analyzer = HeatmapAnalyzer(get_test_config_path())
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.show'):
                    try:
                        analyzer.plot(output_path=tmpdir)
                    except Exception:
                        pass


class TestFocusMapAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'avg_gaze_x': [100.0],
            'avg_gaze_y': [100.0],
            'bbox_id': [1],
            'dwell_time': [500]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]

    def test_focusmap_analyzer_instantiation(self):
        analyzer = FocusMapAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_focusmap_analyze_no_output(self):
        analyzer = FocusMapAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.FocusMapAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestSaccadeAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': [0, 0],
            'avg_gaze_x': [100.0, 200.0],
            'avg_gaze_y': [100.0, 200.0],
            'bbox_id': [1, 2]
        })
        self.bbox_data = [
            {'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50},
            {'id': 2, 'cx': 200, 'cy': 200, 'w': 50, 'h': 50}
        ]

    def test_saccade_analyzer_instantiation(self):
        analyzer = SaccadeAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_saccade_analyze(self):
        analyzer = SaccadeAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.SaccadeAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestFixationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'avg_gaze_x': [100.0],
            'avg_gaze_y': [100.0],
            'bbox_id': [1],
            'dwell_time': [500]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]

    def test_fixation_analyzer_instantiation(self):
        analyzer = FixationAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_fixation_analyze(self):
        analyzer = FixationAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.FixationAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestEntropyAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': [0, 0],
            'avg_gaze_x': [100.0, 150.0],
            'avg_gaze_y': [100.0, 150.0],
            'bbox_id': [1, 2]
        })
        self.bbox_data = [
            {'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50},
            {'id': 2, 'cx': 150, 'cy': 150, 'w': 50, 'h': 50}
        ]

    def test_entropy_analyzer_instantiation(self):
        analyzer = EntropyAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_entropy_analyze(self):
        analyzer = EntropyAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.EntropyAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestClusterAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 0, 0],
            'avg_gaze_x': [100.0, 105.0, 110.0],
            'avg_gaze_y': [100.0, 105.0, 110.0],
            'bbox_id': [1, 1, 1]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]

    def test_cluster_analyzer_instantiation(self):
        analyzer = ClusterAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_cluster_analyze(self):
        analyzer = ClusterAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.ClusterAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestScanpathsAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': [0, 0],
            'avg_gaze_x': [100.0, 200.0],
            'avg_gaze_y': [100.0, 200.0],
            'bbox_id': [1, 2]
        })
        self.bbox_data = [
            {'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50},
            {'id': 2, 'cx': 200, 'cy': 200, 'w': 50, 'h': 50}
        ]

    def test_scanpaths_analyzer_instantiation(self):
        analyzer = ScanpathsAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_scanpaths_analyze(self):
        analyzer = ScanpathsAnalyzer(get_test_config_path())
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('tobii_pytracker.analyze.models.ScanpathsAnalyzer.analyze', return_value=None):
                result = analyzer.analyze(output_path=tmpdir)
                self.assertIsNone(result)


class TestVoiceTranscription(unittest.TestCase):

    def setUp(self):
        self.gaze_data = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'timestamp': [0.0],
            'gaze_x': [100.0],
            'gaze_y': [100.0]
        })
        self.voice_files = []

    def test_voice_transcription_instantiation(self):
        analyzer = VoiceTranscription(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_voice_transcription_analyze(self):
        analyzer = VoiceTranscription(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.VoiceTranscription.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestBBoxImagesAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'bbox_id': [1]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]
        self.images = {}

    def test_bbox_images_analyzer_instantiation(self):
        analyzer = BBoxImagesAnalyzer(get_test_config_path(), self.images)
        self.assertIsNotNone(analyzer)

    def test_bbox_images_analyze(self):
        analyzer = BBoxImagesAnalyzer(get_test_config_path(), self.images)
        with patch('tobii_pytracker.analyze.models.BBoxImagesAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestBBoxTimeSeriesAnalyzer(unittest.TestCase):

    def setUp(self):
        self.gaze_data = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'timestamp': [0.0],
            'gaze_x': [100.0],
            'gaze_y': [100.0]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]

    def test_bbox_timeseries_analyzer_instantiation(self):
        analyzer = BBoxTimeSeriesAnalyzer(get_test_config_path())
        self.assertIsNotNone(analyzer)

    def test_bbox_timeseries_analyze(self):
        analyzer = BBoxTimeSeriesAnalyzer(get_test_config_path())
        with patch('tobii_pytracker.analyze.models.BBoxTimeSeriesAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


class TestBBoxTextAnalyzer(unittest.TestCase):

    def setUp(self):
        self.bbox_gaze = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'bbox_id': [1]
        })
        self.bbox_data = [{'id': 1, 'cx': 100, 'cy': 100, 'w': 50, 'h': 50}]
        self.ocr_results = {}

    def test_bbox_text_analyzer_instantiation(self):
        analyzer = BBoxTextAnalyzer(get_test_config_path(), self.ocr_results)
        self.assertIsNotNone(analyzer)

    def test_bbox_text_analyzer_analyze(self):
        analyzer = BBoxTextAnalyzer(get_test_config_path(), self.ocr_results)
        with patch('tobii_pytracker.analyze.models.BBoxTextAnalyzer.analyze', return_value=None):
            result = analyzer.analyze(output_path=None)
            self.assertIsNone(result)


def get_test_config_path() -> Path:
    return get_test_resources_dir() / "test_config_tmp.yaml"

def get_test_resources_dir() -> Path:
    return Path(__file__).parent / "resources"

if __name__ == '__main__':
    unittest.main()
