import sys
import matplotlib
matplotlib.use("Agg", force=True)

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from tests.support import bootstrap_test_environment


sys.modules['sounddevice'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['tobii_research'] = MagicMock()

bootstrap_test_environment()

from tobii_pytracker.analyze.bbox.parsing import extract_gaze


class TestDataLoaderFunctions(unittest.TestCase):

    @patch('tobii_pytracker.analyze.data_loader.pd.read_csv')
    def test_load_data_csv(self, mock_read_csv):
        from tobii_pytracker.analyze.data_loader import DataLoader
        from tobii_pytracker.configs.custom_config import CustomConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            
            config = MagicMock(spec=CustomConfig)
            config.get_output_config.return_value = {"folder": str(output_dir)}
            
            mock_read_csv.return_value = pd.DataFrame({"col": [1, 2, 3]})
            
            loader = DataLoader(config, root=Path(tmpdir))
            self.assertIsNotNone(loader)

    def test_gaze_point_creation(self):
        from tobii_pytracker.analyze.data_loader import GazePoint
        
        point = GazePoint(x=100.0, y=200.0, pupil_size=3.5, timestamp=0.5)
        self.assertEqual(point.x, 100.0)
        self.assertEqual(point.y, 200.0)
        self.assertEqual(point.pupil_size, 3.5)
        self.assertEqual(point.timestamp, 0.5)

    def test_gaze_point_with_none(self):
        from tobii_pytracker.analyze.data_loader import GazePoint
        
        point = GazePoint(x=None, y=None, pupil_size=None, timestamp=0.0)
        self.assertIsNone(point.x)
        self.assertIsNone(point.y)
        self.assertIsNone(point.pupil_size)


class TestConfigFunctions(unittest.TestCase):

    pass

class TestGazeExtraction(unittest.TestCase):

    def test_extract_gaze_with_arrays(self):
        row = pd.Series({})
        slide_data = pd.DataFrame({
            "avg_gaze_x": np.array([100, 150, np.nan]),
            "avg_gaze_y": np.array([200, 250, np.nan])
        })
        
        result = extract_gaze(row, slide_data)
        self.assertIsNotNone(result)

    def test_extract_gaze_serialized(self):
        row = pd.Series({
            "gaze_data": "[{'avg_gaze_x': 100, 'avg_gaze_y': 200}]"
        })
        slide_data = pd.DataFrame({})
        
        result = extract_gaze(row, slide_data)
        self.assertIsNotNone(result)


class TestParsingEdgeCases(unittest.TestCase):

    def test_extract_image_bboxes_with_bbox_field(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_image_bboxes
        
        raw = [
            {"bbox": {"id": 1, "cx": 100, "cy": 100}},
            {"bbox": {"id": 2, "cx": 200, "cy": 200}}
        ]
        result = extract_image_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_image_bboxes_mixed(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_image_bboxes
        
        raw = [
            {"image_bboxes": [{"id": 1}]},
            {"bbox": {"id": 2}}
        ]
        result = extract_image_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_text_bboxes_default_level(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_text_bboxes
        
        raw = {"words": [{"x": 1}], "lines": [{"x": 2}]}
        result = extract_text_bboxes(raw)  
        self.assertEqual(len(result), 1)

    def test_extract_gaze_points_serialized_string(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({
            "gaze_data": "[{'avg_gaze_x': 100, 'avg_gaze_y': 200}]"
        })
        slide_data = pd.DataFrame({})
        
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        self.assertGreater(len(gaze_x), 0)

    def test_extract_gaze_points_non_dict_items(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({
            "gaze_data": [
                {"avg_gaze_x": 100, "avg_gaze_y": 200},
                "invalid_string",
                None,
            ]
        })
        slide_data = pd.DataFrame({})
        
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 1)


class TestVoiceRecorderMocked(unittest.TestCase):

    @patch('tobii_pytracker.utils.voice.sd.InputStream')
    @patch('tobii_pytracker.utils.voice.sf.write')
    def test_voice_recorder_basic(self, mock_write, mock_stream):
        from tobii_pytracker.utils.voice import VoiceRecorder
        import threading
        
        stop_event = threading.Event()
        
        
        mock_stream.return_value.__enter__.return_value = None
        mock_stream.return_value.__exit__.return_value = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.wav")
            
            
            try:
                
                stop_event.set()
                VoiceRecorder.record_voice(filename, stop_event=stop_event)
            except Exception:
                pass  


class TestScoringEdgeCases(unittest.TestCase):

    def test_analyze_bbox_attention_with_polygon(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"polygon": [[0, 0], [10, 0], [10, 10]]}]}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [5.0],
            "avg_gaze_y": [5.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)

    def test_analyze_bbox_attention_rect_bbox(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"rect_bbox": {"x": 10, "y": 10, "w": 50, "h": 50}}]}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [35.0],
            "avg_gaze_y": [35.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
