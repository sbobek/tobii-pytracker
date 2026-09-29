import sys
import matplotlib
matplotlib.use("Agg", force=True)

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from tests.support import bootstrap_test_environment

sys.modules['sounddevice'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['tobii_research'] = MagicMock()

bootstrap_test_environment()


class TestParsingCoverage(unittest.TestCase):

    def test_extract_gaze_points_non_list_gaze_data(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        
        row = pd.Series({
            "gaze_data": {"not_a_list": "data"}
        })
        slide_data = pd.DataFrame({})
        
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        self.assertEqual(len(gaze_x), 0)
        self.assertEqual(len(gaze_y), 0)

    def test_extract_gaze_with_literal_parse_error(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze
        
        
        row = pd.Series({
            "gaze_data": "{'not': 'a_list'}"
        })
        slide_data = pd.DataFrame({})
        
        with self.assertRaises(ValueError):
            extract_gaze(row, slide_data)

    def test_extract_gaze_missing_data_raises_error(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze
        
        row = pd.Series({})
        slide_data = pd.DataFrame({})
        
        with self.assertRaises(KeyError):
            extract_gaze(row, slide_data)


class TestScoringCoverage(unittest.TestCase):

    def test_analyze_bbox_attention_no_gaze_points(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"bbox": {"cx": 100, "cy": 100, "w": 20, "h": 20}}]}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [500.0],  
            "avg_gaze_y": [500.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        
        self.assertIsNotNone(result)

    def test_analyze_bbox_attention_with_invalid_bbox(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"invalid": "data"}]}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)


class TestVoiceCoverage(unittest.TestCase):

    @patch('tobii_pytracker.utils.voice.sd')
    @patch('tobii_pytracker.utils.voice.sf')
    def test_voice_recorder_callback_with_frames(self, mock_sf, mock_sd):
        from tobii_pytracker.utils.voice import VoiceRecorder
        import threading
        
        
        stop_event = threading.Event()
        stop_event.set()
        
        
        mock_sd.sleep = MagicMock()
        mock_sd.CallbackAbort = Exception
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.wav"
            VoiceRecorder.record_voice(filename, stop_event=stop_event)


class TestDataLoaderCoverage(unittest.TestCase):

    def test_gaze_point_dataclass(self):
        from tobii_pytracker.analyze.data_loader import GazePoint
        
        point = GazePoint(x=100.0, y=200.0, pupil_size=2.5, timestamp=1.0)
        self.assertEqual(point.x, 100.0)
        self.assertEqual(point.y, 200.0)
        self.assertEqual(point.pupil_size, 2.5)
        self.assertEqual(point.timestamp, 1.0)

    def test_gaze_point_all_none(self):
        from tobii_pytracker.analyze.data_loader import GazePoint
        
        point = GazePoint(x=None, y=None, pupil_size=None, timestamp=0.0)
        self.assertIsNone(point.x)
        self.assertIsNone(point.y)
        self.assertIsNone(point.pupil_size)
        self.assertEqual(point.timestamp, 0.0)


class TestParsingWithNonDictItems(unittest.TestCase):

    def test_extract_gaze_points_mixed_types_in_list(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({
            "gaze_data": [
                {"avg_gaze_x": 100, "avg_gaze_y": 200},
                None,
                123,  
                {"avg_gaze_x": 150, "avg_gaze_y": 250}
            ]
        })
        slide_data = pd.DataFrame({})
        
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 2)

    def test_extract_gaze_points_all_nan_values(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({
            "gaze_data": [
                {"avg_gaze_x": np.nan, "avg_gaze_y": np.nan},
                {"avg_gaze_x": np.nan, "avg_gaze_y": np.nan}
            ]
        })
        slide_data = pd.DataFrame({})
        
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 0)


class TestBboxScoringEdgeCases(unittest.TestCase):

    def test_analyze_bbox_with_string_slide_index(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": ["0"],  
            "objects_bboxes": ['{"image_bboxes": []}']
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": ["0"],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)

    def test_analyze_bbox_with_multiple_sets(self):
        from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention
        
        raw_data = pd.DataFrame({
            "set_name": ["set1", "set2"],
            "slide_index": [0, 0],
            "objects_bboxes": [
                '{"image_bboxes": [{"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}]}',
                '{"image_bboxes": []}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1", "set2", "set2"],
            "slide_index": [0, 0, 0, 0],
            "avg_gaze_x": [100.0, 110.0, 200.0, 210.0],
            "avg_gaze_y": [100.0, 110.0, 200.0, 210.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
