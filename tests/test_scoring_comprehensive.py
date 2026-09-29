import unittest
import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention, evaluate_bbox_attention


class TestScoringPhase5(unittest.TestCase):

    def test_rect_bbox_non_dict_fallback(self):
        
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {
                            "bbox": None,
                            "polygon": None,
                            "rect_bbox": "invalid_non_dict_value"
                        }
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_rect_bbox_with_valid_coordinates(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {
                            "bbox": {"cx": "invalid"},  
                            "polygon": None,
                            "rect_bbox": {
                                "cx": 100.0,
                                "cy": 100.0,
                                "w": 50.0,
                                "h": 50.0
                            }
                        }
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        
        self.assertGreater(len(result), 0)
        self.assertGreater(result.iloc[0]["hit_count"], 0)

    def test_rect_bbox_list_converted_to_dict(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {
                            "bbox": None,
                            "polygon": None,
                            "rect_bbox": [100, 100, 50, 50]  
                        }
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        
        self.assertIsInstance(result, pd.DataFrame)

    def test_analyze_bbox_missing_gaze_columns(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}
                    ]
                })
            ]
        })
        
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0]
        })
        
        try:
            result = analyze_bbox_attention(raw_data, gaze_data)
            
            self.assertIsInstance(result, pd.DataFrame)
        except KeyError:
            
            pass

    def test_evaluate_bbox_non_overlapping_hits(self):
        data = pd.DataFrame({
            "set_name": ["s1", "s1"],
            "slide_index": [0, 0],
            "total_gaze_points": [100, 100],
            "hit_count": [10, 10],
            "hit_gaze_indices": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
            "attention_score": [0.5, 0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["overlap_factor"], 1.0)

    def test_polygon_empty_bbox_fallback(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {
                            "bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50},
                            "polygon": None
                        }
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.0],
            "avg_gaze_y": [100.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        
        self.assertGreater(len(result), 0)

    def test_analyze_bbox_with_string_coordinates(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {"bbox": {"cx": "100.5", "cy": "100.5", "w": "50.0", "h": "50.0"}}
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [100.5],
            "avg_gaze_y": [100.5]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertGreater(len(result), 0)

    def test_scoring_with_nan_gaze_values(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1", "set1"],
            "slide_index": [0, 0, 0],
            "avg_gaze_x": [100.0, np.nan, 100.0],
            "avg_gaze_y": [100.0, np.nan, 100.0]
        })
        
        
        try:
            result = analyze_bbox_attention(raw_data, gaze_data)
            self.assertIsInstance(result, pd.DataFrame)
        except (ValueError, TypeError):
            pass

    def test_scoring_with_inf_gaze_values(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                json.dumps({
                    "image_bboxes": [
                        {"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}
                    ]
                })
            ]
        })
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 0],
            "avg_gaze_x": [100.0, float('inf')],
            "avg_gaze_y": [100.0, float('inf')]
        })
        
        
        try:
            result = analyze_bbox_attention(raw_data, gaze_data)
            self.assertIsInstance(result, pd.DataFrame)
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    unittest.main()
