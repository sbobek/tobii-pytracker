import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention, evaluate_bbox_attention
from tobii_pytracker.analyze.bbox.parsing import parse_serialized


class TestEvaluateBboxAttention(unittest.TestCase):

    def test_evaluate_bbox_attention_empty_dataframe(self):
        result = evaluate_bbox_attention(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_evaluate_bbox_attention_none_input(self):
        result = evaluate_bbox_attention(None)
        self.assertTrue(result.empty)

    def test_evaluate_bbox_attention_single_group(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [100],
            "hit_count": [10],
            "hit_gaze_indices": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            "attention_score": [0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 1)
        self.assertIn("coverage_by_bboxes", result.columns)

    def test_evaluate_bbox_attention_multiple_groups(self):
        data = pd.DataFrame({
            "set_name": ["s1", "s1", "s2", "s2"],
            "slide_index": [0, 1, 0, 1],
            "total_gaze_points": [100, 100, 100, 100],
            "hit_count": [10, 20, 30, 40],
            "hit_gaze_indices": [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10]],
            "attention_score": [0.5, 0.6, 0.7, 0.8]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 4)

    def test_evaluate_bbox_attention_overlap_factor(self):
        data = pd.DataFrame({
            "set_name": ["s1", "s1"],
            "slide_index": [0, 0],
            "total_gaze_points": [100, 100],
            "hit_count": [3, 3],
            "hit_gaze_indices": [[1, 2, 3], [3, 4, 5]],
            "attention_score": [0.5, 0.7]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["overlap_factor"], 1.2, places=1)

    def test_evaluate_bbox_attention_with_tuple_indices(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [100],
            "hit_count": [3],
            "hit_gaze_indices": [(1, 2, 3)],
            "attention_score": [0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(result.iloc[0]["unique_gaze_hit_count"], 3)

    def test_evaluate_bbox_attention_with_set_indices(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [100],
            "hit_count": [3],
            "hit_gaze_indices": [{1, 2, 3}],
            "attention_score": [0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(result.iloc[0]["unique_gaze_hit_count"], 3)

    def test_evaluate_bbox_attention_with_ndarray_indices(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [100],
            "hit_count": [3],
            "hit_gaze_indices": [np.array([1, 2, 3])],
            "attention_score": [0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(result.iloc[0]["unique_gaze_hit_count"], 3)

    def test_evaluate_bbox_attention_zero_gaze_points(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [0],
            "hit_count": [0],
            "hit_gaze_indices": [[]],
            "attention_score": [0.0]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(result.iloc[0]["coverage_by_bboxes"], 0.0)

    def test_evaluate_bbox_attention_zero_unique_hits(self):
        data = pd.DataFrame({
            "set_name": ["s1"],
            "slide_index": [0],
            "total_gaze_points": [100],
            "hit_count": [0],
            "hit_gaze_indices": [[]],
            "attention_score": [0.0]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(result.iloc[0]["overlap_factor"], 0.0)

    def test_evaluate_bbox_attention_no_attended_bboxes(self):
        data = pd.DataFrame({
            "set_name": ["s1", "s1", "s1"],
            "slide_index": [0, 0, 0],
            "total_gaze_points": [100, 100, 100],
            "hit_count": [0, 0, 10],
            "hit_gaze_indices": [[], [], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            "attention_score": [0.0, 0.0, 0.5]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 1)
        self.assertGreater(result.iloc[0]["attended_bboxes"], 0)

    def test_evaluate_bbox_attention_attended_ratio(self):
        
        data = pd.DataFrame({
            "set_name": ["s1", "s1", "s1"],
            "slide_index": [0, 0, 0],
            "total_gaze_points": [100, 100, 100],
            "hit_count": [10, 0, 20],
            "hit_gaze_indices": [[1, 2, 3, 4, 5], [], [6, 7]],
            "attention_score": [0.5, 0.0, 0.7]
        })
        
        result = evaluate_bbox_attention(data)
        self.assertEqual(len(result), 1)
        
        self.assertAlmostEqual(result.iloc[0]["attended_bbox_ratio"], 2/3, places=2)


class TestAnalyzeBboxWithFixations(unittest.TestCase):

    def test_analyze_with_fixations_enabled(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}]}'
            ]
        })
        
        
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 0],
            "x_mean": [100.0, 105.0],
            "y_mean": [100.0, 105.0],
            "duration": [16.67, 16.67]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data, use_fixations=True)
        self.assertIsInstance(result, pd.DataFrame)

    def test_analyze_bbox_with_polygon_bbox(self):
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": [
                '{"image_bboxes": [{"polygon": [[100, 100], [150, 100], [150, 150], [100, 150]]}]}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "avg_gaze_x": [125.0],
            "avg_gaze_y": [125.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsInstance(result, pd.DataFrame)


class TestParsingEdgeCases(unittest.TestCase):

    def test_parse_serialized_valid_dict(self):
        literal = "{'key': 'value', 'number': 123}"
        result = parse_serialized(literal)
        
        self.assertIsNotNone(result)

    def test_parse_serialized_valid_list(self):
        literal = "[1, 2, 3, 4, 5]"
        result = parse_serialized(literal)
        self.assertIsNotNone(result)

    def test_parse_serialized_empty_dict(self):
        literal = "{}"
        result = parse_serialized(literal)
        self.assertIsNotNone(result)

    def test_parse_serialized_nested_structures(self):
        literal = "{'outer': {'inner': [1, 2, 3]}}"
        result = parse_serialized(literal)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
