import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.data_loader import GazePoint, DataLoader
from tobii_pytracker.analyze.bbox.geometry import polygon_vertices, point_inside_polygon
from tobii_pytracker.analyze.bbox.parsing import extract_text_bboxes, extract_image_bboxes
from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention


class TestGazePointDataclass(unittest.TestCase):

    def test_gazepoint_creation(self):
        point = GazePoint(x=0.5, y=0.6, pupil_size=5.0, timestamp=0.0)
        self.assertEqual(point.x, 0.5)
        self.assertEqual(point.y, 0.6)

    def test_gazepoint_with_none_values(self):
        point = GazePoint(x=None, y=None, pupil_size=None, timestamp=None)
        self.assertIsNone(point.x)
        self.assertIsNone(point.y)

    def test_gazepoint_negative_values(self):
        point = GazePoint(x=-0.5, y=-0.6, pupil_size=0.0, timestamp=-1.0)
        self.assertEqual(point.x, -0.5)
        self.assertEqual(point.y, -0.6)

    def test_gazepoint_large_values(self):
        point = GazePoint(x=1000.5, y=2000.7, pupil_size=10.0, timestamp=100.0)
        self.assertEqual(point.x, 1000.5)
        self.assertEqual(point.y, 2000.7)


class TestPolygonOperations(unittest.TestCase):

    def test_polygon_vertices_invalid_input(self):
        result = polygon_vertices(None)
        self.assertIsNone(result)

    def test_polygon_vertices_empty_dict(self):
        result = polygon_vertices({})
        self.assertIsNone(result)

    def test_point_inside_polygon_inside(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = point_inside_polygon(50, 50, polygon)
        self.assertTrue(result)

    def test_point_inside_polygon_outside(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = point_inside_polygon(150, 150, polygon)
        self.assertFalse(result)

    def test_point_inside_polygon_on_edge(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = point_inside_polygon(0, 50, polygon)
        self.assertIsNotNone(result)

    def test_point_inside_polygon_vertex(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = point_inside_polygon(0, 0, polygon)
        self.assertIsNotNone(result)

    def test_point_inside_polygon_none(self):
        try:
            result = point_inside_polygon(50, 50, None)
            self.assertFalse(result)
        except ValueError:
            pass


class TestTextBboxExtraction(unittest.TestCase):

    def test_extract_text_bboxes_valid(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0],
            'text_bboxes': [json.dumps({'text_bboxes': [{'cx': 100, 'cy': 100}]})]
        })
        
        result = extract_text_bboxes(raw_data)
        self.assertIsNotNone(result)

    def test_extract_text_bboxes_empty(self):
        raw_data = pd.DataFrame({
            'set_name': [],
            'slide_index': [],
            'text_bboxes': []
        })
        
        result = extract_text_bboxes(raw_data)
        self.assertEqual(len(result), 0)

    def test_extract_text_bboxes_missing_column(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0]
        })
        
        try:
            result = extract_text_bboxes(raw_data)
            self.assertIsNotNone(result)
        except (KeyError, TypeError):
            pass

    def test_extract_text_bboxes_null_values(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1', None],
            'slide_index': [0, 1],
            'text_bboxes': [None, '{}']
        })
        
        try:
            result = extract_text_bboxes(raw_data)
            self.assertIsNotNone(result)
        except (KeyError, TypeError, ValueError):
            pass


class TestImageBboxExtraction(unittest.TestCase):

    def test_extract_image_bboxes_valid(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0],
            'image_bboxes': [json.dumps({'image_bboxes': [{'cx': 100, 'cy': 100}]})]
        })
        
        result = extract_image_bboxes(raw_data)
        self.assertIsNotNone(result)

    def test_extract_image_bboxes_empty(self):
        raw_data = pd.DataFrame({
            'set_name': [],
            'slide_index': [],
            'image_bboxes': []
        })
        
        result = extract_image_bboxes(raw_data)
        self.assertEqual(len(result), 0)

    def test_extract_image_bboxes_multiple_rows(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1', 'set2'],
            'slide_index': [0, 1],
            'image_bboxes': [
                json.dumps({'image_bboxes': []}),
                json.dumps({'image_bboxes': []})
            ]
        })
        
        result = extract_image_bboxes(raw_data)
        self.assertIsNotNone(result)


class TestScoringEdgeCases(unittest.TestCase):

    def test_scoring_with_zero_gaze_points(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0],
            'objects_bboxes': [json.dumps({'image_bboxes': []})]
        })
        
        gaze_data = pd.DataFrame({
            'set_name': [],
            'slide_index': [],
            'avg_gaze_x': [],
            'avg_gaze_y': []
        })
        
        try:
            result = analyze_bbox_attention(raw_data, gaze_data)
            self.assertIsInstance(result, pd.DataFrame)
        except ValueError:
            pass

    def test_scoring_with_inf_attention_score(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0],
            'objects_bboxes': [json.dumps({'image_bboxes': []})]
        })
        
        gaze_data = pd.DataFrame({
            'set_name': ['set1'],
            'slide_index': [0],
            'avg_gaze_x': [float('inf')],
            'avg_gaze_y': [float('inf')]
        })
        
        try:
            result = analyze_bbox_attention(raw_data, gaze_data)
            self.assertIsInstance(result, pd.DataFrame)
        except (ValueError, TypeError):
            pass

    def test_scoring_with_duplicate_gaze_points(self):
        raw_data = pd.DataFrame({
            'set_name': ['set1', 'set1'],
            'slide_index': [0, 0],
            'objects_bboxes': [
                json.dumps({'image_bboxes': [{'bbox': {'cx': 100, 'cy': 100, 'w': 50, 'h': 50}}]}),
                json.dumps({'image_bboxes': [{'bbox': {'cx': 200, 'cy': 200, 'w': 50, 'h': 50}}]})
            ]
        })
        
        gaze_data = pd.DataFrame({
            'set_name': ['set1', 'set1', 'set1'],
            'slide_index': [0, 0, 0],
            'avg_gaze_x': [100.0, 100.0, 200.0],
            'avg_gaze_y': [100.0, 100.0, 200.0]
        })
        
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsInstance(result, pd.DataFrame)


class TestDataTypeConversions(unittest.TestCase):

    def test_gaze_data_numeric_conversion(self):
        gaze_data = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': [0, 1],
            'avg_gaze_x': ['100.5', '200.7'],
            'avg_gaze_y': ['100.5', '200.7']
        })
        
        gaze_data['avg_gaze_x'] = pd.to_numeric(gaze_data['avg_gaze_x'])
        gaze_data['avg_gaze_y'] = pd.to_numeric(gaze_data['avg_gaze_y'])
        
        self.assertEqual(gaze_data['avg_gaze_x'].dtype, np.float64)

    def test_bbox_coordinate_conversion(self):
        bbox_data = {
            'cx': '100',
            'cy': '200',
            'w': '50',
            'h': '60'
        }
        
        cx = float(bbox_data['cx'])
        cy = float(bbox_data['cy'])
        
        self.assertEqual(cx, 100.0)
        self.assertEqual(cy, 200.0)

    def test_series_to_dict_conversion(self):
        series = pd.Series({
            'set_name': 1,
            'slide_index': 0,
            'value': 100
        })
        
        result = series.to_dict()
        self.assertEqual(result['set_name'], 1)
        self.assertEqual(result['slide_index'], 0)


if __name__ == "__main__":
    unittest.main()
