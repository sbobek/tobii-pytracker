import sys
import matplotlib
matplotlib.use("Agg", force=True)

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import tempfile

from tests.support import bootstrap_test_environment


sys.modules['sounddevice'] = MagicMock()
sys.modules['soundfile'] = MagicMock()

bootstrap_test_environment()

from tobii_pytracker.analyze.bbox.geometry import (
    bbox_edges_centered, polygon_vertices, point_inside_polygon,
    polygon_to_plot_coords, point_inside_bbox, get_plot_bounds,
    calculate_points, get_valid_gaze, get_visited_bboxes, get_series_range,
    _get_bbox_bounds
)
from tobii_pytracker.analyze.bbox.parsing import (
    parse_objects_bboxes, parse_input_data, extract_timeseries_bboxes,
    parse_serialized, extract_text_bboxes, extract_gaze_points
)
from tobii_pytracker.analyze.bbox.scoring import analyze_bbox_attention


class TestGeometryFunctions(unittest.TestCase):

    def test_bbox_edges_centered_basic(self):
        bbox = {"cx": 100.0, "cy": 200.0, "w": 50.0, "h": 60.0}
        edges = bbox_edges_centered(bbox)
        self.assertEqual(edges["x_min"], 75.0)
        self.assertEqual(edges["x_max"], 125.0)
        self.assertEqual(edges["y_min"], 170.0)
        self.assertEqual(edges["y_max"], 230.0)

    def test_bbox_edges_centered_with_strings(self):
        bbox = {"cx": "100", "cy": "200", "w": "50", "h": "60"}
        edges = bbox_edges_centered(bbox)
        self.assertEqual(edges["x_min"], 75.0)
        self.assertEqual(edges["x_max"], 125.0)

    def test_polygon_vertices_from_dicts(self):
        value = [{"x": 0, "y": 0}, {"x": 10, "y": 5}, {"x": 10, "y": 15}]
        vertices = polygon_vertices(value)
        self.assertIsNotNone(vertices)
        self.assertEqual(len(vertices), 3)
        np.testing.assert_array_equal(vertices[0], [0, 0])

    def test_polygon_vertices_from_tuples(self):
        value = [(0, 0), (10, 5), (10, 15)]
        vertices = polygon_vertices(value)
        self.assertIsNotNone(vertices)
        self.assertEqual(len(vertices), 3)

    def test_polygon_vertices_none(self):
        self.assertIsNone(polygon_vertices(None))

    def test_polygon_vertices_insufficient_points(self):
        self.assertIsNone(polygon_vertices([(0, 0), (1, 1)]))

    def test_polygon_vertices_invalid_format(self):
        self.assertIsNone(polygon_vertices([{"x": "invalid", "y": 0}]))

    def test_polygon_vertices_type_error(self):
        self.assertIsNone(polygon_vertices("not a list"))

    def test_point_inside_polygon_true(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertTrue(point_inside_polygon(5, 5, polygon))

    def test_point_inside_polygon_false(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertFalse(point_inside_polygon(15, 15, polygon))

    def test_polygon_to_plot_coords(self):
        polygon = np.array([[0, 0], [10, -10]])
        coords = polygon_to_plot_coords(polygon, width=100, height=200)
        self.assertEqual(coords[0, 0], 50.0)
        self.assertEqual(coords[0, 1], 100.0)
        self.assertEqual(coords[1, 0], 60.0)
        self.assertEqual(coords[1, 1], 110.0)

    def test_point_inside_bbox_center(self):
        bbox = {"cx": 100.0, "cy": 100.0, "w": 40.0, "h": 40.0}
        self.assertTrue(point_inside_bbox(100, 100, bbox))

    def test_point_inside_bbox_edge(self):
        bbox = {"cx": 100.0, "cy": 100.0, "w": 40.0, "h": 40.0}
        self.assertTrue(point_inside_bbox(80, 100, bbox))

    def test_point_inside_bbox_with_margin(self):
        bbox = {"cx": 100.0, "cy": 100.0, "w": 20.0, "h": 20.0}
        self.assertTrue(point_inside_bbox(95, 95, bbox, margin=10))

    def test_point_inside_bbox_outside(self):
        bbox = {"cx": 100.0, "cy": 100.0, "w": 20.0, "h": 20.0}
        self.assertFalse(point_inside_bbox(200, 200, bbox))

    def test_get_plot_bounds(self):
        x_min, y_min, x_max, y_max = get_plot_bounds(800, 600)
        self.assertEqual(x_min, 60.0)
        self.assertEqual(y_min, 0.0)
        self.assertEqual(x_max, 800.0)
        self.assertEqual(y_max, 560.0)

    def test_calculate_points(self):
        input_data = np.array([0.2, 0.5, 0.8])
        points_x, points_y = calculate_points(
            input_data, area_x=800, area_y=600,
            plot_x_min=60, plot_y_min=0,
            plot_x_max=800, plot_y_max=560,
            g_min=0.0, g_max=1.0
        )
        self.assertEqual(len(points_x), 3)
        self.assertEqual(len(points_y), 3)

    def test_get_valid_gaze_from_row_scalar(self):
        row = pd.Series({"avg_gaze_x": 100.0, "avg_gaze_y": 200.0})
        bg_data = pd.DataFrame({"avg_gaze_x": [1, 2], "avg_gaze_y": [3, 4]})
        gaze_x, gaze_y = get_valid_gaze(row, bg_data)
        
        if isinstance(gaze_x, (int, float)):
            self.assertEqual(gaze_x, 100.0)
            self.assertEqual(gaze_y, 200.0)
        else:
            
            self.assertTrue(len(gaze_x) > 0)

    def test_get_valid_gaze_from_background_data(self):
        row = pd.Series({})
        bg_data = pd.DataFrame({
            "avg_gaze_x": np.array([100, 200, np.nan]),
            "avg_gaze_y": np.array([150, 250, np.nan])
        })
        gaze_x, gaze_y = get_valid_gaze(row, bg_data)
        np.testing.assert_array_equal(gaze_x, [100, 200])
        np.testing.assert_array_equal(gaze_y, [150, 250])

    def test_get_valid_gaze_all_nan(self):
        row = pd.Series({})
        bg_data = pd.DataFrame({
            "avg_gaze_x": np.array([np.nan, np.nan]),
            "avg_gaze_y": np.array([np.nan, np.nan])
        })
        gaze_x, gaze_y = get_valid_gaze(row, bg_data)
        self.assertEqual(len(gaze_x), 0)
        self.assertEqual(len(gaze_y), 0)

    def test_get_visited_bboxes_match(self):
        timeseries_bboxes = [
            {"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}},
            {"bbox": {"cx": 200, "cy": 200, "w": 50, "h": 50}},
        ]
        gaze_x = np.array([100, 200])
        gaze_y = np.array([100, 200])
        visited = get_visited_bboxes(timeseries_bboxes, gaze_x, gaze_y)
        self.assertEqual(len(visited), 2)

    def test_get_visited_bboxes_no_match(self):
        timeseries_bboxes = [
            {"bbox": {"cx": 100, "cy": 100, "w": 10, "h": 10}},
        ]
        gaze_x = np.array([500, 600])
        gaze_y = np.array([500, 600])
        visited = get_visited_bboxes(timeseries_bboxes, gaze_x, gaze_y)
        self.assertEqual(len(visited), 0)

    def test_get_visited_bboxes_partial(self):
        timeseries_bboxes = [
            {"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}},
        ]
        gaze_x = np.array([100, 500])
        gaze_y = np.array([100, 500])
        visited = get_visited_bboxes(timeseries_bboxes, gaze_x, gaze_y)
        self.assertEqual(len(visited), 1)

    def test_get_series_range_normal(self):
        data = np.array([1, 5, 10])
        g_min, g_max = get_series_range(data)
        self.assertEqual(g_min, 1.0)
        self.assertEqual(g_max, 10.0)

    def test_get_series_range_with_nan(self):
        data = np.array([1, np.nan, 10])
        g_min, g_max = get_series_range(data)
        self.assertEqual(g_min, 1.0)
        self.assertEqual(g_max, 10.0)

    def test_get_series_range_equal_min_max(self):
        data = np.array([5.0, 5.0, 5.0])
        g_min, g_max = get_series_range(data)
        self.assertEqual(g_min, 4.5)
        self.assertEqual(g_max, 5.5)

    def test_get_bbox_bounds(self):
        bbox = {"cx": 100.0, "cy": 100.0, "w": 50.0, "h": 60.0}
        x_min, x_max, y_min, y_max = _get_bbox_bounds(bbox)
        self.assertEqual(x_min, 75.0)
        self.assertEqual(x_max, 125.0)
        self.assertEqual(y_min, 70.0)
        self.assertEqual(y_max, 130.0)


class TestParsingFunctions(unittest.TestCase):

    def test_parse_objects_bboxes_dict_input(self):
        
        value = {"image_bboxes": [{"x": 10}]}
        result = parse_objects_bboxes(value)
        self.assertEqual(result, value)

    def test_parse_objects_bboxes_json_string(self):
        
        value = '{"image_bboxes": [{"x": 10}]}'
        result = parse_objects_bboxes(value)
        self.assertEqual(result["image_bboxes"][0]["x"], 10)

    def test_parse_objects_bboxes_python_literal(self):
        value = "{'image_bboxes': [{'x': 10}]}"
        result = parse_objects_bboxes(value)
        self.assertEqual(result["image_bboxes"][0]["x"], 10)

    def test_parse_objects_bboxes_invalid_string(self):
        
        result = parse_objects_bboxes("not valid json or python")
        self.assertEqual(result, {"image_bboxes": []})

    def test_parse_objects_bboxes_invalid_type(self):
        
        result = parse_objects_bboxes(123)
        self.assertEqual(result, {"image_bboxes": []})

    def test_parse_input_data_string(self):
        
        raw = "[1.0 2.0 3.0]"
        result = parse_input_data(raw)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_parse_input_data_array(self):
        
        raw = [1.0, 2.0, 3.0]
        result = parse_input_data(raw)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_extract_timeseries_bboxes_dict(self):
        
        raw = {"timeseries_bboxes": [{"bbox": 1}, {"bbox": 2}]}
        result = extract_timeseries_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_timeseries_bboxes_list_of_dicts(self):
        
        raw = [
            {"timeseries_bboxes": [{"bbox": 1}]},
            {"timeseries_bboxes": [{"bbox": 2}]},
        ]
        result = extract_timeseries_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_timeseries_bboxes_dict_empty(self):
        
        self.assertEqual(extract_timeseries_bboxes({}), [])

    def test_extract_timeseries_bboxes_list_empty(self):
        
        self.assertEqual(extract_timeseries_bboxes([]), [])

    def test_parse_serialized_python_literal(self):
        
        result = parse_serialized("{'key': 'value'}")
        self.assertEqual(result["key"], "value")

    def test_parse_serialized_empty_string(self):
        
        self.assertIsNone(parse_serialized(""))

    def test_parse_serialized_whitespace(self):
        
        self.assertIsNone(parse_serialized("   "))

    def test_parse_serialized_dict(self):
        
        d = {"key": "value"}
        result = parse_serialized(d)
        self.assertEqual(result, d)

    def test_parse_serialized_invalid_returns_string(self):
        
        result = parse_serialized("not valid python")
        self.assertEqual(result, "not valid python")

    def test_extract_text_bboxes_dict_words(self):
        
        raw = {"words": [{"x": 1}, {"x": 2}]}
        result = extract_text_bboxes(raw, level="words")
        self.assertEqual(len(result), 2)

    def test_extract_text_bboxes_dict_lines(self):
        
        raw = {"lines": [{"x": 1}, {"x": 2}]}
        result = extract_text_bboxes(raw, level="lines")
        self.assertEqual(len(result), 2)

    def test_extract_text_bboxes_list_of_dicts(self):
        
        raw = [
            {"words": [{"x": 1}]},
            {"words": [{"x": 2}]},
        ]
        result = extract_text_bboxes(raw, level="words")
        self.assertEqual(len(result), 2)

    def test_extract_text_bboxes_serialized_string(self):
        
        raw = "{'words': [{'x': 1}]}"
        result = extract_text_bboxes(raw, level="words")
        self.assertEqual(len(result), 1)

    def test_extract_text_bboxes_missing_level(self):
        
        raw = {"other": [{"x": 1}]}
        result = extract_text_bboxes(raw, level="words")
        self.assertEqual(len(result), 0)


class TestScoringFunctions(unittest.TestCase):
    

    def test_analyze_bbox_attention_basic(self):
        
        raw_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 1],
            "objects_bboxes": [
                '{"image_bboxes": [{"bbox": {"cx": 100, "cy": 100, "w": 50, "h": 50}}]}',
                '{"image_bboxes": []}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 1],
            "avg_gaze_x": [100, 200],
            "avg_gaze_y": [100, 200]
        })
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_analyze_bbox_attention_missing_set_name(self):
        
        raw_data = pd.DataFrame({
            "slide_index": [0],
            "objects_bboxes": ['{"image_bboxes": []}']
        })
        gaze_data = pd.DataFrame({"avg_gaze_x": [1], "avg_gaze_y": [1]})
        
        with self.assertRaises(ValueError):
            analyze_bbox_attention(raw_data, gaze_data)

    def test_analyze_bbox_attention_auto_generate_slide_index(self):
        
        raw_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "objects_bboxes": [
                '{"image_bboxes": []}',
                '{"image_bboxes": []}'
            ]
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 1],
            "avg_gaze_x": [100, 200],
            "avg_gaze_y": [100, 200]
        })
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)

    def test_analyze_bbox_attention_with_fixations(self):
        
        raw_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "objects_bboxes": ['{"image_bboxes": []}']
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1"],
            "slide_index": [0],
            "x_mean": [100],
            "y_mean": [100],
            "duration": [100]
        })
        result = analyze_bbox_attention(raw_data, gaze_data, use_fixations=True)
        self.assertIsNotNone(result)

    def test_analyze_bbox_attention_filters_nan(self):
        
        raw_data = pd.DataFrame({
            "set_name": ["set1", "set1"],
            "slide_index": [0, 0],
            "objects_bboxes": ['{"image_bboxes": []}', '{"image_bboxes": []}']
        })
        gaze_data = pd.DataFrame({
            "set_name": ["set1", "set1", "set1"],
            "slide_index": [0, 0, 0],
            "avg_gaze_x": [100.0, np.nan, 50.0],
            "avg_gaze_y": [100.0, np.nan, 50.0]
        })
        result = analyze_bbox_attention(raw_data, gaze_data)
        self.assertIsNotNone(result)


class TestMoreParsingFunctions(unittest.TestCase):
    

    def test_extract_gaze_points_from_slide_data(self):
        
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({})
        slide_data = pd.DataFrame({
            "avg_gaze_x": [100.0, 150.0, np.nan],
            "avg_gaze_y": [200.0, 250.0, np.nan]
        })
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 2)
        self.assertEqual(len(gaze_y), 2)
        np.testing.assert_array_equal(gaze_x, [100.0, 150.0])

    def test_extract_gaze_points_from_gaze_data_column(self):
        
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({
            "gaze_data": [
                {"avg_gaze_x": 100, "avg_gaze_y": 200},
                {"avg_gaze_x": 150, "avg_gaze_y": 250}
            ]
        })
        slide_data = pd.DataFrame({})
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 2)

    def test_extract_gaze_points_empty(self):
        
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze_points
        
        row = pd.Series({})
        slide_data = pd.DataFrame({})
        gaze_x, gaze_y = extract_gaze_points(row, slide_data)
        
        self.assertEqual(len(gaze_x), 0)
        self.assertEqual(len(gaze_y), 0)

    def test_gaze_inside_bbox_true(self):
        
        from tobii_pytracker.analyze.bbox.parsing import gaze_inside_bbox
        
        gaze_x = np.array([100, 110])
        gaze_y = np.array([100, 110])
        bbox = {"cx": 100, "cy": 100, "w": 50, "h": 50}
        
        mask = gaze_inside_bbox(gaze_x, gaze_y, bbox)
        np.testing.assert_array_equal(mask, [True, True])

    def test_gaze_inside_bbox_false(self):
        
        from tobii_pytracker.analyze.bbox.parsing import gaze_inside_bbox
        
        gaze_x = np.array([200, 210])
        gaze_y = np.array([200, 210])
        bbox = {"cx": 100, "cy": 100, "w": 50, "h": 50}
        
        mask = gaze_inside_bbox(gaze_x, gaze_y, bbox)
        np.testing.assert_array_equal(mask, [False, False])

    def test_gaze_inside_bbox_with_padding(self):
        
        from tobii_pytracker.analyze.bbox.parsing import gaze_inside_bbox
        
        gaze_x = np.array([75, 100])
        gaze_y = np.array([75, 100])
        bbox = {"cx": 100, "cy": 100, "w": 50, "h": 50}
        
        
        mask_no_pad = gaze_inside_bbox(gaze_x, gaze_y, bbox, padding=0)
        self.assertTrue(mask_no_pad[0])
        self.assertTrue(mask_no_pad[1])
        
        
        gaze_x2 = np.array([70])
        gaze_y2 = np.array([70])
        mask_out = gaze_inside_bbox(gaze_x2, gaze_y2, bbox, padding=0)
        self.assertFalse(mask_out[0])

    def test_extract_image_bboxes_dict(self):
        
        from tobii_pytracker.analyze.bbox.parsing import extract_image_bboxes
        
        raw = {"image_bboxes": [{"id": 1}, {"id": 2}]}
        result = extract_image_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_image_bboxes_list(self):
        
        from tobii_pytracker.analyze.bbox.parsing import extract_image_bboxes
        
        raw = [
            {"image_bboxes": [{"id": 1}]},
            {"image_bboxes": [{"id": 2}]}
        ]
        result = extract_image_bboxes(raw)
        self.assertEqual(len(result), 2)

    def test_extract_image_bboxes_empty(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_image_bboxes
        
        self.assertEqual(extract_image_bboxes({}), [])
        self.assertEqual(extract_image_bboxes([]), [])

    def test_parse_literal_list(self):
        from tobii_pytracker.analyze.bbox.parsing import parse_literal
        
        result = parse_literal("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])

    def test_parse_literal_dict(self):
        from tobii_pytracker.analyze.bbox.parsing import parse_literal
        
        result = parse_literal("{'key': 'value'}")
        self.assertEqual(result["key"], "value")

    def test_parse_literal_invalid(self):
        from tobii_pytracker.analyze.bbox.parsing import parse_literal
        
        with self.assertRaises(ValueError):
            parse_literal("invalid")

    def test_bbox_contains_gaze_true(self):
        from tobii_pytracker.analyze.bbox.parsing import bbox_contains_gaze
        
        bbox = {"cx": 100, "cy": 100, "w": 50, "h": 50}
        result = bbox_contains_gaze(bbox, np.array([100]), np.array([100]))
        self.assertTrue(np.any(result))

    def test_bbox_contains_gaze_false(self):
        from tobii_pytracker.analyze.bbox.parsing import bbox_contains_gaze
        
        bbox = {"cx": 100, "cy": 100, "w": 50, "h": 50}
        result = bbox_contains_gaze(bbox, np.array([200]), np.array([200]))
        self.assertFalse(np.any(result))

    def test_resolve_image_path_missing(self):
        from tobii_pytracker.analyze.bbox.parsing import resolve_image_path
        
        with self.assertRaises(FileNotFoundError):
            resolve_image_path("/nonexistent/path.jpg")

    def test_resolve_image_path_empty(self):
        from tobii_pytracker.analyze.bbox.parsing import resolve_image_path
        
        with self.assertRaises(ValueError):
            resolve_image_path("")

    def test_extract_gaze_from_gaze_data(self):
        from tobii_pytracker.analyze.bbox.parsing import extract_gaze
        
        row = pd.Series({})
        slide_data = pd.DataFrame({
            "avg_gaze_x": [100.0, 150.0],
            "avg_gaze_y": [200.0, 250.0]
        })
        
        result = extract_gaze(row, slide_data)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
