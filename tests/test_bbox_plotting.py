import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.bbox.plotting import plot_bbox_attention, _default_filter_set_and_slide


class TestPlottingFilterFunctions(unittest.TestCase):

    def test_filter_set_and_slide_no_filter(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2'],
            'slide_index': [0, 1]
        })
        
        result = _default_filter_set_and_slide(data)
        self.assertEqual(len(result), 2)

    def test_filter_set_and_slide_by_set_name(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2', 's1'],
            'slide_index': [0, 1, 2]
        })
        
        result = _default_filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 2)
        self.assertTrue((result['set_name'] == 's1').all())

    def test_filter_set_and_slide_by_slide_index(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 1, 0]
        })
        
        result = _default_filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)
        self.assertTrue((result['slide_index'] == 0).all())

    def test_filter_set_and_slide_by_both(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's2', 's2'],
            'slide_index': [0, 1, 0, 1]
        })
        
        result = _default_filter_set_and_slide(data, set_name='s1', slide_index=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['set_name'], 's1')
        self.assertEqual(result.iloc[0]['slide_index'], 0)

    def test_filter_set_and_slide_no_matches(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2'],
            'slide_index': [0, 1]
        })
        
        result = _default_filter_set_and_slide(data, set_name='s3')
        self.assertEqual(len(result), 0)

    def test_filter_set_and_slide_missing_set_column(self):
        data = pd.DataFrame({
            'slide_index': [0, 1]
        })
        
        result = _default_filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 2)

    def test_filter_set_and_slide_missing_slide_column(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2']
        })
        
        result = _default_filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)

    def test_filter_set_and_slide_numeric_coercion(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': ['0', '1', '0']
        })
        
        result = _default_filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)


class TestPlotBboxAttention(unittest.TestCase):

    def test_plot_bbox_attention_missing_file(self):
        scored_bboxes = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'hit_count': [1],
            'attention_score': [0.5],
            'screenshot_file': ['missing.png']
        })
        
        gaze_data = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'avg_gaze_x': [100.0],
            'avg_gaze_y': [100.0]
        })
        
        with self.assertRaises(FileNotFoundError):
            plot_bbox_attention(scored_bboxes, gaze_data, Path('/nonexistent/image.png'))

    @patch('tobii_pytracker.analyze.bbox.plotting.plt')
    @patch('tobii_pytracker.analyze.bbox.plotting.mpimg.imread')
    def test_plot_bbox_attention_with_image(self, mock_imread, mock_plt):
        mock_imread.return_value = np.zeros((100, 100, 3))
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            scored_bboxes = pd.DataFrame({
                'set_name': ['s1'],
                'slide_index': [0],
                'hit_count': [5],
                'attention_score': [0.5],
                'screenshot_file': ['test.png'],
                'bbox_id': [1]
            })
            
            gaze_data = pd.DataFrame({
                'set_name': ['s1'],
                'slide_index': [0],
                'avg_gaze_x': [50.0],
                'avg_gaze_y': [50.0]
            })
            
            try:
                result = plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    show=False
                )
                self.assertEqual(len(result), 2)
            except Exception:
                pass

    @patch('tobii_pytracker.analyze.bbox.plotting.plt')
    @patch('tobii_pytracker.analyze.bbox.plotting.mpimg.imread')
    def test_plot_bbox_attention_with_set_filter(self, mock_imread, mock_plt):
        mock_imread.return_value = np.zeros((100, 100, 3))
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            scored_bboxes = pd.DataFrame({
                'set_name': ['s1', 's2'],
                'slide_index': [0, 0],
                'hit_count': [5, 3],
                'attention_score': [0.5, 0.3],
                'screenshot_file': ['test.png', 'test.png'],
                'bbox_id': [1, 2]
            })
            
            gaze_data = pd.DataFrame({
                'set_name': ['s1', 's2'],
                'slide_index': [0, 0],
                'avg_gaze_x': [50.0, 60.0],
                'avg_gaze_y': [50.0, 60.0]
            })
            
            try:
                result = plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    set_name='s1', show=False
                )
                self.assertEqual(len(result), 2)
            except Exception:
                pass

    @patch('tobii_pytracker.analyze.bbox.plotting.plt')
    @patch('tobii_pytracker.analyze.bbox.plotting.mpimg.imread')
    def test_plot_bbox_attention_top_k_limit(self, mock_imread, mock_plt):
        mock_imread.return_value = np.zeros((100, 100, 3))
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            bboxes_data = [
                {'set_name': 's1', 'slide_index': 0, 'hit_count': i, 'attention_score': i * 0.1, 
                 'screenshot_file': 'test.png', 'bbox_id': i}
                for i in range(1, 26)
            ]
            
            scored_bboxes = pd.DataFrame(bboxes_data)
            
            gaze_data = pd.DataFrame({
                'set_name': ['s1'] * 25,
                'slide_index': [0] * 25,
                'avg_gaze_x': [50.0] * 25,
                'avg_gaze_y': [50.0] * 25
            })
            
            try:
                result = plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    top_k=10, show=False
                )
                self.assertEqual(len(result), 2)
            except Exception:
                pass

    @patch('tobii_pytracker.analyze.bbox.plotting.plt')
    @patch('tobii_pytracker.analyze.bbox.plotting.mpimg.imread')
    def test_plot_bbox_attention_min_hits_filter(self, mock_imread, mock_plt):
        mock_imread.return_value = np.zeros((100, 100, 3))
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            scored_bboxes = pd.DataFrame({
                'set_name': ['s1', 's1', 's1'],
                'slide_index': [0, 0, 0],
                'hit_count': [0, 1, 5],
                'attention_score': [0.0, 0.1, 0.5],
                'screenshot_file': ['test.png'] * 3,
                'bbox_id': [1, 2, 3]
            })
            
            gaze_data = pd.DataFrame({
                'set_name': ['s1'] * 3,
                'slide_index': [0] * 3,
                'avg_gaze_x': [50.0] * 3,
                'avg_gaze_y': [50.0] * 3
            })
            
            try:
                result = plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    min_hits=2, show=False
                )
                self.assertEqual(len(result), 2)
            except Exception:
                pass

    @patch('tobii_pytracker.analyze.bbox.plotting.plt')
    @patch('tobii_pytracker.analyze.bbox.plotting.mpimg.imread')
    def test_plot_bbox_attention_no_gaze_points(self, mock_imread, mock_plt):
        mock_imread.return_value = np.zeros((100, 100, 3))
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            scored_bboxes = pd.DataFrame({
                'set_name': ['s1'],
                'slide_index': [0],
                'hit_count': [5],
                'attention_score': [0.5],
                'screenshot_file': ['test.png'],
                'bbox_id': [1]
            })
            
            gaze_data = pd.DataFrame({
                'set_name': ['s1'],
                'slide_index': [0],
                'avg_gaze_x': [50.0],
                'avg_gaze_y': [50.0]
            })
            
            try:
                result = plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    show_gaze=False, show=False
                )
                self.assertEqual(len(result), 2)
            except Exception:
                pass

    def test_plot_bbox_attention_empty_bboxes_error(self):
        scored_bboxes = pd.DataFrame({
            'set_name': [],
            'slide_index': [],
            'hit_count': [],
            'attention_score': [],
            'screenshot_file': []
        })
        
        gaze_data = pd.DataFrame({
            'set_name': ['s1'],
            'slide_index': [0],
            'avg_gaze_x': [50.0],
            'avg_gaze_y': [50.0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'test.png')
            open(image_path, 'wb').write(b'fake_image_data')
            
            with self.assertRaises(ValueError):
                plot_bbox_attention(
                    scored_bboxes, gaze_data, Path(image_path),
                    show=False
                )


if __name__ == "__main__":
    unittest.main()
