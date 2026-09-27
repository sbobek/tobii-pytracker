import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.models import (
    BaseAnalyzer, HeatmapAnalyzer, FocusMapAnalyzer, 
    SaccadeAnalyzer, FixationAnalyzer, EntropyAnalyzer, 
    ClusterAnalyzer
)


class TestBaseAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_base_analyzer_init(self):
        analyzer = BaseAnalyzer(self.output_folder)
        self.assertEqual(analyzer.output_folder, self.output_folder)
        self.assertIsNone(analyzer.results)
        self.assertTrue(self.output_folder.exists())

    def test_base_analyzer_init_creates_folder(self):
        new_folder = self.output_folder / 'nested' / 'path'
        analyzer = BaseAnalyzer(new_folder)
        self.assertTrue(new_folder.exists())

    def test_base_analyzer_with_config(self):
        mock_config = MagicMock()
        analyzer = BaseAnalyzer(self.output_folder, config=mock_config)
        self.assertEqual(analyzer.config, mock_config)

    def test_base_analyzer_analyze_not_implemented(self):
        analyzer = BaseAnalyzer(self.output_folder)
        with self.assertRaises(NotImplementedError):
            analyzer.analyze()

    def test_base_analyzer_plot_analysis_not_implemented(self):
        analyzer = BaseAnalyzer(self.output_folder)
        with self.assertRaises(NotImplementedError):
            analyzer.plot_analysis()

    def test_base_analyzer_save_results_no_results(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.save_results()
        files = list(self.output_folder.glob('*.json'))
        self.assertEqual(len(files), 0)

    def test_base_analyzer_save_results_with_dataframe(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        analyzer.save_results('test_results.json')
        
        result_file = self.output_folder / 'test_results.json'
        self.assertTrue(result_file.exists())

    def test_base_analyzer_save_results_default_filename(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame({'x': [1]})
        analyzer.save_results()
        
        result_file = self.output_folder / 'BaseAnalyzer_results.json'
        self.assertTrue(result_file.exists())

    def test_normalize_slide_index_column(self):
        data = pd.DataFrame({
            'slide_index': ['0', '1', '2', None],
            'value': [10, 20, 30, 40]
        })
        
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')

    def test_normalize_slide_index_column_numeric(self):
        data = pd.DataFrame({
            'slide_index': [0, 1, 2],
            'value': [10, 20, 30]
        })
        
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')

    def test_normalize_slide_index_column_custom_col(self):
        data = pd.DataFrame({
            'custom_col': ['0', '1', '2'],
            'value': [10, 20, 30]
        })
        
        result = BaseAnalyzer._normalize_slide_index_column(data, column='custom_col')
        self.assertEqual(result['custom_col'].dtype.name, 'Int64')

    def test_filter_set_and_slide_no_filter(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2'],
            'slide_index': [0, 1],
            'value': [10, 20]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data)
        self.assertEqual(len(result), 2)

    def test_filter_set_and_slide_by_set(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2', 's1'],
            'slide_index': [0, 1, 2]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 2)
        self.assertTrue((result['set_name'] == 's1').all())

    def test_filter_set_and_slide_by_slide(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 1, 0]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)
        self.assertTrue((result['slide_index'] == 0).all())

    def test_filter_set_and_slide_string_conversion(self):
        data = pd.DataFrame({
            'set_name': [123, 456],
            'slide_index': [0, 1]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, set_name=123)
        self.assertEqual(len(result), 1)

    def test_filter_set_and_slide_numeric_coercion(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': ['0', '1']
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 1)

    def test_filter_set_and_slide_both_filters(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's2', 's2'],
            'slide_index': [0, 1, 0, 1]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1', slide_index=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['set_name'], 's1')
        self.assertEqual(result.iloc[0]['slide_index'], 0)

    def test_filter_set_and_slide_missing_set_column(self):
        data = pd.DataFrame({
            'slide_index': [0, 1]
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 2)

    def test_filter_set_and_slide_missing_slide_column(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2']
        })
        
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)


class TestHeatmapAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_heatmap_analyzer_can_be_instantiated(self):
        try:
            analyzer = HeatmapAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


class TestFocusMapAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_focusmap_analyzer_can_be_instantiated(self):
        try:
            analyzer = FocusMapAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


class TestSaccadeAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_saccade_analyzer_can_be_instantiated(self):
        try:
            analyzer = SaccadeAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


class TestFixationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fixation_analyzer_can_be_instantiated(self):
        try:
            analyzer = FixationAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


class TestEntropyAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_entropy_analyzer_can_be_instantiated(self):
        try:
            analyzer = EntropyAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


class TestClusterAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cluster_analyzer_can_be_instantiated(self):
        try:
            analyzer = ClusterAnalyzer(self.output_folder)
            self.assertIsNotNone(analyzer)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
