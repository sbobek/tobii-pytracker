import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.support import bootstrap_test_environment
bootstrap_test_environment()

from tobii_pytracker.analyze.models import BaseAnalyzer


class TestBaseAnalyzerSaveResults(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_results_empty_dataframe(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame()
        analyzer.save_results('empty.json')
        
        result_file = self.output_folder / 'empty.json'
        self.assertTrue(result_file.exists())

    def test_save_results_large_dataframe(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame({
            'col1': list(range(100)),
            'col2': list(range(100, 200)),
            'col3': list(range(200, 300))
        })
        analyzer.save_results('large.json')
        
        result_file = self.output_folder / 'large.json'
        self.assertTrue(result_file.exists())

    def test_save_results_with_special_characters(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Čàrl'],
            'value': [1, 2, 3]
        })
        analyzer.save_results('special.json')
        
        result_file = self.output_folder / 'special.json'
        self.assertTrue(result_file.exists())

    def test_save_results_with_null_values(self):
        analyzer = BaseAnalyzer(self.output_folder)
        analyzer.results = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None]
        })
        analyzer.save_results('nulls.json')
        
        result_file = self.output_folder / 'nulls.json'
        self.assertTrue(result_file.exists())


class TestBaseAnalyzerNormalization(unittest.TestCase):

    def test_normalize_empty_dataframe(self):
        data = pd.DataFrame({'slide_index': []})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(len(result), 0)

    def test_normalize_all_na_values(self):
        data = pd.DataFrame({'slide_index': [None, None, None]})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertTrue(result['slide_index'].isna().all())

    def test_normalize_mixed_types(self):
        data = pd.DataFrame({'slide_index': [0, '1', 2.0, None]})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')

    def test_normalize_float_values(self):
        data = pd.DataFrame({'slide_index': [0.0, 1.0, 2.0]})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')

    def test_normalize_negative_values(self):
        data = pd.DataFrame({'slide_index': [-1, -2, 0, 1]})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')

    def test_normalize_very_large_values(self):
        data = pd.DataFrame({'slide_index': [999999, 1000000, 1000001]})
        result = BaseAnalyzer._normalize_slide_index_column(data)
        self.assertEqual(result['slide_index'].dtype.name, 'Int64')


class TestBaseAnalyzerFiltering(unittest.TestCase):

    def test_filter_empty_dataframe(self):
        data = pd.DataFrame({'set_name': [], 'slide_index': []})
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 0)

    def test_filter_all_rows(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 1, 2]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s2')
        self.assertEqual(len(result), 0)

    def test_filter_with_non_string_set_name(self):
        data = pd.DataFrame({
            'set_name': [1, 2, 3],
            'slide_index': [0, 1, 2]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name=1)
        self.assertEqual(len(result), 1)

    def test_filter_with_float_slide_index(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0.0, 1.0, 2.0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=1)
        self.assertEqual(len(result), 1)

    def test_filter_set_and_slide_no_match_set(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's2'],
            'slide_index': [0, 0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1', slide_index=1)
        self.assertEqual(len(result), 0)

    def test_filter_with_nan_values(self):
        data = pd.DataFrame({
            'set_name': ['s1', np.nan, 's1'],
            'slide_index': [0, 0, 1]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name='s1')
        self.assertEqual(len(result), 2)

    def test_filter_coercion_with_invalid_strings(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1'],
            'slide_index': ['invalid', '1']
        })
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 0)

    def test_filter_zero_slide_index(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [0, 1, 0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=0)
        self.assertEqual(len(result), 2)

    def test_filter_negative_slide_index(self):
        data = pd.DataFrame({
            'set_name': ['s1', 's1', 's1'],
            'slide_index': [-1, 0, 1]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, slide_index=-1)
        self.assertEqual(len(result), 1)


class TestBaseAnalyzerStringConversion(unittest.TestCase):

    def test_filter_set_name_int_to_str(self):
        data = pd.DataFrame({
            'set_name': [1, 2, 3],
            'slide_index': [0, 0, 0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['set_name'], 2)

    def test_filter_set_name_bool(self):
        data = pd.DataFrame({
            'set_name': [True, False, True],
            'slide_index': [0, 0, 0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name=True)
        self.assertEqual(len(result), 2)

    def test_filter_set_name_float(self):
        data = pd.DataFrame({
            'set_name': [1.5, 2.5, 1.5],
            'slide_index': [0, 0, 0]
        })
        result = BaseAnalyzer._filter_set_and_slide(data, set_name=1.5)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
