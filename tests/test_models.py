from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from PIL import Image

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.analyze.models import (
    BaseAnalyzer,
    HeatmapAnalyzer,
    FocusMapAnalyzer,
    SaccadeAnalyzer,
    FixationAnalyzer,
    EntropyAnalyzer,
)


class TestBaseAnalyzer(unittest.TestCase):
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            analyzer = BaseAnalyzer(tmp_path)
            self.assertEqual(analyzer.output_folder, tmp_path)
            self.assertIsNone(analyzer.results)

    def test_analyze_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = BaseAnalyzer(Path(tmp))
            with self.assertRaises(NotImplementedError):
                analyzer.analyze()

    def test_plot_analysis_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = BaseAnalyzer(Path(tmp))
            with self.assertRaises(NotImplementedError):
                analyzer.plot_analysis()

    def test_save_results_with_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = BaseAnalyzer(Path(tmp))
            analyzer.results = None
            analyzer.save_results()

    def test_save_results_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            analyzer = BaseAnalyzer(tmp_path)
            analyzer.results = pd.DataFrame({"col": [1, 2, 3]})
            analyzer.save_results("test_results.json")
            self.assertTrue((tmp_path / "test_results.json").exists())

    def test_save_results_default_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            analyzer = BaseAnalyzer(tmp_path)
            analyzer.results = pd.DataFrame({"col": [1, 2, 3]})
            analyzer.save_results()
            self.assertTrue((tmp_path / "BaseAnalyzer_results.json").exists())


class TestHeatmapAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_creates_output_folder(self):
        analyzer = HeatmapAnalyzer(self.tmp_path / "analysis")
        self.assertTrue((self.tmp_path / "analysis").exists())

    def test_analyze_global(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="global")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["avg_gaze_x"].iloc[0], 20.0)
        self.assertAlmostEqual(result["avg_gaze_y"].iloc[0], 25.0)
        self.assertEqual(result["gaze_count"].iloc[0], 3)

    def test_analyze_per_set(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="set")
        self.assertEqual(len(result), 2)
        self.assertEqual(result["set_name"].tolist(), ["s1", "s2"])
        self.assertEqual(result["gaze_count"].tolist(), [2, 1])

    def test_analyze_per_slide(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="slide")
        self.assertEqual(len(result), 3)
        self.assertEqual(result["gaze_count"].tolist(), [1, 1, 1])

    def test_analyze_invalid_per_mode(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10],
            "avg_gaze_y": [15],
            "set_name": ["s1"],
            "slide_index": [0],
        })
        with self.assertRaises(ValueError):
            analyzer.analyze(data, per="invalid")

    def test_plot_analysis_screenshot_not_found(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10],
            "avg_gaze_y": [15],
        })
        with self.assertRaises(FileNotFoundError):
            analyzer.plot_analysis(data, self.tmp_path / "nonexistent.png")

    def test_plot_analysis_with_valid_screenshot(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        img_path = self.tmp_path / "test.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        data = pd.DataFrame({
            "avg_gaze_x": [10.0, 20.0, -5.0],
            "avg_gaze_y": [5.0, 10.0, -10.0],
        })

        plot_path = self.tmp_path / "heatmap.png"
        with patch("tobii_pytracker.analyze.models.plt.show"):
            analyzer.plot_analysis(data, img_path, save_path=plot_path, show=False)
        self.assertTrue(plot_path.exists())

    def test_plot_analysis_with_custom_title(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        img_path = self.tmp_path / "test.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        data = pd.DataFrame({
            "avg_gaze_x": [10.0],
            "avg_gaze_y": [5.0],
        })

        with patch("tobii_pytracker.analyze.models.plt.show"):
            with patch("tobii_pytracker.analyze.models.plt.subplots") as mock_plot:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_plot.return_value = (mock_fig, mock_ax)

                analyzer.plot_analysis(data, img_path, title="Custom Title", show=False)
                mock_ax.set_title.assert_called_once()
                call_args = mock_ax.set_title.call_args[0][0]
                self.assertIn("Custom Title", call_args)

    def test_plot_analysis_with_flip_y_false(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        img_path = self.tmp_path / "test.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        data = pd.DataFrame({
            "avg_gaze_x": [10.0],
            "avg_gaze_y": [5.0],
        })

        plot_path = self.tmp_path / "heatmap.png"
        with patch("tobii_pytracker.analyze.models.plt.show"):
            analyzer.plot_analysis(data, img_path, flip_y=False, save_path=plot_path, show=False)
        self.assertTrue(plot_path.exists())

    def test_plot_analysis_with_custom_params(self):
        analyzer = HeatmapAnalyzer(self.tmp_path)
        img_path = self.tmp_path / "test.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        data = pd.DataFrame({
            "avg_gaze_x": [10.0, 20.0],
            "avg_gaze_y": [5.0, 10.0],
        })

        plot_path = self.tmp_path / "heatmap.png"
        with patch("tobii_pytracker.analyze.models.plt.show"):
            analyzer.plot_analysis(
                data,
                img_path,
                blur_sigma=5.0,
                bins=50,
                cmap="cool",
                alpha=0.8,
                save_path=plot_path,
                show=False,
            )
        self.assertTrue(plot_path.exists())


class TestFocusMapAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_creates_output_folder(self):
        analyzer = FocusMapAnalyzer(self.tmp_path / "analysis")
        self.assertTrue((self.tmp_path / "analysis").exists())

    def test_analyze_global(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="global")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["avg_gaze_x"].iloc[0], 20.0)
        self.assertAlmostEqual(result["avg_gaze_y"].iloc[0], 25.0)
        self.assertEqual(result["gaze_count"].iloc[0], 3)

    def test_analyze_per_set(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="set")
        self.assertEqual(len(result), 2)

    def test_analyze_per_slide(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10, 20, 30],
            "avg_gaze_y": [15, 25, 35],
            "set_name": ["s1", "s1", "s2"],
            "slide_index": [0, 1, 0],
        })
        result = analyzer.analyze(data, per="slide")
        self.assertEqual(len(result), 3)

    def test_analyze_invalid_per_mode(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10],
            "avg_gaze_y": [15],
            "set_name": ["s1"],
            "slide_index": [0],
        })
        with self.assertRaises(ValueError):
            analyzer.analyze(data, per="invalid")

    def test_plot_analysis_screenshot_not_found(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10],
            "avg_gaze_y": [15],
        })
        with self.assertRaises(FileNotFoundError):
            analyzer.plot_analysis(data, self.tmp_path / "nonexistent.png")

    def test_plot_analysis_with_valid_screenshot(self):
        analyzer = FocusMapAnalyzer(self.tmp_path)
        img_path = self.tmp_path / "test.png"
        Image.new("RGB", (200, 100), color="white").save(img_path)

        data = pd.DataFrame({
            "avg_gaze_x": [10.0, 20.0],
            "avg_gaze_y": [5.0, 10.0],
        })

        plot_path = self.tmp_path / "focus_map.png"
        with patch("tobii_pytracker.analyze.models.plt.show"):
            analyzer.plot_analysis(data, img_path, save_path=plot_path, show=False)
        self.assertTrue(plot_path.exists())


class TestSaccadeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_with_defaults(self):
        analyzer = SaccadeAnalyzer(self.tmp_path)
        self.assertEqual(analyzer.method, "ivt")
        self.assertEqual(analyzer.velocity_threshold, 100.0)
        self.assertFalse(analyzer.filter_micro_saccades)

    def test_init_with_custom_params(self):
        analyzer = SaccadeAnalyzer(
            self.tmp_path,
            method="acceleration",
            velocity_threshold=150.0,
            filter_micro_saccades=True,
            micro_saccade_threshold=20.0,
        )
        self.assertEqual(analyzer.method, "acceleration")
        self.assertEqual(analyzer.velocity_threshold, 150.0)
        self.assertTrue(analyzer.filter_micro_saccades)
        self.assertEqual(analyzer.micro_saccade_threshold, 20.0)

    def test_analyze_empty_data(self):
        analyzer = SaccadeAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [],
            "avg_gaze_y": [],
            "system_time": [],
            "set_name": [],
            "slide_index": [],
        })
        result = analyzer.analyze(data)
        self.assertEqual(len(result), 0)

    def test_analyze_single_sample(self):
        analyzer = SaccadeAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10.0],
            "avg_gaze_y": [5.0],
            "system_time": [0.0],
            "set_name": ["s1"],
            "slide_index": [0],
        })
        result = analyzer.analyze(data)
        self.assertEqual(len(result), 0)

    def test_analyze_with_saccade_ivt_method(self):
        analyzer = SaccadeAnalyzer(self.tmp_path, method="ivt", velocity_threshold=10.0)
        data = pd.DataFrame({
            "avg_gaze_x": [0.0, 10.0, 20.0, 20.0],
            "avg_gaze_y": [0.0, 0.0, 0.0, 0.0],
            "system_time": [0.0, 0.01, 0.02, 0.03],
            "set_name": ["s1", "s1", "s1", "s1"],
            "slide_index": [0, 0, 0, 0],
        })
        result = analyzer.analyze(data)
        self.assertIsNotNone(analyzer.results)
        self.assertIsInstance(result, pd.DataFrame)

    def test_analyze_with_saccade_acceleration_method(self):
        analyzer = SaccadeAnalyzer(self.tmp_path, method="acceleration", acceleration_threshold=1000.0)
        data = pd.DataFrame({
            "avg_gaze_x": [0.0, 1.0, 4.0, 9.0],
            "avg_gaze_y": [0.0, 1.0, 4.0, 9.0],
            "system_time": [0.0, 0.01, 0.02, 0.03],
            "set_name": ["s1", "s1", "s1", "s1"],
            "slide_index": [0, 0, 0, 0],
        })
        result = analyzer.analyze(data)
        self.assertIsNotNone(analyzer.results)

    def test_analyze_invalid_method(self):
        analyzer = SaccadeAnalyzer(self.tmp_path, method="invalid")
        data = pd.DataFrame({
            "avg_gaze_x": [0.0, 10.0],
            "avg_gaze_y": [0.0, 0.0],
            "system_time": [0.0, 0.01],
            "set_name": ["s1", "s1"],
            "slide_index": [0, 0],
        })
        with self.assertRaises(ValueError):
            analyzer.analyze(data)

    def test_analyze_with_micro_saccade_filtering(self):
        analyzer = SaccadeAnalyzer(
            self.tmp_path,
            method="ivt",
            velocity_threshold=10.0,
            filter_micro_saccades=True,
            micro_saccade_threshold=50.0,
        )
        data = pd.DataFrame({
            "avg_gaze_x": [0.0, 5.0, 10.0, 10.0],
            "avg_gaze_y": [0.0, 5.0, 10.0, 10.0],
            "system_time": [0.0, 0.01, 0.02, 0.03],
            "set_name": ["s1", "s1", "s1", "s1"],
            "slide_index": [0, 0, 0, 0],
        })
        result = analyzer.analyze(data)
        self.assertIsNotNone(analyzer.results)


class TestFixationAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_with_defaults(self):
        analyzer = FixationAnalyzer(self.tmp_path)
        self.assertEqual(analyzer.dispersion_threshold, 50.0)
        self.assertEqual(analyzer.min_duration, 0.1)

    def test_init_with_custom_params(self):
        analyzer = FixationAnalyzer(
            self.tmp_path,
            dispersion_threshold=100.0,
            min_duration=0.1,
        )
        self.assertEqual(analyzer.dispersion_threshold, 100.0)
        self.assertEqual(analyzer.min_duration, 0.1)

    def test_analyze_empty_data(self):
        analyzer = FixationAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [],
            "avg_gaze_y": [],
            "system_time": [],
            "set_name": [],
            "slide_index": [],
        })
        result = analyzer.analyze(data)
        self.assertEqual(len(result), 0)

    def test_analyze_single_sample(self):
        analyzer = FixationAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": [10.0],
            "avg_gaze_y": [5.0],
            "system_time": [0.0],
            "set_name": ["s1"],
            "slide_index": [0],
        })
        result = analyzer.analyze(data)
        self.assertEqual(len(result), 0)

    def test_analyze_with_fixation(self):
        analyzer = FixationAnalyzer(self.tmp_path, dispersion_threshold=20.0, min_duration=0.01)
        data = pd.DataFrame({
            "avg_gaze_x": [10.0, 10.5, 11.0, 10.8, 20.0],
            "avg_gaze_y": [5.0, 5.5, 5.2, 5.3, 5.0],
            "system_time": [0.0, 0.01, 0.02, 0.03, 0.04],
            "set_name": ["s1", "s1", "s1", "s1", "s1"],
            "slide_index": [0, 0, 0, 0, 0],
        })
        result = analyzer.analyze(data)
        self.assertIsNotNone(analyzer.results)
        self.assertIsInstance(result, pd.DataFrame)


class TestEntropyAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init(self):
        analyzer = EntropyAnalyzer(self.tmp_path)
        self.assertEqual(analyzer.output_folder, self.tmp_path)
        self.assertIsNone(analyzer.results)

    def test_analyze_global_entropy(self):
        analyzer = EntropyAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": np.random.rand(100) * 100,
            "avg_gaze_y": np.random.rand(100) * 100,
            "set_name": ["s1"] * 100,
            "slide_index": [0] * 100,
        })
        result = analyzer.analyze(data, per="global")
        self.assertEqual(len(result), 1)
        self.assertIn("entropy", result.columns)

    def test_analyze_per_set_entropy(self):
        analyzer = EntropyAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": np.random.rand(20) * 100,
            "avg_gaze_y": np.random.rand(20) * 100,
            "set_name": ["s1"] * 10 + ["s2"] * 10,
            "slide_index": [0] * 20,
        })
        result = analyzer.analyze(data, per="set")
        self.assertEqual(len(result), 2)
        self.assertIn("entropy", result.columns)

    def test_analyze_per_slide_entropy(self):
        analyzer = EntropyAnalyzer(self.tmp_path)
        data = pd.DataFrame({
            "avg_gaze_x": np.random.rand(30) * 100,
            "avg_gaze_y": np.random.rand(30) * 100,
            "set_name": ["s1"] * 15 + ["s2"] * 15,
            "slide_index": [0] * 5 + [1] * 10 + [2] * 15,
        })
        result = analyzer.analyze(data, per="slide")
        self.assertGreaterEqual(len(result), 2)
        self.assertIn("entropy", result.columns)


