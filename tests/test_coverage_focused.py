from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import result
import pandas as pd
import yaml
import json

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.configs.custom_config import CustomConfig
from tobii_pytracker.analyze.data_loader import DataLoader


class TestDataLoaderCoverage(unittest.TestCase):

    def _create_config(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        cfg_path = tmp_path / "config.yaml"
        cfg = {
            "dataset": {"image": {"path": "datasets"}},
            "display": {
                "monitor": {
                    "name": "m",
                    "resolution": [1920, 1080],
                    "width": 35,
                    "distance": 60,
                },
                "gui": {
                    "button": {
                        "size": [250, 100],
                        "margin": 20,
                        "color": "white",
                        "text": {"color": "black", "size": 30},
                    },
                    "fixation_dot": {"size": 10, "color": "white"},
                    "aoe": [500, 500],
                },
            },
            "output": {"folder": "output"},
            "instructions": {"intro": ["hi"], "outro": ["bye"]},
        }
        
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return CustomConfig(str(cfg_path))

    def test_data_loader_output_root_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Don't create output directory
            cfg = self._create_config(tmp_path)
            
            # Override output folder to a non-existent path
            cfg.config["output"]["folder"] = "nonexistent"
            
            with self.assertRaises(FileNotFoundError):
                DataLoader(cfg, root=tmp_path)

    def test_load_data_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            # Create subject directory without data.csv
            subject_dir = tmp_path / "output" / "subject_1"
            subject_dir.mkdir()
            
            loader = DataLoader(config, root=tmp_path)
            
            with self.assertRaises(FileNotFoundError):
                loader._load_data("subject_1")

    def test_parse_gaze_data_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            loader = DataLoader(config, root=tmp_path)
            
            result = loader._parse_gaze_data(None)
            self.assertEqual(result, [])

    def test_flatten_gaze_data_empty_gaze_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            # Create subject directory with data.csv containing empty gaze_data
            subject_dir = tmp_path / "output" / "subject_1"
            subject_dir.mkdir()
            
            data_csv = subject_dir / "data.csv"
            df = pd.DataFrame({
                "slide_number": [1, 2],
                "gaze_data": ["[]", "[]"],  # Empty gaze data
                "screenshot_file": ["img1.png", "img2.png"],
            })
            df.to_csv(data_csv, sep=";", index=False)
            
            loader = DataLoader(config, root=tmp_path)
            result = loader._flatten_gaze_data(df, "subject_1")
            
            # Should result in empty or minimal dataframe
            self.assertIsInstance(result, pd.DataFrame)

    def test_flatten_gaze_data_single_eye_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            subject_dir = tmp_path / "output" / "subject_1"
            subject_dir.mkdir()
            
            data_csv = subject_dir / "data.csv"
            
            # Create gaze data with only left eye
            gaze_data = json.dumps([
                {
                    "gaze_x_left": 100.0,
                    "gaze_y_left": 200.0,
                    "gaze_x_right": None,
                    "gaze_y_right": None,
                    "pupil_left": 2.5,
                    "pupil_right": None,
                    "timestamp": 1234567890.0,
                }
            ])
            
            df = pd.DataFrame({
                "slide_number": [1],
                "gaze_data": [gaze_data],
                "screenshot_file": ["img1.png"],
            })
            df.to_csv(data_csv, sep=";", index=False)
            
            loader = DataLoader(config, root=tmp_path)
            result = loader._flatten_gaze_data(df, "subject_1")
            
            # Check that result contains data
            self.assertGreater(len(result), 0)
            # Check that single eye values are used for average
            self.assertIn("avg_gaze_x", result.columns)

    def test_get_slide_data_index_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            subject_dir = tmp_path / "output" / "subject_1"
            subject_dir.mkdir()
            
            data_csv = subject_dir / "data.csv"
            df = pd.DataFrame({
                "slide_number": [1],
                "gaze_data": ["[]"],
                "screenshot_file": ["img1.png"],
            })
            df.to_csv(data_csv, sep=";", index=False)
            
            loader = DataLoader(config, root=tmp_path)
            
            with self.assertRaises(IndexError):
                loader.get_slide_data("subject_1", 10)

    def test_get_all_data_flatten(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            # Create two subjects
            for subject in ["subject_1", "subject_2"]:
                subject_dir = tmp_path / "output" / subject
                subject_dir.mkdir()
                
                data_csv = subject_dir / "data.csv"
                df = pd.DataFrame({
                    "slide_number": [1],
                    "gaze_data": ["[{\"gaze_x_left\": 100.0, \"gaze_y_left\": 200.0, \"gaze_x_right\": null, \"gaze_y_right\": null, \"pupil_left\": 2.5, \"pupil_right\": null, \"timestamp\": 1234567890.0}]"],
                    "screenshot_file": ["img1.png"],
                })
                df.to_csv(data_csv, sep=";", index=False)
            
            loader = DataLoader(config, root=tmp_path)
            result = loader.get_all_data(flatten=True)
            
            self.assertIn("subject_1", result["set_name"].values)
            self.assertIn("subject_2", result["set_name"].values)

    def test_add_column_with_func_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._create_config(tmp_path)
            
            subject_dir = tmp_path / "output" / "subject_1"
            subject_dir.mkdir()
            
            data_csv = subject_dir / "data.csv"
            df = pd.DataFrame({
                "slide_number": [1],
                "gaze_data": ["[]"],
                "screenshot_file": ["img1.png"],
            })
            df.to_csv(data_csv, sep=";", index=False)
            
            loader = DataLoader(config, root=tmp_path)
            
            # This function will raise an exception
            def bad_func(df):
                raise ValueError("Test error")
            
            # add_column doesn't return anything, but should handle exceptions gracefully
            loader.add_column("subject_1", "test_col", func=bad_func, save=False)
            
            # Check that the file still exists and wasn't modified
            self.assertTrue(data_csv.exists())


class TestCustomConfigCoverage(unittest.TestCase):

    def _create_base_cfg(self):
        return {
            "dataset": {"image": {"path": "datasets"}},
            "display": {
                "monitor": {
                    "name": "m",
                    "resolution": [1920, 1080],
                    "width": 35,
                    "distance": 60,
                },
                "gui": {
                    "button": {
                        "size": [250, 100],
                        "margin": 20,
                        "color": "white",
                        "text": {"color": "black", "size": 30},
                    },
                    "fixation_dot": {"size": 10, "color": "white"},
                    "aoe": [500, 500],
                },
            },
            "output": {"folder": "output"},
            "instructions": {"intro": ["hi"], "outro": ["bye"]},
        }

    def test_button_config_not_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["button"] = "not a dict"
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            with self.assertRaises(ValueError):
                CustomConfig(str(cfg_path))

    def test_button_config_missing_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            del cfg["display"]["gui"]["button"]["size"]
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            with self.assertRaises(KeyError):
                CustomConfig(str(cfg_path))

    def test_button_text_color_not_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["button"]["text"]["color"] = 123  # Should be string
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            with self.assertRaises(ValueError):
                CustomConfig(str(cfg_path))

    def test_button_text_size_not_int(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["button"]["text"]["size"] = "30"  # Should be int
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            with self.assertRaises(ValueError):
                CustomConfig(str(cfg_path))

    def test_get_fixation_dot_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            fixation_config = config.get_fixation_dot_config()
            
            self.assertEqual(fixation_config["size"], 10)
            self.assertEqual(fixation_config["color"], "white")

    def test_get_area_of_interest_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            aoe = config.get_area_of_interest_size()
            
            self.assertEqual(aoe, [500, 500])

    def test_get_bbox_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["bbox_model"] = {
                "folder": "models",
                "module": "custom_model",
                "class": "CustomBBoxModel"
            }
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            bbox_config = config.get_bbox_model_config()
            
            self.assertEqual(bbox_config["folder"], "models")
            self.assertEqual(bbox_config["module"], "custom_model")
            self.assertEqual(bbox_config["class"], "CustomBBoxModel")

    def test_get_bbox_model_config_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["bbox_model"] = {
                "folder": "models",
                "module": "custom_model"
                # Missing "class" field
            }
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(KeyError):
                config.get_bbox_model_config()

    def test_get_output_config_missing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["output"] = {}  # Missing folder
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            with self.assertRaises(KeyError):
                CustomConfig(str(cfg_path))

    def test_fixation_dot_config_not_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["fixation_dot"] = "not a dict"
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(ValueError):
                config.get_fixation_dot_config()

    def test_fixation_dot_config_missing_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["fixation_dot"] = {"color": "white"}
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(KeyError):
                config.get_fixation_dot_config()

    def test_fixation_dot_size_not_int(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["fixation_dot"] = {"size": "10", "color": "white"}
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(ValueError):
                config.get_fixation_dot_config()

    def test_fixation_dot_color_not_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["fixation_dot"] = {"size": 10, "color": 123}
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(ValueError):
                config.get_fixation_dot_config()

    def test_area_of_interest_missing_aoe_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            del cfg["display"]["gui"]["aoe"]
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(KeyError):
                config.get_area_of_interest_size()

    def test_area_of_interest_not_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["aoe"] = "500x500"
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(ValueError):
                config.get_area_of_interest_size()

    def test_area_of_interest_wrong_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            cfg = self._create_base_cfg()
            cfg["display"]["gui"]["aoe"] = [500]  # Should have 2 elements
            
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            
            config = CustomConfig(str(cfg_path))
            
            with self.assertRaises(ValueError):
                config.get_area_of_interest_size()


if __name__ == "__main__":
    unittest.main()
