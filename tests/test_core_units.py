from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import numpy as np
import pandas as pd
import yaml

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.configs.custom_config import CustomConfig
from tobii_pytracker.analyze.data_loader import DataLoader
from tobii_pytracker.datasets.custom_dataset import (
    ImageDataset,
    TextDataset,
    TimeSeriesDataset,
    _contour_to_centered_polygon,
    _to_centered_bbox_from_tl,
)
from tobii_pytracker.runtime_models.custom_model import CustomModel
from tobii_pytracker.utils.custom_logger import CustomLogger


class StubWindow:
    def __init__(self, size=(800, 600)):
        self.size = size

    def flip(self):
        return None


class FakeTextStim:
    def __init__(self, win, text, height, wrapWidth=None, alignText=None, color=None, pos=(0, 0), **_kwargs):
        self.text = text
        self.height = height
        width = max(8, int(len(text) * max(height, 1) * 0.5))
        self.boundingBox = (width, int(max(height, 1)))

    def draw(self):
        return None


class FakeShapeStim:
    def __init__(self, win, vertices, closeShape=False, lineWidth=2.0, lineColor="black", fillColor=None):
        self.vertices = vertices
        self.closeShape = closeShape
        self.lineWidth = lineWidth
        self.lineColor = lineColor
        self.fillColor = fillColor

    def draw(self):
        return None


class TestCustomConfig(unittest.TestCase):
    def test_image_config_loads(self):
        config = CustomConfig(str(Path(__file__).parent / "resources" / "test_config.yaml"))

        self.assertEqual(config.dataset_type, "image")
        self.assertEqual(config.get_dataset_path(), "datasets")
        self.assertEqual(config.get_output_config()["folder"], "output")
        self.assertEqual(config.get_monitor_config()["name"], "test_monitor")
        self.assertEqual(config.get_button_config()["size"], [250, 100])
        self.assertEqual(config.get_fixation_dot_config()["size"], 10)

    def test_text_and_time_series_configs_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_csv = tmp_path / "text.csv"
            text_csv.write_text("label,text\nA,Hello world\n", encoding="utf-8")

            ts_csv = tmp_path / "timeseries.csv"
            ts_csv.write_text("index,label,v1,v2\n0,A,1,2\n", encoding="utf-8")

            base = {
                "display": {
                    "monitor": {
                        "name": "m",
                        "resolution": [800, 600],
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

            text_cfg = tmp_path / "text.yaml"
            text_cfg.write_text(
                yaml.safe_dump(
                    {
                        **base,
                        "dataset": {
                            "text": {
                                "path": str(text_csv),
                                "label_column_name": "label",
                                "text_column_name": "text",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            ts_cfg = tmp_path / "ts.yaml"
            ts_cfg.write_text(
                yaml.safe_dump(
                    {
                        **base,
                        "dataset": {
                            "time_series": {
                                "path": str(ts_csv),
                                "label_column_name": "label",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(CustomConfig(str(text_cfg)).dataset_type, "text")
            self.assertEqual(CustomConfig(str(ts_cfg)).dataset_type, "time_series")

    def test_invalid_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_cfg = Path(tmp) / "bad.yaml"
            bad_cfg.write_text("dataset: {}\n", encoding="utf-8")

            with self.assertRaises(KeyError):
                CustomConfig(str(bad_cfg))

    def test_section_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = {
                "dataset": {"image": {"path": "datasets"}},
                "display": {
                    "monitor": {
                        "name": "m",
                        "resolution": [800, 600],
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

            def write_cfg(name, data):
                path = tmp_path / name
                path.write_text(yaml.safe_dump(data), encoding="utf-8")
                return path

            with self.assertRaises(ValueError):
                CustomConfig(
                    str(
                        write_cfg(
                            "bad_dataset.yaml",
                            {
                                **base,
                                "dataset": {"image": {"path": "datasets", "bbox_model": "invalid"}},
                            },
                        )
                    )
                )

            with self.assertRaises(KeyError):
                CustomConfig(
                    str(
                        write_cfg(
                            "bad_monitor.yaml",
                            {
                                **base,
                                "display": {
                                    **base["display"],
                                    "monitor": {"name": "m", "resolution": [800, 600], "width": 35},
                                },
                            },
                        )
                    )
                )

            with self.assertRaises(ValueError):
                CustomConfig(
                    str(
                        write_cfg(
                            "bad_button.yaml",
                            {
                                **base,
                                "display": {
                                    **base["display"],
                                    "gui": {
                                        **base["display"]["gui"],
                                        "button": "not-a-dict",
                                    },
                                },
                            },
                        )
                    )
                )

            fixation_cfg = CustomConfig(
                str(
                    write_cfg(
                        "bad_fixation.yaml",
                        {
                            **base,
                            "display": {
                                **base["display"],
                                "gui": {
                                    **base["display"]["gui"],
                                    "fixation_dot": {"size": "big", "color": "white"},
                                },
                            },
                        },
                    )
                )
            )
            with self.assertRaises(ValueError):
                fixation_cfg.get_fixation_dot_config()

            with self.assertRaises(KeyError):
                CustomConfig(
                    str(
                        write_cfg(
                            "bad_output.yaml",
                            {
                                **base,
                                "output": {},
                            },
                        )
                    )
                )

            with self.assertRaises(ValueError):
                CustomConfig(
                    str(
                        write_cfg(
                            "bad_instructions.yaml",
                            {
                                **base,
                                "instructions": {"intro": "hi", "outro": ["bye"]},
                            },
                        )
                    )
                )


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.output_root = self.root / "output"
        self.output_root.mkdir()

        self.subject_dir = self.output_root / "subject_a"
        self.subject_dir.mkdir()
        (self.subject_dir / "shot.png").write_bytes(b"png")
        (self.subject_dir / "voice.wav").write_bytes(b"wav")

        df = pd.DataFrame([
            {
                "screenshot_file": "shot.png",
                "input_data": "stimulus",
                "classification": "class_a",
                "user_classification": "class_b",
                "model_prediction": "class_c",
                "voice_file": "voice.wav",
                "voice_start_timestamp": "2026-01-01T00:00:00",
                "gaze_data": [
                    {
                        "gaze_x_left": 10,
                        "gaze_x_right": 14,
                        "gaze_y_left": 20,
                        "gaze_y_right": 24,
                        "pupil_left": 3.0,
                        "pupil_right": 5.0,
                        "event_type": "sample",
                        "event_id": 1,
                        "logged_time": 11.0,
                        "system_time": 12.0,
                    }
                ],
            }
        ])
        df.to_csv(self.subject_dir / "data.csv", sep=";", index=False)

        class StubConfig:
            def get_output_config(self):
                return {"folder": "output"}

        self.config = StubConfig()
        self.loader = DataLoader(self.config, root=self.root)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_discovers_subjects(self):
        self.assertEqual(self.loader.get_subjects(), ["subject_a"])

    def test_parse_gaze_data_variants(self):
        self.assertEqual(self.loader._parse_gaze_data([{"a": 1}]), [{"a": 1}])
        self.assertEqual(self.loader._parse_gaze_data('[{"a": 1}]'), [{"a": 1}])
        self.assertEqual(self.loader._parse_gaze_data("[{'a': 1}]"), [{"a": 1}])
        self.assertEqual(self.loader._parse_gaze_data("bad"), [])

    def test_flatten_and_slide_access(self):
        subject_df = self.loader.get_subject_data("subject_a")
        flattened = self.loader.get_subject_data("subject_a", flatten=True)

        self.assertEqual(len(subject_df), 1)
        self.assertEqual(len(flattened), 1)
        row = flattened.iloc[0]
        self.assertEqual(row["avg_gaze_x"], 12)
        self.assertEqual(row["avg_gaze_y"], 22)
        self.assertEqual(row["avg_pupil_size"], 4.0)

        slide = self.loader.get_slide_data("subject_a", 0)
        self.assertEqual(slide["set_name"], "subject_a")
        self.assertEqual(slide["screenshot_path"].name, "shot.png")
        self.assertEqual(slide["gaze_data"][0]["event_id"], 1)

    def test_add_column_and_plot_gaze(self):
        image_path = self.subject_dir / "shot.png"
        Image.new("RGB", (200, 100), color="white").save(image_path)

        df = pd.DataFrame(
            [
                {
                    "screenshot_file": "output/subject_a/shot.png",
                    "input_data": "stimulus",
                    "classification": "class_a",
                    "user_classification": "class_b",
                    "model_prediction": "class_c",
                    "voice_file": None,
                    "voice_start_timestamp": "2026-01-01T00:00:00",
                    "gaze_data": [
                        {
                            "gaze_x_left": 10,
                            "gaze_x_right": 14,
                            "gaze_y_left": 20,
                            "gaze_y_right": 24,
                            "pupil_left": 3.0,
                            "pupil_right": 5.0,
                            "event_type": "sample",
                            "event_id": 1,
                            "logged_time": 11.0,
                            "system_time": 12.0,
                        },
                        {
                            "gaze_x_left": 20,
                            "gaze_x_right": 28,
                            "gaze_y_left": 30,
                            "gaze_y_right": 34,
                            "pupil_left": 4.0,
                            "pupil_right": 6.0,
                            "event_type": "sample",
                            "event_id": 2,
                            "logged_time": 13.0,
                            "system_time": 14.0,
                        },
                    ],
                }
            ]
        )
        df.to_csv(self.subject_dir / "data.csv", sep=";", index=False)

        extra_subject = self.output_root / "subject_b"
        extra_subject.mkdir()

        self.loader.add_column("trial_id", value=7, subjects=["subject_a", "subject_b"])
        updated = pd.read_csv(self.subject_dir / "data.csv", sep=";")
        self.assertEqual(updated["trial_id"].tolist(), [7])

        self.loader.add_column("trial_id", value=9, subjects=["subject_a"], overwrite=False)
        updated_again = pd.read_csv(self.subject_dir / "data.csv", sep=";")
        self.assertEqual(updated_again["trial_id"].tolist(), [7])

        slide = self.loader.get_slide_data("subject_a", 0)
        self.assertIsNone(slide["voice_file"])

        plot_path = self.root / "plot.png"
        self.loader.plot_gaze(
            "subject_a",
            0,
            save_path=plot_path,
            show=False,
            gradient=True,
            draw_both_eyes=True,
            show_legend=True,
        )
        self.assertTrue(plot_path.exists())


class TestDatasetHelpers(unittest.TestCase):
    def test_bbox_to_polygon_helpers(self):
        bbox = _to_centered_bbox_from_tl(10, 20, 30, 50, 100, 100)
        self.assertEqual(bbox["w"], 20.0)
        self.assertEqual(bbox["h"], 30.0)
        self.assertEqual(bbox["cx"], -30.0)
        self.assertEqual(bbox["cy"], 15.0)

        contour = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=float)
        polygon = _contour_to_centered_polygon(contour, 10, 10, 100, 100)
        self.assertEqual(len(polygon), 3)
        self.assertEqual(polygon[0], (-50.0, 50.0))

    def test_image_grid_bbox_remains_rectangular(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_path = tmp_path / "image.png"
            img_path.write_bytes(b"fake")

            class StubConfig:
                def get_area_of_interest_size(self):
                    return (300, 300)

            image_dataset = ImageDataset.__new__(ImageDataset)
            image_dataset.config = StubConfig()
            image_dataset.logger = CustomLogger("INFO", __name__).logger

            grid = ImageDataset._detect_grid(image_dataset, str(img_path), grid_x=3, grid_y=3)
            self.assertEqual(len(grid), 9)
            self.assertIn("cx", grid[0]["bbox"])
            self.assertIn("w", grid[0]["bbox"])

    def test_image_dataset_loads_bboxes_with_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "datasets" / "class_a"
            dataset_dir.mkdir(parents=True)
            image_path = dataset_dir / "sample.png"
            Image.new("RGB", (100, 100), color="white").save(image_path)

            cfg_path = tmp_path / "image.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "bbox_model": {
                            "folder": "fakepkg",
                            "module": "fake_mod",
                            "class": "DummyBBoxModel",
                        },
                        "dataset": {
                            "image": {
                                "path": str(tmp_path / "datasets"),
                            }
                        },
                        "display": {
                            "monitor": {"name": "m", "resolution": [800, 600], "width": 35, "distance": 60},
                            "gui": {
                                "button": {"size": [250, 100], "margin": 20, "color": "white", "text": {"color": "black", "size": 30}},
                                "fixation_dot": {"size": 10, "color": "white"},
                                "aoe": [500, 500],
                            },
                        },
                        "output": {"folder": "output"},
                        "instructions": {"intro": ["hi"], "outro": ["bye"]},
                        "default_detector": "grid",
                    }
                ),
                encoding="utf-8",
            )

            class DummyBBoxModel(CustomModel):
                def prepare_model(self):
                    self.model = "ready"

                def predict(self, input_data):
                    return [input_data]

                def process(self, path):
                    return [("model_box", 0.9, (10, 10, 60, 60))]

            fake_module = type("Module", (), {"DummyBBoxModel": DummyBBoxModel})()

            with patch("tobii_pytracker.datasets.custom_dataset.importlib.import_module", return_value=fake_module):
                dataset = ImageDataset(CustomConfig(str(cfg_path)), calculate_bboxes=True)

            self.assertEqual(dataset.model.model, "ready")
            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(dataset.data[0]["bboxes"][0]["class"], "model_box")
            self.assertGreater(dataset.data[0]["bboxes"][0]["bbox"]["w"], 0)


class TestModalities(unittest.TestCase):
    def test_text_dataset_draws_and_returns_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "text.csv"
            csv_path.write_text("label,text\nA,Hello world\n", encoding="utf-8")
            cfg_path = tmp_path / "text.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "text": {
                                "path": str(csv_path),
                                "label_column_name": "label",
                                "text_column_name": "text",
                            }
                        },
                        "display": {
                            "monitor": {"name": "m", "resolution": [800, 600], "width": 35, "distance": 60},
                            "gui": {
                                "button": {"size": [250, 100], "margin": 20, "color": "white", "text": {"color": "black", "size": 30}},
                                "fixation_dot": {"size": 10, "color": "white"},
                                "aoe": [500, 500],
                            },
                        },
                        "output": {"folder": "output"},
                        "instructions": {"intro": ["hi"], "outro": ["bye"]},
                    }
                ),
                encoding="utf-8",
            )

            dataset = TextDataset(CustomConfig(str(cfg_path)))
            sample = dataset.data[0]
            with patch("tobii_pytracker.datasets.custom_dataset.visual.TextStim", FakeTextStim):
                result = dataset.draw_stimulus(StubWindow(), sample)

            self.assertIn("words", result)
            self.assertIn("lines", result)
            self.assertGreaterEqual(len(result["words"]), 2)
            self.assertGreaterEqual(len(result["lines"]), 1)

    def test_timeseries_dataset_draws_and_returns_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "timeseries.csv"
            csv_path.write_text("index,label,v1,v2,v3,v4\n0,A,1,2,3,4\n", encoding="utf-8")
            cfg_path = tmp_path / "ts.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "time_series": {
                                "path": str(csv_path),
                                "label_column_name": "label",
                                "bbox_model": "sample",
                            }
                        },
                        "display": {
                            "monitor": {"name": "m", "resolution": [800, 600], "width": 35, "distance": 60},
                            "gui": {
                                "button": {"size": [250, 100], "margin": 20, "color": "white", "text": {"color": "black", "size": 30}},
                                "fixation_dot": {"size": 10, "color": "white"},
                                "aoe": [500, 500],
                            },
                        },
                        "output": {"folder": "output"},
                        "instructions": {"intro": ["hi"], "outro": ["bye"]},
                    }
                ),
                encoding="utf-8",
            )

            dataset = TimeSeriesDataset(CustomConfig(str(cfg_path)), window_size=2)
            sample = dataset.data[0]
            with patch("tobii_pytracker.datasets.custom_dataset.visual.ShapeStim", FakeShapeStim):
                result = dataset.draw_stimulus(StubWindow(), sample)

            self.assertIn("timeseries_bboxes", result)
            self.assertEqual(len(result["timeseries_bboxes"]), 2)


class DummyModel(CustomModel):
    def prepare_model(self):
        self.model = "ready"

    def predict(self, input_data):
        return [input_data]

    def process(self, path):
        return [("class_a", 1.0, (0, 0, 10, 10))]


class FailingModel(CustomModel):
    def prepare_model(self):
        raise ValueError("boom")

    def predict(self, input_data):
        return []

    def process(self, path):
        return []


class TestCustomModel(unittest.TestCase):
    def test_custom_model_initializes_and_calls_prepare(self):
        dataset = type("Dataset", (), {"get_classes": lambda self: ["a", "b"], "is_text": False})()
        model = DummyModel({}, dataset)
        self.assertEqual(model.model, "ready")
        self.assertEqual(model.dataset_class_names, ["a", "b"])
        self.assertFalse(model.is_text)

    def test_custom_model_wraps_prepare_errors(self):
        dataset = type("Dataset", (), {"get_classes": lambda self: ["a"], "is_text": True})()
        with self.assertRaises(RuntimeError):
            FailingModel({}, dataset)


class TestCustomLogger(unittest.TestCase):
    def test_unknown_level_defaults_to_info(self):
        logger = CustomLogger("not-a-level", "test.logger").logger
        self.assertEqual(logger.name, "test.logger")
