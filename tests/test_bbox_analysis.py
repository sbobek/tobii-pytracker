import unittest
from pathlib import Path

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

from tobii_pytracker.configs.custom_config import CustomConfig
from tobii_pytracker.datasets.custom_dataset import ImageDataset

class TestBBoxGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.resources_dir = (
            Path(__file__).parent / "resources"
        )

        cls.test_image_path = (
            cls.resources_dir / "test_image.png"
        )

        cls.config_path = (
            cls.resources_dir / "test_config.yaml"
        )

        if not cls.test_image_path.exists():
            raise FileNotFoundError(
                f"Test image not found: {cls.test_image_path}"
            )

        if not cls.config_path.exists():
            raise FileNotFoundError(
                f"Test config not found: {cls.config_path}"
            )

    def test_grid_bbox_generation(self):

        config = CustomConfig(
            str(self.config_path)
        )

        dataset = ImageDataset(
            config=config,
            calculate_bboxes=False,
        )

        bboxes = dataset._detect_grid(
            str(self.test_image_path),
            grid_x=3,
            grid_y=3,
        )

        self.assertEqual(
            len(bboxes),
            9,
        )

        for record in bboxes:
            self.assertEqual(
                record["class"],
                "grid",
            )

            self.assertEqual(
                record["conf"],
                1.0,
            )

            self.assertIn(
                "bbox",
                record,
            )

            bbox = record["bbox"]

            self.assertIn(
                "cx",
                bbox,
            )

            self.assertIn(
                "cy",
                bbox,
            )

            self.assertIn(
                "w",
                bbox,
            )

            self.assertIn(
                "h",
                bbox,
            )

            self.assertAlmostEqual(
                bbox["w"],
                250.0,
            )

            self.assertAlmostEqual(
                bbox["h"],
                250.0,
            )
            # AOI:
            #       -375          0          +375
            cx = bbox["cx"]
            cy = bbox["cy"]

            self.assertGreaterEqual(
                cx,
                -375,
            )

            self.assertLessEqual(
                cx,
                375,
            )

            self.assertGreaterEqual(
                cy,
                -375,
            )

            self.assertLessEqual(
                cy,
                375,
            )

            x_min = cx - bbox["w"] / 2
            x_max = cx + bbox["w"] / 2

            y_min = cy - bbox["h"] / 2
            y_max = cy + bbox["h"] / 2

            self.assertGreaterEqual(
                x_min,
                -375,
            )

            self.assertLessEqual(
                x_max,
                375,
            )

            self.assertGreaterEqual(
                y_min,
                -375,
            )

            self.assertLessEqual(
                y_max,
                375,
            )

        centers = {
            (
                round(record["bbox"]["cx"]),
                round(record["bbox"]["cy"]),
            )
            for record in bboxes
        }

        expected_centers = {
            (-250, 250),
            (-250, 0),
            (-250, -250),
            (0, 250),
            (0, 0),
            (0, -250),
            (250, 250),
            (250, 0),
            (250, -250),
        }

        self.assertEqual(
            centers,
            expected_centers,
        )

    def test_superpixel_bbox_generation(self):
        config = CustomConfig(
            str(self.config_path)
        )

        dataset = ImageDataset(
            config=config,
            calculate_bboxes=False,
        )

        bboxes = dataset._detect_superpixels(
            str(self.test_image_path),
            n_segments=20,
        )

        self.assertGreater(
            len(bboxes),
            0,
        )

        self.assertLessEqual(
            len(bboxes),
            20,
        )

        for record in bboxes:
            self.assertEqual(
                record["class"],
                "superpixel",
            )

            self.assertIn(
                "conf",
                record,
            )

            self.assertIn(
                "polygon",
                record,
            )

            bbox = record["polygon"]

            self.assertIsInstance(
                bbox,
                list,
            )

            self.assertGreaterEqual(
                len(bbox),
                3,
            )

            for point in bbox:
                self.assertEqual(
                    len(point),
                    2,
                )

            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            self.assertGreaterEqual(
                x_min,
                -375,
            )

            self.assertLessEqual(
                x_max,
                375,
            )

            self.assertGreaterEqual(
                y_min,
                -375,
            )

            self.assertLessEqual(
                y_max,
                375,
            )

            self.assertIn(
                "polygon",
                record,
            )


    def test_bbox_coordinate_system(self):

        config = CustomConfig(
            str(self.config_path)
        )

        dataset = ImageDataset(
            config=config,
            calculate_bboxes=False,
        )

        bboxes = dataset._detect_grid(
            str(self.test_image_path),
            grid_x=3,
            grid_y=3,
        )

        top_left = next(
            record
            for record in bboxes
            if (
                record["bbox"]["cx"] == -250
                and
                record["bbox"]["cy"] == 250
            )
        )

        self.assertEqual(
            top_left["bbox"]["w"],
            250,
        )

        self.assertEqual(
            top_left["bbox"]["h"],
            250,
        )

        center = next(
            record
            for record in bboxes
            if (
                record["bbox"]["cx"] == 0
                and
                record["bbox"]["cy"] == 0
            )
        )

        self.assertEqual(
            center["bbox"]["w"],
            250,
        )

        self.assertEqual(
            center["bbox"]["h"],
            250,
        )

        bottom_right = next(
            record
            for record in bboxes
            if (
                record["bbox"]["cx"] == 250
                and
                record["bbox"]["cy"] == -250
            )
        )

        self.assertEqual(
            bottom_right["bbox"]["w"],
            250,
        )

        self.assertEqual(
            bottom_right["bbox"]["h"],
            250,
        )

    def test_grid_bbox_gaze_mapping(self):

        config = CustomConfig(
            str(self.config_path)
        )

        dataset = ImageDataset(
            config=config,
            calculate_bboxes=False,
        )

        bboxes = dataset._detect_grid(
            str(self.test_image_path),
            grid_x=3,
            grid_y=3,
        )

        gaze_points = [
            (-250, 250),
            (0, 0),
            (250, -250),
        ]

        matched_bboxes = []

        for gaze_x, gaze_y in gaze_points:
            matching = []

            for record in bboxes:
                bbox = record["bbox"]

                x_min = (
                    bbox["cx"]
                    - bbox["w"] / 2
                )

                x_max = (
                    bbox["cx"]
                    + bbox["w"] / 2
                )

                y_min = (
                    bbox["cy"]
                    - bbox["h"] / 2
                )

                y_max = (
                    bbox["cy"]
                    + bbox["h"] / 2
                )

                if (
                    x_min <= gaze_x <= x_max
                    and
                    y_min <= gaze_y <= y_max
                ):
                    matching.append(record)

            self.assertEqual(
                len(matching),
                1,
                (
                    f"Gaze point "
                    f"({gaze_x}, {gaze_y}) "
                    f"should map to exactly one bbox"
                ),
            )

            matched_bboxes.append(
                matching[0]
            )

        self.assertEqual(
            matched_bboxes[0]["bbox"]["cx"],
            -250,
        )

        self.assertEqual(
            matched_bboxes[0]["bbox"]["cy"],
            250,
        )

        self.assertEqual(
            matched_bboxes[1]["bbox"]["cx"],
            0,
        )

        self.assertEqual(
            matched_bboxes[1]["bbox"]["cy"],
            0,
        )

        self.assertEqual(
            matched_bboxes[2]["bbox"]["cx"],
            250,
        )

        self.assertEqual(
            matched_bboxes[2]["bbox"]["cy"],
            -250,
        )