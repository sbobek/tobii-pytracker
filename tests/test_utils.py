from pathlib import Path
import unittest


def get_test_resources_dir() -> Path:
    return Path(__file__).parent / "resources"


def get_test_config_path() -> Path:
    return get_test_resources_dir() / "test_config.yaml"


class TestTestUtilities(unittest.TestCase):
    def test_resources_dir_exists(self):
        self.assertTrue(get_test_resources_dir().exists())

    def test_config_path_points_to_yaml(self):
        config_path = get_test_config_path()
        self.assertEqual(config_path.name, "test_config.yaml")
        self.assertTrue(config_path.exists())
