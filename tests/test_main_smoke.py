from __future__ import annotations

import unittest
from argparse import Namespace
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tests.support import bootstrap_test_environment

bootstrap_test_environment()

if "psychopy.iohub" not in sys.modules:
    iohub = types.ModuleType("psychopy.iohub")
    iohub.launchHubServer = Mock(name="launchHubServer")
    sys.modules["psychopy.iohub"] = iohub

if "sounddevice" not in sys.modules:
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.CallbackAbort = RuntimeError
    sounddevice.InputStream = Mock(name="InputStream")
    sounddevice.sleep = Mock(name="sleep")
    sys.modules["sounddevice"] = sounddevice

if "soundfile" not in sys.modules:
    soundfile = types.ModuleType("soundfile")
    soundfile.write = Mock(name="write")
    sys.modules["soundfile"] = soundfile

from tobii_pytracker import main as main_module


class TestMainSmoke(unittest.TestCase):
    def test_cli_smoke_invokes_main(self):
        args = Namespace(
            config_file="configs/config.yaml",
            eyetracker_config_file="configs/eyetracker_config.yaml",
            enable_eyetracker=False,
            enable_voice=False,
            raw_data=False,
            disable_psychopy=True,
            loop_count=1,
            log_level="info",
        )

        config_obj = Mock(name="config")

        with patch("tobii_pytracker.main.argparse.ArgumentParser.parse_args", return_value=args):
            with patch("tobii_pytracker.main.CustomConfig", return_value=config_obj):
                with patch("tobii_pytracker.main.main") as main_mock:
                    main_module.cli()

        main_mock.assert_called_once_with(
            config=config_obj,
            loop_count=1,
            eyetracker_config_file="configs/eyetracker_config.yaml",
            enable_eyetracker=False,
            enable_voice=False,
            raw_data=False,
            enable_psychopy=False,
        )

    def test_main_headless_without_eyetracker_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = Mock(name="config")
            config.dataset_type = "image"
            config.get_output_config.return_value = {"folder": str(output_dir)}

            dataset = Mock(name="dataset")
            dataset.data = []

            logger = Mock(name="logger")
            main_module.LOGGER = logger

            with patch("tobii_pytracker.main.ImageDataset", return_value=dataset):
                with patch("tobii_pytracker.main.CustomConfig.read_config", return_value={}):
                    with patch("tobii_pytracker.main.eyetracker.is_mouse_eyetracker", return_value=False):
                        with patch("tobii_pytracker.main.eyetracker.get_mouse_move_button_idx", return_value=0):
                            main_module.main(
                                config=config,
                                loop_count=3,
                                eyetracker_config_file="configs/eyetracker_config.yaml",
                                enable_eyetracker=False,
                                enable_voice=False,
                                raw_data=False,
                                enable_psychopy=False,
                            )

            data_files = list(output_dir.glob("*/data.csv"))
            self.assertEqual(len(data_files), 1)
            self.assertTrue(data_files[0].read_text(encoding="utf-8").startswith("screenshot_file;input_data;"))
            logger.info.assert_called()

    def test_main_supports_text_and_timeseries_dataset_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            logger = Mock(name="logger")
            main_module.LOGGER = logger

            for dataset_type, ctor_path in (
                ("text", "tobii_pytracker.main.TextDataset"),
                ("time_series", "tobii_pytracker.main.TimeSeriesDataset"),
            ):
                config = Mock(name=f"config_{dataset_type}")
                config.dataset_type = dataset_type
                config.get_output_config.return_value = {"folder": str(output_dir)}
                dataset = Mock(name=f"{dataset_type}_dataset")
                dataset.data = []

                with patch(ctor_path, return_value=dataset):
                    with patch("tobii_pytracker.main.CustomConfig.read_config", return_value={}):
                        with patch("tobii_pytracker.main.eyetracker.is_mouse_eyetracker", return_value=False):
                            with patch("tobii_pytracker.main.eyetracker.get_mouse_move_button_idx", return_value=0):
                                main_module.main(
                                    config=config,
                                    loop_count=1,
                                    eyetracker_config_file="configs/eyetracker_config.yaml",
                                    enable_eyetracker=False,
                                    enable_voice=False,
                                    raw_data=False,
                                    enable_psychopy=False,
                                )

    def test_main_psychopy_voice_and_mouse_tracker_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = Mock(name="config")
            config.dataset_type = "image"
            config.get_output_config.return_value = {"folder": str(output_dir)}
            config.get_instructions_config.return_value = {"intro": ["hello"], "outro": ["bye"]}

            dataset = Mock(name="dataset")
            dataset.data = [{"class": "good", "data": "dummy.png"}]

            window = Mock(name="window")
            window.winHandle = Mock(name="winHandle")
            mouse = Mock(name="mouse")
            mouse.getPressed.return_value = [False, True, False]
            mouse.isPressedIn.return_value = True

            button_text = Mock(name="button_text")
            button_text.text = "Good"
            buttons = [(Mock(name="rect"), button_text, "functional_next")]

            io = Mock(name="io")
            tracker = Mock(name="tracker")
            logger = Mock(name="logger")
            main_module.LOGGER = logger

            thread = Mock(name="voice_thread")
            thread.is_alive.return_value = True
            stop_event = Mock(name="stop_event")

            with patch("tobii_pytracker.main.ImageDataset", return_value=dataset):
                with patch("tobii_pytracker.main.gui.prepare_monitor", return_value=Mock(name="monitor")):
                    with patch("tobii_pytracker.main.gui.prepare_window", return_value=window):
                        with patch("tobii_pytracker.main.gui.prepare_buttons", return_value=buttons):
                            with patch("tobii_pytracker.main.gui.show_instructions"):
                                with patch(
                                    "tobii_pytracker.main.gui.draw_window",
                                    return_value=("shot.png", {"obj": "bbox"}),
                                ):
                                    with patch("tobii_pytracker.main.CustomConfig.read_config", return_value={}):
                                        with patch(
                                            "tobii_pytracker.main.eyetracker.launch_hub_server",
                                            return_value=(io, tracker),
                                        ):
                                            with patch(
                                                "tobii_pytracker.main.eyetracker.is_mouse_eyetracker",
                                                return_value=True,
                                            ):
                                                with patch(
                                                    "tobii_pytracker.main.eyetracker.get_mouse_move_button_idx",
                                                    return_value=1,
                                                ):
                                                    with patch(
                                                        "tobii_pytracker.main.eyetracker.poll_tracker_events",
                                                        return_value=({"EyeSampleEvent": [object()]}, 1),
                                                    ):
                                                        with patch(
                                                            "tobii_pytracker.main.eyetracker.extract_full_raw_event",
                                                            return_value=[{"sample": 1}],
                                                        ):
                                                            with patch(
                                                                "tobii_pytracker.main.threading.Thread",
                                                                return_value=thread,
                                                            ):
                                                                with patch(
                                                                    "tobii_pytracker.main.threading.Event",
                                                                    return_value=stop_event,
                                                                ):
                                                                    with patch(
                                                                        "tobii_pytracker.main.event.Mouse",
                                                                        return_value=mouse,
                                                                        create=True,
                                                                    ):
                                                                        with patch(
                                                                            "tobii_pytracker.main.core.getTime",
                                                                            side_effect=[0.0, 1.0, 2.0],
                                                                            create=True,
                                                                        ):
                                                                            with patch(
                                                                                "tobii_pytracker.main.core.wait",
                                                                                return_value=None,
                                                                            ):
                                                                                main_module.main(
                                                                                    config=config,
                                                                                    loop_count=1,
                                                                                    eyetracker_config_file="configs/eyetracker_config.yaml",
                                                                                    enable_eyetracker=True,
                                                                                    enable_voice=True,
                                                                                    raw_data=True,
                                                                                    enable_psychopy=True,
                                                                                )

            self.assertGreaterEqual(stop_event.set.call_count, 1)
            self.assertGreaterEqual(thread.join.call_count, 1)

    def test_main_headless_with_eyetracker_creates_raw_stream_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = Mock(name="config")
            config.dataset_type = "image"
            config.get_output_config.return_value = {"folder": str(output_dir)}

            dataset = Mock(name="dataset")
            dataset.data = [{"class": "good", "data": "dummy.png"}]

            io = Mock(name="io")
            tracker = Mock(name="tracker")
            logger = Mock(name="logger")
            main_module.LOGGER = logger

            with patch("tobii_pytracker.main.ImageDataset", return_value=dataset):
                with patch("tobii_pytracker.main.CustomConfig.read_config", return_value={}):
                    with patch("tobii_pytracker.main.eyetracker.launch_hub_server", return_value=(io, tracker)):
                        with patch("tobii_pytracker.main.eyetracker.is_mouse_eyetracker", return_value=False):
                            with patch("tobii_pytracker.main.eyetracker.get_mouse_move_button_idx", return_value=0):
                                with patch(
                                    "tobii_pytracker.main.eyetracker.poll_tracker_events",
                                    side_effect=[({}, 1), KeyboardInterrupt()],
                                ):
                                    with patch(
                                        "tobii_pytracker.main.eyetracker.extract_full_raw_event",
                                        return_value=[{"sample": 1}],
                                    ):
                                        with patch("tobii_pytracker.main.core.getTime", return_value=0.0, create=True):
                                            with patch("tobii_pytracker.main.core.wait", return_value=None):
                                                main_module.main(
                                                    config=config,
                                                    loop_count=1,
                                                    eyetracker_config_file="configs/eyetracker_config.yaml",
                                                    enable_eyetracker=True,
                                                    enable_voice=False,
                                                    raw_data=True,
                                                    enable_psychopy=False,
                                                )

            raw_files = list(output_dir.glob("*/raw_stream.csv"))
            self.assertEqual(len(raw_files), 1)
            tracker.setRecordingState.assert_called_with(False)
            self.assertGreaterEqual(io.quit.call_count, 1)

    def test_main_psychopy_quit_button_triggers_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = Mock(name="config")
            config.dataset_type = "image"
            config.get_output_config.return_value = {"folder": str(output_dir)}
            config.get_instructions_config.return_value = {"intro": ["hello"], "outro": ["bye"]}

            dataset = Mock(name="dataset")
            dataset.data = [{"class": "good", "data": "dummy.png"}]

            window = Mock(name="window")
            window.winHandle = Mock(name="winHandle")
            mouse = Mock(name="mouse")
            mouse.isPressedIn.return_value = True

            button_text = Mock(name="button_text")
            button_text.text = "Good"
            buttons = [(Mock(name="rect"), button_text, "functional_quit")]

            io = Mock(name="io")
            tracker = Mock(name="tracker")
            logger = Mock(name="logger")
            main_module.LOGGER = logger

            with patch("tobii_pytracker.main.ImageDataset", return_value=dataset):
                with patch("tobii_pytracker.main.gui.prepare_monitor", return_value=Mock(name="monitor")):
                    with patch("tobii_pytracker.main.gui.prepare_window", return_value=window):
                        with patch("tobii_pytracker.main.gui.prepare_buttons", return_value=buttons):
                            with patch("tobii_pytracker.main.gui.show_instructions"):
                                with patch(
                                    "tobii_pytracker.main.gui.draw_window",
                                    return_value=("shot.png", {"obj": "bbox"}),
                                ):
                                    with patch("tobii_pytracker.main.CustomConfig.read_config", return_value={}):
                                        with patch(
                                            "tobii_pytracker.main.eyetracker.launch_hub_server",
                                            return_value=(io, tracker),
                                        ):
                                            with patch(
                                                "tobii_pytracker.main.eyetracker.is_mouse_eyetracker",
                                                return_value=False,
                                            ):
                                                with patch(
                                                    "tobii_pytracker.main.eyetracker.get_mouse_move_button_idx",
                                                    return_value=0,
                                                ):
                                                    with patch(
                                                        "tobii_pytracker.main.eyetracker.poll_tracker_events",
                                                        return_value=({}, 0),
                                                    ):
                                                        with patch(
                                                            "tobii_pytracker.main.eyetracker.extract_eye_gaze_events",
                                                            return_value=[],
                                                        ):
                                                            with patch(
                                                                "tobii_pytracker.main.event.Mouse",
                                                                return_value=mouse,
                                                                create=True,
                                                            ):
                                                                with patch(
                                                                    "tobii_pytracker.main.core.getTime",
                                                                    side_effect=[0.0, 1.0],
                                                                    create=True,
                                                                ):
                                                                    with patch(
                                                                        "tobii_pytracker.main.core.wait",
                                                                        return_value=None,
                                                                    ):
                                                                        main_module.main(
                                                                            config=config,
                                                                            loop_count=1,
                                                                            eyetracker_config_file="configs/eyetracker_config.yaml",
                                                                            enable_eyetracker=True,
                                                                            enable_voice=False,
                                                                            raw_data=False,
                                                                            enable_psychopy=True,
                                                                        )

            tracker.clearEvents.assert_called_once()
            tracker.setRecordingState.assert_called_with(False)
            io.quit.assert_called_once()
            window.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
