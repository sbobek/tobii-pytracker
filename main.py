import os
import csv
import argparse
import importlib
from datetime import datetime
from psychopy import event, core

from utils import gui, eyetracker
from utils.custom_logger import CustomLogger
from configs.custom_config import CustomConfig
from datasets.custom_dataset import CustomDataset


LOGGER = None


def main(config, loop_count, eyetracker_config_file, enable_eyetracker, enable_model):
    dataset = CustomDataset(config)

    monitor = gui.prepare_monitor(config)

    window = gui.prepare_window(config, monitor)

    buttons = gui.prepare_buttons(config, window, dataset)

    if enable_eyetracker:
        io, tracker = eyetracker.launch_hub_server(eyetracker_config_file, window)

    if enable_model:
        try:
            custom_model_folder = config.get_model_config()["folder"]
            custom_model_module = config.get_model_config()["module"]
            custom_model_class  = config.get_model_config()["class"]

            model = getattr(importlib.import_module(f"{custom_model_folder}.{custom_model_module}"), custom_model_class)(config, dataset)
        except ImportError as e:
            LOGGER.error(f"Error importing module {custom_model_module}: {e}")
        except AttributeError as e:
            LOGGER.error(f"Error accessing attribute {custom_model_class} in module {custom_model_module}: {e}")

    output_config = config.get_output_config()
    output_folder = os.path.join(output_config["folder"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, "data.csv")

    if loop_count > len(dataset.data):
        loop_count = len(dataset.data)

    try:
        with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ['screenshot_file', 'input_data', 'classification', 'user_classification', 'gaze_data', 'model_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter=';')
            writer.writeheader()

            last_click_time = core.getTime()
            focus_time = 2.0
            debounce_time = focus_time + 2.0

            for i, sample in enumerate(dataset.data):
                if i == loop_count:
                    break

                data = sample['data']
                classification = sample['class'].lower()

                screenshot_path = gui.draw_window(config, window, data, dataset.is_text, buttons, focus_time, output_folder)

                gaze_data = []
                next_data = False

                # Wait for button click
                while not next_data:
                    current_time = core.getTime()
                    gaze_timestamp = round(current_time - last_click_time - focus_time, 4)

                    mouse = event.Mouse(win=window)
                    if enable_eyetracker:
                        gaze_position = eyetracker.get_gaze_position(config, tracker)
                        avg_pupil_size = eyetracker.get_avg_pupil_size(tracker)
                        gaze_data.append((gaze_position, avg_pupil_size, gaze_timestamp))

                    for i, (rect, text, label) in enumerate(buttons):
                        if mouse.isPressedIn(rect) and gaze_timestamp >= debounce_time:
                            model_prediction = []
                            if enable_model:
                                if dataset.is_text:
                                    model_prediciton = model.process(data)
                                else:
                                    model_prediction = model.process(screenshot_path)

                            button_config = config.get_button_config()
                            user_classification = text.text

                            row_data = {
                                'screenshot_file': screenshot_path,
                                'input_data': data,
                                'classification': classification,
                                'user_classification': user_classification.lower(),
                                'gaze_data': gaze_data,
                                'model_prediction': model_prediction
                            }

                            writer.writerow(row_data)

                            if label == "functional_quit":
                                LOGGER.info("Exit button pressed. Quitting...")
                                window.close()
                                if enable_eyetracker:
                                    tracker.setRecordingState(False)
                                    io.quit()
                                exit()

                            last_click_time = current_time
                            next_data = True

                    core.wait(0.01)

        LOGGER.info(f"Successfully saved data to {output_file}")
    except Exception as e:
        LOGGER.error(f"Failed to save the data: {e}")

    window.close()
    if enable_eyetracker:
        tracker.setRecordingState(False)
        io.quit()
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config/config.yaml', help='Path to YAML script config file')
    parser.add_argument('--eyetracker_config_file', type=str, default='configs/config/eyetracker_config.yaml', help='Path to YAML eyetracker config file')
    parser.add_argument('--enable_eyetracker', type=bool, default=False, help='Launch script with launchHubServer (needs connected eyetracker if set to True)')
    parser.add_argument('--enable_model', type=bool, default=False, help='Extend processing for custom model predictions (only for images)')
    parser.add_argument('--loop_count', type=int, default=10, help='Number of times that different data will be displayed before the script exits')
    parser.add_argument('--log_level', type=str, default='info', help='Main logger level ("info", "debug", "warning", "error", "critical")')
    args = parser.parse_args()

    LOGGER = CustomLogger(args.log_level, __name__).logger

    config = CustomConfig(args.config_file)

    main(config=config, loop_count=args.loop_count, eyetracker_config_file=args.eyetracker_config_file, enable_eyetracker=args.enable_eyetracker, enable_model=args.enable_model)
