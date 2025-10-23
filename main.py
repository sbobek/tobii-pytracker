import os
import csv
import argparse
import importlib
import threading
from datetime import datetime
from psychopy import event, core

from utils import gui, eyetracker
from utils.voice import VoiceRecorder
from utils.custom_logger import CustomLogger
from configs.custom_config import CustomConfig
from datasets.custom_dataset import CustomDataset


LOGGER = None

intro_text = [
    "Welcome to the study!",
    "In this experiment, you will see a series of images or text samples.",
    "Please look at each stimulus carefully, then select the appropriate option using the buttons below.",
    "Your gaze and voice may be recorded for analysis.",
    "Press SPACE to begin."
]

outro_text = [
    "Thank you for participating in this study!",
    "",
    "Your responses and recordings have been saved.",
    "You may now close the window or press ESC to exit.",
]


def main(config, loop_count, eyetracker_config_file, enable_eyetracker, enable_model, enable_voice):
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
            model = getattr(
                importlib.import_module(f"{custom_model_folder}.{custom_model_module}"),
                custom_model_class
            )(config, dataset)
        except (ImportError, AttributeError) as e:
            LOGGER.error(f"Error loading custom model: {e}")

    output_config = config.get_output_config()
    output_folder = os.path.join(output_config["folder"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "data.csv")

    if loop_count > len(dataset.data):
        loop_count = len(dataset.data)

    gui.show_instructions(window, intro_text)

    try:
        with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'screenshot_file', 'input_data', 'classification', 'user_classification',
                'gaze_data', 'model_prediction', 'voice_file', 'voice_start_timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter=';')
            writer.writeheader()

            last_click_time = core.getTime()
            focus_time = 2.0
            debounce_time = 1.0 #focus_time + 2.0

            for i, sample in enumerate(dataset.data):
                if i == loop_count:
                    break

                data = sample['data']
                classification = sample['class'].lower()
                screenshot_path = gui.draw_window(config, window, data, dataset.is_text, buttons, focus_time, output_folder)

                gaze_data = []
                next_data = False

                # --- Optional Voice Recording ---
                voice_thread = None
                voice_stop_event = None
                voice_filename = None
                voice_start_time = None

                if enable_voice:
                    voice_filename = os.path.join(output_folder, f"voice_{i:03d}.wav")
                    voice_stop_event = threading.Event()
                    voice_start_time = core.getTime()
                    voice_thread = threading.Thread(
                        target=VoiceRecorder.record_voice,
                        args=(voice_filename,),
                        kwargs={'stop_event': voice_stop_event}
                    )
                    voice_thread.start()

                try:
                    while not next_data:
                        current_time = core.getTime()
                        gaze_timestamp = round(current_time - last_click_time - focus_time, 4)

                        mouse = event.Mouse(win=window)
                        if enable_eyetracker:
                            gaze_position = eyetracker.get_gaze_position(config, tracker)
                            avg_pupil_size = eyetracker.get_avg_pupil_size(tracker)
                            gaze_data.append((gaze_position, avg_pupil_size, gaze_timestamp))

                        for j, (rect, text, label) in enumerate(buttons):
                            if mouse.isPressedIn(rect) and (current_time - last_click_time) > debounce_time:
                                model_prediction = []
                                if enable_model:
                                    model_prediction = model.process(screenshot_path) if not dataset.is_text else model.process(data)

                                user_classification = text.text

                                # --- Stop voice recording ---
                                if enable_voice and voice_thread is not None:
                                    voice_stop_event.set()
                                    voice_thread.join()

                                row_data = {
                                    'screenshot_file': screenshot_path,
                                    'input_data': data,
                                    'classification': classification,
                                    'user_classification': user_classification.lower(),
                                    'gaze_data': gaze_data,
                                    'model_prediction': model_prediction,
                                    'voice_file': voice_filename,
                                    'voice_start_timestamp': round(voice_start_time, 4) if voice_start_time else None
                                }
                                writer.writerow(row_data)

                                if label == "functional_quit":
                                    LOGGER.info("Exit button pressed. Quitting...")
                                    return  # clean exit

                                last_click_time = current_time
                                next_data = True

                        core.wait(0.01)

                except KeyboardInterrupt:
                    LOGGER.info("Keyboard interrupt received. Exiting...")
                    if enable_voice and voice_thread is not None:
                        voice_stop_event.set()
                        voice_thread.join()
                    raise  # re-raise to outer try/finally

    except KeyboardInterrupt:
        LOGGER.info("Experiment interrupted by user.")

    except Exception as e:
        LOGGER.error(f"Failed to save the data: {e}")

    finally:
        # --- Clean up resources ---
        gui.show_instructions(window, outro_text, key='escape')
        if enable_eyetracker:
            tracker.setRecordingState(False)
            io.quit()
        window.close()
        # Ensure voice thread is stopped
        if enable_voice and voice_thread is not None and voice_thread.is_alive():
            voice_stop_event.set()
            voice_thread.join()

    LOGGER.info(f"Data saved to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config/config.yaml', help='Path to YAML script config file')
    parser.add_argument('--eyetracker_config_file', type=str, default='configs/config/eyetracker_config.yaml', help='Path to YAML eyetracker config file')
    parser.add_argument('--enable_eyetracker', type=bool, default=False, help='Launch script with launchHubServer (needs connected eyetracker if set to True)')
    parser.add_argument('--enable_model', type=bool, default=False, help='Extend processing for custom model predictions (only for images)')
    parser.add_argument('--enable_voice', type=bool, default=False, help='Enable voice recording and processing')
    parser.add_argument('--loop_count', type=int, default=10, help='Number of times that different data will be displayed before the script exits')
    parser.add_argument('--log_level', type=str, default='info', help='Main logger level ("info", "debug", "warning", "error", "critical")')
    args = parser.parse_args()

    LOGGER = CustomLogger(args.log_level, __name__).logger

    config = CustomConfig(args.config_file)

    main(config=config, loop_count=args.loop_count, eyetracker_config_file=args.eyetracker_config_file, 
    enable_eyetracker=args.enable_eyetracker, enable_model=args.enable_model, enable_voice=args.enable_voice)
