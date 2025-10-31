import os
import csv
import argparse
import importlib
import threading
from datetime import datetime
from psychopy import event, core

from .utils import gui, eyetracker
from .utils.voice import VoiceRecorder
from .utils.custom_logger import CustomLogger
from .configs.custom_config import CustomConfig
from .datasets.custom_dataset import CustomDataset




LOGGER = None


def main(config, loop_count, eyetracker_config_file,
         enable_eyetracker, enable_model, enable_voice,
         raw_data, enable_psychopy):
    """
    Main execution function for the experiment or data recording session.

    This function handles both PsychoPy GUI-based trials and headless eye-tracker
    recording. Depending on the options, it can:
      - Display stimuli and buttons via PsychoPy
      - Record gaze data and/or full raw eye-tracker samples
      - Record voice input
      - Apply a custom model to predict outputs based on displayed stimuli

    The function ensures proper resource management and logging of all actions
    and gracefully handles keyboard interrupts and unexpected exceptions.

    Parameters
    ----------
    config : CustomConfig
        Configuration object loaded from a YAML file, containing display, model,
        output, and instruction settings.

    loop_count : int
        Maximum number of trials or data items to process. If higher than the
        dataset length, the dataset length is used instead.

    eyetracker_config_file : str
        Path to the YAML configuration file for the Tobii eye-tracker.

    enable_eyetracker : bool
        If True, launches the eye-tracker using `launchHubServer()` and records
        gaze or raw samples.

    enable_model : bool
        If True, initializes a custom model (image/text prediction) and applies
        it to each trial or data item.

    enable_voice : bool
        If True, records voice input during each trial or in headless mode.

    raw_data : bool
        If True, records full Tobii raw samples instead of filtered gaze
        positions and pupil sizes.

    enable_psychopy : bool
        Controls whether the PsychoPy GUI is used:
          - True: Displays GUI windows, stimuli, and buttons, recording gaze
            and model outputs per trial.
          - False: Runs in headless mode, continuously recording raw gaze
            samples (if enabled) and/or voice until interrupted by the user.
            No screenshots, classification, or buttons are used in this mode.

    Returns
    -------
    None
        This function writes experiment or recording data to CSV files in the
        output folder. It does not return any Python objects.

    Files Created
    -------------
    - `data.csv` (PsychoPy mode):
        Contains trial-specific data with the following columns:
            'screenshot_file', 'input_data', 'classification', 'user_classification',
            'gaze_data', 'model_prediction', 'voice_file', 'voice_start_timestamp'
    - `raw_stream.csv` (Headless mode):
        Contains timestamped raw eye-tracker samples when `enable_psychopy=False`
        and `enable_eyetracker=True`.

    Exception Handling
    ------------------
    - KeyboardInterrupt: Gracefully stops trials or headless recording, ensuring
      voice threads and eye-tracker recording are stopped and resources are released.
    - Exception: Logs any unexpected error during:
        - CSV writing
        - Eye-tracker sample retrieval
        - Voice recording
        - Model processing
      Errors do **not** stop the experiment unless critical.

    Notes
    -----
    - When running headless (`enable_psychopy=False`), no screenshots, data, or
      classification columns are written—only raw eye-tracker samples and voice.
    - Voice recording is threaded to avoid blocking the trial loop.
    - Eye-tracker recording is started immediately after initialization and
      stopped at the end, ensuring all buffered samples are captured.

    Examples
    --------
    >>> config = CustomConfig("configs/config.yaml")
    >>> main(
    ...     config=config,
    ...     loop_count=10,
    ...     eyetracker_config_file="configs/eyetracker_config.yaml",
    ...     enable_eyetracker=True,
    ...     enable_model=False,
    ...     enable_voice=True,
    ...     raw_data=True,
    ...     enable_psychopy=False
    ... )
    Running headless will continuously record raw eye-tracker data and voice
    until the user presses Ctrl+C.
    """
    
    dataset = CustomDataset(config)

    monitor, window, buttons = None, None, None

    # --- Optional PsychoPy GUI setup ---
    if enable_psychopy:
        monitor = gui.prepare_monitor(config)
        window = gui.prepare_window(config, monitor)
        buttons = gui.prepare_buttons(config, window, dataset)

    # --- Initialize Eye Tracker ---
    io, tracker = (None, None)
    if enable_eyetracker:
        io, tracker = eyetracker.launch_hub_server(eyetracker_config_file, window if enable_psychopy else None)

    # --- Initialize custom model ---
    model = None
    if enable_model:
        try:
            cfg = config.get_model_config()
            model = getattr(
                importlib.import_module(f"{cfg['folder']}.{cfg['module']}"),
                cfg['class']
            )(config, dataset)
        except (ImportError, AttributeError) as e:
            LOGGER.error(f"Error loading custom model: {e}")

    # --- Output setup ---
    output_config = config.get_output_config()
    output_folder = os.path.join(output_config["folder"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "data.csv")

    if loop_count > len(dataset.data):
        loop_count = len(dataset.data)

    # --- Show instructions if PsychoPy enabled ---
    if enable_psychopy:
        intro_text = config.get_instructions_config()["intro"]
        gui.show_instructions(window, intro_text)

    # --- Main try/finally block to ensure cleanup ---
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
            debounce_time = 0.0

            # --- Headless mode (no PsychoPy GUI) ---
            if not enable_psychopy and enable_eyetracker:
                LOGGER.info("Running in headless mode (no GUI). Press Ctrl+C to stop.")
                try:
                    with open(os.path.join(output_folder, "raw_stream.csv"), "w", newline='', encoding="utf-8") as rawfile:
                        raw_writer = csv.writer(rawfile)
                        raw_writer.writerow(["timestamp", "sample_data"])
                        while True:
                            try:
                                sample = eyetracker.get_full_raw_sample(tracker)
                                if sample:
                                    raw_writer.writerow([datetime.now().isoformat(), sample])
                                core.wait(0.01)
                            except KeyboardInterrupt:
                                LOGGER.info("Recording stopped by user.")
                                break
                            except Exception as e:
                                LOGGER.error(f"Error during headless recording: {e}")
                finally:
                    if tracker:
                        tracker.setRecordingState(False)
                        io.quit()
                return

            # --- PsychoPy GUI loop ---
            for i, sample in enumerate(dataset.data):
                if i == loop_count:
                    break

                data = sample['data']
                classification = sample['class'].lower()
                screenshot_path = gui.draw_window(config, window, data, dataset.is_text, buttons, focus_time, output_folder)

                gaze_data = []
                next_data = False

                voice_thread, voice_stop_event, voice_filename, voice_start_time = None, None, None, None

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
                        gaze_timestamp = current_time #round(current_time - last_click_time - focus_time, 4)
                        mouse = event.Mouse(win=window)

                        if enable_eyetracker:
                            try:
                                if raw_data:
                                    full_sample = eyetracker.get_full_raw_sample(tracker)
                                    if full_sample:
                                        gaze_data.append((gaze_timestamp, full_sample))
                                else:
                                    gaze_position = eyetracker.get_gaze_position(config, tracker)
                                    avg_pupil_size = eyetracker.get_avg_pupil_size(tracker)
                                    gaze_data.append((gaze_position, avg_pupil_size, gaze_timestamp))
                            except Exception as e:
                                import traceback
                                LOGGER.error(f"Eye tracker error: {e}")
                                traceback.print_exc()

                        # --- Check button presses ---
                        for rect, text, label in buttons:
                            if mouse.isPressedIn(rect) and (current_time - last_click_time) > debounce_time:
                                model_prediction = []
                                if enable_model:
                                    model_prediction = model.process(screenshot_path) if not dataset.is_text else model.process(data)

                                user_classification = text.text

                                if enable_voice and voice_thread:
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
                                    return

                                last_click_time = current_time
                                next_data = True

                        core.wait(0.01)

                except KeyboardInterrupt:
                    LOGGER.info("Keyboard interrupt received. Exiting current trial...")
                    if enable_voice and voice_thread:
                        voice_stop_event.set()
                        voice_thread.join()
                    raise
                except Exception as e:
                    LOGGER.error(f"Error during trial loop: {e}")

    except KeyboardInterrupt:
        LOGGER.info("Experiment interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Failed to save the data: {e}")
    finally:
        # --- Cleanup ---
        if enable_psychopy and window:
            outro_text = config.get_instructions_config()["outro"]
            gui.show_instructions(window, outro_text, key='escape')
            window.close()

        if enable_eyetracker and tracker:
            tracker.setRecordingState(False)
            io.quit()

        if enable_voice and voice_thread and voice_thread.is_alive():
            voice_stop_event.set()
            voice_thread.join()

    LOGGER.info(f"Data saved to {output_file}")


def cli():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config.yaml', help='Path to YAML script config file')
    parser.add_argument('--eyetracker_config_file', type=str, default='configs/eyetracker_config.yaml', help='Path to YAML eyetracker config file')

    # Boolean flags
    parser.add_argument('--enable_eyetracker', action='store_true', help='Launch script with launchHubServer (needs connected eyetracker if set)')
    parser.add_argument('--enable_model', action='store_true', help='Extend processing for custom model predictions (only for images)')
    parser.add_argument('--enable_voice', action='store_true', help='Enable voice recording and processing')
    parser.add_argument('--raw_data', action='store_true', help='Record full Tobii raw samples instead of filtered gaze position')
    parser.add_argument('--disable_psychopy', action='store_true', help='Run headless (no GUI) and continuously record gaze + voice until stopped')

    # Other arguments
    parser.add_argument('--loop_count', type=int, default=10, help='Number of times that different data will be displayed before the script exits')
    parser.add_argument('--log_level', type=str, default='info', help='Main logger level ("info", "debug", "warning", "error", "critical")')

    args = parser.parse_args()

    global LOGGER
    LOGGER = CustomLogger(args.log_level, __name__).logger
    config = CustomConfig(args.config_file)

    main(
        config=config,
        loop_count=args.loop_count,
        eyetracker_config_file=args.eyetracker_config_file,
        enable_eyetracker=args.enable_eyetracker,
        enable_model=args.enable_model,
        enable_voice=args.enable_voice,
        raw_data=args.raw_data,
        enable_psychopy=not args.disable_psychopy
    )


if __name__ == "__main__":
    cli()



    