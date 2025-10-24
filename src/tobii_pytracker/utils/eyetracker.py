import yaml
from psychopy.iohub import launchHubServer
from .custom_logger import CustomLogger
from tobii_pytracker.configs.custom_config import CustomConfig

LOGGER = CustomLogger("debug", "eyetracker").logger

def launch_hub_server(eyetracker_config_file, window):
    iohub_config = CustomConfig.read_config(eyetracker_config_file)

    io = launchHubServer(**iohub_config, window=window)
    tracker = io.devices.tracker
    r = tracker.runSetupProcedure()
    LOGGER.debug(r)
    tracker.setRecordingState(True)

    return io, tracker

def get_gaze_position(config, tracker):
    gaze_position = tracker.getPosition()

    area_x, area_y = config.get_area_of_interest_size()
    if gaze_position is not None:
        if ( gaze_position[0] >= -area_x/2 and gaze_position[0] <= area_x/2 ) and ( gaze_position[1] >= -area_y/2 and gaze_position[1] <= area_y/2 ):
            return (round(gaze_position[0]), round(gaze_position[1]))

    return (None, None)

def get_avg_pupil_size(tracker):
    sample = tracker.getLastSample()

    if sample is not None:
        avg_pupil_size = (sample[21] + sample[40]) / 2.0  # sample[21]=left_pupil_measure, sample[40]=right_pupil_measure
        return round(avg_pupil_size, 4)
    
    return None
