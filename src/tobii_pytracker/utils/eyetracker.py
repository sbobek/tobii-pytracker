import yaml
from psychopy.iohub import launchHubServer
from .custom_logger import CustomLogger
from tobii_pytracker.configs.custom_config import CustomConfig
from psychopy.iohub.devices.eyetracker.hw.mouse.eyetracker import EyeTracker as MouseTracker


LOGGER = CustomLogger("debug", "eyetracker").logger

def launch_hub_server(eyetracker_config_file, window):
    """
    Launch and initialize the PsychoPy ioHub server with a Tobii eye tracker.

    This function reads the custom eye-tracker configuration file, initializes
    the ioHub server, connects to the Tobii tracker, runs its calibration/setup
    procedure, and starts data recording.

    Parameters
    ----------
    eyetracker_config_file : str or Path
        Path to the YAML configuration file containing Tobii eye-tracker
        connection parameters (IP address, sampling rate, etc.).
    window : psychopy.visual.Window
        The PsychoPy window object to be associated with the eye tracker.

    Returns
    -------
    tuple
        A tuple ``(io, tracker)`` where:
        - ``io`` : the ioHub connection instance
        - ``tracker`` : the connected Tobii tracker device instance

    Notes
    -----
    - This function must be called before any gaze or pupil data can be recorded.
    - The tracker is set to recording mode automatically
      (`tracker.setRecordingState(True)`).
    - The function runs Tobii’s calibration/setup sequence once via
      `tracker.runSetupProcedure()`.

    Examples
    --------
    >>> io, tracker = launch_hub_server("tobii_config.yaml", window)
    >>> tracker.getPosition()
    (-0.1, 0.2)
    """
    iohub_config = CustomConfig.read_config(eyetracker_config_file)

    io = launchHubServer(**iohub_config, window=window)
    tracker = io.devices.tracker
    r = tracker.runSetupProcedure()
    LOGGER.debug(r)
    tracker.setRecordingState(True)

    return io, tracker

def get_gaze_position(config, tracker):
    """
    Retrieve the current gaze position if it lies within the defined
    area of interest (AOI).

    This function queries the Tobii tracker for the most recent gaze position
    and checks whether it falls within a rectangular AOI defined by the
    experimental configuration. If the gaze lies inside the AOI, the
    coordinates are rounded and returned; otherwise, ``(None, None)`` is
    returned.

    Parameters
    ----------
    config : CustomConfig
        Configuration object providing AOI dimensions via
        `config.get_area_of_interest_size()`.
    tracker : psychopy.iohub.devices.eyetracker.EyeTrackerDevice
        The active Tobii tracker object created by `launchHubServer()`.

    Returns
    -------
    tuple of (float or None, float or None)
        Rounded gaze coordinates ``(x, y)`` if inside the AOI, otherwise
        ``(None, None)``.

    Notes
    -----
    - This function performs a simple spatial filter to ignore gaze samples
      outside the defined AOI.
    - It is non-blocking and returns immediately with the most recent available
      gaze position from the tracker buffer.
    - For full unfiltered raw data, use ``get_full_raw_sample()`` instead.

    Examples
    --------
    >>> gaze = get_gaze_position(config, tracker)
    >>> if gaze != (None, None):
    ...     print(f"Gaze inside AOI at {gaze}")
    """
    gaze_position = tracker.getPosition()

    area_x, area_y = config.get_area_of_interest_size()
    if gaze_position is not None:
        if ( gaze_position[0] >= -area_x/2 and gaze_position[0] <= area_x/2 ) and ( gaze_position[1] >= -area_y/2 and gaze_position[1] <= area_y/2 ):
            return (round(gaze_position[0]), round(gaze_position[1]))

    return (None, None)

def get_full_raw_sample(tracker):
    """
    Retrieve the latest raw Tobii eye-tracker sample with meaningful field names.

    This function returns the most recent gaze sample provided by the PsychoPy
    ioHub Tobii interface. It is designed to extract *all available raw data fields*
    from the tracker, including gaze coordinates, pupil size, gaze origin, and eye
    position for both eyes, without applying any filtering, averaging, or AOI
    (Area of Interest) restriction.

    The function first attempts to call `sample.as_dict()`, which returns a
    fully labeled dictionary of all Tobii-provided fields if supported by the
    current PsychoPy version and Tobii driver. If that is unavailable, it falls
    back to manually labeling known Tobii fields based on the standard sample
    ordering used by PsychoPy's EyeTrackerSampleEvent. Any additional fields
    not matching the expected structure are included as `extra_field_<index>`.

    Parameters
    ----------
    tracker : psychopy.iohub.devices.eyetracker.EyeTrackerDevice
        The active Tobii tracker object created by `launchHubServer()`. It must
        be in a recording state (i.e., `tracker.setRecordingState(True)` called)
        before samples can be retrieved.

    Returns
    -------
    dict or None
        A dictionary containing the most recent Tobii gaze sample with descriptive
        field names. The returned fields typically include (depending on device model
        and configuration):

        - ``timestamp`` : float — system or tracker timestamp for the sample
        - ``left_gaze_x``, ``left_gaze_y``, ``left_gaze_z`` : float — left eye gaze position (3D)
        - ``left_pupil_diameter`` : float — pupil size (mm or device units)
        - ``left_gaze_origin_x``, ``left_gaze_origin_y``, ``left_gaze_origin_z`` : float — gaze origin position
        - ``left_eye_position_x``, ``left_eye_position_y``, ``left_eye_position_z`` : float — eye position in world coordinates
        - ``right_gaze_x``, ``right_gaze_y``, ``right_gaze_z`` : float — right eye gaze position (3D)
        - ``right_pupil_diameter`` : float — pupil size (mm or device units)
        - ``right_gaze_origin_x``, ``right_gaze_origin_y``, ``right_gaze_origin_z`` : float — gaze origin position
        - ``right_eye_position_x``, ``right_eye_position_y``, ``right_eye_position_z`` : float — eye position in world coordinates
        - ``status`` : int — tracking status flag or validity code
        - ``extra_field_<index>`` : any — additional fields returned by the tracker

        Returns ``None`` if no new sample is available.

    Notes
    -----
    - This call is **non-blocking**: it returns immediately with the most recent
      buffered sample or ``None`` if no sample has been received yet.
    - Use this function inside a timed or continuous recording loop to capture
      sequential raw data samples.
    - To record all samples over time, see ``record_all_raw_data()``.

    Examples
    --------
    >>> sample = get_full_raw_sample(tracker)
    >>> if sample:
    ...     print(sample["left_pupil_diameter"], sample["right_pupil_diameter"])

    """
    sample = tracker.getLastSample()
    if sample is None:
        return None

    # Convert PsychoPy iohub sample object (tuple-like) to a dict
    # `getLastSample()` returns an EyeTrackerSampleEvent instance,
    # which has `as_dict()` method returning *all* fields.
    if hasattr(sample, "as_dict"):
        return sample.as_dict()
    else:
        # Fallback: convert tuple to dict with enumerated indices
        return {f"field_{i}": val for i, val in enumerate(sample)}


def get_avg_pupil_size(tracker):
    """
    Compute the average pupil size across both eyes from the most recent sample.

    This function retrieves the last raw Tobii eye-tracker sample and extracts
    the left and right pupil size measurements. It returns the arithmetic mean
    of the two pupil diameters, rounded to four decimal places.

    Parameters
    ----------
    tracker : psychopy.iohub.devices.eyetracker.EyeTrackerDevice
        The active Tobii tracker object created by `launchHubServer()`. It must
        be recording before samples can be retrieved.

    Returns
    -------
    float or None
        Average pupil size across both eyes, rounded to four decimal places.
        Returns ``None`` if no valid sample is available.

    Notes
    -----
    - The sample indices (21 and 40) correspond to left and right pupil
      measurements in the PsychoPy/Tobii sample data structure. These may vary
      slightly depending on device model and SDK version.
    - For full access to all fields, use ``get_full_raw_sample()``.

    Examples
    --------
    >>> avg_size = get_avg_pupil_size(tracker)
    >>> if avg_size:
    ...     print(f"Average pupil size: {avg_size} mm")
    """

    print(tracker.device_class_path)

    if 'mouse' in tracker.device_class_path:
        return None

    sample = tracker.getLastSample()

    if sample is not None:
        avg_pupil_size = (sample[21] + sample[40]) / 2.0  # sample[21]=left_pupil_measure, sample[40]=right_pupil_measure this is only valid for tobii eye trackers
        return round(avg_pupil_size, 4)
    
    return None
