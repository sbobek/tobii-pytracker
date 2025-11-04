import yaml
from psychopy.iohub import launchHubServer
from .custom_logger import CustomLogger
from tobii_pytracker.configs.custom_config import CustomConfig


LOGGER = CustomLogger("debug", "eyetracker").logger
_BUTTON_MAP = {"LEFT_BUTTON": 0, "MIDDLE_BUTTON": 1, "RIGHT_BUTTON": 2}


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

def poll_tracker_events(tracker, buffer, last_event_id=0):
    """
    Poll all new eye-tracker events into a local buffer (no duplicates).
    """
    max_event_id = last_event_id

    try:
        all_events = tracker.getEvents()
    except TypeError:
        LOGGER.debug("tracker.getEvents() failed; continuing with empty list.")
        all_events = []

    if not all_events:
        return buffer, max_event_id

    for ev in all_events:
        eid = getattr(ev, "event_id", None)
        if eid is None or eid <= last_event_id:
            continue

        etype_name = ev.__class__.__name__
        buffer.setdefault(etype_name, []).append(ev)
        max_event_id = max(max_event_id, eid)

    # NOTE: do not call tracker.clearEvents() here, to avoid skipping samples
    return buffer, max_event_id


def extract_full_raw_event(buffer, system_time):
    """Return full raw data from all buffered eye-sample events."""
    results = []
    for etype in list(buffer.keys()):
        events = buffer.pop(etype, [])
        for ev in events:
            rec = {k: getattr(ev, k)
                   for k in dir(ev)
                   if not k.startswith("_") and not callable(getattr(ev, k, None))}
            rec["event_type"] = etype
            rec["system_time"] = system_time
            results.append(rec)
    return results


def extract_eye_gaze_events(buffer, system_time, config):
    """Extract gaze and pupil data from buffered events, only if gaze within AOI."""
    results = []

    # Get AOI bounds
    area_x, area_y = config.get_area_of_interest_size()
    half_x, half_y = area_x / 2, area_y / 2

    def within_aoi(x, y):
        """Check if given (x, y) is inside the AOI."""
        if x is None or y is None:
            return False
        return (-half_x <= x <= half_x) and (-half_y <= y <= half_y)

    for etype in list(buffer.keys()):
        if "EyeSampleEvent" not in etype:
            continue
        events = buffer.get(etype, [])
        if not events:
            continue

        remaining = []
        for ev in events:
            eid = getattr(ev, "event_id", None)
            if eid is None:
                remaining.append(ev)
                continue

            attrs = dir(ev)
            has_left = any(a.startswith("left_") for a in attrs)
            has_right = any(a.startswith("right_") for a in attrs)

            rec = {
                "event_type": etype,
                "event_id": eid,
                "logged_time": getattr(ev, "logged_time", getattr(ev, "time", None)),
                "system_time": system_time,
                "gaze_x_left": None,
                "gaze_y_left": None,
                "gaze_x_right": None,
                "gaze_y_right": None,
                "pupil_left": None,
                "pupil_right": None,
                "avg_gaze_x": None,
                "avg_gaze_y": None,
                "avg_pupil_size": None,
            }

            if has_left or has_right:
                # binocular
                rec["gaze_x_left"] = getattr(ev, "left_gaze_x", None)
                rec["gaze_y_left"] = getattr(ev, "left_gaze_y", None)
                rec["gaze_x_right"] = getattr(ev, "right_gaze_x", None)
                rec["gaze_y_right"] = getattr(ev, "right_gaze_y", None)
                rec["pupil_left"] = getattr(ev, "left_pupil_measure1", None) or getattr(ev, "left_pupil_measure", None)
                rec["pupil_right"] = getattr(ev, "right_pupil_measure1", None) or getattr(ev, "right_pupil_measure", None)
            else:
                # monocular
                rec["gaze_x_left"] = getattr(ev, "gaze_x", None)
                rec["gaze_y_left"] = getattr(ev, "gaze_y", None)
                rec["pupil_left"] = getattr(ev, "pupil_measure1", None) or getattr(ev, "pupil_measure", None)
                rec["pupil_right"] = getattr(ev, "pupil_measure2", None)

            # Compute averages (if both exist or one available)
            gx_l, gx_r = rec["gaze_x_left"], rec["gaze_x_right"]
            gy_l, gy_r = rec["gaze_y_left"], rec["gaze_y_right"]
            pl, pr = rec["pupil_left"], rec["pupil_right"]

            if gx_l is not None and gx_r is not None:
                rec["avg_gaze_x"] = (gx_l + gx_r) / 2
            elif gx_l is not None:
                rec["avg_gaze_x"] = gx_l
            elif gx_r is not None:
                rec["avg_gaze_x"] = gx_r

            if gy_l is not None and gy_r is not None:
                rec["avg_gaze_y"] = (gy_l + gy_r) / 2
            elif gy_l is not None:
                rec["avg_gaze_y"] = gy_l
            elif gy_r is not None:
                rec["avg_gaze_y"] = gy_r

            if pl is not None and pr is not None:
                rec["avg_pupil_size"] = (pl + pr) / 2
            elif pl is not None:
                rec["avg_pupil_size"] = pl
            elif pr is not None:
                rec["avg_pupil_size"] = pr

            # ✅ AOI filtering
            if (
                within_aoi(rec["gaze_x_left"], rec["gaze_y_left"]) or
                within_aoi(rec["gaze_x_right"], rec["gaze_y_right"]) or
                within_aoi(rec["avg_gaze_x"], rec["avg_gaze_y"])
            ):
                results.append(rec)

        # update buffer
        if remaining:
            buffer[etype] = remaining
        else:
            buffer.pop(etype, None)

    return results





def get_tracker_class(iohub_config: dict) -> str:
    """Return the tracker class key name (e.g. 'eyetracker.hw.mouse.EyeTracker')."""
    for key in iohub_config.keys():
        if key.lower().startswith("eyetracker.hw."):
            return key
    raise ValueError("No eyetracker configuration found in iohub_config.")


def is_mouse_eyetracker(iohub_config: dict) -> bool:
    """Return True if the tracker class is the mouse eyetracker."""
    tracker_class = get_tracker_class(iohub_config)
    return "mouse" in tracker_class.lower()


def get_mouse_move_button_idx(iohub_config: dict, default="RIGHT_BUTTON") -> int:
    """
    Read the move button for the mouse eyetracker from iohub_config.
    Returns PsychoPy button index (0=left, 1=middle, 2=right).
    """
    tracker_class = get_tracker_class(iohub_config)
    tracker_conf = iohub_config[tracker_class]

    controls = tracker_conf.get("controls", {})
    move_button = controls.get("move", default)
    move_button = move_button.upper()

    return _BUTTON_MAP.get(move_button, _BUTTON_MAP[default])