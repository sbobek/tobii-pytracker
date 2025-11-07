Release notes
==============

0.1.2 (2025-11-06)
-------------
* Fixed documentation of command-line options
* Added MouseEyetracker emulator, for testing without Tobii hardware
* Fixed timestamp synchronization, now timestamps are raw time from core of Psychopy
* Fixed dependency issues. Psychopy version is now pinned to 2024.1.4 to avoid compatibility issues.
* Fixed issues with mouse gaze data collection (multiple gaze points when no RBS button clicked)
* Refactored code for analyzers and data loaders.
* Added full set of readings from eyetracker (not limited to average pupil size, or average gaze point)
* Added HeatMapAnalyzer, FixationAnalyzer, FocusMapAnalyzer, SaccadeAnalyzer complete code


0.1.1 (2025-10-28)
-------------
* Added raw data recording option with `--raw_data` flag.
* Fixed configuration loading for fixation dot parameters.
* Updated documentation with command-line usage examples.
* Disabling PsychoPy GUI now properly records raw gaze data continuously with no GUI.


0.1.0 (2025-10-24)
-------------
* First working version released to PyPI and GitHub.