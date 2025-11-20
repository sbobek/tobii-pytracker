.. _calibration:

Calibration Procedure
=====================

Before usage of hardware eye trackers, a calibration procedure is required to ensure accurate tracking of the participant's gaze. Calibration typically involves having the participant look at a series of predefined points on the screen while the eye tracker records their eye positions. This data is then used to map the raw eye tracker data to actual screen coordinates.

There are two options of performing calibration with tobii-pytracker. 
One is to use the Tobii Pro Eye Tracker Manager software, which provides a user-friendly interface for calibration and validation. The other option is to perform calibration programmatically using Psychopy's ioHub module, which allows for more customization and integration into experimental scripts.


Calibration with Tobii Pro Eye Tracker Manager
-----------------------------------------------

Download Tobii Pro Eye Tracker Manager from the Tobii website and install it on your computer. Follow the instructions provided in the software to connect to your eye tracker, perform calibration, and validate the calibration results. Once calibration is complete, you can start your experiment using tobii-pytracker to collect eye-tracking data.


Calibration with Psychopy ioHub
----------------------------------

By default, the calibration is handled by the tobii-pytracker software when you start it.
This overrides any previous calibration performed with the Tobii Pro Eye Tracker Manager.
