Troubleshooting
=================

In the following sections we provided common problems with using Tobii-Pytracker with custom datasets and models.

Long importing times
----------------------
When importing Tobii-Pytracker for the first time after installation, you may notice long importing times. This is due to the fact that if you did not install psychopy as shown in the instructions, the package will be installed automatically when importing Tobii-Pytracker for the first time.

Screenshots crop stimuli images
----------------------------------
If you do not set the resolution of your screen correctly, the whole application will run normally, but the screenshots taken during the experiment will be cropped. Please ensure that you set the correct resolution in the configuration file under the `monitor` section in the main configuration file.

For more details see :ref:`Display configuration <display_configuration>` section.

Psychopy version issues
-------------------------
We tested software with Psychopy versions 2024.1.4 up to 2025.1.1.
The software works perfectly on all these versions, but there are some dependencies that need to be fulfilled (read the rest of troubleshooting for more details).
If you encounter any issues related to Psychopy, please ensure that you have the correct version installed.

The only issue that we found is with the latest Psychopy version `>= 2025.1.1` where in case of mouse eyetracker emulation, the calibration does not start. However, calibration is not needed in case of mouse tracker, so it is only the error that can be ignored.

Ultralytics version issues
---------------------------
Ultralytics package is used for YOLO model handling.
Tobii-Pytracker is tested with Ultralytics version 8.3, but even this version installs automatically torch, that may break the previous requirements regarding ``numpy==1.26.4``.
If you encounter any issues related to numpy version, please ensure that you have the correct version installed.


Issues with screen scaling on Windows
--------------------------------------

On Windows 11 by default content scaling is turned on. It means that the resolution set in settings of the screen is additionally scaled, which makes the alignment of mouse emulated eyetracker and screen screenshots not aligned.

Turn off the scaling for safety: . Go to `Settings` -> `Select Display` -> `Select "Turn Off custom scaling and Sign out"`.

Issues with recording data with Tobii
--------------------------------------

If you encounter the following error when trying to record data with Tobii eyetracker in PsychoPy:

.. code-block:: python

    File "...\lib\site-packages\psychopy_eyetracker_tobii\tobii\eyetracker.py", line 458, in            _getIOHubEventObject
    right_gx, right_gy, right_gz = eye_data_event['right_gaze_origin_in_trackbox_coordinate_system']
    KeyError: 'right_gaze_origin_in_trackbox_coordinate_system'

Make sure that you have the correct version of `tobii-research` and `psychopy-eyetracker-tobii` packages installed.

.. code-block:: text
    psychopy-eyetracker-tobii==0.0.3
    tobii-research==2.0.0 


Since `tobii-research` version 2.1.0 there were some changes in the data provided by Tobii eyetrackers.