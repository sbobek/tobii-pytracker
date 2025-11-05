Troubleshooting
=================

In the following sections we provided common problems with using Tobii-Pytracker with custom datasets and models.

Long importing times
----------------------
When importing Tobii-Pytracker for the first time after installation, you may notice long importing times. This is due to the fact that if you did not install psychopy as shown in the instructions, the package will be installed automatically when importing Tobii-Pytracker for the first time.

Screenshots crop stimuli images
----------------------------------
If you do not set the resolution of your screen correctly, the whole application will run normally, but the screenshots taken during the experiment will be cropped. Please ensure that you set the correct resolution in the configuration file under the `monitor` section in the main configuration file.

For more details see :ref:`_display_configuration` section.

Psychopy version issues
-------------------------
We tested software with Psychopy version 2024.1.4.
If you encounter any issues related to Psychopy, please ensure that you have the correct version installed. 

Ultralytics version issues
---------------------------
Ultralytics package is used for YOLO model handling.
Tobii-Pytracker is tested with Ultralytics version 8.3, but even this version installs automatically torch, that may break the previous requirements regarding ``numpy==1.26.4``.
If you encounter any issues related to numpy version, please ensure that you have the correct version installed.

Issues with pyglet and WMFDecoder
----------------------------------
On Windows you may see errors related to pyglet package, such as: `AttributeError: 'WMFDecoder' object has no attribute 'MFShutdown'`.
We do not use pyglet directly, but it is a dependency of Psychopy, hence you may ignore these errors, as they do not affect the functionality of Tobii-Pytracker.

