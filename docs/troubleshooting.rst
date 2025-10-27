Troubleshooting
=================

In the following sections we provided common problems with using Tobii-Pytracker with custom datasets and models.

Long importing times
----------------------
When importing Tobii-Pytracker for the first time after installation, you may notice long importing times. This is due to the fact that if you did not install psychopy as shown in the instructions, the package will be installed automatically when importing Tobii-Pytracker for the first time.

Psychopy version issues
-------------------------
We tested software with Psychopy version 2024.1.4 and 2025.2.1. 
If you encounter any issues related to Psychopy, please ensure that you have the correct version installed. 

Ultralytics version issues
---------------------------
Ultralytics package is used for YOLO model handling.
Tobii-Pytracker is tested with Ultralytics version 8.3, but even this version installs automatically torch, that may break the previous requirements regarding ``numpy==1.26.4``.
If you encounter any issues related to numpy version, please ensure that you have the correct version installed.