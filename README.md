# Eyetrackers

## Tobii Pro Nano
### Requirements:
  * Windows
  * Python 3.10
  * Tobii Pro Eye Tracker Manager: [Dwnload](https://connect.tobii.com/s/etm-downloads?language=en_US)
### Instalation
  - Download and instal Anaconda: [Download](https://www.anaconda.com/)
  - Create virtual environemnt with Python 3.10:
  ``` bash
  conda create -n tobii-env python=3.10
  conda activate
  conda install pip
  pip install tobii-research pandas matplotlib
  ```
### Running
  - Tun Eye Tracker Manager to perform calibration
  - Run:  `python tobii_gaze.py <path_to_save_gaze_data.csv>`
  
  Note that tobii saves coordinates in a system, where (0,0) is in the left upper corner, so for the purpose of visualization this coordinates were changed, but file will contain original tobii coordinates.

### Sample output and format

```
device_time_stamp                                                                          467525500
system_time_stamp                                                                         6405415231
left_gaze_point_on_display_area                             (0.4633694291114807, 0.4872185289859772)
left_gaze_point_in_user_coordinate_system          (-10.157917022705078, 128.29026794433594, 40.8...
left_gaze_point_validity                                                                           1
left_pupil_diameter                                                                         5.655228
left_pupil_validity                                                                                1
left_gaze_origin_in_user_coordinate_system         (-25.868297576904297, 1.4193872213363647, 644....
left_gaze_origin_in_trackbox_coordinate_system     (0.5615569949150085, 0.4811278283596039, 0.489...
left_gaze_origin_validity                                                                          1
right_gaze_point_on_display_area                           (0.4944303631782532, 0.44987085461616516)
right_gaze_point_in_user_coordinate_system         (0.7905667424201965, 135.2486572265625, 43.373...
right_gaze_point_validity                                                                          1
right_pupil_diameter                                                                         5.30722
right_pupil_validity                                                                               1
right_gaze_origin_in_user_coordinate_system        (32.527923583984375, -2.9728522300720215, 640....
right_gaze_origin_in_trackbox_coordinate_system    (0.431782990694046, 0.49570250511169434, 0.483...
right_gaze_origin_validity                                                                         
```

