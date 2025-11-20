==========================
Commandline Usage
==========================

Overview
--------
`tobii-pytracker` is a Python command-line tool for running eye-tracking experiments
or continuous gaze recording using Tobii eye-trackers. The tool supports:

- Displaying stimuli via PsychoPy GUI
- Recording gaze positions and pupil sizes
- Recording full raw Tobii samples
- Voice recording
- Integration with custom models for predictions

Installation
------------
Install the tool via pip:

.. code-block:: bash

    pip install tobii-pytracker

After installation, the `tobii-pytracker` script is available on your PATH.

Command-Line Usage
------------------
Run the tool with:

.. code-block:: bash

    tobii-pytracker [OPTIONS]

General Options
---------------
The following command-line options are available:

- ``--config_file``  
  Path to the YAML configuration file for the experiment (default: ``configs/config.yaml``).  

- ``--eyetracker_config_file``  
  Path to the YAML configuration file for the Tobii eye-tracker (default: ``configs/eyetracker_config.yaml``).  

Boolean Flags
-------------
- ``--enable_eyetracker``  
  Launches the Tobii eye-tracker using PsychoPy ioHub. Requires a connected device.  

- ``--enable_voice``  
  Records voice during trials.  

- ``--raw_data``  
  Records full Tobii raw samples instead of filtered gaze coordinates.  

- ``--disable_psychopy``  
  Runs the tool in headless mode. No PsychoPy GUI is displayed. Continuous
  recording of raw gaze and/or voice occurs until the process is terminated.

Other Options
-------------
- ``--loop_count``  
  Number of trials or data items to display before exiting (default: 10).  

- ``--log_level``  
  Logging level: ``info``, ``debug``, ``warning``, ``error``, ``critical`` (default: ``info``).  

Example Usage
-------------
1. **Run a GUI experiment without eye-tracker just to test the stimuli display**:

.. code-block:: bash

    tobii-pytracker 

1. **Run a GUI experiment with eye-tracker, and voice recording**:

.. code-block:: bash

    tobii-pytracker --enable_eyetracker --enable_voice

2. **Run a headless session recording raw gaze data continuously**:

.. code-block:: bash

    tobii-pytracker --enable_eyetracker --raw_data --disable_psychopy

3. **Run only voice recording with GUI disabled**:

.. code-block:: bash

    tobii-pytracker --enable_voice --disable_psychopy

4. **Run GUI experiment with mouse eyetracker emulation**:

.. code-block:: bash

    tobii-pytracker --eyetracker_config_file ./configs/mouse_eyetracker_config.yaml --enable_eyetracker

File Output
-----------
Depending on mode:

- **PsychoPy mode (`enable_psychopy=True`)**:  

  - ``data.csv`` – contains trial-specific information:
      - ``screenshot_file`` – path to captured screenshot
      - ``input_data`` – data for trial
      - ``classification`` – ground truth label
      - ``user_classification`` – user-selected label
      - ``gaze_data`` – gaze positions or raw samples
      - ``objects_bboxes`` – prediction from custom model
      - ``voice_file`` – recorded voice filename
      - ``voice_start_timestamp`` – timestamp of voice recording start

- **Headless mode (`disable_psychopy=True`)**:  

  - ``raw_stream.csv`` – contains timestamped raw Tobii samples.
  - ``voice_file.wav`` – if voice recording enabled.

Notes
-----
- Headless mode is ideal for continuous monitoring or offline data collection.
- PsychoPy GUI mode allows interactive experiments with user responses.
- Voice recording is threaded and synchronized with gaze sampling.


