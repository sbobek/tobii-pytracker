Windows Installation
=====================

Tobii-Pytracker can be installed from either `PyPI <https://pypi.org/project/tobii-pytracker>`_ or directly from source code `GitHub <https://github.com/sbobek/tobii-pytracker>`_

To install form PyPI::

   pip install tobii-pytracker

To install from source code::

   git clone https://github.com/sbobek/tobii-pytracker
   cd tobii-pytracker
   pip install .
   pip install "psychopy>=2024.1.4,<2025.1.0" --no-deps

To run it in headless mode, just recording all of the gaze data from Tobii eye tracker just run the following command in the environment where tobii-pytracker is installed. 
To run it with mouse emulation, you can use the following command::

   tobii-pytracker --eyetracker_config_file ./configs/mouse_eyetracker_config.yaml --enable_eyetracker 