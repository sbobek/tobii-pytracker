
Linux Installation
==================

The package supports Linux systems running Python 3.10. The commands below
are intended for Debian- and Ubuntu-based distributions.

System dependencies
-------------------

Install the system libraries used by the graphical, audio, camera, and
scientific Python dependencies:

.. code-block:: bash

   sudo apt update
   sudo apt install -y \
      build-essential \
      pkg-config \
      libgl1 \
      libglib2.0-0 \
      libgtk-3-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libxrandr2 \
      libxi6 \
      libxcursor1 \
      libxinerama1 \
      libxkbcommon-x11-0 \
      libx11-xcb1 \
      libxcb1 \
      libfontconfig1 \
      libfreetype6 \
      libasound2 \
      libsndfile1 \
      libportaudio2 \
      libusb-1.0-0 \
      ffmpeg

On newer Ubuntu releases, ``libasound2`` may be provided through a
time64-specific package. In that case, install the package suggested by
``apt`` instead.

Virtual environment
-------------------

Create and activate a virtual environment:

.. code-block:: bash

   conda create -n pytracker-env python=3.10 -y
   conda activate pytracker-env

Upgrade the packaging tools:

.. code-block:: bash

   pip install --upgrade pip wheel

Install ``tobii-pytracker`` from PyPI:

.. code-block:: bash

   pip install tobii-pytracker


Additionally, if you plan to use VoiceTranscriptionAnalyzer, install whisper:

.. code-block:: bash

   pip install  openai-whisper==20250625
  

Display configuration
---------------------

Graphical experiments require a running graphical session. Verify that the
``DISPLAY`` variable is set:

.. code-block:: bash

   echo "$DISPLAY"

When running inside Docker or another headless environment, configure an
X11 display or a virtual display such as Xvfb before starting an experiment.

Tobii eye tracker
------------------

Connect the computer and the eye tracker to the same network. Ensure that
the firewall allows communication between them. The tracker should normally
be discoverable automatically by the Tobii SDK.

If the tracker is not detected, check:

* the tracker and computer have compatible IP addresses;
* the network connection is not isolated by Wi-Fi client isolation;
* the firewall is not blocking Tobii SDK traffic;
* the Tobii SDK is installed in the active virtual environment.

Verify the installation
-----------------------

Run the following command from the activated virtual environment:

.. code-block:: bash

   python -c "import tobii_research, tobii_pytracker; print('Installation successful')"

The command-line interface can be tested with:

.. code-block:: bash

   tobii-pytracker --help