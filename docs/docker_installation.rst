

Docker Installation
===================

The repository includes a Docker image based on Python 3.10 and Debian
Bookworm. The image installs the Linux system libraries required by
PsychoPy, Matplotlib, audio processing, OpenCV, FFmpeg, and the Tobii SDK.

Prerequisites
-------------

Install Docker Engine or Docker Desktop and verify the installation:

.. code-block:: bash

   docker --version



Using the startup script
------------------------

On Linux, macOS, WSL, or another Bash-compatible environment, the repository
provides ``start-docker.sh`` as a shortcut for starting the application with
Docker Compose.

Make the script executable once:

.. code-block:: bash

   chmod +x start-docker.sh

Run it from the repository root:

.. code-block:: bash

   ./start-docker.sh

The script:

* verifies that Docker and Docker Compose are installed;
* creates the ``output`` directory and configures its permissions;
* builds the image when required;
* starts the services using ``docker compose up --build``;
* reports that the VNC server is available at ``localhost:5900``.

To connect with the VNC server use a VNC client such as `TigerVNC <https://tigervnc.org/>`_ or `RealVNC <https://www.realvnc.com/en/connect/download/viewer/>`_.

Press ``Ctrl+C`` to stop the application. The script must be run from a
Bash-compatible terminal. On Windows, use WSL or Git Bash, or run the Docker
commands manually from  as described below.

Build the image
---------------
Alternatively, you can perform all of the steps from ``start-docker.sh`` manually.
Run the following command from the repository root:

.. code-block:: bash

   docker build -t tobii-pytracker .

The Dockerfile installs the Python dependencies from 
``requirements-docker.txt`` and then installs the project in editable mode.

Run the container
-----------------

The default command starts an experiment with the example mouse
eye-tracker configuration:

.. code-block:: bash

   docker run --rm -it tobii-pytracker

The default Dockerfile command is equivalent to:

.. code-block:: bash

   python -m tobii_pytracker.main \
       --eyetracker_config_file ./configs/mouse_eyetracker_config.yaml \
       --enable_eyetracker

Mount local files
-----------------

To use local configurations, recordings, or experiment files, mount a host
directory into the container:

.. code-block:: bash

   docker run --rm -it \
       -v "$PWD/configs:/app/configs" \
       -v "$PWD/data:/app/data" \
       tobii-pytracker

On Windows PowerShell, use:

.. code-block:: powershell

   docker run --rm -it `
       -v "${PWD}\configs:/app/configs" `
       -v "${PWD}\data:/app/data" `
       tobii-pytracker

Graphical applications
----------------------

The image sets ``DISPLAY=:99``. This is suitable for a virtual X11 display,
but ``Xvfb`` must be running before a graphical experiment starts.

To start Xvfb automatically, change the final ``CMD`` instruction in
``Dockerfile`` to:

.. code-block:: dockerfile

   CMD ["sh", "-c", "Xvfb :99 -screen 0 1280x1024x24 & exec python -m tobii_pytracker.main --eyetracker_config_file ./configs/mouse_eyetracker_config.yaml --enable_eyetracker"]

For interactive debugging, start a shell instead:

.. code-block:: bash

   docker run --rm -it --entrypoint /bin/bash tobii-pytracker

Then start the virtual display and the application manually:

.. code-block:: bash

   Xvfb :99 -screen 0 1280x1024x24 &
   python -m tobii_pytracker.main \
       --eyetracker_config_file ./configs/mouse_eyetracker_config.yaml \
       --enable_eyetracker

The Dockerfile also installs ``x11vnc``. It can be used when remote access to
the virtual display is required, but it is not started automatically.

Changing the application behaviour
-----------------------------------

The application command is defined by the final ``CMD`` instruction in
``Dockerfile``. Rebuild the image after changing it.

Disable the eye tracker by removing ``--enable_eyetracker``:

.. code-block:: dockerfile

   CMD ["python", "-m", "tobii_pytracker.main", "--eyetracker_config_file", "./configs/mouse_eyetracker_config.yaml"]

Use another eye-tracker configuration:

.. code-block:: dockerfile

   CMD ["python", "-m", "tobii_pytracker.main", "--eyetracker_config_file", "./configs/my_eyetracker_config.yaml", "--enable_eyetracker"]

Enable voice or another application feature by adding the corresponding
command-line option to ``CMD``. For example, if the application exposes an
``--enable_voice`` option:

.. code-block:: dockerfile

   CMD ["python", "-m", "tobii_pytracker.main", "--eyetracker_config_file", "./configs/mouse_eyetracker_config.yaml", "--enable_voice"]

The exact option names can be checked with:

.. code-block:: bash

   docker run --rm tobii-pytracker \
       python -m tobii_pytracker.main --help

Other useful Dockerfile settings include:

* ``DISPLAY`` — X11 display used by graphical applications;
* ``MPLBACKEND`` — Matplotlib backend; ``Agg`` is suitable for headless use;
* ``PYTHONPATH`` — source directory used by the container;
* ``CMD`` — default command started when the container runs.

For a temporary command change, use ``docker run`` without editing the
Dockerfile:

.. code-block:: bash

   docker run --rm -it --entrypoint python tobii-pytracker \
       -m tobii_pytracker.main --help

Tobii eye tracker access
------------------------

The eye tracker and the host computer must be connected to the same network.
The container must also be able to reach the tracker.

On native Linux, host networking can simplify device discovery:

.. code-block:: bash

   docker run --rm -it --network host tobii-pytracker

On Docker Desktop for Windows or macOS, ``--network host`` does not provide
the same networking behaviour as native Linux. In that case, ensure that the
container can reach the tracker's IP address and that the required network
traffic is not blocked by the host firewall.

If the tracker is not detected, check:

* the tracker and computer have compatible IP addresses;
* the network does not use Wi-Fi client isolation;
* the host and container firewalls allow the Tobii SDK traffic;
* the Tobii SDK is installed in the image;
* the container can reach the tracker's IP address.

Audio and voice features
------------------------

The image includes Linux audio libraries, PortAudio, PulseAudio support,
and FFmpeg. Audio access still depends on the host operating system and
Docker runtime configuration.

If audio devices are required, additional device or PulseAudio configuration
may be necessary. For example, on Linux:

.. code-block:: bash

   docker run --rm -it \
       --device /dev/snd \
       tobii-pytracker

The Python packages used by the feature must also be listed in
``requirements-docker.txt`` or in the project's dependencies.

Rebuild after changes
---------------------

Rebuild the image whenever ``Dockerfile``, ``requirements-docker.txt``,
``pyproject.toml``, or application code affecting installation changes:

.. code-block:: bash

   docker build --no-cache -t tobii-pytracker .

Verify the installation
-----------------------

Run:

.. code-block:: bash

   docker run --rm tobii-pytracker \
       python -c "import tobii_research, tobii_pytracker; print('Installation successful')"

Test the command-line interface:

.. code-block:: bash

   docker run --rm tobii-pytracker \
       tobii-pytracker --help