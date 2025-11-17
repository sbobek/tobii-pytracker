

Configuration
=================

Tobii-Pytracker uses a YAML configuration file to set up various parameters for eye-tracking experiments and data collection. This file allows users to customize settings such as monitor specifications, GUI options, areas of interest, and output preferences.

You need to prepare two separate configuration files:
1. Main configuration file (e.g., `config.yaml`): This file contains general settings for the experiment, including monitor details, GUI options, areas of interest, and output folder specifications.
2. Eyetracker configuration file (e.g., `eyetracker_config.yaml`): This file includes settings specific to the Tobii eye tracker, such as sampling rate, data streams to record, and calibration options.

The default version of the files can be found in the `configs/` directory of the Tobii-Pytracker repository. You can copy these files and modify them according to your experimental requirements.

Main Configuration file
---------------------------

The main file consists of several sections:

1. `dataset`: Specifies the path to the dataset used in the experiment. You can also define text-related parameters if applicable.
2. `display`: Contains monitor specifications (name, resolution, width, distance) and GUI options (button size, margin, color, fixation dot size and color, area of interest dimensions).
3. `output`: Defines the folder where output data will be saved.
4. `model`: Specifies the folder containing models, as well as the module and class names for custom models used in the experiment.
5. `instructions`: Provides introductory and concluding instructions for participants in the experiment.

The example of the main configuration file is shown below:

.. code-block:: yaml

    dataset:
    path: datasets/pdt

    display:
    monitor: 
        name: spectrum_monitor
        resolution:
        - 1536
        - 960
        width: 35
        distance: 60
    gui:
        button:
        size:
            - 250
            - 100
        margin: 20
        color: lightgrey
        text:
            color: black
            size: 30
        fixation_dot: 
        size: 10
        color: white
        aoe:
        - 750
        - 750

    output:
      folder: output

    bbox_model:
      folder: custom_runtime_models
      module: custom_yolo_model
      class: CustomYoloModel

    instructions:
      intro:
          - "Welcome to the study!"
          - ""
          - "In this experiment, you will see a series of images or text samples."
          - "Please look at each stimulus carefully, then select the appropriate option using the buttons below."
          - ""
          - "Press SPACE to begin."
      outro:
          - "Thank you for participating in this study!"
          - ""
          - "Your responses and recordings have been saved."
          - "You may now close the window or press ESC to exit."


Below is a detailed explanation of each section and its parameters, along with examples.

1. ``dataset``
~~~~~~~~~~~~~~~~

Specifies the dataset used for the experiment, including the path to stimuli and labeling details.  
The format of the dataset depends on whether you are using **images** or **text-based stimuli**.

**Example:**

.. code-block:: yaml

    dataset:
      path: datasets/pdt

**Parameters:**

- **``path``** (*str*):  
  Path to the dataset directory or CSV file containing experimental stimuli.

**Image Datasets**

Image datasets must be organized so that each **subdirectory represents a class** (category).  
The GUI automatically generates response buttons according to the class labels extracted from these subdirectory names.

**Example directory structure:**

.. code-block:: text

    ├── dataset
    │   ├── category1
    │   │   ├── file1.jpg
    │   │   ├── file2.jpg
    │   │   └── ...
    │   ├── category2
    │   │   ├── file1.jpg
    │   │   ├── file2.jpg
    │   │   └── ...
    │   └── ...
    └── ...

**Notes:**

- The GUI layout and button labels will automatically match the subfolder names (``category1``, ``category2``, ...).  
- If you are using a **custom model**, ensure that its output class names **exactly match** these subfolder names.

**Text Datasets**

Text-based datasets must be provided in **CSV format**.  
Each row represents a text sample, and the column containing class labels must have a **header name** that matches the configuration setting.

**Example configuration:**

.. code-block:: yaml

    dataset:
      path: path/to/file.csv
      text:
        label_column_name: label_column_name
        text_column_name: text_column_name

**Example file structure:**

.. code-block:: text

    ├── dataset.csv
    └── ...

**Notes:**

- The **first line** of the CSV file **must be a header**.  
- The GUI will automatically generate labels and response options based on the **unique values** in the column specified as ``label_column_name``.  
- The column defined as ``text_column_name`` will be displayed as the main text stimulus during the experiment.

---

.. _display_configuration:

2. ``display``
~~~~~~~~~~~~~~~~

Defines monitor characteristics and graphical user interface (GUI) settings.  
This section includes two main subsections: ``monitor`` and ``gui``.

**a. monitor**

Specifies hardware and geometric properties of the monitor.

**Example:**

.. code-block:: yaml

    display:
      monitor:
        name: spectrum_monitor
        resolution:
          - 1536
          - 960
        width: 35
        distance: 60

**Parameters:**

- **``name``** (*str*):  
  Identifier for the monitor used in the experiment. This can be any descriptive name.
- **``resolution``** (*list[int, int]*):  
  Screen resolution in pixels ``[width, height]``. Note that in case of multiple screens, it assumes the resolution of the primary monitor. If the resolution is provided incorrectly , the application will display correctly, but the screenshots taken during the experiment may have incorrect dimensions or capture incorrect regions.
- **``width``** (*float*):  
  Physical width of the display in centimeters.
- **``distance``** (*float*):  
  Distance between the participant and the display (in centimeters).

**b. gui**

Configures visual elements displayed during the experiment, such as buttons, fixation dots, and areas of interest (AOE).

**Example:**

.. code-block:: yaml

      gui:
        button:
          size:
            - 250
            - 100
          margin: 20
          color: lightgrey
          text:
            color: black
            size: 30
        fixation_dot:
          size: 10
          color: white
        aoe:
          - 750
          - 750

**Parameters:**

- **``button``**:  
  Defines button appearance and behavior.

  - **``size``** (*list[int, int]*): Dimensions ``[width, height]`` in pixels.  
  - **``margin``** (*int*): Padding or spacing around buttons (in pixels).  
  - **``color``** (*str*): Background color.  
  - **``text.color``** (*str*): Text color.  
  - **``text.size``** (*int*): Text font size.

- **``fixation_dot``**:  
  Defines the central dot shown before or between trials.

  - **``size``** (*int*): Diameter in pixels.  
  - **``color``** (*str*): Dot color.

- **``aoe``** (*list[int, int]*):  
  Area of interest size ``[width, height]`` in pixels, determining where stimuli appear. The size of aoe should be chosen based on the expected gaze distribution and the nature of the stimuli to ensure accurate data collection. The stimuli will be centered within this area and scaled to fit the defined dimensions.

---

3. ``output``
~~~~~~~~~~~~~~~~

Defines where experimental results and recordings are saved.

**Example:**

.. code-block:: yaml

    output:
      folder: output

**Parameters:**

- **``folder``** (*str*):  
  Directory where all output files (gaze data, responses, logs) are stored.

---

4. ``bbox_model``
~~~~~~~~~~~~~~~~

Specifies the model used in the experiment, including its location and class definitions.
Every dataset has build-in bounding box detection mode, so this filed is optional and can be omitted.

**Example:**

.. code-block:: yaml

    bbox_model:
      folder: custom_runtime_models
      module: custom_yolo_model
      class: CustomYoloModel

**Parameters:**

- **``folder``** (*str*):  
  Path to the folder containing trained model files.
- **``module``** (*str*):  
  Name of the Python module implementing the model.
- **``class``** (*str*):  
  Name of the model class defined in the module.

**Note:**  
When using a **custom model**, ensure that its class labels **exactly match** the subfolder names in the dataset to maintain consistency between predicted and displayed classes.

---

Custom Model
~~~~~~~~~~~~~~~~

The ``CustomModel`` class (located in ``runtime_models/custom_model.py``) serves as an **abstract base class** that defines the required interface for all custom models.  
By inheriting from ``CustomModel`` and implementing its methods, your model can seamlessly integrate with the Tobii-Pytracker pipeline.

**Required Methods:**

- **``prepare_model(self)``**  
  Load and prepare the model for prediction.

- **``predict(self, input_data)``**  
  Run a forward pass and return raw model predictions.

- **``process(self, path)``**  
  Post-process the prediction results and return formatted output for storage or display.

**Creating Your Own Model Module**

To create your own model:

1. Create a new Python file within the ``custom_runtime_models/`` directory  in folder where you run a script (e.g., ``custom_runtime_models/my_custom_model.py``).  
2. Import the ``CustomModel`` base class:

   .. code-block:: python

       from runtime_models.custom_model import CustomModel

3. Define your new model class inheriting from ``CustomModel``:

   .. code-block:: python

       class MyCustomModel(CustomModel):

           def prepare_model(self):
               # Load and prepare model for prediction
               self.logger.debug("Preparing MyCustomModel...")
               self.model = ...
               self.logger.debug("Model preparation done.")

           def predict(self, input_data):
               # Run prediction with the loaded model
               predictions = ...
               return predictions

           def process(self, data):
               # Post-process the prediction and return formatted results
               predictions = self.predict(data)
               processed_predictions = ...
               return processed_predictions

**Important:**  
The main processing pipeline uses the ``process()`` method to execute and store model predictions.  
Ensure this method returns data in the expected format for downstream components.

---

5. ``instructions``
~~~~~~~~~~~~~~~~~~~~

Defines text messages shown before (intro) and after (outro) the experiment.  
These are displayed within the GUI and can include empty strings for spacing.

**Example:**

.. code-block:: yaml

    instructions:
      intro:
        - "Welcome to the study!"
        - ""
        - "In this experiment, you will see a series of images or text samples."
        - "Please look at each stimulus carefully, then select the appropriate option using the buttons below."
        - ""
        - "Press SPACE to begin."
      outro:
        - "Thank you for participating in this study!"
        - ""
        - "Your responses and recordings have been saved."
        - "You may now close the window or press ESC to exit."

**Parameters:**

- **``intro``** (*list[str]*):  
  Lines of text displayed before the experiment begins.
- **``outro``** (*list[str]*):  
  Lines of text displayed after the experiment ends.

---


Eyetracker configuration file
-------------------------------
The eye tracker configuration file should include the fields required by launchHubServer from the psychopy.iohub module. Refer to the PsychoPy documentation for more details on the specific settings.
The default eyetracker configuration file can be found in the `configs/eyetracker_config.yaml` file of the Tobii-Pytracker repository.

This file also includes mouse emulation settings, which allow you to simulate eye-tracking data using mouse movements. This is particularly useful for testing and development purposes when an actual eye tracker is not available.

In most cases, these default settings should work without any modifications.