Configuration
=============

Tobii-Pytracker uses YAML configuration files to define the dataset, display,
output, model, participant instructions, and eye-tracker settings used during an
experiment.

Two configuration files are used:

1. A main configuration file, for example ``config.yaml``, which defines the
   dataset and experiment interface.
2. An eye-tracker configuration file, for example
   ``eyetracker_config.yaml``, which defines the eye-tracker device, recorded
   streams, sampling, and calibration settings. For calibration details, see
   :ref:`calibration`.

Default configuration files are available in the ``configs/`` directory of the
Tobii-Pytracker repository. Copy them and adjust the values to the experimental
setup rather than editing the defaults directly.

Main configuration file
-----------------------

The main configuration file contains the following top-level sections:

* ``dataset``: selects one stimulus modality and configures its data source and
  built-in bounding-box model.
* ``display``: defines the monitor geometry and graphical user interface.
* ``output``: defines the output directory.
* ``bbox_model``: optionally loads a custom runtime bounding-box model.
* ``instructions``: defines the introductory and concluding participant text.

Complete example
~~~~~~~~~~~~~~~~

The example below activates an image dataset. The text and time-series examples
are included as comments and can be enabled by replacing the active ``image``
subsection. Only one dataset subsection should be active for a study.

.. code-block:: yaml

   dataset:
     image:
       bbox_model: superpixel  # grid | superpixel | saliency
       path: datasets/vehicles

     # text:
     #   label_column_name: sentiment
     #   text_column_name: selected_text
     #   color: white
     #   background_color: black
     #   bbox_model: word  # word | line | sentence
     #   path: datasets/twitter/tweets.csv

     # time_series:
     #   # Dataset: ECG200 (UCR Time Series Archive)
     #   # License: CC BY 4.0
     #   # Source: https://www.timeseriesclassification.com
     #   label_column_name: class
     #   bbox_model: sample  # sample | window
     #   path: datasets/ecg/ecg200.csv

   display:
     monitor:
       name: spectrum_monitor
       resolution:
         - 2560
         - 1440
       width: 35
       distance: 60
       display_number: 0
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

   # bbox_model:
   #   folder: custom_runtime_models
   #   module: custom_yolo_model
   #   class: CustomYoloModel

   instructions:
     intro:
       - "Welcome to the study!"
       - ""
       - "In this experiment, you will see a series of images or text samples."
       - "Please look at each stimulus carefully, then select the appropriate option using the buttons below."
       - ""
       - "Click on the window and press SPACE to begin."
     outro:
       - "Thank you for participating in this study!"
       - ""
       - "Your responses and recordings have been saved."
       - "You may now close the window or press ESC to exit."

Dataset configuration
---------------

The ``dataset`` section now uses a modality-specific subsection. Select exactly
one of the following subsections:

* ``image`` for image stimuli stored in class directories;
* ``text`` for text stimuli loaded from a CSV file;
* ``time_series`` for time-series samples loaded from a CSV file.

The selected subsection determines which dataset class loads and presents the
stimuli. Each modality also defines its own built-in ``bbox_model`` options.
Bounding boxes are expressed in the stimulus coordinate system used by the
experiment and can be included in the recorded output when bounding-box
calculation is enabled.

Image datasets
~~~~~~~~~~~~~~

Image datasets are configured under ``dataset.image``.

.. code-block:: yaml

   dataset:
     image:
       bbox_model: superpixel
       path: datasets/vehicles

Parameters
^^^^^^^^^^

``path`` (str)
   Path to the root directory containing the image dataset.

``bbox_model`` (str)
   Built-in method used to divide or identify image regions. Supported values
   are ``grid``, ``superpixel``, and ``saliency``.

The image directory must contain one subdirectory per class. The class labels
are obtained from the subdirectory names and are used to generate the response
options in the experiment interface.

.. code-block:: text

   datasets/vehicles/
   |-- car/
   |   |-- image_001.jpg
   |   `-- image_002.jpg
   |-- motorcycle/
   |   |-- image_001.jpg
   |   `-- image_002.jpg
   `-- truck/
       |-- image_001.jpg
       `-- image_002.jpg

When a custom model is used, its class names must match the image subdirectory
names exactly.

Text datasets
~~~~~~~~~~~~~

Text datasets are configured under ``dataset.text`` and loaded from a CSV file.
Each row represents one stimulus.

.. code-block:: yaml

   dataset:
     text:
       label_column_name: sentiment
       text_column_name: selected_text
       color: white
       background_color: black
       bbox_model: word
       path: datasets/twitter/tweets.csv

Parameters
^^^^^^^^^^

``path`` (str)
   Path to the CSV file containing the text stimuli.

``label_column_name`` (str)
   Name of the CSV column containing the class label. Unique values from this
   column are used as response classes.

``text_column_name`` (str)
   Name of the CSV column containing the text displayed to the participant.

``color`` (str)
   Color of the displayed text.

``background_color`` (str)
   Background color used when presenting the text stimulus.

``bbox_model`` (str)
   Granularity of the text regions. Supported values are ``word``, ``line``,
   and ``sentence``.

The first row of the CSV file must contain column names, including the columns
specified by ``label_column_name`` and ``text_column_name``.

Time-series datasets
~~~~~~~~~~~~~~~~~~~~

Time-series datasets are configured under ``dataset.time_series`` and loaded
from a CSV file.

.. code-block:: yaml

   dataset:
     time_series:
       label_column_name: class
       bbox_model: sample
       path: datasets/ecg/ecg200.csv

Parameters
^^^^^^^^^^

``path`` (str)
   Path to the CSV file containing the time-series samples.

``label_column_name`` (str)
   Name of the CSV column containing the class assigned to each series.

``bbox_model`` (str)
   Granularity at which time-series regions are represented. Supported values
   are ``sample`` and ``window``.

The CSV file must include a header and the label column specified by
``label_column_name``. Remaining data columns are interpreted as time-series
values by the time-series dataset loader.

.. _display_configuration:

Display Configuration
--------------

The ``display`` section contains the ``monitor`` and ``gui`` subsections.

Monitor configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   display:
     monitor:
       name: spectrum_monitor
       resolution:
         - 2560
         - 1440
       width: 35
       distance: 60
       display_number: 0

Parameters
^^^^^^^^^^

``name`` (str)
   Descriptive identifier for the monitor.

``resolution`` (list[int, int])
   Screen resolution in pixels as ``[width, height]``. An incorrect resolution
   can cause screenshots or captured regions to have incorrect dimensions.

``width`` (float)
   Physical display width in centimetres.

``distance`` (float)
   Distance between the participant and the display in centimetres.

``display_number`` (int)
   Index of the display on which the experiment window is presented. This is
   particularly relevant in multi-monitor setups.

GUI configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   display:
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

Parameters
^^^^^^^^^^

``button.size`` (list[int, int])
   Button dimensions as ``[width, height]`` in pixels.

``button.margin`` (int)
   Spacing around buttons in pixels.

``button.color`` (str)
   Button background color.

``button.text.color`` (str)
   Button text color.

``button.text.size`` (int)
   Button text size.

``fixation_dot.size`` (int)
   Fixation-dot size in pixels.

``fixation_dot.color`` (str)
   Fixation-dot color.

``aoe`` (list[int, int])
   Area of experiment size as ``[width, height]`` in pixels. Stimuli are centred
   and scaled to fit this area.

Output Folder
-------------

.. code-block:: yaml

   output:
     folder: output

``folder`` (str)
   Directory in which experiment outputs, including gaze data, responses, and
   logs, are saved.

BoundingBox Model
-----------------

The optional top-level ``bbox_model`` section loads a custom runtime model. It
is different from the modality-specific ``dataset.<modality>.bbox_model``
setting, which selects one of the built-in region-generation methods.

Omit the top-level section when a built-in dataset bounding-box method is
sufficient.

.. code-block:: yaml

   bbox_model:
     folder: custom_runtime_models
     module: custom_yolo_model
     class: CustomYoloModel

Parameters
^^^^^^^^^^

``folder`` (str)
   Path to the directory containing the custom Python module and any required
   model resources.

``module`` (str)
   Importable Python module name, without the ``.py`` extension.

``class`` (str)
   Name of the custom model class defined in the module.

Custom model interface
~~~~~~~~~~~~~~~~~~~~~~

A custom runtime model should inherit from ``CustomModel`` and implement the
interface required by the processing pipeline:

``prepare_model(self)``
   Loads and prepares the model.

``predict(self, input_data)``
   Runs model inference and returns predictions.

``process(self, data)``
   Converts predictions into the format expected by downstream components.

Example skeleton:

.. code-block:: python

   from runtime_models.custom_model import CustomModel


   class MyCustomModel(CustomModel):
       def prepare_model(self):
           self.model = ...

       def predict(self, input_data):
           predictions = ...
           return predictions

       def process(self, data):
           predictions = self.predict(data)
           processed_predictions = ...
           return processed_predictions

Place the module in the configured ``folder`` relative to the directory from
which Tobii-Pytracker is run. Ensure that the class labels returned by a custom
model match the dataset classes where class matching is required.

Instructions
-------------------

The ``instructions`` section defines the text displayed before and after the
study. Each item is displayed as a separate line; an empty string adds a blank
line.

.. code-block:: yaml

   instructions:
     intro:
       - "Welcome to the study!"
       - ""
       - "In this experiment, you will see a series of images or text samples."
       - "Please look at each stimulus carefully, then select the appropriate option using the buttons below."
       - ""
       - "Click on the window and press SPACE to begin."
     outro:
       - "Thank you for participating in this study!"
       - ""
       - "Your responses and recordings have been saved."
       - "You may now close the window or press ESC to exit."

``intro`` (list[str])
   Lines displayed before the experiment begins.

``outro`` (list[str])
   Lines displayed after the experiment ends.

Eye-tracker configuration file
------------------------------

The eye-tracker configuration file contains the fields required by
``psychopy.iohub.launchHubServer``. It defines the eye-tracker connection,
recorded data streams, and calibration-related settings. See the PsychoPy ioHub
documentation and :ref:`calibration` for the available options.

The default file is available at ``configs/eyetracker_config.yaml`` in the
Tobii-Pytracker repository. It also contains mouse-emulation settings, allowing
the experiment to be tested without a physical eye tracker. In most cases, the
default settings can be used without modification.
