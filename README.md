[![PyPI](https://img.shields.io/pypi/v/tobii-pytracker)](https://pypi.org/project/tobii-pytracker/)  ![License](https://img.shields.io/github/license/sbobek/tobii-pytracker)
 ![PyPI - Downloads](https://img.shields.io/pypi/dm/tobii-pytracker) [![Documentation Status](https://readthedocs.org/projects/tobii-pytracker/badge/?version=latest)](https://tobii-pytracker.readthedocs.io/en/latest/?badge=latest)
# Toolkit for AI-enhanced Eye-tracking data collection


A Python framework for conducting **eyetracking-based experiments** on **perception** and **reasoning** in machine learning (ML) tasks, as well as for **data enrichment**.


The framework integrates multiple data modalities commonly used as inputs for ML models and provides a platform for **rapid experiment design and execution**. It combines the **PsychoPy** platform for basic user input and **eye-tracking hardware calibration (and emulation)**, **state-of-the-art AI models** for data processing and analysis, and a **custom, extensible architecture** built on top of `tobii-pytracker`.


- **Non-intrusive, multimodal data labeling**, including:
  - Classical label assignment  
  - ML-guided object annotations with **free speech input**, which is automatically transcribed into text and aligned with the regions the user refers to (based on **eye gaze area-of-interest detection**)

- **Analytical platform** for experiment results, offering:
  - Classical metrics such as **saccades**, **fixations**, and **heatmaps**  
  - Advanced methods for **analysis and visualization**, including **process mining scanpath analysis** and **concept detection based on eye-gaze clusters**



![Alt text](https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/tobii-pytracker.svg)

## Table of Contents
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Datasets](#Datasets)
- [Custom Model](#CustomModel)
- [Configuration](#Configuration)

## Requirements

- Python 3.10.x - _(Python 3.10 version is __crucial__, the script won't work with any other version)_
- Pip
- [Tobii Pro Eye Tracker Manager](https://connect.tobii.com/s/etm-downloads?language=en_US) (only optional, for some extra features like firmware updates, etc.)

## Installation

1. Clone or download this repository to your local machine.

    ```sh
   conda create --name pytracker-env python=3.10
   conda activate pytracker-env
   git clone https://github.com/sbobek/tobii-pytracker.git
   cd tobii-pytracker
   pip install .
   ```

   Install psychopy, with no-deps, to keep the installation simple and lightweight.
   Note that we need psychopy in a version at least 2024.1.4
   
   ```sh
   pip install "psychopy>=2024.1.4,<2025.1.0" --no-deps
   ```


## Usage

1. To run the script, use the following command (make sure you have activated virtual environment):

    ```sh
    tobii-pytracker
    ```

<sup>*Make sure to connect eyetracker to your device before running the script with `--enable_eyetracker=True`, otherwise it will fail.</sup>

2. Additionally you can specify a few arguments:
- `--config_file` - Path to YAML script config file _(default: configs/config.yaml)_
- `--eyetracker_config_file` - Path to YAML eyetracker config file _(default: configs/eyetracker_config.yaml)_
- `--enable_eyetracker` - Launch script with launchHubServer (requires connected eyetracker) _(default: False)_
- `--enable_voice` - Start voice recording for Think-Aloud Protocol _(default: False)_
- `--raw_data` - Record full Tobii raw samples instead of filtered gaze positions _(default: False)_
- `--disable_psychopy` - Run headless (no GUI) and continuously record gaze + voice until stopped _(default: False)_
- `--loop_count` - Number of stimuli to display before exit _(default: 10)_
- `--log_level` - Logger level ("info", "debug", "warning", "error", "critical") _(default: info)_

For more information and usage you can run `tobii-pytracker --help`

## Datasets

Different datasets can be specified in the config file that will be processed and displayed in the PsychoPy window. They must be downloaded locally on the device where the script is run and follow these structure rules:

1. __Image datasets__ *(.png, .jpg, .jpeg, .bmp, .gif)*:
    ```.
    ├── dataset
    │   ├── category1
    │   │   ├──file1.jpg
    │   │   ├──file2.jpg
    │   │   └──...
    │   ├── category2
    │   │   ├──file1.jpg
    │   │   ├──file2.jpg
    │   │   └──...
    │   └── ...
    └── ...
    ```

    > NOTE: GUI will be created accordingly to this structure and class labels will be extracted from subfolder names within the dataset. If using custom model, remember to make them match class names.

2. __Text datasets__ *(.csv)*:
    ```.
    ├── dataset.csv
    └── ...
    ```

    > NOTE: The first line in .csv file __MUST__ be a header. The GUI will take unique label values from a column with a name specified in config file.

## Custom Model

The custom model for detecting bounding boxes fro possible area of interest is optional. 
Every dataset class has its own bounding-box detection method (words for text, bins dor time-series, superpixels for images).
However, you might want to change it with your custom implementation.
It is desired, when you want to get bounding box from external model at the runtime, because for instance screenshots have format or size which is not used by you detection model and cannot be use post-hoc.

The `CustomModel` class (located in `runtime_models/custom_model.py`) serves as an abstract base class. It defines the essential methods that any custom model implementation must provide. By inheriting from `CustomModel` and implementing these methods, your custom model can seamlessly interact with the rest of the toolkit:

- `prepare_model(self)`: Load and prepare model for prediction
- `predict(self, input_data)`: Run prediction with the loaded model
- `process(self, path)`: Post-process the prediction and return formatted results

### Creating Your Own Model Modules
To create your own model module:

1. Create a new Python file within the specified directory (e.g., `custom_runtime_models/my_custom_model.py`).

2. Import the CustomModel class:
```python
from runtime_models.custom_model import CustomModel
```

3. Define a new class that inherits from `CustomModel`:

```python
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
```

Keep in mind that main processing pipeline uses the `process()` method to save processed model predictions.

## Configuration

The script requires two configuration files in YAML format: one for general settings and another for eye tracker settings.
By default, the configuration files are located in `configs`` directory.

### General Configuration (config.yaml)

The general configuration file should include the following fields:

```yaml
dataset:
  image:
    bbox_model:  superpixel #grid | superpixel | saliency
    path: datasets/vehicles

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
```

> NOTE: If using a dataset with __text__ data, the path should be specified for a file in a `.csv` format, as well as additional `text` field:

```yaml
dataset:
  text:
    label_column_name: sentiment
    text_column_name: selected_text
    color: white
    background_color: black
    bbox_model: word # word|line|sentence
    path: datasets/twitter/tweets.csv
```


> NOTE: If using a dataset with __time_series__ data, the path should be specified for a file in a `.csv` format.

```yaml
dataset:
  time_series:
    label_column_name: class
    bbox_model:  sample #|window
    path: datasets/ecg/ecg200.csv
```
### Eye Tracker Configuration (eyetracker_config.yaml)

The eye tracker configuration file should include the fields required by `launchHubServer` from the `psychopy.iohub` module. Refer to the [PsychoPy documentation](https://www.psychopy.org/api/iohub/device/eyetracker_interface/Tobii_Implementation_Notes.html#default-device-settings) for more details on the specific settings.
