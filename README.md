# Toolkit for AI-enhanced Eye-tracking data collection

This toolkit allows to conduct scientific researches of human classification visual aspect by gathering human gaze data on 2 different types of data. It's modality is defined for images and text. The graphical interface uses PsychoPy library and is created dynamically from a dataset with a strict structure. Main config file allows to override values for different PsychoPy research environments which grants more control over received results. Processing pipeline can be extended with a custom model for additional prediction data to compare with human classification. The data of human eye gaze, as well as average pupil size, is collected with Tobbii eyetracker and saved to a file with the rest of results.

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
- [Tobii Pro Eye Tracker Manager](https://connect.tobii.com/s/etm-downloads?language=en_US)

## Installation

1. Clone or download this repository to your local machine.

    ```sh
   git clone https://github.com/frieZZerr/UJ-AI-Workshops.git
   cd UJ-AI-Workshops
   ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required Python packages:

    ```sh
    ./install_requirements.sh
    ```

## Usage

1. To run the script, use the following command:

    ```sh
    python main.py
    ```

<sup>*Make sure to connect eyetracker to your device before running the script with `--enable_eyetracker=True`, otherwise it will fail.</sup>

2. Additionally you can specify a few arguments:
    - `--config_file` - Path to YAML script config file - _(default = configs/config/config.yaml)_
    - `--eyetracker_config_file` - Path to YAML eyetracker config file - _(default = configs/config/eyetracker_config.yaml)_
    - `--enable_eyetracker` - Launch script with launchHubServer (needs connected eyetracker if set to True) - _(default = False)_
    - `--enable_model` - Extend processing for custom YOLO model predictions (only for images) - _(default = False)_
    - `--loop_count` - Number of times that different data will be displayed before the script exits - _(default = 10)_
    - `--log_level` - Main logger level ("info", "debug", "warning", "error", "critical") - _(default = info)_

For more information and usage you can run `python main.py --help`

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

The `CustomModel` class (located in `models/custom_model.py`) serves as an abstract base class. It defines the essential methods that any custom model implementation must provide. By inheriting from `CustomModel` and implementing these methods, your custom model can seamlessly interact with the rest of the toolkit:

- `prepare_model(self)`: Load and prepare model for prediction
- `predict(self, input_data)`: Run prediction with the loaded model
- `process(self, path)`: Post-process the prediction and return formatted results

### Creating Your Own Model Modules
To create your own model module:

1. Create a new Python file within the specified directory (e.g., `models/my_custom_model.py`).

2. Import the CustomModel class:
```python
from models.custom_model import CustomModel
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

### General Configuration (config.yaml)

The general configuration file should include the following fields:

```yaml
dataset:
  path: path/to/dataset
display:
  monitor:
    name: monitor_name
    resolution:
      - width
      - height
    width: monitor_width
    distance: distance_from_monitor
  gui:
    button:
      size:
        - size_x
        - size_y
      color: color
      text:
        color: text_color
        size: text_size
    aoe:
      - size_x
      - size_y
output:
  folder: folder
  file: file
model:
  folder: folder
  module: module
  class: class
  filename: filename
```

> NOTE: If using a dataset with __text__ data, the path should be specified for a file in a `.csv` format, as well as additional `text` field:

```yaml
dataset:
  path: path/to/file.csv
  text:
    label_column_name: label_column_name
    text_column_name: text_column_name
```

### Eye Tracker Configuration (eyetracker_config.yaml)

The eye tracker configuration file should include the fields required by `launchHubServer` from the `psychopy.iohub` module. Refer to the [PsychoPy documentation](https://www.psychopy.org/api/iohub/device/eyetracker_interface/Tobii_Implementation_Notes.html#default-device-settings) for more details on the specific settings.
