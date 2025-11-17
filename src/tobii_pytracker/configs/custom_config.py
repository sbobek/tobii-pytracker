import os
import yaml
class CustomConfig:
    def __init__(self, filename):
        self.config = self.read_config(filename)
        self.dataset_type = None  # set later
        self.dataset_path = None
        self.check_config()

    # -------------------------
    # YAML LOADING
    # -------------------------
    @staticmethod
    def read_config(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file '{filename}' not found.")
        with open(filename, "r") as f:
            return yaml.safe_load(f)

    # -------------------------
    # MASTER CHECK
    # -------------------------
    def check_config(self):
        self.detect_dataset_type()

        if self.dataset_type == "image":
            self.get_image_dataset_config()
        elif self.dataset_type == "text":
            self.get_text_dataset_config()
        elif self.dataset_type == "time_series":
            self.get_time_series_dataset_config()

        self.get_monitor_config()
        self.get_button_config()
        self.get_output_config()
        self.get_instructions_config()


    def get_dataset_path(self):
        return self.dataset_path
    # ------------------------------------------------------
    #  DATASET TYPE DETECTION
    # ------------------------------------------------------
    def detect_dataset_type(self):
        if "dataset" not in self.config:
            raise KeyError("Missing 'dataset' section in configuration file.")

        dataset_section = self.config["dataset"]
        if not isinstance(dataset_section, dict):
            raise ValueError("'dataset' must be a dictionary.")
        
        # required
        if "path" not in dataset_section:
            raise KeyError("Missing required image dataset field: path")
        if not isinstance(dataset_section["path"], str):
            raise ValueError("'path' must be a string for image dataset.")
        
        self.dataset_path = dataset_section["path"]


        valid_types = {"image", "text", "time_series"}
        present = valid_types.intersection(dataset_section.keys())

        if not present:
            raise KeyError("dataset must contain one of: image, text, time_series")

        if len(present) > 1:
            raise ValueError("Only one dataset type (image/text/time_series) can be defined at once.")

        self.dataset_type = present.pop()

    # ------------------------------------------------------
    #  IMAGE DATASET
    # ------------------------------------------------------
    def get_image_dataset_config(self):
        if self.dataset_type != "image":
            return None

        cfg = self.config["dataset"]["image"]

        # optional bbox_model
        if "bbox_model" in cfg:
            allowed = ["grid", "superpixel", "saliency"]
            if cfg["bbox_model"] not in allowed:
                raise ValueError(f"image bbox_model must be one of: {allowed}")

        return cfg

    # ------------------------------------------------------
    #  TEXT DATASET
    # ------------------------------------------------------
    def get_text_dataset_config(self):
        if self.dataset_type != "text":
            return None

        cfg = self.config["dataset"]["text"]

        required = ["label_column_name", "text_column_name"]
        for r in required:
            if r not in cfg:
                raise KeyError(f"Missing required text dataset field: {r}")


        # optional bbox_model
        if "bbox_model" in cfg:
            allowed = ["word", "line", "sentence"]
            if cfg["bbox_model"] not in allowed:
                raise ValueError(f"text bbox_model must be one of: {allowed}")

        return cfg

    # ------------------------------------------------------
    #  TIME-SERIES DATASET
    # ------------------------------------------------------
    def get_time_series_dataset_config(self):
        if self.dataset_type != "time_series":
            return None

        cfg = self.config["dataset"]["time_series"]

        required = ["label_column_name"]
        for r in required:
            if r not in cfg:
                raise KeyError(f"Missing required time-series dataset field: {r}")

        # optional bbox_model
        if "bbox_model" in cfg:
            allowed = ["sample", "window"]
            if cfg["bbox_model"] not in allowed:
                raise ValueError(f"time_series bbox_model must be one of: {allowed}")

        return cfg
    
    def get_monitor_config(self):
        try:
            monitor_config = self.config["display"]["monitor"]
            
            required_fields = ["name", "resolution", "width", "distance"]
            for field in required_fields:
                if field not in monitor_config:
                    raise KeyError(f"Missing required monitor field: {field}")

            resolution = monitor_config["resolution"]
            if not isinstance(resolution, list) or len(resolution) != 2:
                raise ValueError("Monitor resolution must be a list of two integers (e.g., [1920, 1080]).")

            return monitor_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing monitor configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid monitor configuration: {e}")

    def get_button_config(self):
        try:
            button_config = self.config["display"]["gui"]["button"]
            
            if not isinstance(button_config, dict):
                raise ValueError("Button configuration must be a dictionary.")

            required_fields = ["size", "margin", "color"]
            for field in required_fields:
                if field not in button_config:
                    raise KeyError(f"Missing required button field: {field}")

            def validate_text_field(text_config):
                required_text_fields = ["color", "size"]
                for subfield in required_text_fields:
                    if subfield not in text_config:
                        raise KeyError(f"Missing required 'text' subfield: {subfield}")

                    if subfield == "color" and not isinstance(text_config[subfield], str):
                        raise ValueError("'color' must be a string representing a color.")

                    if subfield == "size" and not isinstance(text_config[subfield], int):
                        raise ValueError("'size' must be an integer.")

            validate_text_field(button_config["text"])

            return button_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing button configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid button configuration: {e}")

    def get_fixation_dot_config(self):
        """
        Retrieve and validate fixation dot configuration from YAML display settings.

        Returns
        -------
        dict
            Dictionary containing fixation dot configuration with validated fields:
            {
                "size": <int>,
                "color": <str>
            }

        Raises
        ------
        KeyError
            If the fixation dot section or any required field is missing.
        ValueError
            If the field values have incorrect types or invalid structure.

        Example YAML
        ------------
        display:
        gui:
            fixation_dot:
            size: 10
            color: white
        """
        try:
            fixation_config = self.config["display"]["gui"]["fixation_dot"]

            if not isinstance(fixation_config, dict):
                raise ValueError("Fixation dot configuration must be a dictionary.")

            required_fields = ["size", "color"]
            for field in required_fields:
                if field not in fixation_config:
                    raise KeyError(f"Missing required fixation_dot field: {field}")

            # Validate types
            if not isinstance(fixation_config["size"], int):
                raise ValueError("'size' in fixation_dot must be an integer (e.g., 10).")

            if not isinstance(fixation_config["color"], str):
                raise ValueError("'color' in fixation_dot must be a string (e.g., 'white').")

            return fixation_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing fixation_dot configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid fixation_dot configuration: {e}")


    def get_output_config(self):
        try:
            output_config = self.config["output"]

            if "folder" not in output_config:
                raise KeyError("Missing required output field: 'folder'.")

            return output_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing output configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid output configuration: {e}")

    def get_bbox_model_config(self):
        try:
            model_config = self.config["bbox_model"]

            required_fields = ["folder", "module", "class"]
            for field in required_fields:
                if field not in model_config:
                    raise KeyError(f"Missing required model field: {field}")

            return model_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing model configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid model configuration: {e}")

    def get_area_of_interest_size(self):
        try:
            area_of_interest = self.config["display"]["gui"]["aoe"]

            if not isinstance(area_of_interest, list) or len(area_of_interest) != 2:
                raise ValueError("Area of interest must be a list of two integers (e.g., [750, 750]).")

            return area_of_interest

        except KeyError as e:
            raise KeyError(f"Invalid or missing aoe field: {e}")

    def get_instructions_config(self):
        """
        Retrieve intro and outro instructions from config.
        Returns a dict with keys: 'intro' and 'outro'.
        """
        try:
            if "instructions" not in self.config:
                raise KeyError("Missing required section: 'instructions'.")

            instructions_config = self.config["instructions"]

            required_fields = ["intro", "outro"]
            for field in required_fields:
                if field not in instructions_config:
                    raise KeyError(f"Missing required instructions field: '{field}'.")

                if not isinstance(instructions_config[field], list):
                    raise ValueError(f"Instruction field '{field}' must be a list of text lines.")

            return instructions_config

        except KeyError as e:
            raise KeyError(f"Invalid or missing instructions configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid instructions configuration: {e}")
