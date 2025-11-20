import os
import numpy as np
from ultralytics import YOLO
from tobii_pytracker.runtime_models.custom_model import CustomModel


class CustomYoloModel(CustomModel):
    def prepare_model(self):
        self.logger.debug("Preparing YOLO model...")
        model_folder = self.config.get_model_config()["folder"]
        model_filename = "yolov8n.pt"
        model_path = os.path.join(model_folder, model_filename)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.model = YOLO(model_path)

        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.predict(dummy_image)
        self.logger.debug("Model preparation done.")

    def predict(self, input_data):
        try:
            return self.model(input_data)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return []

    def process(self, path):
        processed = []
        try:
            prediction = self.predict(path)
            for box in prediction[0].boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                if class_name in self.dataset_class_names:
                    confidence = round(box.conf[0].item(), 2)
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    area_x, area_y = self.config.get_area_of_interest_size()
                    bounding_box = (
                        int(round(x_min - area_x / 2)),
                        int(round(y_min - area_y / 2)),
                        int(round(x_max - area_x / 2)),
                        int(round(y_max - area_y / 2)),
                    )

                    processed.append((class_name, confidence, bounding_box))
                    self.logger.debug(
                        f"Detected object: class_name={class_name}, confidence={confidence:.2f}, "
                        f"bounding_box=({bounding_box[0]}, {bounding_box[1]}, {bounding_box[2]}, {bounding_box[3]})"
                    )
        except Exception as e:
            self.logger.error(f"Error during YOLO processing: {e}")

        return processed
