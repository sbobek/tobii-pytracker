import os
import numpy as np
from abc import ABC, abstractmethod
from utils.custom_logger import CustomLogger


class CustomModel(ABC):
    def __init__(self, config, dataset):
        self.logger = CustomLogger("debug", __name__).logger
        self.config = config
        self.dataset_class_names = dataset.get_classes()
        self.is_text = dataset.is_text

        try:
            self.prepare_model()
        except Exception as e:
            self.logger.error(f"There was an error while loading the model: {e}")
            raise RuntimeError(f"Error while loading custom model: {e}")

    @abstractmethod
    def prepare_model(self):
        """Load and prepare model for prediction"""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Run prediction with the loaded model"""
        pass

    @abstractmethod
    def process(self, path):
        """Post-process the prediction and return formatted results"""
        pass
