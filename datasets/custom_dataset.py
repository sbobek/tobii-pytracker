import os
import random
import pandas as pd

from utils.custom_logger import CustomLogger

class CustomDataset:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.get_dataset_config()['path']
        self.is_text = True if config.get_dataset_text_config() else False

        self.logger = CustomLogger("debug", __name__).logger

        if self.is_text:
            self.data = self.prepare_text()
        else:
            self.data = self.prepare_images()

    def prepare_images(self):
        self.classes = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        self.classes.append('none')  # Additional class

        image_objects = []
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)

            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(root, file)
                        image_objects.append({'class': class_name, 'data': image_path})

        random.shuffle(image_objects)  # Random shuffle

        return image_objects

    def prepare_text(self):
        if self.dataset_path.endswith('.csv'):
            df = pd.read_csv(self.dataset_path, header=0)  # Header must be specified in the first line of data
        else:
            raise ValueError("The dataset must be a CSV file")

        label_column = self.config.get_dataset_text_config()['label_column_name']
        text_column = self.config.get_dataset_text_config()['text_column_name']

        self.classes = [cls.lower() for cls in list(df[label_column].unique())]
        self.classes.append('none')  # Additional class

        df = df.sample(frac=1).reset_index(drop=True)  # Random shuffle

        text_objects = []
        for _, row in df.iterrows():
            text_objects.append({'class': row[label_column].lower(), 'data': row[text_column]})

        return text_objects

    def get_classes(self):
        return self.classes
