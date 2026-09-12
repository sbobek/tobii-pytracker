Extending Tobii-Pytracker
============================
Tobii-Pytracker is designed to be extensible, allowing users to add new features
and customize existing functionality. This guide provides an overview of how to extend Tobii-Pytracker.

Extending CustomDataset
-----------------------
Create a dataset class by inheriting from ``CustomDataset``. A custom dataset is
responsible for loading its source files and converting them into the sample
format used by the GUI and runtime models.

The constructor receives the active configuration object. Call the parent
constructor so that ``config``, ``dataset_path``, ``classes``, ``data``, and
``calculate_bboxes`` are initialized. The two methods that define the dataset
behavior are:

* ``prepare_data()`` loads the dataset and returns a list of samples.
* ``draw_stimulus(window, sample)`` renders one sample and returns a dictionary
  with optional stimulus metadata, such as bounding boxes.

Each sample should normally contain ``class``, ``data``, and ``id``. The value
of ``data`` may be a path, text, array, or another representation understood by
``draw_stimulus``. Set ``self.classes`` to the available labels, including
``"none"`` when that class is useful for the experiment.

.. code-block:: python

   from tobii_pytracker.datasets import CustomDataset
   from psychopy import visual


   class CsvDataset(CustomDataset):
	   def __init__(self, config, calculate_bboxes=False):
		   super().__init__(config, calculate_bboxes)
		   self.data = self.prepare_data()

	   def prepare_data(self):
		   # Load config.get_dataset_path() and return sample dictionaries.
		   return [{"class": "example", "data": "text to show", "id": "1"}]

	   def draw_stimulus(self, window: visual.Window, sample):
		   visual.TextStim(window, text=str(sample["data"]), pos=(0, 0)).draw()
		   window.flip()
		   return {"bboxes": []}

If bounding boxes are needed, return them in the metadata dictionary using the
format expected by the consuming code. For image datasets, the built-in
implementation returns ``image_bboxes`` containing dictionaries with
``class``, ``conf``, and a centered ``bbox``. Dataset-specific configuration
accessors can be used when the format needs validation or additional options.

Extending CustomModel
---------------------
Custom models are loaded with ``CustomModel(config, dataset)``. Inherit from
``CustomModel`` and implement all three abstract methods:

* ``prepare_model()`` loads weights and initializes the inference backend.
* ``predict(input_data)`` runs one prediction using the prepared model.
* ``process(path)`` converts backend output into the project format.

``CustomModel`` provides ``config``, ``dataset_class_names``, ``is_text``, and a
logger. The dataset class names allow a detector to discard predictions that do
not belong to the current experiment. For image bounding-box models,
``process`` should return tuples of ``(class_name, confidence, bbox)`` where
``bbox`` is ``(x_min, y_min, x_max, y_max)`` in top-left image coordinates.

.. code-block:: python

   from tobii_pytracker.runtime_models import CustomModel


   class ExampleModel(CustomModel):
	   def prepare_model(self):
		   self.model = load_my_model(self.config)

	   def predict(self, input_data):
		   return self.model(input_data)

	   def process(self, path):
		   predictions = self.predict(path)
		   return [
			   (label, float(score), (x_min, y_min, x_max, y_max))
			   for label, score, x_min, y_min, x_max, y_max in predictions
			   if label in self.dataset_class_names
		   ]

The model should do backend-specific parsing in ``process`` rather than in the
dataset. The built-in image dataset rescales the returned box to the configured
area of interest and converts it to centered coordinates. Keep model loading in
``prepare_model`` so initialization failures are reported when the model is
constructed, not during a later trial.

Extending BaseAnalyzer
----------------------
Analyzers operate on flattened pandas data. Inherit from ``BaseAnalyzer`` and
call ``super().__init__(output_folder)``. The required public methods are:

* ``analyze(background_data, ...)`` computes metrics and assigns the resulting
  DataFrame to ``self.results``.
* ``plot_analysis(...)`` visualizes the analysis, optionally saving a figure.

``save_results()`` is already implemented by ``BaseAnalyzer`` and writes
``self.results`` as JSON. An analyzer can override it when it needs additional
formats or nested-data handling, as ``VoiceTranscription`` does.

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from tobii_pytracker.analyze.models import BaseAnalyzer


   class DwellTimeAnalyzer(BaseAnalyzer):
	   def analyze(self, background_data: pd.DataFrame, per="slide"):
		   results = (background_data
					  .groupby(["set_name", "slide_index"])
					  .agg(samples=("system_time", "count"))
					  .reset_index())
		   self.results = results
		   return results

	   def plot_analysis(self, background_data, screenshot_path, show=True,
						 save_path=None, **kwargs):
		   # Create a matplotlib visualization and optionally save it.
		   pass

Use the column names produced by ``DataLoader`` or validate required columns
at the beginning of ``analyze``. Grouping should follow the analyzer's chosen
``per`` convention (usually ``global``, ``set``, or ``slide``), and plotting
should accept a filtered subset when a screenshot represents one slide. Keep
analysis independent of file loading where possible so it can be reused with
CSV data, notebooks, and other analyzers.

