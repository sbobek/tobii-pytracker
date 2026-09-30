Data Loading and Visualization
================================

This short example loads recorded data and plots gaze over a stimulus
screenshot. The workflow is modality agnostic: the image, text, and
time-series notebooks use the same ``DataLoader`` calls. Only the selected
example directory, its configuration, and the recorded dataset differ.

Open a Python session or notebook with the project installed from the
repository's ``examples/images``, ``examples/text``, or
``examples/time-series`` directory. Run the following code there so the
modality's configuration, ``output`` folder, and screenshots are found.

Construct a ``DataLoader`` with the example configuration. The loader finds
recorded subjects under the configured output folder:

.. code-block:: python

	from tobii_pytracker.analyze.data_loader import DataLoader
	from tobii_pytracker.configs.custom_config import CustomConfig

	config = CustomConfig("./configs/config.yaml")
	loader = DataLoader(config, root="./")

	subjects = loader.get_subjects()
	subjects[:5]

Inspect a subject's recorded data, or flatten one slide's gaze samples into a
table:

.. code-block:: python

	subject = subjects[0]
	subject_data = loader.get_subject_data(subject)
	slide_data = loader.get_slide_data(subject, 0, flatten=True)
	slide_data.head()

Plot that slide's gaze points over its screenshot:

.. code-block:: python

	loader.plot_gaze(subject, 0, gradient=True, flip_y=True)

The subject name and slide index select the recording to inspect. The
``gradient`` option colors gaze points by their order in the recording.
``plot_gaze`` works for any modality when the selected record has a valid
``screenshot_file``; it overlays gaze on the screenshot for that stimulus.
