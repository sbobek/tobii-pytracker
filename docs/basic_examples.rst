.. _basic_examples:

Basic Examples
==============

The basic examples demonstrate how to configure Tobii-PyTracker, collect
multimodal experimental data, load previously recorded sessions, and apply the
available data analyzers. Examples are provided for image, time-series, and
text stimuli.

Before running an example, install Tobii-PyTracker as described in the
:ref:`Installation Instructions <installation>` and adjust the configuration
files in the example's ``configs`` directory to match the local environment
and eye-tracker settings.

Each example already contains recorded output that can be used to run the
analysis without collecting new data. A new recording is only required when
testing the data-collection workflow or a modified experiment configuration.

Running an Example
------------------

For example, to collect gaze and voice data for the image-classification
experiment using the mouse-emulated eye tracker, open the image example
directory and run:

.. code-block:: console

   cd examples/images
   tobii-pytracker --eyetracker_config_file ./configs/eyetracker_config.yaml --enable_eyetracker --enable_voice

The mouse-emulated eye tracker makes it possible to test the complete
collection workflow without connecting a physical eye tracker. 
Note that the directory contains already recorded output, so running the above command is not required to execute the analysis.

After collection, open the Jupyter notebook provided in the same example
directory and run its cells. The notebook loads sessions from the ``output``
directory, executes the relevant analyzers, and presents the resulting tables
and visualizations.

See the :ref:`Data Analyzers page <data_analyzers>` for descriptions
of the analyzers, their parameters and analysis scopes, and guidance on
interpreting their output.

Image Data Example
------------------

The ``examples/images`` directory contains an image-classification experiment.
During each trial, an image is presented and the participant assigns it to one
of the available categories.

The example includes data collected with a Tobii Pro Spark eye tracker,
together with the corresponding experiment configuration, screenshots, gaze
samples, participant responses, and analysis notebook. It demonstrates the
complete workflow for:

* presenting image stimuli and collecting classification responses;
* recording gaze coordinates and pupil measurements;
* associating gaze samples with individual experiment items;
* summarizing gaze globally, per experiment outcome, or per image;
* detecting fixations and saccades;
* constructing scanpaths;
* generating heatmaps and focus maps;
* measuring gaze entropy and spatial coverage;
* clustering gaze observations into spatial groups;
* assigning gaze observations to image regions.

In the output data, ``set_name`` normally identifies a participant or one
recorded experiment outcome. The ``slide_index`` identifies the image's
position within that experiment. Consequently, analysis performed with
``per="slide"`` produces separate results for every ``set_name`` and
``slide_index`` combination.

The experiment contains files with recorded voice while an image is displayed. The
included voice-transcription workflow demonstrates how ``VoiceTranscription``
uses Whisper to create timestamped transcript segments and align them with
gaze samples from the same item. This makes it possible to inspect where a
participant was looking while a particular statement was spoken.

The image example also demonstrates attention analysis based on automatically
generated or predefined image regions. This can be used to determine which
regions received gaze, estimate gaze coverage, and compare attended and
unattended regions.

Refer to the analysis notebook in ``examples/images`` for the executable
workflow. Detailed explanations of the resulting fixation, saccade, entropy,
clustering, scanpath, region-attention, and voice-alignment outputs are
provided on the :ref:`Data Analyzers page <data_analyzers>`.

Time-Series Data Example
------------------------

The ``examples/time_series`` directory demonstrates the use of Tobii-PyTracker
with time-series stimuli. It is intended for experiments in which participants
inspect signals, plots, or other sequential numerical data presented as
visual items.

The example illustrates how to:

* configure an experiment containing time-series visualizations;
* collect gaze observations for each presented signal;
* associate gaze samples with the corresponding experiment outcome and item;
* inspect the spatial distribution of gaze over individual plots;
* detect periods of stable viewing and transitions between inspected regions;
* compare visual exploration between time-series items.

The example uses the mouse-emulated eye tracker, allowing a controlled gaze-like
trajectory to be reproduced without a physical device. 

For time-series experiments, the analysis remains spatial: the gaze
coordinates describe where the participant looked on the rendered plot. They
should not be confused with the numerical values or timestamps represented by
the plotted signal. Relating gaze to a particular signal segment therefore
requires a known mapping between screenshot coordinates and the plot axes or
predefined regions of interest.

The included notebook loads the recorded output and demonstrates the relevant
gaze summaries and event-based analyses. See the
:ref:`Data Analyzers page <data_analyzers>` for details of heatmaps,
fixations, saccades, entropy, clustering, and scanpaths.

Text Data Example
-----------------

The ``examples/text`` directory demonstrates a text-based experiment using the
mouse-emulated eye tracker. It includes an experiment configuration, example
text stimuli, previously collected output, and a notebook for validating and
analyzing the recorded data.

The example demonstrates how to:

* present text stimuli and collect participant responses;
* record gaze-like observations while textual content is displayed;
* associate each observation with the corresponding text item;
* examine which parts of the displayed text received attention;
* identify stable viewing locations and transitions between them;
* combine gaze-derived measures with task responses.

Text stimuli are stored as experiment items and identified in the output by
``set_name`` and ``slide_index``. Slide-level analysis can therefore be used
to inspect each text item separately, while set-level analysis summarizes the
complete recorded experiment outcome.

When text regions or bounding boxes are defined, region-attention analysis can
be used to determine whether gaze entered a target passage or other area of
interest. Fixation-derived measures can additionally support analyses such as
time to first fixation and dwell time, provided that the relevant textual
regions and timing rules are defined by the experiment.

Because this example uses mouse-emulated gaze, its included output is intended
primarily to demonstrate data flow, item alignment, and analysis execution.
Research conclusions about reading or visual cognition require observations
collected with an appropriate eye tracker and a suitable experimental design.

Open the notebook in ``examples/text`` to run the complete analysis workflow.
See the :ref:`Data Analyzers page <data_analyzers>` for detailed
descriptions and interpretation guidance.