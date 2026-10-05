.. _usage-patterns:

Usage Patterns
==============

Tobii-PyTracker supports three complementary usage layers: experimental data
acquisition, in-depth analysis, and integration with external machine-learning
workflows. The layers separate experiment execution from subsequent analysis
and dataset use, while allowing the same collected records to support both
paths.

.. figure:: https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/tobii-pytracker-workflow.svg
   :alt: Experimental, analytical, and external machine-learning usage layers in Tobii-PyTracker.
   :align: center
   :width: 100%

   Tobii-PyTracker usage patterns. The Experimental Layer collects multimodal
   records, which can be examined in the Analytical Layer or exported as CSV
   files for downstream machine-learning tasks.

Experimental Layer
------------------

The Experimental Layer is responsible for running the experiment and gathering
multimodal data. It can acquire data from supported physical hardware or from
emulated hardware, which makes it possible to develop, demonstrate, and test an
experiment without requiring the target device at every stage.

A typical experimental workflow includes calibration, presentation of
instructions, repeated presentation of configured stimuli, collection of
participant responses, and storage of the resulting records. The repeated part
of the experiment is controlled by the experiment configuration and can contain
additional images, text, or other study material.

The Experimental Layer produces synchronized records that can subsequently be
used in two ways:

* as input to the built-in Analytical Layer; or
* as CSV-formatted data for external processing and machine-learning tasks.

For more info on the Experimental Layer, see the `Basic examples <basic_examples>`_ section.

Analytical Layer
----------------

The Analytical Layer supports detailed examination of the material gathered
during the experimental phase. It combines standard eye-tracking approaches
with custom analytical models provided for more specialized analyses.

The available workflows can include, among others:

* fixation analysis;
* focus maps;
* scanpath and saccade analysis;
* spatial grid and region-based summaries;
* gaze clustering;
* timing analysis;
* alignment of gaze with speech or other synchronized modalities; and
* custom analyses of relationships between the recorded modalities.

These outputs help transform the recorded experiment streams into interpretable
visualizations, measurements, and multimodal summaries. The examples shown in
the workflow figure are representative rather than exhaustive.
For more information on the available analyzers, see the `Analyzers <data_analyzers>`_ section.

External Machine-Learning Layer
-------------------------------

The External Machine-Learning Layer is not part of Tobii-PyTracker itself.
Instead, Tobii-PyTracker can be used as a data-labeling and dataset-creation
tool that produces rich multimodal records for downstream machine-learning
pipelines.

Experimental data are stored in CSV format, which allows the records to be
loaded and processed by commonly used data-analysis and machine-learning
toolkits. Depending on the experiment design, the exported data can combine
gaze measurements, stimulus information, participant responses, timing, and
other synchronized metadata.

Potential downstream uses include:

* creation of gaze-annotated multimodal datasets;
* human-in-the-loop data-labeling workflows;
* training or evaluation of multimodal models;
* development of vision-language models enhanced with human gaze feedback;
* attention and gaze-prediction tasks; and
* preparation of task-specific datasets and benchmarks.

In this usage pattern, Tobii-PyTracker provides the experimental interface and
the structured data output, while model development, training, and evaluation
remain the responsibility of the external machine-learning environment.

Choosing a Usage Pattern
------------------------

The three layers can be combined according to the research objective:

#. Use the **Experimental Layer** to configure and run a study and to collect
   data from physical or emulated hardware.
#. Use the **Analytical Layer** when the objective is to inspect, visualize, and
   evaluate the experiment using standard eye-tracking techniques or custom
   analytical models.
#. Use the exported **CSV data** in an external machine-learning workflow when
   the objective is dataset construction, model development, or another
   downstream computational task.

The Analytical and External Machine-Learning layers are alternative or
complementary destinations for the same experimental records. A study can use
one path or both, depending on whether the collected data are intended for
scientific interpretation, machine-learning applications, or a combination of
the two.
