End-to-End Examples
=====================

Basic smoke-test examples are provided in the `examples <https://github.com/mszac/tobii-pytracker-demo/tree/main/examples>`_ directory.
See the :ref:`Data Analyzers page <data_analyzers>` for more details.


.. _end_to_end_examples_section:

Complete end-to-end tutorials
--------------------

Runnable examples are maintained in the
`tobii-pytracker-demo repository
<https://github.com/mszac/tobii-pytracker-demo>`_.
They demonstrate complete workflows from experiment configuration and data
collection to validation and post-hoc analysis.

The examples are stored in the
`examples directory
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples>`_.


UX A/B Experiment
-----------------

The
`UX AB example
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/ux_ab_demo>`_
is a more complete text-search experiment comparing early and late placement
of target information.

Unlike the smoke tests, this example demonstrates an experimental analysis
workflow. It derives trial-level and condition-level results, including
response accuracy, whether the target was viewed, time to first fixation
(``TTFF``), and dwell time. The analysis uses ``DataLoader`` and
``FixationAnalyzer`` from the installed Tobii-PyTracker package.

The generated results include tabular summaries, a hypothesis summary, plots
comparing experimental conditions, and an example gaze visualization.

See:

* the
  `UX A/B README
  <https://github.com/mszac/tobii-pytracker-demo/blob/main/examples/ux_ab_demo/README.md>`_
  for execution and analysis instructions;
* the
  `experiment description
  <https://github.com/mszac/tobii-pytracker-demo/blob/main/examples/ux_ab_demo/EXPERIMENT.md>`_
  for the experimental design and interpretation;
* the
  `analysis directory
  <https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/ux_ab_demo/analysis>`_
  for the analysis implementation.

Time Series Noise Demo
----------------------

The `time-series noise demo
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/timeseries_noise_demo>`_
demonstrates an experiment in which participants inspect time-series signals
under different noise conditions.

The example illustrates how Tobii-PyTracker can support the study of visual
exploration during signal-analysis tasks. It combines experiment
configuration, presentation of time-series stimuli, gaze recording, and
post-hoc comparison of viewing behaviour across experimental conditions.

The repository contains the experiment assets and analysis materials needed
to examine how gaze is distributed over the presented signals. Consult the
README and analysis files in the corresponding demo directory for its exact
configuration, execution procedure, calculated measures, and interpretation.

Text Search Demo
----------------

The `text-search demo
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/text_search_demo>`_
demonstrates a visual-search task based on textual stimuli.

Participants inspect text content and respond according to the information
presented in each item. The example shows how task responses can be combined
with gaze data to study whether and when relevant information was observed.

The accompanying analysis materials demonstrate measures commonly used in
text-search studies, such as response correctness, fixation-based target
detection, time to first fixation, and viewing time within relevant text
regions.

This demo provides the conceptual basis for the active
`UX AB example
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/ux_ab_demo>`_,
which compares early and late placement of target information. Refer to the
README, experiment description, and analysis files in the corresponding
directory for the complete workflow.

Image Semantic Demo
-------------------

The `image-semantic demo
<https://github.com/mszac/tobii-pytracker-demo/tree/main/examples/image_semantic_demo>`_
demonstrates gaze collection and analysis for images representing different
semantic categories.

The example combines image presentation, participant responses, gaze
recording, and spatial analysis. Its materials illustrate how heatmaps,
fixations, saccades, entropy, and related gaze summaries can be used to compare
visual exploration across image stimuli.

The interpretation should remain descriptive unless the experiment is
extended with an appropriate number of participants, stimuli, repetitions,
and statistical controls. Consult the corresponding demo directory for its
stimuli, configuration, execution instructions, and analysis implementation.



Testing Instructions
--------------------

Platform-specific native testing instructions are available in:

* :ref:`Windows testing documentation <windows_installation>`;
* :ref:`Linux testing documentation <linux_installation>`.

The examples are designed to use Tobii-PyTracker from a the official repository.
Follow the directory
layout and environment instructions in the selected example before running
its commands.

See the :ref:`Data Analyzers page <data_analyzers>` for descriptions
of the analyzers and interpretation of their outputs.