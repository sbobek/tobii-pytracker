.. _bbox_analyzers:

Raw Data-Gaze Alignment Analysis
==============

Bounding-box analyzers align raw gaze samples with modality-specific regions of
interest (ROIs). Each raw gaze point is assigned to the bounding box or boxes
that contain its ``(avg_gaze_x, avg_gaze_y)`` coordinates. The resulting table
can then be used to identify visited regions, count gaze hits, compare alternative
bounding-box generators, and visualize the alignment.


Typical workflow
----------------

A typical analysis consists of four steps:

#. Load the experiment configuration and data.
#. Select gaze samples for one run and one slide using both ``set_name`` and
   ``slide_index``.
#. Run the analyzer to align gaze points with the bounding boxes stored for that
   sample.
#. Plot and, when supported by the analyzer, evaluate the aligned results.

Loading and selecting raw gaze samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DataLoader.get_all_data(flatten=True)`` returns gaze-level rows. Before
analysis, normalize the alignment keys and select exactly one stimulus instance.
Using both keys is important because ``slide_index`` values can repeat across
experiment runs.

.. code-block:: python

   import pandas as pd
   from tobii_pytracker.analyze.data_loader import DataLoader
   from tobii_pytracker.configs.custom_config import CustomConfig

   config = CustomConfig("./configs/config.yaml")
   loader = DataLoader(config, root="./")
   gaze = loader.get_all_data(flatten=True)

   gaze["set_name"] = gaze["set_name"].astype(str)
   gaze["slide_index"] = pd.to_numeric(
       gaze["slide_index"], errors="coerce"
   ).astype("Int64")

   selected_gaze = gaze[
       (gaze["set_name"] == set_name)
       & (gaze["slide_index"] == slide_index)
   ].copy()

The selected rows should contain the raw gaze coordinates, including
``avg_gaze_x`` and ``avg_gaze_y``, as well as the same ``set_name`` and
``slide_index`` values as the stimulus row that contains ``objects_bboxes``.
An empty selection means that no gaze-to-ROI alignment can be performed for that
stimulus.

Alignment model
~~~~~~~~~~~~~~~

For each selected gaze sample, the analyzer tests whether the gaze coordinates
fall inside each modality-specific bounding box. The operation produces scored
bounding boxes that retain the ROI geometry and summarize the gaze points mapped
to each ROI. Overlapping boxes can contain the same gaze sample, so the sum of
per-box hits may be higher than the number of unique gaze points.

For batch processing, first restrict raw stimulus rows to runs with gaze and then
join the two sources by ``set_name`` and ``slide_index``:

.. code-block:: python

   gaze_counts = (
       gaze.groupby(["set_name", "slide_index"])
           .size().rename("gaze_count").reset_index()
   )
   candidates = raw_data.merge(
       gaze_counts, on=["set_name", "slide_index"], how="inner"
   )
   candidates = candidates[candidates["gaze_count"] > 0]

Text bounding boxes
-------------------

``BBoxTextAnalyzer`` aligns gaze samples with word-level bounding boxes. The
shortest usage is:

.. code-block:: python

   from pathlib import Path
   from tobii_pytracker.analyze.models import BBoxTextAnalyzer

   analyzer = BBoxTextAnalyzer(
       output_folder=Path("analysis_outputs/text"),
       config=config,
   )
   results = analyzer.analyze(selected_gaze)
   analyzer.plot_analysis(results)

.. figure:: https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/word-bbox-text.png
   :alt: Gaze samples aligned with word-level text bounding boxes
   :align: center
   :width: 90%

   Example text analysis. Red boxes indicate gaze-visited words, blue boxes
   indicate unvisited words, gray points show gaze samples, and the ``n=`` label
   above a word reports the mapped sample count.

Interpretation
~~~~~~~~~~~~~~

The visualization makes reading coverage explicit. In the example, all displayed
words are outlined in red and therefore received at least one mapped gaze sample.
The labels show that different words accumulated different numbers of samples.
A larger count means that more raw samples fell inside the word box; it should not
be interpreted as a fixation count unless fixation detection was performed
separately before the bounding-box analysis.

Time-series bounding boxes
--------------------------

``BBoxTimeSeriesAnalyzer`` uses bounding boxes generated around sections or
points of a plotted time series.

.. code-block:: python

   from pathlib import Path
   from tobii_pytracker.analyze.models import BBoxTimeSeriesAnalyzer

   analyzer = BBoxTimeSeriesAnalyzer(
       output_folder=Path("analysis_outputs/time_series"),
       config=config,
   )
   results = analyzer.analyze(selected_gaze)
   analyzer.plot_analysis(results)

.. figure:: https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/point-bbox-time-series.png
   :alt: Raw gaze samples aligned with time-series bounding boxes
   :align: center
   :width: 90%

   Example time-series analysis. Small boxes follow the plotted series, raw gaze
   samples form the dark point trace, and red outlines identify boxes visited by
   gaze.

Interpretation
~~~~~~~~~~~~~~

Visited boxes identify the parts of the series that spatially overlap the gaze
trajectory. In the example, visited boxes cluster around the descending lower-left
segment and the trough, while boxes on the upper and right-hand portions remain
unvisited. This supports statements about spatial attention to plotted regions,
but not about causal interpretation of the signal or cognitive processing.

Image bounding boxes
--------------------

``BBoxImagesAnalyzer`` aligns gaze data with ROIs generated for an image, for
example grid, superpixel, or saliency-derived regions. The notebook example also
shows how to evaluate and plot one generated bounding-box set.

.. code-block:: python

   from tobii_pytracker.analyze.models import BBoxImagesAnalyzer

   analyzer = BBoxImagesAnalyzer(
       output_folder="analysis_outputs/images",
       config=config,
   )
   scores = analyzer.analyze(background_data=background_data)
   evaluation = analyzer.evaluate(scores)

   analyzer.plot_analysis(
       scored_bboxes=scores,
       gaze_data=selected_gaze,
       screenshot_path=screenshot_path,
       set_name=set_name,
       slide_index=slide_index,
       top_k=25,
       min_hits=1,
   )

Here, ``background_data`` combines the stimulus row, including
``objects_bboxes``, with each selected ``avg_gaze_x`` and ``avg_gaze_y`` pair.
The example stores image regions under the ``image_bboxes`` key:

.. code-block:: python

   method_raw.at[0, "objects_bboxes"] = {
       "image_bboxes": method_bboxes
   }

.. figure:: https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/bbox_attention_example.png
   :alt: Gaze samples and hit counts over grid image bounding boxes
   :align: center
   :width: 75%

   Example grid analysis. Black rectangles define image ROIs, magenta points show
   the gaze path, and the labels attached to the rectangles report the mapped hit
   counts.

Interpretation
~~~~~~~~~~~~~~

The image example shows unequal attention across the grid. Regions crossed more
often by the displayed gaze path have larger labels, while regions with fewer
mapped samples have smaller labels. Because the grid boxes are non-overlapping,
counts can be compared directly within this partition. For overlapping detectors,
use unique-hit coverage together with the raw per-box counts.

Evaluation measures
-------------------

The example notebooks compare generated bounding-box sets with the following
fields:

``total_gaze_points``
   Number of raw gaze samples available for the analyzed stimulus.

``bbox_hit_count``
   Sum of gaze-to-box assignments. This can exceed ``total_gaze_points`` when
   boxes overlap.

``unique_gaze_hit_count``
   Number of distinct gaze samples assigned to at least one box.

``coverage_by_bboxes``
   Fraction of available gaze samples covered by at least one bounding box.
   Higher values indicate that the ROI set captures more of the observed gaze
   trajectory.

``overlap_factor``
   Degree to which gaze assignments are duplicated across overlapping boxes.
   Interpret high coverage together with this field, because high coverage caused
   by many overlapping regions may be less selective.

``attended_bboxes``
   Number of boxes receiving at least one gaze hit.

``attended_bbox_ratio``
   Fraction of generated boxes that were visited. A low value can indicate a
   sparse attention pattern, but it can also indicate that the generator produced
   many irrelevant or overly fine-grained boxes.

When comparing bounding-box methods, report at least unique coverage, overlap,
and the attended-box ratio together. No single measure determines the best
partition: dense grids, semantic regions, and fine-grained point boxes represent
different analysis targets.

Batch analysis
--------------

The same analyzer can be applied to every slide containing gaze data:

.. code-block:: python

   rows = []
   for slide_index in gaze[gaze["set_name"] == set_name]["slide_index"].unique():
       selected = gaze[
           (gaze["set_name"] == set_name)
           & (gaze["slide_index"] == slide_index)
       ].copy()
       scores = analyzer.analyze(selected)
       analyzer.plot_analysis(scores)

For image analyses or comparisons of alternative generated boxes, append the
single-sample output of ``evaluate`` to ``rows`` and aggregate only after all
samples have been processed. Keep per-sample results as well as aggregated means,
because gaze volume and generated box counts can differ between stimuli.