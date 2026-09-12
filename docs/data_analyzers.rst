Data Analyzers
===================
Tobii-Pytracker includes several data analyzers to help process and visualize eye-tracking data collected during experiments. These analyzers can be used to extract meaningful insights from raw gaze data.

Available Analyzers
-------------------
1. **HeatmapAnalyzer**: Generates gaze heatmaps and overlays them on screenshots.
2. **FocusMapAnalyzer**: Creates focus maps to visualize areas not looked at.
3. **SaccadeAnalyzer**: Detects and analyzes saccades in gaze data.
4. **FixationAnalyzer**: Identifies and analyzes fixations in gaze data.
5. **EntropyAnalyzer**: Computes gaze distribution entropy and convex hull dispersion.
6. **ClusterAnalyzer**: Groups gaze coordinates using DBSCAN, KMeans, or a custom model.
7. **ScanpathsAnalyzer**: Analyzes sequential transitions between fixations.
8. **VoiceTranscription**: Transcribes slide voice recordings and aligns transcript sentences with gaze samples.


Using Analyzers
----------------- 