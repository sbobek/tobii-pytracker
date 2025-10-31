from .base_analyzer import BaseAnalyzer

class HeatmapAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent

    def analyze_one(self, set_name: str, index: int):
        slide = self.parent.get_slide_data(set_name, index)
        # Perform fixation computation on slide['gaze_data']
        pass

    def analyze_subject(self, set_name: str):
        df = self.parent.get_subject_data(set_name)
        # Loop over slides
        pass

    def analyze_all(self):
        for subject in self.parent.subjects:
            self.analyze_subject(subject)


class FocusMapAnalyzer(HeatmapAnalyzer):
    def __init__(self, parent):
        self.parent = parent

    def analyze_one(self, set_name: str, index: int):
        slide = self.parent.get_slide_data(set_name, index)
        # Perform fixation computation on slide['gaze_data']
        pass

    def analyze_subject(self, set_name: str):
        df = self.parent.get_subject_data(set_name)
        # Loop over slides
        pass

    def analyze_all(self):
        for subject in self.parent.subjects:
            self.analyze_subject(subject)


class FixationAnalyzer:
    def __init__(self, parent):
        self.parent = parent

    def analyze_one(self, set_name: str, index: int):
        slide = self.parent.get_slide_data(set_name, index)
        # Perform fixation computation on slide['gaze_data']
        pass

    def analyze_subject(self, set_name: str):
        df = self.parent.get_subject_data(set_name)
        # Loop over slides
        pass

    def analyze_all(self):
        for subject in self.parent.subjects:
            self.analyze_subject(subject)


class SaccadeAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent

    def analyze_one(self, set_name: str, index: int):
        slide = self.parent.get_slide_data(set_name, index)
        # Perform fixation computation on slide['gaze_data']
        pass

    def analyze_subject(self, set_name: str):
        df = self.parent.get_subject_data(set_name)
        # Loop over slides
        pass

    def analyze_all(self):
        for subject in self.parent.subjects:
            self.analyze_subject(subject)

class EntropyAnalyzer(BaseAnalyzer):
    def __init__(self, parent):
        self.parent = parent

    def analyze_one(self, set_name: str, index: int):
        slide = self.parent.get_slide_data(set_name, index)
        # Perform fixation computation on slide['gaze_data']
        pass

    def analyze_subject(self, set_name: str):
        df = self.parent.get_subject_data(set_name)
        # Loop over slides
        pass

    def analyze_all(self):
        for subject in self.parent.subjects:
            self.analyze_subject(subject)

