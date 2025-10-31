class TranscriptAnalyzer:
    def __init__(self, parent):
        self.parent = parent

    def analyze(self):
        """Transcribe all available audio files."""
        for idx in range(len(self.parent.data)):
            self.analyze_one(idx)

    def analyze_one(self, index: int):
        """Transcribe a single audio file."""
        slide = self.parent.get_slide_data(index)
        voice_path = slide['voice_file']
        if not voice_path:
            return None
        start_ts = slide['voice_start_timestamp']
        # Call speech-to-text library here, e.g., Whisper or OpenAI API
        pass