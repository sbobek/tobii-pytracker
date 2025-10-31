from .base_analyzer import BaseAnalyzer
import pandas as pd

class TranscriptAnalyzer(BaseAnalyzer):
    def analyze(self):
        df = pd.read_csv(self.data_csv)
        self.results = self._transcribe(df)
        return self.results

    def analyze_one(self, index: int):
        df = pd.read_csv(self.data_csv)
        return self._transcribe(pd.DataFrame([df.iloc[index]]))

    def analyze_subject(self, set_name: str):
        data_csv = self.output_folder / set_name / "data.csv"
        df = pd.read_csv(data_csv)
        self.results = self._transcribe(df)
        return self.results

    def _transcribe(self, df: pd.DataFrame):
        transcripts = {}
        for _, row in df.iterrows():
            audio_file = row.get("voice_file")
            if not audio_file:
                continue
            # TODO: integrate real transcription model
            transcripts[audio_file] = f"Transcribed text for {audio_file}"
        return transcripts
