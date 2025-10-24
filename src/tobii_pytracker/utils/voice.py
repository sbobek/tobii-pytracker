# utils/voice.py
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np

class VoiceRecorder:
    @staticmethod
    def record_voice(filename, duration=None, samplerate=44100, channels=1, stop_event=None):
        """Record voice until stop_event is set, then save to filename."""
        frames = []

        def callback(indata, frames_count, time_info, status):
            if stop_event.is_set():
                raise sd.CallbackAbort
            frames.append(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            try:
                while not stop_event.is_set():
                    sd.sleep(100)
            except sd.CallbackAbort:
                pass

        if frames:
            audio = np.concatenate(frames, axis=0)
            sf.write(filename, audio, samplerate)
