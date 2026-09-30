import importlib.util
import subprocess, sys

from importlib.metadata import version

__version__ = version("tobii-pytracker")

def _ensure_psychopy():
    if importlib.util.find_spec("psychopy") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psychopy==2024.1.4", "--no-deps"])

_ensure_psychopy()