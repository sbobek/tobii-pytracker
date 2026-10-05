import importlib.util
import platform
import subprocess
import sys

from importlib.metadata import version

__version__ = version("tobii-pytracker")


def _ensure_psychopy():
    if importlib.util.find_spec("psychopy") is None:
        psychopy_version = (
            "psychopy>2024.1.4,<2025.1.0"
            if platform.system() == "Linux"
            else "psychopy>=2024.1.4,<2025.1.0"
        )

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            psychopy_version,
            "--no-deps",
        ])


_ensure_psychopy()