from __future__ import annotations

import sys
import types
from pathlib import Path


def bootstrap_test_environment() -> Path:
    root = Path(__file__).resolve().parent.parent

    if "tobii_pytracker" not in sys.modules:
        pkg = types.ModuleType("tobii_pytracker")
        pkg.__path__ = [str(root / "src" / "tobii_pytracker")]
        sys.modules["tobii_pytracker"] = pkg
    else:
        pkg = sys.modules["tobii_pytracker"]

    for name in [
        "tobii_pytracker.configs",
        "tobii_pytracker.datasets",
        "tobii_pytracker.runtime_models",
        "tobii_pytracker.utils",
        "tobii_pytracker.analyze",
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(root / "src" / "tobii_pytracker" / name.split(".")[-1])]
            sys.modules[name] = module
        setattr(pkg, name.split(".")[-1], sys.modules[name])

    if "psychopy" not in sys.modules:
        psychopy = types.ModuleType("psychopy")
        visual = types.ModuleType("psychopy.visual")
        core = types.ModuleType("psychopy.core")
        monitors = types.ModuleType("psychopy.monitors")
        event = types.ModuleType("psychopy.event")

        class _Stim:
            def __init__(self, *args, **kwargs):
                self.pos = kwargs.get("pos", (0, 0))
                self.text = kwargs.get("text", "")
                self.height = kwargs.get("height", 0)
                self.vertices = kwargs.get("vertices", [])
                self.boundingBox = kwargs.get("boundingBox", (0, 0))

            def draw(self):
                return None

        class Window(_Stim):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.size = kwargs.get("size", (1920, 1080))

            def flip(self):
                return None

        class Rect(_Stim):
            pass

        class Circle(_Stim):
            pass

        class TextStim(_Stim):
            pass

        class ImageStim(_Stim):
            pass

        class ShapeStim(_Stim):
            pass

        class Monitor:
            def __init__(self, name):
                self.name = name

            def setWidth(self, *_args, **_kwargs):
                return None

            def setDistance(self, *_args, **_kwargs):
                return None

            def setSizePix(self, *_args, **_kwargs):
                return None

            def saveMon(self):
                return None

            def getSizePix(self):
                return (1920, 1080)

        visual.Window = Window
        visual.Rect = Rect
        visual.Circle = Circle
        visual.TextStim = TextStim
        visual.ImageStim = ImageStim
        visual.ShapeStim = ShapeStim
        monitors.Monitor = Monitor
        core.wait = lambda *_args, **_kwargs: None
        event.waitKeys = lambda *_args, **_kwargs: []
        event.clearEvents = lambda *_args, **_kwargs: None

        psychopy.visual = visual
        psychopy.core = core
        psychopy.monitors = monitors
        psychopy.event = event

        sys.modules["psychopy"] = psychopy
        sys.modules["psychopy.visual"] = visual
        sys.modules["psychopy.core"] = core
        sys.modules["psychopy.monitors"] = monitors
        sys.modules["psychopy.event"] = event

    if "sklearn" not in sys.modules:
        try:
            import sklearn  # noqa: F401
            import sklearn.cluster  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "scikit-learn is required for tests. Install test dependencies "
                "(e.g. `pip install -r tests/requirements-test.txt`)."
            ) from exc

    if "skimage" not in sys.modules:
        import numpy as np

        skimage = types.ModuleType("skimage")
        segmentation = types.ModuleType("skimage.segmentation")
        measure = types.ModuleType("skimage.measure")

        def slic(img, n_segments=50, compactness=10, start_label=0, channel_axis=-1):
            h, w = img.shape[:2]
            labels = np.zeros((h, w), dtype=int)
            labels[:, w // 2 :] = 1
            return labels

        def find_contours(mask, level):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return []

            x0, x1 = float(xs.min()), float(xs.max() + 1)
            y0, y1 = float(ys.min()), float(ys.max() + 1)
            return [
                np.array(
                    [[y0, x0], [y0, x1], [y1, x1], [y1, x0]],
                    dtype=float,
                )
            ]

        segmentation.slic = slic
        measure.find_contours = find_contours
        skimage.segmentation = segmentation
        skimage.measure = measure
        sys.modules["skimage"] = skimage
        sys.modules["skimage.segmentation"] = segmentation
        sys.modules["skimage.measure"] = measure

    return root
