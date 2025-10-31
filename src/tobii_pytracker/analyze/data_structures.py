from dataclasses import dataclass
from typing import Optional

@dataclass
class GazePoint:
    x: Optional[float]
    y: Optional[float]
    pupil_size: Optional[float]  # renamed from 'event'
    timestamp: float
