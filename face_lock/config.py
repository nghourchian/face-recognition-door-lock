import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    camera: int = 1
    threshold: float = 0.45
    pad_threshold: float = 0.8
    workspace: Path = ROOT
    label: str = "unlabeled"

    def __post_init__(self):
        if self.camera < 0:
            raise ValueError("Camera index must be nonnegative.")
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Identity threshold must be between -1 and 1.")
        if not math.isfinite(self.pad_threshold) or not 0.5 <= self.pad_threshold <= 1:
            raise ValueError("PAD threshold must be between 0.5 and 1.")

    @property
    def models_dir(self):
        return self.workspace / "models"

    @property
    def profile_path(self):
        return self.workspace / "data" / "enrolled_face.npz"

    @property
    def logs_dir(self):
        return self.workspace / "logs"
