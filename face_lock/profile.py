from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import RECOGNITION_MODEL
from .vision import normalize


@dataclass(frozen=True)
class Profile:
    name: str
    centroid: np.ndarray


def load_profile(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            "No enrollment found. Run: python main.py enroll --name Nima"
        )
    with np.load(path, allow_pickle=False) as data:
        if str(data["model"].item()) != RECOGNITION_MODEL:
            raise ValueError("Enrollment uses a different recognition model.")
        vector = np.asarray(data["centroid"], dtype=np.float32)
        if vector.size != 128:
            raise ValueError("Expected a 128-value SFace embedding.")
        name = str(data["name"].item()).strip()
        if not name:
            raise ValueError("Enrollment name is empty.")
        return Profile(name, normalize(vector))


def save_profile(path, name, samples):
    path = Path(path)
    if not name.strip() or not samples:
        raise ValueError("A name and reference samples are required.")
    samples = np.stack([normalize(s) for s in samples])
    if samples.shape[1] != 128:
        raise ValueError("Expected SFace embeddings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    try:
        with temp.open("wb") as output:
            np.savez_compressed(
                output,
                name=np.array(name.strip()),
                samples=samples,
                centroid=normalize(samples.mean(axis=0)),
                model=np.array(RECOGNITION_MODEL),
            )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
