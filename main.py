from pathlib import Path

from face_lock.app import run
from face_lock.config import Config
from face_lock.enrollment import enroll

SETTINGS = Config(
    camera=1,
    threshold=0.45,
    pad_threshold=0.8,
    workspace=Path.home() / "face-lock",    label="unlabeled",
)

ENROLL = False
NAME = "Nima"
OVERWRITE_PROFILE = False


def main():
    if ENROLL:
        enroll(SETTINGS, NAME, overwrite=OVERWRITE_PROFILE)
    else:
        run(SETTINGS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
