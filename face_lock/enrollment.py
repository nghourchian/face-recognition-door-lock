import time

from .models import load_face_models
from .profile import save_profile
from .vision import detect, face_quality, feature

POSES = (
    "Look straight ahead",
    "Turn slightly left",
    "Turn slightly right",
    "Tilt chin slightly up",
    "Tilt chin slightly down",
    "Try a natural smile",
    "Move a little farther back",
    "Look straight ahead again",
)


def enroll(config, name, overwrite=False):
    import cv2

    if not name.strip():
        raise ValueError("Name cannot be empty.")
    if config.profile_path.exists() and not overwrite:
        raise FileExistsError("A profile exists. Use --overwrite to replace it.")
    detector, recognizer = load_face_models(config.models_dir)
    camera = cv2.VideoCapture(config.camera)
    samples = []
    last_capture = -10.0
    title = "Face Lock - Enrollment"
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {config.camera}.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Click the preview. Space captures a sample; Q cancels.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            faces = detect(detector, frame)
            problem = face_quality(frame, faces)
            display = cv2.flip(frame, 1)
            cv2.rectangle(display, (0, 0), (display.shape[1], 85), (25, 25, 25), -1)
            cv2.putText(
                display,
                problem or "Ready - press SPACE",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 220, 100),
                2,
            )
            cv2.putText(
                display,
                f"{len(samples)}/8 | {POSES[len(samples)]}",
                (10, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
            )
            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            ):
                print("Enrollment cancelled; the saved profile was not changed.")
                return
            if key == 32 and problem is None and time.monotonic() - last_capture >= 1:
                samples.append(feature(recognizer, frame, faces[0]))
                last_capture = time.monotonic()
                print(f"Captured {len(samples)}/8")
                if len(samples) == len(POSES):
                    save_profile(config.profile_path, name, samples)
                    print(f"Enrollment saved: {config.profile_path}")
                    return
    finally:
        camera.release()
        cv2.destroyAllWindows()
