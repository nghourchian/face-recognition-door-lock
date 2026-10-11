import queue
import threading
import time

from .controller import PassiveGate


class CameraWorker:
    def __init__(self, config, profile, detector, recognizer, pad):
        self.config = config
        self.profile = profile
        self.detector = detector
        self.recognizer = recognizer
        self.pad = pad
        self.results = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _publish(self, item):
        try:
            self.results.get_nowait()
        except queue.Empty:
            pass
        self.results.put_nowait(item)

    def _run(self):
        import cv2
        import numpy as np

        from .vision import detect, face_quality, feature

        camera = None
        gate = PassiveGate(self.config.pad_threshold)
        try:
            camera = cv2.VideoCapture(self.config.camera)
            if not camera.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.config.camera}. Close other camera apps."
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not self.stop_event.is_set():
                captured = time.monotonic()
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                faces = detect(self.detector, frame)
                problem = face_quality(frame, faces)
                score, pad_scores = None, None
                if problem is None:
                    embedding = feature(self.recognizer, frame, faces[0])
                    score = float(
                        np.clip(np.dot(embedding, self.profile.centroid), -1, 1)
                    )
                    pad_scores = self.pad.scores(frame, faces[0])
                identity_ok = (
                    problem is None
                    and score is not None
                    and score >= self.config.threshold
                )
                pad_ok = gate.update(time.monotonic(), identity_ok, pad_scores)
                preview = cv2.flip(frame, 1)
                if faces is not None:
                    width = frame.shape[1]
                    for x, y, w, h, *_ in faces:
                        cv2.rectangle(
                            preview,
                            (round(width - 1 - x - w), round(y)),
                            (round(width - 1 - x), round(y + h)),
                            (190, 180, 70),
                            2,
                        )
                self._publish(
                    dict(
                        captured=captured,
                        frame=preview,
                        count=0 if faces is None else len(faces),
                        score=score,
                        problem=problem,
                        pad_scores=pad_scores,
                        pad_ok=pad_ok,
                        processing_ms=(time.monotonic() - captured) * 1000,
                    )
                )
        except Exception as error:
            self._publish({"error": str(error)})
        finally:
            if camera is not None:
                camera.release()
