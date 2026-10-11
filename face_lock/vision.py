import cv2
import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError("Invalid face embedding.")
    return vector / norm


def detect(detector, frame):
    height, width = frame.shape[:2]
    sw = min(320, width)
    sh = max(1, round(height * sw / width))
    small = cv2.resize(frame, (sw, sh))
    detector.setInputSize((sw, sh))
    _, faces = detector.detect(small)
    if faces is not None:
        faces = faces.copy()
        faces[:, [0, 2, 4, 6, 8, 10, 12]] *= width / sw
        faces[:, [1, 3, 5, 7, 9, 11, 13]] *= height / sh
    return faces


def face_quality(frame, faces):
    if faces is None or len(faces) == 0:
        return "No face"
    if len(faces) != 1:
        return "One person at a time"
    x, y, w, h = faces[0, :4]
    fh, fw = frame.shape[:2]
    if min(w, h) < 80:
        return "Move closer"
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        return "Keep your whole face in view"
    return None


def feature(recognizer, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    return normalize(recognizer.feature(aligned))
