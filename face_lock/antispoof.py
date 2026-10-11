import math

from .models import ensure_model


class PassiveModels:
    def __init__(self, models_dir):
        import cv2

        self.nets = []
        for name, scale in (("MiniFASNetV2", 2.7), ("MiniFASNetV1SE", 4.0)):
            net = cv2.dnn.readNetFromONNX(str(ensure_model(name, models_dir)))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.nets.append((net, scale))

    @staticmethod
    def crop_bounds(shape, face, scale):
        height, width = shape[:2]
        x, y, w, h = (float(v) for v in face[:4])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("Invalid face crop")
        scale = min(scale, (width - 1) / w, (height - 1) / h)
        cw, ch = w * scale, h * scale
        # Translate the expanded crop inside the frame, preserving its size.
        left = min(max(0, x + w / 2 - cw / 2), width - 1 - cw)
        top = min(max(0, y + h / 2 - ch / 2), height - 1 - ch)
        return int(left), int(top), int(left + cw) + 1, int(top + ch) + 1

    def scores(self, frame, face):
        import cv2
        import numpy as np

        values = []
        for net, scale in self.nets:
            x1, y1, x2, y2 = self.crop_bounds(frame.shape, face, scale)
            patch = cv2.resize(frame[y1:y2, x1:x2], (80, 80))
            # These models expect unscaled BGR pixels.
            tensor = np.ascontiguousarray(
                patch.transpose(2, 0, 1)[None], dtype=np.float32
            )
            net.setInput(tensor)
            logits = net.forward().reshape(-1)
            if logits.size != 3 or not np.isfinite(logits).all():
                raise RuntimeError("Unexpected anti-spoof model output")
            weights = np.exp(logits - logits.max())
            values.append(float(weights[1] / weights.sum()))
        return values
