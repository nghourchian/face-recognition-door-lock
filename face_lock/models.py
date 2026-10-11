import hashlib
import tempfile
import urllib.request
from pathlib import Path

RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"
OPENCV_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
PAD_URL = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"
MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        f"{OPENCV_URL}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "sface": (
        RECOGNITION_MODEL,
        f"{OPENCV_URL}/face_recognition_sface/{RECOGNITION_MODEL}",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "MiniFASNetV2": (
        "MiniFASNetV2.onnx",
        f"{PAD_URL}/MiniFASNetV2.onnx",
        "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    ),
    "MiniFASNetV1SE": (
        "MiniFASNetV1SE.onnx",
        f"{PAD_URL}/MiniFASNetV1SE.onnx",
        "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
    ),
}


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ensure_model(key, directory):
    import cv2

    filename, url, expected = MODELS[key]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.is_file():
        if expected and file_hash(path) != expected:
            raise ValueError(
                f"Model checksum mismatch: {path}. Remove it and download again."
            )
        return path
    print(f"Downloading {filename}...", flush=True)
    temp = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "FaceLock/1.0"})
        with urllib.request.urlopen(request, timeout=60) as source:
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".onnx", delete=False
            ) as output:
                temp = Path(output.name)
                total = 0
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > 100_000_000:
                        raise ValueError("Unexpected model download size.")
                    output.write(block)
        if expected and file_hash(temp) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {filename}")
        cv2.dnn.readNetFromONNX(str(temp))
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return path


def load_face_models(directory):
    import cv2

    detector = cv2.FaceDetectorYN.create(
        str(ensure_model("yunet", directory)), "", (320, 320), 0.8, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(ensure_model("sface", directory)), "")
    return detector, recognizer
