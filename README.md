# Facial Detection Door Lock

Does it only detect? No.

As misleading as the title may sound, this project goes beyond detecting a face. It compares the detected face against an enrolled user and uses eye and mouth actions as liveness checks before allowing access. The interface displays the similarity score, liveness indicators, and simulated lock status.


## Setup on Windows

Use Python 3.12 or 3.13 and a webcam. From Command Prompt in this project folder:

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

Settings are edited directly in `main.py`; there are no command-line options. Camera **1** is the default. Change `camera`, `threshold`, `pad_threshold`, `workspace`, or `label` in `SETTINGS` as needed.

For a new enrollment, set `ENROLL = True` and edit `NAME`, then run `main.py`. Follow the prompts and press Space for each sample. Set `ENROLL = False` afterward to run the lock. Replacing a saved profile also requires `OVERWRITE_PROFILE = True`; leave it false otherwise. Q closes the application.

Models download automatically on first use, are checked against SHA-256 hashes, and are cached in `models/`.

Use only the OpenCV package in `requirements.txt` inside this environment; the different OpenCV pip packages share the same `cv2` namespace.

### Reuse an existing profile

Copy your existing `data/enrolled_face.npz` into this project's `data/` folder. Profiles created by the earlier SFace version of this project remain compatible. You can also copy the four ONNX models into `models/` to avoid downloading them again.

Alternatively, change the workspace line in `main.py` to reuse your original project data:

```python
    workspace=Path.home() / "face-lock",
```

You can also reuse the original virtual environment. Open Command Prompt in the new folder containing `main.py` and run:

```bat
"%USERPROFILE%\face-lock\.venv\Scripts\python.exe" main.py
```

The workspace selects the directory used for models, enrollment data, and logs. It does not change the code being run.

## Unlock behavior

1. Exactly one sufficiently large, fully visible face must be detected.
2. Its cosine similarity must meet the identity threshold (default `0.45`). This score is not a probability.
3. Both anti-spoof scores must meet the PAD threshold (default `0.8`) for at least one second and five observations.
4. The lock requires a further half-second and three qualifying observations before unlocking.
5. It relocks after three seconds, then requires an empty view for one second before accepting another attempt.

A failed identity or PAD check resets the pending confirmation. Stale frames and camera failures prevent unlocking. These checks use application timestamps, not authenticated camera timestamps.

Thresholds are configured in `SETTINGS` in `main.py`. The defaults are prototype settings, not security-certified operating points.

## Project layout

| File | Responsibility |
| --- | --- |
| `main.py` | Editable settings and startup |
| `face_lock/app.py` | Application loop and cleanup |
| `face_lock/config.py` | Settings and workspace paths |
| `face_lock/camera.py` | Camera and inference worker |
| `face_lock/vision.py` | Detection, face quality, embeddings |
| `face_lock/models.py` | Model downloads and integrity checks |
| `face_lock/antispoof.py` | Passive anti-spoof inference |
| `face_lock/controller.py` | Lock state machine and sustained PAD confirmation |
| `face_lock/profile.py` | Enrollment storage and validation |
| `face_lock/enrollment.py` | Enrollment workflow |
| `face_lock/event_log.py` | CSV event logging |
| `face_lock/ui.py` | Preview and status panel |
| `tests/` | Automated checks |


## Scope and next steps

This project demonstrates integration of pretrained vision models, a timed state machine, worker-thread coordination, logging, and automated testing. It does not train a new recognition or anti-spoofing model.

Next steps are measured tests across people and lighting conditions, Raspberry Pi performance measurements, and a separate hardware adapter. Printed photos, screen replays, camera injection, and unfamiliar presentation attacks remain evaluation targets. No false-accept rate or production security claim has been established.

## Third-party components

This project uses pretrained models; their architecture, training, and weights belong to their respective upstream projects.

| Component | Source |
| --- | --- |
| OpenCV and Python bindings | https://github.com/opencv/opencv-python |
| YuNet face detector | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| SFace recognizer | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface |
| MiniFASNet ONNX exports | https://github.com/yakhyo/face-anti-spoofing |
| Original Silent-Face-Anti-Spoofing project | https://github.com/minivision-ai/Silent-Face-Anti-Spoofing |
| NumPy | https://numpy.org/ |

Weights are not bundled in this repository. The download URLs and expected hashes are in `face_lock/models.py`. Consult each upstream project's license and model terms before redistribution or commercial use. This file credits dependencies; it does not grant a license to their code or weights.

