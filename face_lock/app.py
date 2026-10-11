import time


def run(config):
    import cv2

    from face_lock.antispoof import PassiveModels
    from face_lock.camera import CameraWorker
    from face_lock.controller import LockController
    from face_lock.event_log import EventLog
    from face_lock.models import load_face_models
    from face_lock.profile import load_profile
    from face_lock.ui import render

    profile = load_profile(config.profile_path)
    detector, recognizer = load_face_models(config.models_dir)
    pad = PassiveModels(config.models_dir)
    worker = CameraWorker(config, profile, detector, recognizer, pad)
    controller = LockController(config.threshold)
    log = EventLog(config)
    packet = None
    error_message = None
    recent_event = "Waiting for camera"
    started = time.monotonic()
    last_logged = -1.0
    title = "Face Lock"

    def record_events(items):
        nonlocal recent_event
        for event, latency in items:
            recent_event = event.replace("_", " ")
            log.write(event, controller, packet, latency)
            print(recent_event, flush=True)

    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        log.write("START_LOCKED", controller)
        worker.start()
        print(f"Camera {config.camera}. Q quits. Log: {log.path}")
        while True:
            now = time.monotonic()
            record_events(controller.tick(now))
            item = worker.poll()
            if item is not None:
                now = time.monotonic()
                if "error" in item:
                    error_message = item["error"]
                    record_events(controller.fault())
                    print(error_message, flush=True)
                else:
                    packet = item
                    record_events(
                        controller.observe(
                            now,
                            item["captured"],
                            item["count"],
                            item["score"],
                            item["problem"] is None and item["pad_ok"],
                        )
                    )
                    if now - last_logged >= 0.5:
                        log.write("OBSERVATION", controller, packet)
                        last_logged = now
            if packet is None and now - started > 8:
                record_events(controller.fault())
            canvas = render(
                packet,
                controller,
                config,
                profile.name,
                now,
                recent_event,
                error_message,
            )
            cv2.imshow(title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        worker.close()
        controller.state, controller.deadline = "LOCKED", None
        try:
            log.write("STOP_LOCKED", controller, packet)
        finally:
            log.close()
            cv2.destroyAllWindows()
        print(f"Stopped locked. Log: {log.path}")
    if error_message:
        raise RuntimeError(error_message)

