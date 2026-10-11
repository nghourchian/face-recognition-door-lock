def render(packet, c, config, name, now, recent_event, error_message):
    import cv2
    import numpy as np

    canvas = np.full((620, 1020, 3), (24, 22, 20), dtype=np.uint8)

    def text(label, x, y, size=0.6, color=(220, 220, 220), weight=1):
        cv2.putText(
            canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight
        )

    text("FACE Detection Door Lock", 24, 40, 0.8, weight=2)
    text(
        "Look at the camera to unlock",
        24,
        72,
        0.52,
    )
    stale = packet is None or now - packet["captured"] > 1.0 or c.faulted
    if packet is not None:
        frame = packet["frame"]
        h, w = frame.shape[:2]
        scale = min(640 / w, 480 / h)
        image = cv2.resize(frame, (round(w * scale), round(h * scale)))
        ih, iw = image.shape[:2]
        if stale:
            image = (image * 0.25).astype(np.uint8)
        canvas[96 : 96 + ih, 24 : 24 + iw] = image
    color = (110, 220, 100) if c.state == "UNLOCKED" else (70, 180, 240)
    text("UNLOCKED" if c.state == "UNLOCKED" else "LOCKED", 690, 140, 0.95, color, 2)
    if c.state == "UNLOCKED":
        text(f"Relocks in {max(0, c.deadline - now):.1f}s", 690, 180)
    elif c.state == "WAIT_CLEAR":
        text("Waiting for empty frame", 690, 180, 0.52)
        text("Step away for 1 second", 690, 207, 0.52)
    else:
        text("Ready for a match", 690, 180)
    if stale:
        status = "Camera unavailable" if c.faulted else "Starting camera..."
        score_text = "--"
    else:
        score = packet["score"]
        status = packet["problem"] or (name if score >= config.threshold else "Unknown")
        score_text = "--" if score is None else f"{score:.3f}"
    text(status[:29], 690, 265, 0.55)
    text(f"Similarity: {score_text}", 690, 300)
    text(f"Cutoff: {config.threshold:.2f}", 690, 330, 0.53)
    values = (packet or {}).get("pad_scores")
    if values is not None and not stale:
        text(f"PAD V2:    {values[0]:.3f}", 690, 365, 0.53)
        text(f"PAD V1SE:  {values[1]:.3f}", 690, 391, 0.53)
        pad_status = (
            "PAD below cutoff"
            if min(values) < config.pad_threshold
            else ("PAD checks passed" if packet["pad_ok"] else "Collecting evidence...")
        )
    else:
        pad_status = "PAD: no valid result"
    text(pad_status, 690, 420, 0.5)
    progress = (
        min(1.0, (now - c.match_since) / 0.5)
        if c.match_since is not None and not stale
        else 0.0
    )
    text(f"PAD cutoff: {config.pad_threshold:.2f}", 690, 465, 0.48)
    cv2.rectangle(canvas, (690, 435), (985, 447), (60, 60, 60), -1)
    if progress:
        cv2.rectangle(canvas, (690, 435), (690 + round(295 * progress), 447), color, -1)
    text(recent_event[:32], 690, 490, 0.43)
    text(f"Unlocks: {c.unlocks} | Q: quit", 690, 540, 0.5)
    text("Q: quit", 24, 605, 0.48)
    if error_message:
        text("Camera error - see CMD", 30, 120, 0.6, (70, 70, 255), 2)
    return canvas
